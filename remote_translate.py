"""Modal entrypoint that builds a constrained paired translation dataset.

Usage:

    SOMMELIER_TIMEOUT_SECONDS=28800 uv run modal run --detach remote_translate.py \
        --config examples/config.v3-he-full.yaml --run-id he-v3-translate-full \
        --target-language he --mode full --max-rows 60000 \
        --model-id gpt-5.5-2026-04-23 \
        --model-revision gpt-5.5-2026-04-23 \
        --max-new-tokens 512 --translator-interface instruction_chat \
        --max-model-len 0 --output-decoder standard \
        --runtime-backend openai_responses \
        --openai-service-tier flex --openai-max-workers 8 \
        --openai-list-price-limit-usd 50.00

Selection must match the pipeline run that will consume the rows: use the
same config, mode, and --max-rows as the intended remote_pipeline run so
the seeded split picks the same root examples. The full pipeline run
stages rows.<language>.jsonl next to its exported root rows under the
<input stem>.<language>.jsonl convention that data prepare expects. Full
evidence runs publish and consume these rows through an immutable dataset
revision; direct translation-run staging is a smoke-only boundary.

The remote function exports the root dataset's raw rows, runs the same
validation, dedupe, and seeded split selection the pipeline uses (CPU,
seconds), translates exactly the selected rows with the selected pinned
runtime, audits every output against its protected spans, and writes
rows.<language>.jsonl plus translation_summary.json to the artifacts volume under
artifacts/translation/<run_id>/. Progress is checkpointed per row. A full
Hebrew run ID resumes only while terminal outputs are absent and its config,
selection, translator, provider, source revision, and resource identity still
match; accepted rows and their summary make the run ID permanently one-shot.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final, Literal, cast

import modal

from sommelier.config import SommelierConfig
from sommelier.data.openai_evidence import (
    OPENAI_PROVIDER_JOURNAL_FILENAME,
    build_openai_provider_evidence,
)
from sommelier.data.openai_pricing import (
    openai_list_price_ceiling_runtime_summary,
    validated_openai_list_price_limit_usd,
)
from sommelier.data.openai_translate import (
    OPENAI_RESPONSES_SDK_MAX_RETRIES,
    OPENAI_RESPONSES_TIMEOUT_SECONDS,
)
from sommelier.remote.images import (
    OPENAI_TRANSLATION_RUNTIME_VERSIONS,
    SEQ2SEQ_TRANSLATION_RUNTIME_VERSIONS,
    VLLM_TRANSLATION_RUNTIME_VERSIONS,
    openai_translation_image,
    seq2seq_translation_image,
    vllm_translation_image,
)

APP_NAME = "sommelier-translate"

GPU = os.environ.get("SOMMELIER_GPU", "L40S")
TIMEOUT_SECONDS = int(os.environ.get("SOMMELIER_TIMEOUT_SECONDS", str(4 * 60 * 60)))

DEFAULT_MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"
DEFAULT_MODEL_REVISION = "main"
ARTIFACTS_ROOT = Path("/artifacts")

app = modal.App(APP_NAME)

TranslationRuntimeBackend = Literal[
    "vllm_chat",
    "transformers_seq2seq",
    "openai_responses",
]
TranslationInterface = Literal["instruction_chat", "translategemma", "madlad_seq2seq"]
OpenAIServiceTier = Literal["default", "flex"]
VLLM_RUNTIME_BACKEND: Final[TranslationRuntimeBackend] = "vllm_chat"
SEQ2SEQ_RUNTIME_BACKEND: Final[TranslationRuntimeBackend] = "transformers_seq2seq"
OPENAI_RUNTIME_BACKEND: Final[TranslationRuntimeBackend] = "openai_responses"
LOCAL_RUNTIME_DISPATCH: Final = "local"
OPENAI_API_KEY_ENV: Final = "OPENAI_API_KEY"
OPENAI_SECRET_NAME: Final = "openai-api-key"
HF_TOKEN_ENV: Final = "HF_TOKEN"
HF_READ_SECRET_NAME: Final = "huggingface-read-token"
OPENAI_DEFAULT_SERVICE_TIER: Final[OpenAIServiceTier] = "default"
OPENAI_MODEL_SNAPSHOT_PATTERN: Final = re.compile(r"^gpt-[a-z0-9][a-z0-9._-]*-\d{4}-\d{2}-\d{2}$")
VLLM_TRANSLATION_MAX_ATTEMPTS: Final = 3
SEQ2SEQ_TRANSLATION_MAX_ATTEMPTS: Final = 1
OPENAI_TRANSLATION_MAX_ATTEMPTS: Final = 3
OPENAI_TRANSLATION_CHUNK_SIZE: Final = 32
TRANSLATION_RUN_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TRANSLATION_RUN_IDENTITY_SCHEMA: Final = "sommelier.translation_run_identity.v1"
TRANSLATION_RUN_IDENTITY_FILENAME: Final = "translation_run_identity.json"


@dataclass(frozen=True)
class _ValidatedTranslationLaunch:
    config: SommelierConfig
    resolved_target: str
    resolved_interface: TranslationInterface
    max_attempts: int
    resolved_chunk_size: int
    resolved_openai_service_tier: OpenAIServiceTier
    resolved_openai_max_workers: int
    resolved_openai_list_price_limit: Decimal | None
    resolved_allocation_gpu: str | None
    resolved_function_timeout: int


artifacts_volume = modal.Volume.from_name("sommelier-artifacts", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sommelier-hf-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("sommelier-vllm-cache", create_if_missing=True)

# Env vars are set inside the function at runtime: the image mounts the
# package source as its final layer, so no build step may follow it.
vllm_image = vllm_translation_image()
seq2seq_image = seq2seq_translation_image()
openai_image = openai_translation_image()
# Compatibility alias retained for callers that inspect the historical module
# surface. Runtime dispatch below always selects an explicit image.
image = vllm_image


def _validate_translation_run_id(run_id: str) -> str:
    """Reject path traversal before a run ID is joined to the artifact root."""
    if TRANSLATION_RUN_ID_PATTERN.fullmatch(run_id) is None:
        from sommelier.errors import UserInputError

        raise UserInputError(
            f"invalid translation run id: {run_id!r}",
            hint=(
                "Use 1-128 ASCII letters, digits, dots, underscores, or hyphens; "
                "the first character must be alphanumeric."
            ),
        )
    return run_id


def _write_exclusive_text(path: Path, text: str) -> None:
    """Create one durable identity file without replacing an existing inode."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _admit_full_hebrew_translation_run(
    work: Path,
    *,
    run_id: str,
    target_language: str,
    config_yaml: str,
    identity: dict[str, object],
) -> None:
    """Reserve or resume one full run without touching finalized evidence.

    A terminal rows/summary/publication file is an immutable one-shot boundary.
    Before that boundary, the same run ID may resume only when both its exact
    submitted config bytes and complete run-level identity match.
    """
    from sommelier.data.translate import (
        PUBLICATION_MANIFEST_FILENAME,
        SUMMARY_FILENAME,
        rows_filename,
    )
    from sommelier.errors import UserInputError
    from sommelier.redaction import loads_unique_json

    if work.is_symlink() or (work.exists() and not work.is_dir()):
        raise UserInputError(f"translation run path is not a regular directory: {work}")

    final_paths = (
        work / rows_filename(target_language),
        work / SUMMARY_FILENAME,
        work / PUBLICATION_MANIFEST_FILENAME,
    )
    finalized = [path.name for path in final_paths if path.exists() or path.is_symlink()]
    if finalized:
        raise UserInputError(
            f"full translation run id {run_id!r} already has finalized output: "
            + ", ".join(finalized),
            hint=(
                "Keep accepted rows and summary evidence immutable. Use a new --run-id; "
                "only a progress-only run with matching identity may resume."
            ),
        )

    work.mkdir(parents=True, exist_ok=True)
    identity_path = work / TRANSLATION_RUN_IDENTITY_FILENAME
    encoded_identity = json.dumps(identity, indent=2, sort_keys=True) + "\n"
    if identity_path.exists() or identity_path.is_symlink():
        if identity_path.is_symlink() or not identity_path.is_file():
            raise UserInputError("translation run identity is not a regular file")
        try:
            observed_identity = loads_unique_json(identity_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise UserInputError("translation run identity is missing or invalid") from error
        if observed_identity != identity:
            raise UserInputError(
                f"incomplete translation run id {run_id!r} has a different identity",
                hint=(
                    "Resume with the exact same config, source SHA, selection, translator, "
                    "provider, and resource arguments, or use a new --run-id."
                ),
            )
    else:
        existing = sorted(path.name for path in work.iterdir())
        if existing:
            raise UserInputError(
                f"incomplete translation run id {run_id!r} has no run identity",
                hint=(
                    "The existing artifacts cannot be proven resumable under this launch. "
                    "Preserve them and use a new --run-id."
                ),
            )
        try:
            _write_exclusive_text(identity_path, encoded_identity)
        except FileExistsError as error:
            raise UserInputError("translation run identity reservation raced") from error

    config_path = work / "config.yaml"
    if config_path.exists() or config_path.is_symlink():
        if config_path.is_symlink() or not config_path.is_file():
            raise UserInputError("translation run config is not a regular file")
        try:
            observed_config = config_path.read_text(encoding="utf-8")
        except OSError as error:
            raise UserInputError("translation run config cannot be read") from error
        if observed_config != config_yaml:
            raise UserInputError(
                f"incomplete translation run id {run_id!r} has different config bytes",
                hint="Use the exact original config or choose a new --run-id.",
            )
    else:
        try:
            _write_exclusive_text(config_path, config_yaml)
        except FileExistsError:
            observed_config = config_path.read_text(encoding="utf-8")
            if observed_config != config_yaml:
                raise UserInputError(
                    f"incomplete translation run id {run_id!r} has different config bytes"
                )

    # Close the small reservation/finalization race before any model/provider
    # construction. A concurrent completion is terminal for this invocation.
    finalized = [path.name for path in final_paths if path.exists() or path.is_symlink()]
    if finalized:
        raise UserInputError(
            f"full translation run id {run_id!r} became finalized during admission: "
            + ", ".join(finalized),
            hint="Keep the existing evidence immutable and use a new --run-id.",
        )


def _runtime_versions(
    runtime_backend: TranslationRuntimeBackend,
) -> tuple[tuple[str, str], ...]:
    if runtime_backend == OPENAI_RUNTIME_BACKEND:
        return OPENAI_TRANSLATION_RUNTIME_VERSIONS
    if runtime_backend == SEQ2SEQ_RUNTIME_BACKEND:
        return SEQ2SEQ_TRANSLATION_RUNTIME_VERSIONS
    return VLLM_TRANSLATION_RUNTIME_VERSIONS


def _package_versions(
    expected_versions: tuple[tuple[str, str], ...] = VLLM_TRANSLATION_RUNTIME_VERSIONS,
) -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package, _expected in expected_versions:
        if package == "python":
            continue
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "absent"
    return versions


def _validate_full_hebrew_runtime(
    *,
    mode: str,
    target_language: str,
    runtime_backend: TranslationRuntimeBackend,
    package_versions: dict[str, str],
) -> None:
    """Require the backend-specific pinned image before provider/data access.

    Every OpenAI request, including a diagnostic smoke, requires the exact
    CPU-provider image. Local model smoke runs and other languages remain
    observational so they can probe a prospective stack; a full local Hebrew
    producer fails before dataset export, network access, or model loading when
    its distribution or Python patch differs from the selected image identity.
    """
    # Every paid provider request uses the exact CPU image identity, including
    # smoke diagnostics. Local model smoke runs remain observational; their
    # full Hebrew evidence gate retains the established behavior below.
    if runtime_backend == OPENAI_RUNTIME_BACKEND:
        should_validate = True
    else:
        should_validate = mode == "full" and target_language == "he"
    if not should_validate:
        return

    expected = dict(_runtime_versions(runtime_backend))
    if package_versions == expected:
        return
    differences = [
        f"{package}: expected {expected.get(package, 'absent')}, "
        f"observed {package_versions.get(package, 'absent')}"
        for package in sorted(set(expected) | set(package_versions))
        if package_versions.get(package) != expected.get(package)
    ]
    from sommelier.errors import UserInputError

    raise UserInputError(
        f"translation runtime does not match the pinned image ({'; '.join(differences)})",
        hint=(
            "Rebuild the pinned Modal translation image and rerun a smoke probe. "
            "Do not export the full corpus or load a model under a different environment."
        ),
    )


def _validated_openai_service_tier(value: str) -> OpenAIServiceTier:
    if value not in {"default", "flex"}:
        from sommelier.errors import UserInputError

        raise UserInputError(
            f"unsupported OpenAI Responses service tier: {value!r}",
            hint="Choose the explicit diagnostic tier 'default' or the preregistered tier 'flex'.",
        )
    return cast(OpenAIServiceTier, value)


def _validated_openai_max_workers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        from sommelier.errors import UserInputError

        raise UserInputError("OpenAI Responses max workers must be a positive integer")
    return value


def _validate_openai_model_snapshot(model_id: str, model_revision: str) -> None:
    from sommelier.errors import UserInputError

    if model_revision != model_id or OPENAI_MODEL_SNAPSHOT_PATTERN.fullmatch(model_id) is None:
        raise UserInputError(
            "OpenAI Responses requires matching exact dated model snapshots",
            hint=(
                "Pass the same dated provider ID, such as gpt-5.5-2026-04-23, "
                "to --model-id and --model-revision."
            ),
        )
    try:
        date.fromisoformat(model_id[-10:])
    except ValueError as error:
        raise UserInputError("OpenAI Responses model snapshot date is invalid") from error


def _load_translation_model_for_backend(
    translator: Any,
    *,
    runtime_backend: TranslationRuntimeBackend,
    provider_journal_path: Path | None,
    openai_service_tier: OpenAIServiceTier,
    openai_max_workers: int,
    openai_list_price_limit_usd: str,
) -> Any:
    """Load the selected adapter without importing provider code on local paths."""
    if runtime_backend != OPENAI_RUNTIME_BACKEND:
        from sommelier.data.translate import load_translation_model

        return load_translation_model(translator)
    if provider_journal_path is None:  # pragma: no cover - internal invariant
        raise RuntimeError("OpenAI Responses translation requires a provider journal path")

    from sommelier.data.openai_translate import load_openai_responses_translation_model

    return load_openai_responses_translation_model(
        model_snapshot=translator.model_id,
        max_output_tokens=translator.max_new_tokens,
        provider_journal_path=provider_journal_path,
        expected_sdk_version=dict(OPENAI_TRANSLATION_RUNTIME_VERSIONS)["openai"],
        max_retries=OPENAI_RESPONSES_SDK_MAX_RETRIES,
        timeout_seconds=OPENAI_RESPONSES_TIMEOUT_SECONDS,
        service_tier=openai_service_tier,
        max_workers=openai_max_workers,
        openai_list_price_limit_usd=openai_list_price_limit_usd,
    )


def _local_source_identity() -> tuple[str, bool | None]:
    """Return the exact local revision and whether its mounted tree is clean.

    Modal snapshots the local Python source rather than the repository metadata,
    so this identity must be computed by the local entrypoint and passed into the
    remote function. ``None`` keeps the boundary explicit when Git is unavailable.
    """
    from sommelier.manifests import get_git_commit, get_git_worktree_clean

    return get_git_commit(), get_git_worktree_clean()


def _validated_translation_config(
    config_yaml: str,
    *,
    mode: str,
    target_language: str,
) -> tuple[SommelierConfig, str]:
    """Parse and validate the translation config without durable or remote I/O."""
    from sommelier.config import load_config
    from sommelier.errors import UserInputError
    from sommelier.pipeline import apply_smoke_overrides

    with tempfile.TemporaryDirectory(prefix="sommelier-translation-config-") as temporary:
        incoming_config_path = Path(temporary) / "config.yaml"
        incoming_config_path.write_text(config_yaml, encoding="utf-8")
        config = load_config(incoming_config_path)
    if mode not in {"smoke", "full"}:
        raise UserInputError(
            f"unsupported translation mode: {mode!r}",
            hint="Choose --mode smoke or --mode full.",
        )
    if mode == "smoke":
        config = apply_smoke_overrides(config)
    paired_languages = [
        item.language for item in config.datasets if item.source_id_column is not None
    ]
    resolved_target = target_language
    if not resolved_target:
        if len(paired_languages) != 1:
            raise UserInputError(
                "translation target is ambiguous",
                hint="Pass --target-language for one configured paired dataset source.",
            )
        resolved_target = paired_languages[0]
    if resolved_target not in paired_languages:
        raise UserInputError(
            f"target language {resolved_target!r} is not a paired source in the config",
            hint="Add the target under datasets with source_id_column set.",
        )
    if mode == "full" and resolved_target == "he":
        from sommelier.evaluation.data_provenance import (
            validate_hebrew_v3_translation_config,
        )

        validate_hebrew_v3_translation_config(config)
    return config, resolved_target


def _validate_translation_launch_contract(
    config_yaml: str,
    run_id: str,
    mode: str,
    max_rows: int,
    model_id: str,
    model_revision: str,
    max_new_tokens: int,
    translator_interface: str,
    max_model_len: int,
    trust_remote_code: bool,
    output_decoder: str,
    limit: int,
    target_language: str,
    code_revision: str,
    source_tree_clean: bool | None,
    allocation_gpu: str | None,
    function_timeout_seconds: int | None,
    *,
    runtime_backend: TranslationRuntimeBackend,
    openai_service_tier: str = "default",
    openai_max_workers: int = 1,
    openai_list_price_limit_usd: str = "",
) -> _ValidatedTranslationLaunch:
    """Validate one launch without allocating Modal or touching durable artifacts."""
    from sommelier.data.translate import (
        DEFAULT_CHUNK_SIZE,
        translator_interface_for_model,
        validate_hebrew_v3_translation_request,
    )
    from sommelier.errors import UserInputError

    _validate_translation_run_id(run_id)
    resolved_interface = translator_interface_for_model(model_id, translator_interface)
    if runtime_backend == OPENAI_RUNTIME_BACKEND:
        if resolved_interface != "instruction_chat":
            raise UserInputError(
                "OpenAI Responses transport requires the instruction_chat interface",
                hint="Keep provider transport and the translation prompt contract orthogonal.",
            )
        max_attempts = OPENAI_TRANSLATION_MAX_ATTEMPTS
        required_backend = OPENAI_RUNTIME_BACKEND
    elif resolved_interface == "madlad_seq2seq":
        max_attempts = SEQ2SEQ_TRANSLATION_MAX_ATTEMPTS
        required_backend = SEQ2SEQ_RUNTIME_BACKEND
    else:
        max_attempts = VLLM_TRANSLATION_MAX_ATTEMPTS
        required_backend = VLLM_RUNTIME_BACKEND
    if runtime_backend != required_backend:
        raise UserInputError(
            f"translator interface {resolved_interface!r} requires runtime backend "
            f"{required_backend!r}, not {runtime_backend!r}",
            hint="Dispatch the request through the backend-specific Modal function.",
        )

    resolved_chunk_size = (
        OPENAI_TRANSLATION_CHUNK_SIZE
        if runtime_backend == OPENAI_RUNTIME_BACKEND
        else DEFAULT_CHUNK_SIZE
    )
    resolved_openai_service_tier = _validated_openai_service_tier(openai_service_tier)
    resolved_openai_max_workers = _validated_openai_max_workers(openai_max_workers)
    resolved_openai_list_price_limit = None
    if runtime_backend == OPENAI_RUNTIME_BACKEND:
        if allocation_gpu is not None:
            raise UserInputError(
                "OpenAI Responses translation is CPU-only and cannot request a GPU"
            )
        _validate_openai_model_snapshot(model_id, model_revision)
        resolved_openai_list_price_limit = validated_openai_list_price_limit_usd(
            openai_list_price_limit_usd
        )
    elif openai_service_tier != OPENAI_DEFAULT_SERVICE_TIER:
        raise UserInputError("OpenAI service tier is only valid for the openai_responses backend")
    elif openai_max_workers != 1:
        raise UserInputError("OpenAI max workers is only valid for the openai_responses backend")
    elif openai_list_price_limit_usd:
        raise UserInputError(
            "OpenAI list-price limit is only valid for the openai_responses backend"
        )

    config, resolved_target = _validated_translation_config(
        config_yaml,
        mode=mode,
        target_language=target_language,
    )
    resolved_allocation_gpu = (
        None if runtime_backend == OPENAI_RUNTIME_BACKEND else allocation_gpu or GPU
    )
    resolved_function_timeout = (
        TIMEOUT_SECONDS if function_timeout_seconds is None else function_timeout_seconds
    )
    validate_hebrew_v3_translation_request(
        target_language=resolved_target,
        mode=mode,
        model_id=model_id,
        model_revision=model_revision,
        max_new_tokens=max_new_tokens,
        translator_interface=resolved_interface,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        output_decoder=output_decoder,
        max_attempts=max_attempts,
        max_rows=max_rows,
        limit=limit,
        seed=config.project.seed,
        runtime_backend=runtime_backend,
        provider_service_tier=(
            resolved_openai_service_tier if runtime_backend == OPENAI_RUNTIME_BACKEND else None
        ),
        provider_sdk_version=(
            dict(OPENAI_TRANSLATION_RUNTIME_VERSIONS)["openai"]
            if runtime_backend == OPENAI_RUNTIME_BACKEND
            else None
        ),
        provider_timeout_seconds=(
            OPENAI_RESPONSES_TIMEOUT_SECONDS if runtime_backend == OPENAI_RUNTIME_BACKEND else None
        ),
        provider_max_workers=(
            resolved_openai_max_workers if runtime_backend == OPENAI_RUNTIME_BACKEND else None
        ),
        chunk_size=resolved_chunk_size,
        openai_list_price_limit_usd=(
            openai_list_price_limit_usd if runtime_backend == OPENAI_RUNTIME_BACKEND else None
        ),
    )
    if mode == "full":
        if limit:
            raise UserInputError(
                "full translation runs cannot use --limit",
                hint="Use --limit only for smoke quality checks.",
            )
        if (
            re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", code_revision) is None
            or source_tree_clean is not True
        ):
            raise UserInputError(
                "full translation evidence requires a clean, immutable local Git revision",
                hint="Commit the v3 implementation and launch again from a clean worktree.",
            )
        if (
            runtime_backend != OPENAI_RUNTIME_BACKEND
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", model_revision) is None
        ):
            raise UserInputError(
                "full translation evidence requires an immutable translator revision",
                hint="Pass --model-revision with the exact Hugging Face commit SHA.",
            )

    return _ValidatedTranslationLaunch(
        config=config,
        resolved_target=resolved_target,
        resolved_interface=resolved_interface,
        max_attempts=max_attempts,
        resolved_chunk_size=resolved_chunk_size,
        resolved_openai_service_tier=resolved_openai_service_tier,
        resolved_openai_max_workers=resolved_openai_max_workers,
        resolved_openai_list_price_limit=resolved_openai_list_price_limit,
        resolved_allocation_gpu=resolved_allocation_gpu,
        resolved_function_timeout=resolved_function_timeout,
    )


def _run_remote_translation(
    config_yaml: str,
    run_id: str,
    mode: str,
    max_rows: int,
    model_id: str,
    model_revision: str,
    max_new_tokens: int,
    translator_interface: str,
    max_model_len: int,
    trust_remote_code: bool,
    output_decoder: str,
    limit: int,
    target_language: str,
    code_revision: str,
    source_tree_clean: bool | None,
    allocation_gpu: str | None = None,
    function_timeout_seconds: int | None = None,
    *,
    runtime_backend: TranslationRuntimeBackend,
    openai_service_tier: str = "default",
    openai_max_workers: int = 1,
    openai_list_price_limit_usd: str = "",
) -> str:
    launch = _validate_translation_launch_contract(
        config_yaml,
        run_id,
        mode,
        max_rows,
        model_id,
        model_revision,
        max_new_tokens,
        translator_interface,
        max_model_len,
        trust_remote_code,
        output_decoder,
        limit,
        target_language,
        code_revision,
        source_tree_clean,
        allocation_gpu,
        function_timeout_seconds,
        runtime_backend=runtime_backend,
        openai_service_tier=openai_service_tier,
        openai_max_workers=openai_max_workers,
        openai_list_price_limit_usd=openai_list_price_limit_usd,
    )
    if runtime_backend != OPENAI_RUNTIME_BACKEND:
        os.environ.setdefault("HF_HOME", "/hf-cache")
        # Xet's chunked downloader stalled on a multi-GB shard during the first
        # smoke. The regular HTTP path is resumable through the same HF cache and
        # has a deliberately long per-request timeout for large model weights.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
        if runtime_backend == VLLM_RUNTIME_BACKEND:
            os.environ.setdefault("VLLM_CACHE_ROOT", "/vllm-cache")

    from sommelier.artifacts import sha256_file
    from sommelier.data.export import export_raw_rows
    from sommelier.data.load import load_raw_rows
    from sommelier.data.split import all_examples, prepare_split_result
    from sommelier.data.translate import (
        TranslatorInfo,
        progress_filename,
        translate_rows,
        translation_selection_contract_sha256,
        translator_request_sha256,
        write_translation_outputs,
    )

    config = launch.config
    resolved_target = launch.resolved_target
    resolved_interface = launch.resolved_interface
    max_attempts = launch.max_attempts
    resolved_chunk_size = launch.resolved_chunk_size
    resolved_openai_service_tier = launch.resolved_openai_service_tier
    resolved_openai_max_workers = launch.resolved_openai_max_workers
    resolved_openai_list_price_limit = launch.resolved_openai_list_price_limit
    resolved_allocation_gpu = launch.resolved_allocation_gpu
    resolved_function_timeout = launch.resolved_function_timeout
    if runtime_backend == OPENAI_RUNTIME_BACKEND:
        from sommelier.errors import UserInputError

        if not os.environ.get(OPENAI_API_KEY_ENV):
            raise UserInputError(
                "OpenAI Responses translation is missing its API credential",
                hint=(
                    f"Inject {OPENAI_API_KEY_ENV} through the named Modal secret "
                    f"{OPENAI_SECRET_NAME!r}; never pass it as a function argument."
                ),
            )
        if not os.environ.get(HF_TOKEN_ENV):
            raise UserInputError(
                "OpenAI Responses translation is missing its gated-dataset credential",
                hint=(
                    f"Inject {HF_TOKEN_ENV} through the named Modal secret "
                    f"{HF_READ_SECRET_NAME!r}; never pass it as a function argument."
                ),
            )
    source = config.root_dataset

    resolved_model_len = max_model_len or (
        2048 if resolved_interface in {"translategemma", "madlad_seq2seq"} else 8192
    )
    translator = TranslatorInfo(
        model_id=model_id,
        model_revision=model_revision,
        max_new_tokens=max_new_tokens,
        interface=resolved_interface,
        max_model_len=resolved_model_len,
        trust_remote_code=trust_remote_code,
        output_decoder=output_decoder,  # type: ignore[arg-type]
        implementation_revision=code_revision,
        runtime_backend=runtime_backend,
        provider_service_tier=(
            resolved_openai_service_tier if runtime_backend == OPENAI_RUNTIME_BACKEND else None
        ),
        provider_sdk_version=(
            dict(OPENAI_TRANSLATION_RUNTIME_VERSIONS)["openai"]
            if runtime_backend == OPENAI_RUNTIME_BACKEND
            else None
        ),
        provider_timeout_seconds=(
            OPENAI_RESPONSES_TIMEOUT_SECONDS if runtime_backend == OPENAI_RUNTIME_BACKEND else None
        ),
    )
    work = ARTIFACTS_ROOT / "translation" / run_id
    if mode == "full" and resolved_target == "he":
        runtime_identity: dict[str, object] = {
            "backend": runtime_backend,
            "translation_chunk_size": resolved_chunk_size,
            "allocation_gpu": resolved_allocation_gpu,
            "function_timeout_seconds": resolved_function_timeout,
        }
        if runtime_backend == OPENAI_RUNTIME_BACKEND:
            runtime_identity.update(
                {
                    "provider_service_tier": resolved_openai_service_tier,
                    "provider_sdk_version": dict(OPENAI_TRANSLATION_RUNTIME_VERSIONS)["openai"],
                    "provider_timeout_seconds": OPENAI_RESPONSES_TIMEOUT_SECONDS,
                    "provider_max_workers": resolved_openai_max_workers,
                    "openai_list_price_limit_usd": (
                        format(resolved_openai_list_price_limit, "f")
                        if resolved_openai_list_price_limit is not None
                        else None
                    ),
                }
            )
        run_identity: dict[str, object] = {
            "schema_version": TRANSLATION_RUN_IDENTITY_SCHEMA,
            "run_id": run_id,
            "config_sha256": hashlib.sha256(config_yaml.encode("utf-8")).hexdigest(),
            "selection": {
                "contract_sha256": translation_selection_contract_sha256(
                    config,
                    mode="full",
                    max_rows=max_rows,
                    limit=limit,
                ),
                "mode": mode,
                "max_rows": max_rows,
                "limit": limit,
                "seed": config.project.seed,
            },
            "translator": {
                "model_id": model_id,
                "model_revision": model_revision,
                "request_sha256": translator_request_sha256(translator, resolved_target),
                "max_attempts": max_attempts,
                "implementation_revision": code_revision,
            },
            "runtime": runtime_identity,
            "source_code": {
                "git_commit": code_revision,
                "working_tree_clean": source_tree_clean,
            },
        }
        _admit_full_hebrew_translation_run(
            work,
            run_id=run_id,
            target_language=resolved_target,
            config_yaml=config_yaml,
            identity=run_identity,
        )
    else:
        work.mkdir(parents=True, exist_ok=True)
        (work / "config.yaml").write_text(config_yaml, encoding="utf-8")
    config_path = work / "config.yaml"

    expected_versions = _runtime_versions(runtime_backend)
    package_versions = _package_versions(expected_versions)
    _validate_full_hebrew_runtime(
        mode=mode,
        target_language=resolved_target,
        runtime_backend=runtime_backend,
        package_versions=package_versions,
    )
    print(f"[translate] package versions: {package_versions}", flush=True)

    rows_path = work / f"rows.{source.language}.jsonl"
    if not rows_path.exists():
        exported = export_raw_rows(source, rows_path, seed=config.project.seed, max_rows=max_rows)
        print(f"[translate] exported {exported} raw rows", flush=True)
        artifacts_volume.commit()
    rows = load_raw_rows(rows_path)

    # The pipeline's own selection: same validation, dedupe, seed, and
    # counts, so the translated set is exactly the reference run's rows.
    root_result = prepare_split_result(
        rows,
        min_query_chars=config.data.min_query_chars,
        max_query_chars=config.data.max_query_chars,
        n_train=config.data.n_train,
        n_validation=config.data.n_validation,
        n_test=config.data.n_test,
        seed=config.project.seed,
        language=source.language,
    )
    selected_ids = {example["example_id"] for example in all_examples(root_result)}
    selected = [row for row in rows if row["source_id"] in selected_ids]
    if limit:
        selected = selected[:limit]
    print(f"[translate] translating {len(selected)} selected rows", flush=True)

    provider_journal_path = (
        work / OPENAI_PROVIDER_JOURNAL_FILENAME
        if runtime_backend == OPENAI_RUNTIME_BACKEND
        else None
    )
    load_started = time.monotonic()
    try:
        model = _load_translation_model_for_backend(
            translator,
            runtime_backend=runtime_backend,
            provider_journal_path=provider_journal_path,
            openai_service_tier=resolved_openai_service_tier,
            openai_max_workers=resolved_openai_max_workers,
            openai_list_price_limit_usd=(
                format(resolved_openai_list_price_limit, "f")
                if resolved_openai_list_price_limit is not None
                else ""
            ),
        )
    finally:
        # Persist resumable Hugging Face ``*.incomplete`` shards even when a
        # multi-gigabyte checkpoint download is interrupted.  Without this
        # failure-boundary commit, a retry starts the entire download again.
        # vLLM's compile cache follows the same rule so a failed warm-up does
        # not discard completed graphs.
        if runtime_backend != OPENAI_RUNTIME_BACKEND:
            hf_cache_volume.commit()
            if runtime_backend == VLLM_RUNTIME_BACKEND:
                vllm_cache_volume.commit()
    model_load_seconds = time.monotonic() - load_started
    translation_started = time.monotonic()
    translated, stats = translate_rows(
        selected,
        model,
        progress_path=work / progress_filename(resolved_target),
        max_query_chars=config.data.max_query_chars,
        target_language=resolved_target,
        translator=translator,
        max_attempts=max_attempts,
        chunk_size=resolved_chunk_size,
        durable_checkpoint=(
            artifacts_volume.commit if runtime_backend == OPENAI_RUNTIME_BACKEND else None
        ),
    )
    stats["max_attempts"] = max_attempts
    runtime_stats: dict[str, object] = {
        # ``gpu`` is retained for existing summary consumers. The explicit
        # fields bind the evidence to the decorator values passed by the local
        # launcher instead of container-side module globals.
        "gpu": resolved_allocation_gpu,
        "provider": "openai" if runtime_backend == OPENAI_RUNTIME_BACKEND else "modal",
        "execution_provider": "modal",
        "backend": runtime_backend,
        "gpu_allocation_label": resolved_allocation_gpu,
        "function_timeout_seconds": resolved_function_timeout,
        "model_load_seconds": model_load_seconds,
        "translation_seconds": time.monotonic() - translation_started,
        "translation_chunk_size": resolved_chunk_size,
        "boundary": (
            "Modal enforces the recorded outer function timeout. Translation time "
            "includes audited retries and excludes model loading."
        ),
    }
    if runtime_backend == OPENAI_RUNTIME_BACKEND:
        if provider_journal_path is None:  # pragma: no cover - established above
            raise RuntimeError("OpenAI provider journal path was not initialized")
        if resolved_openai_list_price_limit is None:  # pragma: no cover - established above
            raise RuntimeError("OpenAI list-price limit was not initialized")
        stats["provider_evidence"] = build_openai_provider_evidence(
            provider_journal_path,
            model_id,
            resolved_openai_service_tier,
        )
        runtime_stats.update(
            {
                "provider_service_tier": resolved_openai_service_tier,
                "provider_timeout_seconds": OPENAI_RESPONSES_TIMEOUT_SECONDS,
                "provider_max_workers": resolved_openai_max_workers,
                "provider_journal_filename": provider_journal_path.name,
                "openai_list_price_ceiling": openai_list_price_ceiling_runtime_summary(
                    resolved_openai_list_price_limit,
                    service_tier=resolved_openai_service_tier,
                ),
                "credential_source": (
                    f"Modal named secrets {OPENAI_SECRET_NAME!r} and {HF_READ_SECRET_NAME!r} "
                    f"require keys {OPENAI_API_KEY_ENV} and {HF_TOKEN_ENV}"
                ),
            }
        )
    stats["runtime"] = runtime_stats
    stats["environment"] = package_versions
    if runtime_backend != OPENAI_RUNTIME_BACKEND:
        stats["download_policy"] = {
            "hf_hub_disable_xet": os.environ["HF_HUB_DISABLE_XET"],
            "hf_hub_download_timeout_seconds": int(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]),
            "cache_root": "/hf-cache",
        }
    stats["selection"] = {
        "config_sha256": sha256_file(config_path),
        "contract_sha256": translation_selection_contract_sha256(
            config,
            mode=mode,  # type: ignore[arg-type]
            max_rows=max_rows,
            limit=limit,
        ),
        "mode": mode,
        "max_rows": max_rows,
        "limit": limit,
        "seed": config.project.seed,
        "selected_rows": len(selected),
        "selected_source_ids_sha256": hashlib.sha256(
            "\n".join(str(row["source_id"]) for row in selected).encode("utf-8")
        ).hexdigest(),
    }
    stats["source_code"] = {
        "git_commit": code_revision,
        "working_tree_clean": source_tree_clean,
        "boundary": (
            "Computed by the local Modal entrypoint for the source tree mounted into this function."
        ),
    }
    rows_out, summary_path = write_translation_outputs(
        work,
        translated,
        stats,
        translator=translator,
        input_description=(
            f"{source.dataset_id}@{source.dataset_revision} via {rows_path.name}, "
            f"selected {len(selected)} rows with seed {config.project.seed}"
        ),
        target_language=resolved_target,
        input_sha256=sha256_file(rows_path),
    )
    artifacts_volume.commit()
    if runtime_backend != OPENAI_RUNTIME_BACKEND:
        hf_cache_volume.commit()
        if runtime_backend == VLLM_RUNTIME_BACKEND:
            vllm_cache_volume.commit()
    print(f"[translate] stats: {stats}", flush=True)
    print(f"[translate] environment: {stats['environment']}", flush=True)
    return f"rows={rows_out} summary={summary_path}"


@app.function(  # type: ignore[untyped-decorator]
    retries=modal.Retries(max_retries=2, initial_delay=60.0),
    image=vllm_image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={
        "/artifacts": artifacts_volume,
        "/hf-cache": hf_cache_volume,
        "/vllm-cache": vllm_cache_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def run_remote_translation(
    config_yaml: str,
    run_id: str,
    mode: str,
    max_rows: int,
    model_id: str,
    model_revision: str,
    max_new_tokens: int,
    translator_interface: str,
    max_model_len: int,
    trust_remote_code: bool,
    output_decoder: str,
    limit: int,
    target_language: str,
    code_revision: str,
    source_tree_clean: bool | None,
    allocation_gpu: str | None = None,
    function_timeout_seconds: int | None = None,
) -> str:
    """Run chat-style translation in the pinned vLLM runtime."""
    return _run_remote_translation(
        config_yaml,
        run_id,
        mode,
        max_rows,
        model_id,
        model_revision,
        max_new_tokens,
        translator_interface,
        max_model_len,
        trust_remote_code,
        output_decoder,
        limit,
        target_language,
        code_revision,
        source_tree_clean,
        allocation_gpu,
        function_timeout_seconds,
        runtime_backend=VLLM_RUNTIME_BACKEND,
    )


@app.function(  # type: ignore[untyped-decorator]
    retries=modal.Retries(max_retries=2, initial_delay=60.0),
    image=seq2seq_image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={
        "/artifacts": artifacts_volume,
        "/hf-cache": hf_cache_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def run_remote_seq2seq_translation(
    config_yaml: str,
    run_id: str,
    mode: str,
    max_rows: int,
    model_id: str,
    model_revision: str,
    max_new_tokens: int,
    translator_interface: str,
    max_model_len: int,
    trust_remote_code: bool,
    output_decoder: str,
    limit: int,
    target_language: str,
    code_revision: str,
    source_tree_clean: bool | None,
    allocation_gpu: str | None = None,
    function_timeout_seconds: int | None = None,
) -> str:
    """Run MADLAD translation in the pinned Transformers-v4 runtime."""
    return _run_remote_translation(
        config_yaml,
        run_id,
        mode,
        max_rows,
        model_id,
        model_revision,
        max_new_tokens,
        translator_interface,
        max_model_len,
        trust_remote_code,
        output_decoder,
        limit,
        target_language,
        code_revision,
        source_tree_clean,
        allocation_gpu,
        function_timeout_seconds,
        runtime_backend=SEQ2SEQ_RUNTIME_BACKEND,
    )


@app.function(  # type: ignore[untyped-decorator]
    retries=0,
    image=openai_image,
    timeout=TIMEOUT_SECONDS,
    volumes={
        "/artifacts": artifacts_volume,
    },
    secrets=[
        modal.Secret.from_name(
            OPENAI_SECRET_NAME,
            required_keys=[OPENAI_API_KEY_ENV],
        ),
        modal.Secret.from_name(
            HF_READ_SECRET_NAME,
            required_keys=[HF_TOKEN_ENV],
        ),
    ],
)
def run_remote_openai_translation(
    config_yaml: str,
    run_id: str,
    mode: str,
    max_rows: int,
    model_id: str,
    model_revision: str,
    max_new_tokens: int,
    translator_interface: str,
    max_model_len: int,
    trust_remote_code: bool,
    output_decoder: str,
    limit: int,
    target_language: str,
    code_revision: str,
    source_tree_clean: bool | None,
    allocation_gpu: None = None,
    function_timeout_seconds: int | None = None,
    openai_service_tier: str = "default",
    openai_max_workers: int = 1,
    openai_list_price_limit_usd: str = "",
) -> str:
    """Run explicit provider translation in the pinned CPU-only runtime."""
    return _run_remote_translation(
        config_yaml,
        run_id,
        mode,
        max_rows,
        model_id,
        model_revision,
        max_new_tokens,
        translator_interface,
        max_model_len,
        trust_remote_code,
        output_decoder,
        limit,
        target_language,
        code_revision,
        source_tree_clean,
        allocation_gpu,
        function_timeout_seconds,
        runtime_backend=OPENAI_RUNTIME_BACKEND,
        openai_service_tier=openai_service_tier,
        openai_max_workers=openai_max_workers,
        openai_list_price_limit_usd=openai_list_price_limit_usd,
    )


@app.local_entrypoint()  # type: ignore[untyped-decorator]
def main(
    config: str = "examples/config.full.yaml",
    run_id: str = "fr-translate-1",
    mode: str = "full",
    max_rows: int = 0,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    max_new_tokens: int = 1024,
    translator_interface: str = "auto",
    max_model_len: int = 0,
    trust_remote_code: bool = False,
    output_decoder: str = "standard",
    limit: int = 0,
    target_language: str = "",
    runtime_backend: str = LOCAL_RUNTIME_DISPATCH,
    openai_service_tier: str = OPENAI_DEFAULT_SERVICE_TIER,
    openai_max_workers: int = 1,
    openai_list_price_limit_usd: str = "",
) -> None:
    from sommelier.data.translate import translator_interface_for_model

    config_yaml = Path(config).read_text(encoding="utf-8")
    _validate_translation_run_id(run_id)
    code_revision, source_tree_clean = _local_source_identity()
    resolved_interface = translator_interface_for_model(model_id, translator_interface)
    if runtime_backend == OPENAI_RUNTIME_BACKEND:
        from sommelier.errors import UserInputError

        if translator_interface != "instruction_chat":
            raise UserInputError(
                "OpenAI Responses transport requires explicit instruction_chat",
                hint=(
                    "Pass both --runtime-backend openai_responses and "
                    "--translator-interface instruction_chat; runtime backend selection "
                    "is the only paid-inference authorization."
                ),
            )
        if resolved_interface != "instruction_chat":
            raise UserInputError("OpenAI Responses requires the instruction_chat prompt contract")
        resolved_service_tier = _validated_openai_service_tier(openai_service_tier)
        resolved_max_workers = _validated_openai_max_workers(openai_max_workers)
        resolved_list_price_limit = validated_openai_list_price_limit_usd(
            openai_list_price_limit_usd
        )
        launch = _validate_translation_launch_contract(
            config_yaml,
            run_id,
            mode,
            max_rows,
            model_id,
            model_revision,
            max_new_tokens,
            translator_interface,
            max_model_len,
            trust_remote_code,
            output_decoder,
            limit,
            target_language,
            code_revision,
            source_tree_clean,
            None,
            TIMEOUT_SECONDS,
            runtime_backend=OPENAI_RUNTIME_BACKEND,
            openai_service_tier=openai_service_tier,
            openai_max_workers=openai_max_workers,
            openai_list_price_limit_usd=openai_list_price_limit_usd,
        )
        resolved_interface = launch.resolved_interface
        resolved_service_tier = launch.resolved_openai_service_tier
        resolved_max_workers = launch.resolved_openai_max_workers
        assert launch.resolved_openai_list_price_limit is not None
        resolved_list_price_limit = launch.resolved_openai_list_price_limit
        result = run_remote_openai_translation.remote(
            config_yaml,
            run_id,
            mode,
            max_rows,
            model_id,
            model_revision,
            max_new_tokens,
            resolved_interface,
            max_model_len,
            trust_remote_code,
            output_decoder,
            limit,
            target_language,
            code_revision,
            source_tree_clean,
            None,
            TIMEOUT_SECONDS,
            resolved_service_tier,
            resolved_max_workers,
            format(resolved_list_price_limit, "f"),
        )
    elif runtime_backend == LOCAL_RUNTIME_DISPATCH:
        from sommelier.errors import UserInputError

        if openai_service_tier != OPENAI_DEFAULT_SERVICE_TIER:
            raise UserInputError("OpenAI service tier is not valid for a local translation backend")
        if openai_max_workers != 1:
            raise UserInputError("OpenAI max workers is not valid for a local translation backend")
        if openai_list_price_limit_usd:
            raise UserInputError(
                "OpenAI list-price limit is not valid for a local translation backend"
            )
        remote_function = (
            run_remote_seq2seq_translation
            if resolved_interface == "madlad_seq2seq"
            else run_remote_translation
        )
        resolved_backend = (
            SEQ2SEQ_RUNTIME_BACKEND
            if resolved_interface == "madlad_seq2seq"
            else VLLM_RUNTIME_BACKEND
        )
        launch = _validate_translation_launch_contract(
            config_yaml,
            run_id,
            mode,
            max_rows,
            model_id,
            model_revision,
            max_new_tokens,
            translator_interface,
            max_model_len,
            trust_remote_code,
            output_decoder,
            limit,
            target_language,
            code_revision,
            source_tree_clean,
            GPU,
            TIMEOUT_SECONDS,
            runtime_backend=resolved_backend,
        )
        resolved_interface = launch.resolved_interface
        result = remote_function.remote(
            config_yaml,
            run_id,
            mode,
            max_rows,
            model_id,
            model_revision,
            max_new_tokens,
            resolved_interface,
            max_model_len,
            trust_remote_code,
            output_decoder,
            limit,
            target_language,
            code_revision,
            source_tree_clean,
            GPU,
            TIMEOUT_SECONDS,
        )
    else:
        from sommelier.errors import UserInputError

        raise UserInputError(
            f"unsupported translation runtime backend: {runtime_backend!r}",
            hint="Choose 'local' or explicitly choose 'openai_responses'.",
        )
    print(result)
