from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, cast

from sommelier.analysis.tokenization import (
    TOKENIZER_TAX_RECORD_SCHEMA,
    TOKENIZER_TAX_RECORDS_FILENAME,
    TOKENIZER_TAX_REPORT_FILENAME,
    TOKENIZER_TAX_REPORT_SCHEMA,
)
from sommelier.artifacts import (
    ArtifactRef,
    make_artifact_ref,
    read_json_with_schema,
    read_jsonl_with_schema,
)
from sommelier.config import SommelierConfig, compute_config_digest, load_config
from sommelier.data.prepare import paired_input_path
from sommelier.data.semantic_review import (
    SEMANTIC_REVIEW_FILENAME,
    SEMANTIC_REVIEW_SCHEMA,
    SEMANTIC_REVIEW_TEMPLATE_FILENAME,
    SEMANTIC_REVIEW_TEMPLATE_SCHEMA,
)
from sommelier.data.translate import (
    PUBLICATION_MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    TRANSLATION_CONFIG_FILENAME,
    TRANSLATION_PUBLICATION_SCHEMA,
    TRANSLATION_RUN_IDENTITY_FILENAME,
    TRANSLATION_RUN_IDENTITY_SCHEMA,
    TRANSLATION_SUMMARY_SCHEMA,
    validate_full_paired_input_contract,
)
from sommelier.data.types import DROP_SUMMARY_SCHEMA, PREPARED_EXAMPLE_SCHEMA
from sommelier.errors import EvaluationError, SommelierError, UserInputError
from sommelier.evaluation.generate import (
    GENERATION_SCHEMA,
    IMMUTABLE_HF_REVISION,
    INFERENCE_TELEMETRY_FILENAME,
    INFERENCE_TELEMETRY_SCHEMA,
)
from sommelier.formatting.chat import FORMATTED_EXAMPLE_SCHEMA
from sommelier.hebrew_v3_preregistration import require_preregistered_reviewer
from sommelier.runtime_metadata import RUNTIME_METADATA_FILENAME, RUNTIME_METADATA_SCHEMA

MANIFEST_SCHEMA: Final = "sommelier.manifest.v1"
DATA_PROVENANCE_SCHEMA: Final = "sommelier.hebrew_v3_data_provenance.v1"

HEBREW_V3_BASE_MODEL_ID: Final = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
HEBREW_V3_BASE_MODEL_REVISION: Final = "54641c1611fcff44fa4865626462445e0a153fc7"
HEBREW_V3_ROOT_DATASET_ID: Final = "Salesforce/xlam-function-calling-60k"
HEBREW_V3_ROOT_DATASET_REVISION: Final = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
HEBREW_V3_PAIRED_DATASET_ID: Final = "abdelstark/sommelier-xlam-single-call-splits-he"
HEBREW_V3_PROVISIONAL_PAIRED_DATASET_REVISION: Final = "main"
HEBREW_V3_PROJECT_PREFIX: Final = "sommelier-v3-he"
HEBREW_V3_FULL_MAX_ROWS: Final = 60_000
HEBREW_V3_V1_ADAPTER_ID: Final = "abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora"
HEBREW_V3_V1_ADAPTER_REVISION: Final = "45a6e2fa3e29f8393ddf1e9bda51a9461b41ee0e"
HEBREW_V3_TARGET_MODULES: Final = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _error(message: str, *, hint: str | None = None) -> EvaluationError:
    return EvaluationError(
        message,
        hint=hint or "Use one completed Hebrew v3 full-run bundle without editing its artifacts.",
    )


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error(f"{context} must be a JSON object")
    return cast("dict[str, Any]", value)


def _sequence(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(f"{context} must be a JSON array")
    return value


def is_hebrew_v3_config(config: SommelierConfig) -> bool:
    """Return whether a config belongs to the preregistered Hebrew v3 experiment.

    The project prefix identifies the committed example configs, while the
    canonical paired-dataset ID keeps the gate active if the project name alone
    is edited. Merely using Hebrew does not opt an independent experiment into
    Sommelier's fixed v3 contract.
    """
    return config.project.name.startswith(HEBREW_V3_PROJECT_PREFIX) or any(
        source.dataset_id == HEBREW_V3_PAIRED_DATASET_ID for source in config.datasets
    )


ConfigContractErrorFactory = Callable[[str], SommelierError]


def _validate_hebrew_v3_config_contract(
    config: SommelierConfig,
    *,
    allow_provisional_paired_revision: bool,
    error: ConfigContractErrorFactory,
) -> None:
    """Validate the shared Hebrew v3 config, with one phase-specific revision rule."""
    if config.project.seed != 42:
        raise error("Hebrew v3 config does not use preregistered seed 42")
    if config.project.name != "sommelier-v3-he-full" or config.project.artifact_root != Path(
        "artifacts"
    ):
        raise error("Hebrew v3 config changed the preregistered project/output contract")
    if (
        config.model.base_model_id != HEBREW_V3_BASE_MODEL_ID
        or config.model.base_model_revision != HEBREW_V3_BASE_MODEL_REVISION
        or config.model.tokenizer_revision != HEBREW_V3_BASE_MODEL_REVISION
        or config.model.allow_remote_code
    ):
        raise error("Hebrew v3 config substituted the preregistered base or tokenizer")

    if tuple(source.language for source in config.datasets) != ("en", "he"):
        raise error("Hebrew v3 config must contain exactly the en/he dataset pair")
    root, paired = config.datasets
    if (
        root.dataset_id != HEBREW_V3_ROOT_DATASET_ID
        or root.dataset_revision != HEBREW_V3_ROOT_DATASET_REVISION
        or root.source_id_column is not None
    ):
        raise error("Hebrew v3 config substituted the preregistered English root corpus")
    if (
        paired.dataset_id != HEBREW_V3_PAIRED_DATASET_ID
        or paired.source_id_column != "source_example_id"
    ):
        raise error("Hebrew v3 config substituted the preregistered audited Hebrew corpus identity")
    paired_revision_is_allowed = (
        paired.dataset_revision == HEBREW_V3_PROVISIONAL_PAIRED_DATASET_REVISION
        if allow_provisional_paired_revision
        else IMMUTABLE_HF_REVISION.fullmatch(paired.dataset_revision) is not None
    )
    if not paired_revision_is_allowed:
        revision_requirement = (
            f"the provisional revision {HEBREW_V3_PROVISIONAL_PAIRED_DATASET_REVISION!r}"
            if allow_provisional_paired_revision
            else "an immutable commit"
        )
        raise error(f"Hebrew v3 config does not use {revision_requirement} for the Hebrew corpus")
    for source in (root, paired):
        if (
            source.query_column != "query"
            or source.tools_column != "tools"
            or source.answers_column != "answers"
        ):
            raise error("Hebrew v3 config changed the preregistered dataset columns")

    data_contract = (
        config.data.n_train,
        config.data.n_validation,
        config.data.n_test,
        config.data.min_query_chars,
        config.data.max_query_chars,
        config.data.dedupe_key,
    )
    if data_contract != (15_000, 1_000, 1_000, 10, 2_000, "normalized_query"):
        raise error("Hebrew v3 config changed the preregistered cohort contract")

    formatting_contract = (
        config.formatting.template_policy,
        config.formatting.target_format,
        " ".join(config.formatting.system_prompt.split()),
    )
    if formatting_contract != (
        "tokenizer_chat_template",
        "json_tool_call",
        (
            "You are a tool-calling model. Select the correct tool and return only "
            "the JSON tool call. Do not include explanations."
        ),
    ):
        raise error("Hebrew v3 config changed the preregistered formatting contract")

    train_contract = (
        config.train.epochs,
        config.train.per_device_batch_size,
        config.train.gradient_accumulation_steps,
        config.train.learning_rate,
        config.train.scheduler,
        config.train.warmup_ratio,
        config.train.max_sequence_length,
        config.train.quantization,
        config.train.compute_dtype,
        config.train.lora_rank,
        config.train.lora_alpha,
        config.train.lora_dropout,
        tuple(config.train.languages),
        tuple(config.train.target_modules),
    )
    if train_contract != (
        2,
        4,
        4,
        0.0002,
        "cosine",
        0.03,
        4096,
        "nf4-4bit",
        "bfloat16",
        16,
        32,
        0.05,
        ("en", "he"),
        HEBREW_V3_TARGET_MODULES,
    ):
        raise error("Hebrew v3 config changed the preregistered QLoRA contract")

    eval_contract = (
        config.eval.split,
        tuple(config.eval.slices),
        config.eval.temperature,
        config.eval.do_sample,
        config.eval.max_new_tokens,
        config.eval.parser_version,
        config.remote.gpu,
    )
    if eval_contract != (
        "test",
        ("en", "he"),
        0.0,
        False,
        512,
        "sommelier.parser.v1",
        "L40S",
    ):
        raise error("Hebrew v3 config changed the preregistered evaluation hardware contract")

    remote_contract = (
        config.remote.enabled,
        config.remote.data_timeout_seconds,
        config.remote.train_timeout_seconds,
        config.remote.eval_timeout_seconds,
    )
    if remote_contract != (True, 1_800, 43_200, 18_000):
        raise error("Hebrew v3 config changed the preregistered remote planning contract")
    if not config.report.retain_raw_generations or config.report.redact_fields:
        raise error("Hebrew v3 config changed the preregistered report evidence contract")
    if (
        config.tracking.enabled
        or config.tracking.provider != "wandb"
        or config.tracking.project != "sommelier"
    ):
        raise error("Hebrew v3 config changed the preregistered tracking contract")


def validate_hebrew_v3_preregistered_config(config: SommelierConfig) -> None:
    """Reject a post-hoc model, corpus, training, or evaluation substitution.

    Terminal evidence requires the published Hebrew dataset's immutable Hub
    revision. The same shared contract is used by the pre-publication producer,
    whose only permitted mutable value is the committed ``main`` placeholder.
    """
    _validate_hebrew_v3_config_contract(
        config,
        allow_provisional_paired_revision=False,
        error=_error,
    )
    try:
        require_preregistered_reviewer(config, context="Hebrew v3 Phase-B config")
    except UserInputError as error:
        raise _error(str(error), hint=error.hint) from error


def validate_hebrew_v3_translation_config(config: SommelierConfig) -> None:
    """Validate full Hebrew v3 before its not-yet-published corpus can exist.

    Translation necessarily precedes publication, so the canonical ``main``
    placeholder is accepted for the Hebrew revision. Every other project,
    model, root corpus, column, cohort, formatting, QLoRA, evaluation, remote,
    reporting, and tracking field is identical to the terminal validator.
    """

    def user_input_error(message: str) -> UserInputError:
        return UserInputError(
            message,
            hint=(
                "Launch the full producer with examples/config.v3-he-full.yaml; "
                "only its provisional Hebrew dataset revision may remain 'main'."
            ),
        )

    _validate_hebrew_v3_config_contract(
        config,
        allow_provisional_paired_revision=True,
        error=user_input_error,
    )
    require_preregistered_reviewer(config, context="Hebrew v3 Phase-A config")


def _artifact_ref(
    path: Path,
    *,
    artifact_root: Path,
    kind: str,
    schema_version: str,
) -> ArtifactRef:
    if not path.is_file() or path.is_symlink():
        raise _error(f"Hebrew v3 evidence is missing or aliased: {path}")
    return make_artifact_ref(
        path,
        artifact_root=artifact_root,
        kind=kind,
        schema_version=schema_version,
    )


def _assert_ref(value: object, expected: ArtifactRef, *, context: str) -> None:
    reference = _mapping(value, context=context)
    if reference != dict(expected):
        raise _error(f"{context} does not match the exact stored artifact")


def _refs_by_path(value: object, *, context: str) -> dict[str, dict[str, Any]]:
    items = _sequence(value, context=context)
    refs: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(items):
        ref = _mapping(item, context=f"{context}[{index}]")
        path = ref.get("path")
        if not isinstance(path, str) or not path:
            raise _error(f"{context}[{index}].path must be a non-empty string")
        if path in refs:
            raise _error(f"{context} repeats artifact path {path}")
        refs[path] = ref
    return refs


def _assert_exact_refs(
    value: object,
    expected: Mapping[str, ArtifactRef],
    *,
    context: str,
) -> None:
    observed = _refs_by_path(value, context=context)
    if set(observed) != set(expected):
        raise _error(f"{context} artifact set does not match the Hebrew v3 contract")
    for path, artifact in expected.items():
        _assert_ref(observed[path], artifact, context=f"{context} {path}")


def _assert_stage_manifest(
    manifest: Mapping[str, Any],
    *,
    stage: str,
    run_id: str,
    config_sha256: str,
    code_revision: str,
) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise _error(f"Hebrew v3 {stage} manifest has the wrong schema")
    if manifest.get("stage") != stage or manifest.get("status") != "succeeded":
        raise _error(f"Hebrew v3 {stage} manifest is not a succeeded {stage} stage")
    if manifest.get("run_id") != run_id:
        raise _error(f"Hebrew v3 {stage} manifest run_id does not match evaluation")
    if manifest.get("config_sha256") != config_sha256:
        raise _error(f"Hebrew v3 {stage} manifest config does not match evaluation")
    if manifest.get("git_commit") != code_revision:
        raise _error(f"Hebrew v3 {stage} manifest source revision does not match runtime")


def _record_identities(
    records: list[dict[str, Any]],
    *,
    split: str,
    context: str,
) -> list[tuple[str, str, str, str | None]]:
    identities: list[tuple[str, str, str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        example_id = record.get("example_id")
        language = record.get("language")
        source_example_id = record.get("source_example_id")
        if (
            not isinstance(example_id, str)
            or not example_id
            or language not in {"en", "he"}
            or record.get("split") != split
            or (source_example_id is not None and not isinstance(source_example_id, str))
        ):
            raise _error(f"{context} record {index} has an invalid cohort identity")
        key = (cast(str, language), example_id)
        if key in seen:
            raise _error(f"{context} repeats example identity {language}:{example_id}")
        seen.add(key)
        identities.append((example_id, cast(str, language), split, source_example_id))
    return identities


def _same_cohort_identity_set(
    left: list[tuple[str, str, str, str | None]],
    right: list[tuple[str, str, str, str | None]],
) -> bool:
    return len(left) == len(right) and set(left) == set(right)


def _validate_observed_cohorts(
    *,
    run_dir: Path,
) -> dict[str, dict[str, int]]:
    """Prove that the full preregistered cohort, not a truncated run, was scored."""
    expected_english = {"train": 15_000, "validation": 1_000, "test": 1_000}
    observed: dict[str, dict[str, int]] = {}
    formatted_identities: list[tuple[str, str, str, str | None]] = []
    for split in ("train", "validation", "test"):
        prepared = read_jsonl_with_schema(
            run_dir / "data" / f"{split}.jsonl",
            expected_schema=PREPARED_EXAMPLE_SCHEMA,
        )
        formatted = read_jsonl_with_schema(
            run_dir / "formatted" / f"{split}.jsonl",
            expected_schema=FORMATTED_EXAMPLE_SCHEMA,
        )
        prepared_identity = _record_identities(
            prepared, split=split, context=f"prepared {split} cohort"
        )
        formatted_identity = _record_identities(
            formatted, split=split, context=f"formatted {split} cohort"
        )
        if prepared_identity != formatted_identity:
            raise _error(f"formatted {split} cohort is not the ordered output of prepared data")
        counts = {
            language: sum(identity[1] == language for identity in prepared_identity)
            for language in ("en", "he")
        }
        if counts["en"] != expected_english[split]:
            raise _error(
                f"Hebrew v3 {split} cohort has {counts['en']} English rows; "
                f"expected {expected_english[split]}"
            )
        if counts["he"] <= 0:
            raise _error(f"Hebrew v3 {split} cohort contains no Hebrew paired rows")
        observed[split] = {**counts, "total": len(prepared_identity)}
        formatted_identities.extend(formatted_identity)

    token_records = read_jsonl_with_schema(
        run_dir / "analysis" / "tokenization" / TOKENIZER_TAX_RECORDS_FILENAME,
        expected_schema=TOKENIZER_TAX_RECORD_SCHEMA,
    )
    token_identities: list[tuple[str, str, str, str | None]] = []
    for split in ("train", "validation", "test"):
        records = [record for record in token_records if record.get("split") == split]
        token_identities.extend(
            _record_identities(
                records,
                split=split,
                context=f"tokenizer-tax {split} cohort",
            )
        )
    if not _same_cohort_identity_set(token_identities, formatted_identities):
        raise _error("tokenizer-tax records are not the exact formatted full cohort")
    return observed


def _source_input_refs(
    config: SommelierConfig,
    *,
    root_rows_path: Path,
    artifact_root: Path,
    validated: Mapping[str, Mapping[str, Path]],
) -> tuple[dict[str, ArtifactRef], dict[str, ArtifactRef]]:
    refs: dict[str, ArtifactRef] = {}
    evidence: dict[str, ArtifactRef] = {}

    root_ref = _artifact_ref(
        root_rows_path,
        artifact_root=artifact_root,
        kind="raw_dataset",
        schema_version="sommelier.raw_tool_call_row.v1",
    )
    refs[root_ref["path"]] = root_ref
    evidence["root_rows"] = root_ref

    for source in config.datasets:
        if source.source_id_column is None:
            continue
        language = source.language
        expected_paths: dict[str, tuple[Path, str, str]] = {
            "paired_rows": (
                paired_input_path(root_rows_path, language),
                "raw_paired_dataset",
                "sommelier.raw_tool_call_row.v1",
            ),
            "translation_summary": (
                root_rows_path.with_name(f"{Path(SUMMARY_FILENAME).stem}.{language}.json"),
                "translation_summary",
                TRANSLATION_SUMMARY_SCHEMA,
            ),
            "translation_publication": (
                root_rows_path.with_name(
                    f"{Path(PUBLICATION_MANIFEST_FILENAME).stem}.{language}.json"
                ),
                "translation_publication_manifest",
                TRANSLATION_PUBLICATION_SCHEMA,
            ),
        }
        if language == "he":
            expected_paths.update(
                {
                    "semantic_review_template": (
                        root_rows_path.with_name(
                            f"{Path(SEMANTIC_REVIEW_TEMPLATE_FILENAME).stem}.he.json"
                        ),
                        "translation_semantic_review_template",
                        SEMANTIC_REVIEW_TEMPLATE_SCHEMA,
                    ),
                    "semantic_review": (
                        root_rows_path.with_name(f"{Path(SEMANTIC_REVIEW_FILENAME).stem}.he.json"),
                        "translation_semantic_review",
                        SEMANTIC_REVIEW_SCHEMA,
                    ),
                    "translation_config": (
                        root_rows_path.with_name(
                            f"{Path(TRANSLATION_CONFIG_FILENAME).stem}.he.yaml"
                        ),
                        "config",
                        "sommelier.config.v2",
                    ),
                    "translation_run_identity": (
                        root_rows_path.with_name(
                            f"{Path(TRANSLATION_RUN_IDENTITY_FILENAME).stem}.he.json"
                        ),
                        "translation_run_identity",
                        TRANSLATION_RUN_IDENTITY_SCHEMA,
                    ),
                }
            )
        returned = validated.get(language)
        if returned is None or set(returned) != set(expected_paths):
            raise _error(f"validated {language} publication path set is incomplete")
        for key, (expected_path, kind, schema) in expected_paths.items():
            if returned[key].resolve() != expected_path.resolve():
                raise _error(f"validated {language} {key} path was substituted")
            ref = _artifact_ref(
                expected_path,
                artifact_root=artifact_root,
                kind=kind,
                schema_version=schema,
            )
            refs[ref["path"]] = ref
            evidence[f"{language}_{key}"] = ref
    return refs, evidence


def validate_hebrew_v3_data_provenance(
    *,
    run_dir: Path,
    artifact_root: Path,
    run_id: str,
    report_config_sha256: str,
) -> dict[str, Any]:
    """Bind experiment claims to the exact audited Hebrew v3 source and stage chain."""
    run_dir = run_dir.resolve()
    artifact_root = artifact_root.resolve()
    expected_run_dir = artifact_root / "runs" / run_id
    if run_dir != expected_run_dir.resolve():
        raise _error("Hebrew v3 run path does not match its declared run_id")

    config_path = run_dir / "config.resolved.yaml"
    config_ref = _artifact_ref(
        config_path,
        artifact_root=artifact_root,
        kind="config",
        schema_version="sommelier.config.v2",
    )
    if compute_config_digest(config_path.read_text(encoding="utf-8")) != report_config_sha256:
        raise _error("Hebrew v3 resolved config digest does not match evaluation")
    config = load_config(config_path)
    validate_hebrew_v3_preregistered_config(config)

    runtime_path = run_dir / RUNTIME_METADATA_FILENAME
    runtime_ref = _artifact_ref(
        runtime_path,
        artifact_root=artifact_root,
        kind="runtime_metadata",
        schema_version=RUNTIME_METADATA_SCHEMA,
    )
    runtime = _mapping(
        json.loads(runtime_path.read_text(encoding="utf-8")),
        context="Hebrew v3 runtime metadata",
    )
    if runtime.get("schema_version") != RUNTIME_METADATA_SCHEMA:
        raise _error("Hebrew v3 runtime metadata has the wrong schema")
    if runtime.get("run_id") != run_id or runtime.get("config_sha256") != report_config_sha256:
        raise _error("Hebrew v3 runtime identity does not match evaluation")
    source_code = _mapping(runtime.get("source_code"), context="Hebrew v3 source_code")
    code_revision = source_code.get("git_commit")
    if (
        not isinstance(code_revision, str)
        or IMMUTABLE_HF_REVISION.fullmatch(code_revision) is None
        or source_code.get("working_tree_clean") is not True
    ):
        raise _error("Hebrew v3 runtime was not produced from a clean immutable source")

    root_manifest_path = run_dir / "manifest.json"
    root_manifest_ref = _artifact_ref(
        root_manifest_path,
        artifact_root=artifact_root,
        kind="manifest",
        schema_version=MANIFEST_SCHEMA,
    )
    root_manifest = read_json_with_schema(root_manifest_path, expected_schema=MANIFEST_SCHEMA)
    if root_manifest.get("run_id") != run_id or root_manifest.get("status") != "succeeded":
        raise _error("Hebrew v3 root run manifest is not succeeded for this run")
    _assert_ref(root_manifest.get("config"), config_ref, context="Hebrew v3 root config")
    stages = _mapping(root_manifest.get("stages"), context="Hebrew v3 root stages")

    data_manifest_path = run_dir / "data_manifest.json"
    data_manifest_ref = _artifact_ref(
        data_manifest_path,
        artifact_root=artifact_root,
        kind="manifest",
        schema_version=MANIFEST_SCHEMA,
    )
    if stages.get("data") != data_manifest_ref["path"]:
        raise _error("Hebrew v3 root manifest does not bind the data manifest")
    data_manifest = read_json_with_schema(data_manifest_path, expected_schema=MANIFEST_SCHEMA)
    _assert_stage_manifest(
        data_manifest,
        stage="data",
        run_id=run_id,
        config_sha256=report_config_sha256,
        code_revision=code_revision,
    )

    root_rows_path = run_dir / "data" / "source_inputs" / "rows.en.jsonl"
    validated = validate_full_paired_input_contract(config, root_rows_path)
    source_refs, source_evidence = _source_input_refs(
        config,
        root_rows_path=root_rows_path,
        artifact_root=artifact_root,
        validated=validated,
    )
    expected_data_inputs = {config_ref["path"]: config_ref, **source_refs}
    _assert_exact_refs(
        data_manifest.get("inputs"),
        expected_data_inputs,
        context="Hebrew v3 data inputs",
    )

    data_outputs: dict[str, ArtifactRef] = {}
    for split in ("train", "validation", "test"):
        ref = _artifact_ref(
            run_dir / "data" / f"{split}.jsonl",
            artifact_root=artifact_root,
            kind="dataset_split",
            schema_version=PREPARED_EXAMPLE_SCHEMA,
        )
        data_outputs[ref["path"]] = ref
    drop_ref = _artifact_ref(
        run_dir / "data" / "drop_summary.json",
        artifact_root=artifact_root,
        kind="drop_summary",
        schema_version=DROP_SUMMARY_SCHEMA,
    )
    data_outputs[drop_ref["path"]] = drop_ref
    _assert_exact_refs(data_manifest.get("outputs"), data_outputs, context="Hebrew v3 data outputs")

    format_manifest_path = run_dir / "format_manifest.json"
    format_manifest_ref = _artifact_ref(
        format_manifest_path,
        artifact_root=artifact_root,
        kind="manifest",
        schema_version=MANIFEST_SCHEMA,
    )
    if stages.get("format") != format_manifest_ref["path"]:
        raise _error("Hebrew v3 root manifest does not bind the format manifest")
    format_manifest = read_json_with_schema(format_manifest_path, expected_schema=MANIFEST_SCHEMA)
    _assert_stage_manifest(
        format_manifest,
        stage="format",
        run_id=run_id,
        config_sha256=report_config_sha256,
        code_revision=code_revision,
    )
    expected_format_inputs = {
        path: ref for path, ref in data_outputs.items() if ref["kind"] == "dataset_split"
    }
    _assert_exact_refs(
        format_manifest.get("inputs"),
        expected_format_inputs,
        context="Hebrew v3 format inputs",
    )
    formatted_outputs: dict[str, ArtifactRef] = {}
    for split in ("train", "validation", "test"):
        ref = _artifact_ref(
            run_dir / "formatted" / f"{split}.jsonl",
            artifact_root=artifact_root,
            kind="formatted_split",
            schema_version=FORMATTED_EXAMPLE_SCHEMA,
        )
        formatted_outputs[ref["path"]] = ref
    _assert_exact_refs(
        format_manifest.get("outputs"),
        formatted_outputs,
        context="Hebrew v3 format outputs",
    )

    tokenization_manifest_path = run_dir / "tokenization_manifest.json"
    tokenization_manifest_ref = _artifact_ref(
        tokenization_manifest_path,
        artifact_root=artifact_root,
        kind="manifest",
        schema_version=MANIFEST_SCHEMA,
    )
    if stages.get("tokenization") != tokenization_manifest_ref["path"]:
        raise _error("Hebrew v3 root manifest does not bind tokenizer-tax evidence")
    tokenization_manifest = read_json_with_schema(
        tokenization_manifest_path, expected_schema=MANIFEST_SCHEMA
    )
    _assert_stage_manifest(
        tokenization_manifest,
        stage="tokenization",
        run_id=run_id,
        config_sha256=report_config_sha256,
        code_revision=code_revision,
    )
    _assert_exact_refs(
        tokenization_manifest.get("inputs"),
        formatted_outputs,
        context="Hebrew v3 tokenization inputs",
    )
    tokenization_outputs: dict[str, ArtifactRef] = {}
    for filename, kind, schema in (
        (
            TOKENIZER_TAX_RECORDS_FILENAME,
            "tokenizer_tax_records",
            TOKENIZER_TAX_RECORD_SCHEMA,
        ),
        (
            TOKENIZER_TAX_REPORT_FILENAME,
            "tokenizer_tax_report",
            TOKENIZER_TAX_REPORT_SCHEMA,
        ),
    ):
        ref = _artifact_ref(
            run_dir / "analysis" / "tokenization" / filename,
            artifact_root=artifact_root,
            kind=kind,
            schema_version=schema,
        )
        tokenization_outputs[ref["path"]] = ref
    _assert_exact_refs(
        tokenization_manifest.get("outputs"),
        tokenization_outputs,
        context="Hebrew v3 tokenization outputs",
    )
    observed_cohorts = _validate_observed_cohorts(run_dir=run_dir)

    eval_manifest_path = run_dir / "eval-adapter_manifest.json"
    eval_manifest_ref = _artifact_ref(
        eval_manifest_path,
        artifact_root=artifact_root,
        kind="manifest",
        schema_version=MANIFEST_SCHEMA,
    )
    if stages.get("eval-adapter") != eval_manifest_ref["path"]:
        raise _error("Hebrew v3 root manifest does not bind adapter evaluation evidence")
    eval_manifest = read_json_with_schema(eval_manifest_path, expected_schema=MANIFEST_SCHEMA)
    _assert_stage_manifest(
        eval_manifest,
        stage="eval-adapter",
        run_id=run_id,
        config_sha256=report_config_sha256,
        code_revision=code_revision,
    )
    formatted_test_path = (
        (run_dir / "formatted" / "test.jsonl").relative_to(artifact_root).as_posix()
    )
    formatted_test_ref = formatted_outputs[formatted_test_path]
    eval_inputs: dict[str, ArtifactRef] = {formatted_test_ref["path"]: formatted_test_ref}
    for language in ("en", "he"):
        ref = _artifact_ref(
            run_dir / "eval" / "adapter" / f"generations.{language}.jsonl",
            artifact_root=artifact_root,
            kind="generations",
            schema_version=GENERATION_SCHEMA,
        )
        eval_inputs[ref["path"]] = ref
    telemetry_ref = _artifact_ref(
        run_dir / "eval" / "adapter" / INFERENCE_TELEMETRY_FILENAME,
        artifact_root=artifact_root,
        kind="inference_telemetry",
        schema_version=INFERENCE_TELEMETRY_SCHEMA,
    )
    eval_inputs[telemetry_ref["path"]] = telemetry_ref
    _assert_exact_refs(
        eval_manifest.get("inputs"),
        eval_inputs,
        context="Hebrew v3 adapter evaluation inputs",
    )

    return {
        "schema_version": DATA_PROVENANCE_SCHEMA,
        "contract": {
            "seed": 42,
            "root_dataset": {
                "dataset_id": HEBREW_V3_ROOT_DATASET_ID,
                "dataset_revision": HEBREW_V3_ROOT_DATASET_REVISION,
            },
            "paired_dataset": {
                "dataset_id": HEBREW_V3_PAIRED_DATASET_ID,
                "dataset_revision": config.datasets[1].dataset_revision,
            },
            "requested_splits": {"train": 15_000, "validation": 1_000, "test": 1_000},
            "observed_cohorts": observed_cohorts,
            "semantic_review": {
                "sample_size": 200,
                "required_critical_errors": 0,
                "status": "validated",
            },
            "source_code_revision": code_revision,
        },
        "sources": {
            "resolved_config": dict(config_ref),
            "runtime_metadata": dict(runtime_ref),
            "root_manifest": dict(root_manifest_ref),
            "data_manifest": dict(data_manifest_ref),
            "format_manifest": dict(format_manifest_ref),
            "tokenization_manifest": dict(tokenization_manifest_ref),
            "eval_adapter_manifest": dict(eval_manifest_ref),
            **{key: dict(value) for key, value in sorted(source_evidence.items())},
            **{
                f"prepared_{Path(path).stem}": dict(ref)
                for path, ref in sorted(data_outputs.items())
            },
            **{
                f"formatted_{Path(path).stem}": dict(ref)
                for path, ref in sorted(formatted_outputs.items())
            },
            **{
                f"tokenization_{Path(path).stem}": dict(ref)
                for path, ref in sorted(tokenization_outputs.items())
            },
            "v3_inference_telemetry": dict(telemetry_ref),
        },
    }
