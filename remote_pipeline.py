"""Modal entrypoint that runs the sommelier pipeline end to end on a GPU.

Usage:

    uv run modal run remote_pipeline.py --config examples/config.smoke.yaml \
        --mode smoke --max-rows 2500 [--run-id smoke-1]

The remote function writes the config into the artifacts volume, exports
raw rows from the configured Hugging Face dataset, and chains the shared
pipeline stages (data, format, tokenization, base eval, train, adapter eval,
compare)
inside the training image. Artifacts persist on the `sommelier-artifacts`
volume; the Hugging Face cache persists on `sommelier-hf-cache`.

GPU and timeout are read from SOMMELIER_GPU / SOMMELIER_TIMEOUT_SECONDS at
launch time (defaults: A10G, 4 hours). The HF token comes from the local
.env file via a dotenv-backed Modal secret and is never written to
artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import modal

from sommelier.remote.images import (
    PIPELINE_HF_ENV,
    PIPELINE_RUNTIME_VERSIONS,
    train_image,
)

if TYPE_CHECKING:
    from sommelier.config import SommelierConfig

if TYPE_CHECKING:
    from sommelier.runtime_metadata import RemoteExecutionBoundary

APP_NAME = "sommelier-pipeline"
MODAL_MAX_TIMEOUT_SECONDS = 24 * 60 * 60
# A pipeline attempt has no stage-resume contract: retrying after a late
# failure would repeat baseline evaluation and QLoRA training while writing to
# the same run directory. Fail once and require an explicit new run instead of
# silently multiplying compute or obscuring the measured TCO.
PIPELINE_MAX_RETRIES = 0

GPU = os.environ.get("SOMMELIER_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("SOMMELIER_TIMEOUT_SECONDS", str(4 * 60 * 60)))

app = modal.App(APP_NAME)
ARTIFACTS_ROOT = Path("/artifacts")

artifacts_volume = modal.Volume.from_name("sommelier-artifacts", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sommelier-hf-cache", create_if_missing=True)

_INFERENCE_RUNTIME_PACKAGES = tuple(
    package for package, _version in PIPELINE_RUNTIME_VERSIONS if package != "python"
)


def _export_raw_rows(config_path: Path, rows_path: Path, max_rows: int) -> int:
    """Exports the configured HF dataset as raw_tool_call_row.v1 JSONL."""
    from sommelier.config import load_config
    from sommelier.data.export import export_raw_rows

    config = load_config(config_path)
    return export_raw_rows(
        config.root_dataset,
        rows_path,
        seed=config.project.seed,
        max_rows=max_rows,
    )


def _cleanup_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _translation_provenance_sidecar(rows_path: Path, filename: str, language: str) -> Path:
    source_name = Path(filename)
    return rows_path.with_name(f"{source_name.stem}.{language}{source_name.suffix}")


def _wrapped_stages() -> object:
    """Default stages wrapped with timing, GPU cleanup, and volume commits.

    Chaining stages that each load an 8B model in one process needs the
    previous model's CUDA memory released before the next load; committing
    after every stage makes partial progress visible on the volume. The
    tokenization stage persists per-language sequence evidence and rejects
    over-budget training rows before any model loads.
    """
    import time
    from dataclasses import fields

    from sommelier.pipeline import PipelineStages

    base = PipelineStages()

    def wrap(stage_name: str, fn):  # type: ignore[no-untyped-def]
        def inner(paths, config, context, command):  # type: ignore[no-untyped-def]
            print(f"[pipeline] stage {stage_name} starting", flush=True)
            started = time.monotonic()
            outcome = "failed"
            try:
                result = fn(paths, config, context, command)
                outcome = "finished"
                return result
            finally:
                _cleanup_gpu()
                artifacts_volume.commit()
                elapsed = time.monotonic() - started
                print(
                    f"[pipeline] stage {stage_name} {outcome} in {elapsed:.1f}s",
                    flush=True,
                )

        return inner

    kwargs = {
        field.name: wrap(field.name, getattr(base, field.name)) for field in fields(PipelineStages)
    }
    return PipelineStages(**kwargs)


def _stage_paired_rows(
    config_path: Path,
    rows_path: Path,
    translation_run_id: str | None,
    *,
    mode: str,
    max_rows: int,
) -> None:
    """Stages smoke translations or exports pinned full-run paired sources.

    Prepare finds paired sources through the <input stem>.<lang>.jsonl
    convention. Smoke runs may consume an audited translation artifact from
    the artifacts volume. Full runs instead export the paired source at the
    immutable dataset revision in the config, so prepared rows carry the
    published dataset revision and do not depend on mutable staging state.
    """
    import shutil

    from sommelier.config import load_config
    from sommelier.data.load import load_raw_rows
    from sommelier.data.prepare import paired_input_path
    from sommelier.data.split import all_examples, prepare_split_result
    from sommelier.data.translate import (
        PUBLICATION_MANIFEST_FILENAME,
        SUMMARY_FILENAME,
        TranslationStagingContract,
        translation_selection_contract_sha256,
        validate_translation_artifacts,
        validate_translation_publication,
        validate_translation_selection_provenance,
    )
    from sommelier.errors import UserInputError
    from sommelier.pipeline import apply_smoke_overrides

    config = load_config(config_path)
    paired = [source for source in config.datasets if source.source_id_column is not None]
    if not paired:
        return
    if mode not in {"smoke", "full"}:
        raise UserInputError(
            f"unsupported pipeline mode: {mode!r}",
            hint="Choose --mode smoke or --mode full.",
        )
    if mode == "full":
        if translation_run_id is not None:
            raise UserInputError(
                "full pipelines cannot stage rows from --translation-run-id",
                hint=(
                    "Publish the audited paired dataset, pin its immutable revision "
                    "under datasets, and omit --translation-run-id."
                ),
            )
        for source in paired:
            if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source.dataset_revision) is None:
                raise UserInputError(
                    f"full pipeline paired dataset {source.dataset_id!r} does not use "
                    "an immutable revision",
                    hint="Pin dataset_revision to the exact Hugging Face dataset commit SHA.",
                )
        from huggingface_hub import hf_hub_download

        from sommelier.data.export import export_raw_rows
        from sommelier.data.semantic_review import (
            SEMANTIC_REVIEW_FILENAME,
            SEMANTIC_REVIEW_TEMPLATE_FILENAME,
            root_split_assignments,
        )

        root_rows = load_raw_rows(rows_path)
        root_result = prepare_split_result(
            root_rows,
            min_query_chars=config.data.min_query_chars,
            max_query_chars=config.data.max_query_chars,
            n_train=config.data.n_train,
            n_validation=config.data.n_validation,
            n_test=config.data.n_test,
            seed=config.project.seed,
            language=config.root_dataset.language,
        )
        selected_ids = {example["example_id"] for example in all_examples(root_result)}
        ordered_selected_ids = [
            str(row["source_id"]) for row in root_rows if row["source_id"] in selected_ids
        ]
        expected_selection = TranslationStagingContract(
            selection_contract_sha256=translation_selection_contract_sha256(
                config,
                mode="full",
                max_rows=max_rows,
                limit=0,
            ),
            mode="full",
            seed=config.project.seed,
            max_rows=max_rows,
            selected_rows=len(ordered_selected_ids),
            selected_source_ids_sha256=hashlib.sha256(
                "\n".join(ordered_selected_ids).encode("utf-8")
            ).hexdigest(),
            limit=0,
        )
        root_split_by_id = root_split_assignments(config, root_rows)

        for source in paired:
            target = paired_input_path(rows_path, source.language)
            exported = export_raw_rows(
                source,
                target,
                seed=config.project.seed,
                # The published paired dataset is already the audited survivor
                # set. Export it in full; split inheritance selects the cohort.
                max_rows=0,
            )
            if exported <= 0:
                raise UserInputError(
                    f"published paired dataset {source.dataset_id!r} exported no rows",
                    hint="Verify the pinned dataset revision before launching the pipeline.",
                )
            summary_download = Path(
                hf_hub_download(
                    repo_id=source.dataset_id,
                    filename=SUMMARY_FILENAME,
                    repo_type="dataset",
                    revision=source.dataset_revision,
                )
            )
            publication_download = Path(
                hf_hub_download(
                    repo_id=source.dataset_id,
                    filename=PUBLICATION_MANIFEST_FILENAME,
                    repo_type="dataset",
                    revision=source.dataset_revision,
                )
            )
            semantic_review_download: Path | None = None
            semantic_review_template_download: Path | None = None
            if source.language == "he":
                semantic_review_download = Path(
                    hf_hub_download(
                        repo_id=source.dataset_id,
                        filename=SEMANTIC_REVIEW_FILENAME,
                        repo_type="dataset",
                        revision=source.dataset_revision,
                    )
                )
                semantic_review_template_download = Path(
                    hf_hub_download(
                        repo_id=source.dataset_id,
                        filename=SEMANTIC_REVIEW_TEMPLATE_FILENAME,
                        repo_type="dataset",
                        revision=source.dataset_revision,
                    )
                )
            validate_translation_selection_provenance(
                summary_path=summary_download,
                root_rows_path=rows_path,
                target_language=source.language,
                expected=expected_selection,
            )
            validate_translation_publication(
                translated_rows_path=target,
                summary_path=summary_download,
                publication_manifest_path=publication_download,
                target_language=source.language,
                require_full_provenance=True,
                semantic_review_path=semantic_review_download,
                semantic_review_template_path=semantic_review_template_download,
                root_rows_path=rows_path,
                root_split_by_id=root_split_by_id,
                expected_seed=config.project.seed,
            )
            shutil.copy2(
                summary_download,
                _translation_provenance_sidecar(
                    rows_path,
                    SUMMARY_FILENAME,
                    source.language,
                ),
            )
            shutil.copy2(
                publication_download,
                _translation_provenance_sidecar(
                    rows_path,
                    PUBLICATION_MANIFEST_FILENAME,
                    source.language,
                ),
            )
            if (
                semantic_review_download is not None
                and semantic_review_template_download is not None
            ):
                shutil.copy2(
                    semantic_review_download,
                    _translation_provenance_sidecar(
                        rows_path,
                        SEMANTIC_REVIEW_FILENAME,
                        source.language,
                    ),
                )
                shutil.copy2(
                    semantic_review_template_download,
                    _translation_provenance_sidecar(
                        rows_path,
                        SEMANTIC_REVIEW_TEMPLATE_FILENAME,
                        source.language,
                    ),
                )
            print(
                f"[pipeline] exported {exported} published {source.language} rows "
                f"from {source.dataset_id}@{source.dataset_revision}",
                flush=True,
            )
        return
    if translation_run_id is None:
        raise UserInputError(
            "the smoke config declares paired sources but no translation run was named",
            hint="Pass --translation-run-id <id> of a completed smoke translation.",
        )
    selection_config = apply_smoke_overrides(config) if mode == "smoke" else config
    root_rows = load_raw_rows(rows_path)
    root_result = prepare_split_result(
        root_rows,
        min_query_chars=selection_config.data.min_query_chars,
        max_query_chars=selection_config.data.max_query_chars,
        n_train=selection_config.data.n_train,
        n_validation=selection_config.data.n_validation,
        n_test=selection_config.data.n_test,
        seed=selection_config.project.seed,
        language=selection_config.root_dataset.language,
    )
    selected_ids = {example["example_id"] for example in all_examples(root_result)}
    ordered_selected_ids = [
        str(row["source_id"]) for row in root_rows if row["source_id"] in selected_ids
    ]
    selected_source_ids_sha256 = hashlib.sha256(
        "\n".join(ordered_selected_ids).encode("utf-8")
    ).hexdigest()
    expected = TranslationStagingContract(
        selection_contract_sha256=translation_selection_contract_sha256(
            selection_config,
            mode=mode,  # type: ignore[arg-type]
            max_rows=max_rows,
            limit=0,
        ),
        mode=mode,  # type: ignore[arg-type]
        seed=selection_config.project.seed,
        max_rows=max_rows,
        selected_rows=len(ordered_selected_ids),
        selected_source_ids_sha256=selected_source_ids_sha256,
        # Diagnostic translation prefixes are never complete pipeline inputs.
        limit=0,
    )
    for source in paired:
        translated = (
            Path("/artifacts/translation") / translation_run_id / f"rows.{source.language}.jsonl"
        )
        summary_path = translated.parent / "translation_summary.json"
        validate_translation_artifacts(
            summary_path=summary_path,
            translated_rows_path=translated,
            root_rows_path=rows_path,
            target_language=source.language,
            expected=expected,
        )
        publication_path = translated.parent / PUBLICATION_MANIFEST_FILENAME
        validate_translation_publication(
            translated_rows_path=translated,
            summary_path=summary_path,
            publication_manifest_path=publication_path,
            target_language=source.language,
        )
        target = paired_input_path(rows_path, source.language)
        shutil.copy2(translated, target)
        shutil.copy2(
            summary_path,
            _translation_provenance_sidecar(rows_path, SUMMARY_FILENAME, source.language),
        )
        shutil.copy2(
            publication_path,
            _translation_provenance_sidecar(
                rows_path,
                PUBLICATION_MANIFEST_FILENAME,
                source.language,
            ),
        )
        print(f"[pipeline] staged paired rows: {target}", flush=True)


def _package_versions() -> dict[str, str]:
    from importlib.metadata import version
    from platform import python_version

    versions: dict[str, str] = {"python": python_version()}
    for package in _INFERENCE_RUNTIME_PACKAGES:
        try:
            versions[package] = version(package)
        except Exception:
            versions[package] = "absent"
    return versions


def _validate_hebrew_v3_full_runtime(
    config: SommelierConfig,
    *,
    mode: str,
    package_versions: dict[str, str],
) -> None:
    """Require the probe-established runtime before full Hebrew data access.

    Smoke runs intentionally record whatever environment they observe so a new
    compatible stack can be probed. Full Hebrew evidence instead fails closed
    on any Python or distribution drift before export, paired-source staging,
    or model loading can begin.
    """
    from sommelier.errors import UserInputError
    from sommelier.evaluation.data_provenance import is_hebrew_v3_config

    if mode != "full" or not is_hebrew_v3_config(config):
        return

    expected = dict(PIPELINE_RUNTIME_VERSIONS)
    if package_versions == expected:
        return
    differences = [
        f"{package}: expected {expected.get(package, 'absent')}, "
        f"observed {package_versions.get(package, 'absent')}"
        for package in sorted(set(expected) | set(package_versions))
        if package_versions.get(package) != expected.get(package)
    ]
    raise UserInputError(
        "Hebrew v3 full-run runtime does not match the preregistered pipeline "
        f"environment ({'; '.join(differences)})",
        hint=(
            "Rebuild the pinned Modal train image and rerun a smoke probe. Do not "
            "export the full corpus or load a model under a different environment."
        ),
    )


def _apply_pipeline_hf_policy() -> None:
    """Force the image-declared Hugging Face download policy at runtime."""
    for name, value in PIPELINE_HF_ENV:
        os.environ[name] = value


def _required_pipeline_timeout_seconds(
    *,
    data_timeout_seconds: int,
    train_timeout_seconds: int,
    eval_timeout_seconds: int,
    trains_adapter: bool,
) -> int:
    """Return the outer-timeout admission floor from planning estimates.

    The ``*_timeout_seconds`` names are retained for config compatibility. They
    are planning estimates, not enforced per-stage watchdogs: the current
    pipeline is one Modal function and only its outer timeout is provider
    enforced. The estimate includes two sequential evaluations, plus training
    unless a pre-existing adapter was supplied.
    """
    return (
        data_timeout_seconds
        + (train_timeout_seconds if trains_adapter else 0)
        + 2 * eval_timeout_seconds
    )


def _remote_execution_boundary(
    *,
    function_timeout_seconds: int,
    gpu_allocation_label: str,
    stage_planning_estimate_seconds: int,
) -> RemoteExecutionBoundary:
    """Build truthful, machine-readable metadata for the timeout boundary."""
    from sommelier.runtime_metadata import RemoteExecutionBoundary

    planning_headroom_seconds = function_timeout_seconds - stage_planning_estimate_seconds
    return RemoteExecutionBoundary(
        provider="modal",
        function_timeout_seconds=function_timeout_seconds,
        gpu_allocation_label=gpu_allocation_label,
        configured_stage_planning_estimate_seconds=stage_planning_estimate_seconds,
        outer_timeout_planning_headroom_seconds=planning_headroom_seconds,
        per_stage_watchdogs_enforced=False,
        hf_hub_download_policy={
            "disable_xet": dict(PIPELINE_HF_ENV)["HF_HUB_DISABLE_XET"] == "1",
            "download_timeout_seconds": int(dict(PIPELINE_HF_ENV)["HF_HUB_DOWNLOAD_TIMEOUT"]),
            "boundary": (
                "Forced by the pipeline image and again at remote function entry "
                "before Hugging Face dataset or model access."
            ),
        },
        boundary=(
            "Modal enforces the outer function timeout only; legacy-named "
            "remote.*_timeout_seconds values are planning estimates, not "
            "per-stage watchdogs (per_stage_watchdogs_enforced=false); "
            "pipeline stage timers include wrapper-owned GPU cleanup and "
            "artifact-volume commits, while provider billing is separate evidence."
        ),
    )


def _validate_hebrew_v3_full_request(
    config: SommelierConfig,
    *,
    mode: str,
    max_rows: int,
    adapter_id: str | None,
    adapter_revision: str | None,
    translation_run_id: str | None,
) -> None:
    """Fail closed before a paid Hebrew v3 full run reaches data or models.

    Smoke and non-Hebrew configurations deliberately remain flexible.  A
    Hebrew v3 full request is either the no-external-adapter v3 training arm,
    or the one immutable published v1 baseline arm.
    """
    from sommelier.errors import EvaluationError, UserInputError
    from sommelier.evaluation.data_provenance import (
        HEBREW_V3_FULL_MAX_ROWS,
        HEBREW_V3_V1_ADAPTER_ID,
        HEBREW_V3_V1_ADAPTER_REVISION,
        is_hebrew_v3_config,
        validate_hebrew_v3_preregistered_config,
    )

    if mode != "full":
        return
    if not is_hebrew_v3_config(config):
        return

    try:
        validate_hebrew_v3_preregistered_config(config)
    except EvaluationError as error:
        raise UserInputError(
            f"Hebrew v3 full-run preregistration failed: {error}",
            hint=error.hint,
        ) from error

    if max_rows != HEBREW_V3_FULL_MAX_ROWS:
        raise UserInputError(
            "Hebrew v3 full runs require the preregistered "
            f"--max-rows {HEBREW_V3_FULL_MAX_ROWS}, got {max_rows}",
            hint="Use --max-rows 60000 for both full evaluation arms.",
        )
    if translation_run_id is not None:
        raise UserInputError(
            "Hebrew v3 full runs cannot use --translation-run-id",
            hint="Pin the audited Hebrew dataset commit in the full config instead.",
        )

    observed_adapter = (adapter_id, adapter_revision)
    if observed_adapter == (None, None):
        return
    required_adapter = (HEBREW_V3_V1_ADAPTER_ID, HEBREW_V3_V1_ADAPTER_REVISION)
    if observed_adapter != required_adapter:
        raise UserInputError(
            "Hebrew v3 full external-adapter runs require the exact preregistered "
            "v1 baseline adapter",
            hint=(
                f"Use {HEBREW_V3_V1_ADAPTER_ID}@{HEBREW_V3_V1_ADAPTER_REVISION}, "
                "or omit both adapter arguments to train the v3 arm."
            ),
        )


def _validate_remote_launch_boundary(
    config: SommelierConfig,
    *,
    mode: str,
    adapter_id: str | None,
    adapter_revision: str | None,
    code_revision: str,
    source_tree_clean: bool | None,
    gpu_allocation_label: str,
    function_timeout_seconds: int,
) -> int:
    """Reject every locally knowable launch error before Modal dispatch.

    The same pure boundary runs again inside the remote function because it is
    also a callable API surface. It returns the configured sequential planning
    estimate used to describe, but not enforce, individual stage budgets.
    """
    from sommelier.errors import UserInputError

    if mode not in {"smoke", "full"}:
        raise UserInputError(
            f"unsupported pipeline mode: {mode!r}",
            hint="Choose --mode smoke or --mode full.",
        )
    if adapter_id is None and adapter_revision is not None:
        raise UserInputError(
            "--adapter-revision requires --adapter-id",
            hint="Supply both adapter arguments, or omit both to train a new adapter.",
        )
    if config.remote.gpu != gpu_allocation_label:
        raise UserInputError(
            f"remote.gpu={config.remote.gpu!r} does not match the Modal allocation "
            f"{gpu_allocation_label!r}",
            hint="Set SOMMELIER_GPU to the exact remote.gpu value before modal run.",
        )
    configured_stage_planning_estimate = _required_pipeline_timeout_seconds(
        data_timeout_seconds=config.remote.data_timeout_seconds,
        train_timeout_seconds=config.remote.train_timeout_seconds,
        eval_timeout_seconds=config.remote.eval_timeout_seconds,
        trains_adapter=adapter_id is None,
    )
    if function_timeout_seconds > MODAL_MAX_TIMEOUT_SECONDS:
        raise UserInputError(
            f"Modal timeout {function_timeout_seconds}s exceeds the provider maximum "
            f"{MODAL_MAX_TIMEOUT_SECONDS}s",
            hint="Split the pipeline across chained remote functions.",
        )
    if function_timeout_seconds < configured_stage_planning_estimate:
        raise UserInputError(
            f"Modal outer timeout {function_timeout_seconds}s is below the configured "
            "sequential stage planning estimate "
            f"{configured_stage_planning_estimate}s",
            hint=(
                "Raise SOMMELIER_TIMEOUT_SECONDS before modal run. The legacy-named "
                "remote.*_timeout_seconds fields only form this admission estimate; "
                "they do not install per-stage watchdogs."
            ),
        )
    if mode == "full" and (
        re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", code_revision) is None
        or source_tree_clean is not True
    ):
        raise UserInputError(
            "full evidence runs require a clean, immutable local Git revision",
            hint="Commit the v3 implementation and launch again from a clean worktree.",
        )
    return configured_stage_planning_estimate


@app.function(
    retries=PIPELINE_MAX_RETRIES,
    image=train_image(),
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_dotenv(Path(__file__).parent)],
    volumes={"/artifacts": artifacts_volume, "/hf-cache": hf_cache_volume},
)
def run_remote_pipeline(
    config_yaml: str,
    mode: str,
    max_rows: int,
    run_id: str | None = None,
    adapter_id: str | None = None,
    adapter_revision: str | None = None,
    translation_run_id: str | None = None,
    code_revision: str = "unknown",
    source_tree_clean: bool | None = None,
    allocation_gpu: str | None = None,
    function_timeout_seconds: int | None = None,
) -> dict[str, object]:
    _apply_pipeline_hf_policy()
    os.environ.setdefault("HF_HOME", "/hf-cache")
    # Long-sequence batches fragment the allocator; expandable segments
    # avoid OOM from reserved-but-unallocated blocks (set before torch
    # initializes CUDA inside the stages).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from sommelier.pipeline import run_pipeline
    from sommelier.runtime_metadata import SourceCodeProvenance

    work = ARTIFACTS_ROOT
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / f"config-{mode}.yaml"
    config_path.write_text(config_yaml, encoding="utf-8")

    from sommelier.config import load_config

    config = load_config(config_path)
    resolved_allocation_gpu = allocation_gpu or GPU
    resolved_function_timeout = (
        TIMEOUT_SECONDS if function_timeout_seconds is None else function_timeout_seconds
    )
    _validate_hebrew_v3_full_request(
        config,
        mode=mode,
        max_rows=max_rows,
        adapter_id=adapter_id,
        adapter_revision=adapter_revision,
        translation_run_id=translation_run_id,
    )
    configured_stage_planning_estimate = _validate_remote_launch_boundary(
        config,
        mode=mode,
        adapter_id=adapter_id,
        adapter_revision=adapter_revision,
        code_revision=code_revision,
        source_tree_clean=source_tree_clean,
        gpu_allocation_label=resolved_allocation_gpu,
        function_timeout_seconds=resolved_function_timeout,
    )
    package_versions = _package_versions()
    _validate_hebrew_v3_full_runtime(
        config,
        mode=mode,
        package_versions=package_versions,
    )
    print(f"[pipeline] package versions: {package_versions}", flush=True)
    planning_headroom = resolved_function_timeout - configured_stage_planning_estimate
    print(
        "[pipeline] outer timeout admission: "
        f"configured_stage_planning_estimate="
        f"{configured_stage_planning_estimate}s, "
        f"outer_function_timeout={resolved_function_timeout}s, "
        f"planning_headroom={planning_headroom}s, "
        "per_stage_watchdogs_enforced=false",
        flush=True,
    )
    if code_revision != "unknown":
        os.environ["SOMMELIER_GIT_COMMIT"] = code_revision

    print(f"[pipeline] exporting raw rows (max_rows={max_rows})", flush=True)
    rows_path = Path("/tmp/raw_rows.jsonl")
    exported = _export_raw_rows(config_path, rows_path, max_rows)
    print(f"[pipeline] exported {exported} raw rows", flush=True)
    hf_cache_volume.commit()

    _stage_paired_rows(
        config_path,
        rows_path,
        translation_run_id,
        mode=mode,
        max_rows=max_rows,
    )

    try:
        resolved_run_id = run_pipeline(
            config_path,
            mode=mode,  # type: ignore[arg-type]
            input_path=rows_path,
            run_id=run_id,
            project_root=work,
            stages=_wrapped_stages(),  # type: ignore[arg-type]
            adapter_id=adapter_id,
            adapter_revision=adapter_revision,
            package_versions=package_versions,
            source_code=SourceCodeProvenance(
                git_commit=code_revision,
                working_tree_clean=source_tree_clean,
                boundary=("Measured by the local launcher immediately before Modal dispatch."),
            ),
            remote_execution=_remote_execution_boundary(
                function_timeout_seconds=resolved_function_timeout,
                gpu_allocation_label=resolved_allocation_gpu,
                stage_planning_estimate_seconds=configured_stage_planning_estimate,
            ),
        )
    finally:
        artifacts_volume.commit()
        hf_cache_volume.commit()

    run_dir = work / "artifacts" / "runs" / resolved_run_id
    comparison = json.loads(
        (run_dir / "report" / "comparison_report.json").read_text(encoding="utf-8")
    )
    runtime = json.loads((run_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    return {
        "run_id": resolved_run_id,
        # The config value is authoritative; the module-level GPU default
        # is not visible inside the container.
        "gpu": load_config(config_path).remote.gpu,
        "raw_rows": exported,
        "versions": package_versions,
        "metrics": {
            "base": comparison["base"]["metrics"],
            "adapter": comparison["adapter"]["metrics"],
            "deltas": comparison["deltas"],
        },
        "stage_seconds": {
            name: value["elapsed_seconds"] for name, value in runtime["stages"].items()
        },
        "report_path": f"runs/{resolved_run_id}/report/comparison_report.md",
    }


@app.local_entrypoint()
def main(
    config: str = "examples/config.smoke.yaml",
    mode: str = "smoke",
    max_rows: int = 2500,
    run_id: str = "",
    adapter_id: str = "",
    adapter_revision: str = "",
    translation_run_id: str = "",
) -> None:
    from sommelier.config import load_config
    from sommelier.manifests import get_git_commit, get_git_worktree_clean

    config_path = Path(config)
    config_yaml = config_path.read_text(encoding="utf-8")
    resolved_config = load_config(config_path)
    resolved_adapter_id = adapter_id or None
    resolved_adapter_revision = adapter_revision or None
    resolved_translation_run_id = translation_run_id or None
    code_revision = get_git_commit()
    source_tree_clean = get_git_worktree_clean()
    _validate_hebrew_v3_full_request(
        resolved_config,
        mode=mode,
        max_rows=max_rows,
        adapter_id=resolved_adapter_id,
        adapter_revision=resolved_adapter_revision,
        translation_run_id=resolved_translation_run_id,
    )
    _validate_remote_launch_boundary(
        resolved_config,
        mode=mode,
        adapter_id=resolved_adapter_id,
        adapter_revision=resolved_adapter_revision,
        code_revision=code_revision,
        source_tree_clean=source_tree_clean,
        gpu_allocation_label=GPU,
        function_timeout_seconds=TIMEOUT_SECONDS,
    )
    result = run_remote_pipeline.remote(
        config_yaml,
        mode,
        max_rows,
        run_id or None,
        resolved_adapter_id,
        resolved_adapter_revision,
        resolved_translation_run_id,
        code_revision,
        source_tree_clean,
        GPU,
        TIMEOUT_SECONDS,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
