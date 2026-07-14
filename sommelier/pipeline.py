from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from sommelier.config import SommelierConfig, load_config
from sommelier.errors import UserInputError
from sommelier.evaluation.generate import AdapterRef, run_generation
from sommelier.evaluation.report import compare_evaluations, write_evaluation_report
from sommelier.formatting.chat import build_formatted_splits
from sommelier.manifests import create_run_id, set_run_manifest_status
from sommelier.run_context import RunContext, ensure_run_context
from sommelier.runtime_metadata import (
    RemoteExecutionBoundary,
    SourceCodeProvenance,
    initialize_runtime_metadata,
    peak_memory_from_training_metrics,
    record_peak_gpu_memory,
    record_stage_runtime,
)
from sommelier.training.authorization import (
    FullPairedInputValidationCapability,
    _FullPairedInputValidationReceipt,
    _issue_full_paired_input_for_training,
    _validate_full_paired_input_for_pipeline,
)
from sommelier.training.metrics import METRICS_FILENAME
from sommelier.training.qlora import train_adapter

PipelineMode = Literal["smoke", "full"]

SMOKE_MAX_TRAIN: Final = 100
SMOKE_MAX_VALIDATION: Final = 20
SMOKE_MAX_TEST: Final = 20
PIPELINE_RUN_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

StageFn = Callable[["PipelinePaths", SommelierConfig, RunContext, list[str]], None]


@dataclass(frozen=True)
class PipelinePaths:
    """Stage directories inside one run, per the required artifact layout.

    ``external_adapter`` switches the run into baseline shape: the train
    stage is skipped and the adapter evaluation loads the referenced
    published adapter instead of this run's train output.
    """

    input_path: Path
    data_dir: Path
    formatted_dir: Path
    tokenization_dir: Path
    train_dir: Path
    eval_base_dir: Path
    eval_adapter_dir: Path
    report_dir: Path
    external_adapter: AdapterRef | None = None
    full_paired_input_validation_receipt: _FullPairedInputValidationReceipt | None = None


def _stage_prepare(
    paths: PipelinePaths,
    config: SommelierConfig,
    context: RunContext,
    command: list[str],
) -> None:
    import shutil

    from sommelier.artifacts import make_artifact_ref
    from sommelier.data.prepare import paired_input_path, prepare_dataset_from_file
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
        TRANSLATION_SUMMARY_SCHEMA,
    )

    source_dir = paths.data_dir / "source_inputs"
    source_dir.mkdir(parents=True, exist_ok=True)
    root_language = config.root_dataset.language
    staged_root = source_dir / f"rows.{root_language}.jsonl"
    shutil.copy2(paths.input_path, staged_root)
    source_inputs = [
        make_artifact_ref(
            staged_root,
            artifact_root=context.artifact_root,
            kind="raw_dataset",
            schema_version="sommelier.raw_tool_call_row.v1",
        )
    ]
    for source in config.datasets:
        if source.source_id_column is None:
            continue
        paired_source = paired_input_path(paths.input_path, source.language)
        paired_target = paired_input_path(staged_root, source.language)
        shutil.copy2(paired_source, paired_target)
        source_inputs.append(
            make_artifact_ref(
                paired_target,
                artifact_root=context.artifact_root,
                kind="raw_paired_dataset",
                schema_version="sommelier.raw_tool_call_row.v1",
            )
        )
        for filename, schema_version, kind in (
            (SUMMARY_FILENAME, TRANSLATION_SUMMARY_SCHEMA, "translation_summary"),
            (
                PUBLICATION_MANIFEST_FILENAME,
                TRANSLATION_PUBLICATION_SCHEMA,
                "translation_publication_manifest",
            ),
        ):
            name = Path(filename)
            provenance_source = paths.input_path.with_name(
                f"{name.stem}.{source.language}{name.suffix}"
            )
            if not provenance_source.exists():
                raise UserInputError(
                    f"paired dataset provenance not found: {provenance_source}",
                    hint=(
                        "Stage the audited translation summary and publication manifest "
                        "with the paired raw rows."
                    ),
                )
            provenance_target = source_dir / provenance_source.name
            shutil.copy2(provenance_source, provenance_target)
            source_inputs.append(
                make_artifact_ref(
                    provenance_target,
                    artifact_root=context.artifact_root,
                    kind=kind,
                    schema_version=schema_version,
                )
            )
        if source.language == "he":
            config_name = Path(TRANSLATION_CONFIG_FILENAME)
            phase_a_config_source = paths.input_path.with_name(
                f"{config_name.stem}.{source.language}{config_name.suffix}"
            )
            if not phase_a_config_source.exists():
                raise UserInputError(
                    f"paired dataset Phase-A config not found: {phase_a_config_source}",
                    hint=(
                        "Stage translation_config.yaml from the exact published Hebrew "
                        "dataset revision."
                    ),
                )
            phase_a_config_target = source_dir / phase_a_config_source.name
            shutil.copy2(phase_a_config_source, phase_a_config_target)
            source_inputs.append(
                make_artifact_ref(
                    phase_a_config_target,
                    artifact_root=context.artifact_root,
                    kind="config",
                    schema_version="sommelier.config.v2",
                )
            )
            identity_name = Path(TRANSLATION_RUN_IDENTITY_FILENAME)
            identity_source = paths.input_path.with_name(
                f"{identity_name.stem}.{source.language}{identity_name.suffix}"
            )
            if not identity_source.exists():
                raise UserInputError(
                    f"paired dataset pre-provider identity not found: {identity_source}",
                    hint=(
                        "Stage translation_run_identity.json from the exact published Hebrew "
                        "dataset revision."
                    ),
                )
            identity_target = source_dir / identity_source.name
            shutil.copy2(identity_source, identity_target)
            source_inputs.append(
                make_artifact_ref(
                    identity_target,
                    artifact_root=context.artifact_root,
                    kind="translation_run_identity",
                    schema_version="sommelier.translation_run_identity.v1",
                )
            )
        for filename, schema_version, kind in (
            (
                SEMANTIC_REVIEW_FILENAME,
                SEMANTIC_REVIEW_SCHEMA,
                "translation_semantic_review",
            ),
            (
                SEMANTIC_REVIEW_TEMPLATE_FILENAME,
                SEMANTIC_REVIEW_TEMPLATE_SCHEMA,
                "translation_semantic_review_template",
            ),
        ):
            semantic_name = Path(filename)
            semantic_source = paths.input_path.with_name(
                f"{semantic_name.stem}.{source.language}{semantic_name.suffix}"
            )
            if semantic_source.exists():
                semantic_target = source_dir / semantic_source.name
                shutil.copy2(semantic_source, semantic_target)
                source_inputs.append(
                    make_artifact_ref(
                        semantic_target,
                        artifact_root=context.artifact_root,
                        kind=kind,
                        schema_version=schema_version,
                    )
                )

    prepare_dataset_from_file(
        config,
        input_path=staged_root,
        out_dir=paths.data_dir,
        context=context,
        command=command,
        use_gpu=False,
        source_inputs=source_inputs,
    )


def _stage_format(
    paths: PipelinePaths,
    config: SommelierConfig,
    context: RunContext,
    command: list[str],
) -> None:
    build_formatted_splits(
        config,
        data_dir=paths.data_dir,
        out_dir=paths.formatted_dir,
        context=context,
        command=command,
    )


def _stage_tokenization(
    paths: PipelinePaths,
    config: SommelierConfig,
    context: RunContext,
    command: list[str],
) -> None:
    import json

    from sommelier.analysis.tokenization import (
        TOKENIZER_TAX_REPORT_FILENAME,
        analyze_tokenizer_tax,
    )

    analyze_tokenizer_tax(
        config,
        formatted_dir=paths.formatted_dir,
        out_dir=paths.tokenization_dir,
        context=context,
        command=command,
    )
    report = json.loads(
        (paths.tokenization_dir / TOKENIZER_TAX_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    violations: dict[str, int] = {}
    for language in config.train.languages:
        count = sum(
            int(report["languages"][language]["splits"][split]["over_budget"])
            for split in ("train", "validation")
        )
        if count:
            violations[language] = count
    if violations:
        detail = ", ".join(f"{language}={count}" for language, count in sorted(violations.items()))
        raise UserInputError(
            f"formatted training sequences exceed train.max_sequence_length "
            f"{config.train.max_sequence_length} ({detail})",
            hint="Inspect analysis/tokenization/tokenizer_tax_report.json and raise "
            "the sequence budget or revise the predeclared experiment config.",
        )


def _stage_eval_base(
    paths: PipelinePaths,
    config: SommelierConfig,
    context: RunContext,
    command: list[str],
) -> None:
    run_generation(
        config,
        formatted_dir=paths.formatted_dir,
        out_dir=paths.eval_base_dir,
        model_kind="base",
        context=context,
        command=command,
    )
    write_evaluation_report(
        config,
        formatted_dir=paths.formatted_dir,
        eval_dir=paths.eval_base_dir,
        model_kind="base",
        context=context,
        command=command,
    )


def _stage_train(
    paths: PipelinePaths,
    config: SommelierConfig,
    context: RunContext,
    command: list[str],
) -> None:
    if paths.external_adapter is not None:
        # Baseline shape: the run evaluates a published adapter, so there
        # is nothing to train and no train manifest to record.
        print(
            f"[pipeline] skipping train stage: evaluating external adapter "
            f"{paths.external_adapter.source}",
            flush=True,
        )
        return
    full_paired_input_validation: FullPairedInputValidationCapability | None = (
        _issue_full_paired_input_for_training(
            paths.full_paired_input_validation_receipt,
            config,
            context,
            paths.formatted_dir,
            paths.data_dir / "source_inputs" / f"rows.{config.root_dataset.language}.jsonl",
        )
    )
    train_adapter(
        config,
        paths.formatted_dir,
        paths.train_dir,
        context=context,
        command=command,
        full_paired_input_validation=full_paired_input_validation,
    )


def _stage_eval_adapter(
    paths: PipelinePaths,
    config: SommelierConfig,
    context: RunContext,
    command: list[str],
) -> None:
    adapter = paths.external_adapter or AdapterRef(source=str(paths.train_dir))
    run_generation(
        config,
        formatted_dir=paths.formatted_dir,
        out_dir=paths.eval_adapter_dir,
        model_kind="adapter",
        adapter=adapter,
        context=context,
        command=command,
    )
    write_evaluation_report(
        config,
        formatted_dir=paths.formatted_dir,
        eval_dir=paths.eval_adapter_dir,
        model_kind="adapter",
        context=context,
        command=command,
        adapter=adapter,
    )


def _stage_compare(
    paths: PipelinePaths,
    config: SommelierConfig,
    context: RunContext,
    command: list[str],
) -> None:
    compare_evaluations(
        paths.eval_base_dir,
        paths.eval_adapter_dir,
        paths.report_dir,
        command=command,
    )


@dataclass
class PipelineStages:
    """The ordered stage callables; tests inject stubs to verify chaining."""

    prepare: StageFn = field(default=_stage_prepare)
    format: StageFn = field(default=_stage_format)
    tokenization: StageFn = field(default=_stage_tokenization)
    eval_base: StageFn = field(default=_stage_eval_base)
    train: StageFn = field(default=_stage_train)
    eval_adapter: StageFn = field(default=_stage_eval_adapter)
    compare: StageFn = field(default=_stage_compare)

    def ordered(self) -> list[tuple[str, StageFn]]:
        return [
            ("data", self.prepare),
            ("format", self.format),
            ("tokenization", self.tokenization),
            ("eval-base", self.eval_base),
            ("train", self.train),
            ("eval-adapter", self.eval_adapter),
            ("compare", self.compare),
        ]


def apply_smoke_overrides(config: SommelierConfig) -> SommelierConfig:
    """Bounds split sizes for smoke runs without touching other settings."""
    bounded = config.model_copy(deep=True)
    bounded.data.n_train = min(bounded.data.n_train, SMOKE_MAX_TRAIN)
    bounded.data.n_validation = min(bounded.data.n_validation, SMOKE_MAX_VALIDATION)
    bounded.data.n_test = min(bounded.data.n_test, SMOKE_MAX_TEST)
    return bounded


def pipeline_run_id(mode: PipelineMode, run_id: str | None = None) -> str:
    """Builds the run ID; smoke runs get their own prefix so a later full
    run can never overwrite smoke artifacts."""
    base = create_run_id() if run_id is None else run_id
    if PIPELINE_RUN_ID_PATTERN.fullmatch(base) is None:
        raise UserInputError(
            f"invalid pipeline run id: {base!r}",
            hint=(
                "Use 1-128 ASCII letters, digits, dots, underscores, or hyphens; "
                "the first character must be alphanumeric."
            ),
        )
    if mode == "smoke" and not base.startswith("smoke-"):
        return f"smoke-{base}"
    return base


def run_pipeline(
    config_path: Path,
    *,
    mode: PipelineMode,
    input_path: Path,
    run_id: str | None = None,
    project_root: Path | None = None,
    stages: PipelineStages | None = None,
    adapter_id: str | None = None,
    adapter_revision: str | None = None,
    package_versions: dict[str, str] | None = None,
    source_code: SourceCodeProvenance | None = None,
    remote_execution: RemoteExecutionBoundary | None = None,
) -> str:
    """Chains data, format, tokenization, baseline eval, train, adapter eval, and compare.

    Every stage reads and writes inside one run directory under the
    configured artifact root; smoke mode bounds the split sizes through a
    resolved config override and uses a separate run ID namespace. With
    ``adapter_id`` the run takes the baseline shape: training is skipped
    and the adapter evaluation loads the referenced published adapter.
    Stage failures propagate as SommelierError subclasses with their
    documented exit codes; nothing is retried.
    """
    if mode not in {"smoke", "full"}:
        raise UserInputError(
            f"unsupported pipeline mode: {mode!r}",
            hint="Choose --mode smoke or --mode full.",
        )
    resolved_run_id = pipeline_run_id(mode, run_id)
    if not input_path.exists():
        raise UserInputError(
            f"raw input file not found: {input_path}",
            hint="Pass --input with a sommelier.raw_tool_call_row.v1 JSONL file.",
        )

    config = load_config(config_path)
    full_paired_input_validation_receipt = None
    if mode == "smoke":
        config = apply_smoke_overrides(config)
    else:
        full_paired_input_validation_receipt = _validate_full_paired_input_for_pipeline(
            config,
            input_path,
        )
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id=resolved_run_id,
        project_root=project_root or Path.cwd(),
        reject_existing_run=mode == "full",
    )
    external_adapter = None
    if adapter_id is not None:
        adapter_path = Path(adapter_id)
        external_adapter = AdapterRef(
            source=str(adapter_path.resolve()) if adapter_path.exists() else adapter_id,
            revision=adapter_revision,
        )
    paths = PipelinePaths(
        input_path=input_path.resolve(),
        data_dir=context.run_dir / "data",
        formatted_dir=context.run_dir / "formatted",
        tokenization_dir=context.run_dir / "analysis" / "tokenization",
        train_dir=context.run_dir / "train" / "adapter",
        eval_base_dir=context.run_dir / "eval" / "base",
        eval_adapter_dir=context.run_dir / "eval" / "adapter",
        report_dir=context.run_dir / "report",
        external_adapter=external_adapter,
        full_paired_input_validation_receipt=full_paired_input_validation_receipt,
    )

    command = [
        "sommelier",
        "pipeline",
        "run",
        "--config",
        str(config_path),
        "--mode",
        mode,
        "--input",
        str(input_path),
        "--run-id",
        resolved_run_id,
    ]
    if adapter_id is not None:
        command.extend(["--adapter-id", adapter_id])
        if adapter_revision is not None:
            command.extend(["--adapter-revision", adapter_revision])
    initialize_runtime_metadata(
        context.run_dir,
        gpu=config.remote.gpu,
        run_id=context.run_id,
        config_sha256=context.config_sha256,
        packages=package_versions,
        source_code=source_code,
        remote_execution=remote_execution,
    )
    active_stages = stages if stages is not None else PipelineStages()
    try:
        for stage_name, stage_fn in active_stages.ordered():
            started = time.monotonic()
            try:
                stage_fn(paths, config, context, command)
                if stage_name == "train":
                    record_peak_gpu_memory(
                        context.run_dir,
                        peak_memory_from_training_metrics(
                            paths.train_dir.parent / METRICS_FILENAME
                        ),
                    )
            finally:
                record_stage_runtime(
                    context.run_dir,
                    stage=stage_name,
                    elapsed_seconds=time.monotonic() - started,
                    gpu=config.remote.gpu,
                )
    except Exception:
        set_run_manifest_status(run_dir=context.run_dir, status="failed")
        raise
    set_run_manifest_status(run_dir=context.run_dir, status="succeeded")

    return resolved_run_id
