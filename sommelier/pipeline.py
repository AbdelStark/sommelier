from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

from sommelier.config import SommelierConfig, load_config
from sommelier.errors import UserInputError
from sommelier.evaluation.generate import run_generation
from sommelier.evaluation.report import compare_evaluations, write_evaluation_report
from sommelier.formatting.chat import build_formatted_splits
from sommelier.manifests import create_run_id
from sommelier.run_context import RunContext, ensure_run_context
from sommelier.runtime_metadata import (
    initialize_runtime_metadata,
    peak_memory_from_training_metrics,
    record_peak_gpu_memory,
    record_stage_runtime,
)
from sommelier.training.metrics import METRICS_FILENAME
from sommelier.training.qlora import train_adapter

PipelineMode = Literal["smoke", "full"]

SMOKE_MAX_TRAIN: Final = 100
SMOKE_MAX_VALIDATION: Final = 20
SMOKE_MAX_TEST: Final = 20

StageFn = Callable[["PipelinePaths", SommelierConfig, RunContext, list[str]], None]


@dataclass(frozen=True)
class PipelinePaths:
    """Stage directories inside one run, per the required artifact layout."""

    input_path: Path
    data_dir: Path
    formatted_dir: Path
    train_dir: Path
    eval_base_dir: Path
    eval_adapter_dir: Path
    report_dir: Path


def _stage_prepare(
    paths: PipelinePaths,
    config: SommelierConfig,
    context: RunContext,
    command: list[str],
) -> None:
    from sommelier.data.prepare import prepare_dataset_from_file

    prepare_dataset_from_file(
        config,
        input_path=paths.input_path,
        out_dir=paths.data_dir,
        context=context,
        command=command,
        use_gpu=False,
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
    train_adapter(
        config,
        paths.formatted_dir,
        paths.train_dir,
        context=context,
        command=command,
    )


def _stage_eval_adapter(
    paths: PipelinePaths,
    config: SommelierConfig,
    context: RunContext,
    command: list[str],
) -> None:
    run_generation(
        config,
        formatted_dir=paths.formatted_dir,
        out_dir=paths.eval_adapter_dir,
        model_kind="adapter",
        adapter_dir=paths.train_dir,
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
    eval_base: StageFn = field(default=_stage_eval_base)
    train: StageFn = field(default=_stage_train)
    eval_adapter: StageFn = field(default=_stage_eval_adapter)
    compare: StageFn = field(default=_stage_compare)

    def ordered(self) -> list[tuple[str, StageFn]]:
        return [
            ("data", self.prepare),
            ("format", self.format),
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
    run can never overwrite smoke artifacts (RFC-0007)."""
    base = run_id or create_run_id()
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
) -> str:
    """Chains data, format, baseline eval, train, adapter eval, and compare.

    Every stage reads and writes inside one run directory under the
    configured artifact root; smoke mode bounds the split sizes through a
    resolved config override and uses a separate run ID namespace. Stage
    failures propagate as SommelierError subclasses with their documented
    exit codes; nothing is retried.
    """
    if not input_path.exists():
        raise UserInputError(
            f"raw input file not found: {input_path}",
            hint="Pass --input with a sommelier.raw_tool_call_row.v1 JSONL file.",
        )

    config = load_config(config_path)
    if mode == "smoke":
        config = apply_smoke_overrides(config)
    resolved_run_id = pipeline_run_id(mode, run_id)

    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id=resolved_run_id,
        project_root=project_root or Path.cwd(),
    )
    paths = PipelinePaths(
        input_path=input_path.resolve(),
        data_dir=context.run_dir / "data",
        formatted_dir=context.run_dir / "formatted",
        train_dir=context.run_dir / "train" / "adapter",
        eval_base_dir=context.run_dir / "eval" / "base",
        eval_adapter_dir=context.run_dir / "eval" / "adapter",
        report_dir=context.run_dir / "report",
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
    initialize_runtime_metadata(context.run_dir, gpu=config.remote.gpu)
    active_stages = stages if stages is not None else PipelineStages()
    for stage_name, stage_fn in active_stages.ordered():
        started = time.monotonic()
        stage_fn(paths, config, context, command)
        record_stage_runtime(
            context.run_dir,
            stage=stage_name,
            elapsed_seconds=time.monotonic() - started,
            gpu=config.remote.gpu,
        )
        if stage_name == "train":
            record_peak_gpu_memory(
                context.run_dir,
                peak_memory_from_training_metrics(
                    paths.train_dir.parent / METRICS_FILENAME
                ),
            )

    return resolved_run_id
