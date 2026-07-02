from __future__ import annotations

import json
from pathlib import Path

import yaml

from sommelier.pipeline import run_pipeline
from sommelier.runtime_metadata import (
    RUNTIME_METADATA_SCHEMA,
    initialize_runtime_metadata,
    load_runtime_metadata,
    peak_memory_from_training_metrics,
    record_peak_gpu_memory,
    record_stage_runtime,
    runtime_section,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_stage_runtimes_accumulate(tmp_path: Path) -> None:
    record_stage_runtime(tmp_path, stage="data", elapsed_seconds=1.5, gpu="A10G")
    record_stage_runtime(tmp_path, stage="train", elapsed_seconds=120.25, gpu="A10G")

    metadata = load_runtime_metadata(tmp_path)
    assert metadata is not None
    assert metadata["schema_version"] == RUNTIME_METADATA_SCHEMA
    assert metadata["stages"]["data"] == {"elapsed_seconds": 1.5}
    assert metadata["stages"]["train"] == {"elapsed_seconds": 120.25}
    assert metadata["hardware"] == {"gpu": "A10G", "source": "config"}


def test_cost_is_explicitly_unavailable_by_default(tmp_path: Path) -> None:
    metadata = initialize_runtime_metadata(tmp_path, gpu="A10G")
    assert metadata["observed_cost_usd"] is None
    assert metadata["cost_source"] == "unavailable"
    assert metadata["peak_gpu_memory_mb"] is None


def test_peak_memory_recorded_when_available(tmp_path: Path) -> None:
    initialize_runtime_metadata(tmp_path, gpu="A10G")
    record_peak_gpu_memory(tmp_path, 4096)
    metadata = load_runtime_metadata(tmp_path)
    assert metadata is not None
    assert metadata["peak_gpu_memory_mb"] == 4096


def test_peak_memory_none_keeps_unavailable(tmp_path: Path) -> None:
    initialize_runtime_metadata(tmp_path, gpu="A10G")
    record_peak_gpu_memory(tmp_path, None)
    metadata = load_runtime_metadata(tmp_path)
    assert metadata is not None
    assert metadata["peak_gpu_memory_mb"] is None


def test_peak_memory_parsed_from_training_metrics(tmp_path: Path) -> None:
    metrics_path = tmp_path / "training_metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"step": 1, "peak_gpu_memory_mb": None})
        + "\n"
        + json.dumps({"step": 2, "peak_gpu_memory_mb": 2048})
        + "\n",
        encoding="utf-8",
    )
    assert peak_memory_from_training_metrics(metrics_path) == 2048
    assert peak_memory_from_training_metrics(tmp_path / "missing.jsonl") is None


def test_runtime_section_marks_unavailable_explicitly(tmp_path: Path) -> None:
    assert runtime_section(tmp_path) == {"available": False}

    initialize_runtime_metadata(tmp_path, gpu="A10G")
    section = runtime_section(tmp_path)
    assert section["available"] is True
    assert section["cost_source"] == "unavailable"


def test_pipeline_records_runtime_for_every_stage(tmp_path: Path) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        Path("tests/fixtures/preparation_rows.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    from sommelier.config import SommelierConfig
    from sommelier.pipeline import PipelinePaths, PipelineStages
    from sommelier.run_context import RunContext

    def noop(
        paths: PipelinePaths,
        config: SommelierConfig,
        context: RunContext,
        command: list[str],
    ) -> None:
        return None

    run_id = run_pipeline(
        config_path,
        mode="smoke",
        input_path=rows,
        run_id="meta-1",
        project_root=tmp_path,
        stages=PipelineStages(
            prepare=noop,
            format=noop,
            eval_base=noop,
            train=noop,
            eval_adapter=noop,
            compare=noop,
        ),
    )

    run_dir = tmp_path / "artifacts" / "runs" / run_id
    metadata = load_runtime_metadata(run_dir)
    assert metadata is not None
    assert set(metadata["stages"]) == {
        "data",
        "format",
        "eval-base",
        "train",
        "eval-adapter",
        "compare",
    }
    for stage in metadata["stages"].values():
        assert stage["elapsed_seconds"] >= 0.0
    assert metadata["hardware"]["gpu"] == raw["remote"]["gpu"]
    assert metadata["cost_source"] == "unavailable"
