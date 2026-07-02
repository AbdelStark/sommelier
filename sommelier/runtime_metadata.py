from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, TypedDict, cast

from sommelier.artifacts import write_artifact_atomic

RUNTIME_METADATA_SCHEMA: Final = "sommelier.runtime_metadata.v1"
RUNTIME_METADATA_FILENAME: Final = "runtime_metadata.json"

COST_UNAVAILABLE: Final = "unavailable"


class RuntimeMetadata(TypedDict):
    """Observed runtime evidence for one run.

    Cost is observed evidence, never a guarantee: when the provider does
    not expose billing, ``observed_cost_usd`` is None and ``cost_source``
    says ``unavailable`` explicitly instead of implying zero cost.
    """

    schema_version: str
    stages: dict[str, dict[str, float]]
    hardware: dict[str, str]
    peak_gpu_memory_mb: int | None
    observed_cost_usd: float | None
    cost_source: str


def _metadata_path(run_dir: Path) -> Path:
    return run_dir / RUNTIME_METADATA_FILENAME


def load_runtime_metadata(run_dir: Path) -> RuntimeMetadata | None:
    path = _metadata_path(run_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RUNTIME_METADATA_SCHEMA:
        return None
    return cast(RuntimeMetadata, payload)


def _write(run_dir: Path, metadata: RuntimeMetadata) -> None:
    def writer(temp_path: Path) -> None:
        temp_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    write_artifact_atomic(_metadata_path(run_dir), writer)


def initialize_runtime_metadata(run_dir: Path, *, gpu: str) -> RuntimeMetadata:
    """Starts the run's metadata with hardware and explicit cost state."""
    metadata = RuntimeMetadata(
        schema_version=RUNTIME_METADATA_SCHEMA,
        stages={},
        hardware={"gpu": gpu, "source": "config"},
        peak_gpu_memory_mb=None,
        observed_cost_usd=None,
        cost_source=COST_UNAVAILABLE,
    )
    _write(run_dir, metadata)
    return metadata


def record_stage_runtime(
    run_dir: Path,
    *,
    stage: str,
    elapsed_seconds: float,
    gpu: str,
) -> RuntimeMetadata:
    """Records one stage's wall-clock seconds, creating metadata if needed."""
    metadata = load_runtime_metadata(run_dir)
    if metadata is None:
        metadata = initialize_runtime_metadata(run_dir, gpu=gpu)
    metadata["stages"][stage] = {"elapsed_seconds": round(elapsed_seconds, 3)}
    _write(run_dir, metadata)
    return metadata


def record_peak_gpu_memory(run_dir: Path, peak_gpu_memory_mb: int | None) -> None:
    """Stores the training peak GPU memory measurement when available."""
    metadata = load_runtime_metadata(run_dir)
    if metadata is None or peak_gpu_memory_mb is None:
        return
    metadata["peak_gpu_memory_mb"] = peak_gpu_memory_mb
    _write(run_dir, metadata)


def peak_memory_from_training_metrics(metrics_path: Path) -> int | None:
    """Reads the recorded peak GPU memory from training_metrics.jsonl."""
    if not metrics_path.exists():
        return None
    peak: int | None = None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record: dict[str, Any] = json.loads(stripped)
        value = record.get("peak_gpu_memory_mb")
        if isinstance(value, int):
            peak = value
    return peak


def runtime_section(run_dir: Path) -> dict[str, Any]:
    """The runtime section embedded in comparison reports.

    Marks availability explicitly so a missing measurement can never be
    confused with a zero-cost or zero-duration run.
    """
    metadata = load_runtime_metadata(run_dir)
    if metadata is None:
        return {"available": False}
    return {"available": True, **metadata}
