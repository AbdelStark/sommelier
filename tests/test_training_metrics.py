from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sommelier.errors import InvariantViolation
from sommelier.training.metrics import (
    TRAINING_METRIC_SCHEMA,
    build_training_metrics,
    measure_peak_gpu_memory_mb,
    write_training_metrics,
)

HF_STYLE_HISTORY: list[dict[str, object]] = [
    {
        "step": 1,
        "epoch": 0.5,
        "loss": 2.25,
        "learning_rate": 0.0002,
        "num_input_tokens_seen": 128,
    },
    {"step": 2, "epoch": 1.0, "eval_loss": 1.75, "num_input_tokens_seen": 256},
    {"step": 2, "epoch": 1.0, "train_loss": 2.0, "train_runtime": 3.5},
    {"train_summary_without_step": True},
]


def test_history_maps_to_schema_records() -> None:
    metrics = build_training_metrics(HF_STYLE_HISTORY, peak_gpu_memory_mb=512)

    assert len(metrics) == 3
    first = metrics[0]
    assert first["schema_version"] == TRAINING_METRIC_SCHEMA
    assert first["step"] == 1
    assert first["epoch"] == 0.5
    assert first["train_loss"] == 2.25
    assert first["eval_loss"] is None
    assert first["learning_rate"] == 0.0002
    assert first["tokens_seen"] == 128
    assert first["peak_gpu_memory_mb"] is None

    eval_record = metrics[1]
    assert eval_record["eval_loss"] == 1.75
    assert eval_record["train_loss"] is None

    summary = metrics[2]
    assert summary["train_loss"] == 2.0
    assert summary["peak_gpu_memory_mb"] == 512


def test_peak_memory_none_stays_none() -> None:
    metrics = build_training_metrics(HF_STYLE_HISTORY, peak_gpu_memory_mb=None)
    assert all(metric["peak_gpu_memory_mb"] is None for metric in metrics)


def test_metric_field_names_match_rfc() -> None:
    metrics = build_training_metrics(HF_STYLE_HISTORY, peak_gpu_memory_mb=None)
    assert set(metrics[0].keys()) == {
        "schema_version",
        "step",
        "epoch",
        "train_loss",
        "eval_loss",
        "learning_rate",
        "tokens_seen",
        "peak_gpu_memory_mb",
    }


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_losses_fail_closed(bad: float) -> None:
    history: list[dict[str, object]] = [{"step": 1, "epoch": 1.0, "loss": bad}]
    with pytest.raises(InvariantViolation):
        build_training_metrics(history, peak_gpu_memory_mb=None)


def test_writer_produces_schema_versioned_jsonl(tmp_path: Path) -> None:
    metrics = build_training_metrics(HF_STYLE_HISTORY, peak_gpu_memory_mb=64)
    path = tmp_path / "train" / "training_metrics.jsonl"
    write_training_metrics(path, metrics)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert record["schema_version"] == TRAINING_METRIC_SCHEMA


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None,
    reason="torch installed; unavailable-measurement path not reachable",
)
def test_peak_measurement_unavailable_without_torch() -> None:
    assert measure_peak_gpu_memory_mb() is None
