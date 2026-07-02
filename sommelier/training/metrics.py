from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final, TypedDict

from sommelier.errors import InvariantViolation

TRAINING_METRIC_SCHEMA: Final = "sommelier.training_metric.v1"

METRICS_FILENAME: Final = "training_metrics.jsonl"


class TrainingMetric(TypedDict):
    schema_version: str
    step: int
    epoch: float
    train_loss: float | None
    eval_loss: float | None
    learning_rate: float
    tokens_seen: int
    peak_gpu_memory_mb: int | None


class TrainingResult(TypedDict):
    """What a training backend hands back to the stage."""

    history: list[dict[str, object]]
    peak_gpu_memory_mb: int | None


def _finite(name: str, value: object) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise InvariantViolation(
            f"training metric {name} is not finite: {value!r}",
            hint="Non-finite losses indicate divergent training; fix the run "
            "instead of persisting NaN or infinity in artifacts.",
        )
    return number


def _optional_finite(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    return _finite(name, value)


def build_training_metrics(
    history: list[dict[str, object]],
    *,
    peak_gpu_memory_mb: int | None,
) -> list[TrainingMetric]:
    """Maps trainer log history to schema-versioned TrainingMetric records.

    Accepts transformers-style log entries (train steps carry ``loss``/
    ``train_loss``, evaluation entries carry ``eval_loss``). Entries without
    a step are ignored. Peak GPU memory is a run-level measurement taken
    after training, so it is recorded on the final metric only; earlier
    records carry None. Non-finite values fail closed.
    """
    metrics: list[TrainingMetric] = []
    for entry in history:
        if "step" not in entry:
            continue
        train_loss = entry.get("loss", entry.get("train_loss"))
        metrics.append(
            TrainingMetric(
                schema_version=TRAINING_METRIC_SCHEMA,
                step=int(_finite("step", entry["step"])),
                epoch=_finite("epoch", entry.get("epoch", 0.0)),
                train_loss=_optional_finite("train_loss", train_loss),
                eval_loss=_optional_finite("eval_loss", entry.get("eval_loss")),
                learning_rate=_finite("learning_rate", entry.get("learning_rate", 0.0)),
                tokens_seen=int(_finite("tokens_seen", entry.get("num_input_tokens_seen", 0))),
                peak_gpu_memory_mb=None,
            )
        )
    if metrics and peak_gpu_memory_mb is not None:
        metrics[-1]["peak_gpu_memory_mb"] = peak_gpu_memory_mb
    return metrics


def write_training_metrics(path: Path, metrics: list[TrainingMetric]) -> None:
    """Writes training metrics as schema-versioned JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(metric, sort_keys=True) + "\n" for metric in metrics),
        encoding="utf-8",
    )


def measure_peak_gpu_memory_mb() -> int | None:
    """Peak CUDA memory in MiB, or None when no GPU is available.

    torch is optional; without it (or without CUDA) the measurement is
    explicitly unavailable rather than zero.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated() // (1024 * 1024))
