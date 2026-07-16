from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Final, TypedDict

from sommelier.data.types import ToolCall
from sommelier.evaluation.parse import ParseStatus

METRIC_NAMES: Final = (
    "valid_json_rate",
    "function_name_accuracy",
    "argument_exact_match",
    "argument_f1",
    "full_call_exact_match",
)


class MetricValue(TypedDict):
    value: float
    numerator: int
    denominator: int


class ScoredRecord(TypedDict):
    example_id: str
    parse_status: ParseStatus
    parsed_call: ToolCall | None
    gold_call: ToolCall


class MetricComponent(TypedDict):
    """One example's additive numerator/denominator contribution."""

    numerator: int
    denominator: int


def _metric(numerator: int, denominator: int) -> MetricValue:
    value = numerator / denominator if denominator else 0.0
    return MetricValue(value=value, numerator=numerator, denominator=denominator)


def _canonical(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def flatten_arguments(arguments: object, prefix: str = "") -> dict[str, str]:
    """Flattens nested arguments into dotted key paths with canonical values.

    Objects extend the path with ``.key``, lists are compared by index with
    ``[i]``, and every leaf value is serialized as canonical
    scalar JSON. Empty objects and arrays are leaves.
    """
    pairs: dict[str, str] = {}
    if isinstance(arguments, dict) and arguments:
        for key, value in arguments.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            pairs.update(flatten_arguments(value, path))
        return pairs
    if isinstance(arguments, list) and arguments:
        for index, value in enumerate(arguments):
            pairs.update(flatten_arguments(value, f"{prefix}[{index}]"))
        return pairs
    pairs[prefix or "<root>"] = _canonical(arguments)
    return pairs


def _is_ok(record: ScoredRecord) -> bool:
    return record["parse_status"] == "ok" and record["parsed_call"] is not None


def valid_json_rate(records: Sequence[ScoredRecord]) -> MetricValue:
    """Share of examples whose output parsed into a schema-valid tool call.

    Parse failures of any kind count against the metric (INV-DATA-005); the
    denominator is always the full record count.
    """
    return _metric(sum(1 for record in records if _is_ok(record)), len(records))


def function_name_accuracy(records: Sequence[ScoredRecord]) -> MetricValue:
    """Share of examples whose parsed call names the gold function."""
    numerator = sum(
        1
        for record in records
        if _is_ok(record)
        and record["parsed_call"] is not None
        and record["parsed_call"]["name"] == record["gold_call"]["name"]
    )
    return _metric(numerator, len(records))


def argument_exact_match(records: Sequence[ScoredRecord]) -> MetricValue:
    """Share of examples whose arguments equal the gold arguments exactly.

    Equality is canonical-JSON equality of the whole arguments object; the
    function name is not considered here.
    """
    numerator = sum(
        1
        for record in records
        if _is_ok(record)
        and record["parsed_call"] is not None
        and _canonical(record["parsed_call"]["arguments"])
        == _canonical(record["gold_call"]["arguments"])
    )
    return _metric(numerator, len(records))


def argument_f1(records: Sequence[ScoredRecord]) -> MetricValue:
    """Micro-averaged F1 over flattened argument key/value pairs.

    Predicted and gold arguments are flattened to dotted-path/canonical-value
    pairs; a pair matches only when path and value both match. F1 is
    ``2 * matched / (predicted_pairs + gold_pairs)``, so the stored
    numerator is ``2 * matched`` and the denominator is the pooled pair
    count. Records without a parsed call contribute zero predicted pairs
    and still contribute their gold pairs, counting failures against the
    metric.
    """
    matched = 0
    predicted_total = 0
    gold_total = 0
    for record in records:
        gold_pairs = flatten_arguments(record["gold_call"]["arguments"])
        gold_total += len(gold_pairs)
        parsed_call = record["parsed_call"]
        if not _is_ok(record) or parsed_call is None:
            continue
        predicted_pairs = flatten_arguments(parsed_call["arguments"])
        predicted_total += len(predicted_pairs)
        matched += sum(
            1 for path, value in predicted_pairs.items() if gold_pairs.get(path) == value
        )
    return _metric(2 * matched, predicted_total + gold_total)


def full_call_exact_match(records: Sequence[ScoredRecord]) -> MetricValue:
    """Share of examples matching the gold call on name and arguments."""
    numerator = sum(
        1
        for record in records
        if _is_ok(record)
        and record["parsed_call"] is not None
        and record["parsed_call"]["name"] == record["gold_call"]["name"]
        and _canonical(record["parsed_call"]["arguments"])
        == _canonical(record["gold_call"]["arguments"])
    )
    return _metric(numerator, len(records))


def metric_components(record: ScoredRecord) -> dict[str, MetricComponent]:
    """Returns additive per-example components for paired resampling.

    Summing each component over records yields the same numerator and
    denominator as :func:`compute_metrics`. Keeping this primitive here makes
    bootstrap confidence intervals reuse the exact metric semantics rather
    than maintaining a second scoring implementation.
    """
    parsed_call = record["parsed_call"]
    ok = _is_ok(record) and parsed_call is not None
    name_match = False
    arguments_match = False
    if ok and parsed_call is not None:
        name_match = parsed_call["name"] == record["gold_call"]["name"]
        arguments_match = _canonical(parsed_call["arguments"]) == _canonical(
            record["gold_call"]["arguments"]
        )

    gold_pairs = flatten_arguments(record["gold_call"]["arguments"])
    predicted_pairs: dict[str, str] = {}
    if ok and parsed_call is not None:
        predicted_pairs = flatten_arguments(parsed_call["arguments"])
    matched = sum(1 for path, value in predicted_pairs.items() if gold_pairs.get(path) == value)
    return {
        "valid_json_rate": MetricComponent(numerator=int(ok), denominator=1),
        "function_name_accuracy": MetricComponent(numerator=int(name_match), denominator=1),
        "argument_exact_match": MetricComponent(numerator=int(arguments_match), denominator=1),
        "argument_f1": MetricComponent(
            numerator=2 * matched,
            denominator=len(predicted_pairs) + len(gold_pairs),
        ),
        "full_call_exact_match": MetricComponent(
            numerator=int(name_match and arguments_match), denominator=1
        ),
    }


def compute_metrics(records: Sequence[ScoredRecord]) -> dict[str, MetricValue]:
    """Computes the five spec metrics keyed by their spec names."""
    return {
        "valid_json_rate": valid_json_rate(records),
        "function_name_accuracy": function_name_accuracy(records),
        "argument_exact_match": argument_exact_match(records),
        "argument_f1": argument_f1(records),
        "full_call_exact_match": full_call_exact_match(records),
    }
