from __future__ import annotations

from sommelier.data.types import ToolCall
from sommelier.evaluation.metrics import (
    METRIC_NAMES,
    ScoredRecord,
    argument_exact_match,
    argument_f1,
    compute_metrics,
    flatten_arguments,
    full_call_exact_match,
    function_name_accuracy,
    valid_json_rate,
)
from sommelier.evaluation.parse import ParseStatus

GOLD = ToolCall(name="lookup_weather", arguments={"city": "Paris", "units": "metric"})


def record(
    parsed: ToolCall | None,
    status: ParseStatus = "ok",
    gold: ToolCall | None = None,
    example_id: str = "e1",
) -> ScoredRecord:
    return ScoredRecord(
        example_id=example_id,
        parse_status=status,
        parsed_call=parsed,
        gold_call=gold or GOLD,
    )


def exact() -> ScoredRecord:
    return record(ToolCall(name="lookup_weather", arguments={"city": "Paris", "units": "metric"}))


def wrong_name() -> ScoredRecord:
    return record(ToolCall(name="get_weather", arguments={"city": "Paris", "units": "metric"}))


def missing_key() -> ScoredRecord:
    return record(ToolCall(name="lookup_weather", arguments={"city": "Paris"}))


def extra_key() -> ScoredRecord:
    return record(
        ToolCall(
            name="lookup_weather",
            arguments={"city": "Paris", "units": "metric", "lang": "fr"},
        )
    )


def parse_failure(status: ParseStatus) -> ScoredRecord:
    return record(None, status=status)


def test_metric_names_match_spec() -> None:
    assert METRIC_NAMES == (
        "valid_json_rate",
        "function_name_accuracy",
        "argument_exact_match",
        "argument_f1",
        "full_call_exact_match",
    )
    metrics = compute_metrics([exact()])
    assert tuple(metrics.keys()) == METRIC_NAMES


def test_all_metrics_store_numerator_and_denominator() -> None:
    metrics = compute_metrics([exact(), parse_failure("no_json")])
    for name in METRIC_NAMES:
        value = metrics[name]
        assert isinstance(value["numerator"], int)
        assert isinstance(value["denominator"], int)
        assert 0.0 <= value["value"] <= 1.0


def test_valid_json_rate_counts_parse_failures_as_failures() -> None:
    records = [
        exact(),
        parse_failure("no_json"),
        parse_failure("invalid_json"),
        parse_failure("invalid_shape"),
    ]
    result = valid_json_rate(records)
    assert result == {"value": 0.25, "numerator": 1, "denominator": 4}


def test_function_name_accuracy() -> None:
    result = function_name_accuracy([exact(), wrong_name(), parse_failure("no_json")])
    assert result == {"value": 1 / 3, "numerator": 1, "denominator": 3}


def test_argument_exact_match_ignores_name_but_not_values() -> None:
    result = argument_exact_match([exact(), wrong_name(), missing_key(), extra_key()])
    assert result["numerator"] == 2  # exact + wrong_name share gold arguments
    assert result["denominator"] == 4


def test_full_call_exact_match_requires_name_and_arguments() -> None:
    result = full_call_exact_match([exact(), wrong_name(), missing_key(), extra_key()])
    assert result == {"value": 0.25, "numerator": 1, "denominator": 4}


def test_flatten_nested_arguments_with_dotted_paths_and_indices() -> None:
    pairs = flatten_arguments(
        {"a": {"b": [1, {"c": "d"}], "e": None}, "f": True, "g": {}, "h": []}
    )
    assert pairs == {
        "a.b[0]": "1",
        "a.b[1].c": '"d"',
        "a.e": "null",
        "f": "true",
        "g": "{}",
        "h": "[]",
    }


def test_argument_f1_perfect_match() -> None:
    result = argument_f1([exact()])
    assert result == {"value": 1.0, "numerator": 4, "denominator": 4}


def test_argument_f1_missing_key_penalizes_recall() -> None:
    # predicted 1 pair (city correct), gold 2 pairs: F1 = 2*1/(1+2)
    result = argument_f1([missing_key()])
    assert result == {"value": 2 / 3, "numerator": 2, "denominator": 3}


def test_argument_f1_extra_key_penalizes_precision() -> None:
    # predicted 3 pairs (2 correct), gold 2 pairs: F1 = 2*2/(3+2)
    result = argument_f1([extra_key()])
    assert result == {"value": 0.8, "numerator": 4, "denominator": 5}


def test_argument_f1_nested_value_mismatch() -> None:
    gold = ToolCall(name="f", arguments={"a": {"b": 1, "c": 2}})
    predicted = ToolCall(name="f", arguments={"a": {"b": 1, "c": 3}})
    result = argument_f1([record(predicted, gold=gold)])
    # 1 matched of 2 predicted and 2 gold pairs
    assert result == {"value": 0.5, "numerator": 2, "denominator": 4}


def test_argument_f1_counts_parse_failures_in_denominator() -> None:
    result = argument_f1([parse_failure("invalid_json")])
    assert result == {"value": 0.0, "numerator": 0, "denominator": 2}


def test_argument_f1_list_order_matters() -> None:
    gold = ToolCall(name="f", arguments={"items": ["a", "b"]})
    predicted = ToolCall(name="f", arguments={"items": ["b", "a"]})
    result = argument_f1([record(predicted, gold=gold)])
    assert result["numerator"] == 0


def test_empty_records_yield_finite_zero_values() -> None:
    metrics = compute_metrics([])
    for name in METRIC_NAMES:
        assert metrics[name] == {"value": 0.0, "numerator": 0, "denominator": 0}


def test_type_distinctions_in_canonical_values() -> None:
    gold = ToolCall(name="f", arguments={"n": 1})
    predicted = ToolCall(name="f", arguments={"n": "1"})
    assert argument_exact_match([record(predicted, gold=gold)])["numerator"] == 0
    assert argument_f1([record(predicted, gold=gold)])["numerator"] == 0
