from __future__ import annotations

import pytest

from sommelier.data.types import ToolCall
from sommelier.errors import EvaluationError
from sommelier.evaluation.metrics import ScoredRecord, compute_metrics, metric_components
from sommelier.evaluation.statistics import (
    exact_mcnemar_full_call,
    paired_bootstrap_intervals,
)


def _record(example_id: str, *, correct: bool) -> ScoredRecord:
    gold = ToolCall(name="lookup", arguments={"city": "Paris"})
    parsed = gold if correct else ToolCall(name="other", arguments={"city": "Rome"})
    return ScoredRecord(
        example_id=example_id,
        parse_status="ok",
        parsed_call=parsed,
        gold_call=gold,
    )


def test_metric_components_sum_to_the_authoritative_metrics() -> None:
    records = [_record("one", correct=True), _record("two", correct=False)]
    metrics = compute_metrics(records)
    components = [metric_components(record) for record in records]
    for name, metric in metrics.items():
        assert sum(item[name]["numerator"] for item in components) == metric["numerator"]
        assert sum(item[name]["denominator"] for item in components) == metric["denominator"]


def test_paired_bootstrap_is_seeded_and_records_its_contract() -> None:
    reference = [_record(str(index), correct=False) for index in range(6)]
    candidate = [_record(str(index), correct=index < 3) for index in range(6)]
    first = paired_bootstrap_intervals(reference, candidate, seed=17, resamples=200)
    second = paired_bootstrap_intervals(reference, candidate, seed=17, resamples=200)
    assert first == second
    assert first["method"] == "sommelier.paired_bootstrap.v1"
    assert first["confidence_level"] == 0.95
    assert first["resamples"] == 200
    interval = first["intervals"]["full_call_exact_match"]
    assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0


def test_paired_bootstrap_rejects_different_example_identity() -> None:
    with pytest.raises(EvaluationError, match="identities differ"):
        paired_bootstrap_intervals(
            [_record("root", correct=False)],
            [_record("translation", correct=True)],
            seed=1,
            resamples=10,
        )


def test_exact_mcnemar_records_discordant_counts_and_known_two_sided_p_value() -> None:
    reference = [_record(str(index), correct=index == 0) for index in range(6)]
    candidate = [_record(str(index), correct=index != 0) for index in range(6)]

    result = exact_mcnemar_full_call(reference, candidate)

    assert result == {
        "method": "sommelier.exact_mcnemar.v1",
        "metric": "full_call_exact_match",
        "alternative": "two-sided",
        "pairs": 6,
        "discordant_pairs": 6,
        "discordant_counts": {
            "reference_correct_candidate_incorrect": 1,
            "reference_incorrect_candidate_correct": 5,
        },
        "p_value": 0.21875,
    }


def test_exact_mcnemar_returns_one_without_discordant_pairs() -> None:
    records = [_record("one", correct=True), _record("two", correct=False)]

    result = exact_mcnemar_full_call(records, records)

    assert result["discordant_pairs"] == 0
    assert result["p_value"] == 1.0


@pytest.mark.parametrize(
    ("reference", "candidate", "message"),
    [
        ([], [], "equally sized non-empty"),
        ([_record("", correct=False)], [_record("", correct=True)], "non-empty example"),
        (
            [_record("same", correct=False), _record("same", correct=True)],
            [_record("same", correct=True), _record("same", correct=False)],
            "unique example",
        ),
        (
            [_record("reference", correct=False)],
            [_record("candidate", correct=True)],
            "identities differ",
        ),
    ],
)
def test_exact_mcnemar_rejects_invalid_pair_identities(
    reference: list[ScoredRecord],
    candidate: list[ScoredRecord],
    message: str,
) -> None:
    with pytest.raises(EvaluationError, match=message):
        exact_mcnemar_full_call(reference, candidate)
