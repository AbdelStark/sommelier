from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import sommelier.data.openai_evidence as evidence_module
from sommelier.data.openai_evidence import (
    OPENAI_PROVIDER_JOURNAL_FILENAME,
    build_openai_provider_evidence,
    validate_openai_provider_evidence,
)
from sommelier.data.openai_translate import (
    OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
    OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA,
    OPENAI_RESPONSES_SAFETY_IDENTIFIER,
    OPENAI_RESPONSES_SDK_MAX_RETRIES,
    OPENAI_RESPONSES_TIMEOUT_SECONDS,
    openai_flex_resource_unavailable_retry_policy,
)
from sommelier.errors import UserInputError

MODEL_SNAPSHOT = "gpt-5.5-2026-04-23"


def _provider_aggregate(
    *,
    usage_complete: bool = True,
    provider_errors: int = 0,
) -> dict[str, object]:
    request_errors = 0 if usage_complete else 1
    responses_missing_usage = 0 if usage_complete else 1
    responses = 2
    return {
        "schema_version": OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA,
        "journal_schema_version": OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
        "journal_sha256": "a" * 64,
        "requested_model": MODEL_SNAPSHOT,
        "returned_models": [MODEL_SNAPSHOT],
        "requested_service_tier": "flex",
        "requested_timeout_seconds": OPENAI_RESPONSES_TIMEOUT_SECONDS,
        "returned_service_tiers": ["flex"],
        "safety_identifier": OPENAI_RESPONSES_SAFETY_IDENTIFIER,
        "client_injected": False,
        "sdk_max_retries": OPENAI_RESPONSES_SDK_MAX_RETRIES,
        "resource_unavailable_retry_policy": (openai_flex_resource_unavailable_retry_policy()),
        "max_canonical_request_body_utf8_bytes": 1024,
        "max_response_input_tokens": 600,
        "unique_requests": 2,
        "unique_source_attempts": 2,
        "usage_complete": usage_complete,
        "counts": {
            "records": responses + request_errors,
            "responses": responses,
            "replayable_responses": responses - provider_errors,
            "replays": 0,
            "durable_journal_replays": 0,
            "batch_coalesced_replays": 0,
            "request_errors": request_errors,
            "resource_unavailable_events": 0,
            "resolved_resource_unavailable_events": 0,
            "pending_resource_unavailable_events": 0,
            "unresolved_resource_unavailable_events": 0,
            "provider_error_responses": provider_errors,
            "error_records": request_errors + provider_errors,
            "model_mismatch_responses": 0,
            "service_tier_mismatch_responses": 0,
            "refusal_responses": 0,
            "incomplete_responses": 0,
            "responses_missing_usage": responses_missing_usage,
        },
        "usage": {
            "input_tokens": 1000,
            "cached_input_tokens": 200,
            "output_tokens": 100,
            "reasoning_output_tokens": 25,
            "total_tokens": 1100,
        },
    }


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregate: dict[str, object] | None = None,
) -> dict[str, object]:
    monkeypatch.setattr(
        evidence_module,
        "aggregate_openai_responses_provider_journal",
        lambda _path: aggregate if aggregate is not None else _provider_aggregate(),
    )
    return build_openai_provider_evidence(
        tmp_path / OPENAI_PROVIDER_JOURNAL_FILENAME,
        MODEL_SNAPSHOT,
        "flex",
    )


def test_build_prices_cached_input_and_does_not_double_count_reasoning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build(tmp_path, monkeypatch)

    estimate = evidence["list_price_estimate"]
    assert isinstance(estimate, dict)
    # ((800 * $5/M) + (200 * $0.5/M) + (100 * $30/M)) * 0.5
    # Reasoning is a diagnostic subset of the 100 output tokens, not another charge.
    assert estimate["calculated_usd"] == "0.003550000"
    assert estimate["billing_evidence"] is False
    assert "invoice" in str(estimate["boundary"])
    pricing = estimate["pricing"]
    assert isinstance(pricing, dict)
    assert pricing["standard_usd_per_million_tokens"] == {
        "input": "5",
        "cached_input": "0.5",
        "output_including_reasoning": "30",
    }
    assert pricing["service_tier_multiplier"] == "0.5"
    assert pricing["all_responses_base_rate_eligible"] is True
    assert evidence["max_response_input_tokens"] == 600

    validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", True)


def test_build_withholds_estimate_and_validation_rejects_unavailable_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build(
        tmp_path,
        monkeypatch,
        _provider_aggregate(usage_complete=False),
    )

    estimate = evidence["list_price_estimate"]
    assert isinstance(estimate, dict)
    assert estimate["available"] is False
    assert "calculated_usd" not in estimate
    assert estimate["billing_evidence"] is False
    with pytest.raises(UserInputError, match="usage is incomplete"):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)


def test_resolved_availability_events_remain_clean_and_transport_policy_is_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = _provider_aggregate()
    counts = aggregate["counts"]
    assert isinstance(counts, dict)
    counts.update(
        records=4,
        resource_unavailable_events=2,
        resolved_resource_unavailable_events=2,
    )

    evidence = _build(tmp_path, monkeypatch, aggregate)

    assert evidence["transport"] == {
        "client_injected": False,
        "sdk_max_retries": 0,
        "resource_unavailable_retry_policy": (openai_flex_resource_unavailable_retry_policy()),
    }
    built_counts = evidence["counts"]
    assert isinstance(built_counts, dict)
    assert built_counts["resolved_resource_unavailable_events"] == 2
    validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", True)


def test_pending_availability_retry_withholds_estimate_and_fails_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = _provider_aggregate()
    counts = aggregate["counts"]
    assert isinstance(counts, dict)
    counts.update(
        records=3,
        resource_unavailable_events=1,
        pending_resource_unavailable_events=1,
    )

    evidence = _build(tmp_path, monkeypatch, aggregate)

    estimate = evidence["list_price_estimate"]
    assert isinstance(estimate, dict)
    assert estimate["available"] is False
    with pytest.raises(UserInputError, match="pending availability retries"):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)


def test_validation_recomputes_and_rejects_cost_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build(tmp_path, monkeypatch)
    estimate = evidence["list_price_estimate"]
    assert isinstance(estimate, dict)
    estimate["calculated_usd"] = "0.000000000"

    with pytest.raises(UserInputError, match="cost has drifted"):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("requested_model", "gpt-5.5-2026-05-01"),
        ("returned_models", ["gpt-5.5-2026-05-01"]),
        ("requested_service_tier", "default"),
        ("returned_service_tiers", ["default"]),
        ("request_timeout_seconds", 60.0),
    ],
)
def test_validation_rejects_model_and_tier_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    evidence = _build(tmp_path, monkeypatch)
    identity = evidence["identity"]
    assert isinstance(identity, dict)
    identity[field] = replacement

    with pytest.raises(UserInputError, match="returned identity has drifted"):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("filename", "renamed.jsonl"),
        ("schema_version", "sommelier.openai_responses_provider_journal.v1"),
        ("summary_schema_version", "sommelier.openai_responses_provider_journal_summary.v1"),
        ("sha256", "b" * 63),
    ],
)
def test_validation_rejects_journal_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    evidence = _build(tmp_path, monkeypatch)
    journal = evidence["journal"]
    assert isinstance(journal, dict)
    journal[field] = replacement

    with pytest.raises(UserInputError, match="journal"):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)


def test_validation_rejects_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build(tmp_path, monkeypatch)
    counts = evidence["counts"]
    assert isinstance(counts, dict)
    counts["records"] = 3

    with pytest.raises(UserInputError, match="record counts are inconsistent"):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)


def test_validation_rejects_source_attempt_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build(tmp_path, monkeypatch)
    evidence["unique_source_attempts"] = 0

    with pytest.raises(UserInputError, match="source-attempt count"):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)

    evidence = _build(tmp_path, monkeypatch)
    evidence["unique_source_attempts"] = 1
    with pytest.raises(UserInputError, match="unique requests exceed"):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("cached_input_tokens", 1001, "cached input"),
        ("reasoning_output_tokens", 101, "reasoning output"),
        ("total_tokens", 1099, "total token count"),
    ],
)
def test_validation_rejects_token_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: int,
    message: str,
) -> None:
    evidence = _build(tmp_path, monkeypatch)
    usage = evidence["usage"]
    assert isinstance(usage, dict)
    usage[field] = replacement

    with pytest.raises(UserInputError, match=message):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)


def test_validation_rejects_pricing_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build(tmp_path, monkeypatch)
    estimate = evidence["list_price_estimate"]
    assert isinstance(estimate, dict)
    pricing = estimate["pricing"]
    assert isinstance(pricing, dict)
    pricing["checked_date"] = "2026-07-14"

    with pytest.raises(UserInputError, match="pricing has drifted"):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)


def test_validation_rejects_response_long_context_bound_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build(tmp_path, monkeypatch)
    evidence["max_response_input_tokens"] = 272_001

    with pytest.raises(UserInputError, match="base-rate input-token threshold"):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)


def test_clean_validation_rejects_provider_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _build(
        tmp_path,
        monkeypatch,
        _provider_aggregate(provider_errors=1),
    )

    validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", False)
    with pytest.raises(UserInputError, match="provider or request errors"):
        validate_openai_provider_evidence(evidence, MODEL_SNAPSHOT, "flex", True)


def test_builder_rejects_aggregate_model_tier_and_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for path, replacement, message in (
        (("returned_models",), ["gpt-5.5-2026-05-01"], "model identity"),
        (("returned_service_tiers",), ["default"], "service tier"),
        (("requested_timeout_seconds",), 60.0, "timeout"),
        (("client_injected",), True, "transport policy"),
        (("schema_version",), "sommelier.journal_summary.v2", "unsupported schema"),
    ):
        aggregate = deepcopy(_provider_aggregate())
        aggregate[path[0]] = replacement
        monkeypatch.setattr(
            evidence_module,
            "aggregate_openai_responses_provider_journal",
            lambda _journal, aggregate=aggregate: aggregate,
        )
        with pytest.raises(UserInputError, match=message):
            build_openai_provider_evidence(
                tmp_path / OPENAI_PROVIDER_JOURNAL_FILENAME,
                MODEL_SNAPSHOT,
                "flex",
            )
