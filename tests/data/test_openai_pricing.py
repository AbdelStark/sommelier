from decimal import Decimal

import pytest

from sommelier.data.openai_pricing import (
    OPENAI_LIST_PRICE_CEILING_BOUNDARY,
    OPENAI_LIST_PRICE_CEILING_METHOD,
    OPENAI_LIST_PRICE_INPUT_OVERHEAD_TOKENS_PER_REQUEST,
    decimal_usd_string,
    missing_batch_list_price_upper_bound_usd,
    missing_request_list_price_upper_bound_usd,
    observed_openai_list_price_usd,
    openai_list_price_ceiling_runtime_summary,
    validate_openai_base_rate_request_body_bytes,
    validate_openai_list_price_ceiling_runtime_summary,
    validated_openai_list_price_limit_usd,
)
from sommelier.errors import UserInputError


@pytest.mark.parametrize(
    "value",
    [None, 50, 50.0, "", "0", "0.00", "-1", "+1", ".5", "1e2", " 50.00"],
)
def test_list_price_limit_rejects_missing_non_string_and_non_positive_values(
    value: object,
) -> None:
    with pytest.raises(UserInputError, match="list-price limit"):
        validated_openai_list_price_limit_usd(value)


def test_list_price_limit_preserves_plain_decimal_precision() -> None:
    assert validated_openai_list_price_limit_usd("50.00") == Decimal("50.00")


def test_observed_list_price_uses_uncached_cached_output_and_flex_rates() -> None:
    observed = observed_openai_list_price_usd(
        {
            "input_tokens": 1_000_000,
            "cached_input_tokens": 250_000,
            "output_tokens": 100_000,
        },
        service_tier="flex",
    )

    assert decimal_usd_string(observed) == "3.437500000"


def test_missing_request_bound_reserves_bytes_overhead_and_full_output() -> None:
    bound = missing_request_list_price_upper_bound_usd(
        canonical_request_body_utf8_bytes=2048,
        max_output_tokens=512,
        service_tier="flex",
    )

    expected_input_tokens = 2048 + OPENAI_LIST_PRICE_INPUT_OVERHEAD_TOKENS_PER_REQUEST
    expected = Decimal(expected_input_tokens) * Decimal("2.5") / Decimal(1_000_000)
    expected += Decimal(512) * Decimal("15") / Decimal(1_000_000)
    assert bound == expected


def test_missing_batch_bound_sums_only_missing_unique_requests() -> None:
    per_request = missing_request_list_price_upper_bound_usd(
        canonical_request_body_utf8_bytes=1024,
        max_output_tokens=512,
        service_tier="flex",
    )

    assert (
        missing_batch_list_price_upper_bound_usd(
            [1024, 1024, 1024],
            max_output_tokens=512,
            service_tier="flex",
        )
        == per_request * 3
    )


def test_base_rate_guard_rejects_body_bound_above_long_context_threshold() -> None:
    validate_openai_base_rate_request_body_bytes(267_904)

    with pytest.raises(UserInputError, match="base-rate threshold"):
        validate_openai_base_rate_request_body_bytes(267_905)


def test_runtime_summary_records_decimal_limit_method_and_boundary() -> None:
    summary = openai_list_price_ceiling_runtime_summary(
        Decimal("50.00"),
        service_tier="flex",
    )

    assert summary["limit_usd"] == "50.00"
    assert summary["method"] == OPENAI_LIST_PRICE_CEILING_METHOD
    assert summary["boundary"] == OPENAI_LIST_PRICE_CEILING_BOUNDARY
    assert "invoice" in str(summary["boundary"])
    assert "spend hard cap" in str(summary["boundary"])


def test_runtime_summary_validation_requires_estimate_at_or_below_limit() -> None:
    summary = openai_list_price_ceiling_runtime_summary(
        Decimal("50.00"),
        service_tier="flex",
    )

    validate_openai_list_price_ceiling_runtime_summary(
        summary,
        expected_service_tier="flex",
        calculated_usd="50.000000000",
    )
    with pytest.raises(UserInputError, match="estimate exceeds"):
        validate_openai_list_price_ceiling_runtime_summary(
            summary,
            expected_service_tier="flex",
            calculated_usd="50.000000001",
        )


def test_runtime_summary_validation_rejects_tier_or_method_drift() -> None:
    summary = openai_list_price_ceiling_runtime_summary(
        Decimal("50.00"),
        service_tier="flex",
    )
    summary["method"] = "weaker_method"

    with pytest.raises(UserInputError, match="contract has drifted"):
        validate_openai_list_price_ceiling_runtime_summary(
            summary,
            expected_service_tier="flex",
            calculated_usd="1.000000000",
        )
