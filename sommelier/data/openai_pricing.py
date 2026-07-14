"""Pinned OpenAI public-list-price arithmetic for local admission guards.

All currency inputs and persisted values are decimal strings.  The helpers in
this module deliberately do not model invoices, account credits, taxes, or
project-level usage.  They provide deterministic arithmetic against one dated
GPT-5.5 public price snapshot so the provider adapter can stop before starting
another request batch when its local evidence approaches an operator-supplied
ceiling.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from decimal import ROUND_CEILING, Decimal
from typing import Final, Literal, cast

from sommelier.errors import UserInputError

OpenAIServiceTier = Literal["default", "flex"]

OPENAI_PRICING_CHECKED_DATE: Final = "2026-07-13"
OPENAI_PRICING_SOURCE: Final = "https://developers.openai.com/api/docs/models/gpt-5.5"
OPENAI_STANDARD_INPUT_USD_PER_MILLION: Final = Decimal("5")
OPENAI_STANDARD_CACHED_INPUT_USD_PER_MILLION: Final = Decimal("0.5")
OPENAI_STANDARD_OUTPUT_USD_PER_MILLION: Final = Decimal("30")
OPENAI_FLEX_PRICE_MULTIPLIER: Final = Decimal("0.5")

# The request-body byte count is a tokenizer-independent local proxy rather
# than a provider protocol guarantee.  Reserve a deliberately large fixed
# allowance for message framing, structured-output metadata, and special-token
# accounting that is not exposed by the API request body.
OPENAI_LIST_PRICE_INPUT_OVERHEAD_TOKENS_PER_REQUEST: Final = 4096
OPENAI_LONG_CONTEXT_THRESHOLD_INPUT_TOKENS: Final = 272_000
OPENAI_LONG_CONTEXT_INPUT_PRICE_MULTIPLIER: Final = Decimal("2")
OPENAI_LONG_CONTEXT_OUTPUT_PRICE_MULTIPLIER: Final = Decimal("1.5")

OPENAI_LIST_PRICE_CEILING_METHOD: Final = (
    "journal_usage_plus_missing_request_utf8_bytes_and_4096_input_token_reserve_"
    "plus_max_output_tokens_at_pinned_public_list_prices"
)
OPENAI_LIST_PRICE_CEILING_BOUNDARY: Final = (
    "Local pre-request admission and post-response stop guard under the pinned GPT-5.5 "
    "public list-price table. Each missing request treats canonical request-body UTF-8 "
    "bytes plus 4096 tokens as uncached input and reserves its full max output. Provider "
    "protocol and special-token accounting are not publicly bounded by request-body bytes, "
    "so this is not an invoice, observed charge, account/project spend hard cap, or substitute "
    "for provider-side spend controls."
)

_MILLION: Final = Decimal(1_000_000)
_USD_QUANTUM: Final = Decimal("0.000000001")
_POSITIVE_DECIMAL_STRING = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_RUNTIME_SUMMARY_KEYS: Final = frozenset(
    {
        "limit_usd",
        "currency",
        "method",
        "input_token_overhead_per_request",
        "pricing_checked_date",
        "pricing_source",
        "service_tier",
        "long_context_threshold_input_tokens",
        "long_context_policy",
        "boundary",
    }
)


def validated_openai_list_price_limit_usd(value: object) -> Decimal:
    """Parse one explicit, finite, positive plain-decimal USD string."""
    if not isinstance(value, str) or _POSITIVE_DECIMAL_STRING.fullmatch(value) is None:
        raise UserInputError(
            "OpenAI list-price limit must be an explicit positive decimal string",
            hint="Pass --openai-list-price-limit-usd with a value such as 50.00.",
        )
    parsed = Decimal(value)
    if not parsed.is_finite() or parsed <= 0:
        raise UserInputError(
            "OpenAI list-price limit must be greater than zero",
            hint="Pass --openai-list-price-limit-usd with a value such as 50.00.",
        )
    return parsed


def _validated_nonnegative_decimal_string(value: object, *, field: str) -> Decimal:
    if not isinstance(value, str) or _POSITIVE_DECIMAL_STRING.fullmatch(value) is None:
        raise UserInputError(f"OpenAI {field} must be a non-negative decimal string")
    parsed = Decimal(value)
    if not parsed.is_finite() or parsed < 0:
        raise UserInputError(f"OpenAI {field} must be a non-negative decimal string")
    return parsed


def validated_openai_service_tier(value: object) -> OpenAIServiceTier:
    if value not in {"default", "flex"}:
        raise UserInputError("OpenAI list-price calculation requires default or flex service")
    return cast(OpenAIServiceTier, value)


def openai_service_tier_price_multiplier(service_tier: OpenAIServiceTier) -> Decimal:
    return OPENAI_FLEX_PRICE_MULTIPLIER if service_tier == "flex" else Decimal("1")


def openai_list_price_usd(
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    service_tier: OpenAIServiceTier,
    long_context: bool = False,
) -> Decimal:
    """Calculate exact USD from token counts and the pinned public rate table."""
    counts = (input_tokens, cached_input_tokens, output_tokens)
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
        raise UserInputError("OpenAI list-price token counts must be non-negative integers")
    if cached_input_tokens > input_tokens:
        raise UserInputError("OpenAI cached input tokens cannot exceed input tokens")
    service_tier = validated_openai_service_tier(service_tier)
    if not isinstance(long_context, bool):
        raise UserInputError("OpenAI long-context price selection must be boolean")

    input_multiplier = OPENAI_LONG_CONTEXT_INPUT_PRICE_MULTIPLIER if long_context else Decimal("1")
    output_multiplier = (
        OPENAI_LONG_CONTEXT_OUTPUT_PRICE_MULTIPLIER if long_context else Decimal("1")
    )
    uncached_input_tokens = input_tokens - cached_input_tokens
    return (
        (
            Decimal(uncached_input_tokens)
            * OPENAI_STANDARD_INPUT_USD_PER_MILLION
            * input_multiplier
            + Decimal(cached_input_tokens)
            * OPENAI_STANDARD_CACHED_INPUT_USD_PER_MILLION
            * input_multiplier
            + Decimal(output_tokens) * OPENAI_STANDARD_OUTPUT_USD_PER_MILLION * output_multiplier
        )
        * openai_service_tier_price_multiplier(service_tier)
        / _MILLION
    )


def observed_openai_list_price_usd(
    usage: Mapping[str, object],
    *,
    service_tier: OpenAIServiceTier,
) -> Decimal:
    """Calculate the journal estimate from complete aggregate usage fields."""
    fields = ("input_tokens", "cached_input_tokens", "output_tokens")
    counts: dict[str, int] = {}
    for field in fields:
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise UserInputError(f"OpenAI provider journal has invalid usage.{field}")
        counts[field] = value
    return openai_list_price_usd(
        input_tokens=counts["input_tokens"],
        cached_input_tokens=counts["cached_input_tokens"],
        output_tokens=counts["output_tokens"],
        service_tier=service_tier,
    )


def missing_request_list_price_upper_bound_usd(
    *,
    canonical_request_body_utf8_bytes: int,
    max_output_tokens: int,
    service_tier: OpenAIServiceTier,
) -> Decimal:
    """Return the local conservative bound reserved for one missing request."""
    if (
        isinstance(canonical_request_body_utf8_bytes, bool)
        or not isinstance(canonical_request_body_utf8_bytes, int)
        or canonical_request_body_utf8_bytes <= 0
    ):
        raise UserInputError("OpenAI canonical request-body byte count must be positive")
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens <= 0
    ):
        raise UserInputError("OpenAI max output tokens must be positive")
    input_token_bound = openai_request_input_token_bound(canonical_request_body_utf8_bytes)
    return openai_list_price_usd(
        input_tokens=input_token_bound,
        cached_input_tokens=0,
        output_tokens=max_output_tokens,
        service_tier=service_tier,
        long_context=input_token_bound > OPENAI_LONG_CONTEXT_THRESHOLD_INPUT_TOKENS,
    )


def openai_request_input_token_bound(canonical_request_body_utf8_bytes: int) -> int:
    """Return the body-byte proxy plus the fixed provider-framing reserve."""
    if (
        isinstance(canonical_request_body_utf8_bytes, bool)
        or not isinstance(canonical_request_body_utf8_bytes, int)
        or canonical_request_body_utf8_bytes <= 0
    ):
        raise UserInputError("OpenAI canonical request-body byte count must be positive")
    return canonical_request_body_utf8_bytes + OPENAI_LIST_PRICE_INPUT_OVERHEAD_TOKENS_PER_REQUEST


def validate_openai_base_rate_request_body_bytes(
    canonical_request_body_utf8_bytes: int,
) -> None:
    """Fail when the local request bound would cross GPT-5.5 long-context rates."""
    input_token_bound = openai_request_input_token_bound(canonical_request_body_utf8_bytes)
    if input_token_bound > OPENAI_LONG_CONTEXT_THRESHOLD_INPUT_TOKENS:
        raise UserInputError(
            "OpenAI request input bound exceeds the pinned base-rate threshold",
            hint=(
                "Reduce the request before provider access; this translator's published "
                "list-price evidence is intentionally limited to GPT-5.5 base-rate requests."
            ),
        )


def validate_openai_base_rate_response_input_tokens(input_tokens: int) -> None:
    """Fail when observed provider usage would require long-context rates."""
    if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0:
        raise UserInputError("OpenAI response input-token count must be non-negative")
    if input_tokens > OPENAI_LONG_CONTEXT_THRESHOLD_INPUT_TOKENS:
        raise UserInputError(
            "OpenAI response crossed the pinned base-rate input-token threshold",
            hint=(
                "The response may remain journaled, but a base-rate-only run must stop "
                "instead of publishing an understated list-price estimate."
            ),
        )


def missing_batch_list_price_upper_bound_usd(
    canonical_request_body_utf8_bytes: Iterable[int],
    *,
    max_output_tokens: int,
    service_tier: OpenAIServiceTier,
) -> Decimal:
    """Sum the full uncached-input/full-output reserve for a missing batch."""
    return sum(
        (
            missing_request_list_price_upper_bound_usd(
                canonical_request_body_utf8_bytes=body_bytes,
                max_output_tokens=max_output_tokens,
                service_tier=service_tier,
            )
            for body_bytes in canonical_request_body_utf8_bytes
        ),
        start=Decimal("0"),
    )


def decimal_usd_string(value: Decimal) -> str:
    """Serialize a non-negative USD Decimal without binary floating point."""
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise UserInputError("OpenAI list-price USD value must be a non-negative Decimal")
    return format(value.quantize(_USD_QUANTUM, rounding=ROUND_CEILING), "f")


def openai_list_price_ceiling_runtime_summary(
    limit_usd: Decimal,
    *,
    service_tier: OpenAIServiceTier,
) -> dict[str, object]:
    """Return the persisted runtime contract for the local admission guard."""
    if not isinstance(limit_usd, Decimal) or not limit_usd.is_finite() or limit_usd <= 0:
        raise UserInputError("OpenAI list-price limit must be a positive Decimal")
    service_tier = validated_openai_service_tier(service_tier)
    return {
        "limit_usd": format(limit_usd, "f"),
        "currency": "USD",
        "method": OPENAI_LIST_PRICE_CEILING_METHOD,
        "input_token_overhead_per_request": (OPENAI_LIST_PRICE_INPUT_OVERHEAD_TOKENS_PER_REQUEST),
        "pricing_checked_date": OPENAI_PRICING_CHECKED_DATE,
        "pricing_source": OPENAI_PRICING_SOURCE,
        "service_tier": service_tier,
        "long_context_threshold_input_tokens": OPENAI_LONG_CONTEXT_THRESHOLD_INPUT_TOKENS,
        "long_context_policy": "reject_request_bound_above_threshold_before_provider_call",
        "boundary": OPENAI_LIST_PRICE_CEILING_BOUNDARY,
    }


def validate_openai_list_price_ceiling_runtime_summary(
    payload: object,
    *,
    expected_service_tier: OpenAIServiceTier,
    calculated_usd: object,
) -> None:
    """Validate a persisted ceiling and require its estimate not to exceed it."""
    if not isinstance(payload, Mapping) or set(payload) != _RUNTIME_SUMMARY_KEYS:
        raise UserInputError("full provider translation has invalid list-price limit evidence")
    limit_usd = validated_openai_list_price_limit_usd(payload.get("limit_usd"))
    expected = openai_list_price_ceiling_runtime_summary(
        limit_usd,
        service_tier=expected_service_tier,
    )
    if dict(payload) != expected:
        raise UserInputError("full provider translation list-price limit contract has drifted")
    estimate_usd = _validated_nonnegative_decimal_string(
        calculated_usd,
        field="provider list-price estimate",
    )
    if estimate_usd > limit_usd:
        raise UserInputError(
            "full provider translation list-price estimate exceeds its explicit limit"
        )


__all__ = [
    "OPENAI_FLEX_PRICE_MULTIPLIER",
    "OPENAI_LIST_PRICE_CEILING_BOUNDARY",
    "OPENAI_LIST_PRICE_CEILING_METHOD",
    "OPENAI_LIST_PRICE_INPUT_OVERHEAD_TOKENS_PER_REQUEST",
    "OPENAI_LONG_CONTEXT_INPUT_PRICE_MULTIPLIER",
    "OPENAI_LONG_CONTEXT_OUTPUT_PRICE_MULTIPLIER",
    "OPENAI_LONG_CONTEXT_THRESHOLD_INPUT_TOKENS",
    "OPENAI_PRICING_CHECKED_DATE",
    "OPENAI_PRICING_SOURCE",
    "OPENAI_STANDARD_CACHED_INPUT_USD_PER_MILLION",
    "OPENAI_STANDARD_INPUT_USD_PER_MILLION",
    "OPENAI_STANDARD_OUTPUT_USD_PER_MILLION",
    "OpenAIServiceTier",
    "decimal_usd_string",
    "missing_batch_list_price_upper_bound_usd",
    "missing_request_list_price_upper_bound_usd",
    "observed_openai_list_price_usd",
    "openai_list_price_ceiling_runtime_summary",
    "openai_list_price_usd",
    "openai_request_input_token_bound",
    "openai_service_tier_price_multiplier",
    "validate_openai_base_rate_request_body_bytes",
    "validate_openai_base_rate_response_input_tokens",
    "validated_openai_list_price_limit_usd",
    "validated_openai_service_tier",
    "validate_openai_list_price_ceiling_runtime_summary",
]
