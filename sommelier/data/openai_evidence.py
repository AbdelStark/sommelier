"""Auditable OpenAI Responses usage and public-list-price evidence.

This module deliberately keeps three claims separate:

* the provider journal records API response metadata and usage;
* the evidence payload is a content-free aggregate of that journal; and
* the USD value is a deterministic estimate from a dated public price table,
  not an invoice or an observed charge.

The rates are specific to GPT-5.5.  Callers therefore must supply an exact
dated GPT-5.5 model snapshot and an explicit ``default`` or ``flex`` service
tier.  Reasoning tokens are reported for auditability but are not billed a
second time because they are already included in output tokens.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Final, Literal, cast

from sommelier.data.openai_pricing import (
    OPENAI_FLEX_PRICE_MULTIPLIER,
    OPENAI_LIST_PRICE_INPUT_OVERHEAD_TOKENS_PER_REQUEST,
    OPENAI_LONG_CONTEXT_INPUT_PRICE_MULTIPLIER,
    OPENAI_LONG_CONTEXT_OUTPUT_PRICE_MULTIPLIER,
    OPENAI_LONG_CONTEXT_THRESHOLD_INPUT_TOKENS,
    OPENAI_PRICING_CHECKED_DATE,
    OPENAI_PRICING_SOURCE,
    OPENAI_STANDARD_CACHED_INPUT_USD_PER_MILLION,
    OPENAI_STANDARD_INPUT_USD_PER_MILLION,
    OPENAI_STANDARD_OUTPUT_USD_PER_MILLION,
    decimal_usd_string,
    observed_openai_list_price_usd,
    openai_service_tier_price_multiplier,
    validate_openai_base_rate_request_body_bytes,
    validate_openai_base_rate_response_input_tokens,
)
from sommelier.data.openai_translate import (
    OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
    OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA,
    OPENAI_RESPONSES_SAFETY_IDENTIFIER,
    OPENAI_RESPONSES_SDK_MAX_RETRIES,
    OPENAI_RESPONSES_TIMEOUT_SECONDS,
    aggregate_openai_responses_provider_journal,
    openai_flex_resource_unavailable_retry_policy,
)
from sommelier.errors import UserInputError

OpenAIServiceTier = Literal["default", "flex"]

OPENAI_PROVIDER_EVIDENCE_SCHEMA: Final = "sommelier.openai_provider_evidence.v2"
OPENAI_PROVIDER_JOURNAL_FILENAME: Final = "openai_responses_provider.jsonl"

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_GPT55_SNAPSHOT_PATTERN: Final = re.compile(r"^gpt-5\.5-(\d{4}-\d{2}-\d{2})$")

_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "schema_version",
        "journal",
        "identity",
        "unique_requests",
        "unique_source_attempts",
        "max_canonical_request_body_utf8_bytes",
        "max_response_input_tokens",
        "usage_complete",
        "transport",
        "counts",
        "usage",
        "list_price_estimate",
    }
)
_TRANSPORT_KEYS: Final = frozenset(
    {
        "client_injected",
        "sdk_max_retries",
        "resource_unavailable_retry_policy",
    }
)
_JOURNAL_KEYS: Final = frozenset(
    {
        "filename",
        "schema_version",
        "summary_schema_version",
        "sha256",
        "publication_boundary",
    }
)
_IDENTITY_KEYS: Final = frozenset(
    {
        "requested_model",
        "returned_models",
        "requested_service_tier",
        "returned_service_tiers",
        "request_timeout_seconds",
        "safety_identifier",
    }
)
_COUNT_KEYS: Final = frozenset(
    {
        "records",
        "responses",
        "replayable_responses",
        "replays",
        "durable_journal_replays",
        "batch_coalesced_replays",
        "request_errors",
        "resource_unavailable_events",
        "resolved_resource_unavailable_events",
        "pending_resource_unavailable_events",
        "unresolved_resource_unavailable_events",
        "provider_error_responses",
        "error_records",
        "model_mismatch_responses",
        "service_tier_mismatch_responses",
        "refusal_responses",
        "incomplete_responses",
        "responses_missing_usage",
    }
)
_USAGE_KEYS: Final = frozenset(
    {
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "total_tokens",
    }
)
_PRICING_KEYS: Final = frozenset(
    {
        "checked_date",
        "source",
        "standard_usd_per_million_tokens",
        "service_tier_multiplier",
        "input_token_overhead_per_request",
        "long_context_threshold_input_tokens",
        "long_context_price_multipliers",
        "request_bound_policy",
        "response_usage_policy",
        "all_requests_base_rate_eligible",
        "all_responses_base_rate_eligible",
    }
)
_RATE_KEYS: Final = frozenset({"input", "cached_input", "output_including_reasoning"})
_LONG_CONTEXT_MULTIPLIER_KEYS: Final = frozenset({"input", "output"})
_AVAILABLE_ESTIMATE_KEYS: Final = frozenset(
    {
        "available",
        "calculated_usd",
        "currency",
        "billing_evidence",
        "method",
        "pricing",
        "boundary",
    }
)
_CLEAN_COUNT_KEYS: Final = (
    "error_records",
    "request_errors",
    "provider_error_responses",
    "model_mismatch_responses",
    "service_tier_mismatch_responses",
    "responses_missing_usage",
    "pending_resource_unavailable_events",
    "unresolved_resource_unavailable_events",
)

_PUBLICATION_BOUNDARY: Final = (
    "The raw journal remains in the durable producer artifacts; this content-free "
    "aggregate is published in the translation summary."
)
_ESTIMATE_METHOD: Final = "provider_usage_x_pinned_public_list_prices"
_ESTIMATE_BOUNDARY: Final = (
    "Calculated from API usage fields and the pinned public list-price snapshot; "
    "reasoning tokens are already included in output tokens. Every journaled request's "
    "canonical UTF-8 body byte count plus the fixed input-token reserve and every returned "
    "response usage.input_tokens value are independently validated at or below the GPT-5.5 "
    "long-context threshold, so base rates apply. This is not an invoice or observed "
    "billing artifact."
)
_UNAVAILABLE_REASON: Final = "provider_usage_or_identity_incomplete"


def _mapping(
    value: object,
    *,
    field: str,
    exact_keys: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != exact_keys:
        raise UserInputError(f"OpenAI provider evidence has invalid {field}")
    return cast(Mapping[str, object], value)


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UserInputError(f"OpenAI provider evidence has invalid {field}")
    return value


def _validated_model_snapshot(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise UserInputError(f"OpenAI provider evidence has invalid {field}")
    match = _GPT55_SNAPSHOT_PATTERN.fullmatch(value)
    if match is None:
        raise UserInputError(
            f"OpenAI provider evidence {field} is not an exact dated GPT-5.5 snapshot"
        )
    try:
        date.fromisoformat(match.group(1))
    except ValueError as error:
        raise UserInputError(
            f"OpenAI provider evidence {field} has an invalid snapshot date"
        ) from error
    return value


def _validated_service_tier(value: object, *, field: str) -> OpenAIServiceTier:
    if value not in {"default", "flex"}:
        raise UserInputError(f"OpenAI provider evidence has invalid {field}")
    return cast(OpenAIServiceTier, value)


def _validated_counts(value: object) -> dict[str, int]:
    mapping = _mapping(value, field="counts", exact_keys=_COUNT_KEYS)
    counts = {key: _nonnegative_int(mapping[key], field=f"counts.{key}") for key in _COUNT_KEYS}
    if counts["records"] != (
        counts["responses"]
        + counts["replays"]
        + counts["request_errors"]
        + counts["resource_unavailable_events"]
    ):
        raise UserInputError("OpenAI provider evidence record counts are inconsistent")
    if counts["replays"] != (counts["durable_journal_replays"] + counts["batch_coalesced_replays"]):
        raise UserInputError("OpenAI provider evidence replay counts are inconsistent")
    response_subcounts = (
        "replayable_responses",
        "provider_error_responses",
        "model_mismatch_responses",
        "service_tier_mismatch_responses",
        "refusal_responses",
        "incomplete_responses",
        "responses_missing_usage",
    )
    if any(counts[key] > counts["responses"] for key in response_subcounts):
        raise UserInputError("OpenAI provider evidence response counts are inconsistent")
    if counts["resource_unavailable_events"] != (
        counts["resolved_resource_unavailable_events"]
        + counts["pending_resource_unavailable_events"]
        + counts["unresolved_resource_unavailable_events"]
    ):
        raise UserInputError(
            "OpenAI provider evidence resource-unavailable counts are inconsistent"
        )
    if counts["error_records"] != (
        counts["request_errors"]
        + counts["responses"]
        - counts["replayable_responses"]
        + counts["unresolved_resource_unavailable_events"]
    ):
        raise UserInputError("OpenAI provider evidence error counts are inconsistent")
    return counts


def _validated_usage(value: object) -> dict[str, int]:
    mapping = _mapping(value, field="usage", exact_keys=_USAGE_KEYS)
    usage = {key: _nonnegative_int(mapping[key], field=f"usage.{key}") for key in _USAGE_KEYS}
    if usage["cached_input_tokens"] > usage["input_tokens"]:
        raise UserInputError("OpenAI provider cached input exceeds total input tokens")
    if usage["reasoning_output_tokens"] > usage["output_tokens"]:
        raise UserInputError("OpenAI provider reasoning output exceeds total output tokens")
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        raise UserInputError("OpenAI provider total token count is inconsistent")
    return usage


def _calculated_usd(usage: Mapping[str, int], service_tier: OpenAIServiceTier) -> str:
    return decimal_usd_string(
        observed_openai_list_price_usd(
            usage,
            service_tier=service_tier,
        )
    )


def _pricing_payload(
    service_tier: OpenAIServiceTier,
    *,
    max_canonical_request_body_utf8_bytes: int,
    max_response_input_tokens: int,
) -> dict[str, object]:
    if max_canonical_request_body_utf8_bytes > 0:
        validate_openai_base_rate_request_body_bytes(max_canonical_request_body_utf8_bytes)
    validate_openai_base_rate_response_input_tokens(max_response_input_tokens)
    return {
        "checked_date": OPENAI_PRICING_CHECKED_DATE,
        "source": OPENAI_PRICING_SOURCE,
        "standard_usd_per_million_tokens": {
            "input": str(OPENAI_STANDARD_INPUT_USD_PER_MILLION),
            "cached_input": str(OPENAI_STANDARD_CACHED_INPUT_USD_PER_MILLION),
            "output_including_reasoning": str(OPENAI_STANDARD_OUTPUT_USD_PER_MILLION),
        },
        "service_tier_multiplier": str(openai_service_tier_price_multiplier(service_tier)),
        "input_token_overhead_per_request": (OPENAI_LIST_PRICE_INPUT_OVERHEAD_TOKENS_PER_REQUEST),
        "long_context_threshold_input_tokens": OPENAI_LONG_CONTEXT_THRESHOLD_INPUT_TOKENS,
        "long_context_price_multipliers": {
            "input": str(OPENAI_LONG_CONTEXT_INPUT_PRICE_MULTIPLIER),
            "output": str(OPENAI_LONG_CONTEXT_OUTPUT_PRICE_MULTIPLIER),
        },
        "request_bound_policy": (
            "canonical_request_body_utf8_bytes_plus_fixed_overhead_below_threshold"
        ),
        "response_usage_policy": "max_usage_input_tokens_at_or_below_threshold",
        "all_requests_base_rate_eligible": True,
        "all_responses_base_rate_eligible": True,
    }


def _validated_unique_requests(value: object, *, records: int) -> int:
    unique_requests = _nonnegative_int(value, field="unique_requests")
    if unique_requests > records:
        raise UserInputError("OpenAI provider unique-request count exceeds journal records")
    return unique_requests


def _validated_unique_source_attempts(value: object, *, records: int) -> int:
    unique_source_attempts = _nonnegative_int(value, field="unique_source_attempts")
    if unique_source_attempts > records or (records > 0 and unique_source_attempts == 0):
        raise UserInputError(
            "OpenAI provider unique source-attempt count is inconsistent with journal records"
        )
    return unique_source_attempts


def _validated_transport(value: object) -> dict[str, object]:
    transport = _mapping(value, field="transport", exact_keys=_TRANSPORT_KEYS)
    expected = {
        "client_injected": False,
        "sdk_max_retries": OPENAI_RESPONSES_SDK_MAX_RETRIES,
        "resource_unavailable_retry_policy": (openai_flex_resource_unavailable_retry_policy()),
    }
    if dict(transport) != expected:
        raise UserInputError("OpenAI provider evidence transport policy has drifted")
    return expected


def build_openai_provider_evidence(
    journal_path: Path,
    expected_model: str,
    expected_service_tier: OpenAIServiceTier,
) -> dict[str, object]:
    """Build content-free provider evidence from a validated raw journal."""
    expected_model = _validated_model_snapshot(expected_model, field="expected model")
    expected_service_tier = _validated_service_tier(
        expected_service_tier,
        field="expected service tier",
    )
    if journal_path.name != OPENAI_PROVIDER_JOURNAL_FILENAME:
        raise UserInputError("OpenAI provider journal does not use the canonical evidence filename")

    aggregate = _mapping(
        aggregate_openai_responses_provider_journal(journal_path),
        field="journal aggregate",
        exact_keys=frozenset(
            {
                "schema_version",
                "journal_schema_version",
                "journal_sha256",
                "requested_model",
                "returned_models",
                "requested_service_tier",
                "requested_timeout_seconds",
                "returned_service_tiers",
                "safety_identifier",
                "client_injected",
                "sdk_max_retries",
                "resource_unavailable_retry_policy",
                "max_canonical_request_body_utf8_bytes",
                "max_response_input_tokens",
                "unique_requests",
                "unique_source_attempts",
                "usage_complete",
                "counts",
                "usage",
            }
        ),
    )
    if aggregate["schema_version"] != OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA:
        raise UserInputError("OpenAI provider journal aggregate has an unsupported schema")
    if aggregate["journal_schema_version"] != OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA:
        raise UserInputError("OpenAI provider journal has an unsupported schema")
    journal_sha256 = aggregate["journal_sha256"]
    if not isinstance(journal_sha256, str) or _SHA256_PATTERN.fullmatch(journal_sha256) is None:
        raise UserInputError("OpenAI provider journal has an invalid SHA-256")
    if aggregate["requested_model"] != expected_model or aggregate["returned_models"] != [
        expected_model
    ]:
        raise UserInputError(
            "OpenAI provider journal model identity does not match the translation request"
        )
    if aggregate["requested_service_tier"] != expected_service_tier or aggregate[
        "returned_service_tiers"
    ] != [expected_service_tier]:
        raise UserInputError(
            "OpenAI provider journal service tier does not match the translation request"
        )
    if aggregate["requested_timeout_seconds"] != OPENAI_RESPONSES_TIMEOUT_SECONDS:
        raise UserInputError(
            "OpenAI provider journal timeout does not match the pinned request timeout"
        )
    if aggregate["safety_identifier"] != OPENAI_RESPONSES_SAFETY_IDENTIFIER:
        raise UserInputError("OpenAI provider journal has an unexpected safety identifier")
    transport = _validated_transport(
        {
            "client_injected": aggregate["client_injected"],
            "sdk_max_retries": aggregate["sdk_max_retries"],
            "resource_unavailable_retry_policy": aggregate["resource_unavailable_retry_policy"],
        }
    )
    if not isinstance(aggregate["usage_complete"], bool):
        raise UserInputError("OpenAI provider journal has invalid usage completeness")

    counts = _validated_counts(aggregate["counts"])
    usage = _validated_usage(aggregate["usage"])
    unique_requests = _validated_unique_requests(
        aggregate["unique_requests"],
        records=counts["records"],
    )
    unique_source_attempts = _validated_unique_source_attempts(
        aggregate["unique_source_attempts"],
        records=counts["records"],
    )
    if unique_requests > unique_source_attempts:
        raise UserInputError("OpenAI provider unique requests exceed attributed source attempts")
    usage_complete = aggregate["usage_complete"]
    identity_complete = (
        counts["request_errors"] == 0
        and counts["pending_resource_unavailable_events"] == 0
        and counts["unresolved_resource_unavailable_events"] == 0
        and counts["model_mismatch_responses"] == 0
        and counts["service_tier_mismatch_responses"] == 0
        and counts["responses_missing_usage"] == 0
    )
    max_canonical_request_body_utf8_bytes = _nonnegative_int(
        aggregate["max_canonical_request_body_utf8_bytes"],
        field="max canonical request-body UTF-8 bytes",
    )
    if (counts["records"] > 0) != (max_canonical_request_body_utf8_bytes > 0):
        raise UserInputError(
            "OpenAI provider request-body byte bound is inconsistent with journal records"
        )
    if max_canonical_request_body_utf8_bytes > 0:
        validate_openai_base_rate_request_body_bytes(max_canonical_request_body_utf8_bytes)
    max_response_input_tokens = _nonnegative_int(
        aggregate["max_response_input_tokens"],
        field="max response input tokens",
    )
    validate_openai_base_rate_response_input_tokens(max_response_input_tokens)
    if usage_complete and identity_complete:
        estimate: dict[str, object] = {
            "available": True,
            "calculated_usd": _calculated_usd(usage, expected_service_tier),
            "currency": "USD",
            "billing_evidence": False,
            "method": _ESTIMATE_METHOD,
        }
    else:
        estimate = {
            "available": False,
            "reason": _UNAVAILABLE_REASON,
            "billing_evidence": False,
            "method": _ESTIMATE_METHOD,
        }
    estimate["pricing"] = _pricing_payload(
        expected_service_tier,
        max_canonical_request_body_utf8_bytes=max_canonical_request_body_utf8_bytes,
        max_response_input_tokens=max_response_input_tokens,
    )
    estimate["boundary"] = _ESTIMATE_BOUNDARY

    return {
        "schema_version": OPENAI_PROVIDER_EVIDENCE_SCHEMA,
        "journal": {
            "filename": OPENAI_PROVIDER_JOURNAL_FILENAME,
            "schema_version": OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
            "summary_schema_version": OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA,
            "sha256": journal_sha256,
            "publication_boundary": _PUBLICATION_BOUNDARY,
        },
        "identity": {
            "requested_model": expected_model,
            "returned_models": [expected_model],
            "requested_service_tier": expected_service_tier,
            "returned_service_tiers": [expected_service_tier],
            "request_timeout_seconds": OPENAI_RESPONSES_TIMEOUT_SECONDS,
            "safety_identifier": OPENAI_RESPONSES_SAFETY_IDENTIFIER,
        },
        "unique_requests": unique_requests,
        "unique_source_attempts": unique_source_attempts,
        "max_canonical_request_body_utf8_bytes": (max_canonical_request_body_utf8_bytes),
        "max_response_input_tokens": max_response_input_tokens,
        "usage_complete": usage_complete,
        "transport": transport,
        "counts": counts,
        "usage": usage,
        "list_price_estimate": estimate,
    }


def _validate_journal_payload(value: object) -> None:
    journal = _mapping(value, field="journal", exact_keys=_JOURNAL_KEYS)
    if (
        journal["filename"] != OPENAI_PROVIDER_JOURNAL_FILENAME
        or journal["schema_version"] != OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA
        or journal["summary_schema_version"] != OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA
        or journal["publication_boundary"] != _PUBLICATION_BOUNDARY
    ):
        raise UserInputError("OpenAI provider evidence journal identity has drifted")
    sha256 = journal["sha256"]
    if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
        raise UserInputError("OpenAI provider evidence journal SHA-256 has drifted")


def _validate_identity_payload(
    value: object,
    *,
    expected_model: str,
    expected_service_tier: OpenAIServiceTier,
) -> None:
    identity = _mapping(value, field="identity", exact_keys=_IDENTITY_KEYS)
    if (
        identity["requested_model"] != expected_model
        or identity["returned_models"] != [expected_model]
        or identity["requested_service_tier"] != expected_service_tier
        or identity["returned_service_tiers"] != [expected_service_tier]
        or identity["request_timeout_seconds"] != OPENAI_RESPONSES_TIMEOUT_SECONDS
        or identity["safety_identifier"] != OPENAI_RESPONSES_SAFETY_IDENTIFIER
    ):
        raise UserInputError("OpenAI provider evidence returned identity has drifted")


def _validate_pricing(
    value: object,
    *,
    service_tier: OpenAIServiceTier,
    max_canonical_request_body_utf8_bytes: int,
    max_response_input_tokens: int,
) -> None:
    pricing = _mapping(value, field="pricing", exact_keys=_PRICING_KEYS)
    rates = _mapping(
        pricing["standard_usd_per_million_tokens"],
        field="price rates",
        exact_keys=_RATE_KEYS,
    )
    long_context_multipliers = _mapping(
        pricing["long_context_price_multipliers"],
        field="long-context price multipliers",
        exact_keys=_LONG_CONTEXT_MULTIPLIER_KEYS,
    )
    expected = _pricing_payload(
        service_tier,
        max_canonical_request_body_utf8_bytes=max_canonical_request_body_utf8_bytes,
        max_response_input_tokens=max_response_input_tokens,
    )
    expected_rates = cast(dict[str, object], expected["standard_usd_per_million_tokens"])
    expected_long_context = cast(dict[str, object], expected["long_context_price_multipliers"])
    if (
        pricing["checked_date"] != expected["checked_date"]
        or pricing["source"] != expected["source"]
        or dict(rates) != expected_rates
        or dict(long_context_multipliers) != expected_long_context
        or pricing["service_tier_multiplier"] != expected["service_tier_multiplier"]
        or pricing["input_token_overhead_per_request"]
        != expected["input_token_overhead_per_request"]
        or pricing["long_context_threshold_input_tokens"]
        != expected["long_context_threshold_input_tokens"]
        or pricing["request_bound_policy"] != expected["request_bound_policy"]
        or pricing["response_usage_policy"] != expected["response_usage_policy"]
        or pricing["all_requests_base_rate_eligible"] is not True
        or pricing["all_responses_base_rate_eligible"] is not True
    ):
        raise UserInputError("OpenAI provider evidence pricing has drifted")


def _validate_available_estimate(
    value: object,
    *,
    usage: Mapping[str, int],
    service_tier: OpenAIServiceTier,
    max_canonical_request_body_utf8_bytes: int,
    max_response_input_tokens: int,
) -> None:
    estimate = _mapping(
        value,
        field="list-price estimate",
        exact_keys=_AVAILABLE_ESTIMATE_KEYS,
    )
    if (
        estimate["available"] is not True
        or estimate["currency"] != "USD"
        or estimate["billing_evidence"] is not False
        or estimate["method"] != _ESTIMATE_METHOD
        or estimate["boundary"] != _ESTIMATE_BOUNDARY
    ):
        raise UserInputError("OpenAI provider list-price estimate contract has drifted")
    _validate_pricing(
        estimate["pricing"],
        service_tier=service_tier,
        max_canonical_request_body_utf8_bytes=max_canonical_request_body_utf8_bytes,
        max_response_input_tokens=max_response_input_tokens,
    )
    expected_cost = _calculated_usd(usage, service_tier)
    if estimate["calculated_usd"] != expected_cost:
        raise UserInputError("OpenAI provider list-price estimate cost has drifted")


def validate_openai_provider_evidence(
    payload: object,
    expected_model: str,
    expected_service_tier: OpenAIServiceTier,
    require_clean: bool,
) -> None:
    """Validate evidence before it is accepted as publication-grade.

    Validation always requires complete provider usage and exact returned
    model/tier identity. ``require_clean`` additionally rules out every
    provider/request error category used by the production release gate.
    """
    expected_model = _validated_model_snapshot(expected_model, field="expected model")
    expected_service_tier = _validated_service_tier(
        expected_service_tier,
        field="expected service tier",
    )
    if not isinstance(require_clean, bool):
        raise UserInputError("OpenAI provider evidence require_clean must be boolean")

    evidence = _mapping(payload, field="payload", exact_keys=_TOP_LEVEL_KEYS)
    if evidence["schema_version"] != OPENAI_PROVIDER_EVIDENCE_SCHEMA:
        raise UserInputError("OpenAI provider evidence schema has drifted")
    _validate_journal_payload(evidence["journal"])
    _validate_identity_payload(
        evidence["identity"],
        expected_model=expected_model,
        expected_service_tier=expected_service_tier,
    )
    counts = _validated_counts(evidence["counts"])
    _validated_transport(evidence["transport"])
    unique_requests = _validated_unique_requests(
        evidence["unique_requests"],
        records=counts["records"],
    )
    unique_source_attempts = _validated_unique_source_attempts(
        evidence["unique_source_attempts"],
        records=counts["records"],
    )
    if unique_requests > unique_source_attempts:
        raise UserInputError("OpenAI provider unique requests exceed attributed source attempts")
    usage = _validated_usage(evidence["usage"])
    max_canonical_request_body_utf8_bytes = _nonnegative_int(
        evidence["max_canonical_request_body_utf8_bytes"],
        field="max canonical request-body UTF-8 bytes",
    )
    if (counts["records"] > 0) != (max_canonical_request_body_utf8_bytes > 0):
        raise UserInputError(
            "OpenAI provider request-body byte bound is inconsistent with evidence records"
        )
    if max_canonical_request_body_utf8_bytes > 0:
        validate_openai_base_rate_request_body_bytes(max_canonical_request_body_utf8_bytes)
    max_response_input_tokens = _nonnegative_int(
        evidence["max_response_input_tokens"],
        field="max response input tokens",
    )
    validate_openai_base_rate_response_input_tokens(max_response_input_tokens)
    if evidence["usage_complete"] is not True:
        raise UserInputError("OpenAI provider evidence usage is incomplete")
    if counts["pending_resource_unavailable_events"]:
        raise UserInputError("OpenAI provider evidence has pending availability retries")
    _validate_available_estimate(
        evidence["list_price_estimate"],
        usage=usage,
        service_tier=expected_service_tier,
        max_canonical_request_body_utf8_bytes=max_canonical_request_body_utf8_bytes,
        max_response_input_tokens=max_response_input_tokens,
    )
    if require_clean and any(counts[key] != 0 for key in _CLEAN_COUNT_KEYS):
        raise UserInputError("OpenAI provider evidence contains provider or request errors")


__all__ = [
    "OPENAI_FLEX_PRICE_MULTIPLIER",
    "OPENAI_PRICING_CHECKED_DATE",
    "OPENAI_PRICING_SOURCE",
    "OPENAI_PROVIDER_EVIDENCE_SCHEMA",
    "OPENAI_PROVIDER_JOURNAL_FILENAME",
    "OPENAI_STANDARD_CACHED_INPUT_USD_PER_MILLION",
    "OPENAI_STANDARD_INPUT_USD_PER_MILLION",
    "OPENAI_STANDARD_OUTPUT_USD_PER_MILLION",
    "OpenAIServiceTier",
    "build_openai_provider_evidence",
    "validate_openai_provider_evidence",
]
