"""OpenAI Responses adapter for constrained instruction-chat translation.

The provider boundary is deliberately narrower than the shared translation
pipeline: it builds the canonical instruction-chat conversation, requests one
strict assistant JSON object, and returns the exact decoded provider bytes in
``DecodedTranslationCompletion``.  Parsing, protected-span restoration, and
row auditing remain centralized in :mod:`sommelier.data.translate`.

Every observed response is appended and flushed to a provider journal before
it is returned to the row pipeline. Durably journaled responses are replayed
by deterministic request-payload SHA, which avoids rebilling an already
observed response after a local crash. This is not an exactly-once provider
protocol: if OpenAI accepts a request but the process dies before the response
is received and fsynced, a resume cannot distinguish that call from one that
never ran and may issue it again. ``store=False`` intentionally rules out
provider-side recovery. Source-row and attempt attribution is journal-only and
never enters the provider body or its canonical SHA. Identical provider bodies
may therefore still share one response while each consumer receives its own
attributed replay record.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Final, Literal, Protocol, cast

from sommelier.data.openai_pricing import (
    OpenAIServiceTier,
    decimal_usd_string,
    missing_batch_list_price_upper_bound_usd,
    observed_openai_list_price_usd,
    validate_openai_base_rate_request_body_bytes,
    validate_openai_base_rate_response_input_tokens,
    validated_openai_list_price_limit_usd,
)
from sommelier.data.translate import (
    INSTRUCTION_CHAT_ASSISTANT_PAYLOAD_SCHEMA,
    DecodedTranslationCompletion,
    TranslationModel,
    TranslationRequest,
    build_translation_conversation,
)
from sommelier.errors import ExternalDependencyError, UserInputError

OPENAI_RESPONSES_SDK_VERSION: Final = "2.45.0"
OPENAI_RESPONSES_TIMEOUT_SECONDS: Final = 900.0
OPENAI_RESPONSES_SDK_MAX_RETRIES: Final = 0
OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA: Final = "sommelier.openai_responses_provider_journal.v2"
OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA: Final = (
    "sommelier.openai_responses_provider_journal_summary.v2"
)
OPENAI_RESPONSES_TEXT_FORMAT_NAME: Final = "sommelier_instruction_chat_assistant_payload_v1"
OPENAI_RESPONSES_SAFETY_IDENTIFIER: Final = (
    "sommelier_" + hashlib.sha256(b"sommelier:offline-translation-producer:v1").hexdigest()[:32]
)
OPENAI_FLEX_RESOURCE_UNAVAILABLE_HTTP_STATUS: Final = 429
OPENAI_FLEX_RESOURCE_UNAVAILABLE_ERROR_CODE: Final = "resource_unavailable"
OPENAI_FLEX_RESOURCE_UNAVAILABLE_BACKOFF_SECONDS: Final = (
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
)

_EXACT_MODEL_SNAPSHOT = re.compile(r"^gpt-[a-z0-9][a-z0-9._-]*-\d{4}-\d{2}-\d{2}$")
_EXACT_SDK_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_COMPLETION_DISPOSITIONS = {"complete", "incomplete", "not_generated"}
_REPLAY_SOURCES = {"durable_journal", "batch_coalesced"}
_KNOWN_RETURNED_SERVICE_TIERS = {"auto", "default", "flex", "scale", "priority"}


class OpenAIResponsesResource(Protocol):
    """Minimal Responses surface used by the adapter and test doubles."""

    def create(self, **kwargs: object) -> object: ...


class OpenAIResponsesClient(Protocol):
    """Minimal injected-client contract; importing the SDK stays optional."""

    @property
    def responses(self) -> OpenAIResponsesResource: ...


@dataclass(frozen=True)
class _PreparedRequest:
    body: dict[str, object]
    sha256: str
    canonical_body_utf8_bytes: int
    batch_position: int
    source_id: str
    attempt: int


@dataclass(frozen=True)
class _ObservedResponse:
    response_id: str | None
    request_id: str | None
    returned_model: str | None
    status: str | None
    returned_service_tier: str | None
    provider_created_at: int | float | None
    incomplete_reason: str | None
    response_error_code: str | None
    response_error_type: str | None
    raw_output: str
    refusals: tuple[str, ...]
    usage: dict[str, object]


@dataclass(frozen=True)
class _CachedCompletion:
    completion: DecodedTranslationCompletion
    canonical_body_utf8_bytes: int


@dataclass(frozen=True)
class _PendingAvailabilityRetry:
    source_id: str
    attempt: int
    next_provider_call_attempt: int
    backoff_seconds: float


def openai_flex_resource_unavailable_retry_policy() -> dict[str, object]:
    """Return the fixed, journaled Flex availability retry contract."""
    return {
        "eligible_service_tier": "flex",
        "http_status": OPENAI_FLEX_RESOURCE_UNAVAILABLE_HTTP_STATUS,
        "error_code": OPENAI_FLEX_RESOURCE_UNAVAILABLE_ERROR_CODE,
        "max_retries": len(OPENAI_FLEX_RESOURCE_UNAVAILABLE_BACKOFF_SECONDS),
        "backoff_seconds": list(OPENAI_FLEX_RESOURCE_UNAVAILABLE_BACKOFF_SECONDS),
        "switch_service_tier": False,
    }


def _assistant_text_format() -> dict[str, object]:
    """Return a fresh strict schema so an SDK/test double cannot mutate it."""
    return {
        "format": {
            "type": "json_schema",
            "name": OPENAI_RESPONSES_TEXT_FORMAT_NAME,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "schema_version": {
                        "type": "string",
                        "enum": [INSTRUCTION_CHAT_ASSISTANT_PAYLOAD_SCHEMA],
                    },
                    "target_text": {"type": "string"},
                },
                "required": ["schema_version", "target_text"],
                "additionalProperties": False,
            },
        }
    }


def _canonical_request_bytes(body: Mapping[str, object]) -> bytes:
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_request_sha256(body: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_request_bytes(body)).hexdigest()


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return cast(object | None, value.get(name))
    return cast(object | None, getattr(value, name, None))


def _string_field(value: object, name: str) -> str | None:
    candidate = _field(value, name)
    return candidate if isinstance(candidate, str) else None


def _number_field(value: object, name: str) -> int | float | None:
    candidate = _field(value, name)
    if isinstance(candidate, bool) or not isinstance(candidate, int | float):
        return None
    return candidate if math.isfinite(float(candidate)) else None


def _is_flex_resource_unavailable(error: Exception) -> bool:
    """Match only the SDK's structured 429 resource-unavailable response.

    Exception text is intentionally ignored: it is neither a stable API
    contract nor safe journal material.  The pinned SDK exposes the decoded
    API error object as ``body`` and its HTTP status as ``status_code``.
    """
    status_code = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    return (
        status_code == OPENAI_FLEX_RESOURCE_UNAVAILABLE_HTTP_STATUS
        and isinstance(body, Mapping)
        and body.get("code") == OPENAI_FLEX_RESOURCE_UNAVAILABLE_ERROR_CODE
    )


def _items(value: object | None) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    return []


def _collect_output(response: object) -> tuple[str, tuple[str, ...]]:
    text_parts: list[str] = []
    refusals: list[str] = []
    saw_output_text = False
    for output_item in _items(_field(response, "output")):
        for content_item in _items(_field(output_item, "content")):
            content_type = _string_field(content_item, "type")
            if content_type == "output_text":
                text = _string_field(content_item, "text")
                if text is not None:
                    saw_output_text = True
                    text_parts.append(text)
            elif content_type == "refusal":
                refusal = _string_field(content_item, "refusal")
                if refusal is not None:
                    refusals.append(refusal)
    if not saw_output_text:
        output_text = _string_field(response, "output_text")
        if output_text is not None:
            text_parts.append(output_text)
    return "".join(text_parts), tuple(refusals)


def _usage_details(usage: object | None) -> dict[str, object]:
    if usage is None:
        return {}
    result: dict[str, object] = {}
    for field_name in ("input_tokens", "output_tokens", "total_tokens"):
        value = _number_field(usage, field_name)
        if value is not None:
            result[field_name] = value
    for details_name, numeric_fields in (
        ("input_tokens_details", ("cached_tokens",)),
        ("output_tokens_details", ("reasoning_tokens",)),
    ):
        details = _field(usage, details_name)
        sanitized: dict[str, object] = {}
        if details is not None:
            for field_name in numeric_fields:
                value = _number_field(details, field_name)
                if value is not None:
                    sanitized[field_name] = value
        if sanitized:
            result[details_name] = sanitized
    return result


def _response_usage_for_pricing(usage: Mapping[str, object]) -> dict[str, object]:
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    input_details = usage.get("input_tokens_details")
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens < 0
        or isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens != input_tokens + output_tokens
        or not isinstance(input_details, Mapping)
    ):
        raise UserInputError(
            "OpenAI response lacks complete usage for the post-response list-price guard"
        )
    validate_openai_base_rate_response_input_tokens(input_tokens)
    cached_input_tokens = input_details.get("cached_tokens")
    if (
        isinstance(cached_input_tokens, bool)
        or not isinstance(cached_input_tokens, int)
        or cached_input_tokens < 0
        or cached_input_tokens > input_tokens
    ):
        raise UserInputError(
            "OpenAI response lacks complete cached-input usage for the list-price guard"
        )
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
    }


def _observe_response(response: object) -> _ObservedResponse:
    raw_output, refusals = _collect_output(response)
    incomplete_details = _field(response, "incomplete_details")
    response_error = _field(response, "error")
    request_id = _string_field(response, "_request_id") or _string_field(response, "request_id")
    return _ObservedResponse(
        response_id=_string_field(response, "id"),
        request_id=request_id,
        returned_model=_string_field(response, "model"),
        status=_string_field(response, "status"),
        returned_service_tier=_string_field(response, "service_tier"),
        provider_created_at=_number_field(response, "created_at"),
        incomplete_reason=(
            _string_field(incomplete_details, "reason") if incomplete_details is not None else None
        ),
        response_error_code=(
            _string_field(response_error, "code") if response_error is not None else None
        ),
        response_error_type=(
            _string_field(response_error, "type") if response_error is not None else None
        ),
        raw_output=raw_output,
        refusals=refusals,
        usage=_usage_details(_field(response, "usage")),
    )


def _classify_response(
    observed: _ObservedResponse,
) -> DecodedTranslationCompletion | None:
    if observed.response_error_code is not None or observed.response_error_type is not None:
        return None
    if observed.status == "completed":
        if observed.refusals:
            return DecodedTranslationCompletion(
                text=observed.raw_output,
                disposition="incomplete" if observed.raw_output else "not_generated",
                finish_reason="refusal",
            )
        if not observed.raw_output:
            return DecodedTranslationCompletion(
                text="",
                disposition="not_generated",
                finish_reason="missing_output_text",
            )
        return DecodedTranslationCompletion(
            text=observed.raw_output,
            disposition="complete",
            finish_reason="completed",
        )
    if observed.status == "incomplete":
        reason = observed.incomplete_reason or "unknown"
        return DecodedTranslationCompletion(
            text=observed.raw_output,
            disposition="incomplete" if observed.raw_output else "not_generated",
            finish_reason=f"incomplete:{reason}",
        )
    return None


def _default_client(
    *,
    expected_sdk_version: str,
    max_retries: int,
    timeout_seconds: float,
) -> OpenAIResponsesClient:
    try:
        installed_version = package_version("openai")
    except PackageNotFoundError:
        raise ExternalDependencyError(
            "OpenAI Responses translation requires the optional openai SDK",
            hint=f"Install the exact runtime pin openai=={expected_sdk_version}.",
        ) from None
    if installed_version != expected_sdk_version:
        raise ExternalDependencyError(
            "OpenAI Responses SDK does not match the pinned runtime",
            hint=(f"Install openai=={expected_sdk_version}; observed openai=={installed_version}."),
        )
    try:
        module = import_module("openai")
        client_type = getattr(module, "OpenAI", None)
        if not callable(client_type):
            raise AttributeError("OpenAI client constructor is unavailable")
        client = client_type(
            max_retries=max_retries,
            timeout=timeout_seconds,
        )
    except Exception:
        raise ExternalDependencyError(
            "OpenAI Responses client could not be initialized",
            hint="Provide OPENAI_API_KEY through the remote secret boundary.",
        ) from None
    return cast(OpenAIResponsesClient, client)


class _OpenAIResponsesTranslationModel:
    def __init__(
        self,
        *,
        model_snapshot: str,
        max_output_tokens: int,
        provider_journal_path: Path,
        service_tier: OpenAIServiceTier,
        client: OpenAIResponsesClient | None,
        expected_sdk_version: str,
        max_retries: int,
        timeout_seconds: float,
        max_workers: int,
        openai_list_price_limit_usd: str,
    ) -> None:
        self._model_snapshot = model_snapshot
        self._max_output_tokens = max_output_tokens
        self._journal_path = provider_journal_path
        self._service_tier = service_tier
        self._client = client
        self._client_injected = client is not None
        self._expected_sdk_version = expected_sdk_version
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds
        self._max_workers = max_workers
        self._list_price_limit_usd = validated_openai_list_price_limit_usd(
            openai_list_price_limit_usd
        )
        self._client_lock = threading.Lock()
        self._journal_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._price_lock = threading.Lock()
        # A single admission critical section covers journal replay discovery,
        # the complete missing-batch reservation, provider access, and the
        # post-response stop check. Internal request workers remain parallel,
        # but two caller threads can never both admit against the same observed
        # spend before either batch is journaled.
        self._translate_batch_lock = threading.Lock()
        self._running_list_price_usd = Decimal("0")
        # Validate the complete existing ledger before replaying or appending
        # to it. A trailing, explicitly scheduled Flex availability retry is
        # resumable; terminal or malformed provider failures fail closed.
        self._running_list_price_usd = self._observed_list_price_usd()
        if self._running_list_price_usd > self._list_price_limit_usd:
            raise UserInputError(
                "OpenAI existing provider list-price estimate exceeds the explicit limit",
                hint=(
                    f"Observed {decimal_usd_string(self._running_list_price_usd)} USD versus "
                    f"limit {format(self._list_price_limit_usd, 'f')} USD. Preserve and review "
                    "the journal before authorizing another paid batch."
                ),
            )
        self._replay_cache = self._load_replay_cache()
        self._pending_availability_retries = self._load_pending_availability_retries()

    def _client_for_use(self) -> OpenAIResponsesClient:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = _default_client(
                        expected_sdk_version=self._expected_sdk_version,
                        max_retries=self._max_retries,
                        timeout_seconds=self._timeout_seconds,
                    )
        client = self._client
        if client is None:  # pragma: no cover - guarded by the lock above
            raise RuntimeError("OpenAI Responses client was not initialized")
        return client

    def _prepare_request(
        self,
        request: TranslationRequest,
        *,
        batch_position: int,
    ) -> _PreparedRequest:
        if not isinstance(request.source_id, str) or not request.source_id:
            raise UserInputError(
                "OpenAI Responses translation requires a non-empty source_id attribution"
            )
        if (
            isinstance(request.attempt, bool)
            or not isinstance(request.attempt, int)
            or request.attempt <= 0
        ):
            raise UserInputError(
                "OpenAI Responses translation requires a positive attempt attribution"
            )
        conversation, _replacements = build_translation_conversation(
            request,
            "instruction_chat",
        )
        body: dict[str, object] = {
            "model": self._model_snapshot,
            "input": conversation,
            "text": _assistant_text_format(),
            "reasoning": {"effort": "none"},
            "safety_identifier": OPENAI_RESPONSES_SAFETY_IDENTIFIER,
            "store": False,
            "background": False,
            "service_tier": self._service_tier,
            "max_output_tokens": self._max_output_tokens,
            "truncation": "disabled",
        }
        canonical_body = _canonical_request_bytes(body)
        canonical_body_utf8_bytes = len(canonical_body)
        validate_openai_base_rate_request_body_bytes(canonical_body_utf8_bytes)
        return _PreparedRequest(
            body=body,
            sha256=hashlib.sha256(canonical_body).hexdigest(),
            canonical_body_utf8_bytes=canonical_body_utf8_bytes,
            batch_position=batch_position,
            source_id=request.source_id,
            attempt=request.attempt,
        )

    def _base_journal_record(self, prepared: _PreparedRequest) -> dict[str, object]:
        return {
            "schema_version": OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
            "observed_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "request_sha256": prepared.sha256,
            "batch_position": prepared.batch_position,
            "attribution": {
                "source_id": prepared.source_id,
                "attempt": prepared.attempt,
            },
            "request": {
                "api": "responses",
                "requested_model": self._model_snapshot,
                "requested_service_tier": self._service_tier,
                "canonical_request_body_utf8_bytes": prepared.canonical_body_utf8_bytes,
                "max_output_tokens": self._max_output_tokens,
                "reasoning_effort": "none",
                "safety_identifier": OPENAI_RESPONSES_SAFETY_IDENTIFIER,
                "store": False,
                "background": False,
                "truncation": "disabled",
                "sdk_version_expected": self._expected_sdk_version,
                "sdk_retries": self._max_retries,
                "timeout_seconds": self._timeout_seconds,
                "client_injected": self._client_injected,
                "resource_unavailable_retry_policy": (
                    openai_flex_resource_unavailable_retry_policy()
                ),
            },
        }

    def _response_journal_record(
        self,
        prepared: _PreparedRequest,
        observed: _ObservedResponse,
        completion: DecodedTranslationCompletion | None,
        *,
        elapsed_seconds: float,
        provider_call_attempt: int,
    ) -> dict[str, object]:
        record = self._base_journal_record(prepared)
        model_matches = observed.returned_model == self._model_snapshot
        service_tier_matches = observed.returned_service_tier == self._service_tier
        replayable = model_matches and service_tier_matches and completion is not None
        record.update(
            {
                "event": "response",
                "elapsed_seconds": elapsed_seconds,
                "provider_call_attempt": provider_call_attempt,
                "replayable": replayable,
                "response": {
                    "id": observed.response_id,
                    "request_id": observed.request_id,
                    "returned_model": observed.returned_model,
                    "model_matches_request": model_matches,
                    "status": observed.status,
                    "returned_service_tier": observed.returned_service_tier,
                    "service_tier_matches_request": service_tier_matches,
                    "provider_created_at": observed.provider_created_at,
                    "incomplete_reason": observed.incomplete_reason,
                    "error_code": observed.response_error_code,
                    "error_type": observed.response_error_type,
                },
                "completion": (
                    {
                        "disposition": completion.disposition,
                        "finish_reason": completion.finish_reason,
                    }
                    if completion is not None
                    else None
                ),
                "raw_output": observed.raw_output,
                "raw_output_sha256": hashlib.sha256(
                    observed.raw_output.encode("utf-8")
                ).hexdigest(),
                "refusals": list(observed.refusals),
                "usage": observed.usage,
            }
        )
        return record

    def _replay_journal_record(
        self,
        prepared: _PreparedRequest,
        completion: DecodedTranslationCompletion,
        *,
        source: Literal["durable_journal", "batch_coalesced"],
    ) -> dict[str, object]:
        record = self._base_journal_record(prepared)
        record.update(
            {
                "event": "replay",
                "replay": {
                    "source": source,
                    "raw_output_sha256": hashlib.sha256(
                        completion.text.encode("utf-8")
                    ).hexdigest(),
                },
                "completion": {
                    "disposition": completion.disposition,
                    "finish_reason": completion.finish_reason,
                },
                "usage": {},
            }
        )
        return record

    def _request_error_journal_record(
        self,
        prepared: _PreparedRequest,
        error: Exception,
        *,
        elapsed_seconds: float,
        provider_call_attempt: int,
    ) -> dict[str, object]:
        record = self._base_journal_record(prepared)
        record.update(
            {
                "event": "request_error",
                "elapsed_seconds": elapsed_seconds,
                "provider_call_attempt": provider_call_attempt,
                "replayable": False,
                # Exception messages and HTTP response/header objects are
                # intentionally excluded: SDK errors can carry credentials or
                # provider headers. The class is sufficient for local triage.
                "error": {"type": type(error).__name__},
                "raw_output": None,
                "raw_output_sha256": None,
                "refusals": [],
                "usage": {},
            }
        )
        return record

    def _resource_unavailable_journal_record(
        self,
        prepared: _PreparedRequest,
        error: Exception,
        *,
        elapsed_seconds: float,
        provider_call_attempt: int,
        retry_scheduled: bool,
        backoff_seconds: float | None,
    ) -> dict[str, object]:
        record = self._base_journal_record(prepared)
        record.update(
            {
                "event": "resource_unavailable",
                "elapsed_seconds": elapsed_seconds,
                "provider_call_attempt": provider_call_attempt,
                "replayable": False,
                "resource_unavailable": {
                    "error_type": type(error).__name__,
                    "http_status": OPENAI_FLEX_RESOURCE_UNAVAILABLE_HTTP_STATUS,
                    "error_code": OPENAI_FLEX_RESOURCE_UNAVAILABLE_ERROR_CODE,
                    "retry_scheduled": retry_scheduled,
                    "backoff_seconds": backoff_seconds,
                },
                "raw_output": None,
                "raw_output_sha256": None,
                "refusals": [],
                "usage": {},
            }
        )
        return record

    def _append_journal(self, record: Mapping[str, object]) -> None:
        line = (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        try:
            with self._journal_lock:
                self._journal_path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    self._journal_path,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    0o600,
                )
                with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.flush()
                    try:
                        os.fsync(stream.fileno())
                    except OSError as error:
                        unsupported = {errno.EINVAL}
                        if hasattr(errno, "ENOTSUP"):
                            unsupported.add(errno.ENOTSUP)
                        if error.errno not in unsupported:
                            raise
        except OSError as error:
            raise ExternalDependencyError(
                f"could not append OpenAI provider journal: {self._journal_path}",
                hint="Keep the provider journal on a writable durable artifacts volume.",
            ) from error

    def _load_replay_cache(self) -> dict[str, _CachedCompletion]:
        if not self._journal_path.exists():
            return {}
        if not self._journal_path.is_file():
            raise UserInputError(f"OpenAI provider journal is not a file: {self._journal_path}")
        cache: dict[str, _CachedCompletion] = {}
        try:
            lines = self._journal_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ExternalDependencyError(
                f"could not read OpenAI provider journal: {self._journal_path}"
            ) from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} is not valid JSON",
                    hint="Preserve the journal for audit; do not silently rebill unknown calls.",
                ) from error
            if not isinstance(payload, dict):
                raise UserInputError(f"OpenAI provider journal line {line_number} is not an object")
            if payload.get("schema_version") != OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has an unsupported schema"
                )
            _journal_attribution(
                payload.get("attribution"),
                line_number=line_number,
            )
            if payload.get("event") != "response" or payload.get("replayable") is not True:
                continue
            request_sha256 = payload.get("request_sha256")
            raw_output = payload.get("raw_output")
            raw_output_sha256 = payload.get("raw_output_sha256")
            completion = payload.get("completion")
            request_metadata = payload.get("request")
            response_metadata = payload.get("response")
            if (
                not isinstance(request_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None
                or not isinstance(raw_output, str)
                or not isinstance(raw_output_sha256, str)
                or not isinstance(completion, dict)
                or not isinstance(request_metadata, dict)
                or not isinstance(response_metadata, dict)
            ):
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has invalid replay fields"
                )
            if hashlib.sha256(raw_output.encode("utf-8")).hexdigest() != raw_output_sha256:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} failed its raw-output hash"
                )
            canonical_body_utf8_bytes = request_metadata.get("canonical_request_body_utf8_bytes")
            if (
                isinstance(canonical_body_utf8_bytes, bool)
                or not isinstance(canonical_body_utf8_bytes, int)
                or canonical_body_utf8_bytes <= 0
            ):
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has an invalid "
                    "canonical request-body byte count"
                )
            validate_openai_base_rate_request_body_bytes(canonical_body_utf8_bytes)
            if (
                request_metadata.get("requested_model") != self._model_snapshot
                or request_metadata.get("requested_service_tier") != self._service_tier
                or response_metadata.get("returned_model") != self._model_snapshot
                or response_metadata.get("model_matches_request") is not True
                or response_metadata.get("returned_service_tier") != self._service_tier
                or response_metadata.get("service_tier_matches_request") is not True
            ):
                continue
            disposition = completion.get("disposition")
            finish_reason = completion.get("finish_reason")
            if disposition not in _COMPLETION_DISPOSITIONS or not (
                finish_reason is None or isinstance(finish_reason, str)
            ):
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has an invalid completion"
                )
            completion_value = DecodedTranslationCompletion(
                text=raw_output,
                disposition=cast(
                    Literal["complete", "incomplete", "not_generated"],
                    disposition,
                ),
                finish_reason=finish_reason,
            )
            previous = cache.get(request_sha256)
            if (
                previous is not None
                and previous.canonical_body_utf8_bytes != canonical_body_utf8_bytes
            ):
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} changes the canonical "
                    "request-body byte count for one request"
                )
            cache[request_sha256] = _CachedCompletion(
                completion=completion_value,
                canonical_body_utf8_bytes=canonical_body_utf8_bytes,
            )
        return cache

    def _load_pending_availability_retries(
        self,
    ) -> dict[str, _PendingAvailabilityRetry]:
        """Recover a trailing fsynced Flex retry without losing attribution."""
        if not self._journal_path.exists():
            return {}
        try:
            lines = self._journal_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ExternalDependencyError(
                f"could not read OpenAI provider journal: {self._journal_path}"
            ) from error
        pending: dict[str, _PendingAvailabilityRetry] = {}
        for line in lines:
            if not line.strip():
                continue
            payload = cast(dict[str, object], json.loads(line))
            event = payload["event"]
            request_sha256 = cast(str, payload["request_sha256"])
            if event in {"response", "request_error"}:
                pending.pop(request_sha256, None)
                continue
            if event != "resource_unavailable":
                continue
            metadata = cast(dict[str, object], payload["resource_unavailable"])
            if metadata["retry_scheduled"] is not True:
                pending.pop(request_sha256, None)
                continue
            attribution = cast(dict[str, object], payload["attribution"])
            provider_call_attempt = cast(int, payload["provider_call_attempt"])
            backoff_seconds = cast(float, metadata["backoff_seconds"])
            candidate = _PendingAvailabilityRetry(
                source_id=cast(str, attribution["source_id"]),
                attempt=cast(int, attribution["attempt"]),
                next_provider_call_attempt=provider_call_attempt + 1,
                backoff_seconds=backoff_seconds,
            )
            previous = pending.get(request_sha256)
            if previous is not None and (
                previous.source_id != candidate.source_id or previous.attempt != candidate.attempt
            ):
                raise UserInputError(
                    "OpenAI provider journal has ambiguous pending availability retries"
                )
            pending[request_sha256] = candidate
        return pending

    def _request_one(self, prepared: _PreparedRequest) -> DecodedTranslationCompletion:
        pending = self._pending_availability_retries.get(prepared.sha256)
        provider_call_attempt = 1
        if pending is not None:
            if (pending.source_id, pending.attempt) != (
                prepared.source_id,
                prepared.attempt,
            ):
                raise UserInputError(
                    "OpenAI pending availability retry attribution is absent from this batch",
                    hint=(
                        "Resume with the same selected rows and source-attempt ledger; do not "
                        "silently reassign a provider call to another consumer."
                    ),
                )
            provider_call_attempt = pending.next_provider_call_attempt
            time.sleep(pending.backoff_seconds)

        while True:
            started = time.monotonic()
            try:
                response = self._client_for_use().responses.create(**prepared.body)
                break
            except Exception as error:
                elapsed = time.monotonic() - started
                is_retryable = self._service_tier == "flex" and _is_flex_resource_unavailable(error)
                if is_retryable:
                    retry_index = provider_call_attempt - 1
                    retry_scheduled = retry_index < len(
                        OPENAI_FLEX_RESOURCE_UNAVAILABLE_BACKOFF_SECONDS
                    )
                    backoff_seconds = (
                        OPENAI_FLEX_RESOURCE_UNAVAILABLE_BACKOFF_SECONDS[retry_index]
                        if retry_scheduled
                        else None
                    )
                    self._append_journal(
                        self._resource_unavailable_journal_record(
                            prepared,
                            error,
                            elapsed_seconds=elapsed,
                            provider_call_attempt=provider_call_attempt,
                            retry_scheduled=retry_scheduled,
                            backoff_seconds=backoff_seconds,
                        )
                    )
                    if retry_scheduled:
                        assert backoff_seconds is not None
                        time.sleep(backoff_seconds)
                        provider_call_attempt += 1
                        continue
                    raise ExternalDependencyError(
                        "OpenAI Flex resource-unavailable retries were exhausted",
                        hint=(
                            "The exact 429 resource_unavailable events are fsynced in the "
                            "provider journal; the adapter never changed service tier."
                        ),
                    ) from None

                self._append_journal(
                    self._request_error_journal_record(
                        prepared,
                        error,
                        elapsed_seconds=elapsed,
                        provider_call_attempt=provider_call_attempt,
                    )
                )
                raise ExternalDependencyError(
                    "OpenAI Responses translation request failed",
                    hint=(
                        "Inspect the provider journal error type. SDK retries are disabled, "
                        "and only exact Flex 429 resource_unavailable errors are retried."
                    ),
                ) from None

        elapsed = time.monotonic() - started
        observed = _observe_response(response)
        completion = _classify_response(observed)
        self._append_journal(
            self._response_journal_record(
                prepared,
                observed,
                completion,
                elapsed_seconds=elapsed,
                provider_call_attempt=provider_call_attempt,
            )
        )
        self._pending_availability_retries.pop(prepared.sha256, None)

        if observed.returned_model != self._model_snapshot:
            raise ExternalDependencyError(
                "OpenAI Responses returned a different model than requested",
                hint=(f"Requested {self._model_snapshot!r}; received {observed.returned_model!r}."),
            )
        if observed.returned_service_tier != self._service_tier:
            raise ExternalDependencyError(
                "OpenAI Responses returned a different service tier than requested",
                hint=(
                    f"Requested {self._service_tier!r}; "
                    f"received {observed.returned_service_tier!r}."
                ),
            )
        if completion is None:
            status = observed.status or "missing"
            raise ExternalDependencyError(
                f"OpenAI Responses returned an unusable response status: {status}",
                hint="Inspect the fsynced provider journal for the safe error code and status.",
            )

        self._check_response_list_price(observed.usage)

        with self._cache_lock:
            self._replay_cache[prepared.sha256] = _CachedCompletion(
                completion=completion,
                canonical_body_utf8_bytes=prepared.canonical_body_utf8_bytes,
            )
        return completion

    def _check_response_list_price(self, usage: Mapping[str, object]) -> None:
        """Apply the stop guard immediately after each accepted response."""
        response_price = observed_openai_list_price_usd(
            _response_usage_for_pricing(usage),
            service_tier=self._service_tier,
        )
        with self._price_lock:
            self._running_list_price_usd += response_price
            observed = self._running_list_price_usd
        if observed > self._list_price_limit_usd:
            raise UserInputError(
                "OpenAI observed provider list-price estimate exceeds the explicit limit",
                hint=(
                    f"Observed at least {decimal_usd_string(observed)} USD versus limit "
                    f"{format(self._list_price_limit_usd, 'f')} USD. The response journal "
                    "is preserved; no additional provider batch may be admitted."
                ),
            )

    def _observed_list_price_usd(
        self,
        *,
        require_client_match: bool = False,
    ) -> Decimal:
        """Return the complete journal estimate or fail closed on unknown usage."""
        if not self._journal_path.exists():
            return Decimal("0")
        aggregate = aggregate_openai_responses_provider_journal(self._journal_path)
        counts = aggregate.get("counts")
        if not isinstance(counts, Mapping):  # pragma: no cover - aggregate contract
            raise RuntimeError("OpenAI provider aggregate is missing counts")
        records = counts.get("records")
        if records == 0:
            return Decimal("0")
        responses = counts.get("responses")
        expected_returned_models = [self._model_snapshot] if responses else []
        expected_returned_tiers = [self._service_tier] if responses else []
        if (
            aggregate.get("usage_complete") is not True
            or aggregate.get("requested_model") != self._model_snapshot
            or aggregate.get("returned_models") != expected_returned_models
            or aggregate.get("requested_service_tier") != self._service_tier
            or aggregate.get("returned_service_tiers") != expected_returned_tiers
            or aggregate.get("requested_timeout_seconds") != self._timeout_seconds
            or aggregate.get("sdk_max_retries") != OPENAI_RESPONSES_SDK_MAX_RETRIES
            or aggregate.get("resource_unavailable_retry_policy")
            != openai_flex_resource_unavailable_retry_policy()
            or (
                require_client_match
                and aggregate.get("client_injected") is not self._client_injected
            )
        ):
            raise UserInputError(
                "OpenAI provider journal cannot establish a complete list-price estimate",
                hint=(
                    "Preserve the journal for audit and resolve missing usage or provider "
                    "identity drift before authorizing another paid batch."
                ),
            )
        usage = aggregate.get("usage")
        if not isinstance(usage, Mapping):  # pragma: no cover - aggregate contract
            raise RuntimeError("OpenAI provider aggregate is missing usage")
        return observed_openai_list_price_usd(
            usage,
            service_tier=self._service_tier,
        )

    def _check_missing_batch_list_price(
        self,
        missing: Sequence[_PreparedRequest],
    ) -> None:
        if not missing:
            return
        # A durable replay may be consumed without instantiating the SDK, even
        # when its test/transport origin differs. Any *new* provider call must
        # retain the existing journal's client boundary so a paid batch cannot
        # silently mix injected transport policy with the pinned SDK client.
        observed = self._observed_list_price_usd(require_client_match=True)
        with self._price_lock:
            self._running_list_price_usd = observed
        batch_bound = missing_batch_list_price_upper_bound_usd(
            (prepared.canonical_body_utf8_bytes for prepared in missing),
            max_output_tokens=self._max_output_tokens,
            service_tier=self._service_tier,
        )
        projected = observed + batch_bound
        if projected > self._list_price_limit_usd:
            raise UserInputError(
                "OpenAI missing provider batch exceeds the explicit list-price limit",
                hint=(
                    f"Observed journal estimate {decimal_usd_string(observed)} USD plus "
                    f"batch upper bound {decimal_usd_string(batch_bound)} USD exceeds "
                    f"limit {format(self._list_price_limit_usd, 'f')} USD. Raise the "
                    "explicit limit only after reviewing the journal and provider-side "
                    "account/project spend controls."
                ),
            )

    def _check_observed_list_price(self) -> None:
        observed = self._observed_list_price_usd()
        if observed > self._list_price_limit_usd:
            raise UserInputError(
                "OpenAI observed provider list-price estimate exceeds the explicit limit",
                hint=(
                    f"Observed {decimal_usd_string(observed)} USD versus limit "
                    f"{format(self._list_price_limit_usd, 'f')} USD. The response journal "
                    "is preserved; do not authorize another paid batch without review."
                ),
            )

    def _translate_batch_admitted(
        self,
        requests: list[TranslationRequest],
    ) -> list[DecodedTranslationCompletion]:
        prepared_requests = [
            self._prepare_request(request, batch_position=position)
            for position, request in enumerate(requests)
        ]
        results: list[DecodedTranslationCompletion | None] = [None] * len(requests)
        missing_by_sha: dict[str, _PreparedRequest] = {}
        positions_by_sha: dict[str, list[int]] = {}
        with self._cache_lock:
            cached = dict(self._replay_cache)
        for prepared in prepared_requests:
            replayed = cached.get(prepared.sha256)
            if replayed is not None:
                if replayed.canonical_body_utf8_bytes != prepared.canonical_body_utf8_bytes:
                    raise UserInputError(
                        "OpenAI replay request-body byte count does not match its prepared body"
                    )
                self._append_journal(
                    self._replay_journal_record(
                        prepared,
                        replayed.completion,
                        source="durable_journal",
                    )
                )
                results[prepared.batch_position] = replayed.completion
                continue
            missing_by_sha.setdefault(prepared.sha256, prepared)
            positions_by_sha.setdefault(prepared.sha256, []).append(prepared.batch_position)

        for request_sha256, pending in self._pending_availability_retries.items():
            if request_sha256 not in missing_by_sha:
                raise UserInputError(
                    "OpenAI pending availability retry is absent from this request batch",
                    hint="Resume the same deterministic row selection and attempt ledger.",
                )
            matching = next(
                (
                    prepared
                    for prepared in prepared_requests
                    if prepared.sha256 == request_sha256
                    and prepared.source_id == pending.source_id
                    and prepared.attempt == pending.attempt
                ),
                None,
            )
            if matching is None:
                raise UserInputError(
                    "OpenAI pending availability retry attribution is absent from this batch"
                )
            missing_by_sha[request_sha256] = matching

        missing = list(missing_by_sha.values())
        self._check_missing_batch_list_price(missing)
        if self._max_workers == 1:
            generated = [self._request_one(prepared) for prepared in missing]
        else:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                generated = list(executor.map(self._request_one, missing))
        self._check_observed_list_price()
        for prepared, completion in zip(missing, generated, strict=True):
            for position in positions_by_sha[prepared.sha256]:
                if position != prepared.batch_position:
                    self._append_journal(
                        self._replay_journal_record(
                            prepared_requests[position],
                            completion,
                            source="batch_coalesced",
                        )
                    )
                results[position] = completion

        if any(result is None for result in results):
            raise RuntimeError("OpenAI Responses adapter did not preserve its request batch")
        return [cast(DecodedTranslationCompletion, result) for result in results]

    def translate_batch(
        self,
        requests: list[TranslationRequest],
    ) -> list[DecodedTranslationCompletion]:
        """Translate one batch under the serialized paid-call admission boundary."""
        with self._translate_batch_lock:
            return self._translate_batch_admitted(requests)


def load_openai_responses_translation_model(
    *,
    model_snapshot: str,
    max_output_tokens: int,
    provider_journal_path: Path,
    service_tier: OpenAIServiceTier = "default",
    client: OpenAIResponsesClient | None = None,
    expected_sdk_version: str = OPENAI_RESPONSES_SDK_VERSION,
    max_retries: int = 0,
    timeout_seconds: float = OPENAI_RESPONSES_TIMEOUT_SECONDS,
    max_workers: int = 1,
    openai_list_price_limit_usd: str,
) -> TranslationModel:
    """Build a lazy Responses translation adapter with an auditable journal.

    The SDK is imported only when the first non-replayed request is sent. Its
    installed version must exactly equal ``expected_sdk_version``. The default
    SDK retry count is required to remain zero. Exact Flex HTTP 429
    ``resource_unavailable`` errors use the separate journal-visible fixed
    backoff policy; no other error is retried. Injected clients are accepted
    for network-free tests and remain responsible for hidden transport policy.
    """
    match = _EXACT_MODEL_SNAPSHOT.fullmatch(model_snapshot)
    if match is None:
        raise UserInputError(
            "OpenAI Responses model must be an exact dated GPT snapshot",
            hint="Pass a model such as gpt-5.5-2026-04-23, never a moving alias.",
        )
    try:
        date.fromisoformat(model_snapshot[-10:])
    except ValueError as error:
        raise UserInputError("OpenAI Responses model snapshot date is invalid") from error
    if (
        not isinstance(max_output_tokens, int)
        or isinstance(max_output_tokens, bool)
        or max_output_tokens <= 0
    ):
        raise UserInputError("OpenAI Responses max_output_tokens must be a positive integer")
    if service_tier not in {"default", "flex"}:
        raise UserInputError(
            "OpenAI Responses service tier must be explicitly default or flex",
            hint="Use default for diagnostics or flex for the pinned production run; never auto.",
        )
    if _EXACT_SDK_VERSION.fullmatch(expected_sdk_version) is None:
        raise UserInputError("OpenAI Responses SDK version must be an exact x.y.z pin")
    if max_retries != OPENAI_RESPONSES_SDK_MAX_RETRIES or isinstance(max_retries, bool):
        raise UserInputError(
            "OpenAI Responses SDK max_retries must remain exactly zero",
            hint="Use only the journal-visible Flex resource-unavailable retry policy.",
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise UserInputError("OpenAI Responses timeout_seconds must be positive and finite")
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers <= 0:
        raise UserInputError("OpenAI Responses max_workers must be a positive integer")
    return _OpenAIResponsesTranslationModel(
        model_snapshot=model_snapshot,
        max_output_tokens=max_output_tokens,
        provider_journal_path=provider_journal_path,
        service_tier=service_tier,
        client=client,
        expected_sdk_version=expected_sdk_version,
        max_retries=max_retries,
        timeout_seconds=float(timeout_seconds),
        max_workers=max_workers,
        openai_list_price_limit_usd=openai_list_price_limit_usd,
    )


def _journal_mapping(value: object, *, line_number: int, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise UserInputError(f"OpenAI provider journal line {line_number} has invalid {field}")
    return cast(dict[str, object], value)


def _journal_string(value: object, *, line_number: int, field: str) -> str:
    if not isinstance(value, str):
        raise UserInputError(f"OpenAI provider journal line {line_number} has invalid {field}")
    return value


def _journal_bool(value: object, *, line_number: int, field: str) -> bool:
    if not isinstance(value, bool):
        raise UserInputError(f"OpenAI provider journal line {line_number} has invalid {field}")
    return value


def _journal_token_count(value: object, *, line_number: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UserInputError(f"OpenAI provider journal line {line_number} has invalid {field}")
    return value


def _journal_attribution(
    value: object,
    *,
    line_number: int,
) -> tuple[str, int]:
    attribution = _journal_mapping(
        value,
        line_number=line_number,
        field="source-attempt attribution",
    )
    if set(attribution) != {"source_id", "attempt"}:
        raise UserInputError(
            f"OpenAI provider journal line {line_number} has invalid source-attempt attribution"
        )
    source_id = _journal_string(
        attribution.get("source_id"),
        line_number=line_number,
        field="attribution.source_id",
    )
    attempt = attribution.get("attempt")
    if not source_id or isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise UserInputError(
            f"OpenAI provider journal line {line_number} has invalid source-attempt attribution"
        )
    return source_id, attempt


_JOURNAL_BASE_RECORD_KEYS: Final = frozenset(
    {
        "schema_version",
        "observed_at",
        "request_sha256",
        "batch_position",
        "attribution",
        "request",
        "event",
    }
)
_JOURNAL_EVENT_RECORD_KEYS: Final = {
    "response": frozenset(
        {
            "elapsed_seconds",
            "provider_call_attempt",
            "replayable",
            "response",
            "completion",
            "raw_output",
            "raw_output_sha256",
            "refusals",
            "usage",
        }
    ),
    "replay": frozenset({"replay", "completion", "usage"}),
    "request_error": frozenset(
        {
            "elapsed_seconds",
            "provider_call_attempt",
            "replayable",
            "error",
            "raw_output",
            "raw_output_sha256",
            "refusals",
            "usage",
        }
    ),
    "resource_unavailable": frozenset(
        {
            "elapsed_seconds",
            "provider_call_attempt",
            "replayable",
            "resource_unavailable",
            "raw_output",
            "raw_output_sha256",
            "refusals",
            "usage",
        }
    ),
}
_RESOURCE_UNAVAILABLE_METADATA_KEYS: Final = frozenset(
    {
        "error_type",
        "http_status",
        "error_code",
        "retry_scheduled",
        "backoff_seconds",
    }
)
_REQUEST_METADATA_KEYS: Final = frozenset(
    {
        "api",
        "requested_model",
        "requested_service_tier",
        "canonical_request_body_utf8_bytes",
        "max_output_tokens",
        "reasoning_effort",
        "safety_identifier",
        "store",
        "background",
        "truncation",
        "sdk_version_expected",
        "sdk_retries",
        "timeout_seconds",
        "client_injected",
        "resource_unavailable_retry_policy",
    }
)
_RESPONSE_METADATA_KEYS: Final = frozenset(
    {
        "id",
        "request_id",
        "returned_model",
        "model_matches_request",
        "status",
        "returned_service_tier",
        "service_tier_matches_request",
        "provider_created_at",
        "incomplete_reason",
        "error_code",
        "error_type",
    }
)


def _validate_retry_policy(value: object, *, line_number: int) -> None:
    policy = _journal_mapping(
        value,
        line_number=line_number,
        field="resource-unavailable retry policy",
    )
    if policy != openai_flex_resource_unavailable_retry_policy():
        raise UserInputError(
            f"OpenAI provider journal line {line_number} has a drifted retry policy"
        )


def _journal_provider_call_attempt(value: object, *, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UserInputError(
            f"OpenAI provider journal line {line_number} has invalid provider call attempt"
        )
    return value


def _validate_journal_record_shape(
    record: Mapping[str, object],
    *,
    event: str,
    line_number: int,
) -> None:
    event_keys = _JOURNAL_EVENT_RECORD_KEYS.get(event)
    if event_keys is None or set(record) != _JOURNAL_BASE_RECORD_KEYS | event_keys:
        raise UserInputError(
            f"OpenAI provider journal line {line_number} has invalid {event} record fields"
        )
    observed_at = record.get("observed_at")
    if not isinstance(observed_at, str):
        raise UserInputError(
            f"OpenAI provider journal line {line_number} has invalid observation time"
        )
    try:
        parsed_observed_at = datetime.fromisoformat(observed_at)
    except ValueError as error:
        raise UserInputError(
            f"OpenAI provider journal line {line_number} has invalid observation time"
        ) from error
    if parsed_observed_at.tzinfo is None:
        raise UserInputError(
            f"OpenAI provider journal line {line_number} has naive observation time"
        )
    batch_position = record.get("batch_position")
    if (
        isinstance(batch_position, bool)
        or not isinstance(batch_position, int)
        or batch_position < 0
    ):
        raise UserInputError(
            f"OpenAI provider journal line {line_number} has invalid batch position"
        )
    if event != "replay":
        elapsed_seconds = record.get("elapsed_seconds")
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, int | float)
            or not math.isfinite(float(elapsed_seconds))
            or elapsed_seconds < 0
        ):
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has invalid elapsed time"
            )


def _validated_journal_completion(
    value: object,
    *,
    line_number: int,
) -> tuple[str, str | None]:
    completion = _journal_mapping(value, line_number=line_number, field="completion")
    if set(completion) != {"disposition", "finish_reason"}:
        raise UserInputError(f"OpenAI provider journal line {line_number} has invalid completion")
    disposition = completion.get("disposition")
    finish_reason = completion.get("finish_reason")
    if disposition not in _COMPLETION_DISPOSITIONS or not (
        finish_reason is None or isinstance(finish_reason, str)
    ):
        raise UserInputError(f"OpenAI provider journal line {line_number} has invalid completion")
    return disposition, finish_reason


def _accumulate_journal_usage(
    value: object,
    totals: dict[str, int],
    *,
    line_number: int,
) -> tuple[bool, int | None]:
    usage = _journal_mapping(value, line_number=line_number, field="usage")
    allowed = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_tokens_details",
        "output_tokens_details",
    }
    if set(usage) - allowed:
        raise UserInputError(
            f"OpenAI provider journal line {line_number} has unsupported usage fields"
        )
    complete = {"input_tokens", "output_tokens", "total_tokens"}.issubset(usage)
    token_counts: dict[str, int] = {}
    for source, target in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        if source in usage:
            token_count = _journal_token_count(
                usage[source],
                line_number=line_number,
                field=f"usage.{source}",
            )
            if source == "input_tokens":
                try:
                    validate_openai_base_rate_response_input_tokens(token_count)
                except UserInputError:
                    raise UserInputError(
                        f"OpenAI provider journal line {line_number} crosses the pinned "
                        "base-rate input-token threshold"
                    ) from None
            token_counts[source] = token_count
            totals[target] += token_count
    if complete and token_counts["total_tokens"] != (
        token_counts["input_tokens"] + token_counts["output_tokens"]
    ):
        raise UserInputError(
            f"OpenAI provider journal line {line_number} has inconsistent total-token usage"
        )
    input_details_value = usage.get("input_tokens_details")
    if input_details_value is not None:
        input_details = _journal_mapping(
            input_details_value,
            line_number=line_number,
            field="usage.input_tokens_details",
        )
        if set(input_details) != {"cached_tokens"}:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has invalid cached-token usage"
            )
        cached_tokens = _journal_token_count(
            input_details["cached_tokens"],
            line_number=line_number,
            field="usage.input_tokens_details.cached_tokens",
        )
        if "input_tokens" in token_counts and cached_tokens > token_counts["input_tokens"]:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has cached input exceeding "
                "total input usage"
            )
        totals["cached_input_tokens"] += cached_tokens
    else:
        complete = False
    output_details_value = usage.get("output_tokens_details")
    if output_details_value is not None:
        output_details = _journal_mapping(
            output_details_value,
            line_number=line_number,
            field="usage.output_tokens_details",
        )
        if set(output_details) != {"reasoning_tokens"}:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has invalid reasoning-token usage"
            )
        reasoning_tokens = _journal_token_count(
            output_details["reasoning_tokens"],
            line_number=line_number,
            field="usage.output_tokens_details.reasoning_tokens",
        )
        if "output_tokens" in token_counts and reasoning_tokens > token_counts["output_tokens"]:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has reasoning output exceeding "
                "total output usage"
            )
        totals["reasoning_output_tokens"] += reasoning_tokens
    else:
        complete = False
    return complete, token_counts.get("input_tokens")


def aggregate_openai_responses_provider_journal(
    provider_journal_path: Path,
) -> dict[str, object]:
    """Return a validated, content-free accounting summary of one journal.

    The helper is read-only and deterministic for a given byte sequence. It
    validates the append records and their raw-output hashes but returns only
    whitelisted model/tier identities, counts, token totals, and the complete
    journal SHA. Prompts, decoded output, refusals, response IDs, error text,
    request hashes, credentials, and headers are never copied into the result.
    A journal may contain provider mismatches, but all records must belong to
    one requested model and one requested service tier.
    """
    try:
        journal_bytes = provider_journal_path.read_bytes()
    except OSError as error:
        raise ExternalDependencyError(
            f"could not read OpenAI provider journal: {provider_journal_path}"
        ) from error
    try:
        journal_text = journal_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise UserInputError("OpenAI provider journal is not valid UTF-8") from error

    counts = {
        "records": 0,
        "responses": 0,
        "replayable_responses": 0,
        "replays": 0,
        "durable_journal_replays": 0,
        "batch_coalesced_replays": 0,
        "request_errors": 0,
        "resource_unavailable_events": 0,
        "resolved_resource_unavailable_events": 0,
        "pending_resource_unavailable_events": 0,
        "unresolved_resource_unavailable_events": 0,
        "provider_error_responses": 0,
        "error_records": 0,
        "model_mismatch_responses": 0,
        "service_tier_mismatch_responses": 0,
        "refusal_responses": 0,
        "incomplete_responses": 0,
        "responses_missing_usage": 0,
    }
    usage_totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }
    requested_models: set[str] = set()
    requested_service_tiers: set[str] = set()
    requested_timeout_seconds: set[float] = set()
    client_injected_values: set[bool] = set()
    returned_models: set[str] = set()
    returned_service_tiers: set[str] = set()
    request_hashes: set[str] = set()
    canonical_body_bytes_by_request: dict[str, int] = {}
    source_attempts: set[tuple[str, int]] = set()
    request_hash_by_source_attempt: dict[tuple[str, int], str] = {}
    attempts_by_source: dict[str, set[int]] = {}
    replayable_output_hashes: dict[str, set[str]] = {}
    replayable_completions: dict[str, set[tuple[str, str | None]]] = {}
    provider_events_by_source_attempt: dict[
        tuple[str, int], list[tuple[str, int, bool | None]]
    ] = {}
    resource_unavailable_events_by_source_attempt: dict[tuple[str, int], int] = {}
    usage_complete = True
    max_response_input_tokens = 0

    for line_number, line in enumerate(journal_text.splitlines(), start=1):
        if not line.strip():
            continue
        counts["records"] += 1
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as error:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} is not valid JSON"
            ) from error
        record = _journal_mapping(decoded, line_number=line_number, field="record")
        if record.get("schema_version") != OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has an unsupported schema"
            )
        event = _journal_string(
            record.get("event"),
            line_number=line_number,
            field="event",
        )
        _validate_journal_record_shape(
            record,
            event=event,
            line_number=line_number,
        )
        source_attempt = _journal_attribution(
            record.get("attribution"),
            line_number=line_number,
        )
        source_attempts.add(source_attempt)
        request_sha256 = _journal_string(
            record.get("request_sha256"),
            line_number=line_number,
            field="request_sha256",
        )
        if re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has invalid request_sha256"
            )
        request_hashes.add(request_sha256)
        previous_request_hash = request_hash_by_source_attempt.setdefault(
            source_attempt,
            request_sha256,
        )
        if previous_request_hash != request_sha256:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} assigns multiple requests "
                "to one source attempt"
            )
        source_id, attempt = source_attempt
        attempts_by_source.setdefault(source_id, set()).add(attempt)
        request = _journal_mapping(
            record.get("request"),
            line_number=line_number,
            field="request metadata",
        )
        if set(request) != _REQUEST_METADATA_KEYS:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has invalid request metadata fields"
            )
        requested_model = _journal_string(
            request.get("requested_model"),
            line_number=line_number,
            field="requested model",
        )
        requested_tier = _journal_string(
            request.get("requested_service_tier"),
            line_number=line_number,
            field="requested service tier",
        )
        if _EXACT_MODEL_SNAPSHOT.fullmatch(requested_model) is None:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has a moving requested model"
            )
        if requested_tier not in {"default", "flex"}:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has an unsupported requested tier"
            )
        if request.get("safety_identifier") != OPENAI_RESPONSES_SAFETY_IDENTIFIER:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has an invalid safety identifier"
            )
        max_output_tokens = request.get("max_output_tokens")
        canonical_request_body_utf8_bytes = request.get("canonical_request_body_utf8_bytes")
        sdk_retries = request.get("sdk_retries")
        timeout_seconds = request.get("timeout_seconds")
        client_injected = request.get("client_injected")
        retry_policy = request.get("resource_unavailable_retry_policy")
        if (
            request.get("api") != "responses"
            or request.get("reasoning_effort") != "none"
            or request.get("store") is not False
            or request.get("background") is not False
            or request.get("truncation") != "disabled"
            or isinstance(canonical_request_body_utf8_bytes, bool)
            or not isinstance(canonical_request_body_utf8_bytes, int)
            or canonical_request_body_utf8_bytes <= 0
            or isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
            or _EXACT_SDK_VERSION.fullmatch(
                _journal_string(
                    request.get("sdk_version_expected"),
                    line_number=line_number,
                    field="expected SDK version",
                )
            )
            is None
            or isinstance(sdk_retries, bool)
            or not isinstance(sdk_retries, int)
            or sdk_retries < 0
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int | float)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
            or not isinstance(client_injected, bool)
        ):
            raise UserInputError(
                f"OpenAI provider journal line {line_number} violates the request contract"
            )
        if sdk_retries != OPENAI_RESPONSES_SDK_MAX_RETRIES:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} enables hidden SDK retries"
            )
        _validate_retry_policy(retry_policy, line_number=line_number)
        validate_openai_base_rate_request_body_bytes(canonical_request_body_utf8_bytes)
        previous_body_bytes = canonical_body_bytes_by_request.setdefault(
            request_sha256,
            canonical_request_body_utf8_bytes,
        )
        if previous_body_bytes != canonical_request_body_utf8_bytes:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} changes the canonical "
                "request-body byte count for one request"
            )
        requested_models.add(requested_model)
        requested_service_tiers.add(requested_tier)
        requested_timeout_seconds.add(float(timeout_seconds))
        if event != "replay":
            client_injected_values.add(client_injected)

        if event == "response":
            provider_call_attempt = _journal_provider_call_attempt(
                record.get("provider_call_attempt"),
                line_number=line_number,
            )
            provider_events_by_source_attempt.setdefault(source_attempt, []).append(
                (event, provider_call_attempt, None)
            )
            counts["responses"] += 1
            response = _journal_mapping(
                record.get("response"),
                line_number=line_number,
                field="response metadata",
            )
            if set(response) != _RESPONSE_METADATA_KEYS:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has invalid response fields"
                )
            for optional_string_field in (
                "id",
                "request_id",
                "incomplete_reason",
                "error_code",
                "error_type",
            ):
                optional_value = response.get(optional_string_field)
                if optional_value is not None and not isinstance(optional_value, str):
                    raise UserInputError(
                        f"OpenAI provider journal line {line_number} has invalid response fields"
                    )
            provider_created_at = response.get("provider_created_at")
            if provider_created_at is not None and (
                isinstance(provider_created_at, bool)
                or not isinstance(provider_created_at, int | float)
                or not math.isfinite(float(provider_created_at))
            ):
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has invalid response fields"
                )
            returned_model = _journal_string(
                response.get("returned_model"),
                line_number=line_number,
                field="returned model",
            )
            returned_tier = _journal_string(
                response.get("returned_service_tier"),
                line_number=line_number,
                field="returned service tier",
            )
            returned_models.add(returned_model)
            returned_service_tiers.add(returned_tier)
            if _EXACT_MODEL_SNAPSHOT.fullmatch(returned_model) is None:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has an invalid returned model"
                )
            if returned_tier not in _KNOWN_RETURNED_SERVICE_TIERS:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has an invalid returned tier"
                )
            _journal_string(
                response.get("status"),
                line_number=line_number,
                field="response status",
            )
            model_matches = _journal_bool(
                response.get("model_matches_request"),
                line_number=line_number,
                field="model match",
            )
            service_tier_matches = _journal_bool(
                response.get("service_tier_matches_request"),
                line_number=line_number,
                field="service-tier match",
            )
            if model_matches != (returned_model == requested_model):
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has inconsistent model identity"
                )
            if service_tier_matches != (returned_tier == requested_tier):
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has inconsistent tier identity"
                )
            if not model_matches:
                counts["model_mismatch_responses"] += 1
                usage_complete = False
            if not service_tier_matches:
                counts["service_tier_mismatch_responses"] += 1
                usage_complete = False

            raw_output = _journal_string(
                record.get("raw_output"),
                line_number=line_number,
                field="raw output",
            )
            raw_output_sha256 = _journal_string(
                record.get("raw_output_sha256"),
                line_number=line_number,
                field="raw-output SHA",
            )
            if hashlib.sha256(raw_output.encode("utf-8")).hexdigest() != raw_output_sha256:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} failed its raw-output hash"
                )
            refusals = record.get("refusals")
            if not isinstance(refusals, list) or any(
                not isinstance(refusal, str) for refusal in refusals
            ):
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has invalid refusals"
                )
            if refusals:
                counts["refusal_responses"] += 1
            if response.get("status") == "incomplete":
                counts["incomplete_responses"] += 1

            completion_value = record.get("completion")
            completion_identity = (
                _validated_journal_completion(completion_value, line_number=line_number)
                if completion_value is not None
                else None
            )
            replayable = _journal_bool(
                record.get("replayable"),
                line_number=line_number,
                field="replayable",
            )
            expected_replayable = (
                model_matches and service_tier_matches and completion_identity is not None
            )
            if replayable != expected_replayable:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has inconsistent replayability"
                )
            if replayable:
                counts["replayable_responses"] += 1
                replayable_output_hashes.setdefault(request_sha256, set()).add(raw_output_sha256)
                assert completion_identity is not None
                replayable_completions.setdefault(request_sha256, set()).add(completion_identity)
            provider_error = completion_identity is None
            if provider_error:
                counts["provider_error_responses"] += 1
            if provider_error or not model_matches or not service_tier_matches:
                counts["error_records"] += 1
            response_usage_complete, response_input_tokens = _accumulate_journal_usage(
                record.get("usage"),
                usage_totals,
                line_number=line_number,
            )
            if response_input_tokens is not None:
                max_response_input_tokens = max(
                    max_response_input_tokens,
                    response_input_tokens,
                )
            if not response_usage_complete:
                counts["responses_missing_usage"] += 1
                usage_complete = False
        elif event == "replay":
            counts["replays"] += 1
            replay = _journal_mapping(
                record.get("replay"),
                line_number=line_number,
                field="replay metadata",
            )
            if set(replay) != {"source", "raw_output_sha256"}:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has invalid replay fields"
                )
            source = _journal_string(
                replay.get("source"),
                line_number=line_number,
                field="replay source",
            )
            if source not in _REPLAY_SOURCES:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has an invalid replay source"
                )
            counts[f"{source}_replays"] += 1
            replayed_output_sha256 = _journal_string(
                replay.get("raw_output_sha256"),
                line_number=line_number,
                field="replayed raw-output SHA",
            )
            completion_identity = _validated_journal_completion(
                record.get("completion"),
                line_number=line_number,
            )
            if replayed_output_sha256 not in replayable_output_hashes.get(
                request_sha256, set()
            ) or completion_identity not in replayable_completions.get(request_sha256, set()):
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has no prior replayable response"
                )
            if record.get("usage") != {}:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} assigns usage to a replay"
                )
        elif event == "request_error":
            provider_call_attempt = _journal_provider_call_attempt(
                record.get("provider_call_attempt"),
                line_number=line_number,
            )
            provider_events_by_source_attempt.setdefault(source_attempt, []).append(
                (event, provider_call_attempt, None)
            )
            counts["request_errors"] += 1
            counts["error_records"] += 1
            usage_complete = False
            error_metadata = _journal_mapping(
                record.get("error"),
                line_number=line_number,
                field="request error metadata",
            )
            if set(error_metadata) != {"type"}:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has invalid request error fields"
                )
            _journal_string(
                error_metadata.get("type"),
                line_number=line_number,
                field="request error type",
            )
            if record.get("raw_output") is not None or record.get("raw_output_sha256") is not None:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} exposes request-error output"
                )
            if record.get("usage") != {}:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} assigns usage to a request error"
                )
        elif event == "resource_unavailable":
            provider_call_attempt = _journal_provider_call_attempt(
                record.get("provider_call_attempt"),
                line_number=line_number,
            )
            metadata = _journal_mapping(
                record.get("resource_unavailable"),
                line_number=line_number,
                field="resource-unavailable metadata",
            )
            if set(metadata) != _RESOURCE_UNAVAILABLE_METADATA_KEYS:
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} has invalid "
                    "resource-unavailable metadata"
                )
            _journal_string(
                metadata.get("error_type"),
                line_number=line_number,
                field="resource-unavailable error type",
            )
            retry_scheduled = _journal_bool(
                metadata.get("retry_scheduled"),
                line_number=line_number,
                field="resource-unavailable retry flag",
            )
            retry_index = provider_call_attempt - 1
            expected_retry_scheduled = retry_index < len(
                OPENAI_FLEX_RESOURCE_UNAVAILABLE_BACKOFF_SECONDS
            )
            expected_backoff = (
                OPENAI_FLEX_RESOURCE_UNAVAILABLE_BACKOFF_SECONDS[retry_index]
                if expected_retry_scheduled
                else None
            )
            if (
                requested_tier != "flex"
                or metadata.get("http_status") != OPENAI_FLEX_RESOURCE_UNAVAILABLE_HTTP_STATUS
                or metadata.get("error_code") != OPENAI_FLEX_RESOURCE_UNAVAILABLE_ERROR_CODE
                or retry_scheduled != expected_retry_scheduled
                or metadata.get("backoff_seconds") != expected_backoff
                or record.get("replayable") is not False
                or record.get("raw_output") is not None
                or record.get("raw_output_sha256") is not None
                or record.get("refusals") != []
                or record.get("usage") != {}
            ):
                raise UserInputError(
                    f"OpenAI provider journal line {line_number} violates the "
                    "resource-unavailable retry contract"
                )
            counts["resource_unavailable_events"] += 1
            resource_unavailable_events_by_source_attempt[source_attempt] = (
                resource_unavailable_events_by_source_attempt.get(source_attempt, 0) + 1
            )
            provider_events_by_source_attempt.setdefault(source_attempt, []).append(
                (event, provider_call_attempt, retry_scheduled)
            )
        else:
            raise UserInputError(
                f"OpenAI provider journal line {line_number} has an unsupported event"
            )

    for source_attempt, provider_events in provider_events_by_source_attempt.items():
        terminal_seen = False
        for index, (event, provider_call_attempt, scheduled) in enumerate(
            provider_events,
            start=1,
        ):
            if provider_call_attempt != index or terminal_seen:
                raise UserInputError(
                    "OpenAI provider journal has a non-contiguous or post-terminal "
                    "provider call-attempt ledger"
                )
            if event == "resource_unavailable":
                if scheduled is False:
                    terminal_seen = True
                elif index == len(provider_events):
                    continue
            else:
                terminal_seen = True
        last_event, _last_attempt, last_retry_scheduled = provider_events[-1]
        availability_events = resource_unavailable_events_by_source_attempt.get(
            source_attempt,
            0,
        )
        if availability_events == 0:
            continue
        if last_event == "response":
            counts["resolved_resource_unavailable_events"] += availability_events
        elif last_event == "resource_unavailable" and last_retry_scheduled is True:
            counts["pending_resource_unavailable_events"] += availability_events
        else:
            counts["unresolved_resource_unavailable_events"] += availability_events

    if counts["unresolved_resource_unavailable_events"]:
        counts["error_records"] += counts["unresolved_resource_unavailable_events"]
        usage_complete = False
    if counts["resource_unavailable_events"] != (
        counts["resolved_resource_unavailable_events"]
        + counts["pending_resource_unavailable_events"]
        + counts["unresolved_resource_unavailable_events"]
    ):
        raise RuntimeError("OpenAI availability event classification is inconsistent")

    for request_sha256, output_hashes in replayable_output_hashes.items():
        completions = replayable_completions.get(request_sha256, set())
        if len(output_hashes) > 1 or len(completions) > 1:
            raise UserInputError(
                "OpenAI provider journal has conflicting replayable responses for one request"
            )

    if len(requested_models) > 1:
        raise UserInputError("OpenAI provider journal mixes requested models")
    if len(requested_service_tiers) > 1:
        raise UserInputError("OpenAI provider journal mixes requested service tiers")
    if len(requested_timeout_seconds) > 1:
        raise UserInputError("OpenAI provider journal mixes requested timeouts")
    if len(client_injected_values) > 1:
        raise UserInputError("OpenAI provider journal mixes injected and pinned SDK clients")
    for source_id, attempts in attempts_by_source.items():
        expected_attempts = set(range(1, max(attempts) + 1))
        if attempts != expected_attempts:
            raise UserInputError(
                f"OpenAI provider journal has non-contiguous attempts for source_id {source_id!r}"
            )

    return {
        "schema_version": OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA,
        "journal_schema_version": OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
        "journal_sha256": hashlib.sha256(journal_bytes).hexdigest(),
        "requested_model": next(iter(requested_models), None),
        "returned_models": sorted(returned_models),
        "requested_service_tier": next(iter(requested_service_tiers), None),
        "requested_timeout_seconds": next(iter(requested_timeout_seconds), None),
        "returned_service_tiers": sorted(returned_service_tiers),
        "safety_identifier": OPENAI_RESPONSES_SAFETY_IDENTIFIER,
        "client_injected": next(iter(client_injected_values), None),
        "sdk_max_retries": OPENAI_RESPONSES_SDK_MAX_RETRIES,
        "resource_unavailable_retry_policy": (openai_flex_resource_unavailable_retry_policy()),
        "max_canonical_request_body_utf8_bytes": max(
            canonical_body_bytes_by_request.values(),
            default=0,
        ),
        "max_response_input_tokens": max_response_input_tokens,
        "unique_requests": len(request_hashes),
        "unique_source_attempts": len(source_attempts),
        "usage_complete": usage_complete,
        "counts": counts,
        "usage": usage_totals,
    }
