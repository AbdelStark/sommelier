from __future__ import annotations

import hashlib
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from sommelier.data.openai_translate import (
    OPENAI_FLEX_RESOURCE_UNAVAILABLE_BACKOFF_SECONDS,
    OPENAI_FLEX_RESOURCE_UNAVAILABLE_ERROR_CODE,
    OPENAI_FLEX_RESOURCE_UNAVAILABLE_HTTP_STATUS,
    OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
    OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA,
    OPENAI_RESPONSES_SAFETY_IDENTIFIER,
    OPENAI_RESPONSES_SDK_VERSION,
    OPENAI_RESPONSES_TEXT_FORMAT_NAME,
    OPENAI_RESPONSES_TIMEOUT_SECONDS,
    aggregate_openai_responses_provider_journal,
)
from sommelier.data.openai_translate import (
    load_openai_responses_translation_model as _load_openai_responses_translation_model,
)
from sommelier.data.translate import (
    INSTRUCTION_CHAT_ASSISTANT_PAYLOAD_SCHEMA,
    DecodedTranslationCompletion,
    TranslationRequest,
)
from sommelier.errors import ExternalDependencyError, UserInputError

MODEL_SNAPSHOT = "gpt-5.5-2026-04-23"
load_openai_responses_translation_model = partial(
    _load_openai_responses_translation_model,
    openai_list_price_limit_usd="1000.00",
)


class _FakeResponses:
    def __init__(self, outcomes: list[object | Exception]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if not self._outcomes:
            raise AssertionError("unexpected provider call")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list[object | Exception]) -> None:
        self.responses = _FakeResponses(outcomes)


class _FakeStatusError(Exception):
    def __init__(self, *, status_code: int, code: str, secret: str = "") -> None:
        super().__init__(f"provider message must stay private {secret}")
        self.status_code = status_code
        self.body = {
            "code": code,
            "type": "server_error",
            "message": f"provider message must stay private {secret}",
        }


def _request(
    query: str = "Show the weather",
    *,
    source_id: str = "root-1",
    attempt: int = 1,
) -> TranslationRequest:
    return TranslationRequest(
        query=query,
        protected_spans=(),
        target_language="he",
        semantic_context="{}",
        source_id=source_id,
        attempt=attempt,
    )


def _assistant_payload(target_text: str) -> str:
    return json.dumps(
        {
            "schema_version": INSTRUCTION_CHAT_ASSISTANT_PAYLOAD_SCHEMA,
            "target_text": target_text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _response(
    *,
    raw_output: str = "",
    model: str = MODEL_SNAPSHOT,
    status: str = "completed",
    service_tier: str = "default",
    refusal: str | None = None,
    incomplete_reason: str | None = None,
    error: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    content: list[dict[str, object]] = []
    if raw_output:
        content.append({"type": "output_text", "text": raw_output, "annotations": []})
    if refusal is not None:
        content.append({"type": "refusal", "refusal": refusal})
    response: dict[str, object] = {
        "id": "resp_123",
        "_request_id": "req_123",
        "model": model,
        "status": status,
        "service_tier": service_tier,
        "created_at": 1_783_900_000,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": status,
                "content": content,
            }
        ],
        "usage": {
            "input_tokens": 101,
            "output_tokens": 17,
            "total_tokens": 118,
            "input_tokens_details": {"cached_tokens": 23},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "incomplete_details": (
            {"reason": incomplete_reason} if incomplete_reason is not None else None
        ),
        "error": error,
    }
    if extra:
        response.update(extra)
    return response


def _journal_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        assert isinstance(payload, dict)
        records.append(cast(dict[str, object], payload))
    return records


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def test_success_uses_strict_responses_contract_and_journals_usage(
    tmp_path: Path,
) -> None:
    raw_output = _assistant_payload("הצג את מזג האוויר")
    secret = "sk-test-never-write-this"
    fake = _FakeClient(
        [
            _response(
                raw_output=raw_output,
                service_tier="flex",
                extra={
                    "headers": {"authorization": f"Bearer {secret}"},
                    "api_key": secret,
                },
            )
        ]
    )
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        service_tier="flex",
        client=fake,
    )

    outputs = model.translate_batch([_request()])

    assert outputs == [
        DecodedTranslationCompletion(
            text=raw_output,
            disposition="complete",
            finish_reason="completed",
        )
    ]
    assert len(fake.responses.calls) == 1
    call = fake.responses.calls[0]
    assert set(call) == {
        "model",
        "input",
        "text",
        "reasoning",
        "safety_identifier",
        "store",
        "background",
        "service_tier",
        "max_output_tokens",
        "truncation",
    }
    assert call["model"] == MODEL_SNAPSHOT
    assert call["store"] is False
    assert call["background"] is False
    assert call["service_tier"] == "flex"
    assert call["reasoning"] == {"effort": "none"}
    assert call["safety_identifier"] == OPENAI_RESPONSES_SAFETY_IDENTIFIER
    assert call["max_output_tokens"] == 256
    assert call["truncation"] == "disabled"
    text = _mapping(call["text"])
    text_format = _mapping(text["format"])
    assert text_format == {
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

    records = _journal_records(journal)
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA
    assert record["event"] == "response"
    assert record["replayable"] is True
    assert record["attribution"] == {"source_id": "root-1", "attempt": 1}
    assert (
        record["request_sha256"]
        == hashlib.sha256(
            json.dumps(
                call,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert record["raw_output"] == raw_output
    assert record["raw_output_sha256"] == hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
    assert record["usage"] == {
        "input_tokens": 101,
        "input_tokens_details": {"cached_tokens": 23},
        "output_tokens": 17,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 118,
    }
    request_metadata = _mapping(record["request"])
    response_metadata = _mapping(record["response"])
    assert request_metadata["sdk_version_expected"] == OPENAI_RESPONSES_SDK_VERSION
    assert request_metadata["sdk_retries"] == 0
    assert request_metadata["timeout_seconds"] == OPENAI_RESPONSES_TIMEOUT_SECONDS
    assert request_metadata["safety_identifier"] == OPENAI_RESPONSES_SAFETY_IDENTIFIER
    assert request_metadata["requested_service_tier"] == "flex"
    assert response_metadata["returned_model"] == MODEL_SNAPSHOT
    assert response_metadata["model_matches_request"] is True
    assert response_metadata["returned_service_tier"] == "flex"
    assert response_metadata["service_tier_matches_request"] is True
    assert secret not in journal.read_text(encoding="utf-8")
    assert "source_id" not in call
    assert "attempt" not in call


def test_refusal_returns_not_generated_and_is_journaled(tmp_path: Path) -> None:
    fake = _FakeClient([_response(refusal="I cannot translate that request.")])
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=fake,
    )

    outputs = model.translate_batch([_request()])

    assert outputs == [
        DecodedTranslationCompletion(
            text="",
            disposition="not_generated",
            finish_reason="refusal",
        )
    ]
    record = _journal_records(journal)[0]
    assert record["refusals"] == ["I cannot translate that request."]
    assert record["completion"] == {
        "disposition": "not_generated",
        "finish_reason": "refusal",
    }


def test_incomplete_response_preserves_partial_output(tmp_path: Path) -> None:
    partial = '{"schema_version":"sommelier.instruction_chat_assistant_payload.v1"'
    fake = _FakeClient(
        [
            _response(
                raw_output=partial,
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        ]
    )
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=64,
        provider_journal_path=journal,
        client=fake,
    )

    outputs = model.translate_batch([_request()])

    assert outputs == [
        DecodedTranslationCompletion(
            text=partial,
            disposition="incomplete",
            finish_reason="incomplete:max_output_tokens",
        )
    ]
    record = _journal_records(journal)[0]
    assert record["raw_output"] == partial
    assert record["replayable"] is True
    assert _mapping(record["response"])["incomplete_reason"] == "max_output_tokens"


def test_model_mismatch_is_journaled_then_rejected(tmp_path: Path) -> None:
    fake = _FakeClient(
        [_response(raw_output=_assistant_payload("טקסט"), model="gpt-5.5-2099-01-01")]
    )
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=fake,
    )

    with pytest.raises(ExternalDependencyError, match="different model"):
        model.translate_batch([_request()])

    record = _journal_records(journal)[0]
    assert record["replayable"] is False
    response_metadata = _mapping(record["response"])
    assert response_metadata["returned_model"] == "gpt-5.5-2099-01-01"
    assert response_metadata["model_matches_request"] is False
    summary = aggregate_openai_responses_provider_journal(journal)
    assert summary["returned_models"] == ["gpt-5.5-2099-01-01"]
    assert _mapping(summary["counts"])["model_mismatch_responses"] == 1
    assert summary["usage_complete"] is False


def test_service_tier_mismatch_is_journaled_then_rejected(tmp_path: Path) -> None:
    fake = _FakeClient([_response(raw_output=_assistant_payload("טקסט"), service_tier="default")])
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        service_tier="flex",
        client=fake,
    )

    with pytest.raises(ExternalDependencyError, match="different service tier"):
        model.translate_batch([_request()])

    record = _journal_records(journal)[0]
    assert record["replayable"] is False
    response_metadata = _mapping(record["response"])
    assert response_metadata["returned_service_tier"] == "default"
    assert response_metadata["service_tier_matches_request"] is False
    summary = aggregate_openai_responses_provider_journal(journal)
    assert summary["returned_service_tiers"] == ["default"]
    assert _mapping(summary["counts"])["service_tier_mismatch_responses"] == 1
    assert summary["usage_complete"] is False


def test_durable_response_replays_without_sdk_or_network(tmp_path: Path) -> None:
    raw_output = _assistant_payload("הצג את מזג האוויר")
    journal = tmp_path / "provider.jsonl"
    first_client = _FakeClient([_response(raw_output=raw_output)])
    first = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=first_client,
    )
    expected = first.translate_batch([_request()])
    assert len(first_client.responses.calls) == 1

    # No client is injected and the optional SDK is not installed in the test
    # environment. A matching durable record must therefore replay before the
    # lazy SDK boundary is touched.
    resumed = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
    )

    assert resumed.translate_batch([_request(source_id="root-1", attempt=1)]) == expected
    records = _journal_records(journal)
    assert len(records) == 2
    assert records[1]["event"] == "replay"
    assert records[1]["replay"] == {
        "source": "durable_journal",
        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
    }
    assert records[1]["attribution"] == {
        "source_id": "root-1",
        "attempt": 1,
    }
    summary = aggregate_openai_responses_provider_journal(journal)
    assert summary["unique_source_attempts"] == 1


def test_new_provider_call_rejects_existing_client_transport_drift(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "provider.jsonl"
    injected = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=_FakeClient([_response(raw_output=_assistant_payload("א"))]),
    )
    injected.translate_batch([_request("query a", source_id="root-a")])

    # Replay-only access remains possible for audit/tests, but a new paid call
    # cannot silently continue an injected-client ledger with the pinned SDK.
    pinned_sdk = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
    )
    with pytest.raises(UserInputError, match="complete list-price estimate"):
        pinned_sdk.translate_batch([_request("query b", source_id="root-b")])

    assert [record["event"] for record in _journal_records(journal)] == ["response"]


def test_identical_request_hashes_coalesce_with_exact_consumer_attribution(
    tmp_path: Path,
) -> None:
    fake = _FakeClient([_response(raw_output=_assistant_payload("טקסט"))])
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=tmp_path / "provider.jsonl",
        client=fake,
        max_workers=4,
    )

    outputs = model.translate_batch(
        [
            _request(source_id="root-a", attempt=1),
            _request(source_id="root-b", attempt=1),
        ]
    )

    assert len(outputs) == 2
    assert outputs[0] == outputs[1]
    assert len(fake.responses.calls) == 1
    records = _journal_records(tmp_path / "provider.jsonl")
    assert [record["event"] for record in records] == ["response", "replay"]
    assert records[0]["request_sha256"] == records[1]["request_sha256"]
    assert records[0]["attribution"] == {"source_id": "root-a", "attempt": 1}
    assert records[1]["attribution"] == {"source_id": "root-b", "attempt": 1}
    assert _mapping(records[1]["replay"])["source"] == "batch_coalesced"
    assert "source_id" not in fake.responses.calls[0]
    assert "attempt" not in fake.responses.calls[0]
    summary = aggregate_openai_responses_provider_journal(tmp_path / "provider.jsonl")
    assert summary["unique_requests"] == 1
    assert summary["unique_source_attempts"] == 2


def test_failed_response_is_journaled_without_provider_message(tmp_path: Path) -> None:
    secret = "sk-provider-message-secret"
    fake = _FakeClient(
        [
            _response(
                status="failed",
                error={
                    "code": "server_error",
                    "type": "server_error",
                    "message": f"Authorization: Bearer {secret}",
                },
                extra={"headers": {"authorization": secret}},
            )
        ]
    )
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=fake,
    )

    with pytest.raises(ExternalDependencyError, match="unusable response status"):
        model.translate_batch([_request()])

    journal_text = journal.read_text(encoding="utf-8")
    assert secret not in journal_text
    record = _journal_records(journal)[0]
    assert record["replayable"] is False
    response_metadata = _mapping(record["response"])
    assert response_metadata["error_code"] == "server_error"
    assert response_metadata["error_type"] == "server_error"


def test_transport_error_has_no_hidden_retry_and_never_journals_secret(
    tmp_path: Path,
) -> None:
    secret = "sk-transport-secret"
    fake = _FakeClient([RuntimeError(f"Authorization: Bearer {secret}")])
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=fake,
    )

    with pytest.raises(ExternalDependencyError, match="request failed") as caught:
        model.translate_batch([_request()])

    assert len(fake.responses.calls) == 1
    assert secret not in str(caught.value)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert secret not in journal.read_text(encoding="utf-8")
    record = _journal_records(journal)[0]
    assert record["event"] == "request_error"
    assert record["attribution"] == {"source_id": "root-1", "attempt": 1}
    assert record["error"] == {"type": "RuntimeError"}
    assert record["raw_output"] is None


def test_exact_flex_resource_unavailable_retries_are_journaled_and_resolve_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-resource-unavailable-secret"

    def unavailable() -> _FakeStatusError:
        return _FakeStatusError(
            status_code=OPENAI_FLEX_RESOURCE_UNAVAILABLE_HTTP_STATUS,
            code=OPENAI_FLEX_RESOURCE_UNAVAILABLE_ERROR_CODE,
            secret=secret,
        )

    fake = _FakeClient(
        [
            unavailable(),
            unavailable(),
            _response(raw_output=_assistant_payload("טקסט"), service_tier="flex"),
        ]
    )
    slept: list[float] = []
    monkeypatch.setattr("sommelier.data.openai_translate.time.sleep", slept.append)
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        service_tier="flex",
        client=fake,
    )

    output = model.translate_batch([_request()])

    assert len(output) == 1
    assert slept == [1.0, 2.0]
    assert len(fake.responses.calls) == 3
    assert {call["service_tier"] for call in fake.responses.calls} == {"flex"}
    records = _journal_records(journal)
    assert [record["event"] for record in records] == [
        "resource_unavailable",
        "resource_unavailable",
        "response",
    ]
    assert [record["provider_call_attempt"] for record in records] == [1, 2, 3]
    assert all(record["attribution"] == {"source_id": "root-1", "attempt": 1} for record in records)
    assert _mapping(records[0]["resource_unavailable"]) == {
        "error_type": "_FakeStatusError",
        "http_status": 429,
        "error_code": "resource_unavailable",
        "retry_scheduled": True,
        "backoff_seconds": 1.0,
    }
    assert secret not in journal.read_text(encoding="utf-8")
    summary = aggregate_openai_responses_provider_journal(journal)
    counts = _mapping(summary["counts"])
    assert counts["resource_unavailable_events"] == 2
    assert counts["resolved_resource_unavailable_events"] == 2
    assert counts["pending_resource_unavailable_events"] == 0
    assert counts["unresolved_resource_unavailable_events"] == 0
    assert counts["error_records"] == 0
    assert summary["usage_complete"] is True


@pytest.mark.parametrize(
    ("service_tier", "status_code", "code"),
    [
        ("flex", 429, "rate_limit_exceeded"),
        ("flex", 500, "resource_unavailable"),
        ("default", 429, "resource_unavailable"),
    ],
)
def test_only_exact_flex_resource_unavailable_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service_tier: str,
    status_code: int,
    code: str,
) -> None:
    fake = _FakeClient([_FakeStatusError(status_code=status_code, code=code)])
    slept: list[float] = []
    monkeypatch.setattr("sommelier.data.openai_translate.time.sleep", slept.append)
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        service_tier=service_tier,  # type: ignore[arg-type]
        client=fake,
    )

    with pytest.raises(ExternalDependencyError, match="request failed"):
        model.translate_batch([_request()])

    assert len(fake.responses.calls) == 1
    assert slept == []
    assert _journal_records(journal)[0]["event"] == "request_error"


def test_exhausted_flex_resource_unavailable_is_terminal_and_evidence_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-flex-exhaustion-trace-secret"
    outcomes: list[object | Exception] = [
        _FakeStatusError(status_code=429, code="resource_unavailable", secret=secret)
        for _ in range(len(OPENAI_FLEX_RESOURCE_UNAVAILABLE_BACKOFF_SECONDS) + 1)
    ]
    fake = _FakeClient(outcomes)
    slept: list[float] = []
    monkeypatch.setattr("sommelier.data.openai_translate.time.sleep", slept.append)
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        service_tier="flex",
        client=fake,
    )

    with pytest.raises(ExternalDependencyError, match="retries were exhausted") as caught:
        model.translate_batch([_request()])

    assert secret not in "".join(traceback.format_exception(caught.value))
    assert slept == list(OPENAI_FLEX_RESOURCE_UNAVAILABLE_BACKOFF_SECONDS)
    assert len(fake.responses.calls) == len(OPENAI_FLEX_RESOURCE_UNAVAILABLE_BACKOFF_SECONDS) + 1
    records = _journal_records(journal)
    assert all(record["event"] == "resource_unavailable" for record in records)
    assert _mapping(records[-1]["resource_unavailable"])["retry_scheduled"] is False
    assert _mapping(records[-1]["resource_unavailable"])["backoff_seconds"] is None
    summary = aggregate_openai_responses_provider_journal(journal)
    counts = _mapping(summary["counts"])
    assert counts["resolved_resource_unavailable_events"] == 0
    assert counts["pending_resource_unavailable_events"] == 0
    assert counts["unresolved_resource_unavailable_events"] == len(records)
    assert counts["error_records"] == len(records)
    assert summary["usage_complete"] is False


def test_pending_flex_resource_unavailable_resume_preserves_call_attempt_and_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "provider.jsonl"
    first = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        service_tier="flex",
        client=_FakeClient([_FakeStatusError(status_code=429, code="resource_unavailable")]),
    )

    def interrupt_after_journal(_seconds: float) -> None:
        raise RuntimeError("simulated process interruption during backoff")

    monkeypatch.setattr("sommelier.data.openai_translate.time.sleep", interrupt_after_journal)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        first.translate_batch([_request()])
    pending_summary = aggregate_openai_responses_provider_journal(journal)
    assert _mapping(pending_summary["counts"])["pending_resource_unavailable_events"] == 1
    assert pending_summary["usage_complete"] is True

    slept: list[float] = []
    monkeypatch.setattr("sommelier.data.openai_translate.time.sleep", slept.append)
    resumed_client = _FakeClient(
        [_response(raw_output=_assistant_payload("טקסט"), service_tier="flex")]
    )
    resumed = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        service_tier="flex",
        client=resumed_client,
    )

    resumed.translate_batch([_request(source_id="root-1", attempt=1)])

    assert slept == [1.0]
    records = _journal_records(journal)
    assert records[-1]["provider_call_attempt"] == 2
    assert records[-1]["attribution"] == {"source_id": "root-1", "attempt": 1}
    counts = _mapping(aggregate_openai_responses_provider_journal(journal)["counts"])
    assert counts["resolved_resource_unavailable_events"] == 1
    assert counts["pending_resource_unavailable_events"] == 0


def test_pre_batch_and_post_response_list_price_guards_stop_provider_access(
    tmp_path: Path,
) -> None:
    pre_client = _FakeClient([])
    pre_model = _load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=tmp_path / "pre.jsonl",
        client=pre_client,
        openai_list_price_limit_usd="0.000000001",
    )
    with pytest.raises(UserInputError, match="missing provider batch"):
        pre_model.translate_batch([_request()])
    assert pre_client.responses.calls == []

    high_usage_response = _response(raw_output=_assistant_payload("טקסט"))
    usage = _mapping(high_usage_response["usage"])
    usage.update(input_tokens=100_000, output_tokens=1_000_000, total_tokens=1_100_000)
    post_client = _FakeClient([high_usage_response, _response(raw_output=_assistant_payload("ב"))])
    post_model = _load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=tmp_path / "post.jsonl",
        client=post_client,
        openai_list_price_limit_usd="0.10",
    )
    with pytest.raises(UserInputError, match="observed provider list-price"):
        post_model.translate_batch(
            [_request("query a", source_id="root-a"), _request("query b", source_id="root-b")]
        )
    assert len(post_client.responses.calls) == 1


def test_concurrent_batches_share_one_serialized_list_price_admission(
    tmp_path: Path,
) -> None:
    first_entered = threading.Barrier(2)
    release_first = threading.Event()
    response = _response(raw_output=_assistant_payload("טקסט"))
    usage = _mapping(response["usage"])
    usage.update(input_tokens=4_000, output_tokens=500, total_tokens=4_500)

    class BlockingResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            first_entered.wait(timeout=5)
            assert release_first.wait(timeout=5)
            return response

    responses = BlockingResponses()
    client = cast(_FakeClient, SimpleNamespace(responses=responses))
    model = _load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=tmp_path / "concurrent.jsonl",
        client=client,
        openai_list_price_limit_usd="0.05",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            model.translate_batch,
            [_request("query a", source_id="root-a")],
        )
        first_entered.wait(timeout=5)
        second = executor.submit(
            model.translate_batch,
            [_request("query b", source_id="root-b")],
        )
        release_first.set()
        assert len(first.result(timeout=5)) == 1
        with pytest.raises(UserInputError, match="missing provider batch"):
            second.result(timeout=5)

    assert len(responses.calls) == 1


def test_client_initialization_traceback_suppresses_raw_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-client-init-trace-secret"
    monkeypatch.setattr(
        "sommelier.data.openai_translate.package_version",
        lambda _package: OPENAI_RESPONSES_SDK_VERSION,
    )
    monkeypatch.setattr(
        "sommelier.data.openai_translate.import_module",
        lambda _module: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=tmp_path / "client-init.jsonl",
    )

    with pytest.raises(ExternalDependencyError, match="request failed") as caught:
        model.translate_batch([_request()])

    assert secret not in "".join(traceback.format_exception(caught.value))


def test_paid_adapter_requires_explicit_limit_and_zero_sdk_retries(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="openai_list_price_limit_usd"):
        _load_openai_responses_translation_model(  # type: ignore[call-arg]
            model_snapshot=MODEL_SNAPSHOT,
            max_output_tokens=256,
            provider_journal_path=tmp_path / "provider.jsonl",
            client=_FakeClient([]),
        )
    with pytest.raises(UserInputError, match="list-price limit"):
        _load_openai_responses_translation_model(
            model_snapshot=MODEL_SNAPSHOT,
            max_output_tokens=256,
            provider_journal_path=tmp_path / "provider.jsonl",
            client=_FakeClient([]),
            openai_list_price_limit_usd="",
        )
    with pytest.raises(UserInputError, match="exactly zero"):
        _load_openai_responses_translation_model(
            model_snapshot=MODEL_SNAPSHOT,
            max_output_tokens=256,
            provider_journal_path=tmp_path / "provider.jsonl",
            client=_FakeClient([]),
            max_retries=1,
            openai_list_price_limit_usd="1.00",
        )


def test_actual_long_context_usage_fails_closed_after_journaling(tmp_path: Path) -> None:
    response = _response(raw_output=_assistant_payload("טקסט"))
    usage = _mapping(response["usage"])
    usage.update(input_tokens=272_001, output_tokens=1, total_tokens=272_002)
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=_FakeClient([response]),
    )

    with pytest.raises(UserInputError, match="base-rate input-token threshold"):
        model.translate_batch([_request()])

    assert _journal_records(journal)[0]["event"] == "response"
    with pytest.raises(UserInputError, match="base-rate input-token threshold"):
        aggregate_openai_responses_provider_journal(journal)


def test_provider_journal_aggregation_is_validated_and_content_free(
    tmp_path: Path,
) -> None:
    secret = "sk-summary-secret"
    refusal = "Sensitive refusal text that must not enter the summary"
    journal = tmp_path / "provider.jsonl"
    fake = _FakeClient(
        [
            _response(raw_output=_assistant_payload("תוצאה א")),
            _response(
                raw_output="{partial",
                status="incomplete",
                incomplete_reason="max_output_tokens",
            ),
            _response(refusal=refusal),
            RuntimeError(f"Authorization: Bearer {secret}"),
        ]
    )
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=fake,
    )
    model.translate_batch(
        [
            _request("query a", source_id="root-a"),
            _request("query b", source_id="root-b"),
            _request("query c", source_id="root-c"),
        ]
    )
    model.translate_batch([_request("query a", source_id="root-a")])
    with pytest.raises(ExternalDependencyError, match="request failed"):
        model.translate_batch([_request("query d", source_id="root-d")])

    summary = aggregate_openai_responses_provider_journal(journal)

    assert summary["schema_version"] == OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA
    assert summary["journal_schema_version"] == OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA
    assert summary["journal_sha256"] == hashlib.sha256(journal.read_bytes()).hexdigest()
    assert summary["requested_model"] == MODEL_SNAPSHOT
    assert summary["returned_models"] == [MODEL_SNAPSHOT]
    assert summary["requested_service_tier"] == "default"
    assert summary["requested_timeout_seconds"] == OPENAI_RESPONSES_TIMEOUT_SECONDS
    assert summary["returned_service_tiers"] == ["default"]
    assert summary["safety_identifier"] == OPENAI_RESPONSES_SAFETY_IDENTIFIER
    assert summary["unique_requests"] == 4
    assert summary["unique_source_attempts"] == 4
    assert summary["usage_complete"] is False
    assert summary["counts"] == {
        "records": 5,
        "responses": 3,
        "replayable_responses": 3,
        "replays": 1,
        "durable_journal_replays": 1,
        "batch_coalesced_replays": 0,
        "request_errors": 1,
        "resource_unavailable_events": 0,
        "resolved_resource_unavailable_events": 0,
        "pending_resource_unavailable_events": 0,
        "unresolved_resource_unavailable_events": 0,
        "provider_error_responses": 0,
        "error_records": 1,
        "model_mismatch_responses": 0,
        "service_tier_mismatch_responses": 0,
        "refusal_responses": 1,
        "incomplete_responses": 1,
        "responses_missing_usage": 0,
    }
    assert summary["usage"] == {
        "input_tokens": 303,
        "cached_input_tokens": 69,
        "output_tokens": 51,
        "reasoning_output_tokens": 0,
        "total_tokens": 354,
    }
    assert summary["max_response_input_tokens"] == 101
    rendered = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    assert "תוצאה א" not in rendered
    assert "{partial" not in rendered
    assert refusal not in rendered
    assert secret not in rendered


def test_provider_journal_aggregation_rejects_mixed_requested_tiers(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "provider.jsonl"
    default_model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        service_tier="default",
        client=_FakeClient([_response(raw_output=_assistant_payload("א"))]),
    )
    default_model.translate_batch([_request("query a")])
    with pytest.raises(UserInputError, match="complete list-price estimate"):
        load_openai_responses_translation_model(
            model_snapshot=MODEL_SNAPSHOT,
            max_output_tokens=256,
            provider_journal_path=journal,
            service_tier="flex",
            client=_FakeClient(
                [_response(raw_output=_assistant_payload("ב"), service_tier="flex")]
            ),
        )


def test_provider_journal_aggregation_rejects_mixed_requested_models(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "provider.jsonl"
    first_model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=_FakeClient([_response(raw_output=_assistant_payload("א"))]),
    )
    first_model.translate_batch([_request("query a")])
    other_snapshot = "gpt-5.5-2026-05-01"
    with pytest.raises(UserInputError, match="complete list-price estimate"):
        load_openai_responses_translation_model(
            model_snapshot=other_snapshot,
            max_output_tokens=256,
            provider_journal_path=journal,
            client=_FakeClient(
                [
                    _response(
                        raw_output=_assistant_payload("ב"),
                        model=other_snapshot,
                    )
                ]
            ),
        )


def test_provider_journal_aggregation_rejects_mixed_requested_timeouts(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "provider.jsonl"
    first_model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        timeout_seconds=OPENAI_RESPONSES_TIMEOUT_SECONDS,
        client=_FakeClient([_response(raw_output=_assistant_payload("א"))]),
    )
    first_model.translate_batch([_request("query a", source_id="root-1")])
    with pytest.raises(UserInputError, match="complete list-price estimate"):
        load_openai_responses_translation_model(
            model_snapshot=MODEL_SNAPSHOT,
            max_output_tokens=256,
            provider_journal_path=journal,
            timeout_seconds=60.0,
            client=_FakeClient([_response(raw_output=_assistant_payload("ב"))]),
        )


def test_provider_journal_aggregation_rejects_tampered_output_hash(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=_FakeClient([_response(raw_output=_assistant_payload("א"))]),
    )
    model.translate_batch([_request()])
    journal.write_text(
        journal.read_text(encoding="utf-8").replace(
            '"raw_output_sha256":"', '"raw_output_sha256":"0'
        ),
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="raw-output hash"):
        aggregate_openai_responses_provider_journal(journal)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("total_tokens", 119, "inconsistent total-token usage"),
        ("cached_tokens", 102, "cached input exceeding"),
        ("reasoning_tokens", 18, "reasoning output exceeding"),
    ],
)
def test_provider_journal_aggregation_rejects_per_response_usage_drift(
    tmp_path: Path,
    field: str,
    replacement: int,
    message: str,
) -> None:
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=_FakeClient([_response(raw_output=_assistant_payload("א"))]),
    )
    model.translate_batch([_request()])
    record = _journal_records(journal)[0]
    usage = _mapping(record["usage"])
    if field == "cached_tokens":
        _mapping(usage["input_tokens_details"])[field] = replacement
    elif field == "reasoning_tokens":
        _mapping(usage["output_tokens_details"])[field] = replacement
    else:
        usage[field] = replacement
    journal.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match=message):
        aggregate_openai_responses_provider_journal(journal)


def test_provider_journal_aggregation_rejects_tampered_attribution(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=_FakeClient([_response(raw_output=_assistant_payload("א"))]),
    )
    model.translate_batch([_request()])
    journal.write_text(
        journal.read_text(encoding="utf-8").replace('"attempt":1', '"attempt":0'),
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="source-attempt attribution"):
        aggregate_openai_responses_provider_journal(journal)


def test_provider_journal_rejects_multiple_requests_for_one_source_attempt(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=_FakeClient(
            [
                _response(raw_output=_assistant_payload("א")),
                _response(raw_output=_assistant_payload("ב")),
            ]
        ),
    )
    model.translate_batch([_request("query a", source_id="root-1", attempt=1)])
    with pytest.raises(UserInputError, match="multiple requests"):
        model.translate_batch([_request("query b", source_id="root-1", attempt=1)])


def test_provider_journal_rejects_noncontiguous_source_attempts(tmp_path: Path) -> None:
    journal = tmp_path / "provider.jsonl"
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=journal,
        client=_FakeClient(
            [
                _response(raw_output=_assistant_payload("א")),
                _response(raw_output=_assistant_payload("ב")),
            ]
        ),
    )
    model.translate_batch([_request("query a", source_id="root-1", attempt=1)])
    with pytest.raises(UserInputError, match="non-contiguous attempts"):
        model.translate_batch([_request("query b", source_id="root-1", attempt=3)])


@pytest.mark.parametrize(
    "candidate",
    [
        TranslationRequest(
            query="Show the weather",
            protected_spans=(),
            target_language="he",
            semantic_context="{}",
            attempt=1,
        ),
        TranslationRequest(
            query="Show the weather",
            protected_spans=(),
            target_language="he",
            semantic_context="{}",
            source_id="root-1",
            attempt=0,
        ),
    ],
)
def test_provider_rejects_missing_or_invalid_local_attribution(
    tmp_path: Path,
    candidate: TranslationRequest,
) -> None:
    fake = _FakeClient([])
    model = load_openai_responses_translation_model(
        model_snapshot=MODEL_SNAPSHOT,
        max_output_tokens=256,
        provider_journal_path=tmp_path / "provider.jsonl",
        client=fake,
    )

    with pytest.raises(UserInputError, match="attribution"):
        model.translate_batch([candidate])

    assert fake.responses.calls == []


@pytest.mark.parametrize(
    ("model_snapshot", "service_tier"),
    [
        ("gpt-5.5", "default"),
        (MODEL_SNAPSHOT, "auto"),
    ],
)
def test_rejects_moving_model_alias_and_auto_tier(
    tmp_path: Path,
    model_snapshot: str,
    service_tier: str,
) -> None:
    with pytest.raises(UserInputError):
        load_openai_responses_translation_model(
            model_snapshot=model_snapshot,
            max_output_tokens=256,
            provider_journal_path=tmp_path / "provider.jsonl",
            service_tier=service_tier,  # type: ignore[arg-type]
            client=_FakeClient([]),
        )
