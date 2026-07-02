from __future__ import annotations

from typing import Any

import pytest

from sommelier.errors import SchemaValidationError
from sommelier.serving.schemas import (
    ServeRequest,
    build_serve_response,
    validate_serve_request,
)


def valid_payload() -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "You are a tool-calling model."},
            {"role": "user", "content": "What is the weather in Paris today?"},
        ],
        "tools": [
            {
                "name": "lookup_weather",
                "description": "Look up weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }


def test_valid_request_passes() -> None:
    request: ServeRequest = validate_serve_request(valid_payload())
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 256
    assert request["messages"][1]["content"].startswith("What is the weather")
    assert request["tools"][0]["name"] == "lookup_weather"


def mutate(**changes: Any) -> dict[str, Any]:
    payload = valid_payload()
    payload.update(changes)
    return payload


@pytest.mark.parametrize(
    "payload",
    [
        "not an object",
        {},
        mutate(extra="field"),
        {key: value for key, value in valid_payload().items() if key != "tools"},
        mutate(messages=[]),
        mutate(messages=[{"role": "user"}]),
        mutate(messages=[{"role": "tool", "content": "x"}]),
        mutate(messages=[{"role": "user", "content": 42}]),
        mutate(messages=[{"role": "user", "content": "x", "name": "n"}]),
        mutate(tools=[]),
        mutate(tools=["not an object"]),
        mutate(tools=[{"name": "", "description": "d", "parameters": {}}]),
        mutate(tools=[{"name": "f", "description": "d", "parameters": "not object"}]),
        mutate(tools=[{"name": "f", "parameters": {}}]),
        mutate(temperature=0.7),
        mutate(temperature="0.0"),
        mutate(temperature=True),
        mutate(max_tokens=0),
        mutate(max_tokens=-5),
        mutate(max_tokens=2.5),
        mutate(max_tokens=True),
    ],
    ids=[
        "non_object",
        "empty",
        "extra_field",
        "missing_tools",
        "empty_messages",
        "message_missing_content",
        "bad_role",
        "non_string_content",
        "message_extra_key",
        "empty_tools",
        "tool_not_object",
        "tool_empty_name",
        "tool_bad_parameters",
        "tool_missing_description",
        "nonzero_temperature",
        "string_temperature",
        "bool_temperature",
        "zero_max_tokens",
        "negative_max_tokens",
        "float_max_tokens",
        "bool_max_tokens",
    ],
)
def test_invalid_requests_fail_closed(payload: object) -> None:
    with pytest.raises(SchemaValidationError):
        validate_serve_request(payload)


def test_response_reuses_parser_ok() -> None:
    response = build_serve_response('{"arguments":{"city":"Paris"},"name":"lookup_weather"}')
    assert response["parse_status"] == "ok"
    assert response["parsed_call"] is not None
    assert response["parsed_call"]["name"] == "lookup_weather"
    assert response["model_kind"] == "adapter"


@pytest.mark.parametrize(
    ("raw_text", "status"),
    [
        ("no tools for that", "no_json"),
        ('{"name":"f","arguments":{', "invalid_json"),
        ('{"name":"f"}', "invalid_shape"),
    ],
)
def test_response_reports_parse_failures(raw_text: str, status: str) -> None:
    response = build_serve_response(raw_text)
    assert response["parse_status"] == status
    assert response["parsed_call"] is None
    assert response["raw_text"] == raw_text


def test_integer_temperature_zero_is_accepted() -> None:
    request = validate_serve_request(mutate(temperature=0))
    assert request["temperature"] == 0.0
