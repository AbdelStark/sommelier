from __future__ import annotations

import pytest

from sommelier.evaluation.parse import PARSER_VERSION, parse_tool_call

VALID_CALL = '{"arguments":{"city":"Paris"},"name":"lookup_weather"}'


def test_parser_version_constant() -> None:
    assert PARSER_VERSION == "sommelier.parser.v1"


@pytest.mark.parametrize(
    "text",
    [
        VALID_CALL,
        f"[{VALID_CALL}]",
        f"Sure, calling the tool now: {VALID_CALL}",
        f"{VALID_CALL} Let me know if you need anything else.",
        f"```json\n{VALID_CALL}\n```",
        '{"name": "lookup_weather", "arguments": {"city": "Paris"}}',
    ],
    ids=["object", "array", "prefix_prose", "suffix_prose", "fenced", "key_order"],
)
def test_ok_extraction(text: str) -> None:
    call, status = parse_tool_call(text)
    assert status == "ok"
    assert call is not None
    assert call["name"] == "lookup_weather"
    assert call["arguments"] == {"city": "Paris"}


def test_ok_with_nested_arguments() -> None:
    text = '{"name":"f","arguments":{"a":{"b":[1,2,{"c":"d"}]},"e":null}}'
    call, status = parse_tool_call(text)
    assert status == "ok"
    assert call is not None
    assert call["arguments"] == {"a": {"b": [1, 2, {"c": "d"}]}, "e": None}


def test_ok_with_braces_inside_strings() -> None:
    text = '{"name":"f","arguments":{"s":"closing } and ] inside"}}'
    call, status = parse_tool_call(text)
    assert status == "ok"
    assert call is not None


def test_first_balanced_span_wins() -> None:
    text = f'{{"name":"first","arguments":{{}}}} then {VALID_CALL}'
    call, status = parse_tool_call(text)
    assert status == "ok"
    assert call is not None
    assert call["name"] == "first"


@pytest.mark.parametrize(
    "text",
    ["", "I cannot call any tool for that request.", "name: lookup_weather"],
    ids=["empty", "prose", "yaml_like"],
)
def test_no_json(text: str) -> None:
    call, status = parse_tool_call(text)
    assert call is None
    assert status == "no_json"


@pytest.mark.parametrize(
    "text",
    [
        '{"name":"f","arguments":{',
        '[{"name":"f"]',
        "{'name': 'f', 'arguments': {}}",
        '{"name": f, "arguments": {}}',
        '{"name":"f",}',
    ],
    ids=["unclosed", "mismatched", "single_quotes", "bare_token", "trailing_comma"],
)
def test_invalid_json(text: str) -> None:
    call, status = parse_tool_call(text)
    assert call is None
    assert status == "invalid_json"


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        f"[{VALID_CALL},{VALID_CALL}]",
        '{"name":"f"}',
        '{"arguments":{}}',
        '{"name":"","arguments":{}}',
        '{"name":42,"arguments":{}}',
        '{"name":"f","arguments":[]}',
        '{"name":"f","arguments":"x"}',
        '{"name":"f","arguments":{},"id":1}',
        '["not a call"]',
        '{"tool":"f","args":{}}',
    ],
    ids=[
        "empty_array",
        "two_calls",
        "missing_arguments",
        "missing_name",
        "empty_name",
        "non_string_name",
        "array_arguments",
        "string_arguments",
        "extra_key",
        "array_of_scalar",
        "wrong_keys",
    ],
)
def test_invalid_shape(text: str) -> None:
    call, status = parse_tool_call(text)
    assert call is None
    assert status == "invalid_shape"


def test_no_repair_of_invalid_json() -> None:
    # A repairing parser would fix the trailing comma; ours must not.
    call, status = parse_tool_call('{"name":"f","arguments":{"a":1,}}')
    assert call is None
    assert status == "invalid_json"


def test_deterministic_across_calls() -> None:
    text = f"noise {VALID_CALL} noise"
    assert parse_tool_call(text) == parse_tool_call(text)
