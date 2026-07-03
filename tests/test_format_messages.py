from __future__ import annotations

import json

import pytest

from sommelier.errors import SchemaValidationError
from sommelier.formatting.chat import (
    ChatMessage,
    build_messages,
    canonical_json,
    validate_assistant_target,
    validate_gold_calls,
)

SYSTEM_PROMPT = (
    "You are a tool-calling model. Select the correct tool and return only "
    "the JSON tool call. Do not include explanations."
)

TOOLS = [
    {
        "name": "lookup_weather",
        "description": "Look up weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]

GOLD_CALLS = [{"name": "lookup_weather", "arguments": {"city": "Paris"}}]


def build() -> list[ChatMessage]:
    return build_messages(
        query="What is the weather in Paris today?",
        tools=list(TOOLS),
        gold_calls=GOLD_CALLS,
        system_prompt=SYSTEM_PROMPT,
    )


def test_messages_have_canonical_roles_and_content() -> None:
    messages = build()

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"].startswith(SYSTEM_PROMPT)
    assert "\n\nAvailable tools:\n" in messages[0]["content"]
    assert messages[1]["content"] == "What is the weather in Paris today?"
    assert messages[2]["content"] == canonical_json(GOLD_CALLS)


def test_canonical_json_sorts_keys_and_uses_compact_separators() -> None:
    payload = {"b": 1, "a": {"z": True, "m": [1, 2]}}
    assert canonical_json(payload) == '{"a":{"m":[1,2],"z":true},"b":1}'


def test_canonical_json_is_input_order_independent() -> None:
    first = {"name": "f", "arguments": {"x": 1, "y": 2}}
    second = {"arguments": {"y": 2, "x": 1}, "name": "f"}
    assert canonical_json(first) == canonical_json(second)


def test_assistant_target_is_pure_json_tool_call() -> None:
    messages = build()
    parsed = json.loads(messages[2]["content"])
    assert isinstance(parsed, list)
    assert parsed[0]["name"] == "lookup_weather"
    assert parsed[0]["arguments"] == {"city": "Paris"}


def test_validate_assistant_target_accepts_canonical_call() -> None:
    target = canonical_json(GOLD_CALLS)
    calls = validate_assistant_target(target)
    assert calls[0]["name"] == "lookup_weather"


@pytest.mark.parametrize(
    "target",
    [
        "Sure! " + canonical_json(GOLD_CALLS),
        canonical_json(GOLD_CALLS) + " Hope this helps.",
        "```json\n" + canonical_json(GOLD_CALLS) + "\n```",
        "not json at all",
    ],
)
def test_validate_assistant_target_rejects_explanatory_text(target: str) -> None:
    with pytest.raises(SchemaValidationError):
        validate_assistant_target(target)


def test_validate_assistant_target_rejects_non_canonical_ordering() -> None:
    target = json.dumps([{"name": "f", "arguments": {"b": 1, "a": 2}}])
    with pytest.raises(SchemaValidationError):
        validate_assistant_target(target)


@pytest.mark.parametrize(
    "gold_calls",
    [
        [],
        "not a list",
        [{"arguments": {}}],
        [{"name": "", "arguments": {}}],
        [{"name": "f"}],
        [{"name": "f", "arguments": "not an object"}],
        ["not an object"],
    ],
)
def test_validate_gold_calls_rejects_bad_shapes(gold_calls: object) -> None:
    with pytest.raises(SchemaValidationError):
        validate_gold_calls(gold_calls, context="test")


def test_build_messages_rejects_empty_gold_calls() -> None:
    with pytest.raises(SchemaValidationError):
        build_messages(
            query="q",
            tools=list(TOOLS),
            gold_calls=[],
            system_prompt=SYSTEM_PROMPT,
        )
