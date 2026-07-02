from __future__ import annotations

from typing import Literal, TypedDict, cast

from sommelier.data.types import JsonObject, ToolCall, ToolSchema
from sommelier.errors import SchemaValidationError
from sommelier.evaluation.parse import ParseStatus, parse_tool_call
from sommelier.formatting.chat import ChatMessage

REQUEST_FIELDS = frozenset({"messages", "tools", "temperature", "max_tokens"})

ALLOWED_ROLES = ("system", "user", "assistant")


class ServeRequest(TypedDict):
    """RFC-0010 request shape: chat messages plus tool schemas.

    Serving is deterministic like evaluation: temperature must be exactly
    0.0 and max_tokens positive. Unknown fields are rejected so silent
    client drift cannot change behavior.
    """

    messages: list[ChatMessage]
    tools: list[ToolSchema]
    temperature: float
    max_tokens: int


class ServeResponse(TypedDict):
    raw_text: str
    parsed_call: ToolCall | None
    parse_status: ParseStatus
    model_kind: Literal["adapter"]


def _fail(message: str) -> SchemaValidationError:
    return SchemaValidationError(
        f"invalid serve request: {message}",
        hint="Match the RFC-0010 request shape: messages, tools, "
        "temperature (0.0), and max_tokens.",
    )


def _validate_messages(value: object) -> list[ChatMessage]:
    if not isinstance(value, list) or not value:
        raise _fail("messages must be a non-empty list")
    messages: list[ChatMessage] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise _fail(f"messages[{index}] must be an object")
        if set(item.keys()) != {"role", "content"}:
            raise _fail(f"messages[{index}] must have exactly role and content")
        role = item["role"]
        content = item["content"]
        if role not in ALLOWED_ROLES:
            raise _fail(f"messages[{index}].role must be one of {ALLOWED_ROLES}")
        if not isinstance(content, str):
            raise _fail(f"messages[{index}].content must be a string")
        messages.append(cast(ChatMessage, {"role": role, "content": content}))
    return messages


def _validate_tools(value: object) -> list[ToolSchema]:
    if not isinstance(value, list) or not value:
        raise _fail("tools must be a non-empty list")
    tools: list[ToolSchema] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise _fail(f"tools[{index}] must be an object")
        name = item.get("name")
        description = item.get("description")
        parameters = item.get("parameters")
        if not isinstance(name, str) or not name:
            raise _fail(f"tools[{index}].name must be a non-empty string")
        if not isinstance(description, str):
            raise _fail(f"tools[{index}].description must be a string")
        if not isinstance(parameters, dict):
            raise _fail(f"tools[{index}].parameters must be an object")
        tools.append(
            ToolSchema(
                name=name,
                description=description,
                parameters=cast(JsonObject, parameters),
            )
        )
    return tools


def validate_serve_request(payload: object) -> ServeRequest:
    """Validates a request payload into a ServeRequest, or fails closed."""
    if not isinstance(payload, dict):
        raise _fail("payload must be a JSON object")
    unknown = set(payload.keys()) - REQUEST_FIELDS
    if unknown:
        raise _fail(f"unknown fields: {sorted(unknown)}")
    missing = REQUEST_FIELDS - set(payload.keys())
    if missing:
        raise _fail(f"missing fields: {sorted(missing)}")

    messages = _validate_messages(payload["messages"])
    tools = _validate_tools(payload["tools"])

    temperature = payload["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise _fail("temperature must be a number")
    if float(temperature) != 0.0:
        raise _fail("temperature must be 0.0; serving decodes deterministically")

    max_tokens = payload["max_tokens"]
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise _fail("max_tokens must be an integer")
    if max_tokens <= 0:
        raise _fail("max_tokens must be positive")

    return ServeRequest(
        messages=messages,
        tools=tools,
        temperature=0.0,
        max_tokens=max_tokens,
    )


def build_serve_response(raw_text: str) -> ServeResponse:
    """Builds the response for generated text, reusing the RFC-0005 parser.

    Parse failures are reported in parse_status, never repaired; the raw
    text is always returned for inspection.
    """
    parsed_call, parse_status = parse_tool_call(raw_text)
    return ServeResponse(
        raw_text=raw_text,
        parsed_call=parsed_call,
        parse_status=parse_status,
        model_kind="adapter",
    )
