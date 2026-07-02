from __future__ import annotations

import json
from typing import Final, Literal

from sommelier.data.types import ToolCall

PARSER_VERSION: Final = "sommelier.parser.v1"

ParseStatus = Literal["ok", "no_json", "invalid_json", "invalid_shape"]

_OPENERS = {"{": "}", "[": "]"}


def _extract_first_balanced_span(text: str) -> str | None:
    """Returns the first balanced JSON object or array substring, if any.

    Walks the text from the first ``{`` or ``[``, tracking bracket depth
    while skipping double-quoted strings and escape sequences. Returns None
    when no opening bracket exists or the brackets never balance.
    """
    start = -1
    for index, char in enumerate(text):
        if char in _OPENERS:
            start = index
            break
    if start == -1:
        return None

    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _OPENERS:
            stack.append(_OPENERS[char])
        elif char in ("}", "]"):
            if not stack or char != stack.pop():
                return None
            if not stack:
                return text[start : index + 1]
    return None


def _validate_call_shape(payload: object) -> ToolCall | None:
    """Accepts exactly ``{"name": str, "arguments": object}``; no repair."""
    if not isinstance(payload, dict):
        return None
    if set(payload.keys()) != {"name", "arguments"}:
        return None
    name = payload["name"]
    arguments = payload["arguments"]
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(arguments, dict):
        return None
    return ToolCall(name=name, arguments=arguments)


def parse_tool_call(text: str) -> tuple[ToolCall | None, ParseStatus]:
    """Conservatively parses one tool call from generated text (RFC-0005).

    Extraction takes the first balanced JSON object or array in the text;
    surrounding prose is ignored, but nothing inside the span is repaired.

    Statuses:

    - ``no_json``: the text contains no opening ``{`` or ``[``.
    - ``invalid_json``: an opening bracket exists but no balanced span can
      be extracted, or the span is not valid JSON.
    - ``invalid_shape``: the JSON parses but is not a single
      ``{"name": str, "arguments": object}`` call, or a one-element array
      containing exactly such an object. Extra keys, empty names,
      non-object arguments, empty arrays, and multi-call arrays all fail.
    - ``ok``: the call is returned as a ToolCall.

    Invalid statuses return ``(None, status)``; failures count against
    metrics rather than aborting evaluation (INV-DATA-005).
    """
    span = _extract_first_balanced_span(text)
    if span is None:
        has_bracket = any(char in _OPENERS for char in text)
        return None, "invalid_json" if has_bracket else "no_json"

    try:
        payload = json.loads(span)
    except json.JSONDecodeError:
        return None, "invalid_json"

    if isinstance(payload, list):
        if len(payload) != 1:
            return None, "invalid_shape"
        payload = payload[0]

    call = _validate_call_shape(payload)
    if call is None:
        return None, "invalid_shape"
    return call, "ok"
