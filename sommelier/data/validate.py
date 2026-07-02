from __future__ import annotations

import json
from typing import cast

from sommelier.data.normalize import query_digest
from sommelier.data.types import (
    DropReason,
    JsonObject,
    PreparedExample,
    RawToolCallRow,
    ToolCall,
    ToolSchema,
)


def parse_tools(raw: str) -> list[ToolSchema] | DropReason:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "invalid_tools_json"
    if not isinstance(payload, list):
        return "invalid_tool_shape"
    tools: list[ToolSchema] = []
    for item in payload:
        if not isinstance(item, dict):
            return "invalid_tool_shape"
        name = item.get("name")
        description = item.get("description")
        parameters = item.get("parameters")
        if not isinstance(name, str) or not name:
            return "invalid_tool_shape"
        if not isinstance(description, str):
            return "invalid_tool_shape"
        if not isinstance(parameters, dict):
            return "invalid_tool_shape"
        tools.append(
            ToolSchema(
                name=name,
                description=description,
                parameters=cast(JsonObject, parameters),
            )
        )
    return tools


def parse_gold_calls(raw: str) -> list[ToolCall] | DropReason:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "invalid_answers_json"
    if not isinstance(payload, list) or not payload:
        return "invalid_answer_shape"
    calls: list[ToolCall] = []
    for item in payload:
        if not isinstance(item, dict):
            return "invalid_answer_shape"
        name = item.get("name")
        arguments = item.get("arguments")
        if not isinstance(name, str) or not name:
            return "invalid_answer_shape"
        if not isinstance(arguments, dict):
            return "invalid_answer_shape"
        calls.append(
            ToolCall(
                name=name,
                arguments=cast(JsonObject, arguments),
            )
        )
    if len(calls) != 1:
        # v1 trains and scores exactly one tool call per example: the
        # parser rejects multi-call outputs, so keeping multi-call golds
        # would score a faithful model as a failure. Declared filter with
        # its own drop reason per the data policy.
        return "multi_call_answer"
    return calls


def validate_raw_row(
    row: RawToolCallRow,
    *,
    min_query_chars: int,
    max_query_chars: int,
) -> PreparedExample | DropReason:
    query = row.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return "missing_query"

    tools_raw = row.get("tools", "")
    if not isinstance(tools_raw, str) or not tools_raw.strip():
        return "missing_tools"

    answers_raw = row.get("answers", "")
    if not isinstance(answers_raw, str) or not answers_raw.strip():
        return "missing_answers"

    if len(query.strip()) < min_query_chars:
        return "query_too_short"
    if len(query.strip()) > max_query_chars:
        return "query_too_long"

    tools_result = parse_tools(tools_raw)
    if isinstance(tools_result, str):
        return tools_result

    calls_result = parse_gold_calls(answers_raw)
    if isinstance(calls_result, str):
        return calls_result

    return PreparedExample(
        schema_version="sommelier.prepared_example.v1",
        example_id=row["source_id"],
        source_id=row["source_id"],
        query=query.strip(),
        tools=tools_result,
        gold_calls=calls_result,
        split="train",
        query_sha256=query_digest(query),
        source_revision=row["source_revision"],
    )
