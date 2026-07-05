from __future__ import annotations

from typing import Any, Final, Literal, NotRequired, TypedDict

PREPARED_EXAMPLE_SCHEMA: Final = "sommelier.prepared_example.v2"
DROP_SUMMARY_SCHEMA: Final = "sommelier.drop_summary.v2"

DropReason = Literal[
    "missing_query",
    "missing_tools",
    "missing_answers",
    "query_too_short",
    "query_too_long",
    "invalid_tools_json",
    "invalid_answers_json",
    "invalid_tool_shape",
    "invalid_answer_shape",
    "multi_call_answer",
    "duplicate_query",
    "missing_source_example",
    "duplicate_source_example",
    "pair_tools_mismatch",
    "pair_answers_mismatch",
    "cross_split_duplicate",
]

JsonObject = dict[str, Any]


class RawToolCallRow(TypedDict):
    schema_version: Literal["sommelier.raw_tool_call_row.v1"]
    source_id: str
    query: str
    tools: str
    answers: str
    source_revision: str
    # Paired sources only: the example_id of the root row this row is a
    # translation of. Absent on root source rows.
    source_example_id: NotRequired[str]


class ToolSchema(TypedDict):
    name: str
    description: str
    parameters: JsonObject


class ToolCall(TypedDict):
    name: str
    arguments: JsonObject


class PreparedExample(TypedDict):
    schema_version: Literal["sommelier.prepared_example.v2"]
    example_id: str
    source_id: str
    language: str
    source_example_id: str | None
    query: str
    tools: list[ToolSchema]
    gold_calls: list[ToolCall]
    split: Literal["train", "validation", "test"]
    query_sha256: str
    source_revision: str


SplitName = Literal["train", "validation", "test"]
