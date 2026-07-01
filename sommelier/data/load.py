from __future__ import annotations

from pathlib import Path

from sommelier.artifacts import read_jsonl_with_schema
from sommelier.data.types import RawToolCallRow
from sommelier.errors import SchemaValidationError, UserInputError


def load_raw_rows(path: Path) -> list[RawToolCallRow]:
    if not path.exists():
        raise UserInputError(
            f"raw dataset file not found: {path}",
            hint="Pass --input with a JSONL file of sommelier.raw_tool_call_row.v1 records.",
        )
    records = read_jsonl_with_schema(
        path,
        expected_schema="sommelier.raw_tool_call_row.v1",
    )
    rows: list[RawToolCallRow] = []
    for index, record in enumerate(records, start=1):
        try:
            rows.append(
                RawToolCallRow(
                    schema_version="sommelier.raw_tool_call_row.v1",
                    source_id=str(record["source_id"]),
                    query=str(record["query"]),
                    tools=str(record["tools"]),
                    answers=str(record["answers"]),
                    source_revision=str(record["source_revision"]),
                )
            )
        except KeyError as error:
            raise SchemaValidationError(
                f"{path}:{index} missing required field {error}",
                hint="Each raw row needs source_id, query, tools, answers, and source_revision.",
            ) from error
    return rows
