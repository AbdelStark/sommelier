from __future__ import annotations

from pathlib import Path

from sommelier.artifacts import read_jsonl_with_schema
from sommelier.data.types import RawToolCallRow
from sommelier.errors import SchemaValidationError, UserInputError


def load_raw_rows(
    path: Path,
    *,
    require_source_example_id: bool = False,
) -> list[RawToolCallRow]:
    """Loads raw rows; paired-source files must name their root example on
    every row (``require_source_example_id=True``)."""
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
            row = RawToolCallRow(
                schema_version="sommelier.raw_tool_call_row.v1",
                source_id=str(record["source_id"]),
                query=str(record["query"]),
                tools=str(record["tools"]),
                answers=str(record["answers"]),
                source_revision=str(record["source_revision"]),
            )
        except KeyError as error:
            raise SchemaValidationError(
                f"{path}:{index} missing required field {error}",
                hint="Each raw row needs source_id, query, tools, answers, and source_revision.",
            ) from error
        source_example_id = record.get("source_example_id")
        if source_example_id is not None:
            row["source_example_id"] = str(source_example_id)
        elif require_source_example_id:
            raise SchemaValidationError(
                f"{path}:{index} missing source_example_id",
                hint="Paired-source rows must name the root example they translate.",
            )
        rows.append(row)
    return rows
