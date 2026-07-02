import json
from pathlib import Path

import pytest

from sommelier.data.types import RawToolCallRow
from sommelier.data.validate import parse_gold_calls, parse_tools, validate_raw_row

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _raw_row(**overrides: str) -> RawToolCallRow:
    base = RawToolCallRow(
        schema_version="sommelier.raw_tool_call_row.v1",
        source_id="row-1",
        query="What is the weather in Paris today?",
        tools=(
            '[{"name":"lookup_weather","description":"Look up weather.",'
            '"parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]'
        ),
        answers='[{"name":"lookup_weather","arguments":{"city":"Paris"}}]',
        source_revision="fixture",
    )
    if not overrides:
        return base
    return RawToolCallRow(
        schema_version=base["schema_version"],
        source_id=overrides.get("source_id", base["source_id"]),
        query=overrides.get("query", base["query"]),
        tools=overrides.get("tools", base["tools"]),
        answers=overrides.get("answers", base["answers"]),
        source_revision=overrides.get("source_revision", base["source_revision"]),
    )


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"query": ""}, "missing_query"),
        ({"query": "   "}, "missing_query"),
        ({"tools": ""}, "missing_tools"),
        ({"answers": ""}, "missing_answers"),
        ({"query": "short"}, "query_too_short"),
        ({"query": "x" * 2001}, "query_too_long"),
        ({"tools": "not-json"}, "invalid_tools_json"),
        ({"answers": "not-json"}, "invalid_answers_json"),
        ({"tools": '{"name":"lookup_weather"}'}, "invalid_tool_shape"),
        ({"answers": "[]"}, "invalid_answer_shape"),
        ({"answers": '[{"arguments":{"city":"Paris"}}]'}, "invalid_answer_shape"),
    ],
)
def test_validate_raw_row_drop_reasons(
    overrides: dict[str, str],
    expected: str,
) -> None:
    result = validate_raw_row(_raw_row(**overrides), min_query_chars=10, max_query_chars=2000)
    assert result == expected


def test_validate_raw_row_accepts_valid_row() -> None:
    result = validate_raw_row(_raw_row(), min_query_chars=10, max_query_chars=2000)
    assert isinstance(result, dict)
    assert result["schema_version"] == "sommelier.prepared_example.v1"
    assert result["gold_calls"][0]["name"] == "lookup_weather"


def test_parse_tools_rejects_non_list() -> None:
    assert parse_tools('{"name":"lookup_weather"}') == "invalid_tool_shape"


def test_parse_gold_calls_requires_non_empty_list() -> None:
    assert parse_gold_calls("[]") == "invalid_answer_shape"


def test_fixture_rows_are_valid() -> None:
    path = FIXTURES_DIR / "preparation_rows.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        row = RawToolCallRow(
            schema_version="sommelier.raw_tool_call_row.v1",
            source_id=str(payload["source_id"]),
            query=str(payload["query"]),
            tools=str(payload["tools"]),
            answers=str(payload["answers"]),
            source_revision=str(payload["source_revision"]),
        )
        result = validate_raw_row(row, min_query_chars=10, max_query_chars=2000)
        assert isinstance(result, dict)
