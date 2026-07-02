from __future__ import annotations

import json
from pathlib import Path

import pytest

from sommelier.data.split import prepare_split_result
from sommelier.data.types import RawToolCallRow


def _raw_row(source_id: str, query: str) -> RawToolCallRow:
    tools = (
        '[{"name":"lookup_weather","description":"Look up weather.",'
        '"parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]'
    )
    answers = '[{"name":"lookup_weather","arguments":{"city":"Paris"}}]'
    return RawToolCallRow(
        schema_version="sommelier.raw_tool_call_row.v1",
        source_id=source_id,
        query=query,
        tools=tools,
        answers=answers,
        source_revision="fixture",
    )


@pytest.mark.parametrize("seed", [1, 2, 7, 42])
def test_split_disjointness_property(seed: int) -> None:
    rows = [
        _raw_row(f"row-{index}", f"Property weather query {seed}-{index}")
        for index in range(20)
    ]
    result = prepare_split_result(
        rows,
        min_query_chars=10,
        max_query_chars=2000,
        n_train=8,
        n_validation=4,
        n_test=4,
        seed=seed,
    )
    hashes = [
        *[example["query_sha256"] for example in result.train],
        *[example["query_sha256"] for example in result.validation],
        *[example["query_sha256"] for example in result.test],
    ]
    assert len(hashes) == len(set(hashes))


def test_preparation_fixture_file_has_valid_and_unique_rows() -> None:
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
    path = fixtures_dir / "preparation_rows.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    assert len(rows) >= 10
    queries = [row["query"] for row in rows]
    assert len(queries) == len(set(queries))
