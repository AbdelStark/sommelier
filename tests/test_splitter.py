import json
from pathlib import Path

import pytest

from sommelier.config import load_config
from sommelier.data.prepare import prepare_dataset_from_file
from sommelier.data.split import prepare_split_result, split_examples
from sommelier.data.types import PreparedExample, RawToolCallRow
from sommelier.data.validate import validate_raw_row
from sommelier.errors import UserInputError
from sommelier.run_context import ensure_run_context

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


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


def _prepared(source_id: str, query: str) -> PreparedExample:
    result = validate_raw_row(_raw_row(source_id, query), min_query_chars=10, max_query_chars=2000)
    assert isinstance(result, dict)
    return result


def test_split_is_deterministic_for_seed() -> None:
    rows = [
        _raw_row(f"row-{index}", f"Query number {index} about weather in Paris")
        for index in range(10)
    ]
    first = prepare_split_result(
        rows,
        min_query_chars=10,
        max_query_chars=2000,
        n_train=4,
        n_validation=2,
        n_test=2,
        seed=42,
    )
    second = prepare_split_result(
        rows,
        min_query_chars=10,
        max_query_chars=2000,
        n_train=4,
        n_validation=2,
        n_test=2,
        seed=42,
    )
    assert [example["example_id"] for example in first.train] == [
        example["example_id"] for example in second.train
    ]


def test_query_cannot_appear_in_multiple_splits() -> None:
    rows = [
        _raw_row(f"row-{index}", f"Unique weather query {index} in Paris")
        for index in range(12)
    ]
    result = prepare_split_result(
        rows,
        min_query_chars=10,
        max_query_chars=2000,
        n_train=5,
        n_validation=3,
        n_test=2,
        seed=7,
    )
    all_hashes = [
        *[example["query_sha256"] for example in result.train],
        *[example["query_sha256"] for example in result.validation],
        *[example["query_sha256"] for example in result.test],
    ]
    assert len(all_hashes) == len(set(all_hashes))


def test_duplicate_queries_are_dropped_before_split() -> None:
    rows = [
        _raw_row("row-1", "What is the weather in Paris today?"),
        _raw_row("row-2", "What   is   the   weather   in   paris   today?"),
    ]
    rows.extend(_raw_row(f"row-{index}", f"Other weather query {index}") for index in range(3, 8))
    result = prepare_split_result(
        rows,
        min_query_chars=10,
        max_query_chars=2000,
        n_train=3,
        n_validation=1,
        n_test=1,
        seed=1,
    )
    assert result.drop_counts["duplicate_query"] == 1


def test_insufficient_rows_fail_before_split() -> None:
    examples = [_prepared(f"row-{index}", f"Valid weather query {index}") for index in range(3)]
    with pytest.raises(UserInputError):
        split_examples(examples, n_train=2, n_validation=2, n_test=2, seed=1)


def test_drop_summary_written(tmp_path: Path) -> None:
    import yaml

    config_path = tmp_path / "config.smoke.yaml"
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 2
    raw["data"]["n_validation"] = 1
    raw["data"]["n_test"] = 1
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id="drop-summary",
        project_root=tmp_path,
    )
    out_dir = tmp_path / "artifacts" / "runs" / "drop-summary" / "data"
    prepare_dataset_from_file(
        config,
        input_path=FIXTURES_DIR / "preparation_rows.jsonl",
        out_dir=out_dir,
        context=context,
        command=["sommelier", "data", "prepare"],
    )
    summary = json.loads((out_dir / "drop_summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "sommelier.drop_summary.v1"
    assert summary["deduplicated_rows"] >= 4
