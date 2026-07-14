"""Offline preparation and formatting coverage for the Hebrew v3 path."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from sommelier.config import SommelierConfig, load_config
from sommelier.data.prepare import (
    build_fixture_prepared_examples,
    prepare_dataset_from_file,
)
from sommelier.formatting.chat import build_formatted_splits_fixture
from sommelier.run_context import RunContext, ensure_run_context

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
HEBREW_CHAR = re.compile(r"[\u0590-\u05ff]")


@pytest.mark.parametrize(
    ("filename", "n_train", "gpu"),
    [
        ("config.v3-he-smoke.yaml", 100, "A10G"),
        ("config.v3-he-full.yaml", 15000, "L40S"),
    ],
)
def test_hebrew_v3_example_configs_load(filename: str, n_train: int, gpu: str) -> None:
    config = load_config(EXAMPLES_DIR / filename)

    assert [source.language for source in config.datasets] == ["en", "he"]
    assert config.train.languages == ["en", "he"]
    assert config.eval.slices == ["en", "he"]
    assert config.data.n_train == n_train
    assert config.remote.gpu == gpu
    assert config.dataset_for("he").source_id_column == "source_example_id"
    assert config.model.base_model_revision != "main"
    assert config.root_dataset.dataset_revision != "main"


def _fixture_config(tmp_path: Path) -> tuple[SommelierConfig, Path]:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.v3-he-smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 20
    raw["data"]["n_validation"] = 5
    raw["data"]["n_test"] = 5
    config_path = tmp_path / "config.v3-he-fixture.yaml"
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_config(config_path), config_path


def _prepare_fixture(
    tmp_path: Path,
) -> tuple[SommelierConfig, RunContext, Path]:
    config, config_path = _fixture_config(tmp_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id="hebrew-files",
        project_root=tmp_path,
    )
    data_dir = context.run_dir / "data"
    prepare_dataset_from_file(
        config,
        input_path=FIXTURES_DIR / "preparation_rows.jsonl",
        out_dir=data_dir,
        context=context,
        command=["test"],
    )
    return config, context, data_dir


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_hebrew_fixture_flows_through_prepare_and_format(tmp_path: Path) -> None:
    config, context, data_dir = _prepare_fixture(tmp_path)

    summary = json.loads((data_dir / "drop_summary.json").read_text(encoding="utf-8"))
    assert summary["languages"]["en"]["split_sizes"] == {
        "train": 20,
        "validation": 5,
        "test": 5,
    }
    assert summary["languages"]["he"]["split_sizes"] == {
        "train": 20,
        "validation": 5,
        "test": 5,
    }

    formatted_dir = context.run_dir / "formatted"
    build_formatted_splits_fixture(
        config,
        data_dir=data_dir,
        out_dir=formatted_dir,
        context=context,
        command=["test"],
    )

    for split, expected in (("train", 20), ("validation", 5), ("test", 5)):
        prepared = _read_jsonl(data_dir / f"{split}.jsonl")
        formatted = _read_jsonl(formatted_dir / f"{split}.jsonl")
        assert len(prepared) == len(formatted) == expected * 2

        prepared_by_id = {str(row["example_id"]): row for row in prepared}
        formatted_by_id = {str(row["example_id"]): row for row in formatted}
        hebrew_rows = [row for row in prepared if row["language"] == "he"]
        assert len(hebrew_rows) == expected
        for hebrew in hebrew_rows:
            assert HEBREW_CHAR.search(str(hebrew["query"]))
            root_id = str(hebrew["source_example_id"])
            root = prepared_by_id[root_id]
            assert hebrew["split"] == root["split"]
            assert hebrew["tools"] == root["tools"]
            assert hebrew["gold_calls"] == root["gold_calls"]

            formatted_hebrew = formatted_by_id[str(hebrew["example_id"])]
            formatted_root = formatted_by_id[root_id]
            messages = formatted_hebrew["messages"]
            assert isinstance(messages, list)
            assert messages[1]["content"] == hebrew["query"]
            assert formatted_hebrew["target_text"] == formatted_root["target_text"]
            assert formatted_hebrew["source_example_id"] == root_id


def test_synthetic_hebrew_fixture_uses_hebrew_queries(tmp_path: Path) -> None:
    config, _ = _fixture_config(tmp_path)
    prepared = build_fixture_prepared_examples(config)

    hebrew_rows = [
        row for split_rows in prepared.values() for row in split_rows if row["language"] == "he"
    ]
    assert len(hebrew_rows) == 30
    assert all(HEBREW_CHAR.search(row["query"]) for row in hebrew_rows)
    assert all(row["source_example_id"] is not None for row in hebrew_rows)
