from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sommelier.config import DatasetSourceConfig
from sommelier.data.export import export_raw_rows


@pytest.mark.parametrize(
    ("source_id_column", "expected_data_files"),
    [(None, None), ("source_example_id", "rows.he.jsonl")],
)
def test_export_selects_only_paired_row_file_when_provenance_sidecars_coexist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_id_column: str | None,
    expected_data_files: str | None,
) -> None:
    observed: dict[str, Any] = {}

    def load_dataset(dataset_id: str, **kwargs: object) -> list[dict[str, object]]:
        observed.update({"dataset_id": dataset_id, **kwargs})
        row: dict[str, object] = {
            "id": "row-1",
            "query": "Find one item",
            "tools": "[]",
            "answers": "[]",
        }
        if source_id_column is not None:
            row[source_id_column] = "root-1"
        return [row]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    source = DatasetSourceConfig(
        language="he" if source_id_column is not None else "en",
        dataset_id="example/publication",
        dataset_revision="a" * 40,
        source_id_column=source_id_column,
    )

    assert export_raw_rows(source, tmp_path / "rows.jsonl", seed=42) == 1
    assert observed["dataset_id"] == "example/publication"
    assert observed["split"] == "train"
    assert observed["revision"] == "a" * 40
    if expected_data_files is None:
        assert "data_files" not in observed
    else:
        assert observed["data_files"] == expected_data_files
