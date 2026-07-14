import json
from pathlib import Path

import pytest

from sommelier.artifacts import read_json_with_schema, read_jsonl_with_schema
from sommelier.data.load import load_raw_rows
from sommelier.errors import SchemaValidationError


def test_read_json_with_supported_schema(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_text(
        json.dumps({"schema_version": "sommelier.manifest.v1", "stage": "data"}),
        encoding="utf-8",
    )
    payload = read_json_with_schema(path, expected_schema="sommelier.manifest.v1")
    assert payload["stage"] == "data"


@pytest.mark.parametrize(
    "schema_version",
    [
        "sommelier.experiment_report.v1",
        "sommelier.experiment_report.v2",
    ],
)
def test_read_json_with_experiment_report_schema(
    tmp_path: Path,
    schema_version: str,
) -> None:
    path = tmp_path / "experiment_report.json"
    path.write_text(
        json.dumps({"schema_version": schema_version}),
        encoding="utf-8",
    )
    payload = read_json_with_schema(
        path,
        expected_schema=schema_version,
    )
    assert payload["schema_version"] == schema_version


def test_read_json_rejects_missing_schema(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"stage": "data"}), encoding="utf-8")
    with pytest.raises(SchemaValidationError):
        read_json_with_schema(path)


def test_read_json_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    path.write_text(
        json.dumps({"schema_version": "sommelier.manifest.v99"}),
        encoding="utf-8",
    )
    with pytest.raises(SchemaValidationError):
        read_json_with_schema(path)


def test_read_jsonl_with_schema(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "sommelier.prepared_example.v1",
                        "example_id": "a",
                    }
                ),
                json.dumps(
                    {
                        "schema_version": "sommelier.prepared_example.v1",
                        "example_id": "b",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    records = read_jsonl_with_schema(
        path,
        expected_schema="sommelier.prepared_example.v1",
    )
    assert [record["example_id"] for record in records] == ["a", "b"]


@pytest.mark.parametrize(
    "duplicated_fields",
    [
        '"source_id":"root","source_id":"do-not-disclose"',
        '"query":"safe query","query":"do-not-disclose"',
    ],
)
def test_load_raw_rows_rejects_duplicate_source_fields_without_echoing_values(
    tmp_path: Path,
    duplicated_fields: str,
) -> None:
    path = tmp_path / "source.jsonl"
    path.write_text(
        "{"
        '"schema_version":"sommelier.raw_tool_call_row.v1",'
        f"{duplicated_fields},"
        '"source_id":"root","query":"safe query",'
        '"tools":"[]","answers":"[]","source_revision":"fixture"'
        "}\n",
        encoding="utf-8",
    )

    with pytest.raises(SchemaValidationError, match="contains invalid JSON") as caught:
        load_raw_rows(path)

    assert "do-not-disclose" not in str(caught.value)
