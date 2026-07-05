from __future__ import annotations

import json
from pathlib import Path

from sommelier.config import DatasetSourceConfig
from sommelier.errors import ExternalDependencyError


def export_raw_rows(
    source: DatasetSourceConfig,
    out_path: Path,
    *,
    seed: int,
    max_rows: int = 0,
) -> int:
    """Exports a Hugging Face dataset as sommelier.raw_tool_call_row.v1 JSONL.

    The configured column names map into the canonical raw-row fields here
    and nowhere else. ``max_rows`` of zero exports everything; a positive
    value takes a seeded shuffle's prefix, which is how smoke runs bound
    their input.
    """
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ExternalDependencyError(
            "dataset export requires the datasets package",
            hint="Run the export remotely or install the datasets extra.",
        ) from error

    dataset = load_dataset(
        source.dataset_id,
        split="train",
        revision=source.dataset_revision,
    )
    if max_rows and max_rows < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_rows))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            record = {
                "schema_version": "sommelier.raw_tool_call_row.v1",
                "source_id": f"{source.dataset_id}:{row.get('id', index)}",
                "query": str(row[source.query_column]),
                "tools": str(row[source.tools_column]),
                "answers": str(row[source.answers_column]),
                "source_revision": source.dataset_revision,
            }
            if source.source_id_column is not None:
                record["source_example_id"] = str(row[source.source_id_column])
            handle.write(json.dumps(record) + "\n")
    return len(dataset)
