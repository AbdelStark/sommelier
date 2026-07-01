from __future__ import annotations

from pathlib import Path
from typing import Literal

from sommelier.artifacts import ArtifactRef, make_artifact_ref, read_jsonl_with_schema
from sommelier.config import SommelierConfig
from sommelier.run_context import (
    RunContext,
    query_digest,
    record_stage_success,
    write_jsonl_records,
)

SplitName = Literal["train", "validation", "test"]


def _fixture_prepared_example(
    *,
    example_id: str,
    split: SplitName,
    query: str,
    source_revision: str,
) -> dict[str, object]:
    return {
        "schema_version": "sommelier.prepared_example.v1",
        "example_id": example_id,
        "source_id": f"fixture:{example_id}",
        "query": query,
        "tools": [
            {
                "name": "lookup_weather",
                "description": "Look up weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "gold_calls": [
            {
                "name": "lookup_weather",
                "arguments": {"city": "Paris"},
            }
        ],
        "split": split,
        "query_sha256": query_digest(query),
        "source_revision": source_revision,
    }


PreparedExamples = dict[SplitName, list[dict[str, object]]]


def build_fixture_prepared_examples(config: SommelierConfig) -> PreparedExamples:
    split_sizes: dict[SplitName, int] = {
        "train": config.data.n_train,
        "validation": config.data.n_validation,
        "test": config.data.n_test,
    }
    prepared: dict[SplitName, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split, count in split_sizes.items():
        for index in range(count):
            query = f"Fixture {split} request {index + 1}: what is the weather in Paris?"
            prepared[split].append(
                _fixture_prepared_example(
                    example_id=f"{split}-{index + 1}",
                    split=split,
                    query=query,
                    source_revision=config.dataset.dataset_revision,
                )
            )
    return prepared


def prepare_dataset_fixture(
    config: SommelierConfig,
    *,
    out_dir: Path,
    context: RunContext,
    command: list[str],
) -> list[ArtifactRef]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = build_fixture_prepared_examples(config)
    outputs: list[ArtifactRef] = []
    for split, records in prepared.items():
        split_path = out_dir / f"{split}.jsonl"
        write_jsonl_records(split_path, records)
        outputs.append(
            make_artifact_ref(
                split_path,
                artifact_root=context.artifact_root,
                kind="dataset_split",
                schema_version="sommelier.prepared_example.v1",
            )
        )
    record_stage_success(
        context,
        stage="data",
        command=command,
        seed=config.project.seed,
        inputs=[context.config_ref],
        outputs=outputs,
    )
    return outputs


def validate_fixture_files(fixtures_dir: Path) -> None:
    for path in sorted(fixtures_dir.glob("*.jsonl")):
        read_jsonl_with_schema(path)
