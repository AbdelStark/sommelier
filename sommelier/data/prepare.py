from __future__ import annotations

import json
from pathlib import Path

from sommelier.artifacts import (
    ArtifactRef,
    make_artifact_ref,
    read_jsonl_with_schema,
    write_artifact_atomic,
)
from sommelier.config import SommelierConfig
from sommelier.data.load import load_raw_rows
from sommelier.data.normalize import query_digest
from sommelier.data.split import SplitResult, assert_split_disjointness, prepare_split_result
from sommelier.data.types import PreparedExample, RawToolCallRow, SplitName
from sommelier.run_context import RunContext, record_stage_success, write_jsonl_records

FixturePreparedExamples = dict[SplitName, list[PreparedExample]]


def build_fixture_prepared_examples(config: SommelierConfig) -> FixturePreparedExamples:
    split_sizes: dict[SplitName, int] = {
        "train": config.data.n_train,
        "validation": config.data.n_validation,
        "test": config.data.n_test,
    }
    prepared: dict[SplitName, list[PreparedExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split, count in split_sizes.items():
        for index in range(count):
            query = f"Fixture {split} request {index + 1}: what is the weather in Paris?"
            prepared[split].append(
                PreparedExample(
                    schema_version="sommelier.prepared_example.v1",
                    example_id=f"{split}-{index + 1}",
                    source_id=f"fixture:{split}-{index + 1}",
                    query=query,
                    tools=[
                        {
                            "name": "lookup_weather",
                            "description": "Look up weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        }
                    ],
                    gold_calls=[{"name": "lookup_weather", "arguments": {"city": "Paris"}}],
                    split=split,
                    query_sha256=query_digest(query),
                    source_revision=config.dataset.dataset_revision,
                )
            )
    return prepared


def build_drop_summary(result: SplitResult, config: SommelierConfig) -> dict[str, object]:
    return {
        "schema_version": "sommelier.drop_summary.v1",
        "counts": dict(result.drop_counts),
        "valid_rows": result.valid_rows,
        "deduplicated_rows": result.deduplicated_rows,
        "requested": {
            "train": config.data.n_train,
            "validation": config.data.n_validation,
            "test": config.data.n_test,
        },
    }


def _write_split_outputs(
    result: SplitResult,
    config: SommelierConfig,
    *,
    out_dir: Path,
    context: RunContext,
) -> list[ArtifactRef]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[ArtifactRef] = []
    split_map: dict[SplitName, list[PreparedExample]] = {
        "train": result.train,
        "validation": result.validation,
        "test": result.test,
    }
    for split, records in split_map.items():
        split_path = out_dir / f"{split}.jsonl"
        write_jsonl_records(split_path, [dict(record) for record in records])
        outputs.append(
            make_artifact_ref(
                split_path,
                artifact_root=context.artifact_root,
                kind="dataset_split",
                schema_version="sommelier.prepared_example.v1",
            )
        )

    drop_summary_path = out_dir / "drop_summary.json"
    drop_summary = build_drop_summary(result, config)

    def writer(temp_path: Path) -> None:
        temp_path.write_text(json.dumps(drop_summary, indent=2, sort_keys=True), encoding="utf-8")

    outputs.append(
        write_artifact_atomic(
            drop_summary_path,
            writer,
            artifact_root=context.artifact_root,
            kind="drop_summary",
            schema_version="sommelier.drop_summary.v1",
        )
    )
    return outputs


def prepare_dataset(
    config: SommelierConfig,
    *,
    raw_rows: list[RawToolCallRow],
    out_dir: Path,
    context: RunContext,
    command: list[str],
) -> list[ArtifactRef]:
    result = prepare_split_result(
        raw_rows,
        min_query_chars=config.data.min_query_chars,
        max_query_chars=config.data.max_query_chars,
        n_train=config.data.n_train,
        n_validation=config.data.n_validation,
        n_test=config.data.n_test,
        seed=config.project.seed,
    )
    assert_split_disjointness(result)
    outputs = _write_split_outputs(result, config, out_dir=out_dir, context=context)
    record_stage_success(
        context,
        stage="data",
        command=command,
        seed=config.project.seed,
        inputs=[context.config_ref],
        outputs=outputs,
    )
    return outputs


def prepare_dataset_from_file(
    config: SommelierConfig,
    *,
    input_path: Path,
    out_dir: Path,
    context: RunContext,
    command: list[str],
    use_gpu: bool = False,
) -> list[ArtifactRef]:
    if use_gpu:
        from sommelier.data.gpu import load_and_coarse_filter

        raw_rows = load_and_coarse_filter(config, input_path)
    else:
        raw_rows = load_raw_rows(input_path)
    return prepare_dataset(
        config,
        raw_rows=raw_rows,
        out_dir=out_dir,
        context=context,
        command=command,
    )


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
        write_jsonl_records(split_path, [dict(record) for record in records])
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
