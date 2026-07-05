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
from sommelier.data.split import (
    SplitResult,
    assert_multilingual_disjointness,
    examples_for_split,
    pair_split_result,
    prepare_split_result,
)
from sommelier.data.types import (
    DROP_SUMMARY_SCHEMA,
    PREPARED_EXAMPLE_SCHEMA,
    PreparedExample,
    RawToolCallRow,
    SplitName,
)
from sommelier.errors import UserInputError
from sommelier.run_context import RunContext, record_stage_success, write_jsonl_records

FixturePreparedExamples = dict[SplitName, list[PreparedExample]]


def paired_input_path(input_path: Path, language: str) -> Path:
    """Convention: a paired source's rows sit next to the root rows file,
    named by language (``rows.jsonl`` pairs with ``rows.fr.jsonl``)."""
    return input_path.with_name(f"{input_path.stem}.{language}{input_path.suffix}")


def build_fixture_prepared_examples(config: SommelierConfig) -> FixturePreparedExamples:
    """Synthesizes prepared examples for every configured dataset source.

    Paired sources get one fixture row per root row, linked through
    ``source_example_id``, so a multilingual config exercises the same
    split shape in fixture mode that the real path guarantees.
    """
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
    for source in config.datasets:
        is_root = source.source_id_column is None
        for split, count in split_sizes.items():
            for index in range(count):
                root_example_id = f"{split}-{index + 1}"
                if is_root:
                    query = f"Fixture {split} request {index + 1}: what is the weather in Paris?"
                    example_id = root_example_id
                else:
                    query = (
                        f"Fixture {split} demande {index + 1}: "
                        "quel temps fait-il a Paris?"
                    )
                    example_id = f"{root_example_id}-{source.language}"
                prepared[split].append(
                    PreparedExample(
                        schema_version=PREPARED_EXAMPLE_SCHEMA,
                        example_id=example_id,
                        source_id=f"fixture:{example_id}",
                        language=source.language,
                        source_example_id=None if is_root else root_example_id,
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
                        gold_calls=[
                            {"name": "lookup_weather", "arguments": {"city": "Paris"}}
                        ],
                        split=split,
                        query_sha256=query_digest(query),
                        source_revision=source.dataset_revision,
                    )
                )
    return prepared


def build_drop_summary(
    results: dict[str, SplitResult],
    config: SommelierConfig,
) -> dict[str, object]:
    languages: dict[str, object] = {}
    for language, result in results.items():
        languages[language] = {
            "counts": dict(result.drop_counts),
            "valid_rows": result.valid_rows,
            "deduplicated_rows": result.deduplicated_rows,
            "split_sizes": {
                "train": len(result.train),
                "validation": len(result.validation),
                "test": len(result.test),
            },
        }
    return {
        "schema_version": DROP_SUMMARY_SCHEMA,
        "languages": languages,
        "requested": {
            "train": config.data.n_train,
            "validation": config.data.n_validation,
            "test": config.data.n_test,
        },
    }


def _write_split_outputs(
    results: dict[str, SplitResult],
    config: SommelierConfig,
    *,
    out_dir: Path,
    context: RunContext,
) -> list[ArtifactRef]:
    """Writes one file per split, root language rows first, then each
    paired language in configuration order (each already following the
    root split's example order)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[ArtifactRef] = []
    ordered_languages = [
        source.language for source in config.datasets if source.language in results
    ]
    splits: tuple[SplitName, ...] = ("train", "validation", "test")
    for split in splits:
        split_path = out_dir / f"{split}.jsonl"
        records: list[PreparedExample] = []
        for language in ordered_languages:
            records.extend(examples_for_split(results[language], split))
        write_jsonl_records(split_path, [dict(record) for record in records])
        outputs.append(
            make_artifact_ref(
                split_path,
                artifact_root=context.artifact_root,
                kind="dataset_split",
                schema_version=PREPARED_EXAMPLE_SCHEMA,
            )
        )

    drop_summary_path = out_dir / "drop_summary.json"
    drop_summary = build_drop_summary(results, config)

    def writer(temp_path: Path) -> None:
        temp_path.write_text(json.dumps(drop_summary, indent=2, sort_keys=True), encoding="utf-8")

    outputs.append(
        write_artifact_atomic(
            drop_summary_path,
            writer,
            artifact_root=context.artifact_root,
            kind="drop_summary",
            schema_version=DROP_SUMMARY_SCHEMA,
        )
    )
    return outputs


def prepare_dataset(
    config: SommelierConfig,
    *,
    rows_by_language: dict[str, list[RawToolCallRow]],
    out_dir: Path,
    context: RunContext,
    command: list[str],
) -> list[ArtifactRef]:
    configured = {source.language for source in config.datasets}
    provided = set(rows_by_language)
    if provided != configured:
        missing = ", ".join(sorted(configured - provided)) or "none"
        unexpected = ", ".join(sorted(provided - configured)) or "none"
        raise UserInputError(
            f"raw rows do not match configured dataset sources "
            f"(missing: {missing}; unexpected: {unexpected})",
            hint="Provide one raw rows file per configured dataset language.",
        )

    root_language = config.root_dataset.language
    root_rows = rows_by_language[root_language]
    root_result = prepare_split_result(
        root_rows,
        min_query_chars=config.data.min_query_chars,
        max_query_chars=config.data.max_query_chars,
        n_train=config.data.n_train,
        n_validation=config.data.n_validation,
        n_test=config.data.n_test,
        seed=config.project.seed,
        language=root_language,
    )
    results: dict[str, SplitResult] = {root_language: root_result}
    for source in config.datasets:
        if source.language == root_language:
            continue
        results[source.language] = pair_split_result(
            root_result,
            root_rows,
            rows_by_language[source.language],
            min_query_chars=config.data.min_query_chars,
            max_query_chars=config.data.max_query_chars,
            language=source.language,
        )
    assert_multilingual_disjointness(results, root_language=root_language)
    outputs = _write_split_outputs(results, config, out_dir=out_dir, context=context)
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
    paired_input_paths: dict[str, Path] | None = None,
) -> list[ArtifactRef]:
    """Loads the root rows from ``input_path`` and each paired source's rows
    from an explicit override or the :func:`paired_input_path` convention.

    The GPU coarse filter applies to the root rows only: paired rows are
    few enough that Python validation is the cheap part, and every rule the
    filter approximates is re-checked there anyway.
    """
    root_language = config.root_dataset.language
    if use_gpu:
        from sommelier.data.gpu import load_and_coarse_filter

        root_rows = load_and_coarse_filter(config, input_path)
    else:
        root_rows = load_raw_rows(input_path)

    overrides = dict(paired_input_paths or {})
    paired_languages = {
        source.language for source in config.datasets if source.source_id_column is not None
    }
    unknown_overrides = sorted(set(overrides) - paired_languages)
    if unknown_overrides:
        raise UserInputError(
            f"paired input given for unconfigured language: {', '.join(unknown_overrides)}",
            hint="Each --paired-input language must match a dataset source with "
            "source_id_column set; root rows come from --input.",
        )

    rows_by_language: dict[str, list[RawToolCallRow]] = {root_language: root_rows}
    for source in config.datasets:
        if source.source_id_column is None:
            continue
        source_path = overrides.get(source.language, paired_input_path(input_path, source.language))
        if not source_path.exists():
            raise UserInputError(
                f"raw rows file for paired language {source.language!r} not found: "
                f"{source_path}",
                hint="Pass --paired-input "
                f"{source.language}=<path> or place the file next to --input "
                f"as {paired_input_path(input_path, source.language).name}.",
            )
        rows_by_language[source.language] = load_raw_rows(
            source_path,
            require_source_example_id=True,
        )
    return prepare_dataset(
        config,
        rows_by_language=rows_by_language,
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
                schema_version=PREPARED_EXAMPLE_SCHEMA,
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
