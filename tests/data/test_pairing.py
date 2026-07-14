from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from sommelier.artifacts import make_artifact_ref
from sommelier.config import SommelierConfig, load_config
from sommelier.data.prepare import (
    build_fixture_prepared_examples,
    paired_input_path,
    prepare_dataset,
    prepare_dataset_from_file,
)
from sommelier.data.split import (
    assert_multilingual_disjointness,
    pair_split_result,
    prepare_split_result,
)
from sommelier.data.types import RawToolCallRow, SplitName
from sommelier.errors import SchemaValidationError, UserInputError
from sommelier.run_context import ensure_run_context, write_jsonl_records

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

TOOLS = '[{"name":"lookup_weather","description":"d","parameters":{}}]'
ANSWERS = '[{"name":"lookup_weather","arguments":{"city":"Paris"}}]'


def _root_row(index: int) -> RawToolCallRow:
    return RawToolCallRow(
        schema_version="sommelier.raw_tool_call_row.v1",
        source_id=f"en-{index}",
        query=f"What is the weather in city number {index} today?",
        tools=TOOLS,
        answers=ANSWERS,
        source_revision="fixture",
    )


def _paired_row(index: int, **overrides: str) -> RawToolCallRow:
    row = RawToolCallRow(
        schema_version="sommelier.raw_tool_call_row.v1",
        source_id=f"fr-{index}",
        query=f"Quel temps fait-il dans la ville numero {index} aujourd'hui?",
        tools=TOOLS,
        answers=ANSWERS,
        source_revision="fixture-fr",
    )
    row["source_example_id"] = f"en-{index}"
    for key, value in overrides.items():
        row[key] = value  # type: ignore[literal-required]
    return row


def _root_result(rows: list[RawToolCallRow]) -> Any:
    return prepare_split_result(
        rows,
        min_query_chars=10,
        max_query_chars=2000,
        n_train=4,
        n_validation=2,
        n_test=2,
        seed=42,
        language="en",
    )


def _pair(
    root_result: Any,
    root_rows: list[RawToolCallRow],
    paired_rows: list[RawToolCallRow],
) -> Any:
    return pair_split_result(
        root_result,
        root_rows,
        paired_rows,
        min_query_chars=10,
        max_query_chars=2000,
        language="fr",
    )


def test_paired_rows_inherit_root_splits_in_root_order() -> None:
    root_rows = [_root_row(index) for index in range(10)]
    paired_rows = [_paired_row(index) for index in range(10)]
    root_result = _root_result(root_rows)
    paired_result = _pair(root_result, root_rows, paired_rows)

    for split_name in ("train", "validation", "test"):
        root_split = getattr(root_result, split_name)
        paired_split = getattr(paired_result, split_name)
        assert [example["source_example_id"] for example in paired_split] == [
            example["example_id"] for example in root_split
        ]
        for example in paired_split:
            assert example["language"] == "fr"
            assert example["split"] == split_name


def test_missing_source_example_is_dropped_and_counted() -> None:
    root_rows = [_root_row(index) for index in range(10)]
    # en-999 never existed; en-11 exists as a raw row but is not selected
    # into any split (the root run only needs 8 of 10 rows), so both drop.
    paired_rows = [_paired_row(0), _paired_row(999)]
    root_result = _root_result(root_rows)
    paired_result = _pair(root_result, root_rows, paired_rows)
    total = len(paired_result.train) + len(paired_result.validation) + len(paired_result.test)
    assert total + paired_result.drop_counts["missing_source_example"] == 2


def _selected_index(root_result: Any, position: int = 0) -> int:
    """Index of a root row that actually made it into the train split."""
    return int(root_result.train[position]["example_id"].removeprefix("en-"))


def test_duplicate_source_example_keeps_first() -> None:
    root_rows = [_root_row(index) for index in range(10)]
    root_result = _root_result(root_rows)
    index = _selected_index(root_result)
    duplicate = _paired_row(
        index, source_id=f"fr-{index}-bis", query="Une autre traduction du meme temps?"
    )
    paired_rows = [_paired_row(index), duplicate]
    paired_result = _pair(root_result, root_rows, paired_rows)
    assert paired_result.drop_counts["duplicate_source_example"] == 1
    kept = [
        example
        for split in (paired_result.train, paired_result.validation, paired_result.test)
        for example in split
        if example["source_example_id"] == f"en-{index}"
    ]
    assert [example["source_id"] for example in kept] == [f"fr-{index}"]


def test_mutated_tools_and_answers_are_rejected() -> None:
    root_rows = [_root_row(index) for index in range(10)]
    root_result = _root_result(root_rows)
    mutated_tools = _paired_row(
        _selected_index(root_result, 0),
        tools='[{"name":"other_tool","description":"d","parameters":{}}]',
    )
    mutated_answers = _paired_row(
        _selected_index(root_result, 1),
        answers='[{"name":"lookup_weather","arguments":{"city":"Lyon"}}]',
    )
    paired_result = _pair(root_result, root_rows, [mutated_tools, mutated_answers])
    assert paired_result.drop_counts["pair_tools_mismatch"] == 1
    assert paired_result.drop_counts["pair_answers_mismatch"] == 1
    assert not paired_result.train and not paired_result.validation and not paired_result.test


def test_paired_language_dedupes_by_normalized_query() -> None:
    root_rows = [_root_row(index) for index in range(10)]
    first = _paired_row(0, query="La meme question sur le temps a Paris?")
    second = _paired_row(1, query="La  MEME  question sur le temps a Paris?")
    root_result = _root_result(root_rows)
    paired_result = _pair(root_result, root_rows, [first, second])
    assert paired_result.drop_counts["duplicate_query"] == 1


def test_cross_split_duplicate_is_dropped() -> None:
    root_rows = [_root_row(index) for index in range(10)]
    root_result = _root_result(root_rows)
    # A "translation" that came out identical to a root row's query. If the
    # colliding root row sits in a different split than the pair's root,
    # the pair must drop; a same-split collision would be a plain
    # duplicate within that split file and is allowed by the invariant.
    collision_target = root_result.train[0]
    donor_root_id = root_result.test[0]["example_id"]
    donor_index = int(donor_root_id.removeprefix("en-"))
    colliding = _paired_row(donor_index, query=collision_target["query"])
    paired_result = _pair(root_result, root_rows, [colliding])
    assert paired_result.drop_counts["cross_split_duplicate"] == 1
    assert not paired_result.test


def test_multilingual_disjointness_rejects_example_id_collision() -> None:
    root_rows = [_root_row(index) for index in range(10)]
    paired_rows = [_paired_row(index, source_id=f"en-{index}") for index in range(10)]
    root_result = _root_result(root_rows)
    paired_result = _pair(root_result, root_rows, paired_rows)
    with pytest.raises(UserInputError, match="appears more than once"):
        assert_multilingual_disjointness(
            {"en": root_result, "fr": paired_result}, root_language="en"
        )


def test_root_split_assignment_is_independent_of_paired_sources() -> None:
    root_rows = [_root_row(index) for index in range(10)]
    with_pairs = _root_result(root_rows)
    without_pairs = _root_result(root_rows)
    for split_name in ("train", "validation", "test"):
        assert [example["example_id"] for example in getattr(with_pairs, split_name)] == [
            example["example_id"] for example in getattr(without_pairs, split_name)
        ]


def _bilingual_config(tmp_path: Path, n_train: int = 4) -> SommelierConfig:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = n_train
    raw["data"]["n_validation"] = 2
    raw["data"]["n_test"] = 2
    french = dict(raw["datasets"][0])
    french["language"] = "fr"
    french["dataset_id"] = "fixture/french"
    french["source_id_column"] = "source_example_id"
    raw["datasets"].append(french)
    raw["eval"]["slices"] = ["en", "fr"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(config_path)


def _write_rows(path: Path, rows: list[RawToolCallRow]) -> None:
    write_jsonl_records(path, [dict(row) for row in rows])


def test_prepare_dataset_requires_rows_for_every_source(tmp_path: Path) -> None:
    config = _bilingual_config(tmp_path)
    context = ensure_run_context(
        config,
        config_path=tmp_path / "config.yaml",
        run_id="pairing-missing",
        project_root=tmp_path,
    )
    with pytest.raises(UserInputError, match="missing: fr"):
        prepare_dataset(
            config,
            rows_by_language={"en": [_root_row(index) for index in range(10)]},
            out_dir=tmp_path / "out",
            context=context,
            command=["test"],
        )


def test_prepare_dataset_requires_paired_coverage_in_every_split(tmp_path: Path) -> None:
    config = _bilingual_config(tmp_path)
    context = ensure_run_context(
        config,
        config_path=tmp_path / "config.yaml",
        run_id="pairing-coverage",
        project_root=tmp_path,
    )
    with pytest.raises(UserInputError, match="paired dataset has no surviving rows"):
        prepare_dataset(
            config,
            rows_by_language={
                "en": [_root_row(index) for index in range(10)],
                "fr": [_paired_row(0)],
            },
            out_dir=tmp_path / "out",
            context=context,
            command=["test"],
        )


def test_prepare_dataset_from_file_uses_paired_path_convention(tmp_path: Path) -> None:
    config = _bilingual_config(tmp_path)
    context = ensure_run_context(
        config,
        config_path=tmp_path / "config.yaml",
        run_id="pairing-files",
        project_root=tmp_path,
    )
    input_path = tmp_path / "rows.jsonl"
    _write_rows(input_path, [_root_row(index) for index in range(10)])
    _write_rows(
        paired_input_path(input_path, "fr"),
        [_paired_row(index) for index in range(10)],
    )
    out_dir = tmp_path / "artifacts" / "runs" / "pairing-files" / "data"
    prepare_dataset_from_file(
        config,
        input_path=input_path,
        out_dir=out_dir,
        context=context,
        command=["test"],
    )

    for split_name, expected in (("train", 4), ("validation", 2), ("test", 2)):
        records = [
            json.loads(line)
            for line in (out_dir / f"{split_name}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        by_language: dict[str, int] = {}
        for record in records:
            by_language[record["language"]] = by_language.get(record["language"], 0) + 1
            assert record["schema_version"] == "sommelier.prepared_example.v2"
        assert by_language == {"en": expected, "fr": expected}
        # Root rows come first, paired rows after, in root split order.
        english = [record for record in records if record["language"] == "en"]
        french = [record for record in records if record["language"] == "fr"]
        assert [record["source_example_id"] for record in french] == [
            record["example_id"] for record in english
        ]

    summary = json.loads((out_dir / "drop_summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "sommelier.drop_summary.v2"
    assert set(summary["languages"]) == {"en", "fr"}
    assert summary["languages"]["fr"]["split_sizes"] == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }


def test_data_manifest_records_checksummed_raw_and_translation_inputs(tmp_path: Path) -> None:
    config = _bilingual_config(tmp_path)
    context = ensure_run_context(
        config,
        config_path=tmp_path / "config.yaml",
        run_id="pairing-provenance",
        project_root=tmp_path,
    )
    out_dir = context.run_dir / "data"
    source_dir = out_dir / "source_inputs"
    input_path = source_dir / "rows.en.jsonl"
    paired_path = paired_input_path(input_path, "fr")
    _write_rows(input_path, [_root_row(index) for index in range(10)])
    _write_rows(paired_path, [_paired_row(index) for index in range(10)])
    summary_path = source_dir / "translation_summary.fr.json"
    summary_path.write_text("{}", encoding="utf-8")
    publication_path = source_dir / "translation_publication.fr.json"
    publication_path.write_text("{}", encoding="utf-8")
    source_inputs = [
        make_artifact_ref(
            input_path,
            artifact_root=context.artifact_root,
            kind="raw_dataset",
            schema_version="sommelier.raw_tool_call_row.v1",
        ),
        make_artifact_ref(
            paired_path,
            artifact_root=context.artifact_root,
            kind="raw_paired_dataset",
            schema_version="sommelier.raw_tool_call_row.v1",
        ),
        make_artifact_ref(
            summary_path,
            artifact_root=context.artifact_root,
            kind="translation_summary",
            schema_version="sommelier.translation_summary.v2",
        ),
        make_artifact_ref(
            publication_path,
            artifact_root=context.artifact_root,
            kind="translation_publication_manifest",
            schema_version="sommelier.translation_publication_manifest.v1",
        ),
    ]

    prepare_dataset_from_file(
        config,
        input_path=input_path,
        out_dir=out_dir,
        context=context,
        command=["test"],
        source_inputs=source_inputs,
    )

    manifest = json.loads((context.run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    assert [item["kind"] for item in manifest["inputs"]] == [
        "config",
        "raw_dataset",
        "raw_paired_dataset",
        "translation_summary",
        "translation_publication_manifest",
    ]
    for item in manifest["inputs"][1:]:
        assert len(item["sha256"]) == 64


def test_prepare_dataset_from_file_is_deterministic(tmp_path: Path) -> None:
    config = _bilingual_config(tmp_path)
    input_path = tmp_path / "rows.jsonl"
    _write_rows(input_path, [_root_row(index) for index in range(10)])
    _write_rows(
        paired_input_path(input_path, "fr"),
        [_paired_row(index) for index in range(10)],
    )
    outputs: list[str] = []
    for run_id in ("pairing-det-1", "pairing-det-2"):
        context = ensure_run_context(
            config,
            config_path=tmp_path / "config.yaml",
            run_id=run_id,
            project_root=tmp_path,
        )
        out_dir = tmp_path / "artifacts" / "runs" / run_id / "data"
        prepare_dataset_from_file(
            config,
            input_path=input_path,
            out_dir=out_dir,
            context=context,
            command=["test"],
        )
        outputs.append(
            "".join(
                (out_dir / f"{split}.jsonl").read_text(encoding="utf-8")
                for split in ("train", "validation", "test")
            )
        )
    assert outputs[0] == outputs[1]


def test_byte_check_uses_the_exact_row_an_example_came_from() -> None:
    root_rows = [_root_row(index) for index in range(10)]
    root_result = _root_result(root_rows)
    index = _selected_index(root_result)
    # Two rows share one source_id: the first carries mutated answers, the
    # second is clean but has a different query so both survive validation.
    # The byte check must reject the first row's example instead of blessing
    # it against the second row's clean payload.
    mutated = _paired_row(index, answers='[{"name":"lookup_weather","arguments":{"city":"Lyon"}}]')
    decoy = _paired_row(index, query="Une autre question sur le temps a Paris?")
    decoy["source_id"] = mutated["source_id"]
    paired_result = _pair(root_result, root_rows, [mutated, decoy])
    assert paired_result.drop_counts["pair_answers_mismatch"] == 1
    kept = [
        example
        for split in (paired_result.train, paired_result.validation, paired_result.test)
        for example in split
    ]
    assert [example["query"] for example in kept] == [decoy["query"]]


def test_unknown_paired_input_override_is_rejected(tmp_path: Path) -> None:
    config = _bilingual_config(tmp_path)
    context = ensure_run_context(
        config,
        config_path=tmp_path / "config.yaml",
        run_id="pairing-override",
        project_root=tmp_path,
    )
    input_path = tmp_path / "rows.jsonl"
    _write_rows(input_path, [_root_row(index) for index in range(10)])
    with pytest.raises(UserInputError, match="unconfigured language: de"):
        prepare_dataset_from_file(
            config,
            input_path=input_path,
            out_dir=tmp_path / "out",
            context=context,
            command=["test"],
            paired_input_paths={"de": tmp_path / "rows.de.jsonl"},
        )


def test_missing_paired_file_names_language_and_convention(tmp_path: Path) -> None:
    config = _bilingual_config(tmp_path)
    context = ensure_run_context(
        config,
        config_path=tmp_path / "config.yaml",
        run_id="pairing-missing-file",
        project_root=tmp_path,
    )
    input_path = tmp_path / "rows.jsonl"
    _write_rows(input_path, [_root_row(index) for index in range(10)])
    with pytest.raises(UserInputError, match="paired language 'fr'"):
        prepare_dataset_from_file(
            config,
            input_path=input_path,
            out_dir=tmp_path / "out",
            context=context,
            command=["test"],
        )


def test_fixture_builder_pairs_every_configured_language(tmp_path: Path) -> None:
    config = _bilingual_config(tmp_path, n_train=3)
    prepared = build_fixture_prepared_examples(config)
    for split_name, expected in (("train", 3), ("validation", 2), ("test", 2)):
        rows = prepared[cast(SplitName, split_name)]
        by_language: dict[str, int] = {}
        for record in rows:
            by_language[record["language"]] = by_language.get(record["language"], 0) + 1
        assert by_language == {"en": expected, "fr": expected}
        french = [record for record in rows if record["language"] == "fr"]
        english_ids = [record["example_id"] for record in rows if record["language"] == "en"]
        assert [record["source_example_id"] for record in french] == english_ids
        assert all(record["gold_calls"] == rows[0]["gold_calls"] for record in rows)


def test_paired_rows_require_source_example_id(tmp_path: Path) -> None:
    config = _bilingual_config(tmp_path)
    context = ensure_run_context(
        config,
        config_path=tmp_path / "config.yaml",
        run_id="pairing-schema",
        project_root=tmp_path,
    )
    input_path = tmp_path / "rows.jsonl"
    _write_rows(input_path, [_root_row(index) for index in range(10)])
    bare_rows = [_paired_row(index) for index in range(10)]
    for row in bare_rows:
        del row["source_example_id"]
    _write_rows(paired_input_path(input_path, "fr"), bare_rows)
    with pytest.raises(SchemaValidationError, match="source_example_id"):
        prepare_dataset_from_file(
            config,
            input_path=input_path,
            out_dir=tmp_path / "out",
            context=context,
            command=["test"],
        )
