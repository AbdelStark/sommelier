from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from sommelier.analysis.tokenization import (
    TOKENIZER_TAX_RECORD_SCHEMA,
    TOKENIZER_TAX_RECORDS_FILENAME,
    TOKENIZER_TAX_REPORT_FILENAME,
    TOKENIZER_TAX_REPORT_SCHEMA,
    analyze_tokenizer_tax,
)
from sommelier.config import SommelierConfig, load_config
from sommelier.errors import SchemaValidationError
from sommelier.run_context import RunContext, ensure_run_context, write_jsonl_records

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


class HebrewTaxTokenizer:
    """One token per non-Hebrew character, two per Hebrew character."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        tokens: list[int] = []
        for character in text:
            tokens.append(ord(character))
            if "\u0590" <= character <= "\u05ff":
                tokens.append(ord(character) + 1_000_000)
        return tokens


class EmptyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return []


def _config_and_context(tmp_path: Path) -> tuple[SommelierConfig, RunContext, Path]:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    hebrew = dict(raw["datasets"][0])
    hebrew["language"] = "he"
    hebrew["dataset_id"] = "fixture/hebrew"
    hebrew["dataset_revision"] = "hebrew-rev-1"
    hebrew["source_id_column"] = "source_example_id"
    raw["datasets"].append(hebrew)
    raw["train"]["languages"] = ["en", "he"]
    raw["train"]["max_sequence_length"] = 4
    raw["eval"]["slices"] = ["en", "he"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_config(config_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id="tokenizer-tax-test",
        project_root=Path.cwd(),
    )
    return config, context, config_path


def _record(
    *,
    example_id: str,
    split: str,
    language: str,
    query: str,
    source_example_id: str | None,
    target: str = "T",
) -> dict[str, object]:
    return {
        "schema_version": "sommelier.formatted_example.v2",
        "example_id": example_id,
        "split": split,
        "language": language,
        "source_example_id": source_example_id,
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": target},
        ],
        # Tiny strings make all expected token counts transparent.
        "prompt_text": query,
        "target_text": target,
        "full_text": query + target,
        "prompt_sha256": "fixture",
        "tokenizer_id": "fixture/tokenizer",
        "tokenizer_revision": "fixture-rev",
        "template_policy": "tokenizer_chat_template",
    }


def _write_fixture(
    formatted_dir: Path,
    *,
    mutation: Any | None = None,
) -> None:
    for split in ("train", "validation", "test"):
        root_id = f"{split}-root"
        records = [
            _record(
                example_id=root_id,
                split=split,
                language="en",
                query="go",
                source_example_id=None,
            )
        ]
        # Deliberately leave the Hebrew test pair absent: coverage is evidence,
        # and translation drops must not be hidden by matching different roots.
        if split != "test":
            records.append(
                _record(
                    example_id=f"{split}-he",
                    split=split,
                    language="he",
                    query="אב",
                    source_example_id=root_id,
                )
            )
        if mutation is not None:
            mutation(split, records)
        write_jsonl_records(formatted_dir / f"{split}.jsonl", records)


def _run(tmp_path: Path) -> tuple[RunContext, Path, dict[str, Any], list[dict[str, Any]]]:
    config, context, _ = _config_and_context(tmp_path)
    formatted_dir = context.run_dir / "formatted"
    _write_fixture(formatted_dir)
    out_dir = context.run_dir / "analysis" / "tokenization"
    analyze_tokenizer_tax(
        config,
        formatted_dir=formatted_dir,
        out_dir=out_dir,
        context=context,
        command=["test", "tokenizer-tax"],
        tokenizer=HebrewTaxTokenizer(),
    )
    report = json.loads((out_dir / TOKENIZER_TAX_REPORT_FILENAME).read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (out_dir / TOKENIZER_TAX_RECORDS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return context, out_dir, report, records


def test_analysis_writes_exact_paired_counts_coverage_and_manifest(tmp_path: Path) -> None:
    context, out_dir, report, records = _run(tmp_path)
    resolved_config = yaml.safe_load(
        (context.run_dir / "config.resolved.yaml").read_text(encoding="utf-8")
    )

    assert report["schema_version"] == TOKENIZER_TAX_REPORT_SCHEMA
    assert report["tokenizer"] == {
        "id": resolved_config["model"]["base_model_id"],
        "revision": resolved_config["model"]["tokenizer_revision"],
    }
    assert report["max_sequence_length"] == 4
    assert set(report["inputs"]) == {"train", "validation", "test"}
    assert all(len(item["sha256"]) == 64 for item in report["inputs"].values())

    english = report["languages"]["en"]["all"]
    hebrew = report["languages"]["he"]["all"]
    assert english["examples"] == 3
    assert english["counts"]["query_chars"]["total"] == 6
    assert english["counts"]["query_utf8_bytes"]["total"] == 6
    assert english["counts"]["query_tokens"]["total"] == 6
    assert hebrew["examples"] == 2
    assert hebrew["counts"]["query_chars"]["total"] == 4
    assert hebrew["counts"]["query_utf8_bytes"]["total"] == 8
    assert hebrew["counts"]["query_tokens"]["total"] == 8
    assert hebrew["counts"]["query_tokens"] == {
        "total": 8,
        "mean": 4.0,
        "p50": 4,
        "p95": 4,
        "p99": 4,
        "max": 4,
    }
    assert hebrew["rates"] == {
        "query_tokens_per_character": 2.0,
        "query_tokens_per_utf8_byte": 1.0,
        "query_tokens_per_whitespace_word": 4.0,
    }
    assert english["over_budget"] == 0
    assert hebrew["over_budget"] == 2

    paired = report["pairing"]["he"]["all"]
    assert paired["coverage"] == {"paired": 2, "roots": 3, "ratio": 2 / 3}
    assert paired["metrics"]["query_chars"]["ratio"] == 1.0
    assert paired["metrics"]["query_utf8_bytes"]["ratio"] == 2.0
    assert paired["metrics"]["query_tokens"]["ratio"] == 2.0
    assert paired["metrics"]["prompt_tokens"]["ratio"] == 2.0
    assert paired["metrics"]["target_tokens"]["ratio"] == 1.0
    assert paired["metrics"]["full_tokens"]["ratio"] == 5 / 3
    assert report["pairing"]["he"]["splits"]["test"]["coverage"] == {
        "paired": 0,
        "roots": 1,
        "ratio": 0.0,
    }
    assert report["pairing"]["he"]["splits"]["test"]["metrics"]["query_tokens"]["ratio"] is None
    assert report["training_workload"] == {
        "languages": ["en", "he"],
        "examples_per_epoch": 2,
        "non_padding_full_tokens_per_epoch": 8,
        "epochs": 1,
        "projected_non_padding_full_tokens": 8,
        "boundary": (
            "Excludes dynamic padding and is a deterministic lower bound on tokens "
            "processed by training."
        ),
    }

    assert len(records) == 5
    assert all(record["schema_version"] == TOKENIZER_TAX_RECORD_SCHEMA for record in records)
    hebrew_record = next(record for record in records if record["example_id"] == "train-he")
    assert hebrew_record["root_example_id"] == "train-root"
    assert hebrew_record["ratios_to_root"]["query_tokens"] == 2.0
    assert hebrew_record["ratios_to_root"]["target_tokens"] == 1.0
    assert hebrew_record["over_budget"] is True

    manifest = json.loads((context.run_dir / "tokenization_manifest.json").read_text())
    assert manifest["stage"] == "tokenization"
    assert manifest["status"] == "succeeded"
    assert len(manifest["inputs"]) == 3
    assert {output["kind"] for output in manifest["outputs"]} == {
        "tokenizer_tax_records",
        "tokenizer_tax_report",
    }
    run_manifest = json.loads((context.run_dir / "manifest.json").read_text())
    assert "tokenization" in run_manifest["stages"]
    assert (out_dir / TOKENIZER_TAX_REPORT_FILENAME).exists()


def test_outputs_are_byte_deterministic_for_the_same_inputs(tmp_path: Path) -> None:
    config, context, _ = _config_and_context(tmp_path)
    formatted_dir = context.run_dir / "formatted"
    _write_fixture(formatted_dir)
    out_dir = context.run_dir / "analysis" / "tokenization"

    def run() -> tuple[bytes, bytes]:
        analyze_tokenizer_tax(
            config,
            formatted_dir=formatted_dir,
            out_dir=out_dir,
            context=context,
            command=["test"],
            tokenizer=HebrewTaxTokenizer(),
        )
        return (
            (out_dir / TOKENIZER_TAX_RECORDS_FILENAME).read_bytes(),
            (out_dir / TOKENIZER_TAX_REPORT_FILENAME).read_bytes(),
        )

    assert run() == run()


def test_missing_root_identity_fails_closed(tmp_path: Path) -> None:
    config, context, _ = _config_and_context(tmp_path)
    formatted_dir = context.run_dir / "formatted"

    def mutate(split: str, records: list[dict[str, object]]) -> None:
        if split == "train":
            records[1]["source_example_id"] = "does-not-exist"

    _write_fixture(formatted_dir, mutation=mutate)
    with pytest.raises(SchemaValidationError, match="references missing root"):
        analyze_tokenizer_tax(
            config,
            formatted_dir=formatted_dir,
            out_dir=context.run_dir / "analysis",
            context=context,
            command=["test"],
            tokenizer=HebrewTaxTokenizer(),
        )


def test_duplicate_root_identity_fails_closed(tmp_path: Path) -> None:
    config, context, _ = _config_and_context(tmp_path)
    formatted_dir = context.run_dir / "formatted"

    def mutate(split: str, records: list[dict[str, object]]) -> None:
        if split == "validation":
            records[0]["example_id"] = "train-root"

    _write_fixture(formatted_dir, mutation=mutate)
    with pytest.raises(SchemaValidationError, match="duplicate formatted example_id"):
        analyze_tokenizer_tax(
            config,
            formatted_dir=formatted_dir,
            out_dir=context.run_dir / "analysis",
            context=context,
            command=["test"],
            tokenizer=HebrewTaxTokenizer(),
        )


def test_duplicate_language_pair_fails_closed(tmp_path: Path) -> None:
    config, context, _ = _config_and_context(tmp_path)
    formatted_dir = context.run_dir / "formatted"

    def mutate(split: str, records: list[dict[str, object]]) -> None:
        if split == "train":
            duplicate = dict(records[1])
            duplicate["example_id"] = "train-he-duplicate"
            records.append(duplicate)

    _write_fixture(formatted_dir, mutation=mutate)
    with pytest.raises(SchemaValidationError, match="duplicate 'he' pair"):
        analyze_tokenizer_tax(
            config,
            formatted_dir=formatted_dir,
            out_dir=context.run_dir / "analysis",
            context=context,
            command=["test"],
            tokenizer=HebrewTaxTokenizer(),
        )


def test_paired_target_mismatch_fails_closed(tmp_path: Path) -> None:
    config, context, _ = _config_and_context(tmp_path)
    formatted_dir = context.run_dir / "formatted"

    def mutate(split: str, records: list[dict[str, object]]) -> None:
        if split == "train":
            records[1]["target_text"] = "different"

    _write_fixture(formatted_dir, mutation=mutate)
    with pytest.raises(SchemaValidationError, match="target differs from root"):
        analyze_tokenizer_tax(
            config,
            formatted_dir=formatted_dir,
            out_dir=context.run_dir / "analysis",
            context=context,
            command=["test"],
            tokenizer=HebrewTaxTokenizer(),
        )


def test_zero_token_encoding_is_rejected_before_ratio_serialization(tmp_path: Path) -> None:
    config, context, _ = _config_and_context(tmp_path)
    formatted_dir = context.run_dir / "formatted"
    _write_fixture(formatted_dir)
    with pytest.raises(SchemaValidationError, match="encoded to zero tokens"):
        analyze_tokenizer_tax(
            config,
            formatted_dir=formatted_dir,
            out_dir=context.run_dir / "analysis",
            context=context,
            command=["test"],
            tokenizer=EmptyTokenizer(),
        )


def test_every_persisted_ratio_is_finite_or_explicitly_unavailable(tmp_path: Path) -> None:
    _, _, report, records = _run(tmp_path)

    def assert_finite(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                assert_finite(child)
        elif isinstance(value, list):
            for child in value:
                assert_finite(child)
        elif isinstance(value, float):
            assert math.isfinite(value)

    assert_finite(report)
    assert_finite(records)
