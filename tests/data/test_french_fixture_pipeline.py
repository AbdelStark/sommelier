"""Offline end-to-end coverage of the bilingual path on real fixture files.

A fresh clone with no network runs this: raw English rows plus the
hand-written French fixture flow through prepare (pairing and drops),
fixture formatting, slice-aware evaluation with a stub generator, and
the gated comparison with its language gaps.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from sommelier.config import SommelierConfig, load_config
from sommelier.data.prepare import prepare_dataset_from_file
from sommelier.evaluation.generate import DecodingConfig, run_generation
from sommelier.evaluation.report import compare_evaluations, write_evaluation_report
from sommelier.formatting.chat import build_formatted_splits_fixture
from sommelier.run_context import RunContext, ensure_run_context

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def bilingual_config(tmp_path: Path) -> tuple[SommelierConfig, Path]:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 20
    raw["data"]["n_validation"] = 5
    raw["data"]["n_test"] = 5
    french = dict(raw["datasets"][0])
    french["language"] = "fr"
    french["dataset_id"] = "fixture/french-pairs"
    french["source_id_column"] = "source_example_id"
    raw["datasets"].append(french)
    raw["eval"]["slices"] = ["en", "fr"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(config_path), config_path


def run_prepare(tmp_path: Path) -> tuple[SommelierConfig, RunContext, Path]:
    config, config_path = bilingual_config(tmp_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id="french-files",
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


def test_french_fixture_prepare_pairs_and_drops(tmp_path: Path) -> None:
    _, _, data_dir = run_prepare(tmp_path)

    summary = json.loads((data_dir / "drop_summary.json").read_text(encoding="utf-8"))
    english = summary["languages"]["en"]
    french = summary["languages"]["fr"]
    # All 30 root rows are valid and every one is selected (20+5+5).
    assert english["split_sizes"] == {"train": 20, "validation": 5, "test": 5}
    # 26 pairs survive (24 plain, the unicode-heavy row, the untranslated
    # row); the four edge rows drop with their exact reasons.
    assert french["counts"]["missing_source_example"] == 1
    assert french["counts"]["pair_answers_mismatch"] == 1
    assert french["counts"]["pair_tools_mismatch"] == 1
    assert french["counts"]["duplicate_source_example"] == 1
    total_fr = sum(french["split_sizes"].values())
    assert total_fr == 26

    # Every surviving pair sits in its root's split, byte-identical gold.
    for split in ("train", "validation", "test"):
        records = [
            json.loads(line)
            for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        by_id = {record["example_id"]: record for record in records}
        for record in records:
            if record["language"] != "fr":
                continue
            root = by_id[record["source_example_id"]]
            assert root["split"] == record["split"]
            assert root["tools"] == record["tools"]
            assert root["gold_calls"] == record["gold_calls"]


def test_french_fixture_full_offline_chain(tmp_path: Path) -> None:
    config, context, data_dir = run_prepare(tmp_path)
    formatted_dir = context.run_dir / "formatted"
    build_formatted_splits_fixture(
        config,
        data_dir=data_dir,
        out_dir=formatted_dir,
        context=context,
        command=["test"],
    )

    test_records = [
        json.loads(line)
        for line in (formatted_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    gold_by_digest: dict[str, str] = {
        str(record["prompt_sha256"]): str(record["target_text"])[1:-1]
        for record in test_records
    }

    class GapGenerator:
        """Correct on English prompts, prose on French ones."""

        def generate(self, prompt_text: str, *, decoding: DecodingConfig) -> str:
            import hashlib

            digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            if "Quel temps" in prompt_text or "météo" in prompt_text:
                return "Je ne peux pas aider avec cela."
            return gold_by_digest[digest]

    class FullGenerator:
        def generate(self, prompt_text: str, *, decoding: DecodingConfig) -> str:
            import hashlib

            digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            return gold_by_digest[digest]

    for model_kind, generator in (("base", GapGenerator()), ("adapter", FullGenerator())):
        eval_dir = context.run_dir / "eval" / model_kind
        run_generation(
            config,
            formatted_dir=formatted_dir,
            out_dir=eval_dir,
            model_kind=model_kind,  # type: ignore[arg-type]
            context=context,
            command=["test"],
            generator=generator,
        )
        write_evaluation_report(
            config,
            formatted_dir=formatted_dir,
            eval_dir=eval_dir,
            model_kind=model_kind,  # type: ignore[arg-type]
            context=context,
            command=["test"],
        )

    out_dir = context.run_dir / "report"
    compare_evaluations(
        context.run_dir / "eval" / "base",
        context.run_dir / "eval" / "adapter",
        out_dir,
        command=["test"],
    )
    comparison = json.loads((out_dir / "comparison_report.json").read_text(encoding="utf-8"))
    assert set(comparison["slices"]) == {"en", "fr"}
    assert comparison["slices"]["en"]["examples"] == 5
    # The French test slice can run short of the root (edge rows dropped).
    assert 1 <= comparison["slices"]["fr"]["examples"] <= 5
    # The base model fails French while the adapter answers both: the gap
    # is negative for base and closes for the adapter, except for pairs
    # whose French text is the untranslated English row.
    gaps = comparison["language_gaps"]
    assert gaps["reference"] == "en"
    assert gaps["base"]["fr"]["valid_json_rate"] <= 0.0
    assert gaps["adapter"]["fr"]["valid_json_rate"] == 0.0
    markdown = (out_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert "## Language Gaps" in markdown
