from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from sommelier.config import SommelierConfig, load_config
from sommelier.data.prepare import prepare_dataset_fixture
from sommelier.errors import EvaluationError, SchemaValidationError, UserInputError
from sommelier.evaluation.generate import AdapterRef, DecodingConfig, run_generation
from sommelier.evaluation.report import compare_evaluations, write_evaluation_report
from sommelier.formatting.chat import build_formatted_splits_fixture
from sommelier.run_context import RunContext, ensure_run_context

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

GOOD_CALL = '{"arguments":{"city":"Paris"},"name":"lookup_weather"}'


class LanguageAwareGenerator:
    """Answers correctly for English prompts, with prose for French ones."""

    def __init__(self, french_text: str = "Je ne peux pas aider.") -> None:
        self.french_text = french_text

    def generate(self, prompt_text: str, *, decoding: DecodingConfig) -> str:
        if "demande" in prompt_text:
            return self.french_text
        return GOOD_CALL


class FixedGenerator:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, prompt_text: str, *, decoding: DecodingConfig) -> str:
        return self.text


def setup_bilingual_run(
    tmp_path: Path,
    run_id: str = "slices-test",
) -> tuple[SommelierConfig, RunContext, Path]:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 2
    raw["data"]["n_validation"] = 1
    raw["data"]["n_test"] = 2
    french = dict(raw["datasets"][0])
    french["language"] = "fr"
    french["source_id_column"] = "source_example_id"
    raw["datasets"].append(french)
    raw["eval"]["slices"] = ["en", "fr"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id=run_id,
        project_root=tmp_path,
    )
    data_dir = context.run_dir / "data"
    formatted_dir = context.run_dir / "formatted"
    prepare_dataset_fixture(config, out_dir=data_dir, context=context, command=["test"])
    build_formatted_splits_fixture(
        config,
        data_dir=data_dir,
        out_dir=formatted_dir,
        context=context,
        command=["test"],
    )
    return config, context, formatted_dir


def evaluate(
    config: SommelierConfig,
    context: RunContext,
    formatted_dir: Path,
    model_kind: str,
    generator: Any,
    adapter: AdapterRef | None = None,
) -> Path:
    eval_dir = context.run_dir / "eval" / model_kind
    run_generation(
        config,
        formatted_dir=formatted_dir,
        out_dir=eval_dir,
        model_kind=model_kind,  # type: ignore[arg-type]
        context=context,
        command=["test"],
        generator=generator,
        adapter=adapter,
    )
    write_evaluation_report(
        config,
        formatted_dir=formatted_dir,
        eval_dir=eval_dir,
        model_kind=model_kind,  # type: ignore[arg-type]
        context=context,
        command=["test"],
        adapter=adapter,
    )
    return eval_dir


def test_bilingual_generation_writes_one_file_per_slice(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_bilingual_run(tmp_path)
    eval_dir = evaluate(config, context, formatted_dir, "base", LanguageAwareGenerator())

    for slice_language in ("en", "fr"):
        records = [
            json.loads(line)
            for line in (eval_dir / f"generations.{slice_language}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(records) == 2
        assert all(record["language"] == slice_language for record in records)


def test_bilingual_report_carries_per_slice_and_overall_metrics(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_bilingual_run(tmp_path)
    eval_dir = evaluate(config, context, formatted_dir, "base", LanguageAwareGenerator())

    report = json.loads((eval_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert set(report["slices"]) == {"en", "fr"}
    assert report["slices"]["en"]["metrics"]["valid_json_rate"]["value"] == 1.0
    assert report["slices"]["fr"]["metrics"]["valid_json_rate"]["value"] == 0.0
    # Overall pools both slices: 2 valid of 4.
    assert report["metrics"]["valid_json_rate"]["value"] == 0.5
    assert report["metrics"]["valid_json_rate"]["denominator"] == 4
    assert (
        report["slices"]["en"]["prompt_set_sha256"]
        != report["slices"]["fr"]["prompt_set_sha256"]
    )


def test_language_gaps_measure_fr_against_en(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_bilingual_run(tmp_path)
    base_dir = evaluate(config, context, formatted_dir, "base", LanguageAwareGenerator())
    adapter_dir = evaluate(
        config,
        context,
        formatted_dir,
        "adapter",
        FixedGenerator(GOOD_CALL),
        adapter=AdapterRef(source="abdelstark/example-adapter", revision="v1.0"),
    )
    out_dir = context.run_dir / "report"
    compare_evaluations(base_dir, adapter_dir, out_dir, command=["test"])

    comparison = json.loads((out_dir / "comparison_report.json").read_text(encoding="utf-8"))
    gaps = comparison["language_gaps"]
    assert gaps["reference"] == "en"
    # Base: fr answers prose while en answers correctly, so the gap is -1.
    assert gaps["base"]["fr"]["valid_json_rate"] == -1.0
    # Adapter: both slices answer correctly, so the gap closes to 0.
    assert gaps["adapter"]["fr"]["valid_json_rate"] == 0.0
    assert comparison["slices"]["fr"]["deltas"]["valid_json_rate"] == 1.0
    assert comparison["adapter"]["adapter_source"] == {
        "source": "abdelstark/example-adapter",
        "revision": "v1.0",
        "kind": "huggingface_repo",
    }
    markdown = (out_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert "## Language Gaps" in markdown
    assert "`fr` minus `en`" in markdown


def test_zero_row_slice_is_an_error(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_bilingual_run(tmp_path)
    test_path = formatted_dir / "test.jsonl"
    english_only = [
        line
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["language"] == "en"
    ]
    test_path.write_text("".join(line + "\n" for line in english_only), encoding="utf-8")

    with pytest.raises(UserInputError, match="eval.slices includes fr"):
        run_generation(
            config,
            formatted_dir=formatted_dir,
            out_dir=context.run_dir / "eval" / "base",
            model_kind="base",
            context=context,
            command=["test"],
            generator=FixedGenerator(GOOD_CALL),
        )


def test_comparison_rejects_differing_slice_sets(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_bilingual_run(tmp_path)
    base_dir = evaluate(config, context, formatted_dir, "base", LanguageAwareGenerator())
    adapter_dir = evaluate(
        config, context, formatted_dir, "adapter", FixedGenerator(GOOD_CALL)
    )
    report_path = adapter_dir / "evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["slices"]["fr"]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(EvaluationError, match="slices differ"):
        compare_evaluations(
            base_dir, adapter_dir, context.run_dir / "report", command=["test"]
        )


def test_comparison_rejects_per_slice_prompt_mismatch(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_bilingual_run(tmp_path)
    base_dir = evaluate(config, context, formatted_dir, "base", LanguageAwareGenerator())
    adapter_dir = evaluate(
        config, context, formatted_dir, "adapter", FixedGenerator(GOOD_CALL)
    )
    report_path = adapter_dir / "evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["slices"]["fr"]["prompt_set_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(EvaluationError, match="prompt_set_sha256 for slice fr"):
        compare_evaluations(
            base_dir, adapter_dir, context.run_dir / "report", command=["test"]
        )


def test_comparison_rejects_v1_reports(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_bilingual_run(tmp_path)
    base_dir = evaluate(config, context, formatted_dir, "base", LanguageAwareGenerator())
    adapter_dir = evaluate(
        config, context, formatted_dir, "adapter", FixedGenerator(GOOD_CALL)
    )
    report_path = base_dir / "evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema_version"] = "sommelier.evaluation_report.v1"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="evaluation_report.v2"):
        compare_evaluations(
            base_dir, adapter_dir, context.run_dir / "report", command=["test"]
        )


def test_adapter_source_recorded_in_manifest_details(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_bilingual_run(tmp_path)
    evaluate(
        config,
        context,
        formatted_dir,
        "adapter",
        FixedGenerator(GOOD_CALL),
        adapter=AdapterRef(source="abdelstark/example-adapter", revision="v1.0"),
    )
    manifest = json.loads((context.run_dir / "eval_manifest.json").read_text(encoding="utf-8"))
    assert manifest["details"]["adapter_source"]["source"] == "abdelstark/example-adapter"
    assert manifest["details"]["adapter_source"]["revision"] == "v1.0"
    assert manifest["details"]["eval_slices"] == ["en", "fr"]
