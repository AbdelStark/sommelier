from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sommelier.config import SommelierConfig, load_config
from sommelier.data.prepare import prepare_dataset_fixture
from sommelier.errors import EvaluationError, InvariantViolation, UserInputError
from sommelier.evaluation.generate import DecodingConfig, run_generation
from sommelier.evaluation.report import (
    compare_evaluations,
    find_run_layout,
    write_evaluation_report,
)
from sommelier.formatting.chat import build_formatted_splits_fixture
from sommelier.run_context import RunContext, ensure_run_context

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

GOOD_CALL = '{"arguments":{"city":"Paris"},"name":"lookup_weather"}'


class FixedGenerator:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, prompt_text: str, *, decoding: DecodingConfig) -> str:
        return self.text


def setup_run(tmp_path: Path) -> tuple[SommelierConfig, RunContext, Path]:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 2
    raw["data"]["n_validation"] = 1
    raw["data"]["n_test"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id="report-test",
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
    text: str,
) -> Path:
    eval_dir = context.run_dir / "eval" / model_kind
    run_generation(
        config,
        formatted_dir=formatted_dir,
        out_dir=eval_dir,
        model_kind=model_kind,  # type: ignore[arg-type]
        context=context,
        command=["test"],
        generator=FixedGenerator(text),
    )
    write_evaluation_report(
        config,
        formatted_dir=formatted_dir,
        eval_dir=eval_dir,
        model_kind=model_kind,  # type: ignore[arg-type]
        context=context,
        command=["test"],
    )
    return eval_dir


def setup_comparison(tmp_path: Path) -> tuple[RunContext, Path, Path, Path]:
    config, context, formatted_dir = setup_run(tmp_path)
    base_dir = evaluate(config, context, formatted_dir, "base", "no tool call here")
    adapter_dir = evaluate(config, context, formatted_dir, "adapter", GOOD_CALL)
    out_dir = context.run_dir / "report"
    return context, base_dir, adapter_dir, out_dir


def test_evaluation_report_shape(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    eval_dir = evaluate(config, context, formatted_dir, "base", GOOD_CALL)

    report = json.loads((eval_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "sommelier.evaluation_report.v1"
    assert report["model_kind"] == "base"
    assert report["split"] == "test"
    assert report["parser_version"] == "sommelier.parser.v1"
    assert set(report["metrics"]) == {
        "valid_json_rate",
        "function_name_accuracy",
        "argument_exact_match",
        "argument_f1",
        "full_call_exact_match",
    }
    assert report["metrics"]["valid_json_rate"]["value"] == 1.0
    assert report["test_split_sha256"]
    assert report["prompt_set_sha256"]
    assert report["decoding"]["temperature"] == 0.0
    assert report["config_sha256"] == context.config_sha256


def test_comparison_writes_report_and_deltas(tmp_path: Path) -> None:
    context, base_dir, adapter_dir, out_dir = setup_comparison(tmp_path)

    compare_evaluations(base_dir, adapter_dir, out_dir, command=["test"])

    comparison = json.loads((out_dir / "comparison_report.json").read_text(encoding="utf-8"))
    assert comparison["schema_version"] == "sommelier.comparison_report.v1"
    assert comparison["base"]["metrics"]["valid_json_rate"]["value"] == 0.0
    assert comparison["adapter"]["metrics"]["valid_json_rate"]["value"] == 1.0
    assert comparison["deltas"]["valid_json_rate"] == 1.0
    assert comparison["deltas"]["function_name_accuracy"] == 1.0
    assert comparison["shared"]["parser_version"] == "sommelier.parser.v1"
    assert comparison["runtime"] == {"available": False}

    manifest = json.loads(
        (context.run_dir / "report_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stage"] == "report"
    assert manifest["status"] == "succeeded"

    markdown_path = out_dir / "comparison_report.md"
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Sommelier Comparison Report" in markdown
    assert "## Limitations" in markdown
    output_kinds = {ref["kind"] for ref in manifest["outputs"]}
    assert output_kinds == {"comparison_report", "comparison_report_markdown"}


def tamper_report(eval_dir: Path, field: str, value: object) -> None:
    path = eval_dir / "evaluation_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report[field] = value
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test_split_sha256", "0" * 64),
        ("prompt_set_sha256", "0" * 64),
        ("parser_version", "sommelier.parser.v0"),
        ("decoding", {"temperature": 0.5, "do_sample": True, "max_new_tokens": 8}),
        ("config_sha256", "0" * 64),
        ("split", "validation"),
    ],
)
def test_comparison_rejects_mismatched_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    _, base_dir, adapter_dir, out_dir = setup_comparison(tmp_path)
    tamper_report(adapter_dir, field, value)

    with pytest.raises(EvaluationError, match=field if field != "split" else "split"):
        compare_evaluations(base_dir, adapter_dir, out_dir, command=["test"])
    assert not (out_dir / "comparison_report.json").exists()


def test_comparison_rejects_swapped_model_kinds(tmp_path: Path) -> None:
    _, base_dir, adapter_dir, out_dir = setup_comparison(tmp_path)
    with pytest.raises(EvaluationError):
        compare_evaluations(adapter_dir, base_dir, out_dir, command=["test"])


def test_report_rejects_prompt_digest_mismatch(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    eval_dir = context.run_dir / "eval" / "base"
    run_generation(
        config,
        formatted_dir=formatted_dir,
        out_dir=eval_dir,
        model_kind="base",
        context=context,
        command=["test"],
        generator=FixedGenerator(GOOD_CALL),
    )
    generations_path = eval_dir / "generations.jsonl"
    lines = [
        json.loads(line) for line in generations_path.read_text(encoding="utf-8").splitlines()
    ]
    lines[0]["prompt_sha256"] = "0" * 64
    generations_path.write_text(
        "\n".join(json.dumps(line, sort_keys=True) for line in lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InvariantViolation):
        write_evaluation_report(
            config,
            formatted_dir=formatted_dir,
            eval_dir=eval_dir,
            model_kind="base",
            context=context,
            command=["test"],
        )


def test_comparison_renders_runtime_metadata_when_present(tmp_path: Path) -> None:
    from sommelier.runtime_metadata import (
        initialize_runtime_metadata,
        record_stage_runtime,
    )

    context, base_dir, adapter_dir, out_dir = setup_comparison(tmp_path)
    initialize_runtime_metadata(context.run_dir, gpu="A10G")
    record_stage_runtime(context.run_dir, stage="train", elapsed_seconds=42.0, gpu="A10G")

    compare_evaluations(base_dir, adapter_dir, out_dir, command=["test"])

    comparison = json.loads((out_dir / "comparison_report.json").read_text(encoding="utf-8"))
    runtime = comparison["runtime"]
    assert runtime["available"] is True
    assert runtime["stages"]["train"]["elapsed_seconds"] == 42.0
    assert runtime["hardware"]["gpu"] == "A10G"
    assert runtime["cost_source"] == "unavailable"
    assert runtime["observed_cost_usd"] is None


def test_find_run_layout_rejects_foreign_paths(tmp_path: Path) -> None:
    with pytest.raises(UserInputError):
        find_run_layout(tmp_path / "not-a-run")


def test_comparison_requires_out_dir_in_producing_run(tmp_path: Path) -> None:
    _, base_dir, adapter_dir, _ = setup_comparison(tmp_path)
    foreign_out = tmp_path / "artifacts" / "runs" / "other-run" / "report"
    foreign_out.mkdir(parents=True)

    with pytest.raises(UserInputError):
        compare_evaluations(base_dir, adapter_dir, foreign_out, command=["test"])
