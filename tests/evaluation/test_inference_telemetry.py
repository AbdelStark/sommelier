from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from sommelier.config import SommelierConfig, load_config
from sommelier.data.prepare import prepare_dataset_fixture
from sommelier.errors import EvaluationError
from sommelier.evaluation.generate import (
    INFERENCE_TELEMETRY_FILENAME,
    INFERENCE_TELEMETRY_SCHEMA,
    DecodingConfig,
    run_generation,
)
from sommelier.evaluation.report import write_evaluation_report
from sommelier.formatting.chat import build_formatted_splits_fixture
from sommelier.run_context import RunContext, ensure_run_context

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
GOOD_CALL = '{"arguments":{"city":"Paris"},"name":"lookup_weather"}'


class LanguageAwareGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt_text: str, *, decoding: DecodingConfig) -> str:
        self.prompts.append(prompt_text)
        if "demande" in prompt_text:
            return "Je ne peux pas aider."
        return GOOD_CALL


class SequenceClock:
    def __init__(self, values: Iterator[float]) -> None:
        self._values = values
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self._values)


def _setup_bilingual_run(
    tmp_path: Path,
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
        run_id="telemetry-test",
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


def _run_with_deterministic_timings(
    tmp_path: Path,
) -> tuple[
    SommelierConfig,
    RunContext,
    Path,
    Path,
    LanguageAwareGenerator,
    SequenceClock,
]:
    config, context, formatted_dir = _setup_bilingual_run(tmp_path)
    eval_dir = context.run_dir / "eval" / "base"
    # Two examples per slice. English generate calls take 1 s and 3 s;
    # French calls take 2 s and 4 s. Gaps between calls are not counted.
    clock = SequenceClock(iter([0.0, 1.0, 10.0, 13.0, 20.0, 22.0, 30.0, 34.0]))
    generator = LanguageAwareGenerator()
    run_generation(
        config,
        formatted_dir=formatted_dir,
        out_dir=eval_dir,
        model_kind="base",
        context=context,
        command=["test"],
        generator=generator,
        clock=clock,
    )
    return config, context, formatted_dir, eval_dir, generator, clock


def test_sequential_generation_writes_per_language_aggregate_telemetry(
    tmp_path: Path,
) -> None:
    config, context, formatted_dir, eval_dir, generator, clock = _run_with_deterministic_timings(
        tmp_path
    )

    telemetry = json.loads((eval_dir / INFERENCE_TELEMETRY_FILENAME).read_text(encoding="utf-8"))
    assert telemetry["schema_version"] == INFERENCE_TELEMETRY_SCHEMA
    assert telemetry["run_id"] == context.run_id
    assert telemetry["model_kind"] == "base"
    assert telemetry["measurement"] == {
        "scope": "generator.generate_end_to_end_call_wall_time",
        "aggregation": "sum_of_per_example_call_intervals",
        "clock": "monotonic_seconds",
        "model_load_included": False,
        "parsing_and_artifact_io_included": False,
    }
    expected_operations = [
        "prompt_tokenization",
        "input_device_transfer",
        "model_generate",
        "generated_token_decode",
    ]
    assert telemetry["timed_call_contract"] == {
        "callable": "TextGenerator.generate",
        "default_transformers_implementation_includes": expected_operations,
        "explicit_device_synchronization": False,
    }
    assert telemetry["warmup"] == {
        "calls": 1,
        "timed": False,
        "prompt_source": "first_example_in_first_configured_slice",
        "output_disposition": "discarded",
        "call_scope": "generator.generate_end_to_end_call_wall_time",
        "default_transformers_implementation_includes": expected_operations,
        "uses_measured_decoding": True,
    }
    assert telemetry["sequential_run"] == {
        "boundary": "single_run_generation_invocation_after_model_load",
        "concurrency": 1,
        "single_model_instance": True,
        "slice_order": ["en", "fr"],
        "example_order": "formatted_test_order_within_slice",
    }
    assert telemetry["hardware"] == {
        "gpu_label": config.remote.gpu,
        "gpu_count": 1,
        "source": "config.remote.gpu",
    }
    assert telemetry["slices"]["en"]["examples"] == 2
    assert telemetry["slices"]["en"]["elapsed_seconds"] == 4.0
    assert telemetry["slices"]["en"]["seconds_per_example"] == 2.0
    assert telemetry["slices"]["fr"]["examples"] == 2
    assert telemetry["slices"]["fr"]["elapsed_seconds"] == 6.0
    assert telemetry["slices"]["fr"]["seconds_per_example"] == 3.0
    assert telemetry["total"] == {
        "examples": 4,
        "elapsed_seconds": 10.0,
        "seconds_per_example": 2.5,
    }
    assert telemetry["slices"]["en"]["generation_artifact"]["sha256"]

    formatted = [
        json.loads(line)
        for line in (formatted_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    measured_prompts = [
        str(row["prompt_text"])
        for language in config.eval.slices
        for row in formatted
        if row["language"] == language
    ]
    assert generator.prompts == [measured_prompts[0], *measured_prompts]
    assert clock.calls == 2 * len(measured_prompts)

    manifest = json.loads((context.run_dir / "eval-base_manifest.json").read_text(encoding="utf-8"))
    assert manifest["details"]["execution_mode"] == "sequential"
    assert manifest["details"]["inference_telemetry"].endswith(INFERENCE_TELEMETRY_FILENAME)
    assert {output["schema_version"] for output in manifest["outputs"]} == {
        "sommelier.generation.v2",
        INFERENCE_TELEMETRY_SCHEMA,
    }


def test_evaluation_report_derives_gpu_seconds_and_marks_zero_success(
    tmp_path: Path,
) -> None:
    config, context, formatted_dir, eval_dir, _, _ = _run_with_deterministic_timings(tmp_path)

    write_evaluation_report(
        config,
        formatted_dir=formatted_dir,
        eval_dir=eval_dir,
        model_kind="base",
        context=context,
        command=["test"],
    )

    report = json.loads((eval_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    telemetry = json.loads((eval_dir / INFERENCE_TELEMETRY_FILENAME).read_text(encoding="utf-8"))
    efficiency = report["inference_efficiency"]
    assert efficiency["available"] is True
    assert efficiency["timed_call_contract"] == telemetry["timed_call_contract"]
    assert efficiency["warmup"] == telemetry["warmup"]
    assert efficiency["telemetry_artifact"]["schema_version"] == (INFERENCE_TELEMETRY_SCHEMA)
    english = efficiency["slices"]["en"]
    assert english["gpu_seconds_per_full_call_exact_success"] == {
        "available": True,
        "value": 2.0,
        "reason": None,
        "unit": "gpu_seconds_per_full_call_exact_success",
        "full_call_exact_successes": 2,
        "basis": "generation_elapsed_seconds_x_configured_gpu_count",
    }
    french = efficiency["slices"]["fr"]
    assert french["gpu_seconds_per_full_call_exact_success"] == {
        "available": False,
        "value": None,
        "reason": "zero_full_call_exact_successes",
        "unit": "gpu_seconds_per_full_call_exact_success",
        "full_call_exact_successes": 0,
        "basis": "generation_elapsed_seconds_x_configured_gpu_count",
    }
    assert efficiency["overall"]["gpu_seconds_per_full_call_exact_success"]["value"] == 5.0
    assert "cost" not in efficiency


def test_evaluation_report_remains_compatible_without_telemetry(tmp_path: Path) -> None:
    config, context, formatted_dir, eval_dir, _, _ = _run_with_deterministic_timings(tmp_path)
    (eval_dir / INFERENCE_TELEMETRY_FILENAME).unlink()

    write_evaluation_report(
        config,
        formatted_dir=formatted_dir,
        eval_dir=eval_dir,
        model_kind="base",
        context=context,
        command=["test"],
    )

    report = json.loads((eval_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    assert report["inference_efficiency"] == {
        "available": False,
        "reason": "inference_telemetry_artifact_missing",
    }


def test_evaluation_report_rejects_telemetry_from_different_generations(
    tmp_path: Path,
) -> None:
    config, context, formatted_dir, eval_dir, _, _ = _run_with_deterministic_timings(tmp_path)
    telemetry_path = eval_dir / INFERENCE_TELEMETRY_FILENAME
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry["slices"]["en"]["examples"] = 1
    telemetry_path.write_text(json.dumps(telemetry, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(EvaluationError, match="example count for en"):
        write_evaluation_report(
            config,
            formatted_dir=formatted_dir,
            eval_dir=eval_dir,
            model_kind="base",
            context=context,
            command=["test"],
        )


def test_evaluation_report_rejects_missing_timed_call_contract(
    tmp_path: Path,
) -> None:
    config, context, formatted_dir, eval_dir, _, _ = _run_with_deterministic_timings(tmp_path)
    telemetry_path = eval_dir / INFERENCE_TELEMETRY_FILENAME
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    del telemetry["timed_call_contract"]
    telemetry_path.write_text(json.dumps(telemetry, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(EvaluationError, match="timed-call contract"):
        write_evaluation_report(
            config,
            formatted_dir=formatted_dir,
            eval_dir=eval_dir,
            model_kind="base",
            context=context,
            command=["test"],
        )


def test_evaluation_report_rejects_tampered_warmup_contract(
    tmp_path: Path,
) -> None:
    config, context, formatted_dir, eval_dir, _, _ = _run_with_deterministic_timings(tmp_path)
    telemetry_path = eval_dir / INFERENCE_TELEMETRY_FILENAME
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry["warmup"]["calls"] = 0
    telemetry_path.write_text(json.dumps(telemetry, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(EvaluationError, match="warmup contract"):
        write_evaluation_report(
            config,
            formatted_dir=formatted_dir,
            eval_dir=eval_dir,
            model_kind="base",
            context=context,
            command=["test"],
        )
