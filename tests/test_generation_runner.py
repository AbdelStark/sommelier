from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from sommelier.config import SommelierConfig, load_config
from sommelier.data.prepare import prepare_dataset_fixture
from sommelier.errors import (
    EvaluationError,
    ExternalDependencyError,
    UserInputError,
)
from sommelier.evaluation.generate import (
    AdapterRef,
    DecodingConfig,
    load_model_generator,
    run_generation,
    validate_decoding,
)
from sommelier.formatting.chat import build_formatted_splits_fixture
from sommelier.run_context import RunContext, ensure_run_context

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


class EchoTargetGenerator:
    """Returns the canonical gold call for even prompts, prose for odd ones."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.decodings: list[DecodingConfig] = []
        self._responses_by_prompt: dict[str, str] = {}

    def generate(self, prompt_text: str, *, decoding: DecodingConfig) -> str:
        self.prompts.append(prompt_text)
        self.decodings.append(decoding)
        if prompt_text not in self._responses_by_prompt:
            response = (
                '{"arguments":{"city":"Paris"},"name":"lookup_weather"}'
                if not self._responses_by_prompt
                else "I cannot help with that."
            )
            self._responses_by_prompt[prompt_text] = response
        return self._responses_by_prompt[prompt_text]


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
        run_id="eval-test",
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


def test_generations_persist_with_decoding_and_parse_status(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    out_dir = context.run_dir / "eval" / "base"
    generator = EchoTargetGenerator()

    run_generation(
        config,
        formatted_dir=formatted_dir,
        out_dir=out_dir,
        model_kind="base",
        context=context,
        command=["test"],
        generator=generator,
    )

    lines = (out_dir / "generations.en.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    formatted = [
        json.loads(line)
        for line in (formatted_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == len(formatted)

    for record in records:
        assert record["schema_version"] == "sommelier.generation.v2"
        assert record["language"] == "en"
        assert record["model_kind"] == "base"
        assert record["decoding"] == {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": config.eval.max_new_tokens,
        }
        assert record["parse_status"] in {"ok", "no_json"}
        if record["parse_status"] == "ok":
            assert record["parsed_call"]["name"] == "lookup_weather"
        else:
            assert record["parsed_call"] is None
        assert record["raw_text"]

    statuses = {record["parse_status"] for record in records}
    assert statuses == {"ok", "no_json"}


def test_prompts_come_from_stored_prompt_text(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    generator = EchoTargetGenerator()

    run_generation(
        config,
        formatted_dir=formatted_dir,
        out_dir=context.run_dir / "eval" / "base",
        model_kind="base",
        context=context,
        command=["test"],
        generator=generator,
    )

    formatted = [
        json.loads(line)
        for line in (formatted_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    measured_prompts = [str(example["prompt_text"]) for example in formatted]
    assert generator.prompts == [measured_prompts[0], *measured_prompts]
    assert generator.decodings[0] == generator.decodings[1]

    records = [
        json.loads(line)
        for line in (context.run_dir / "eval" / "base" / "generations.en.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["prompt_sha256"] for record in records] == [
        example["prompt_sha256"] for example in formatted
    ]


def test_eval_manifest_recorded(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    run_generation(
        config,
        formatted_dir=formatted_dir,
        out_dir=context.run_dir / "eval" / "base",
        model_kind="base",
        context=context,
        command=["test"],
        generator=EchoTargetGenerator(),
    )

    manifest = json.loads((context.run_dir / "eval-base_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "eval-base"
    assert manifest["status"] == "succeeded"
    assert manifest["outputs"][0]["schema_version"] == "sommelier.generation.v2"
    assert manifest["details"]["eval_slices"] == ["en"]


def test_base_and_adapter_eval_manifests_do_not_overwrite_each_other(
    tmp_path: Path,
) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    for model_kind in ("base", "adapter"):
        run_generation(
            config,
            formatted_dir=formatted_dir,
            out_dir=context.run_dir / "eval" / model_kind,
            model_kind=model_kind,
            context=context,
            command=["test"],
            adapter=(
                AdapterRef(source="example/adapter", revision="a" * 40)
                if model_kind == "adapter"
                else None
            ),
            generator=EchoTargetGenerator(),
        )

    base_manifest = context.run_dir / "eval-base_manifest.json"
    adapter_manifest = context.run_dir / "eval-adapter_manifest.json"
    assert base_manifest.exists()
    assert adapter_manifest.exists()
    root = json.loads((context.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (
        root["stages"]["eval-base"] == base_manifest.relative_to(context.artifact_root).as_posix()
    )
    assert (
        root["stages"]["eval-adapter"]
        == adapter_manifest.relative_to(context.artifact_root).as_posix()
    )


def test_decoding_validation_rejects_sampling(tmp_path: Path) -> None:
    config, _, _ = setup_run(tmp_path)

    config.eval.temperature = 0.7
    with pytest.raises(EvaluationError):
        validate_decoding(config)

    config.eval.temperature = 0.0
    config.eval.do_sample = True
    with pytest.raises(EvaluationError):
        validate_decoding(config)

    config.eval.do_sample = False
    config.eval.max_new_tokens = 0
    with pytest.raises(EvaluationError):
        validate_decoding(config)


def test_adapter_requires_adapter_dir(tmp_path: Path) -> None:
    config, _, _ = setup_run(tmp_path)
    with pytest.raises(UserInputError):
        load_model_generator(config, "adapter", None)


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is not None,
    reason="transformers installed; missing-dependency path not reachable",
)
def test_base_generator_requires_model_stack(tmp_path: Path) -> None:
    config, _, _ = setup_run(tmp_path)
    with pytest.raises(ExternalDependencyError):
        load_model_generator(config, "base")
