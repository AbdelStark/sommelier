from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from weakref import WeakKeyDictionary

import pytest
import yaml

import sommelier.training.authorization as authorization_module
from sommelier.config import SommelierConfig, load_config
from sommelier.data.prepare import prepare_dataset_fixture
from sommelier.errors import ArtifactNotFoundError, ExternalDependencyError, UserInputError
from sommelier.formatting.chat import build_formatted_splits_fixture
from sommelier.run_context import RunContext, ensure_run_context
from sommelier.training.authorization import FullPairedInputValidationCapability
from sommelier.training.metrics import TrainingResult
from sommelier.training.qlora import (
    QLORA_DEVICE_MAP,
    build_default_trainer,
    configure_qlora_base_model,
    qlora_kbit_preparation_kwargs,
    qlora_lora_kwargs,
    qlora_model_load_kwargs,
    qlora_quantization_kwargs,
    qlora_tokenizer_load_kwargs,
    qlora_training_argument_kwargs,
    train_adapter,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


class StubTrainer:
    def __init__(self) -> None:
        self.train_examples: list[dict[str, object]] = []
        self.validation_examples: list[dict[str, object]] = []

    def train(
        self,
        train_examples: list[dict[str, object]],
        validation_examples: list[dict[str, object]],
        adapter_dir: Path,
    ) -> TrainingResult:
        self.train_examples = train_examples
        self.validation_examples = validation_examples
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"stub-weights")
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 16}), encoding="utf-8")
        return TrainingResult(
            history=[{"step": 1, "epoch": 1.0, "loss": 1.5, "learning_rate": 2e-4}],
            peak_gpu_memory_mb=None,
        )


class EmptyTrainer(StubTrainer):
    def train(
        self,
        train_examples: list[dict[str, object]],
        validation_examples: list[dict[str, object]],
        adapter_dir: Path,
    ) -> TrainingResult:
        return TrainingResult(history=[], peak_gpu_memory_mb=None)


def setup_run(tmp_path: Path) -> tuple[SommelierConfig, RunContext, Path]:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 2
    raw["data"]["n_validation"] = 1
    raw["data"]["n_test"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id="train-test",
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


def setup_hebrew_v3_full_run(tmp_path: Path) -> tuple[SommelierConfig, RunContext, Path]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (EXAMPLES_DIR / "config.v3-he-full.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = load_config(config_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id="hebrew-v3-direct-train",
        project_root=tmp_path,
    )
    formatted_dir = context.run_dir / "formatted"
    formatted_dir.mkdir(parents=True)
    fixture_dir = EXAMPLES_DIR.parent / "tests" / "fixtures" / "training" / "formatted"
    for split in ("train", "validation"):
        english = [
            json.loads(line)
            for line in (fixture_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        hebrew = []
        for record in english:
            translated = dict(record)
            translated["example_id"] = f"{record['example_id']}-he"
            translated["language"] = "he"
            translated["source_example_id"] = record["example_id"]
            hebrew.append(translated)
        (formatted_dir / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in [*english, *hebrew]) + "\n",
            encoding="utf-8",
        )
    return config, context, formatted_dir


def test_train_stage_saves_adapter_and_manifest(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    adapter_dir = context.run_dir / "train" / "adapter"
    trainer = StubTrainer()

    manifest = train_adapter(
        config,
        formatted_dir,
        adapter_dir,
        context=context,
        command=["test"],
        trainer=trainer,
    )

    assert (adapter_dir / "adapter_model.safetensors").exists()
    assert manifest["stage"] == "train"
    assert manifest["status"] == "succeeded"
    assert manifest["run_id"] == "train-test"

    input_kinds = [ref["kind"] for ref in manifest["inputs"]]
    assert input_kinds == ["formatted_split", "formatted_split"]
    assert all(len(ref["sha256"]) == 64 for ref in manifest["inputs"])

    output_paths = [ref["path"] for ref in manifest["outputs"]]
    assert any(path.endswith("adapter_model.safetensors") for path in output_paths)

    metrics_path = context.run_dir / "train" / "training_metrics.jsonl"
    assert metrics_path.exists()
    metrics_refs = [ref for ref in manifest["outputs"] if ref["kind"] == "training_metrics"]
    assert len(metrics_refs) == 1
    assert metrics_refs[0]["schema_version"] == "sommelier.training_metric.v1"
    metric_record = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert metric_record["train_loss"] == 1.5

    on_disk = json.loads((context.run_dir / "train_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["stage"] == "train"


def test_hebrew_v3_full_train_requires_validated_pipeline_capability(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_hebrew_v3_full_run(tmp_path)
    adapter_dir = context.run_dir / "train" / "adapter"
    trainer = StubTrainer()

    with pytest.raises(UserInputError, match="full pipeline"):
        train_adapter(
            config,
            formatted_dir,
            adapter_dir,
            context=context,
            command=["test"],
            trainer=trainer,
        )

    assert trainer.train_examples == []
    assert not adapter_dir.exists()


def test_full_paired_input_capability_cannot_be_constructed_by_library_callers() -> None:
    with pytest.raises(TypeError, match="issued internally"):
        FullPairedInputValidationCapability()


def test_full_paired_input_capability_registry_cannot_be_injected_via_module_global() -> None:
    assert not hasattr(authorization_module, "_ISSUED_CAPABILITIES")
    assert not any(
        isinstance(value, WeakKeyDictionary) for value in vars(authorization_module).values()
    )


def test_hebrew_v3_full_train_rejects_unregistered_capability(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_hebrew_v3_full_run(tmp_path)
    adapter_dir = context.run_dir / "train" / "adapter"
    trainer = StubTrainer()
    forged = object.__new__(FullPairedInputValidationCapability)

    with pytest.raises(UserInputError, match="no valid paired-input capability"):
        train_adapter(
            config,
            formatted_dir,
            adapter_dir,
            context=context,
            command=["test"],
            trainer=trainer,
            full_paired_input_validation=forged,
        )

    assert trainer.train_examples == []
    assert not adapter_dir.exists()


def test_production_qlora_setup_is_explicit_and_eval_matches_train_batch(tmp_path: Path) -> None:
    config = load_config(EXAMPLES_DIR / "config.v3-he-full.yaml")
    torch = SimpleNamespace(bfloat16="bfloat16")
    quantization = object()

    assert qlora_tokenizer_load_kwargs(config) == {
        "revision": config.model.tokenizer_revision,
        "trust_remote_code": False,
    }
    assert qlora_quantization_kwargs(torch) == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    }
    assert qlora_model_load_kwargs(config, quantization_config=quantization) == {
        "revision": config.model.base_model_revision,
        "trust_remote_code": False,
        "quantization_config": quantization,
        "device_map": QLORA_DEVICE_MAP,
    }
    assert qlora_kbit_preparation_kwargs() == {
        "use_gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    assert qlora_lora_kwargs(config) == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": config.train.target_modules,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    configure_qlora_base_model(model)
    assert model.config.use_cache is False

    arguments = qlora_training_argument_kwargs(config, output_dir=tmp_path / "trainer")
    assert arguments["per_device_train_batch_size"] == 4
    assert arguments["per_device_eval_batch_size"] == 4
    assert arguments["gradient_accumulation_steps"] == 4
    assert arguments["bf16"] is True
    assert arguments["fp16"] is False
    assert arguments["data_seed"] == config.project.seed
    assert arguments["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


def test_train_stage_feeds_train_and_validation_splits(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    trainer = StubTrainer()

    train_adapter(
        config,
        formatted_dir,
        context.run_dir / "train" / "adapter",
        context=context,
        command=["test"],
        trainer=trainer,
    )

    assert [example["split"] for example in trainer.train_examples] == ["train", "train"]
    assert [example["split"] for example in trainer.validation_examples] == ["validation"]


def setup_bilingual_run(
    tmp_path: Path,
    train_languages: list[str],
) -> tuple[SommelierConfig, RunContext, Path]:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 2
    raw["data"]["n_validation"] = 1
    raw["data"]["n_test"] = 1
    french = dict(raw["datasets"][0])
    french["language"] = "fr"
    french["source_id_column"] = "source_example_id"
    raw["datasets"].append(french)
    raw["train"]["languages"] = train_languages
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id="train-test-bilingual",
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


def test_train_stage_records_language_counts(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    manifest = train_adapter(
        config,
        formatted_dir,
        context.run_dir / "train" / "adapter",
        context=context,
        command=["test"],
        trainer=StubTrainer(),
    )
    details = manifest["details"]
    assert details["train_languages"] == ["en"]
    assert details["train_examples_by_language"] == {"en": 2}
    assert details["validation_examples_by_language"] == {"en": 1}


def test_train_stage_filters_to_configured_languages(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_bilingual_run(tmp_path, ["fr"])
    trainer = StubTrainer()
    manifest = train_adapter(
        config,
        formatted_dir,
        context.run_dir / "train" / "adapter",
        context=context,
        command=["test"],
        trainer=trainer,
    )
    assert {example["language"] for example in trainer.train_examples} == {"fr"}
    assert {example["language"] for example in trainer.validation_examples} == {"fr"}
    assert manifest["details"]["train_examples_by_language"] == {"fr": 2}


def test_train_stage_errors_on_language_with_no_rows(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_bilingual_run(tmp_path, ["en", "fr"])
    train_path = formatted_dir / "train.jsonl"
    english_only = [
        line
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["language"] == "en"
    ]
    train_path.write_text("".join(line + "\n" for line in english_only), encoding="utf-8")

    with pytest.raises(UserInputError, match="train.languages includes fr"):
        train_adapter(
            config,
            formatted_dir,
            context.run_dir / "train" / "adapter",
            context=context,
            command=["test"],
            trainer=StubTrainer(),
        )


def test_train_stage_rejects_missing_split(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    (formatted_dir / "validation.jsonl").unlink()

    with pytest.raises(ArtifactNotFoundError):
        train_adapter(
            config,
            formatted_dir,
            context.run_dir / "train" / "adapter",
            context=context,
            command=["test"],
            trainer=StubTrainer(),
        )


def test_train_stage_rejects_empty_split(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    (formatted_dir / "train.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(UserInputError):
        train_adapter(
            config,
            formatted_dir,
            context.run_dir / "train" / "adapter",
            context=context,
            command=["test"],
            trainer=StubTrainer(),
        )


def test_train_stage_rejects_trainer_without_artifacts(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)

    with pytest.raises(UserInputError):
        train_adapter(
            config,
            formatted_dir,
            context.run_dir / "train" / "adapter",
            context=context,
            command=["test"],
            trainer=EmptyTrainer(),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is not None,
    reason="transformers installed; missing-dependency path not reachable",
)
def test_default_trainer_requires_training_stack(tmp_path: Path) -> None:
    config, _, _ = setup_run(tmp_path)
    with pytest.raises(ExternalDependencyError):
        build_default_trainer(config)
