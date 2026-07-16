from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

import sommelier.training.qlora_preflight as preflight
from sommelier.config import SommelierConfig, load_config
from sommelier.errors import InvariantViolation, UserInputError
from sommelier.formatting.chat import FORMATTED_EXAMPLE_SCHEMA
from sommelier.remote.images import PIPELINE_RUNTIME_VERSIONS
from sommelier.training.collators import CompletionOnlyCollator
from sommelier.training.qlora_preflight import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    EMPTY_GIT_STATUS_SHA256,
    EVAL_ROWS,
    EXPECTED_EVAL_FORWARD_BATCHES,
    EXPECTED_TRAIN_MICROBATCHES,
    GPU_ALLOCATION,
    MAX_SEQUENCE_LENGTH,
    MIN_SYNTHETIC_SEQUENCE_TOKENS,
    OPTIMIZER_STEPS,
    TARGET_MODULES,
    TOKENIZER_REVISION,
    TRAIN_ROWS,
    SourceProvenance,
    artifact_digests,
    build_synthetic_formatted_splits,
    preflight_contract,
    run_qlora_shape_preflight,
    validate_artifact_manifest,
    validate_config_yaml_identity,
    validate_preflight_artifacts,
    validate_preflight_config,
    validate_preflight_report,
    validate_run_id,
    validate_runtime_versions,
    validate_single_cuda_device_map,
    validate_source_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_CONFIG_PATH = REPO_ROOT / "examples/config.v3-he-full.yaml"


class CharacterTokenizer:
    pad_token_id: int | None = 0
    eos_token_id: int | None = 1
    pad_token: str | None = "<pad>"
    eos_token: str | None = "<eos>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        # Four-codepoint chunks keep the fake fast while making each repeated
        # phrase fine-grained enough to exercise the 4080-4096 token window.
        return list(range((len(text) + 3) // 4))

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        if tokenize:
            raise AssertionError("the preflight must request rendered text")
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in conversation
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def save_pretrained(self, save_directory: str) -> object:
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        return (str(destination),)


def _config() -> SommelierConfig:
    return load_config(FULL_CONFIG_PATH)


def _source(
    *,
    clean: bool = True,
    boundary: str = "test launcher provenance",
) -> SourceProvenance:
    status = b"" if clean else b" M sommelier/training/qlora_preflight.py\0"
    return SourceProvenance(
        git_commit="a" * 40,
        working_tree_clean=clean,
        git_status_sha256=hashlib.sha256(status).hexdigest(),
        boundary=boundary,
    )


def test_contract_binds_exact_full_shape_without_dataset_or_provider_access() -> None:
    contract = preflight_contract()

    assert contract["base_model_id"] == BASE_MODEL_ID
    assert contract["base_model_revision"] == BASE_MODEL_REVISION
    assert contract["tokenizer_revision"] == TOKENIZER_REVISION
    assert contract["gpu_allocation"] == GPU_ALLOCATION
    assert contract["max_sequence_length"] == MAX_SEQUENCE_LENGTH
    assert contract["per_device_train_batch_size"] == 4
    assert contract["per_device_eval_batch_size"] == 4
    assert contract["gradient_accumulation_steps"] == 4
    assert contract["optimizer_steps_exercised"] == OPTIMIZER_STEPS
    assert contract["train_microbatches_exercised"] == EXPECTED_TRAIN_MICROBATCHES
    assert contract["eval_forward_batches_exercised"] == EXPECTED_EVAL_FORWARD_BATCHES
    assert contract["quantization"] == {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "double_quant": True,
    }
    assert cast(dict[str, object], contract["lora"])["target_modules"] == list(TARGET_MODULES)
    assert contract["dataset_access_required"] is False
    assert contract["provider_access_required"] is False
    assert contract["release_evidence_eligible"] is False


def test_full_hebrew_config_matches_preflight_contract() -> None:
    validate_preflight_config(_config())
    validate_config_yaml_identity(
        _config(),
        FULL_CONFIG_PATH.read_text(encoding="utf-8"),
    )


def test_input_yaml_must_resolve_to_the_exact_supplied_config() -> None:
    different_yaml = FULL_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "sommelier-v3-he-full",
        "different-project",
    )

    with pytest.raises(InvariantViolation, match="does not resolve to the supplied config"):
        validate_config_yaml_identity(_config(), different_yaml)


@pytest.mark.parametrize(
    "field",
    [
        "system_prompt",
        "base_model_revision",
        "batch_size",
        "gradient_accumulation",
        "sequence_length",
        "target_modules",
        "gpu",
    ],
)
def test_config_drift_is_rejected_before_optional_runtime_imports(field: str) -> None:
    config = _config().model_copy(deep=True)
    if field == "system_prompt":
        config.formatting.system_prompt = "different prompt"
    elif field == "base_model_revision":
        config.model.base_model_revision = "b" * 40
    elif field == "batch_size":
        config.train.per_device_batch_size = 8
    elif field == "gradient_accumulation":
        config.train.gradient_accumulation_steps = 2
    elif field == "sequence_length":
        config.train.max_sequence_length = 2048
    elif field == "target_modules":
        config.train.target_modules = ["q_proj"]
    elif field == "gpu":
        config.remote.gpu = "A100"
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(field)

    with pytest.raises(InvariantViolation, match="config drift"):
        validate_preflight_config(config)


@pytest.mark.parametrize(
    "run_id",
    ["", "../escape", "/absolute", " space", "name/child", "x" * 129],
)
def test_run_id_rejects_unsafe_or_ambiguous_paths(run_id: str) -> None:
    with pytest.raises(UserInputError, match="invalid QLoRA preflight run id"):
        validate_run_id(run_id)


def test_run_id_accepts_bounded_ascii_artifact_name() -> None:
    assert validate_run_id("he-v3.l40s_shape-001") == "he-v3.l40s_shape-001"


def test_source_provenance_accepts_consistent_clean_and_dirty_measurements() -> None:
    validate_source_provenance(_source(clean=True))
    validate_source_provenance(_source(clean=False))


@pytest.mark.parametrize(
    "source",
    [
        SourceProvenance(
            git_commit="unknown",
            working_tree_clean=True,
            git_status_sha256=EMPTY_GIT_STATUS_SHA256,
            boundary="test",
        ),
        SourceProvenance(
            git_commit="a" * 40,
            working_tree_clean=True,
            git_status_sha256="not-a-sha",
            boundary="test",
        ),
        SourceProvenance(
            git_commit="a" * 40,
            working_tree_clean=False,
            git_status_sha256=EMPTY_GIT_STATUS_SHA256,
            boundary="test",
        ),
        SourceProvenance(
            git_commit="a" * 40,
            working_tree_clean=True,
            git_status_sha256="b" * 64,
            boundary="test",
        ),
    ],
)
def test_source_provenance_rejects_unbound_or_inconsistent_measurements(
    source: SourceProvenance,
) -> None:
    with pytest.raises(UserInputError):
        validate_source_provenance(source)


def test_runtime_gate_rejects_any_pinned_distribution_drift() -> None:
    observed = dict(PIPELINE_RUNTIME_VERSIONS)
    observed["transformers"] = "drifted"

    with pytest.raises(InvariantViolation, match="transformers='drifted'"):
        validate_runtime_versions(observed)


def test_synthetic_splits_are_paired_hashed_and_near_4096_tokens() -> None:
    tokenizer = CharacterTokenizer()
    config = _config()

    splits, token_ledger = build_synthetic_formatted_splits(tokenizer, config)

    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": TRAIN_ROWS,
        "validation": EVAL_ROWS,
    }
    assert len(token_ledger) == TRAIN_ROWS + EVAL_ROWS
    assert {str(entry["language"]) for entry in token_ledger} == {"en", "he"}
    assert all(
        MIN_SYNTHETIC_SEQUENCE_TOKENS <= cast(int, entry["full_tokens"]) <= MAX_SEQUENCE_LENGTH
        for entry in token_ledger
    )
    assert all(cast(int, entry["target_tokens"]) > 0 for entry in token_ledger)

    all_rows = [*splits["train"], *splits["validation"]]
    assert len({str(row["example_id"]) for row in all_rows}) == len(all_rows)
    for row in all_rows:
        prompt = str(row["prompt_text"])
        full_text = str(row["full_text"])
        assert row["schema_version"] == FORMATTED_EXAMPLE_SCHEMA
        assert row["tokenizer_id"] == BASE_MODEL_ID
        assert row["tokenizer_revision"] == TOKENIZER_REVISION
        assert row["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert full_text.startswith(prompt)
        assert str(row["target_text"]) in full_text[len(prompt) :]

    train_sources = {str(row["source_example_id"]) for row in splits["train"]}
    validation_sources = {str(row["source_example_id"]) for row in splits["validation"]}
    assert train_sources.isdisjoint(validation_sources)
    for rows in splits.values():
        paired_languages: dict[str, set[str]] = {}
        for row in rows:
            paired_languages.setdefault(str(row["source_example_id"]), set()).add(
                str(row["language"])
            )
        assert all(languages == {"en", "he"} for languages in paired_languages.values())

    collator = CompletionOnlyCollator(
        cast(Any, tokenizer),
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    batch = collator(splits["validation"])
    assert len(batch["input_ids"]) == EVAL_ROWS
    assert max(len(row) for row in batch["input_ids"]) <= MAX_SEQUENCE_LENGTH
    assert all(any(label != -100 for label in labels) for labels in batch["labels"])


def test_synthetic_split_rejects_duplicate_rows_hidden_by_language_sets() -> None:
    splits, token_ledger = build_synthetic_formatted_splits(CharacterTokenizer(), _config())
    train_rows = splits["train"]
    pair_zero = str(train_rows[0]["source_example_id"])
    pair_one = str(train_rows[1]["source_example_id"])
    for row in train_rows:
        if row["source_example_id"] == pair_one:
            row["source_example_id"] = pair_zero

    assert sum(row["source_example_id"] == pair_zero for row in train_rows) == 4
    with pytest.raises(InvariantViolation, match="pair count drift"):
        preflight._validate_synthetic_formatted_splits(splits, token_ledger)


class _FakeParameter:
    def __init__(self, count: int, *, requires_grad: bool) -> None:
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._count


class _FakeModel:
    def __init__(self, capture: dict[str, Any]) -> None:
        self.capture = capture
        self.config = SimpleNamespace(use_cache=True)
        self.is_gradient_checkpointing = False
        self.hf_device_map = {"": 0}

    def named_modules(self) -> list[tuple[str, object]]:
        return [
            (f"model.layers.0.{name}", SimpleNamespace(lora_A=object())) for name in TARGET_MODULES
        ]

    def parameters(self) -> list[_FakeParameter]:
        return [
            _FakeParameter(1_000, requires_grad=False),
            _FakeParameter(100, requires_grad=True),
        ]

    def save_pretrained(self, directory: str, *, safe_serialization: bool) -> None:
        self.capture["safe_serialization"] = safe_serialization
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "adapter_model.safetensors").write_bytes(b"synthetic-adapter")
        (destination / "adapter_config.json").write_text("{}\n", encoding="utf-8")


class _FakeCuda:
    def __init__(self, *, device_count: int = 1, device_name: str = "NVIDIA L40S") -> None:
        self._device_count = device_count
        self._device_name = device_name

    def is_available(self) -> bool:
        return True

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self._device_name

    def get_device_properties(self, index: int) -> object:
        assert index == 0
        return SimpleNamespace(total_memory=48 * 1024 * 1024 * 1024)

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (8, 9)

    def device_count(self) -> int:
        return self._device_count

    def reset_peak_memory_stats(self) -> None:
        return None

    def max_memory_allocated(self) -> int:
        return 31 * 1024 * 1024

    def max_memory_reserved(self) -> int:
        return 37 * 1024 * 1024

    def synchronize(self) -> None:
        return None


@pytest.mark.parametrize(
    "device_map",
    [
        {"": "cpu"},
        {"": "disk"},
        {"": 1},
        {"layer.0": 0, "layer.1": "cuda:1"},
        {},
        None,
    ],
)
def test_preflight_rejects_unproved_or_offloaded_device_maps(device_map: object) -> None:
    model = SimpleNamespace(hf_device_map=device_map)
    with pytest.raises(InvariantViolation, match="hf_device_map|non-CUDA-0"):
        validate_single_cuda_device_map(model)


@pytest.mark.parametrize("placement", [0, "0", "cuda", "cuda:0"])
def test_preflight_accepts_only_cuda_zero_device_map_encodings(placement: object) -> None:
    assert validate_single_cuda_device_map(SimpleNamespace(hf_device_map={"": placement})) == {
        "": "cuda:0"
    }


@pytest.mark.parametrize(
    ("cuda", "message"),
    [
        (_FakeCuda(device_count=2), "exactly one visible GPU"),
        (_FakeCuda(device_name="NVIDIA A100"), "expected L40S"),
    ],
)
def test_hardware_gate_rejects_multi_gpu_or_non_l40s(cuda: _FakeCuda, message: str) -> None:
    torch = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda="12.8"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90_100)),
    )
    with pytest.raises(InvariantViolation, match=message):
        preflight._hardware_metadata(torch)


def _install_fake_training_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], _FakeModel]:
    capture: dict[str, Any] = {"tensor_shapes": []}
    tokenizer = CharacterTokenizer()
    model = _FakeModel(capture)

    torch_module = ModuleType("torch")
    fake_cuda = _FakeCuda()

    def tensor(data: list[list[int]], *, dtype: object) -> object:
        cast(list[tuple[int, int]], capture["tensor_shapes"]).append((len(data), len(data[0])))
        return {"data": data, "dtype": dtype}

    setattr(torch_module, "cuda", fake_cuda)
    setattr(torch_module, "bfloat16", "bfloat16")
    setattr(torch_module, "long", "long")
    setattr(torch_module, "tensor", tensor)
    setattr(torch_module, "version", SimpleNamespace(cuda="12.8"))
    setattr(
        torch_module,
        "backends",
        SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90_100)),
    )

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> CharacterTokenizer:
            capture["tokenizer_load"] = {"model_id": model_id, **kwargs}
            return tokenizer

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeModel:
            capture["model_load"] = {"model_id": model_id, **kwargs}
            return model

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            capture["quantization"] = kwargs

    class FakeTrainingArguments:
        def __init__(self, **kwargs: object) -> None:
            capture["training_arguments"] = kwargs

    class FakeTrainer:
        def __init__(
            self,
            *,
            model: object,
            args: object,
            train_dataset: list[dict[str, object]],
            eval_dataset: list[dict[str, object]],
            data_collator: Callable[[list[dict[str, object]]], dict[str, object]],
        ) -> None:
            del model, args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.data_collator = data_collator
            self.state = SimpleNamespace(global_step=0, log_history=[])

        def train(self) -> object:
            for start in range(0, len(self.train_dataset), 4):
                self.data_collator(self.train_dataset[start : start + 4])
            self.state.global_step = 1
            self.state.log_history = [{"loss": 0.5, "step": 1}]
            return SimpleNamespace(metrics={"train_loss": 0.5})

        def evaluate(self) -> dict[str, float]:
            self.data_collator(self.eval_dataset)
            return {"eval_loss": 0.25}

    transformers_module = ModuleType("transformers")
    for name, value in {
        "AutoModelForCausalLM": FakeAutoModelForCausalLM,
        "AutoTokenizer": FakeAutoTokenizer,
        "BitsAndBytesConfig": FakeBitsAndBytesConfig,
        "Trainer": FakeTrainer,
        "TrainingArguments": FakeTrainingArguments,
    }.items():
        setattr(transformers_module, name, value)

    class FakeLoraConfig:
        def __init__(self, **kwargs: object) -> None:
            capture["lora"] = kwargs

    def prepare_model_for_kbit_training(
        prepared_model: _FakeModel,
        **kwargs: object,
    ) -> _FakeModel:
        capture["kbit_preparation"] = kwargs
        prepared_model.is_gradient_checkpointing = True
        return prepared_model

    def get_peft_model(prepared_model: _FakeModel, _config: object) -> _FakeModel:
        capture["peft_wrapped"] = True
        return prepared_model

    peft_module = ModuleType("peft")
    setattr(peft_module, "LoraConfig", FakeLoraConfig)
    setattr(peft_module, "get_peft_model", get_peft_model)
    setattr(peft_module, "prepare_model_for_kbit_training", prepare_model_for_kbit_training)

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setattr(
        preflight,
        "collect_runtime_versions",
        lambda: dict(PIPELINE_RUNTIME_VERSIONS),
    )
    return capture, model


def test_full_orchestrator_exercises_four_microbatches_one_step_and_one_eval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, _model = _install_fake_training_stack(monkeypatch)
    config_yaml = FULL_CONFIG_PATH.read_text(encoding="utf-8")
    output_dir = tmp_path / "shape-run"

    report = run_qlora_shape_preflight(
        _config(),
        config_yaml=config_yaml,
        output_dir=output_dir,
        run_id="he-v3-shape-test",
        source=_source(),
    )

    assert report["status"] == "succeeded"
    assert report["diagnostic_only"] is True
    assert report["release_evidence_eligible"] is False
    assert report["provider_accessed"] is False
    assert report["dataset_accessed"] is False
    execution = cast(dict[str, object], report["execution"])
    assert execution["optimizer_steps"] == 1
    assert execution["train_microbatches"] == 4
    assert execution["eval_forward_batches"] == 1
    assert report["peak_gpu_memory_mib"] == {
        "allocated_mib": 31,
        "reserved_mib": 37,
    }
    model_wiring = cast(dict[str, object], report["model_wiring"])
    assert model_wiring["hf_device_map"] == {"": "cuda:0"}

    assert capture["tokenizer_load"] == {
        "model_id": BASE_MODEL_ID,
        "revision": TOKENIZER_REVISION,
        "trust_remote_code": False,
    }
    model_load = cast(dict[str, object], capture["model_load"])
    assert model_load["model_id"] == BASE_MODEL_ID
    assert model_load["revision"] == BASE_MODEL_REVISION
    assert model_load["device_map"] == "auto"
    assert model_load["trust_remote_code"] is False
    assert capture["quantization"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    }
    assert capture["lora"] == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": list(TARGET_MODULES),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    assert capture["kbit_preparation"] == {
        "use_gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    training_arguments = cast(dict[str, object], capture["training_arguments"])
    assert training_arguments["max_steps"] == 1
    assert training_arguments["per_device_train_batch_size"] == 4
    assert training_arguments["per_device_eval_batch_size"] == 4
    assert training_arguments["gradient_accumulation_steps"] == 4
    assert training_arguments["bf16"] is True
    assert training_arguments["fp16"] is False
    assert training_arguments["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert training_arguments["eval_strategy"] == "no"
    assert training_arguments["save_strategy"] == "no"
    assert capture["safe_serialization"] is True

    tensor_shapes = cast(list[tuple[int, int]], capture["tensor_shapes"])
    assert len(tensor_shapes) == 15  # three tensors for each of five observed batches
    assert all(batch_rows == 4 for batch_rows, _sequence in tensor_shapes)
    assert all(
        MIN_SYNTHETIC_SEQUENCE_TOKENS <= sequence <= MAX_SEQUENCE_LENGTH
        for _batch_rows, sequence in tensor_shapes
    )

    persisted = json.loads((output_dir / "preflight_report.json").read_text())
    assert persisted == report
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text())
    assert manifest["schema_version"] == ARTIFACT_MANIFEST_SCHEMA_VERSION
    validate_preflight_report(report)
    validate_artifact_manifest(manifest, artifact_root=output_dir)
    validate_preflight_artifacts(report, manifest, artifact_root=output_dir)
    manifest_paths = {entry["path"] for entry in manifest["artifacts"]}
    assert "config.input.yaml" in manifest_paths
    assert "formatted/train.jsonl" in manifest_paths
    assert "formatted/validation.jsonl" in manifest_paths
    assert "adapter/adapter_model.safetensors" in manifest_paths
    assert "preflight_report.json" not in manifest_paths
    assert "artifact_manifest.json" not in manifest_paths

    invalid_report = dict(report)
    invalid_report["unregistered_field"] = True
    with pytest.raises(InvariantViolation, match="unexpected unregistered_field"):
        validate_preflight_report(invalid_report)

    invalid_manifest = dict(manifest)
    invalid_manifest["artifacts"] = [*manifest["artifacts"], manifest["artifacts"][0]]
    with pytest.raises(InvariantViolation, match="unique and sorted"):
        validate_artifact_manifest(invalid_manifest)

    mismatched_report = json.loads(json.dumps(report))
    mismatched_report["artifact_hashes"]["files"] = manifest["artifacts"][:-1]
    with pytest.raises(InvariantViolation, match="disagree with the artifact manifest"):
        validate_preflight_artifacts(mismatched_report, manifest, artifact_root=output_dir)

    (output_dir / "config.input.yaml").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(InvariantViolation, match="does not match the on-disk"):
        validate_artifact_manifest(manifest, artifact_root=output_dir)


def test_config_contract_failure_is_persisted_after_output_reservation(tmp_path: Path) -> None:
    config = _config().model_copy(deep=True)
    config.train.per_device_batch_size = 8
    config_yaml = FULL_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "per_device_batch_size: 4",
        "per_device_batch_size: 8",
    )
    output_dir = tmp_path / "invalid-config-run"

    with pytest.raises(InvariantViolation, match="config drift"):
        run_qlora_shape_preflight(
            config,
            config_yaml=config_yaml,
            output_dir=output_dir,
            run_id="invalid-config-shape-test",
            source=_source(),
        )

    report = json.loads((output_dir / "preflight_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure"]["type"] == "InvariantViolation"
    assert "resolved_config" not in report
    validate_preflight_report(report)
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text())
    validate_artifact_manifest(manifest, artifact_root=output_dir)


def test_runtime_failure_is_redacted_and_persisted_before_reraise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_training_stack(monkeypatch)
    secret = "hf_abcdefghijklmnopqrstuvwxyz123456"
    monkeypatch.setenv("HF_TOKEN", secret)

    def fail_version_collection() -> dict[str, str]:
        raise RuntimeError(f"runtime probe failed with {secret}")

    monkeypatch.setattr(preflight, "collect_runtime_versions", fail_version_collection)
    output_dir = tmp_path / "failed-run"

    with pytest.raises(RuntimeError, match="runtime probe failed"):
        run_qlora_shape_preflight(
            _config(),
            config_yaml=FULL_CONFIG_PATH.read_text(encoding="utf-8"),
            output_dir=output_dir,
            run_id="failed-shape-test",
            source=_source(boundary=f"launcher at {Path.home()} with {secret}"),
        )

    report = json.loads((output_dir / "preflight_report.json").read_text())
    assert report["status"] == "failed"
    assert report["failure"]["type"] == "RuntimeError"
    assert secret not in report["failure"]["message"]
    assert "[redacted]" in report["failure"]["message"]
    assert secret not in report["source_code"]["boundary"]
    assert str(Path.home()) not in report["source_code"]["boundary"]
    assert report["provider_accessed"] is False
    assert report["dataset_accessed"] is False
    assert (output_dir / "artifact_manifest.json").is_file()


def test_existing_attempt_is_never_overwritten(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(UserInputError, match="output already exists"):
        run_qlora_shape_preflight(
            _config(),
            config_yaml=FULL_CONFIG_PATH.read_text(encoding="utf-8"),
            output_dir=output_dir,
            run_id="existing-shape-test",
            source=_source(),
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(output_dir.iterdir()) == [sentinel]


def test_empty_existing_attempt_is_never_resumed(tmp_path: Path) -> None:
    output_dir = tmp_path / "empty-existing"
    output_dir.mkdir()

    with pytest.raises(UserInputError, match="output already exists"):
        run_qlora_shape_preflight(
            _config(),
            config_yaml=FULL_CONFIG_PATH.read_text(encoding="utf-8"),
            output_dir=output_dir,
            run_id="empty-existing-shape-test",
            source=_source(),
        )

    assert list(output_dir.iterdir()) == []


def test_artifact_manifest_refuses_symbolic_links(tmp_path: Path) -> None:
    source = tmp_path / "outside.txt"
    source.write_text("outside", encoding="utf-8")
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "linked.txt").symlink_to(source)

    with pytest.raises(InvariantViolation, match="symbolic link"):
        artifact_digests(root)
