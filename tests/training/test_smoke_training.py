from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml

from sommelier.config import load_config
from sommelier.run_context import RunContext, ensure_run_context
from sommelier.training.collators import CompletionOnlyCollator
from sommelier.training.metrics import TrainingResult
from sommelier.training.qlora import train_adapter

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "training" / "formatted"
EXAMPLES_DIR = REPO_ROOT / "examples"


class CharTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(char) for char in text]


class OneStepTrainer:
    """Stubbed one-step path: one collated forward batch, then save.

    Exercises the same collator the real QLoRA backend uses, so the fixture
    proves the prompt-boundary contract end to end without GPU packages.
    """

    def __init__(self) -> None:
        self.batches: list[dict[str, object]] = []

    def train(
        self,
        train_examples: list[dict[str, object]],
        validation_examples: list[dict[str, object]],
        adapter_dir: Path,
    ) -> TrainingResult:
        collator = CompletionOnlyCollator(CharTokenizer(), max_sequence_length=512)
        batch = collator(train_examples)
        self.batches.append(dict(batch))
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"one-step-weights")
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"r": 16, "step": 1}), encoding="utf-8"
        )
        return TrainingResult(
            history=[
                {
                    "step": 1,
                    "epoch": 1.0,
                    "loss": 2.5,
                    "learning_rate": 2e-4,
                    "num_input_tokens_seen": sum(
                        sum(mask) for mask in batch["attention_mask"]
                    ),
                },
            ],
            peak_gpu_memory_mb=None,
        )


def setup_run(tmp_path: Path) -> tuple[RunContext, Path]:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id="smoke-train",
        project_root=tmp_path,
    )
    formatted_dir = context.run_dir / "formatted"
    formatted_dir.mkdir(parents=True)
    for split in ("train", "validation"):
        shutil.copy(FIXTURE_DIR / f"{split}.jsonl", formatted_dir / f"{split}.jsonl")
    return context, formatted_dir


def test_fixture_is_internally_coherent() -> None:
    for split in ("train", "validation"):
        for line in (FIXTURE_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert record["schema_version"] == "sommelier.formatted_example.v2"
            prompt_text = record["prompt_text"]
            assert record["full_text"].startswith(prompt_text)
            assert record["target_text"] in record["full_text"]
            expected = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            assert record["prompt_sha256"] == expected


def test_one_step_smoke_writes_adapter_and_manifest(tmp_path: Path) -> None:
    context, formatted_dir = setup_run(tmp_path)
    config = load_config(tmp_path / "config.yaml")
    adapter_dir = context.run_dir / "train" / "adapter"
    trainer = OneStepTrainer()

    manifest = train_adapter(
        config,
        formatted_dir,
        adapter_dir,
        context=context,
        command=["smoke"],
        trainer=trainer,
    )

    assert (adapter_dir / "adapter_model.safetensors").exists()
    assert (adapter_dir / "adapter_config.json").exists()

    manifest_path = context.run_dir / "train_manifest.json"
    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["stage"] == "train"
    assert on_disk["status"] == "succeeded"
    assert on_disk == json.loads(json.dumps(manifest))

    metrics_path = context.run_dir / "train" / "training_metrics.jsonl"
    assert metrics_path.exists()
    metric = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert metric["step"] == 1
    assert metric["tokens_seen"] > 0

    run_manifest = json.loads(
        (context.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert "train" in run_manifest["stages"]


def test_one_step_batch_masks_prompts_for_fixture(tmp_path: Path) -> None:
    context, formatted_dir = setup_run(tmp_path)
    config = load_config(tmp_path / "config.yaml")
    trainer = OneStepTrainer()

    train_adapter(
        config,
        formatted_dir,
        context.run_dir / "train" / "adapter",
        context=context,
        command=["smoke"],
        trainer=trainer,
    )

    batch = trainer.batches[0]
    labels = batch["labels"]
    fixture = [
        json.loads(line)
        for line in (FIXTURE_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert isinstance(labels, list)
    for row, record in zip(labels, fixture, strict=True):
        prompt_length = len(record["prompt_text"])
        assert all(label == -100 for label in row[:prompt_length])
        assert any(label != -100 for label in row[prompt_length:])
