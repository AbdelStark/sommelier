from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol

from sommelier.artifacts import ArtifactRef, make_artifact_ref
from sommelier.config import SommelierConfig
from sommelier.errors import (
    ExternalDependencyError,
    ResourceError,
    SommelierError,
    UserInputError,
)
from sommelier.manifests import (
    StageManifest,
    build_stage_manifest,
    update_run_manifest,
    write_stage_manifest,
)
from sommelier.run_context import RunContext, read_jsonl_records
from sommelier.tracking import track_stage_metrics
from sommelier.training.collators import CompletionOnlyCollator
from sommelier.training.metrics import (
    METRICS_FILENAME,
    TRAINING_METRIC_SCHEMA,
    TrainingResult,
    build_training_metrics,
    measure_peak_gpu_memory_mb,
    write_training_metrics,
)

FORMATTED_SCHEMA: Final = "sommelier.formatted_example.v1"


class AdapterTrainer(Protocol):
    """The training surface the stage depends on.

    The default implementation wraps transformers/peft/trl; tests inject
    stubs so the stage contract is exercised without GPU dependencies.
    ``train`` writes adapter files into ``adapter_dir`` and returns the
    raw log history plus the peak GPU memory measurement when available.
    """

    def train(
        self,
        train_examples: list[dict[str, object]],
        validation_examples: list[dict[str, object]],
        adapter_dir: Path,
    ) -> TrainingResult: ...


def _read_split(formatted_dir: Path, split: str) -> list[dict[str, object]]:
    records = read_jsonl_records(formatted_dir / f"{split}.jsonl")
    if not records:
        raise UserInputError(
            f"formatted {split} split is empty: {formatted_dir / f'{split}.jsonl'}",
            hint="Run sommelier format build with non-empty splits before training.",
        )
    for record in records:
        if record.get("schema_version") != FORMATTED_SCHEMA:
            raise UserInputError(
                f"{formatted_dir / f'{split}.jsonl'}: expected {FORMATTED_SCHEMA} records",
                hint="Rebuild the formatted splits with the current pipeline version.",
            )
    return records


def map_resource_failure(
    error: BaseException,
    config: SommelierConfig,
) -> ResourceError | None:
    """Maps OOM and timeout failures to actionable ResourceErrors (exit 4).

    The hint names the exact config fields to change; Sommelier never
    retries with silently altered batch, sequence, or GPU settings
    (RFC-0004). Unrecognized failures return None and propagate unchanged.
    """
    lowered = str(error).lower()
    is_oom = (
        type(error).__name__ == "OutOfMemoryError" or "out of memory" in lowered
    )
    if is_oom:
        return ResourceError(
            "training ran out of GPU memory",
            hint=(
                "Current settings: "
                f"train.per_device_batch_size={config.train.per_device_batch_size}, "
                f"train.gradient_accumulation_steps="
                f"{config.train.gradient_accumulation_steps}, "
                f"train.max_sequence_length={config.train.max_sequence_length}, "
                f"remote.gpu={config.remote.gpu}. Reduce the batch size or "
                "sequence length, or select a larger GPU; Sommelier does not "
                "change these values automatically."
            ),
        )
    is_timeout = isinstance(error, TimeoutError) or "timed out" in lowered
    if is_timeout:
        return ResourceError(
            "training exceeded its time budget",
            hint=(
                f"remote.train_timeout_seconds={config.remote.train_timeout_seconds}. "
                "Raise the timeout, reduce train.epochs, or shrink the split "
                "sizes; Sommelier does not retry automatically."
            ),
        )
    return None


def build_default_trainer(config: SommelierConfig) -> AdapterTrainer:
    """Builds the QLoRA trainer: 4-bit base model, LoRA from config.

    torch/transformers/peft/bitsandbytes are optional extras imported
    inside the stage. Hyperparameters come from the validated config and
    are never adjusted to fit hardware (RFC-0004).
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise ExternalDependencyError(
            "adapter training requires the torch and transformers packages",
            hint="Run training remotely or install the train extra stack.",
        ) from error
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as error:
        raise ExternalDependencyError(
            "adapter training requires the peft package",
            hint="Run training remotely or install the train extra stack.",
        ) from error

    class _QLoraTrainer:
        def train(
            self,
            train_examples: list[dict[str, object]],
            validation_examples: list[dict[str, object]],
            adapter_dir: Path,
        ) -> TrainingResult:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.base_model_id,
                revision=config.model.tokenizer_revision,
                trust_remote_code=config.model.allow_remote_code,
            )
            if tokenizer.pad_token_id is None:
                # Padding-only metadata; does not alter training
                # hyperparameters.
                tokenizer.pad_token = tokenizer.eos_token

            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.model.base_model_id,
                revision=config.model.base_model_revision,
                trust_remote_code=config.model.allow_remote_code,
                quantization_config=quantization,
                device_map="auto",
            )
            model = prepare_model_for_kbit_training(model)
            lora = LoraConfig(
                r=config.train.lora_rank,
                lora_alpha=config.train.lora_alpha,
                lora_dropout=config.train.lora_dropout,
                target_modules=list(config.train.target_modules),
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora)

            collator = CompletionOnlyCollator(
                tokenizer,
                max_sequence_length=config.train.max_sequence_length,
            )

            def torch_collate(batch: list[dict[str, object]]) -> dict[str, object]:
                collated = collator(batch)
                return {
                    "input_ids": torch.tensor(collated["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(
                        collated["attention_mask"], dtype=torch.long
                    ),
                    "labels": torch.tensor(collated["labels"], dtype=torch.long),
                }

            arguments = TrainingArguments(
                output_dir=str(adapter_dir.parent / "trainer_state"),
                num_train_epochs=config.train.epochs,
                per_device_train_batch_size=config.train.per_device_batch_size,
                gradient_accumulation_steps=config.train.gradient_accumulation_steps,
                learning_rate=config.train.learning_rate,
                lr_scheduler_type=config.train.scheduler,
                warmup_ratio=config.train.warmup_ratio,
                bf16=True,
                # Explicit rather than relying on peft's kbit-preparation
                # defaults: an 8B model at batch 8 without checkpointing
                # exceeded a 44 GiB GPU in the first full run.
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=1,
                report_to=[],
                seed=config.project.seed,
                include_num_input_tokens_seen=True,
                # The completion-only collator consumes prompt_text and
                # full_text; the Trainer's RemoveColumnsCollator wrapper
                # would strip them before our collator runs.
                remove_unused_columns=False,
            )
            trainer = Trainer(
                model=model,
                args=arguments,
                train_dataset=train_examples,
                eval_dataset=validation_examples,
                data_collator=torch_collate,
            )
            trainer.train()
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            return TrainingResult(
                history=list(trainer.state.log_history),
                peak_gpu_memory_mb=measure_peak_gpu_memory_mb(),
            )

    return _QLoraTrainer()


def train_adapter(
    config: SommelierConfig,
    formatted_dir: Path,
    out_dir: Path,
    *,
    context: RunContext,
    command: list[str],
    trainer: AdapterTrainer | None = None,
) -> StageManifest:
    """Trains the LoRA adapter on the formatted train split (RFC-0004).

    Reads train and validation splits, trains through the injected or
    default QLoRA trainer, saves the adapter directory under ``out_dir``,
    and records the train stage manifest whose inputs carry the formatted
    split digests. Validation data is used for eval loss only; the test
    split is never read here (INV-DATA-003).
    """
    train_examples = _read_split(formatted_dir, "train")
    validation_examples = _read_split(formatted_dir, "validation")

    active_trainer = trainer if trainer is not None else build_default_trainer(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = active_trainer.train(train_examples, validation_examples, out_dir)
    except SommelierError:
        raise
    except Exception as error:
        mapped = map_resource_failure(error, config)
        if mapped is not None:
            raise mapped from error
        raise

    metrics = build_training_metrics(
        result["history"],
        peak_gpu_memory_mb=result["peak_gpu_memory_mb"],
    )
    metrics_path = out_dir.parent / METRICS_FILENAME
    write_training_metrics(metrics_path, metrics)
    track_stage_metrics(
        config,
        context,
        stage="train",
        records=[dict(metric) for metric in metrics],
    )

    adapter_files = sorted(path for path in out_dir.rglob("*") if path.is_file())
    if not adapter_files:
        raise UserInputError(
            f"trainer wrote no adapter files under {out_dir}",
            hint="The training backend must save adapter weights before returning.",
        )

    input_refs = [
        make_artifact_ref(
            formatted_dir / f"{split}.jsonl",
            artifact_root=context.artifact_root,
            kind="formatted_split",
            schema_version=FORMATTED_SCHEMA,
        )
        for split in ("train", "validation")
    ]
    output_refs: list[ArtifactRef] = [
        make_artifact_ref(
            path,
            artifact_root=context.artifact_root,
            kind="adapter_weights",
            schema_version="",
        )
        for path in adapter_files
    ]
    output_refs.append(
        make_artifact_ref(
            metrics_path,
            artifact_root=context.artifact_root,
            kind="training_metrics",
            schema_version=TRAINING_METRIC_SCHEMA,
        )
    )

    manifest = build_stage_manifest(
        stage="train",
        run_id=context.run_id,
        config_sha256=context.config_sha256,
        dependency_lock_sha256=context.dependency_lock_sha256,
        command=command,
        seed=config.project.seed,
        inputs=input_refs,
        outputs=output_refs,
        status="succeeded",
    )
    stage_ref = write_stage_manifest(
        manifest,
        run_dir=context.run_dir,
        artifact_root=context.artifact_root,
    )
    update_run_manifest(
        run_dir=context.run_dir,
        artifact_root=context.artifact_root,
        stage="train",
        stage_manifest_ref=stage_ref,
        status="running",
    )
    return manifest
