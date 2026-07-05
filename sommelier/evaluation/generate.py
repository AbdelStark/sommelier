from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, TypedDict

from sommelier.artifacts import ArtifactRef, make_artifact_ref
from sommelier.config import SommelierConfig
from sommelier.errors import (
    EvaluationError,
    ExternalDependencyError,
    SchemaValidationError,
    UserInputError,
)
from sommelier.evaluation.parse import parse_tool_call
from sommelier.formatting.chat import FORMATTED_EXAMPLE_SCHEMA as FORMATTED_SCHEMA
from sommelier.run_context import (
    RunContext,
    read_jsonl_records,
    record_stage_success,
    write_jsonl_records,
)

GENERATION_SCHEMA: Final = "sommelier.generation.v2"

ModelKind = Literal["base", "adapter"]


@dataclass(frozen=True)
class AdapterRef:
    """Where the adapter weights come from: a local directory or a
    Hugging Face repo id with a pinned revision."""

    source: str
    revision: str | None = None

    @property
    def is_local(self) -> bool:
        return Path(self.source).exists()

    def describe(self) -> dict[str, str | None]:
        return {
            "source": self.source,
            "revision": None if self.is_local else (self.revision or "main"),
            "kind": "local_directory" if self.is_local else "huggingface_repo",
        }


class DecodingConfig(TypedDict):
    temperature: float
    do_sample: bool
    max_new_tokens: int


class TextGenerator(Protocol):
    """One deterministic completion per prompt; implementations load models."""

    def generate(self, prompt_text: str, *, decoding: DecodingConfig) -> str: ...


def validate_decoding(config: SommelierConfig) -> DecodingConfig:
    """Builds the decoding config, rejecting anything non-deterministic.

    Reference evaluation requires temperature 0.0 and sampling disabled;
    any other setting fails instead of being silently coerced.
    """
    if config.eval.temperature != 0.0:
        raise EvaluationError(
            f"reference evaluation requires temperature 0.0, got {config.eval.temperature}",
            hint="Set eval.temperature to 0.0 for deterministic decoding.",
        )
    if config.eval.do_sample:
        raise EvaluationError(
            "reference evaluation requires do_sample false",
            hint="Set eval.do_sample to false for deterministic decoding.",
        )
    if config.eval.max_new_tokens <= 0:
        raise EvaluationError(
            f"eval.max_new_tokens must be positive, got {config.eval.max_new_tokens}",
            hint="Set eval.max_new_tokens to a positive token budget.",
        )
    return DecodingConfig(
        temperature=0.0,
        do_sample=False,
        max_new_tokens=config.eval.max_new_tokens,
    )


def load_model_generator(
    config: SommelierConfig,
    model_kind: ModelKind,
    adapter: AdapterRef | None = None,
) -> TextGenerator:
    """Loads the base model (optionally with an adapter) for greedy decoding.

    The adapter may be a local directory (a run's own train output) or a
    published Hugging Face repo pinned to a revision, which is how an
    existing adapter is evaluated without retraining. torch/transformers/
    peft are optional extras imported inside this stage function, never at
    package import time.
    """
    if model_kind == "adapter" and adapter is None:
        raise UserInputError(
            "adapter evaluation requires an adapter source",
            hint="Pass an adapter directory or a Hugging Face repo id.",
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ExternalDependencyError(
            "model generation requires the torch and transformers packages",
            hint="Run evaluation remotely or install the eval extra stack.",
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_id,
        revision=config.model.tokenizer_revision,
        trust_remote_code=config.model.allow_remote_code,
    )
    # device_map="auto" is an accelerate/CUDA sharding path; on Apple
    # Silicon or CPU it leaves modules on the meta device and breaks
    # adapter dispatch, so load normally there and move the whole model.
    load_kwargs: dict[str, object] = {}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_id,
        revision=config.model.base_model_revision,
        trust_remote_code=config.model.allow_remote_code,
        dtype="auto",
        **load_kwargs,
    )
    if not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
    if model_kind == "adapter" and adapter is not None:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise ExternalDependencyError(
                "adapter evaluation requires the peft package",
                hint="Run evaluation remotely or install the eval extra stack.",
            ) from error
        adapter_kwargs: dict[str, object] = {}
        if not adapter.is_local:
            adapter_kwargs["revision"] = adapter.revision or "main"
        model = PeftModel.from_pretrained(model, adapter.source, **adapter_kwargs)
    model.eval()

    class _TransformersGenerator:
        def generate(self, prompt_text: str, *, decoding: DecodingConfig) -> str:
            # prompt_text is the rendered chat template and already contains
            # any special tokens (e.g. begin_of_text); adding them again
            # would double the BOS and change the evaluated prompt.
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    do_sample=decoding["do_sample"],
                    max_new_tokens=decoding["max_new_tokens"],
                )
            new_tokens = output[0][inputs["input_ids"].shape[1] :]
            text: str = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return text

    return _TransformersGenerator()


def slice_filename(slice_language: str) -> str:
    return f"generations.{slice_language}.jsonl"


def read_test_slices(
    config: SommelierConfig,
    formatted_dir: Path,
) -> dict[str, list[dict[str, object]]]:
    """Reads the formatted test split and partitions it by eval slice.

    Every configured slice must be non-empty: evaluating a slice with no
    rows would silently report on nothing.
    """
    test_path = formatted_dir / "test.jsonl"
    examples = read_jsonl_records(test_path)
    if not examples:
        raise UserInputError(
            f"formatted test split is empty: {test_path}",
            hint="Run sommelier format build with a non-empty test split.",
        )
    for example in examples:
        if example.get("schema_version") != FORMATTED_SCHEMA:
            raise SchemaValidationError(
                f"{test_path}: expected {FORMATTED_SCHEMA} records",
                hint="Rebuild the formatted split with the current pipeline version.",
            )
    by_slice: dict[str, list[dict[str, object]]] = {
        slice_language: [] for slice_language in config.eval.slices
    }
    for example in examples:
        language = str(example.get("language"))
        if language in by_slice:
            by_slice[language].append(example)
    empty_slices = sorted(
        slice_language for slice_language, rows in by_slice.items() if not rows
    )
    if empty_slices:
        raise UserInputError(
            f"eval.slices includes {', '.join(empty_slices)} but the formatted "
            "test split has no rows for it",
            hint="Prepare and format data for every configured eval slice, or "
            "remove it from eval.slices.",
        )
    return by_slice


def run_generation(
    config: SommelierConfig,
    *,
    formatted_dir: Path,
    out_dir: Path,
    model_kind: ModelKind,
    context: RunContext,
    command: list[str],
    adapter: AdapterRef | None = None,
    generator: TextGenerator | None = None,
) -> list[ArtifactRef]:
    """Generates one output per formatted test prompt, one file per slice.

    Prompts come exclusively from the stored ``prompt_text`` of the
    formatted test split (evaluation never rebuilds prompts). Every
    record is written to ``generations.<slice>.jsonl`` with its language,
    parse status, and the deterministic decoding config, even when
    parsing fails.
    """
    decoding = validate_decoding(config)
    by_slice = read_test_slices(config, formatted_dir)
    test_path = formatted_dir / "test.jsonl"

    active_generator = generator or load_model_generator(config, model_kind, adapter)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_refs: list[ArtifactRef] = []
    for slice_language in config.eval.slices:
        records: list[dict[str, object]] = []
        for example in by_slice[slice_language]:
            prompt_text = str(example["prompt_text"])
            raw_text = active_generator.generate(prompt_text, decoding=decoding)
            parsed_call, parse_status = parse_tool_call(raw_text)
            records.append(
                {
                    "schema_version": GENERATION_SCHEMA,
                    "example_id": example["example_id"],
                    "model_kind": model_kind,
                    "language": slice_language,
                    "prompt_sha256": example["prompt_sha256"],
                    "raw_text": raw_text,
                    "parsed_call": parsed_call,
                    "parse_status": parse_status,
                    "decoding": dict(decoding),
                }
            )
        generations_path = out_dir / slice_filename(slice_language)
        write_jsonl_records(generations_path, records)
        output_refs.append(
            make_artifact_ref(
                generations_path,
                artifact_root=context.artifact_root,
                kind="generations",
                schema_version=GENERATION_SCHEMA,
            )
        )

    input_ref = make_artifact_ref(
        test_path,
        artifact_root=context.artifact_root,
        kind="formatted_split",
        schema_version=FORMATTED_SCHEMA,
    )
    details: dict[str, object] = {"eval_slices": list(config.eval.slices)}
    if adapter is not None:
        details["adapter_source"] = adapter.describe()
    record_stage_success(
        context,
        stage="eval",
        command=command,
        seed=config.project.seed,
        inputs=[input_ref],
        outputs=output_refs,
        details=details,
    )
    return output_refs
