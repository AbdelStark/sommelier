from __future__ import annotations

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

GENERATION_SCHEMA: Final = "sommelier.generation.v1"

ModelKind = Literal["base", "adapter"]


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
    adapter_dir: Path | None = None,
) -> TextGenerator:
    """Loads the base model (optionally with an adapter) for greedy decoding.

    torch/transformers/peft are optional extras imported inside this stage
    function, never at package import time.
    """
    if model_kind == "adapter" and adapter_dir is None:
        raise UserInputError(
            "adapter evaluation requires an adapter directory",
            hint="Pass adapter_dir pointing at the trained adapter artifacts.",
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
    if model_kind == "adapter":
        try:
            from peft import PeftModel
        except ImportError as error:
            raise ExternalDependencyError(
                "adapter evaluation requires the peft package",
                hint="Run evaluation remotely or install the eval extra stack.",
            ) from error
        model = PeftModel.from_pretrained(model, adapter_dir)
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


def run_generation(
    config: SommelierConfig,
    *,
    formatted_dir: Path,
    out_dir: Path,
    model_kind: ModelKind,
    context: RunContext,
    command: list[str],
    adapter_dir: Path | None = None,
    generator: TextGenerator | None = None,
) -> ArtifactRef:
    """Generates one output per formatted test prompt and persists records.

    Prompts come exclusively from the stored ``prompt_text`` of the
    formatted test split (evaluation never rebuilds prompts).
    Every record is written to ``generations.jsonl`` with its parse status
    and the deterministic decoding config, even when parsing fails.
    """
    decoding = validate_decoding(config)
    test_path = formatted_dir / "test.jsonl"
    examples = read_jsonl_records(test_path)
    if not examples:
        raise UserInputError(
            f"formatted test split is empty: {test_path}",
            hint="Run sommelier format build with a non-empty test split.",
        )

    active_generator = generator or load_model_generator(config, model_kind, adapter_dir)

    records: list[dict[str, object]] = []
    for example in examples:
        if example.get("schema_version") != FORMATTED_SCHEMA:
            raise SchemaValidationError(
                f"{test_path}: expected {FORMATTED_SCHEMA} records",
                hint="Rebuild the formatted split with the current pipeline version.",
            )
        prompt_text = str(example["prompt_text"])
        raw_text = active_generator.generate(prompt_text, decoding=decoding)
        parsed_call, parse_status = parse_tool_call(raw_text)
        records.append(
            {
                "schema_version": GENERATION_SCHEMA,
                "example_id": example["example_id"],
                "model_kind": model_kind,
                "prompt_sha256": example["prompt_sha256"],
                "raw_text": raw_text,
                "parsed_call": parsed_call,
                "parse_status": parse_status,
                "decoding": dict(decoding),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = out_dir / "generations.jsonl"
    write_jsonl_records(generations_path, records)

    input_ref = make_artifact_ref(
        test_path,
        artifact_root=context.artifact_root,
        kind="formatted_split",
        schema_version=FORMATTED_SCHEMA,
    )
    output_ref = make_artifact_ref(
        generations_path,
        artifact_root=context.artifact_root,
        kind="generations",
        schema_version=GENERATION_SCHEMA,
    )
    record_stage_success(
        context,
        stage="eval",
        command=command,
        seed=config.project.seed,
        inputs=[input_ref],
        outputs=[output_ref],
    )
    return output_ref
