from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, TypedDict

from sommelier.artifacts import ArtifactRef, make_artifact_ref, write_artifact_atomic
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
INFERENCE_TELEMETRY_SCHEMA: Final = "sommelier.inference_telemetry.v2"
INFERENCE_TELEMETRY_FILENAME: Final = "inference_telemetry.json"

GENERATION_TIMING_SCOPE: Final = "generator.generate_end_to_end_call_wall_time"
GENERATION_TIMING_AGGREGATION: Final = "sum_of_per_example_call_intervals"
SEQUENTIAL_RUN_BOUNDARY: Final = "single_run_generation_invocation_after_model_load"
DEFAULT_GENERATOR_TIMED_OPERATIONS: Final = (
    "prompt_tokenization",
    "input_device_transfer",
    "model_generate",
    "generated_token_decode",
)
INFERENCE_WARMUP_CALLS: Final = 1
INFERENCE_WARMUP_PROMPT_SOURCE: Final = "first_example_in_first_configured_slice"

ModelKind = Literal["base", "adapter"]
EvaluationStage = Literal["eval-base", "eval-adapter"]
IMMUTABLE_HF_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def inference_timed_call_contract() -> dict[str, object]:
    """Canonical v2 boundary around each measured generator call."""
    return {
        "callable": "TextGenerator.generate",
        "default_transformers_implementation_includes": list(DEFAULT_GENERATOR_TIMED_OPERATIONS),
        "explicit_device_synchronization": False,
    }


def inference_warmup_contract() -> dict[str, object]:
    """Canonical v2 warmup performed once before timing starts."""
    return {
        "calls": INFERENCE_WARMUP_CALLS,
        "timed": False,
        "prompt_source": INFERENCE_WARMUP_PROMPT_SOURCE,
        "output_disposition": "discarded",
        "call_scope": GENERATION_TIMING_SCOPE,
        "default_transformers_implementation_includes": list(DEFAULT_GENERATOR_TIMED_OPERATIONS),
        "uses_measured_decoding": True,
    }


def adapter_tree_sha256(path: Path) -> str:
    """Content identity for a local adapter directory.

    Relative paths are part of the digest, traversal order is canonical, and
    symlinks are rejected so a later target change cannot alter evaluated
    weights without changing the recorded identity.
    """
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise UserInputError(
            f"adapter directory is empty: {path}",
            hint="Point --adapter at a saved PEFT adapter directory.",
        )
    for file_path in files:
        if file_path.is_symlink():
            raise UserInputError(
                f"adapter directory contains a symlink: {file_path}",
                hint="Materialize adapter files before evaluation.",
            )
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_adapter_artifact_path(path: Path) -> str | None:
    """Return the canonical artifact-root-relative path for a run adapter.

    Local paths recorded inside a remote container are not portable after a
    run bundle is downloaded.  The ``runs/<run-id>/...`` suffix is portable
    because every artifact reference in a Sommelier manifest uses the same
    artifact-root-relative namespace.
    """
    parts = path.resolve().parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "runs" and index + 1 < len(parts):
            return Path(*parts[index:]).as_posix()
    return None


def evaluation_stage(model_kind: ModelKind) -> EvaluationStage:
    """Stable manifest identity for one model arm of evaluation."""
    return "eval-base" if model_kind == "base" else "eval-adapter"


@dataclass(frozen=True)
class AdapterRef:
    """Where the adapter weights come from: a local directory or a
    Hugging Face repo id with a pinned revision."""

    source: str
    revision: str | None = None

    @property
    def is_local(self) -> bool:
        return Path(self.source).exists()

    def describe(self) -> dict[str, object]:
        description: dict[str, object] = {
            "source": self.source,
            "revision": None if self.is_local else (self.revision or "main"),
            "kind": "local_directory" if self.is_local else "huggingface_repo",
        }
        if self.is_local:
            local_path = Path(self.source)
            description["tree_sha256"] = adapter_tree_sha256(local_path)
            description["artifact_path"] = canonical_adapter_artifact_path(local_path)
            description["revision_is_immutable"] = True
        else:
            revision = self.revision or "main"
            description["tree_sha256"] = None
            description["artifact_path"] = None
            description["revision_is_immutable"] = bool(IMMUTABLE_HF_REVISION.fullmatch(revision))
        return description


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


def gpu_count_from_label(gpu_label: str) -> int:
    """Returns the allocation count encoded by Modal-style ``GPU:count`` labels."""
    match = re.search(r":([1-9][0-9]*)$", gpu_label)
    return int(match.group(1)) if match is not None else 1


def _rounded_seconds(value: float) -> float:
    return round(value, 6)


def _write_inference_telemetry(
    config: SommelierConfig,
    *,
    out_dir: Path,
    model_kind: ModelKind,
    context: RunContext,
    decoding: DecodingConfig,
    slice_elapsed_seconds: dict[str, float],
    generation_refs: dict[str, ArtifactRef],
) -> ArtifactRef:
    """Writes aggregate inference timing without making generations volatile.

    The timed interval is each end-to-end ``TextGenerator.generate`` call. For
    the default Transformers implementation this includes prompt tokenization,
    input device transfer, ``model.generate``, and generated-token decoding.
    Model loading, the one warmup call, tool-call parsing, artifact serialization,
    and time between calls are outside the measurement. Slices and examples are
    processed sequentially by one model instance, so elapsed seconds can be
    attributed to a language slice without overlap.
    """
    slices: dict[str, dict[str, object]] = {}
    total_examples = 0
    total_elapsed_seconds = 0.0
    for slice_language in config.eval.slices:
        examples = len(read_jsonl_records(out_dir / slice_filename(slice_language)))
        elapsed_seconds = slice_elapsed_seconds[slice_language]
        total_examples += examples
        total_elapsed_seconds += elapsed_seconds
        slices[slice_language] = {
            "examples": examples,
            "elapsed_seconds": _rounded_seconds(elapsed_seconds),
            "seconds_per_example": _rounded_seconds(elapsed_seconds / examples),
            "generation_artifact": generation_refs[slice_language],
        }

    payload: dict[str, object] = {
        "schema_version": INFERENCE_TELEMETRY_SCHEMA,
        "run_id": context.run_id,
        "model_kind": model_kind,
        "decoding": dict(decoding),
        "measurement": {
            "scope": GENERATION_TIMING_SCOPE,
            "aggregation": GENERATION_TIMING_AGGREGATION,
            "clock": "monotonic_seconds",
            "model_load_included": False,
            "parsing_and_artifact_io_included": False,
        },
        "timed_call_contract": inference_timed_call_contract(),
        "warmup": inference_warmup_contract(),
        "sequential_run": {
            "boundary": SEQUENTIAL_RUN_BOUNDARY,
            "concurrency": 1,
            "single_model_instance": True,
            "slice_order": list(config.eval.slices),
            "example_order": "formatted_test_order_within_slice",
        },
        "hardware": {
            "gpu_label": config.remote.gpu,
            "gpu_count": gpu_count_from_label(config.remote.gpu),
            "source": "config.remote.gpu",
        },
        "slices": slices,
        "total": {
            "examples": total_examples,
            "elapsed_seconds": _rounded_seconds(total_elapsed_seconds),
            "seconds_per_example": _rounded_seconds(total_elapsed_seconds / total_examples),
        },
    }
    telemetry_path = out_dir / INFERENCE_TELEMETRY_FILENAME

    def writer(temp_path: Path) -> None:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return write_artifact_atomic(
        telemetry_path,
        writer,
        artifact_root=context.artifact_root,
        kind="inference_telemetry",
        schema_version=INFERENCE_TELEMETRY_SCHEMA,
    )


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
    empty_slices = sorted(slice_language for slice_language, rows in by_slice.items() if not rows)
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
    clock: Callable[[], float] = time.perf_counter,
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

    # Prime the same end-to-end generate path once for every evaluation arm.
    # The first configured slice is guaranteed non-empty by ``read_test_slices``;
    # using its first stored prompt makes the warmup choice deterministic. The
    # result is deliberately discarded, and the telemetry clock is not touched.
    warmup_prompt = str(by_slice[config.eval.slices[0]][0]["prompt_text"])
    active_generator.generate(warmup_prompt, decoding=decoding)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_refs: list[ArtifactRef] = []
    generation_refs: dict[str, ArtifactRef] = {}
    slice_elapsed_seconds: dict[str, float] = {}
    for slice_language in config.eval.slices:
        records: list[dict[str, object]] = []
        elapsed_seconds = 0.0
        for example in by_slice[slice_language]:
            prompt_text = str(example["prompt_text"])
            started = clock()
            raw_text = active_generator.generate(prompt_text, decoding=decoding)
            finished = clock()
            if not math.isfinite(started) or not math.isfinite(finished) or finished < started:
                raise EvaluationError(
                    "inference telemetry clock returned a non-monotonic value",
                    hint="Use a finite monotonic clock for generation timing.",
                )
            elapsed_seconds += finished - started
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
        generation_ref = make_artifact_ref(
            generations_path,
            artifact_root=context.artifact_root,
            kind="generations",
            schema_version=GENERATION_SCHEMA,
        )
        generation_refs[slice_language] = generation_ref
        output_refs.append(generation_ref)
        slice_elapsed_seconds[slice_language] = elapsed_seconds

    telemetry_ref = _write_inference_telemetry(
        config,
        out_dir=out_dir,
        model_kind=model_kind,
        context=context,
        decoding=decoding,
        slice_elapsed_seconds=slice_elapsed_seconds,
        generation_refs=generation_refs,
    )
    output_refs.append(telemetry_ref)

    input_ref = make_artifact_ref(
        test_path,
        artifact_root=context.artifact_root,
        kind="formatted_split",
        schema_version=FORMATTED_SCHEMA,
    )
    details: dict[str, object] = {
        "eval_slices": list(config.eval.slices),
        "inference_telemetry": telemetry_ref["path"],
        "execution_mode": "sequential",
    }
    if adapter is not None:
        details["adapter_source"] = adapter.describe()
    record_stage_success(
        context,
        stage=evaluation_stage(model_kind),
        command=command,
        seed=config.project.seed,
        inputs=[input_ref],
        outputs=output_refs,
        details=details,
    )
    return output_refs
