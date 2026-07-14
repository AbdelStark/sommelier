from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol, TypedDict, cast

import yaml

from sommelier.artifacts import write_artifact_atomic
from sommelier.config import SommelierConfig
from sommelier.errors import InvariantViolation, SecurityPolicyError, UserInputError
from sommelier.formatting.chat import FORMATTED_EXAMPLE_SCHEMA
from sommelier.formatting.templates import render_formatted_example
from sommelier.remote.images import PIPELINE_RUNTIME_VERSIONS
from sommelier.security import redact_text, validate_no_secrets
from sommelier.training.collators import CompletionOnlyCollator, find_prompt_token_count
from sommelier.training.qlora import (
    configure_qlora_base_model,
    qlora_kbit_preparation_kwargs,
    qlora_lora_kwargs,
    qlora_model_load_kwargs,
    qlora_quantization_kwargs,
    qlora_tokenizer_load_kwargs,
    qlora_training_argument_kwargs,
)

PREFLIGHT_SCHEMA_VERSION: Final = "sommelier.qlora_shape_preflight.v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION: Final = "sommelier.qlora_shape_preflight_artifact_manifest.v1"
DIAGNOSTIC_BOUNDARY: Final = (
    "Diagnostic resource-fit evidence only. This run uses synthetic formatted rows, "
    "does not access experiment datasets or providers, and is ineligible for release "
    "accuracy, training, cost, or reproducibility claims."
)

BASE_MODEL_ID: Final = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
BASE_MODEL_REVISION: Final = "54641c1611fcff44fa4865626462445e0a153fc7"
TOKENIZER_REVISION: Final = BASE_MODEL_REVISION
GPU_ALLOCATION: Final = "L40S"
MAX_SEQUENCE_LENGTH: Final = 4096
MIN_SYNTHETIC_SEQUENCE_TOKENS: Final = 4080
TRAIN_ROWS: Final = 16
EVAL_ROWS: Final = 4
OPTIMIZER_STEPS: Final = 1
EXPECTED_TRAIN_MICROBATCHES: Final = 4
EXPECTED_EVAL_FORWARD_BATCHES: Final = 1
EXPECTED_SYSTEM_PROMPT: Final = (
    "You are a tool-calling model. Select the correct tool and return only "
    "the JSON tool call. Do not include explanations."
)
TARGET_MODULES: Final = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
RUNTIME_PACKAGES: Final = tuple(
    package for package, _version in PIPELINE_RUNTIME_VERSIONS if package != "python"
)

RUN_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX_REVISION_PATTERN: Final = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
EMPTY_GIT_STATUS_SHA256: Final = hashlib.sha256(b"").hexdigest()
ARTIFACT_HASH_BOUNDARY: Final = (
    "Hashes cover every regular file present under the diagnostic run directory "
    "when the manifest is built, except preflight_report.json and "
    "artifact_manifest.json. Those two self-referential files are deliberately "
    "excluded."
)


class SourceProvenance(TypedDict):
    git_commit: str
    working_tree_clean: bool
    git_status_sha256: str
    boundary: str


class ArtifactDigest(TypedDict):
    path: str
    bytes: int
    sha256: str


class PreflightTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None
    pad_token: str | None
    eos_token: str | None

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def save_pretrained(self, save_directory: str) -> object: ...


def validate_run_id(run_id: str) -> str:
    """Rejects path traversal and ambiguous diagnostic output names."""
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise UserInputError(
            f"invalid QLoRA preflight run id: {run_id!r}",
            hint=(
                "Use 1-128 ASCII letters, digits, dots, underscores, or hyphens; "
                "the first character must be alphanumeric."
            ),
        )
    return run_id


def preflight_contract() -> dict[str, object]:
    """Returns the immutable Hebrew-v3 resource shape exercised by the smoke."""
    return {
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "allow_remote_code": False,
        "seed": 42,
        "epochs_declared": 2,
        "optimizer_steps_exercised": OPTIMIZER_STEPS,
        "optimizer_step_cap_is_diagnostic_override": True,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "train_microbatches_exercised": EXPECTED_TRAIN_MICROBATCHES,
        "eval_forward_batches_exercised": EXPECTED_EVAL_FORWARD_BATCHES,
        "learning_rate": 0.0002,
        "scheduler": "cosine",
        "warmup_ratio": 0.03,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "minimum_synthetic_sequence_tokens": MIN_SYNTHETIC_SEQUENCE_TOKENS,
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "double_quant": True,
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": list(TARGET_MODULES),
        },
        "gradient_checkpointing": {
            "enabled": True,
            "use_reentrant": False,
        },
        "languages": ["en", "he"],
        "formatting": {
            "template_policy": "tokenizer_chat_template",
            "target_format": "json_tool_call",
            "system_prompt": EXPECTED_SYSTEM_PROMPT,
        },
        "synthetic_train_rows": TRAIN_ROWS,
        "synthetic_eval_rows": EVAL_ROWS,
        "gpu_allocation": GPU_ALLOCATION,
        "dataset_access_required": False,
        "provider_access_required": False,
        "release_evidence_eligible": False,
    }


def validate_preflight_config(config: SommelierConfig) -> None:
    """Binds the diagnostic to the full Hebrew-v3 model and training shape.

    Dataset identities and revisions are deliberately not part of this gate:
    the diagnostic consumes no dataset rows, which lets it run before the
    audited Hebrew dataset is published. The complete resolved config is still
    stored and hashed in the diagnostic artifacts.
    """
    expected: dict[str, object] = {
        "project.seed": 42,
        "model.base_model_id": BASE_MODEL_ID,
        "model.base_model_revision": BASE_MODEL_REVISION,
        "model.tokenizer_revision": TOKENIZER_REVISION,
        "model.allow_remote_code": False,
        "formatting.system_prompt": EXPECTED_SYSTEM_PROMPT,
        "formatting.template_policy": "tokenizer_chat_template",
        "formatting.target_format": "json_tool_call",
        "train.epochs": 2,
        "train.per_device_batch_size": 4,
        "train.gradient_accumulation_steps": 4,
        "train.learning_rate": 0.0002,
        "train.scheduler": "cosine",
        "train.warmup_ratio": 0.03,
        "train.max_sequence_length": MAX_SEQUENCE_LENGTH,
        "train.quantization": "nf4-4bit",
        "train.compute_dtype": "bfloat16",
        "train.lora_rank": 16,
        "train.lora_alpha": 32,
        "train.lora_dropout": 0.05,
        "train.languages": ["en", "he"],
        "train.target_modules": list(TARGET_MODULES),
        "remote.enabled": True,
        "remote.gpu": GPU_ALLOCATION,
    }
    actual: dict[str, object] = {
        "project.seed": config.project.seed,
        "model.base_model_id": config.model.base_model_id,
        "model.base_model_revision": config.model.base_model_revision,
        "model.tokenizer_revision": config.model.tokenizer_revision,
        "model.allow_remote_code": config.model.allow_remote_code,
        "formatting.system_prompt": config.formatting.system_prompt.strip(),
        "formatting.template_policy": config.formatting.template_policy,
        "formatting.target_format": config.formatting.target_format,
        "train.epochs": config.train.epochs,
        "train.per_device_batch_size": config.train.per_device_batch_size,
        "train.gradient_accumulation_steps": config.train.gradient_accumulation_steps,
        "train.learning_rate": config.train.learning_rate,
        "train.scheduler": config.train.scheduler,
        "train.warmup_ratio": config.train.warmup_ratio,
        "train.max_sequence_length": config.train.max_sequence_length,
        "train.quantization": config.train.quantization,
        "train.compute_dtype": config.train.compute_dtype,
        "train.lora_rank": config.train.lora_rank,
        "train.lora_alpha": config.train.lora_alpha,
        "train.lora_dropout": config.train.lora_dropout,
        "train.languages": list(config.train.languages),
        "train.target_modules": list(config.train.target_modules),
        "remote.enabled": config.remote.enabled,
        "remote.gpu": config.remote.gpu,
    }
    mismatches = [
        f"{field}={actual[field]!r} (expected {value!r})"
        for field, value in expected.items()
        if actual[field] != value
    ]
    if mismatches:
        raise InvariantViolation(
            "Hebrew-v3 QLoRA shape preflight config drift: " + "; ".join(mismatches),
            hint=(
                "Run this diagnostic only with examples/config.v3-he-full.yaml at "
                "the preregistered model, training, and L40S resource shape."
            ),
        )


def validate_config_yaml_identity(config: SommelierConfig, config_yaml: str) -> None:
    """Binds the exact stored YAML bytes to the resolved config used by the run."""
    try:
        raw = yaml.safe_load(config_yaml)
        if not isinstance(raw, dict):
            raise TypeError("configuration document is not a mapping")
        validate_no_secrets(raw, context="QLoRA preflight config")
        reparsed = SommelierConfig.model_validate(raw)
    except SecurityPolicyError:
        raise
    except Exception as error:
        raise UserInputError(
            "QLoRA preflight config YAML is not a valid secret-free v2 config",
            hint="Pass the exact bytes that were loaded into the supplied SommelierConfig.",
        ) from error
    if reparsed.model_dump(mode="json") != config.model_dump(mode="json"):
        raise InvariantViolation(
            "QLoRA preflight config YAML does not resolve to the supplied config",
            hint="Do not combine a resolved config object with bytes from another file.",
        )


def validate_source_provenance(source: SourceProvenance) -> None:
    """Requires an exact commit and a digest of the launcher's dirty state."""
    if HEX_REVISION_PATTERN.fullmatch(source["git_commit"]) is None:
        raise UserInputError(
            "QLoRA preflight requires an immutable local Git commit",
            hint="Launch from a Git checkout whose HEAD resolves to a 40- or 64-hex object id.",
        )
    if SHA256_PATTERN.fullmatch(source["git_status_sha256"]) is None:
        raise UserInputError("QLoRA preflight source status digest is not a SHA-256")
    clean = source["working_tree_clean"]
    empty_digest = source["git_status_sha256"] == EMPTY_GIT_STATUS_SHA256
    if not isinstance(clean, bool) or clean != empty_digest:
        raise UserInputError(
            "QLoRA preflight source clean flag disagrees with its Git status digest",
            hint="Re-measure source provenance with the dedicated local Modal launcher.",
        )


def collect_runtime_versions() -> dict[str, str]:
    """Collects the exact Python/training distributions checked by the image gate."""
    return {
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in RUNTIME_PACKAGES},
    }


def validate_runtime_versions(observed: dict[str, str]) -> None:
    expected = dict(PIPELINE_RUNTIME_VERSIONS)
    if observed != expected:
        fields = sorted(set(observed) | set(expected))
        drift = [
            f"{field}={observed.get(field)!r} (expected {expected.get(field)!r})"
            for field in fields
            if observed.get(field) != expected.get(field)
        ]
        raise InvariantViolation(
            "QLoRA preflight runtime drift: " + "; ".join(drift),
            hint="Rebuild the pinned training image; do not use a drifted diagnostic result.",
        )


def _tool_schema() -> list[object]:
    return [
        {
            "name": "diagnostic_lookup",
            "description": "Return the requested diagnostic value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        }
    ]


def _render_synthetic_candidate(
    tokenizer: PreflightTokenizer,
    config: SommelierConfig,
    *,
    language: str,
    split: str,
    example_id: str,
    repetitions: int,
) -> dict[str, object]:
    phrase = (
        " deterministic English resource-shape token"
        if language == "en"
        else " אסימון אבחון עברי דטרמיניסטי לבדיקת משאבים"
    )
    query_prefix = (
        f"Diagnostic {split} only: preserve this long English context."
        if language == "en"
        else f"אבחון {split} בלבד: יש לשמור את ההקשר העברי הארוך הזה."
    )
    source_pair = example_id.rsplit("-", 1)[-1]
    prepared: dict[str, object] = {
        "example_id": example_id,
        "split": split,
        "language": language,
        "source_example_id": f"synthetic-{split}-pair-{source_pair}",
        "query": query_prefix + phrase * repetitions,
        "tools": _tool_schema(),
        "gold_calls": [
            {
                "name": "diagnostic_lookup",
                "arguments": {"value": "fit-check"},
            }
        ],
    }
    return render_formatted_example(
        prepared,
        tokenizer=tokenizer,
        tokenizer_id=config.model.base_model_id,
        tokenizer_revision=config.model.tokenizer_revision,
        system_prompt=config.formatting.system_prompt,
        template_policy=config.formatting.template_policy,
    )


def _near_limit_record(
    tokenizer: PreflightTokenizer,
    config: SommelierConfig,
    *,
    language: str,
    split: str,
    example_id: str,
) -> tuple[dict[str, object], int, int]:
    """Finds the longest repeated-language record that stays within 4096 tokens."""

    def render(repetitions: int) -> tuple[dict[str, object], int]:
        record = _render_synthetic_candidate(
            tokenizer,
            config,
            language=language,
            split=split,
            example_id=example_id,
            repetitions=repetitions,
        )
        tokens = tokenizer.encode(str(record["full_text"]), add_special_tokens=False)
        return record, len(tokens)

    lower = 1
    upper = 1
    best_record: dict[str, object] | None = None
    best_length = 0
    while upper <= 65_536:
        record, length = render(upper)
        if length <= MAX_SEQUENCE_LENGTH:
            best_record = record
            best_length = length
            lower = upper
            upper *= 2
            continue
        break
    if best_record is None:
        raise InvariantViolation(
            f"synthetic {language} base record exceeds {MAX_SEQUENCE_LENGTH} tokens"
        )
    if upper > 65_536:
        raise InvariantViolation(
            f"could not bound synthetic {language} sequence length",
            hint="Inspect the pinned tokenizer before running the GPU diagnostic.",
        )

    low = lower + 1
    high = upper - 1
    while low <= high:
        middle = (low + high) // 2
        record, length = render(middle)
        if length <= MAX_SEQUENCE_LENGTH:
            if length > best_length:
                best_record = record
                best_length = length
            low = middle + 1
        else:
            high = middle - 1

    if best_length < MIN_SYNTHETIC_SEQUENCE_TOKENS:
        raise InvariantViolation(
            f"synthetic {language} sequence reached only {best_length} tokens; "
            f"required at least {MIN_SYNTHETIC_SEQUENCE_TOKENS}",
            hint="Do not treat a materially shorter sequence as a full-shape preflight.",
        )

    prompt_tokens = find_prompt_token_count(
        cast(Any, tokenizer),
        prompt_text=str(best_record["prompt_text"]),
        full_text=str(best_record["full_text"]),
        context=f"synthetic {example_id}",
    )
    if prompt_tokens >= best_length:
        raise InvariantViolation(f"synthetic {example_id} has no assistant target tokens")
    return best_record, best_length, prompt_tokens


def build_synthetic_formatted_splits(
    tokenizer: PreflightTokenizer,
    config: SommelierConfig,
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    """Builds deterministic EN+HE formatted rows near the 4096-token limit.

    Sixteen training rows produce four real batch-4 microbatches, matching the
    configured gradient-accumulation factor before the single optimizer step.
    Four validation rows produce exactly one batch-4 evaluation forward.
    """
    layouts = (("train", 8), ("validation", 2))
    splits: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    lengths: list[dict[str, object]] = []
    for split, rows_per_language in layouts:
        for language in ("en", "he"):
            prototype_id = f"{split}-{language}-000"
            prototype, token_count, prompt_token_count = _near_limit_record(
                tokenizer,
                config,
                language=language,
                split=split,
                example_id=prototype_id,
            )
            for index in range(rows_per_language):
                record = dict(prototype)
                record["example_id"] = f"{split}-{language}-{index:03d}"
                record["source_example_id"] = f"synthetic-{split}-pair-{index:03d}"
                splits[split].append(record)
                lengths.append(
                    {
                        "example_id": record["example_id"],
                        "split": split,
                        "language": language,
                        "full_tokens": token_count,
                        "prompt_tokens": prompt_token_count,
                        "target_tokens": token_count - prompt_token_count,
                    }
                )
    _validate_synthetic_formatted_splits(splits, lengths)
    return splits, lengths


def _validate_synthetic_formatted_splits(
    splits: dict[str, list[dict[str, object]]],
    lengths: list[dict[str, object]],
) -> None:
    """Proves the generated records retain the formatted-example contract."""
    expected_rows = {"train": TRAIN_ROWS, "validation": EVAL_ROWS}
    if {split: len(rows) for split, rows in splits.items()} != expected_rows:
        raise InvariantViolation("synthetic QLoRA preflight split cardinality drift")
    if len(lengths) != TRAIN_ROWS + EVAL_ROWS:
        raise InvariantViolation("synthetic QLoRA preflight token ledger cardinality drift")

    seen_example_ids: set[str] = set()
    source_splits: dict[str, str] = {}
    for split, rows in splits.items():
        languages: dict[str, int] = {"en": 0, "he": 0}
        source_languages: dict[str, dict[str, int]] = {}
        for record in rows:
            example_id = str(record.get("example_id"))
            language = str(record.get("language"))
            source_id = str(record.get("source_example_id"))
            prompt_text = str(record.get("prompt_text"))
            target_text = str(record.get("target_text"))
            full_text = str(record.get("full_text"))
            if record.get("schema_version") != FORMATTED_EXAMPLE_SCHEMA:
                raise InvariantViolation(f"synthetic {example_id} has formatted schema drift")
            if record.get("split") != split or language not in languages:
                raise InvariantViolation(f"synthetic {example_id} has split/language drift")
            if not source_id or source_id == "None":
                raise InvariantViolation(f"synthetic {example_id} has no source pair id")
            if example_id in seen_example_ids:
                raise InvariantViolation(f"duplicate synthetic example id: {example_id}")
            seen_example_ids.add(example_id)
            if hashlib.sha256(prompt_text.encode("utf-8")).hexdigest() != record.get(
                "prompt_sha256"
            ):
                raise InvariantViolation(f"synthetic {example_id} has an invalid prompt digest")
            if (
                not full_text.startswith(prompt_text)
                or target_text not in full_text[len(prompt_text) :]
            ):
                raise InvariantViolation(f"synthetic {example_id} has an invalid target boundary")
            if source_id in source_splits and source_splits[source_id] != split:
                raise InvariantViolation(f"synthetic source id crosses splits: {source_id}")
            source_splits[source_id] = split
            languages[language] += 1
            pair_counts = source_languages.setdefault(source_id, {"en": 0, "he": 0})
            pair_counts[language] += 1
        expected_per_language = expected_rows[split] // 2
        if languages != {"en": expected_per_language, "he": expected_per_language}:
            raise InvariantViolation(f"synthetic {split} language balance drift")
        if len(source_languages) != expected_per_language:
            raise InvariantViolation(
                f"synthetic {split} pair count drift: observed {len(source_languages)}, "
                f"expected {expected_per_language}"
            )
        if any(pair_counts != {"en": 1, "he": 1} for pair_counts in source_languages.values()):
            raise InvariantViolation(f"synthetic {split} pairing drift")


def _write_text(path: Path, content: str) -> None:
    def writer(temp_path: Path) -> None:
        temp_path.write_text(content, encoding="utf-8")

    write_artifact_atomic(path, writer)


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    _write_text(
        path,
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_digests(
    root: Path,
    *,
    excluded_relative_paths: frozenset[str] = frozenset(),
) -> list[ArtifactDigest]:
    """Hashes existing diagnostic artifacts in stable relative-path order."""
    symlinks = sorted(path for path in root.rglob("*") if path.is_symlink())
    if symlinks:
        relative = symlinks[0].relative_to(root).as_posix()
        raise InvariantViolation(
            f"QLoRA preflight artifact tree contains a symbolic link: {relative}"
        )
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded_relative_paths
    )
    return [
        ArtifactDigest(
            path=path.relative_to(root).as_posix(),
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in files
    ]


def _closed_mapping(
    value: object,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise InvariantViolation(f"{context} must be a JSON object with string keys")
    payload = cast(Mapping[str, object], value)
    keys = set(payload)
    missing = sorted(required - keys)
    extra = sorted(keys - required - optional)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise InvariantViolation(f"{context} contract drift: {'; '.join(details)}")
    return payload


def _validated_digest_entries(value: object, *, context: str) -> list[ArtifactDigest]:
    if not isinstance(value, list):
        raise InvariantViolation(f"{context} must be a JSON array")
    validated: list[ArtifactDigest] = []
    paths: list[str] = []
    excluded = {"preflight_report.json", "artifact_manifest.json"}
    for index, raw_entry in enumerate(value):
        entry = _closed_mapping(
            raw_entry,
            required=frozenset({"path", "bytes", "sha256"}),
            context=f"{context}[{index}]",
        )
        path = entry["path"]
        byte_count = entry["bytes"]
        digest = entry["sha256"]
        if not isinstance(path, str) or not path:
            raise InvariantViolation(f"{context}[{index}].path must be a non-empty string")
        posix_path = PurePosixPath(path)
        if (
            posix_path.is_absolute()
            or path != posix_path.as_posix()
            or not posix_path.parts
            or any(part in {"", ".", ".."} for part in posix_path.parts)
            or path in excluded
        ):
            raise InvariantViolation(f"{context}[{index}].path is not a canonical artifact path")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise InvariantViolation(f"{context}[{index}].bytes must be a non-negative integer")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise InvariantViolation(f"{context}[{index}].sha256 is not a SHA-256")
        paths.append(path)
        validated.append(ArtifactDigest(path=path, bytes=byte_count, sha256=digest))
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise InvariantViolation(f"{context} paths must be unique and sorted")
    return validated


def validate_artifact_manifest(
    manifest: object,
    *,
    artifact_root: Path | None = None,
) -> list[ArtifactDigest]:
    """Validates the closed digest manifest and optionally rehashes its tree."""
    payload = _closed_mapping(
        manifest,
        required=frozenset({"schema_version", "diagnostic_only", "artifacts"}),
        context="QLoRA preflight artifact manifest",
    )
    if payload["schema_version"] != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise InvariantViolation("QLoRA preflight artifact manifest schema drift")
    if payload["diagnostic_only"] is not True:
        raise InvariantViolation("QLoRA preflight artifact manifest lost diagnostic-only status")
    entries = _validated_digest_entries(
        payload["artifacts"],
        context="QLoRA preflight artifact manifest artifacts",
    )
    if artifact_root is not None:
        observed = artifact_digests(
            artifact_root,
            excluded_relative_paths=frozenset({"artifact_manifest.json", "preflight_report.json"}),
        )
        if entries != observed:
            raise InvariantViolation(
                "QLoRA preflight artifact manifest does not match the on-disk artifact tree"
            )
    return entries


def _validate_peak_memory(value: object) -> None:
    payload = _closed_mapping(
        value,
        required=frozenset({"allocated_mib", "reserved_mib"}),
        context="QLoRA preflight peak memory",
    )
    for field in ("allocated_mib", "reserved_mib"):
        measurement = payload[field]
        if measurement is not None and (
            isinstance(measurement, bool) or not isinstance(measurement, int) or measurement < 0
        ):
            raise InvariantViolation(
                f"QLoRA preflight peak memory {field} must be null or a non-negative integer"
            )


def validate_preflight_report(
    report: object,
    *,
    require_terminal: bool = True,
    require_artifact_hashes: bool = True,
) -> None:
    """Validates the closed preflight report contract for reads and writes."""
    common = frozenset(
        {
            "schema_version",
            "run_id",
            "status",
            "diagnostic_only",
            "release_evidence_eligible",
            "boundary",
            "provider_accessed",
            "dataset_accessed",
            "config_sha256",
            "contract",
        }
    )
    optional = frozenset(
        {
            "resolved_config",
            "source_code",
            "versions",
            "hardware",
            "synthetic_dataset",
            "model_wiring",
            "execution",
            "peak_gpu_memory_mib",
            "elapsed_seconds",
            "failure",
            "peak_memory_measurement_failure",
            "artifact_hashes",
        }
    )
    payload = _closed_mapping(
        report,
        required=common,
        optional=optional,
        context="QLoRA preflight report",
    )
    if payload["schema_version"] != PREFLIGHT_SCHEMA_VERSION:
        raise InvariantViolation("QLoRA preflight report schema drift")
    run_id = payload["run_id"]
    if not isinstance(run_id, str):
        raise InvariantViolation("QLoRA preflight report run_id must be a string")
    validate_run_id(run_id)
    status = payload["status"]
    if status not in {"running", "succeeded", "failed"}:
        raise InvariantViolation("QLoRA preflight report has an invalid status")
    if require_terminal and status not in {"succeeded", "failed"}:
        raise InvariantViolation("QLoRA preflight report is not terminal")
    if payload["diagnostic_only"] is not True or payload["release_evidence_eligible"] is not False:
        raise InvariantViolation("QLoRA preflight report lost its diagnostic claim boundary")
    if payload["provider_accessed"] is not False or payload["dataset_accessed"] is not False:
        raise InvariantViolation("QLoRA preflight report claims forbidden external data access")
    if payload["boundary"] != DIAGNOSTIC_BOUNDARY:
        raise InvariantViolation("QLoRA preflight report boundary text drift")
    config_sha256 = payload["config_sha256"]
    if not isinstance(config_sha256, str) or SHA256_PATTERN.fullmatch(config_sha256) is None:
        raise InvariantViolation("QLoRA preflight report config_sha256 is not a SHA-256")
    if payload["contract"] != preflight_contract():
        raise InvariantViolation("QLoRA preflight report embedded contract drift")

    if "resolved_config" in payload:
        if not isinstance(payload["resolved_config"], Mapping):
            raise InvariantViolation("QLoRA preflight resolved_config must be a JSON object")
        try:
            resolved_config = SommelierConfig.model_validate(payload["resolved_config"])
        except Exception as error:
            raise InvariantViolation("QLoRA preflight resolved_config is invalid") from error
        validate_preflight_config(resolved_config)
    if "source_code" in payload:
        source = _closed_mapping(
            payload["source_code"],
            required=frozenset(
                {"git_commit", "working_tree_clean", "git_status_sha256", "boundary"}
            ),
            context="QLoRA preflight source provenance",
        )
        if (
            not isinstance(source["git_commit"], str)
            or not isinstance(source["working_tree_clean"], bool)
            or not isinstance(source["git_status_sha256"], str)
            or not isinstance(source["boundary"], str)
        ):
            raise InvariantViolation("QLoRA preflight source provenance types are invalid")
        validate_source_provenance(cast(SourceProvenance, dict(source)))
    if "versions" in payload:
        versions = payload["versions"]
        if not isinstance(versions, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in versions.items()
        ):
            raise InvariantViolation("QLoRA preflight versions must map strings to strings")
        validate_runtime_versions(dict(versions))
    if "hardware" in payload:
        hardware = _closed_mapping(
            payload["hardware"],
            required=frozenset(
                {
                    "allocation_label",
                    "device_count",
                    "device_name",
                    "compute_capability",
                    "total_memory_mib",
                    "cuda_runtime",
                    "cudnn",
                }
            ),
            context="QLoRA preflight hardware",
        )
        if (
            hardware["allocation_label"] != GPU_ALLOCATION
            or isinstance(hardware["device_count"], bool)
            or hardware["device_count"] != 1
            or not isinstance(hardware["device_name"], str)
            or "L40S" not in str(hardware["device_name"]).upper()
        ):
            raise InvariantViolation("QLoRA preflight hardware does not prove one L40S")
        if (
            isinstance(hardware["total_memory_mib"], bool)
            or not isinstance(hardware["total_memory_mib"], int)
            or hardware["total_memory_mib"] <= 0
        ):
            raise InvariantViolation("QLoRA preflight hardware memory is invalid")
    if "synthetic_dataset" in payload:
        synthetic = _closed_mapping(
            payload["synthetic_dataset"],
            required=frozenset({"schema_version", "source", "rows", "languages", "token_lengths"}),
            context="QLoRA preflight synthetic dataset",
        )
        if synthetic["schema_version"] != FORMATTED_EXAMPLE_SCHEMA:
            raise InvariantViolation("QLoRA preflight synthetic dataset schema drift")
        if synthetic["source"] != "generated in-container; no dataset API or published rows":
            raise InvariantViolation("QLoRA preflight synthetic dataset source drift")
        rows = _closed_mapping(
            synthetic["rows"],
            required=frozenset({"train", "validation"}),
            context="QLoRA preflight synthetic dataset rows",
        )
        if rows != {"train": TRAIN_ROWS, "validation": EVAL_ROWS}:
            raise InvariantViolation("QLoRA preflight synthetic dataset row-count drift")
        languages = _closed_mapping(
            synthetic["languages"],
            required=frozenset({"train", "validation"}),
            context="QLoRA preflight synthetic dataset languages",
        )
        for split, expected_count in (("train", TRAIN_ROWS // 2), ("validation", EVAL_ROWS // 2)):
            counts = _closed_mapping(
                languages[split],
                required=frozenset({"en", "he"}),
                context=f"QLoRA preflight synthetic {split} languages",
            )
            if counts != {"en": expected_count, "he": expected_count}:
                raise InvariantViolation(f"QLoRA preflight synthetic {split} language-count drift")
        token_lengths = synthetic["token_lengths"]
        if not isinstance(token_lengths, list) or len(token_lengths) != TRAIN_ROWS + EVAL_ROWS:
            raise InvariantViolation("QLoRA preflight token ledger cardinality drift")
        ledger_ids: set[str] = set()
        for index, raw_entry in enumerate(token_lengths):
            entry = _closed_mapping(
                raw_entry,
                required=frozenset(
                    {
                        "example_id",
                        "split",
                        "language",
                        "full_tokens",
                        "prompt_tokens",
                        "target_tokens",
                    }
                ),
                context=f"QLoRA preflight token ledger[{index}]",
            )
            example_id = entry["example_id"]
            if not isinstance(example_id, str) or example_id in ledger_ids:
                raise InvariantViolation("QLoRA preflight token ledger example ids are invalid")
            ledger_ids.add(example_id)
            full_tokens = entry["full_tokens"]
            prompt_tokens = entry["prompt_tokens"]
            target_tokens = entry["target_tokens"]
            if (
                isinstance(full_tokens, bool)
                or not isinstance(full_tokens, int)
                or not MIN_SYNTHETIC_SEQUENCE_TOKENS <= full_tokens <= MAX_SEQUENCE_LENGTH
                or isinstance(prompt_tokens, bool)
                or not isinstance(prompt_tokens, int)
                or isinstance(target_tokens, bool)
                or not isinstance(target_tokens, int)
                or prompt_tokens <= 0
                or target_tokens <= 0
                or prompt_tokens + target_tokens != full_tokens
            ):
                raise InvariantViolation("QLoRA preflight token ledger length drift")
            if entry["split"] not in {"train", "validation"} or entry["language"] not in {
                "en",
                "he",
            }:
                raise InvariantViolation("QLoRA preflight token ledger split/language drift")
    if "model_wiring" in payload:
        wiring = _closed_mapping(
            payload["model_wiring"],
            required=frozenset(
                {
                    "target_module_counts",
                    "parameters",
                    "gradient_checkpointing_active",
                    "use_cache",
                    "hf_device_map",
                }
            ),
            context="QLoRA preflight model wiring",
        )
        if wiring["gradient_checkpointing_active"] is not True or wiring["use_cache"] is not False:
            raise InvariantViolation("QLoRA preflight checkpointing/cache model contract drift")
        target_counts = _closed_mapping(
            wiring["target_module_counts"],
            required=frozenset(TARGET_MODULES),
            context="QLoRA preflight target-module counts",
        )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in target_counts.values()
        ):
            raise InvariantViolation("QLoRA preflight target-module counts are invalid")
        parameters = _closed_mapping(
            wiring["parameters"],
            required=frozenset({"trainable", "total_visible", "trainable_fraction"}),
            context="QLoRA preflight parameter counts",
        )
        trainable = parameters["trainable"]
        total_visible = parameters["total_visible"]
        fraction = parameters["trainable_fraction"]
        if (
            isinstance(trainable, bool)
            or not isinstance(trainable, int)
            or isinstance(total_visible, bool)
            or not isinstance(total_visible, int)
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0 < trainable < total_visible
            or not 0 < float(fraction) < 1
        ):
            raise InvariantViolation("QLoRA preflight parameter counts are invalid")
        device_map = wiring["hf_device_map"]
        if (
            not isinstance(device_map, Mapping)
            or not device_map
            or any(placement != "cuda:0" for placement in device_map.values())
        ):
            raise InvariantViolation("QLoRA preflight report has an invalid hf_device_map")
    if "execution" in payload:
        execution = _closed_mapping(
            payload["execution"],
            required=frozenset(
                {
                    "optimizer_steps",
                    "train_microbatches",
                    "train_batch_size_per_device",
                    "eval_forward_batches",
                    "eval_batch_size_per_device",
                    "train_metrics",
                    "eval_metrics",
                }
            ),
            context="QLoRA preflight execution",
        )
        expected_execution = {
            "optimizer_steps": OPTIMIZER_STEPS,
            "train_microbatches": EXPECTED_TRAIN_MICROBATCHES,
            "train_batch_size_per_device": 4,
            "eval_forward_batches": EXPECTED_EVAL_FORWARD_BATCHES,
            "eval_batch_size_per_device": 4,
        }
        if any(execution[field] != expected for field, expected in expected_execution.items()):
            raise InvariantViolation("QLoRA preflight execution shape drift")
        if not isinstance(execution["train_metrics"], Mapping) or not isinstance(
            execution["eval_metrics"], Mapping
        ):
            raise InvariantViolation("QLoRA preflight execution metrics must be JSON objects")
        train_loss = execution["train_metrics"].get("train_loss")
        eval_loss = execution["eval_metrics"].get("eval_loss")
        if any(
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            for loss in (train_loss, eval_loss)
        ):
            raise InvariantViolation("QLoRA preflight execution losses must be finite")

    if status in {"succeeded", "failed"}:
        elapsed = payload.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise InvariantViolation("QLoRA preflight terminal elapsed_seconds is invalid")
        if "peak_gpu_memory_mib" not in payload:
            raise InvariantViolation("QLoRA preflight terminal report is missing peak memory")
        _validate_peak_memory(payload["peak_gpu_memory_mib"])
    if status == "succeeded":
        required_success = {
            "resolved_config",
            "source_code",
            "versions",
            "hardware",
            "synthetic_dataset",
            "model_wiring",
            "execution",
        }
        missing_success = sorted(required_success - set(payload))
        if missing_success:
            raise InvariantViolation(
                "QLoRA preflight success report is incomplete: " + ", ".join(missing_success)
            )
        if "failure" in payload:
            raise InvariantViolation("QLoRA preflight success report contains failure evidence")
    if status == "failed":
        failure = _closed_mapping(
            payload.get("failure"),
            required=frozenset({"type", "message", "cuda_out_of_memory"}),
            context="QLoRA preflight failure",
        )
        if not isinstance(failure["type"], str) or not isinstance(failure["message"], str):
            raise InvariantViolation("QLoRA preflight failure type/message must be strings")
        if not isinstance(failure["cuda_out_of_memory"], bool):
            raise InvariantViolation("QLoRA preflight failure OOM flag must be boolean")
    if "peak_memory_measurement_failure" in payload:
        measurement_failure = _closed_mapping(
            payload["peak_memory_measurement_failure"],
            required=frozenset({"type", "message"}),
            context="QLoRA preflight peak-memory measurement failure",
        )
        if not isinstance(measurement_failure["type"], str) or not isinstance(
            measurement_failure["message"], str
        ):
            raise InvariantViolation(
                "QLoRA preflight peak-memory measurement failure fields must be strings"
            )

    hashes = payload.get("artifact_hashes")
    if require_terminal and require_artifact_hashes and hashes is None:
        raise InvariantViolation("QLoRA preflight terminal report is missing artifact hashes")
    if hashes is not None:
        artifact_hashes = _closed_mapping(
            hashes,
            required=frozenset({"manifest_path", "manifest_sha256", "files", "boundary"}),
            context="QLoRA preflight report artifact hashes",
        )
        if artifact_hashes["manifest_path"] != "artifact_manifest.json":
            raise InvariantViolation("QLoRA preflight report manifest path drift")
        manifest_sha = artifact_hashes["manifest_sha256"]
        if not isinstance(manifest_sha, str) or SHA256_PATTERN.fullmatch(manifest_sha) is None:
            raise InvariantViolation("QLoRA preflight report manifest digest is not a SHA-256")
        _validated_digest_entries(
            artifact_hashes["files"],
            context="QLoRA preflight report artifact files",
        )
        if artifact_hashes["boundary"] != ARTIFACT_HASH_BOUNDARY:
            raise InvariantViolation("QLoRA preflight artifact-hash boundary text drift")


def validate_preflight_artifacts(
    report: object,
    manifest: object,
    *,
    artifact_root: Path,
) -> None:
    """Cross-validates a terminal report, digest manifest, and on-disk tree."""
    validate_preflight_report(report)
    manifest_entries = validate_artifact_manifest(manifest, artifact_root=artifact_root)
    report_payload = cast(Mapping[str, object], report)
    hashes = cast(Mapping[str, object], report_payload["artifact_hashes"])
    report_entries = _validated_digest_entries(
        hashes["files"],
        context="QLoRA preflight report artifact files",
    )
    if report_entries != manifest_entries:
        raise InvariantViolation(
            "QLoRA preflight report artifact files disagree with the artifact manifest"
        )
    manifest_path = artifact_root / "artifact_manifest.json"
    if hashes["manifest_sha256"] != sha256_file(manifest_path):
        raise InvariantViolation(
            "QLoRA preflight report manifest digest disagrees with artifact_manifest.json"
        )
    config_path = artifact_root / "config.input.yaml"
    if config_path.is_file():
        config_yaml = config_path.read_text(encoding="utf-8")
        if (
            report_payload["config_sha256"]
            != hashlib.sha256(config_yaml.encode("utf-8")).hexdigest()
        ):
            raise InvariantViolation(
                "QLoRA preflight report config digest disagrees with config.input.yaml"
            )
        resolved_payload = report_payload.get("resolved_config")
        if resolved_payload is not None:
            resolved_config = SommelierConfig.model_validate(resolved_payload)
            validate_config_yaml_identity(resolved_config, config_yaml)
    elif report_payload["status"] == "succeeded":
        raise InvariantViolation("QLoRA preflight success artifacts are missing config.input.yaml")


def _hardware_metadata(torch: Any) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise InvariantViolation(
            "QLoRA shape preflight requires CUDA",
            hint="Launch the dedicated Modal entrypoint on its fixed L40S allocation.",
        )
    device_count = int(torch.cuda.device_count())
    if device_count != 1:
        raise InvariantViolation(
            f"QLoRA shape preflight requires exactly one visible GPU, observed {device_count}",
            hint=(
                "Launch the dedicated Modal entrypoint with its fixed single-L40S allocation; "
                "do not accept a sharded or multi-GPU resource-fit result."
            ),
        )
    name = str(torch.cuda.get_device_name(0))
    if "L40S" not in name.upper():
        raise InvariantViolation(
            f"QLoRA shape preflight expected L40S hardware, observed {name!r}",
            hint="Do not substitute a different accelerator for this resource-fit diagnostic.",
        )
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    return {
        "allocation_label": GPU_ALLOCATION,
        "device_count": device_count,
        "device_name": name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_mib": int(properties.total_memory // (1024 * 1024)),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
    }


def validate_single_cuda_device_map(model: object) -> dict[str, str]:
    """Proves that Accelerate placed every base-model component on CUDA device 0."""
    raw_map = getattr(model, "hf_device_map", None)
    if not isinstance(raw_map, Mapping) or not raw_map:
        raise InvariantViolation(
            "QLoRA preflight base model has no non-empty hf_device_map",
            hint="The automatic loader placement cannot be audited; do not use this run.",
        )

    normalized: dict[str, str] = {}
    invalid: list[str] = []
    for raw_name, placement in raw_map.items():
        name = str(raw_name)
        if isinstance(placement, bool):
            valid = False
        elif isinstance(placement, int):
            valid = placement == 0
        else:
            valid = str(placement).strip().lower() in {"0", "cuda", "cuda:0"}
        if not valid:
            invalid.append(f"{name}={placement!r}")
        else:
            normalized[name] = "cuda:0"
    if invalid:
        raise InvariantViolation(
            "QLoRA preflight rejected non-CUDA-0 model placement: " + ", ".join(invalid),
            hint=(
                "CPU, disk, and other-GPU offload do not prove the registered single-L40S "
                "resource shape."
            ),
        )
    return dict(sorted(normalized.items()))


def _peak_memory(torch: Any) -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {"allocated_mib": None, "reserved_mib": None}
    return {
        "allocated_mib": int(torch.cuda.max_memory_allocated() // (1024 * 1024)),
        "reserved_mib": int(torch.cuda.max_memory_reserved() // (1024 * 1024)),
    }


def _target_module_counts(model: Any) -> dict[str, int]:
    counts = {module: 0 for module in TARGET_MODULES}
    for name, layer in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in counts and hasattr(layer, "lora_A"):
            counts[leaf] += 1
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise InvariantViolation(
            "QLoRA preflight did not attach LoRA to target modules: " + ", ".join(missing)
        )
    return counts


def _trainable_parameter_counts(model: Any) -> dict[str, int | float]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    if trainable <= 0:
        raise InvariantViolation("QLoRA preflight model has no trainable adapter parameters")
    return {
        "trainable": trainable,
        "total_visible": total,
        "trainable_fraction": trainable / total,
    }


def _finalize_report(
    output_dir: Path,
    report: dict[str, object],
) -> dict[str, object]:
    report_path = output_dir / "preflight_report.json"
    report.pop("artifact_hashes", None)
    # Persist the terminal state before hashing. If hashing or manifest writing
    # itself fails, the diagnostic still leaves an inspectable failure/success
    # record rather than disappearing completely.
    validate_preflight_report(report, require_artifact_hashes=False)
    _write_json(report_path, report)
    excluded = frozenset({"artifact_manifest.json", "preflight_report.json"})
    digests = artifact_digests(output_dir, excluded_relative_paths=excluded)
    manifest = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "diagnostic_only": True,
        "artifacts": digests,
    }
    validate_artifact_manifest(manifest, artifact_root=output_dir)
    manifest_path = output_dir / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    report["artifact_hashes"] = {
        "manifest_path": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "files": digests,
        "boundary": ARTIFACT_HASH_BOUNDARY,
    }
    validate_preflight_artifacts(
        report,
        manifest,
        artifact_root=output_dir,
    )
    _write_json(report_path, report)
    return report


def run_qlora_shape_preflight(
    config: SommelierConfig,
    *,
    config_yaml: str,
    output_dir: Path,
    run_id: str,
    source: SourceProvenance,
) -> dict[str, object]:
    """Executes one full-shape QLoRA optimizer step and one eval forward.

    Heavy ML imports stay inside this optional-runtime boundary. Once the safe
    run id has reserved a new output directory, config/provenance/runtime/
    training failures (including CUDA OOM) write a redacted terminal report
    with any peak-memory measurement and hashes of the regular files present
    before re-raising.
    """
    validate_run_id(run_id)
    if output_dir.exists():
        raise UserInputError(
            f"QLoRA preflight output already exists: {output_dir}",
            hint="Use a new --run-id; diagnostic attempts are never overwritten or resumed.",
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as error:
        raise UserInputError(
            f"QLoRA preflight output already exists: {output_dir}",
            hint="Use a new --run-id; diagnostic attempts are never overwritten or resumed.",
        ) from error
    report: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "diagnostic_only": True,
        "release_evidence_eligible": False,
        "boundary": DIAGNOSTIC_BOUNDARY,
        "provider_accessed": False,
        "dataset_accessed": False,
        "config_sha256": hashlib.sha256(config_yaml.encode("utf-8")).hexdigest(),
        "contract": preflight_contract(),
    }

    torch: Any | None = None
    started = time.monotonic()
    try:
        validate_preflight_report(
            report,
            require_terminal=False,
            require_artifact_hashes=False,
        )
        _write_json(output_dir / "preflight_report.json", report)
        validate_preflight_config(config)
        validate_config_yaml_identity(config, config_yaml)
        validate_source_provenance(source)
        report["resolved_config"] = config.model_dump(mode="json")
        report["source_code"] = {
            **source,
            "boundary": redact_text(source["boundary"]),
        }
        _write_text(output_dir / "config.input.yaml", config_yaml)

        import torch as imported_torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )

        torch = imported_torch
        versions = collect_runtime_versions()
        validate_runtime_versions(versions)
        report["versions"] = versions
        hardware = _hardware_metadata(torch)
        report["hardware"] = hardware
        torch.cuda.reset_peak_memory_stats()

        tokenizer = cast(
            PreflightTokenizer,
            AutoTokenizer.from_pretrained(
                config.model.base_model_id,
                **qlora_tokenizer_load_kwargs(config),
            ),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise InvariantViolation("pinned tokenizer has neither a pad token nor an EOS token")

        splits, token_lengths = build_synthetic_formatted_splits(tokenizer, config)
        formatted_dir = output_dir / "formatted"
        _write_jsonl(formatted_dir / "train.jsonl", splits["train"])
        _write_jsonl(formatted_dir / "validation.jsonl", splits["validation"])
        report["synthetic_dataset"] = {
            "schema_version": FORMATTED_EXAMPLE_SCHEMA,
            "source": "generated in-container; no dataset API or published rows",
            "rows": {"train": len(splits["train"]), "validation": len(splits["validation"])},
            "languages": {
                split: {
                    language: sum(row["language"] == language for row in rows)
                    for language in ("en", "he")
                }
                for split, rows in splits.items()
            },
            "token_lengths": token_lengths,
        }

        quantization = BitsAndBytesConfig(**qlora_quantization_kwargs(torch))
        model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_id,
            **qlora_model_load_kwargs(
                config,
                quantization_config=quantization,
            ),
        )
        hf_device_map = validate_single_cuda_device_map(model)
        configure_qlora_base_model(model)
        model = prepare_model_for_kbit_training(
            model,
            **qlora_kbit_preparation_kwargs(),
        )
        model = get_peft_model(
            model,
            LoraConfig(**qlora_lora_kwargs(config)),
        )
        if not bool(getattr(model, "is_gradient_checkpointing", False)):
            raise InvariantViolation("gradient checkpointing is not active on the QLoRA model")
        report["model_wiring"] = {
            "target_module_counts": _target_module_counts(model),
            "parameters": _trainable_parameter_counts(model),
            "gradient_checkpointing_active": True,
            "use_cache": bool(model.config.use_cache),
            "hf_device_map": hf_device_map,
        }

        collator = CompletionOnlyCollator(
            cast(Any, tokenizer),
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
        batch_phase = "train"
        observed_batches = {"train": 0, "eval": 0}

        def torch_collate(batch: list[dict[str, object]]) -> dict[str, object]:
            if len(batch) != config.train.per_device_batch_size:
                raise InvariantViolation(
                    f"QLoRA preflight {batch_phase} batch has {len(batch)} rows; "
                    f"expected {config.train.per_device_batch_size}"
                )
            observed_batches[batch_phase] += 1
            collated = collator(batch)
            return {
                "input_ids": torch.tensor(collated["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(collated["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(collated["labels"], dtype=torch.long),
            }

        trainer_state_dir = output_dir / "trainer_state"
        training_kwargs = qlora_training_argument_kwargs(
            config,
            output_dir=trainer_state_dir,
        )
        training_kwargs.update(
            {
                "max_steps": OPTIMIZER_STEPS,
                "eval_strategy": "no",
                "disable_tqdm": True,
            }
        )
        arguments = TrainingArguments(**training_kwargs)
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=splits["train"],
            eval_dataset=splits["validation"],
            data_collator=torch_collate,
        )
        train_output = trainer.train()
        if int(trainer.state.global_step) != OPTIMIZER_STEPS:
            raise InvariantViolation(
                f"QLoRA preflight completed {trainer.state.global_step} optimizer steps; "
                f"expected {OPTIMIZER_STEPS}"
            )
        if observed_batches["train"] != EXPECTED_TRAIN_MICROBATCHES:
            raise InvariantViolation(
                f"QLoRA preflight observed {observed_batches['train']} train microbatches; "
                f"expected {EXPECTED_TRAIN_MICROBATCHES}"
            )
        train_loss = float(train_output.metrics.get("train_loss", math.nan))
        if not math.isfinite(train_loss):
            raise InvariantViolation("QLoRA preflight optimizer step did not return finite loss")
        batch_phase = "eval"
        eval_metrics = dict(trainer.evaluate())
        if observed_batches["eval"] != EXPECTED_EVAL_FORWARD_BATCHES:
            raise InvariantViolation(
                f"QLoRA preflight observed {observed_batches['eval']} eval forward batches; "
                f"expected {EXPECTED_EVAL_FORWARD_BATCHES}"
            )
        eval_loss = float(eval_metrics.get("eval_loss", math.nan))
        if not math.isfinite(eval_loss):
            raise InvariantViolation("QLoRA preflight eval forward did not return finite loss")
        torch.cuda.synchronize()

        adapter_dir = output_dir / "adapter"
        model.save_pretrained(str(adapter_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(adapter_dir))
        _write_json(output_dir / "trainer_log_history.json", trainer.state.log_history)
        _write_json(output_dir / "eval_metrics.json", eval_metrics)
        report["execution"] = {
            "optimizer_steps": int(trainer.state.global_step),
            "train_microbatches": observed_batches["train"],
            "train_batch_size_per_device": config.train.per_device_batch_size,
            "eval_forward_batches": observed_batches["eval"],
            "eval_batch_size_per_device": config.train.per_device_batch_size,
            "train_metrics": dict(train_output.metrics),
            "eval_metrics": eval_metrics,
        }
        report["peak_gpu_memory_mib"] = _peak_memory(torch)
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["status"] = "succeeded"
        return _finalize_report(output_dir, report)
    except Exception as error:
        report.pop("artifact_hashes", None)
        report["status"] = "failed"
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if torch is None:
            report["peak_gpu_memory_mib"] = {
                "allocated_mib": None,
                "reserved_mib": None,
            }
        else:
            try:
                report["peak_gpu_memory_mib"] = _peak_memory(torch)
            except Exception as measurement_error:
                report["peak_gpu_memory_mib"] = {
                    "allocated_mib": None,
                    "reserved_mib": None,
                }
                report["peak_memory_measurement_failure"] = {
                    "type": type(measurement_error).__name__,
                    "message": redact_text(str(measurement_error))[:1000],
                }
        message = redact_text(str(error))[:1000]
        report["failure"] = {
            "type": type(error).__name__,
            "message": message,
            "cuda_out_of_memory": (
                type(error).__name__ == "OutOfMemoryError" or "out of memory" in message.lower()
            ),
        }
        _finalize_report(output_dir, report)
        raise
