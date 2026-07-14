from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

from sommelier.analysis.tokenization import (
    TOKENIZER_TAX_RECORD_SCHEMA,
    TOKENIZER_TAX_RECORDS_FILENAME,
    TOKENIZER_TAX_REPORT_FILENAME,
    TOKENIZER_TAX_REPORT_SCHEMA,
)
from sommelier.artifacts import (
    read_json_with_schema,
    read_jsonl_with_schema,
    sha256_file,
    write_artifact_atomic,
)
from sommelier.config import SommelierConfig, compute_config_digest, load_config
from sommelier.errors import EvaluationError, UserInputError
from sommelier.evaluation.data_provenance import (
    HEBREW_V3_V1_ADAPTER_ID,
    HEBREW_V3_V1_ADAPTER_REVISION,
    validate_hebrew_v3_data_provenance,
)
from sommelier.evaluation.generate import (
    GENERATION_SCHEMA,
    IMMUTABLE_HF_REVISION,
    INFERENCE_TELEMETRY_FILENAME,
    INFERENCE_TELEMETRY_SCHEMA,
    ModelKind,
    adapter_tree_sha256,
    evaluation_stage,
    slice_filename,
)
from sommelier.evaluation.metrics import METRIC_NAMES, ScoredRecord, compute_metrics
from sommelier.evaluation.parse import parse_tool_call
from sommelier.evaluation.report import (
    EVALUATION_REPORT_SCHEMA,
    REPORT_FILENAME,
    build_scored_records,
    find_run_layout,
    paired_set_digest,
    prompt_set_digest,
)
from sommelier.evaluation.statistics import (
    exact_mcnemar_full_call,
    paired_bootstrap_intervals,
)
from sommelier.evaluation.tco import (
    ArtifactEvidence,
    InferenceArmInput,
    LocalAdapterInput,
    TCOIdentity,
    TokenizerTaxInput,
    TrainingInput,
    build_sovereign_tco_evidence,
)
from sommelier.formatting.chat import FORMATTED_EXAMPLE_SCHEMA
from sommelier.manifests import get_git_commit, get_git_worktree_clean
from sommelier.runtime_metadata import RUNTIME_METADATA_FILENAME, RUNTIME_METADATA_SCHEMA
from sommelier.training.metrics import (
    METRICS_FILENAME,
    TRAINING_METRIC_SCHEMA,
)

EXPERIMENT_REPORT_SCHEMA: Final = "sommelier.experiment_report.v1"
EXPERIMENT_REPORT_FILENAME: Final = "experiment_report.json"
HEBREW_V3_PREREGISTRATION_SCHEMA: Final = "sommelier.hebrew_v3_preregistration.v1"
HEBREW_V3_ENGLISH_NON_INFERIORITY_MARGIN: Final = 0.01
HEBREW_V3_BOOTSTRAP_SEED: Final = 42
HEBREW_V3_BOOTSTRAP_RESAMPLES: Final = 2000
REQUIRED_SLICES: Final = frozenset({"en", "he"})
MODEL_IDENTITY_FIELDS: Final = (
    "base_model_id",
    "base_model_revision",
    "tokenizer_id",
    "tokenizer_revision",
)


@dataclass(frozen=True)
class _LoadedArm:
    name: str
    eval_dir: Path
    run_dir: Path
    artifact_root: Path
    report_path: Path
    report: dict[str, Any]
    scored: dict[str, list[ScoredRecord]]
    example_ids: dict[str, list[str]]
    prompt_set_sha256: dict[str, str]
    paired_set_sha256: dict[str, str]
    artifact_paths: dict[str, Path]


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(
            f"{context} must be a JSON object",
            hint="Regenerate the evaluation report with the current pipeline.",
        )
    return cast("dict[str, Any]", value)


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(
            f"{context} must be a non-empty string",
            hint="Regenerate the evaluation report with complete provenance.",
        )
    return value


def _sequence(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvaluationError(
            f"{context} must be a JSON array",
            hint="Regenerate the evaluation evidence with the current pipeline.",
        )
    return value


def _ordered_id_digest(example_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(example_ids).encode("utf-8")).hexdigest()


def _validate_adapter_source(name: str, value: object) -> None:
    adapter_source = _mapping(value, context=f"{name} adapter_source")
    _string(adapter_source.get("source"), context=f"{name} adapter_source.source")
    if adapter_source.get("revision_is_immutable") is not True:
        raise EvaluationError(
            f"{name} adapter arm does not have an immutable adapter identity",
            hint=(
                "Use an exact Hugging Face commit revision or a local adapter "
                "recorded with its tree_sha256."
            ),
        )

    kind = adapter_source.get("kind")
    if kind == "huggingface_repo":
        revision = adapter_source.get("revision")
        if (
            not isinstance(revision, str)
            or IMMUTABLE_HF_REVISION.fullmatch(revision) is None
            or adapter_source.get("tree_sha256") is not None
            or adapter_source.get("artifact_path") is not None
        ):
            raise EvaluationError(
                f"{name} adapter arm has an invalid Hugging Face adapter identity",
                hint="Bind the adapter to an exact 40-64 character hexadecimal commit.",
            )
        return

    if kind == "local_directory":
        tree_sha256 = adapter_source.get("tree_sha256")
        artifact_path = adapter_source.get("artifact_path")
        if (
            not isinstance(tree_sha256, str)
            or len(tree_sha256) != 64
            or any(character not in "0123456789abcdef" for character in tree_sha256)
            or adapter_source.get("revision") is not None
            or not isinstance(artifact_path, str)
            or not artifact_path.startswith("runs/")
            or artifact_path.startswith("/")
            or ".." in artifact_path.split("/")
        ):
            raise EvaluationError(
                f"{name} adapter arm has an invalid local adapter identity",
                hint=(
                    "Record the local adapter with its exact tree_sha256 and canonical "
                    "runs/<run-id>/... artifact_path."
                ),
            )
        return

    raise EvaluationError(
        f"{name} adapter arm has an unsupported adapter source kind",
        hint="Use a huggingface_repo or local_directory adapter identity.",
    )


def _validate_inputs(
    english_non_inferiority_margin: float,
    seed: int,
    resamples: int,
) -> None:
    if (
        isinstance(english_non_inferiority_margin, bool)
        or not math.isfinite(english_non_inferiority_margin)
        or english_non_inferiority_margin <= 0.0
    ):
        raise UserInputError(
            "english_non_inferiority_margin must be a positive finite number",
            hint="Predeclare the tolerated absolute English full-call regression.",
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise UserInputError("seed must be an integer")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise UserInputError("resamples must be a positive integer")
    supplied = {
        "english_non_inferiority_margin": english_non_inferiority_margin,
        "seed": seed,
        "resamples": resamples,
    }
    preregistered = {
        "english_non_inferiority_margin": HEBREW_V3_ENGLISH_NON_INFERIORITY_MARGIN,
        "seed": HEBREW_V3_BOOTSTRAP_SEED,
        "resamples": HEBREW_V3_BOOTSTRAP_RESAMPLES,
    }
    if supplied != preregistered:
        raise UserInputError(
            "experiment parameters do not match the committed Hebrew v3 preregistration",
            hint=(
                "Use --english-non-inferiority-margin 0.01 --seed 42 "
                "--resamples 2000; do not tune claim gates after viewing outcomes."
            ),
        )


def _finalizer_source_identity() -> dict[str, object]:
    revision = get_git_commit()
    clean = get_git_worktree_clean()
    if IMMUTABLE_HF_REVISION.fullmatch(revision) is None or clean is not True:
        raise EvaluationError(
            "experiment finalizer is not running from a clean immutable source revision",
            hint=(
                "Checkout the exact clean commit used by the three full evaluation runs "
                "before creating the terminal claim artifact."
            ),
        )
    return {
        "git_commit": revision,
        "working_tree_clean": True,
        "boundary": "Observed before loading experiment outcome artifacts.",
    }


def _validate_pair_digest(
    report: dict[str, Any],
    examples: dict[str, list[dict[str, Any]]],
) -> str:
    reference_by_id = {
        _string(example.get("example_id"), context="English example_id"): example
        for example in examples["en"]
    }
    entries: list[dict[str, str]] = []
    seen_roots: set[str] = set()
    for target in examples["he"]:
        target_id = _string(target.get("example_id"), context="Hebrew example_id")
        root_id = _string(
            target.get("source_example_id"),
            context=f"Hebrew example {target_id} source_example_id",
        )
        if root_id in seen_roots:
            raise EvaluationError(
                f"Hebrew paired cohort repeats English example {root_id}",
                hint="Keep exactly one Hebrew evaluation row per English root.",
            )
        seen_roots.add(root_id)
        root = reference_by_id.get(root_id)
        if root is None:
            raise EvaluationError(
                f"Hebrew example {target_id} references missing English root {root_id}",
                hint="Use the same paired test cohort for every experiment arm.",
            )
        entries.append(
            {
                "reference_example_id": root_id,
                "target_example_id": target_id,
                "reference_prompt_sha256": _string(
                    root.get("prompt_sha256"), context=f"English prompt digest {root_id}"
                ),
                "target_prompt_sha256": _string(
                    target.get("prompt_sha256"), context=f"Hebrew prompt digest {target_id}"
                ),
            }
        )

    paired_slices = _mapping(report.get("paired_slices"), context="paired_slices")
    hebrew_pair = _mapping(paired_slices.get("he"), context="paired_slices.he")
    reported_digest = _string(
        hebrew_pair.get("pair_set_sha256"), context="paired_slices.he.pair_set_sha256"
    )
    actual_digest = paired_set_digest(entries)
    if reported_digest != actual_digest:
        raise EvaluationError(
            "Hebrew paired cohort digest does not match formatted artifacts",
            hint="Regenerate the evaluation report from the stored formatted split.",
        )
    if hebrew_pair.get("pairs") != len(entries):
        raise EvaluationError(
            "Hebrew paired cohort count does not match formatted artifacts",
            hint="Regenerate the evaluation report from the stored formatted split.",
        )
    return actual_digest


def _load_arm(name: str, eval_dir: Path, *, expected_kind: ModelKind) -> _LoadedArm:
    resolved_eval_dir = eval_dir.resolve()
    report_path = resolved_eval_dir / REPORT_FILENAME
    if not report_path.exists():
        raise UserInputError(
            f"evaluation report not found: {report_path}",
            hint="Pass an eval directory containing evaluation_report.json.",
        )
    report = read_json_with_schema(
        report_path,
        expected_schema=EVALUATION_REPORT_SCHEMA,
    )
    artifact_root, run_dir, inferred_run_id = find_run_layout(resolved_eval_dir)
    if report.get("run_id") != inferred_run_id:
        raise EvaluationError(
            f"{name} report run_id does not match its artifact path",
            hint="Keep each report inside the run that produced it.",
        )
    if report.get("model_kind") != expected_kind:
        raise EvaluationError(
            f"{name} arm must contain {expected_kind} generations",
            hint="Pass base first, followed by the v1 and v3 adapter eval directories.",
        )
    if expected_kind == "adapter":
        _validate_adapter_source(name, report.get("adapter_source"))

    slices = _mapping(report.get("slices"), context=f"{name} slices")
    slice_names = set(slices)
    missing = sorted(REQUIRED_SLICES - slice_names)
    if missing:
        raise EvaluationError(
            f"{name} arm is missing required slice(s): {', '.join(missing)}",
            hint="Evaluate every arm on both English and Hebrew.",
        )

    formatted_path = run_dir / "formatted" / "test.jsonl"
    if not formatted_path.exists():
        raise UserInputError(
            f"formatted test artifact not found: {formatted_path}",
            hint="Keep the formatted test split with every evaluation run.",
        )
    if report.get("test_split_sha256") != sha256_file(formatted_path):
        raise EvaluationError(
            f"{name} test split digest does not match its formatted artifact",
            hint="Do not edit evaluation inputs after report generation.",
        )
    formatted = read_jsonl_with_schema(
        formatted_path,
        expected_schema=FORMATTED_EXAMPLE_SCHEMA,
    )
    examples: dict[str, list[dict[str, Any]]] = {language: [] for language in slice_names}
    for example in formatted:
        language = example.get("language")
        if isinstance(language, str) and language in examples:
            examples[language].append(example)

    scored: dict[str, list[ScoredRecord]] = {}
    example_ids: dict[str, list[str]] = {}
    prompt_digests: dict[str, str] = {}
    artifact_paths: dict[str, Path] = {
        "evaluation_report": report_path,
        "formatted_test": formatted_path,
    }
    report_decoding = _mapping(report.get("decoding"), context=f"{name} decoding")
    if report.get("parser_version") != "sommelier.parser.v1":
        raise EvaluationError(
            f"{name} report uses an unsupported parser version",
            hint="Regenerate the report with sommelier.parser.v1.",
        )
    for language in sorted(slice_names):
        slice_report = _mapping(slices[language], context=f"{name} slice {language}")
        language_examples = examples[language]
        if not language_examples:
            raise EvaluationError(
                f"{name} formatted test split has no {language} examples",
                hint="Evaluate non-empty, identical slices in every arm.",
            )
        ids = [
            _string(example.get("example_id"), context=f"{name} {language} example_id")
            for example in language_examples
        ]
        if len(set(ids)) != len(ids):
            raise EvaluationError(f"{name} {language} slice contains duplicate example IDs")

        actual_prompt_digest = prompt_set_digest(
            [
                _string(
                    example.get("prompt_sha256"),
                    context=f"{name} {language} prompt digest",
                )
                for example in language_examples
            ]
        )
        if slice_report.get("prompt_set_sha256") != actual_prompt_digest:
            raise EvaluationError(
                f"{name} {language} prompt digest does not match formatted artifacts",
                hint="Regenerate the report from the stored formatted test split.",
            )
        if slice_report.get("examples") != len(language_examples):
            raise EvaluationError(
                f"{name} {language} example count does not match formatted artifacts"
            )

        generations_path = resolved_eval_dir / slice_filename(language)
        generations = read_jsonl_with_schema(
            generations_path,
            expected_schema=GENERATION_SCHEMA,
        )
        generation_ids = [
            _string(
                generation.get("example_id"),
                context=f"{name} {language} generation example_id",
            )
            for generation in generations
        ]
        if generation_ids != ids:
            raise EvaluationError(
                f"{name} {language} generation example IDs differ from formatted order",
                hint="Regenerate outputs over the stored ordered test slice.",
            )
        for generation in generations:
            if generation.get("model_kind") != expected_kind:
                raise EvaluationError(f"{name} {language} generations have the wrong model kind")
            if generation.get("language") != language:
                raise EvaluationError(f"{name} {language} generations carry a different language")
            if generation.get("decoding") != report_decoding:
                raise EvaluationError(
                    f"{name} {language} generation decoding does not match the report",
                    hint="Regenerate the evaluation report from the stored generations.",
                )
            raw_text = generation.get("raw_text")
            if not isinstance(raw_text, str):
                raise EvaluationError(f"{name} {language} generation raw_text must be a string")
            reparsed_call, reparsed_status = parse_tool_call(raw_text)
            if (
                generation.get("parsed_call") != reparsed_call
                or generation.get("parse_status") != reparsed_status
            ):
                raise EvaluationError(
                    f"{name} {language} stored parse result does not match raw_text",
                    hint=(
                        "Regenerate generations or restore the parser-produced "
                        "parsed_call and parse_status fields."
                    ),
                )

        reported_artifact = _string(
            slice_report.get("generation_artifact"),
            context=f"{name} {language} generation_artifact",
        )
        if (artifact_root / reported_artifact).resolve() != generations_path.resolve():
            raise EvaluationError(
                f"{name} {language} generation artifact path does not match the report"
            )

        language_scored = build_scored_records(language_examples, generations)
        actual_metrics = compute_metrics(language_scored)
        if slice_report.get("metrics") != actual_metrics:
            raise EvaluationError(
                f"{name} {language} metrics do not match generation artifacts",
                hint="Regenerate the evaluation report instead of editing metrics.",
            )
        scored[language] = language_scored
        example_ids[language] = ids
        prompt_digests[language] = actual_prompt_digest
        artifact_paths[f"generations.{language}"] = generations_path

    paired_digest = _validate_pair_digest(report, examples)
    return _LoadedArm(
        name=name,
        eval_dir=resolved_eval_dir,
        run_dir=run_dir,
        artifact_root=artifact_root,
        report_path=report_path,
        report=report,
        scored=scored,
        example_ids=example_ids,
        prompt_set_sha256=prompt_digests,
        paired_set_sha256={"he": paired_digest},
        artifact_paths=artifact_paths,
    )


def _shared_identity(arms: list[_LoadedArm]) -> dict[str, Any]:
    reference = arms[0]
    reference_slices = set(reference.scored)
    reference_model = _mapping(
        reference.report.get("model_identity"), context="base model_identity"
    )
    model_identity = {
        field: _string(reference_model.get(field), context=f"model_identity.{field}")
        for field in MODEL_IDENTITY_FIELDS
    }
    for field in ("base_model_revision", "tokenizer_revision"):
        if IMMUTABLE_HF_REVISION.fullmatch(model_identity[field]) is None:
            raise EvaluationError(
                f"experiment rejected: mutable model_identity.{field}",
                hint="Evaluate every arm with exact immutable model and tokenizer commits.",
            )
    for arm in arms[1:]:
        if set(arm.scored) != reference_slices:
            raise EvaluationError(
                "experiment arms evaluated different slice sets",
                hint="Use identical evaluation slices for base, v1, and v3.",
            )
        candidate_model = _mapping(
            arm.report.get("model_identity"), context=f"{arm.name} model_identity"
        )
        for field, expected in model_identity.items():
            if candidate_model.get(field) != expected:
                raise EvaluationError(
                    f"experiment rejected: mismatched model_identity.{field}",
                    hint="Evaluate every arm with the same pinned base and tokenizer.",
                )
        for field in ("split", "parser_version", "decoding", "test_split_sha256"):
            if arm.report.get(field) != reference.report.get(field):
                raise EvaluationError(
                    f"experiment rejected: mismatched {field}",
                    hint="Regenerate every arm under one evaluation contract.",
                )
        for language in sorted(reference_slices):
            if arm.prompt_set_sha256[language] != reference.prompt_set_sha256[language]:
                raise EvaluationError(f"experiment rejected: mismatched {language} prompt cohort")
            if arm.example_ids[language] != reference.example_ids[language]:
                raise EvaluationError(f"experiment rejected: mismatched {language} example IDs")
        if arm.paired_set_sha256 != reference.paired_set_sha256:
            raise EvaluationError("experiment rejected: mismatched paired cohort digest")

    return {
        "model_identity": model_identity,
        "split": reference.report["split"],
        "parser_version": reference.report["parser_version"],
        "decoding": reference.report["decoding"],
        "test_split_sha256": reference.report["test_split_sha256"],
        "slices": {
            language: {
                "examples": len(reference.example_ids[language]),
                "example_ids_sha256": _ordered_id_digest(reference.example_ids[language]),
                "prompt_set_sha256": reference.prompt_set_sha256[language],
            }
            for language in sorted(reference_slices)
        },
        "paired_cohorts": {
            "he": {
                "pairs": len(reference.example_ids["he"]),
                "pair_set_sha256": reference.paired_set_sha256["he"],
            }
        },
    }


def _arm_payload(arm: _LoadedArm) -> dict[str, Any]:
    scored_all = [record for language in sorted(arm.scored) for record in arm.scored[language]]

    def portable_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(arm.artifact_root.resolve()).as_posix()
        except ValueError as error:
            raise EvaluationError(
                f"experiment artifact path escapes artifact root: {path}",
                hint="Keep every arm artifact inside its checksummed run bundle.",
            ) from error

    return {
        "run_id": arm.report["run_id"],
        "model_kind": arm.report["model_kind"],
        "config_sha256": arm.report["config_sha256"],
        "adapter_source": arm.report.get("adapter_source"),
        "metrics": {
            "overall": compute_metrics(scored_all),
            "slices": {
                language: compute_metrics(arm.scored[language]) for language in sorted(arm.scored)
            },
        },
        "artifacts": {
            key: {
                "path": portable_path(path),
                "sha256": sha256_file(path),
            }
            for key, path in sorted(arm.artifact_paths.items())
        },
    }


def _artifact_evidence(
    arm: _LoadedArm,
    path: Path,
    *,
    kind: str,
    schema_version: str,
) -> ArtifactEvidence:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(arm.artifact_root)
    except ValueError as error:
        raise EvaluationError(
            f"TCO evidence path escapes artifact root: {path}",
            hint="Keep all evidence inside the run artifact root.",
        ) from error
    if not resolved.is_file():
        raise UserInputError(
            f"TCO evidence artifact not found: {resolved}",
            hint="Keep the measured artifact with the v3 run.",
        )
    return ArtifactEvidence(
        path=relative.as_posix(),
        kind=kind,
        schema_version=schema_version,
        sha256=sha256_file(resolved),
        bytes=resolved.stat().st_size,
    )


def _resolved_tco_config(
    arm: _LoadedArm,
) -> tuple[SommelierConfig | None, ArtifactEvidence | None]:
    path = arm.run_dir / "config.resolved.yaml"
    if not path.exists():
        return None, None
    raw = path.read_text(encoding="utf-8")
    digest = compute_config_digest(raw)
    if digest != arm.report.get("config_sha256"):
        raise EvaluationError(
            "v3 resolved config digest does not match its evaluation report",
            hint="Do not combine evaluation and training evidence from different runs.",
        )
    config = load_config(path)
    model_identity = _mapping(arm.report.get("model_identity"), context="v3 model_identity")
    expected_model = {
        "base_model_id": config.model.base_model_id,
        "base_model_revision": config.model.base_model_revision,
        "tokenizer_id": config.model.base_model_id,
        "tokenizer_revision": config.model.tokenizer_revision,
    }
    for field, expected in expected_model.items():
        if model_identity.get(field) != expected:
            raise EvaluationError(
                f"v3 resolved config does not match model_identity.{field}",
                hint="Use the resolved config that produced the v3 evaluation.",
            )
    return (
        config,
        _artifact_evidence(
            arm,
            path,
            kind="config",
            schema_version="sommelier.config.v2",
        ),
    )


def _tokenizer_tco_input(arm: _LoadedArm) -> TokenizerTaxInput | None:
    directory = arm.run_dir / "analysis" / "tokenization"
    report_path = directory / TOKENIZER_TAX_REPORT_FILENAME
    records_path = directory / TOKENIZER_TAX_RECORDS_FILENAME
    manifest_path = arm.run_dir / "tokenization_manifest.json"
    present = [path.exists() for path in (report_path, records_path, manifest_path)]
    if not any(present):
        return None
    if not all(present):
        raise EvaluationError(
            "v3 tokenizer-tax evidence is incomplete",
            hint="Keep the report, records, and tokenization manifest together.",
        )
    report = read_json_with_schema(
        report_path,
        expected_schema=TOKENIZER_TAX_REPORT_SCHEMA,
    )
    records = read_jsonl_with_schema(
        records_path,
        expected_schema=TOKENIZER_TAX_RECORD_SCHEMA,
    )
    manifest = read_json_with_schema(
        manifest_path,
        expected_schema="sommelier.manifest.v1",
    )
    formatted_inputs = {
        split: _artifact_evidence(
            arm,
            arm.run_dir / "formatted" / f"{split}.jsonl",
            kind="formatted_split",
            schema_version=FORMATTED_EXAMPLE_SCHEMA,
        )
        for split in ("train", "validation", "test")
    }
    return TokenizerTaxInput(
        report=report,
        records=records,
        manifest=manifest,
        report_artifact=_artifact_evidence(
            arm,
            report_path,
            kind="tokenizer_tax_report",
            schema_version=TOKENIZER_TAX_REPORT_SCHEMA,
        ),
        records_artifact=_artifact_evidence(
            arm,
            records_path,
            kind="tokenizer_tax_records",
            schema_version=TOKENIZER_TAX_RECORD_SCHEMA,
        ),
        manifest_artifact=_artifact_evidence(
            arm,
            manifest_path,
            kind="manifest",
            schema_version="sommelier.manifest.v1",
        ),
        formatted_inputs=formatted_inputs,
    )


def _json_object(path: Path, *, context: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, context=context)


def _training_tco_input(arm: _LoadedArm) -> TrainingInput:
    runtime_path = arm.run_dir / RUNTIME_METADATA_FILENAME
    metrics_path = arm.run_dir / "train" / METRICS_FILENAME
    manifest_path = arm.run_dir / "train_manifest.json"

    observed_runtime = (
        _json_object(runtime_path, context="v3 runtime metadata") if runtime_path.exists() else None
    )
    metrics = (
        read_jsonl_with_schema(metrics_path, expected_schema=TRAINING_METRIC_SCHEMA)
        if metrics_path.exists()
        else None
    )
    metrics_artifact = (
        _artifact_evidence(
            arm,
            metrics_path,
            kind="training_metrics",
            schema_version=TRAINING_METRIC_SCHEMA,
        )
        if metrics is not None
        else None
    )
    manifest = (
        read_json_with_schema(
            manifest_path,
            expected_schema="sommelier.manifest.v1",
        )
        if manifest_path.exists()
        else None
    )
    manifest_artifact = (
        _artifact_evidence(
            arm,
            manifest_path,
            kind="manifest",
            schema_version="sommelier.manifest.v1",
        )
        if manifest is not None
        else None
    )
    runtime_stages = (
        _mapping(observed_runtime.get("stages"), context="v3 runtime stages")
        if observed_runtime is not None
        else {}
    )
    runtime = (
        observed_runtime
        if "train" in runtime_stages or metrics is not None or manifest is not None
        else None
    )
    runtime_artifact = (
        _artifact_evidence(
            arm,
            runtime_path,
            kind="runtime_metadata",
            schema_version=RUNTIME_METADATA_SCHEMA,
        )
        if runtime is not None
        else None
    )

    adapter_identity_value = arm.report.get("adapter_source")
    adapter_identity = (
        _mapping(adapter_identity_value, context="v3 adapter_source")
        if adapter_identity_value is not None
        else None
    )
    local_adapter: LocalAdapterInput | None = None
    if adapter_identity is not None and adapter_identity.get("kind") == "local_directory":
        artifact_path = _string(
            adapter_identity.get("artifact_path"),
            context="v3 local adapter artifact_path",
        )
        source = (arm.artifact_root / artifact_path).resolve()
        expected = (arm.run_dir / "train" / "adapter").resolve()
        if source == expected and source.is_dir():
            files = tuple(
                _artifact_evidence(
                    arm,
                    path,
                    kind="adapter_weights",
                    schema_version="",
                )
                for path in sorted(source.rglob("*"))
                if path.is_file() and not path.is_symlink()
            )
            local_adapter = LocalAdapterInput(
                tree_sha256=adapter_tree_sha256(source),
                files=files,
            )

    return TrainingInput(
        runtime_metadata=runtime,
        runtime_artifact=runtime_artifact,
        metrics=metrics,
        metrics_artifact=metrics_artifact,
        manifest=manifest,
        manifest_artifact=manifest_artifact,
        adapter_identity=adapter_identity,
        local_adapter=local_adapter,
    )


def _validate_preregistered_adapter_arms(
    v1: _LoadedArm,
    v3: _LoadedArm,
) -> None:
    """Validate the preregistered v1 baseline and locally trained v3 adapter.

    The experiment is specifically a comparison against one immutable published
    baseline.  Its v3 candidate must be the adapter produced by the v3 run's
    succeeded train stage, rather than another published checkpoint or an
    arbitrary local directory.  This check intentionally happens independently
    of metric computation so adapter substitution cannot inherit the
    preregistered claims.
    """
    v1_identity = _mapping(v1.report.get("adapter_source"), context="v1 adapter_source")
    v3_identity = _mapping(v3.report.get("adapter_source"), context="v3 adapter_source")
    if v1_identity == v3_identity or v1_identity.get("source") == v3_identity.get("source"):
        raise EvaluationError(
            "preregistered v1 and v3 adapters must be distinct",
            hint="Use the fixed published v1 baseline and the v3 run's own adapter.",
        )

    required_v1_identity: dict[str, object] = {
        "source": HEBREW_V3_V1_ADAPTER_ID,
        "revision": HEBREW_V3_V1_ADAPTER_REVISION,
        "kind": "huggingface_repo",
        "tree_sha256": None,
        "artifact_path": None,
        "revision_is_immutable": True,
    }
    if v1_identity != required_v1_identity:
        raise EvaluationError(
            "v1 adapter does not match the committed Hebrew v3 baseline",
            hint=(f"Use {HEBREW_V3_V1_ADAPTER_ID}@{HEBREW_V3_V1_ADAPTER_REVISION}."),
        )

    v3_run_id = _string(v3.report.get("run_id"), context="v3 run_id")
    expected_artifact_path = f"runs/{v3_run_id}/train/adapter"
    expected_identity_fields = {
        "source",
        "revision",
        "kind",
        "tree_sha256",
        "artifact_path",
        "revision_is_immutable",
    }
    source = _string(v3_identity.get("source"), context="v3 adapter_source.source")
    portable_source = source.replace("\\", "/").rstrip("/")
    source_names_expected_artifact = portable_source == expected_artifact_path or (
        portable_source.endswith(f"/{expected_artifact_path}")
    )
    if (
        set(v3_identity) != expected_identity_fields
        or v3_identity.get("kind") != "local_directory"
        or v3_identity.get("revision") is not None
        or v3_identity.get("artifact_path") != expected_artifact_path
        or v3_identity.get("revision_is_immutable") is not True
        or not source_names_expected_artifact
    ):
        raise EvaluationError(
            "v3 adapter is not the canonical local adapter produced by its run",
            hint=f"Evaluate {expected_artifact_path}, not an external or substituted adapter.",
        )

    canonical_directory = v3.run_dir / "train" / "adapter"
    expected_directory = canonical_directory.resolve()
    bundled_directory = (v3.artifact_root / expected_artifact_path).resolve()
    if (
        bundled_directory != expected_directory
        or not canonical_directory.is_dir()
        or canonical_directory.is_symlink()
    ):
        raise EvaluationError(
            "v3 canonical local adapter directory is unavailable",
            hint=f"Restore the complete {expected_artifact_path} training artifact.",
        )
    observed_tree_sha256 = adapter_tree_sha256(expected_directory)
    if v3_identity.get("tree_sha256") != observed_tree_sha256:
        raise EvaluationError(
            "v3 adapter tree sha256 does not match the evaluated local adapter",
            hint="Regenerate evaluation from the unmodified v3 training output.",
        )

    training = _training_tco_input(v3)
    if (
        training.runtime_metadata is None
        or training.runtime_artifact is None
        or not training.metrics
        or training.metrics_artifact is None
        or training.manifest is None
        or training.manifest_artifact is None
        or training.local_adapter is None
    ):
        raise EvaluationError(
            "v3 adapter training evidence is unavailable or incomplete",
            hint=(
                "Keep runtime metadata, non-empty training metrics, the succeeded train "
                "manifest, and every local adapter file with the v3 run."
            ),
        )

    manifest = _mapping(training.manifest, context="v3 train manifest")
    if (
        manifest.get("stage") != "train"
        or manifest.get("status") != "succeeded"
        or manifest.get("run_id") != v3_run_id
        or manifest.get("config_sha256") != v3.report.get("config_sha256")
    ):
        raise EvaluationError(
            "v3 adapter is not bound to a succeeded matching train manifest",
            hint="Regenerate the v3 adapter and evaluation in one completed run.",
        )

    expected_adapter_outputs = {
        artifact.path: artifact.payload() for artifact in training.local_adapter.files
    }
    manifest_outputs = [
        _mapping(value, context="v3 train output")
        for value in _sequence(manifest.get("outputs"), context="v3 train outputs")
    ]
    adapter_output_list = [
        output for output in manifest_outputs if output.get("kind") == "adapter_weights"
    ]
    manifest_adapter_outputs = {
        _string(output.get("path"), context="v3 train adapter output path"): output
        for output in adapter_output_list
    }
    if len(adapter_output_list) != len(expected_adapter_outputs) or set(
        manifest_adapter_outputs
    ) != set(expected_adapter_outputs):
        raise EvaluationError(
            "v3 train manifest does not bind the evaluated adapter file set",
            hint="Restore the train manifest and all adapter files from the same run.",
        )
    for path, expected in expected_adapter_outputs.items():
        observed = manifest_adapter_outputs[path]
        for field, expected_value in expected.items():
            if observed.get(field) != expected_value:
                raise EvaluationError(
                    f"v3 train manifest adapter output {path} {field} does not match",
                    hint="Do not edit or replace adapter files after training.",
                )

    assert training.metrics_artifact is not None
    metrics_outputs = [
        output for output in manifest_outputs if output.get("kind") == "training_metrics"
    ]
    expected_metrics_output = training.metrics_artifact.payload()
    if len(metrics_outputs) != 1 or any(
        metrics_outputs[0].get(field) != expected_value
        for field, expected_value in expected_metrics_output.items()
    ):
        raise EvaluationError(
            "v3 train manifest does not bind the observed training metrics",
            hint="Restore training_metrics.jsonl and its matching train manifest.",
        )

    runtime = _mapping(training.runtime_metadata, context="v3 training runtime metadata")
    stages = _mapping(runtime.get("stages"), context="v3 training runtime stages")
    _mapping(stages.get("train"), context="v3 training runtime train stage")
    runtime_source = _mapping(runtime.get("source_code"), context="v3 training runtime source code")
    if (
        runtime.get("run_id") != v3_run_id
        or runtime.get("config_sha256") != v3.report.get("config_sha256")
        or runtime_source.get("working_tree_clean") is not True
        or runtime_source.get("git_commit") != manifest.get("git_commit")
        or not isinstance(runtime_source.get("git_commit"), str)
        or IMMUTABLE_HF_REVISION.fullmatch(runtime_source["git_commit"]) is None
    ):
        raise EvaluationError(
            "v3 training runtime does not match its immutable train manifest",
            hint="Launch and retain the full v3 run from one clean committed revision.",
        )

    root_manifest_path = v3.run_dir / "manifest.json"
    if not root_manifest_path.exists():
        raise EvaluationError(
            "v3 training evidence has no root run manifest",
            hint="Keep the succeeded root manifest with the v3 run.",
        )
    root_manifest = read_json_with_schema(
        root_manifest_path,
        expected_schema="sommelier.manifest.v1",
    )
    root_stages = _mapping(root_manifest.get("stages"), context="v3 root run stages")
    if (
        root_manifest.get("run_id") != v3_run_id
        or root_manifest.get("status") != "succeeded"
        or root_stages.get("train") != training.manifest_artifact.path
    ):
        raise EvaluationError(
            "v3 root run manifest does not bind succeeded training evidence",
            hint="Complete the v3 pipeline and retain its root and train manifests.",
        )


def _inference_tco_input(arm: _LoadedArm) -> InferenceArmInput:
    config, config_artifact = _resolved_tco_config(arm)
    if config is None or config_artifact is None:
        raise EvaluationError(
            f"{arm.name} TCO evidence has no resolved config",
            hint="Keep config.resolved.yaml with every completed experiment arm.",
        )
    expected_decoding = {
        "temperature": config.eval.temperature,
        "do_sample": config.eval.do_sample,
        "max_new_tokens": config.eval.max_new_tokens,
    }
    if arm.report.get("decoding") != expected_decoding:
        raise EvaluationError(f"{arm.name} evaluation decoding does not match its resolved config")
    if set(config.eval.slices) != {"en", "he"}:
        raise EvaluationError(
            f"{arm.name} resolved config does not declare the required en/he slices"
        )
    efficiency_value = arm.report.get("inference_efficiency")
    efficiency = (
        _mapping(efficiency_value, context=f"{arm.name} inference_efficiency")
        if efficiency_value is not None
        else None
    )
    telemetry: dict[str, Any] | None = None
    telemetry_artifact: ArtifactEvidence | None = None
    if efficiency is not None and efficiency.get("available") is True:
        reference = _mapping(
            efficiency.get("telemetry_artifact"),
            context=f"{arm.name} telemetry_artifact",
        )
        declared = _string(reference.get("path"), context=f"{arm.name} telemetry artifact path")
        telemetry_path = (arm.artifact_root / declared).resolve()
        expected_path = (arm.eval_dir / INFERENCE_TELEMETRY_FILENAME).resolve()
        if telemetry_path != expected_path:
            raise EvaluationError(
                f"{arm.name} telemetry artifact path does not match its eval directory",
                hint="Keep inference telemetry beside the generations it measured.",
            )
        telemetry = read_json_with_schema(
            telemetry_path,
            expected_schema=INFERENCE_TELEMETRY_SCHEMA,
        )
        telemetry_artifact = _artifact_evidence(
            arm,
            telemetry_path,
            kind="inference_telemetry",
            schema_version=INFERENCE_TELEMETRY_SCHEMA,
        )

    generation_artifacts = {
        language: _artifact_evidence(
            arm,
            arm.artifact_paths[f"generations.{language}"],
            kind="generations",
            schema_version=GENERATION_SCHEMA,
        )
        for language in ("en", "he")
    }
    per_language_successes = {
        language: compute_metrics(arm.scored[language])["full_call_exact_match"]["numerator"]
        for language in ("en", "he")
    }
    model_kind = _string(arm.report.get("model_kind"), context=f"{arm.name} model_kind")
    if model_kind not in {"base", "adapter"}:
        raise EvaluationError(f"{arm.name} has an unsupported model kind")
    eval_stage = evaluation_stage(cast("ModelKind", model_kind))
    eval_manifest_path = arm.run_dir / f"{eval_stage}_manifest.json"
    if not eval_manifest_path.exists():
        raise EvaluationError(
            f"{arm.name} TCO evidence has no {eval_stage} manifest",
            hint="Keep the succeeded model-specific eval manifest with every arm.",
        )
    run_manifest_path = arm.run_dir / "manifest.json"
    if not run_manifest_path.exists():
        raise EvaluationError(
            f"{arm.name} TCO evidence has no root run manifest",
            hint="Keep the succeeded root manifest with every experiment arm.",
        )
    runtime_path = arm.run_dir / RUNTIME_METADATA_FILENAME
    if not runtime_path.exists():
        raise EvaluationError(
            f"{arm.name} TCO evidence has no runtime metadata",
            hint="Keep runtime_metadata.json with every completed experiment arm.",
        )
    return InferenceArmInput(
        name=arm.name,
        run_id=_string(arm.report.get("run_id"), context=f"{arm.name} run_id"),
        model_kind=model_kind,
        config_sha256=_string(arm.report.get("config_sha256"), context=f"{arm.name} config_sha256"),
        efficiency=efficiency,
        telemetry=telemetry,
        telemetry_artifact=telemetry_artifact,
        evaluation_report_artifact=_artifact_evidence(
            arm,
            arm.report_path,
            kind="evaluation_report",
            schema_version=EVALUATION_REPORT_SCHEMA,
        ),
        generation_artifacts=generation_artifacts,
        actual_examples={language: len(arm.scored[language]) for language in ("en", "he")},
        exact_successes={
            **per_language_successes,
            "overall": sum(per_language_successes.values()),
        },
        decoding=_mapping(arm.report.get("decoding"), context=f"{arm.name} decoding"),
        evaluation_manifest=read_json_with_schema(
            eval_manifest_path,
            expected_schema="sommelier.manifest.v1",
        ),
        evaluation_manifest_artifact=_artifact_evidence(
            arm,
            eval_manifest_path,
            kind="manifest",
            schema_version="sommelier.manifest.v1",
        ),
        run_manifest=read_json_with_schema(
            run_manifest_path,
            expected_schema="sommelier.manifest.v1",
        ),
        run_manifest_artifact=_artifact_evidence(
            arm,
            run_manifest_path,
            kind="manifest",
            schema_version="sommelier.manifest.v1",
        ),
        resolved_config_artifact=config_artifact,
        configured_gpu_label=config.remote.gpu,
        runtime_metadata=_json_object(
            runtime_path,
            context=f"{arm.name} runtime metadata",
        ),
        runtime_artifact=_artifact_evidence(
            arm,
            runtime_path,
            kind="runtime_metadata",
            schema_version=RUNTIME_METADATA_SCHEMA,
        ),
    )


def _bind_v3_run_manifest(
    arm: _LoadedArm,
    *,
    config_artifact: ArtifactEvidence | None,
    tokenizer: TokenizerTaxInput | None,
    training: TrainingInput,
    evidence: dict[str, Any],
) -> None:
    manifest_path = arm.run_dir / "manifest.json"
    if not manifest_path.exists():
        raise EvaluationError(
            "v3 TCO artifacts exist without a root run manifest",
            hint="Keep the completed run manifest with its runtime evidence.",
        )
    manifest = read_json_with_schema(
        manifest_path,
        expected_schema="sommelier.manifest.v1",
    )
    if manifest.get("run_id") != arm.report.get("run_id"):
        raise EvaluationError("v3 root run manifest run_id does not match evaluation")
    if manifest.get("status") != "succeeded":
        raise EvaluationError(
            "v3 root run manifest is not succeeded",
            hint="TCO evidence requires a completed pipeline run.",
        )
    if config_artifact is not None:
        config_ref = _mapping(manifest.get("config"), context="v3 run config ref")
        for field in ("path", "sha256", "bytes", "kind", "schema_version"):
            if config_ref.get(field) != config_artifact.payload()[field]:
                raise EvaluationError(
                    f"v3 root run manifest config {field} does not match",
                    hint="Do not combine runtime evidence with a different config.",
                )
    stages = _mapping(manifest.get("stages"), context="v3 run stages")
    if tokenizer is not None and stages.get("tokenization") != (tokenizer.manifest_artifact.path):
        raise EvaluationError("v3 root run manifest does not bind tokenization evidence")
    if training.manifest_artifact is not None and stages.get("train") != (
        training.manifest_artifact.path
    ):
        raise EvaluationError("v3 root run manifest does not bind training evidence")
    evidence["sources"]["run_manifest"] = _artifact_evidence(
        arm,
        manifest_path,
        kind="manifest",
        schema_version="sommelier.manifest.v1",
    ).payload()


def _sovereign_tco_payload(
    base: _LoadedArm,
    v1: _LoadedArm,
    v3: _LoadedArm,
) -> dict[str, Any]:
    config, config_artifact = _resolved_tco_config(v3)
    runtime_path = v3.run_dir / RUNTIME_METADATA_FILENAME
    if not runtime_path.exists():
        raise EvaluationError(
            "v3 TCO evidence has no runtime metadata",
            hint="Keep runtime_metadata.json with the completed v3 run.",
        )
    runtime = _json_object(runtime_path, context="v3 runtime metadata")
    runtime_source = _mapping(runtime.get("source_code"), context="v3 runtime source code")
    source_code_revision = _string(
        runtime_source.get("git_commit"), context="v3 runtime git commit"
    )
    if (
        IMMUTABLE_HF_REVISION.fullmatch(source_code_revision) is None
        or runtime_source.get("working_tree_clean") is not True
    ):
        raise EvaluationError(
            "v3 runtime was not produced from a clean immutable source revision",
            hint="Launch the full pipeline from a clean committed worktree.",
        )
    model_identity = _mapping(v3.report.get("model_identity"), context="v3 model_identity")
    identity = TCOIdentity(
        run_id=_string(v3.report.get("run_id"), context="v3 run_id"),
        config_sha256=_string(v3.report.get("config_sha256"), context="v3 config_sha256"),
        tokenizer_id=_string(model_identity.get("tokenizer_id"), context="v3 tokenizer_id"),
        tokenizer_revision=_string(
            model_identity.get("tokenizer_revision"),
            context="v3 tokenizer_revision",
        ),
        test_split_sha256=_string(
            v3.report.get("test_split_sha256"), context="v3 test split sha256"
        ),
        train_languages=(tuple(config.train.languages) if config is not None else None),
        train_epochs=(config.train.epochs if config is not None else None),
        configured_gpu_label=(config.remote.gpu if config is not None else None),
        resolved_config_artifact=config_artifact,
        source_code_revision=source_code_revision,
    )
    tokenizer = _tokenizer_tco_input(v3)
    training = _training_tco_input(v3)
    evidence = build_sovereign_tco_evidence(
        identity,
        tokenizer_tax=tokenizer,
        training=training,
        inference_arms=[_inference_tco_input(arm) for arm in (base, v1, v3)],
        require_three_arm_matrix=True,
    )
    tokenization_evidence = _mapping(
        evidence.get("paired_tokenization"), context="paired tokenizer-tax evidence"
    )
    training_evidence = _mapping(evidence.get("qlora_training"), context="QLoRA training evidence")
    inference_evidence = _mapping(
        evidence.get("inference_efficiency"), context="inference TCO evidence"
    )
    inference_arms = _mapping(inference_evidence.get("arms"), context="inference TCO arms")
    if tokenization_evidence.get("available") is not True:
        raise EvaluationError("Hebrew v3 experiment is missing paired tokenizer-tax evidence")
    if (
        training_evidence.get("available") is not True
        or _mapping(
            training_evidence.get("train_stage_runtime"),
            context="QLoRA train-stage runtime",
        ).get("available")
        is not True
        or _mapping(
            training_evidence.get("adapter_storage"),
            context="QLoRA adapter storage",
        ).get("available")
        is not True
    ):
        raise EvaluationError(
            "Hebrew v3 experiment is missing measured QLoRA runtime or adapter evidence"
        )
    if (
        set(inference_arms) != {"base", "v1_en", "v3_en_he"}
        or any(
            _mapping(inference_arms[name], context=f"{name} inference evidence").get("available")
            is not True
            for name in inference_arms
        )
        or _mapping(
            inference_evidence.get("cross_arm_comparability"),
            context="inference cross-arm comparability",
        ).get("available")
        is not True
    ):
        raise EvaluationError(
            "Hebrew v3 experiment requires measured comparable inference telemetry "
            "for base, v1, and v3"
        )
    _bind_v3_run_manifest(
        v3,
        config_artifact=config_artifact,
        tokenizer=tokenizer,
        training=training,
        evidence=evidence,
    )
    return evidence


def _metric_deltas(
    reference: list[ScoredRecord], candidate: list[ScoredRecord]
) -> dict[str, float]:
    reference_metrics = compute_metrics(reference)
    candidate_metrics = compute_metrics(candidate)
    return {
        name: candidate_metrics[name]["value"] - reference_metrics[name]["value"]
        for name in METRIC_NAMES
    }


def _claim(
    *,
    passed: bool,
    estimate: float,
    lower: float,
    upper: float,
    criterion: str,
    statement: str,
    margin: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": passed,
        "metric": "full_call_exact_match",
        "estimate": estimate,
        "ci95": {"lower": lower, "upper": upper},
        "criterion": criterion,
    }
    if margin is not None:
        result["margin"] = margin
    if passed:
        result["statement"] = statement
    return result


def write_experiment_report(
    base_eval_dir: Path,
    v1_en_eval_dir: Path,
    v3_en_he_eval_dir: Path,
    out_dir: Path,
    *,
    english_non_inferiority_margin: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Writes the gated base/v1/v3 experiment report and returns its payload.

    The preregistered source, adapter, training, evaluation, and TCO identities
    are independently checked before uncertainty or claims are computed.
    """
    _validate_inputs(english_non_inferiority_margin, seed, resamples)
    finalizer_source = _finalizer_source_identity()
    base = _load_arm("base", base_eval_dir, expected_kind="base")
    v1 = _load_arm("v1_en", v1_en_eval_dir, expected_kind="adapter")
    v3 = _load_arm("v3_en_he", v3_en_he_eval_dir, expected_kind="adapter")
    arms = [base, v1, v3]
    shared = _shared_identity(arms)
    data_provenance = validate_hebrew_v3_data_provenance(
        run_dir=v3.run_dir,
        artifact_root=v3.artifact_root,
        run_id=_string(v3.report.get("run_id"), context="v3 run_id"),
        report_config_sha256=_string(v3.report.get("config_sha256"), context="v3 config_sha256"),
    )
    if data_provenance["contract"]["source_code_revision"] != finalizer_source["git_commit"]:
        raise EvaluationError(
            "experiment finalizer revision does not match the full-run source revision",
            hint="Checkout the exact source commit recorded by the v3 runtime metadata.",
        )
    _validate_preregistered_adapter_arms(v1, v3)
    tco_evidence = _sovereign_tco_payload(base, v1, v3)
    _mapping(tco_evidence.get("sources"), context="sovereign TCO sources")["data_provenance"] = (
        data_provenance["sources"]
    )

    comparison: dict[str, Any] = {}
    for offset, language in enumerate(("en", "he")):
        comparison[language] = {
            "deltas": _metric_deltas(v1.scored[language], v3.scored[language]),
            "ci95": paired_bootstrap_intervals(
                v1.scored[language],
                v3.scored[language],
                seed=seed + offset,
                resamples=resamples,
            ),
            "mcnemar": exact_mcnemar_full_call(
                v1.scored[language],
                v3.scored[language],
            ),
        }

    en_delta = comparison["en"]["deltas"]["full_call_exact_match"]
    he_delta = comparison["he"]["deltas"]["full_call_exact_match"]
    en_interval = comparison["en"]["ci95"]["intervals"]["full_call_exact_match"]
    he_interval = comparison["he"]["ci95"]["intervals"]["full_call_exact_match"]
    hebrew_passed = bool(he_interval["lower"] > 0.0)
    english_passed = bool(en_interval["lower"] >= -english_non_inferiority_margin)
    claims = {
        "hebrew_full_call_uplift": _claim(
            passed=hebrew_passed,
            estimate=he_delta,
            lower=he_interval["lower"],
            upper=he_interval["upper"],
            criterion="95% paired-bootstrap lower bound > 0",
            statement=(
                "The v3 en+he adapter improves Hebrew full-call exact match "
                "over the v1 English adapter on the gated cohort."
            ),
        ),
        "english_full_call_non_inferiority": _claim(
            passed=english_passed,
            estimate=en_delta,
            lower=en_interval["lower"],
            upper=en_interval["upper"],
            margin=english_non_inferiority_margin,
            criterion="95% paired-bootstrap lower bound >= -margin",
            statement=(
                "The v3 en+he adapter is non-inferior to the v1 English adapter "
                "on English full-call exact match at the declared margin."
            ),
        ),
    }
    approved_claims = [
        claim["statement"] for claim in claims.values() if claim["passed"] and "statement" in claim
    ]
    report: dict[str, Any] = {
        "schema_version": EXPERIMENT_REPORT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "preregistration": {
            "schema_version": HEBREW_V3_PREREGISTRATION_SCHEMA,
            "status": "committed_in_source_before_full_results",
            "english_non_inferiority_margin": (HEBREW_V3_ENGLISH_NON_INFERIORITY_MARGIN),
            "bootstrap": {
                "seed": HEBREW_V3_BOOTSTRAP_SEED,
                "resamples": HEBREW_V3_BOOTSTRAP_RESAMPLES,
                "confidence_level": 0.95,
                "method": "sommelier.paired_bootstrap.v1",
            },
            "primary_claim_rules": {
                "hebrew_full_call_uplift": "95% lower bound > 0",
                "english_full_call_non_inferiority": ("95% lower bound >= -0.01"),
            },
            "finalizer_source_code": finalizer_source,
        },
        "shared_evaluation_identity": shared,
        "data_provenance": data_provenance,
        "bootstrap": {
            "seed": seed,
            "resamples": resamples,
            "confidence_level": 0.95,
        },
        "arms": {
            "base": _arm_payload(base),
            "v1_en": _arm_payload(v1),
            "v3_en_he": _arm_payload(v3),
        },
        "comparisons": {"v3_vs_v1": comparison},
        "sovereign_tco_evidence": tco_evidence,
        "claims": claims,
        "all_claims_passed": hebrew_passed and english_passed,
        "approved_claims": approved_claims,
    }

    output_path = out_dir.resolve() / EXPERIMENT_REPORT_FILENAME

    def writer(temp_path: Path) -> None:
        temp_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    write_artifact_atomic(
        output_path,
        writer,
        kind="experiment_report",
        schema_version=EXPERIMENT_REPORT_SCHEMA,
    )
    return report
