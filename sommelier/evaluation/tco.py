"""Bounded sovereign-agentic TCO evidence derived from existing artifacts.

This module is deliberately pure: callers load bytes and construct immutable
artifact evidence, while :func:`build_sovereign_tco_evidence` validates the
join and returns JSON-compatible measurements. It never prices GPU time or
extrapolates full-fine-tuning savings without matching evidence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

from sommelier.analysis.tokenization import (
    TOKENIZER_TAX_RECORD_SCHEMA,
    TOKENIZER_TAX_REPORT_SCHEMA,
)
from sommelier.errors import EvaluationError
from sommelier.evaluation.generate import (
    GENERATION_TIMING_AGGREGATION,
    GENERATION_TIMING_SCOPE,
    INFERENCE_TELEMETRY_SCHEMA,
    SEQUENTIAL_RUN_BOUNDARY,
    gpu_count_from_label,
    inference_timed_call_contract,
    inference_warmup_contract,
)
from sommelier.runtime_metadata import RUNTIME_METADATA_SCHEMA
from sommelier.training.metrics import TRAINING_METRIC_SCHEMA

SOVEREIGN_TCO_EVIDENCE_SCHEMA: Final = "sommelier.sovereign_tco_evidence.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_ADAPTER_WEIGHT_FILE = re.compile(r"^adapter_model(?:-[0-9]+-of-[0-9]+)?\.(?:safetensors|bin)$")
_SPLITS: Final = ("train", "validation", "test")
_TOKEN_RATIO_FIELDS: Final = ("query_tokens", "prompt_tokens", "full_tokens")
_REQUIRED_INFERENCE_PACKAGES: Final = (
    "python",
    "torch",
    "transformers",
    "tokenizers",
    "accelerate",
    "peft",
    "datasets",
    "huggingface_hub",
)


@dataclass(frozen=True)
class ArtifactEvidence:
    """Observed identity and size of one local evidence artifact."""

    path: str
    kind: str
    schema_version: str
    sha256: str
    bytes: int

    def payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class TCOIdentity:
    """Expected v3 identity, independently established by the experiment."""

    run_id: str
    config_sha256: str
    tokenizer_id: str
    tokenizer_revision: str
    test_split_sha256: str
    train_languages: tuple[str, ...] | None
    train_epochs: int | None
    configured_gpu_label: str | None
    resolved_config_artifact: ArtifactEvidence | None = None
    source_code_revision: str | None = None


@dataclass(frozen=True)
class TokenizerTaxInput:
    report: Mapping[str, Any]
    records: Sequence[Mapping[str, Any]]
    manifest: Mapping[str, Any]
    report_artifact: ArtifactEvidence
    records_artifact: ArtifactEvidence
    manifest_artifact: ArtifactEvidence
    formatted_inputs: Mapping[str, ArtifactEvidence]


@dataclass(frozen=True)
class LocalAdapterInput:
    tree_sha256: str
    files: tuple[ArtifactEvidence, ...]


@dataclass(frozen=True)
class TrainingInput:
    runtime_metadata: Mapping[str, Any] | None
    runtime_artifact: ArtifactEvidence | None
    metrics: Sequence[Mapping[str, Any]] | None
    metrics_artifact: ArtifactEvidence | None
    manifest: Mapping[str, Any] | None
    manifest_artifact: ArtifactEvidence | None
    adapter_identity: Mapping[str, Any] | None
    local_adapter: LocalAdapterInput | None


@dataclass(frozen=True)
class InferenceArmInput:
    name: str
    run_id: str
    model_kind: str
    config_sha256: str
    efficiency: Mapping[str, Any] | None
    telemetry: Mapping[str, Any] | None
    telemetry_artifact: ArtifactEvidence | None
    evaluation_report_artifact: ArtifactEvidence
    generation_artifacts: Mapping[str, ArtifactEvidence]
    actual_examples: Mapping[str, int]
    exact_successes: Mapping[str, int]
    decoding: Mapping[str, Any]
    evaluation_manifest: Mapping[str, Any]
    evaluation_manifest_artifact: ArtifactEvidence
    run_manifest: Mapping[str, Any]
    run_manifest_artifact: ArtifactEvidence
    resolved_config_artifact: ArtifactEvidence
    configured_gpu_label: str
    runtime_metadata: Mapping[str, Any]
    runtime_artifact: ArtifactEvidence


def _error(message: str) -> EvaluationError:
    return EvaluationError(
        message,
        hint="Do not combine or edit evidence from different runs; regenerate the artifacts.",
    )


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error(f"{context} must be a JSON object")
    return cast("dict[str, Any]", value)


def _sequence(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(f"{context} must be a JSON array")
    return value


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(f"{context} must be a non-empty string")
    return value


def _integer(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: object, *, context: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(f"{context} must be a finite number >= {minimum}")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise _error(f"{context} must be a finite number >= {minimum}")
    return number


def _validate_artifact(artifact: ArtifactEvidence, *, context: str) -> None:
    if not artifact.path or artifact.path.startswith("/") or ".." in artifact.path.split("/"):
        raise _error(f"{context} has an unsafe artifact path")
    if not artifact.kind:
        raise _error(f"{context} is missing its artifact kind")
    if not _SHA256.fullmatch(artifact.sha256):
        raise _error(f"{context} has an invalid sha256")
    if isinstance(artifact.bytes, bool) or artifact.bytes < 0:
        raise _error(f"{context} has an invalid byte count")


def _run_path(run_id: str, suffix: str) -> str:
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise _error("TCO run_id is not safe for run-relative evidence paths")
    return f"runs/{run_id}/{suffix}"


def _assert_artifact_identity(
    artifact: ArtifactEvidence,
    *,
    expected_path: str,
    expected_kind: str,
    expected_schema: str,
    context: str,
) -> None:
    _validate_artifact(artifact, context=context)
    if artifact.path != expected_path:
        raise _error(f"{context} is not at its expected run-relative path")
    if artifact.kind != expected_kind:
        raise _error(f"{context} has the wrong artifact kind")
    if artifact.schema_version != expected_schema:
        raise _error(f"{context} has the wrong artifact schema")


def _assert_ref(
    value: object,
    artifact: ArtifactEvidence,
    *,
    context: str,
    require_kind: bool = True,
) -> None:
    _validate_artifact(artifact, context=context)
    reference = _mapping(value, context=context)
    expected: dict[str, object] = {
        "path": artifact.path,
        "sha256": artifact.sha256,
        "bytes": artifact.bytes,
    }
    if require_kind:
        expected.update(
            {
                "kind": artifact.kind,
                "schema_version": artifact.schema_version,
            }
        )
    for field, expected_value in expected.items():
        if reference.get(field) != expected_value:
            raise _error(f"{context} {field} does not match the observed artifact")


def _validate_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    stage: str,
    identity: TCOIdentity,
) -> dict[str, Any]:
    payload = dict(manifest)
    if payload.get("schema_version") != "sommelier.manifest.v1":
        raise _error(f"{stage} manifest has the wrong schema")
    if payload.get("stage") != stage or payload.get("status") != "succeeded":
        raise _error(f"{stage} manifest is not a succeeded {stage} stage")
    if payload.get("run_id") != identity.run_id:
        raise _error(f"{stage} manifest run_id does not match v3")
    if payload.get("config_sha256") != identity.config_sha256:
        raise _error(f"{stage} manifest config_sha256 does not match v3")
    git_commit = payload.get("git_commit")
    if identity.source_code_revision is not None:
        if not _GIT_COMMIT.fullmatch(identity.source_code_revision):
            raise _error("v3 source-code identity is not an immutable revision")
        if git_commit != identity.source_code_revision:
            raise _error(f"{stage} manifest git_commit does not match v3 runtime")
    elif git_commit != "unknown" and (
        not isinstance(git_commit, str) or not _GIT_COMMIT.fullmatch(git_commit)
    ):
        raise _error(f"{stage} manifest has no immutable source-code identity")
    return payload


def _ratio(
    numerator: int,
    denominator: int,
    *,
    context: str,
) -> float | None:
    if denominator == 0:
        if numerator != 0:
            raise _error(f"{context} has a zero denominator with a non-zero numerator")
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        raise _error(f"{context} is not finite")
    return value


def _same_number(actual: object, expected: float | None, *, context: str) -> None:
    if expected is None:
        if actual is not None:
            raise _error(f"{context} must be explicitly unavailable")
        return
    number = _number(actual, context=context)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
        raise _error(f"{context} does not match recomputed evidence")


def _tokenizer_evidence(
    identity: TCOIdentity,
    source: TokenizerTaxInput | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if source is None:
        return (
            {
                "available": False,
                "reason": "tokenizer_tax_evidence_missing",
            },
            {},
        )
    if identity.train_languages is None or identity.train_epochs is None:
        raise _error("tokenizer-tax evidence exists but resolved training config is missing")

    report = dict(source.report)
    if report.get("schema_version") != TOKENIZER_TAX_REPORT_SCHEMA:
        raise _error("tokenizer-tax report has the wrong schema")
    if report.get("run_id") != identity.run_id:
        raise _error("tokenizer-tax report run_id does not match v3")
    if report.get("config_sha256") != identity.config_sha256:
        raise _error("tokenizer-tax report config_sha256 does not match v3")
    tokenizer = _mapping(report.get("tokenizer"), context="tokenizer-tax tokenizer")
    if tokenizer != {"id": identity.tokenizer_id, "revision": identity.tokenizer_revision}:
        raise _error("tokenizer-tax identity does not match the evaluated tokenizer")
    if report.get("root_language") != "en":
        raise _error("sovereign TCO evidence requires English as the paired root language")

    tokenization_prefix = "analysis/tokenization"
    _assert_artifact_identity(
        source.report_artifact,
        expected_path=_run_path(
            identity.run_id, f"{tokenization_prefix}/tokenizer_tax_report.json"
        ),
        expected_kind="tokenizer_tax_report",
        expected_schema=TOKENIZER_TAX_REPORT_SCHEMA,
        context="tokenizer-tax report artifact",
    )
    _assert_artifact_identity(
        source.records_artifact,
        expected_path=_run_path(
            identity.run_id, f"{tokenization_prefix}/tokenizer_tax_records.jsonl"
        ),
        expected_kind="tokenizer_tax_records",
        expected_schema=TOKENIZER_TAX_RECORD_SCHEMA,
        context="tokenizer-tax records artifact",
    )
    _assert_artifact_identity(
        source.manifest_artifact,
        expected_path=_run_path(identity.run_id, "tokenization_manifest.json"),
        expected_kind="manifest",
        expected_schema="sommelier.manifest.v1",
        context="tokenization manifest artifact",
    )
    if set(source.formatted_inputs) != set(_SPLITS):
        raise _error("tokenizer-tax formatted input set is incomplete")
    report_inputs = _mapping(report.get("inputs"), context="tokenizer-tax inputs")
    for split in _SPLITS:
        artifact = source.formatted_inputs[split]
        _assert_artifact_identity(
            artifact,
            expected_path=_run_path(identity.run_id, f"formatted/{split}.jsonl"),
            expected_kind="formatted_split",
            expected_schema="sommelier.formatted_example.v2",
            context=f"tokenizer-tax {split} formatted artifact",
        )
        _assert_ref(
            report_inputs.get(split),
            artifact,
            context=f"tokenizer-tax {split} input",
            require_kind=False,
        )
    if source.formatted_inputs["test"].sha256 != identity.test_split_sha256:
        raise _error("tokenizer-tax test split does not match evaluation")

    records_ref = _mapping(report.get("records"), context="tokenizer-tax records")
    if records_ref.get("path") != source.records_artifact.path:
        raise _error("tokenizer-tax records path does not match the observed artifact")
    if records_ref.get("sha256") != source.records_artifact.sha256:
        raise _error("tokenizer-tax records sha256 does not match the observed artifact")
    if records_ref.get("count") != len(source.records):
        raise _error("tokenizer-tax records count does not match the observed artifact")

    manifest = _validate_manifest_identity(
        source.manifest,
        stage="tokenization",
        identity=identity,
    )
    manifest_inputs = _sequence(manifest.get("inputs"), context="tokenization inputs")
    if len(manifest_inputs) != len(_SPLITS):
        raise _error("tokenization manifest input count is inconsistent")
    for split, reference in zip(_SPLITS, manifest_inputs, strict=True):
        _assert_ref(
            reference,
            source.formatted_inputs[split],
            context=f"tokenization manifest {split} input",
        )
    manifest_outputs = _sequence(manifest.get("outputs"), context="tokenization outputs")
    if len(manifest_outputs) != 2:
        raise _error("tokenization manifest output count is inconsistent")
    expected_outputs = {
        source.records_artifact.path: source.records_artifact,
        source.report_artifact.path: source.report_artifact,
    }
    manifest_output_paths = {
        _string(
            _mapping(reference, context="tokenization output").get("path"),
            context="tokenization output path",
        )
        for reference in manifest_outputs
    }
    if manifest_output_paths != set(expected_outputs):
        raise _error("tokenization manifest output set is inconsistent")
    for reference in manifest_outputs:
        payload = _mapping(reference, context="tokenization output")
        path = _string(payload.get("path"), context="tokenization output path")
        output_artifact = expected_outputs.get(path)
        if output_artifact is None:
            raise _error("tokenization manifest references an unexpected output")
        _assert_ref(payload, output_artifact, context=f"tokenization output {path}")

    root_records: dict[str, dict[str, Any]] = {}
    target_records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(source.records):
        record = dict(raw_record)
        if record.get("schema_version") != TOKENIZER_TAX_RECORD_SCHEMA:
            raise _error(f"tokenizer-tax record {index} has the wrong schema")
        example_id = _string(
            record.get("example_id"), context=f"tokenizer-tax record {index} example_id"
        )
        language = _string(record.get("language"), context=f"tokenizer-tax record {index} language")
        if language == "en":
            if example_id in root_records:
                raise _error(f"tokenizer-tax records repeat English root {example_id}")
            root_records[example_id] = record
        elif language == "he":
            target_records.append(record)
    if not root_records or not target_records:
        raise _error("tokenizer-tax records must contain English roots and Hebrew pairs")

    pairing = _mapping(report.get("pairing"), context="tokenizer-tax pairing")
    hebrew_pairing = _mapping(pairing.get("he"), context="tokenizer-tax pairing.he")
    scopes: dict[str, Any] = {}
    for scope in ("all", *_SPLITS):
        roots = [
            record
            for record in root_records.values()
            if scope == "all" or record.get("split") == scope
        ]
        targets = [
            record for record in target_records if scope == "all" or record.get("split") == scope
        ]
        report_scope = (
            _mapping(hebrew_pairing.get("all"), context="tokenizer pairing all")
            if scope == "all"
            else _mapping(
                _mapping(hebrew_pairing.get("splits"), context="tokenizer pairing splits").get(
                    scope
                ),
                context=f"tokenizer pairing {scope}",
            )
        )
        coverage = _mapping(report_scope.get("coverage"), context=f"{scope} coverage")
        expected_coverage = {
            "paired": len(targets),
            "roots": len(roots),
            "ratio": len(targets) / len(roots),
        }
        if coverage != expected_coverage:
            raise _error(f"tokenizer-tax {scope} coverage does not match records")

        token_ratios: dict[str, Any] = {}
        report_metrics = _mapping(report_scope.get("metrics"), context=f"{scope} metrics")
        for field in _TOKEN_RATIO_FIELDS:
            paired_total = 0
            matched_root_total = 0
            seen_roots: set[str] = set()
            for target in targets:
                root_id = _string(target.get("root_example_id"), context=f"{scope} Hebrew root id")
                if root_id in seen_roots:
                    raise _error(f"tokenizer-tax {scope} repeats Hebrew pair {root_id}")
                seen_roots.add(root_id)
                root = root_records.get(root_id)
                if root is None:
                    raise _error(f"tokenizer-tax Hebrew pair references missing root {root_id}")
                if root.get("split") != target.get("split"):
                    raise _error(f"tokenizer-tax pair {root_id} crosses split boundaries")
                target_counts = _mapping(target.get("counts"), context=f"{root_id} target counts")
                root_counts = _mapping(root.get("counts"), context=f"{root_id} root counts")
                paired_total += _integer(
                    target_counts.get(field), context=f"{root_id} target {field}"
                )
                matched_root_total += _integer(
                    root_counts.get(field), context=f"{root_id} root {field}"
                )
            expected_ratio = _ratio(
                paired_total,
                matched_root_total,
                context=f"{scope} {field} ratio",
            )
            reported = _mapping(report_metrics.get(field), context=f"{scope} {field}")
            if reported.get("paired_total") != paired_total:
                raise _error(f"tokenizer-tax {scope} {field} paired total is inconsistent")
            if reported.get("matched_root_total") != matched_root_total:
                raise _error(f"tokenizer-tax {scope} {field} root total is inconsistent")
            _same_number(
                reported.get("ratio"),
                expected_ratio,
                context=f"tokenizer-tax {scope} {field} ratio",
            )
            token_ratios[field] = {
                "paired_hebrew_tokens": paired_total,
                "matched_english_tokens": matched_root_total,
                "hebrew_to_english_ratio": expected_ratio,
            }
        scopes[scope] = {
            "coverage": expected_coverage,
            "token_ratios": token_ratios,
        }

    workload = _mapping(report.get("training_workload"), context="training workload")
    if tuple(workload.get("languages", ())) != identity.train_languages:
        raise _error("tokenizer-tax training languages do not match resolved config")
    epochs = _integer(workload.get("epochs"), context="training workload epochs", minimum=1)
    if epochs != identity.train_epochs:
        raise _error("tokenizer-tax epochs do not match resolved config")
    selected = [
        record
        for record in (*root_records.values(), *target_records)
        if record.get("split") == "train" and record.get("language") in identity.train_languages
    ]
    tokens_per_epoch = sum(
        _integer(
            _mapping(record.get("counts"), context="training record counts").get("full_tokens"),
            context="training record full_tokens",
        )
        for record in selected
    )
    projected_tokens = tokens_per_epoch * epochs
    expected_workload = {
        "languages": list(identity.train_languages),
        "examples_per_epoch": len(selected),
        "non_padding_full_tokens_per_epoch": tokens_per_epoch,
        "epochs": epochs,
        "projected_non_padding_full_tokens": projected_tokens,
    }
    for field, expected in expected_workload.items():
        if workload.get(field) != expected:
            raise _error(f"training workload {field} does not match tokenizer records")
    boundary = _string(workload.get("boundary"), context="training workload boundary")

    sources = {
        "tokenizer_tax_report": source.report_artifact.payload(),
        "tokenizer_tax_records": source.records_artifact.payload(),
        "tokenization_manifest": source.manifest_artifact.payload(),
        "formatted_inputs": {split: source.formatted_inputs[split].payload() for split in _SPLITS},
    }
    return (
        {
            "available": True,
            "evidence_kind": "deterministic_measurement_from_pinned_tokenizer",
            "reference_language": "en",
            "target_language": "he",
            "paired_scopes": scopes,
            "projected_training_workload": {
                **expected_workload,
                "evidence_kind": "deterministic_projection",
                "boundary": boundary,
            },
        },
        sources,
    )


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "value": None, "reason": reason}


def _training_evidence(
    identity: TCOIdentity,
    source: TrainingInput | None,
    tokenizer: TokenizerTaxInput | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if source is None:
        return (
            {
                "available": False,
                "reason": "training_runtime_and_adapter_evidence_missing",
                "source_code": _unavailable("source_code_provenance_not_recorded"),
                "remote_execution": _unavailable("remote_execution_boundary_not_recorded"),
                "currency_cost": _unavailable("provider_billing_evidence_not_supplied"),
                "full_finetune_savings": _unavailable(
                    "matched_full_finetune_evidence_not_supplied"
                ),
            },
            {},
        )

    sources: dict[str, Any] = {}
    runtime = dict(source.runtime_metadata) if source.runtime_metadata is not None else None
    metrics = [dict(metric) for metric in source.metrics] if source.metrics is not None else None
    if (runtime is None) != (source.runtime_artifact is None):
        raise _error("runtime metadata payload and artifact identity are incomplete")
    if (metrics is None) != (source.metrics_artifact is None):
        raise _error("training metrics payload and artifact identity are incomplete")
    if (source.manifest is None) != (source.manifest_artifact is None):
        raise _error("train manifest payload and artifact identity are incomplete")
    if source.local_adapter is not None and source.adapter_identity is None:
        raise _error("local adapter files have no evaluated adapter identity")

    has_training_claim_evidence = bool(
        runtime is not None or metrics is not None or source.local_adapter is not None
    )
    if (
        identity.source_code_revision is not None
        and has_training_claim_evidence
        and runtime is None
    ):
        raise _error("training evidence has no runtime source-code provenance")
    if has_training_claim_evidence and source.manifest is None:
        raise _error("training evidence is not bound by a succeeded train manifest")

    manifest_payload: dict[str, Any] | None = None
    if source.manifest is not None:
        assert source.manifest_artifact is not None
        _assert_artifact_identity(
            source.manifest_artifact,
            expected_path=_run_path(identity.run_id, "train_manifest.json"),
            expected_kind="manifest",
            expected_schema="sommelier.manifest.v1",
            context="train manifest artifact",
        )
        manifest_payload = _validate_manifest_identity(
            source.manifest,
            stage="train",
            identity=identity,
        )
        sources["train_manifest"] = source.manifest_artifact.payload()
        details = _mapping(manifest_payload.get("details"), context="train manifest details")
        if (
            identity.train_languages is not None
            and tuple(details.get("train_languages", ())) != identity.train_languages
        ):
            raise _error("train manifest languages do not match resolved config")

    elapsed: float | None = None
    gpu_label: str | None = None
    gpu_count: int | None = None
    peak_runtime: int | None = None
    currency_cost: dict[str, Any]
    if runtime is None:
        runtime_section = _unavailable("runtime_metadata_missing")
        currency_cost = _unavailable("provider_billing_evidence_not_supplied")
        source_code_section = _unavailable("source_code_provenance_not_recorded")
        remote_execution_section = _unavailable("remote_execution_boundary_not_recorded")
    else:
        if source.runtime_artifact is None:
            raise _error("runtime metadata has no observed artifact identity")
        _assert_artifact_identity(
            source.runtime_artifact,
            expected_path=_run_path(identity.run_id, "runtime_metadata.json"),
            expected_kind="runtime_metadata",
            expected_schema=RUNTIME_METADATA_SCHEMA,
            context="runtime metadata artifact",
        )
        sources["runtime_metadata"] = source.runtime_artifact.payload()
        if runtime.get("schema_version") != RUNTIME_METADATA_SCHEMA:
            raise _error("runtime metadata has the wrong schema")
        if runtime.get("run_id") != identity.run_id:
            raise _error("runtime metadata run_id does not match v3")
        if runtime.get("config_sha256") != identity.config_sha256:
            raise _error("runtime metadata config_sha256 does not match v3")
        hardware = _mapping(runtime.get("hardware"), context="runtime hardware")
        gpu_label = _string(hardware.get("gpu"), context="runtime GPU label")
        if hardware.get("source") != "config":
            raise _error("runtime GPU label is not bound to config")
        if identity.configured_gpu_label is None:
            raise _error("runtime metadata exists but resolved GPU config is missing")
        if gpu_label != identity.configured_gpu_label:
            raise _error("runtime GPU label does not match resolved config")
        gpu_count = gpu_count_from_label(gpu_label)
        stages = _mapping(runtime.get("stages"), context="runtime stages")
        train_stage = _mapping(stages.get("train"), context="runtime train stage")
        elapsed = _number(train_stage.get("elapsed_seconds"), context="train elapsed seconds")
        if elapsed <= 0.0:
            raise _error("train elapsed seconds must be positive for TCO evidence")
        peak_value = runtime.get("peak_gpu_memory_mb")
        if peak_value is not None:
            peak_runtime = _integer(peak_value, context="runtime peak GPU memory", minimum=1)
        runtime_section = {
            "available": True,
            "elapsed_seconds": elapsed,
            "boundary": (
                "Pipeline-observed train-stage wall clock from stage dispatch through "
                "stage-wrapper completion; includes model/tokenizer load, training, "
                "adapter save, metric serialization, and any wrapper-owned post-stage "
                "cleanup or artifact-volume commit."
            ),
            "configured_gpu": {
                "label": gpu_label,
                "count": gpu_count,
                "source": "resolved config via runtime metadata",
            },
            "configured_gpu_hours": elapsed * gpu_count / 3600.0,
            "configured_gpu_hours_kind": "observed_wall_time_x_configured_gpu_count",
        }
        source_code = runtime.get("source_code")
        if source_code is None:
            if identity.source_code_revision is not None:
                raise _error("training runtime has no source-code provenance")
            source_code_section = _unavailable("source_code_provenance_not_recorded")
        else:
            source_payload = _mapping(source_code, context="source-code provenance")
            git_commit = _string(source_payload.get("git_commit"), context="source-code git commit")
            if git_commit != "unknown" and not _GIT_COMMIT.fullmatch(git_commit):
                raise _error("source-code git commit is not an immutable hex revision")
            clean = source_payload.get("working_tree_clean")
            if clean is not None and not isinstance(clean, bool):
                raise _error("source-code working_tree_clean must be boolean or null")
            source_snapshot_available = git_commit != "unknown" and clean is True
            if identity.source_code_revision is not None:
                if git_commit != identity.source_code_revision:
                    raise _error("training runtime git_commit does not match v3 runtime")
                if clean is not True:
                    raise _error("training runtime source tree was not recorded clean")
            source_code_section = {
                "available": source_snapshot_available,
                "git_commit": git_commit,
                "working_tree_clean": clean,
                "boundary": _string(source_payload.get("boundary"), context="source-code boundary"),
                "reason": (
                    None
                    if source_snapshot_available
                    else (
                        "git_commit_not_recorded"
                        if git_commit == "unknown"
                        else "working_tree_not_recorded_clean"
                    )
                ),
            }
            if manifest_payload is not None and git_commit != "unknown":
                if manifest_payload.get("git_commit") != git_commit:
                    raise _error("train manifest git_commit does not match runtime source code")
        remote_execution = runtime.get("remote_execution")
        if remote_execution is None:
            remote_execution_section = _unavailable("remote_execution_boundary_not_recorded")
        else:
            remote_payload = _mapping(remote_execution, context="remote execution boundary")
            allocation_label = _string(
                remote_payload.get("gpu_allocation_label"),
                context="remote GPU allocation label",
            )
            if allocation_label != gpu_label:
                raise _error("remote GPU allocation label does not match runtime config")
            remote_execution_section = {
                "available": True,
                "provider": _string(remote_payload.get("provider"), context="remote provider"),
                "function_timeout_seconds": _integer(
                    remote_payload.get("function_timeout_seconds"),
                    context="remote function timeout",
                    minimum=1,
                ),
                "gpu_allocation_label": allocation_label,
                "boundary": _string(
                    remote_payload.get("boundary"),
                    context="remote execution boundary text",
                ),
            }
        observed_cost = runtime.get("observed_cost_usd")
        cost_source = runtime.get("cost_source")
        if observed_cost is not None or cost_source != "unavailable":
            raise _error(
                "runtime currency cost cannot be accepted without a joined "
                "schema-versioned provider billing artifact"
            )
        # Runtime metadata deliberately cannot promote a hand-entered number
        # into observed billing evidence. A future available branch must join
        # a provider artifact, account/resource interval, and manifest ref.
        currency_cost = _unavailable("provider_billing_evidence_not_supplied")

    tokens_seen: int | None = None
    peak_metrics: int | None = None
    if metrics is None:
        tokens_section = _unavailable("training_metrics_missing")
    else:
        assert source.metrics_artifact is not None
        _assert_artifact_identity(
            source.metrics_artifact,
            expected_path=_run_path(identity.run_id, "train/training_metrics.jsonl"),
            expected_kind="training_metrics",
            expected_schema=TRAINING_METRIC_SCHEMA,
            context="training metrics artifact",
        )
        sources["training_metrics"] = source.metrics_artifact.payload()
        previous_positive_tokens = 0
        seen_peaks: list[int] = []
        for index, metric in enumerate(metrics):
            if metric.get("schema_version") != TRAINING_METRIC_SCHEMA:
                raise _error(f"training metric {index} has the wrong schema")
            current = _integer(
                metric.get("tokens_seen"), context=f"training metric {index} tokens_seen"
            )
            if current > 0:
                if current < previous_positive_tokens:
                    raise _error("positive training metrics tokens_seen is not monotonic")
                previous_positive_tokens = current
            peak = metric.get("peak_gpu_memory_mb")
            if peak is not None:
                seen_peaks.append(
                    _integer(peak, context="training metric peak GPU memory", minimum=1)
                )
        if len(set(seen_peaks)) > 1:
            raise _error("training metrics disagree on peak GPU memory")
        peak_metrics = seen_peaks[-1] if seen_peaks else None
        tokens_seen = previous_positive_tokens if previous_positive_tokens > 0 else None
        tokens_section = (
            {
                "available": True,
                "value": tokens_seen,
                "unit": "trainer_reported_input_tokens",
                "source_label": "maximum_positive_transformers.num_input_tokens_seen",
                "boundary": (
                    "Backend-reported input tokens; padding semantics are backend-defined "
                    "and this value is not substituted for projected non-padding tokens."
                ),
            }
            if tokens_seen is not None
            else _unavailable("trainer_did_not_report_nonzero_tokens_seen")
        )

    if peak_runtime is not None and peak_metrics is not None and peak_runtime != peak_metrics:
        raise _error("runtime and training metrics disagree on peak GPU memory")
    peak_memory = peak_runtime if peak_runtime is not None else peak_metrics
    peak_section = (
        {
            "available": True,
            "value": peak_memory,
            "unit": "MiB",
            "evidence_kind": "observed_peak_allocated_gpu_memory",
        }
        if peak_memory is not None
        else _unavailable("peak_gpu_memory_not_recorded")
    )
    throughput = (
        {
            "available": True,
            "value": tokens_seen / elapsed,
            "unit": "trainer_reported_input_tokens_per_train_stage_second",
            "boundary": "Uses the end-to-end train-stage wall clock boundary.",
        }
        if tokens_seen is not None and elapsed is not None and elapsed > 0
        else _unavailable("tokens_seen_or_positive_train_elapsed_unavailable")
    )

    if manifest_payload is not None:
        if tokenizer is not None:
            manifest_inputs = _sequence(
                manifest_payload.get("inputs"), context="train manifest inputs"
            )
            expected_inputs = [
                tokenizer.formatted_inputs["train"],
                tokenizer.formatted_inputs["validation"],
            ]
            if len(manifest_inputs) != len(expected_inputs):
                raise _error("train manifest input count is inconsistent")
            for reference, artifact in zip(manifest_inputs, expected_inputs, strict=True):
                _assert_ref(reference, artifact, context="train manifest formatted input")

    if source.metrics_artifact is not None:
        if manifest_payload is None:
            raise _error("training metrics artifact exists without a train manifest")
        matching = [
            output
            for output in _sequence(manifest_payload.get("outputs"), context="train outputs")
            if isinstance(output, dict) and output.get("kind") == "training_metrics"
        ]
        if len(matching) != 1:
            raise _error("train manifest must bind exactly one training metrics artifact")
        _assert_ref(matching[0], source.metrics_artifact, context="train metrics manifest output")

    adapter_section: dict[str, Any]
    adapter_identity = (
        dict(source.adapter_identity) if source.adapter_identity is not None else None
    )
    if adapter_identity is None:
        adapter_section = _unavailable("adapter_identity_missing")
    elif adapter_identity.get("kind") != "local_directory":
        if source.local_adapter is not None:
            raise _error("local adapter files disagree with the evaluated adapter kind")
        adapter_section = _unavailable("evaluated_adapter_is_not_a_local_run_artifact")
    elif source.local_adapter is None:
        adapter_section = _unavailable("local_adapter_files_missing")
    else:
        if adapter_identity.get("revision_is_immutable") is not True:
            raise _error("evaluated local adapter identity is not marked immutable")
        reported_tree = _string(adapter_identity.get("tree_sha256"), context="adapter tree_sha256")
        reported_path = _string(
            adapter_identity.get("artifact_path"), context="adapter artifact_path"
        )
        expected_adapter_path = _run_path(identity.run_id, "train/adapter")
        if reported_path != expected_adapter_path:
            raise _error("evaluated local adapter does not name the canonical v3 run-relative path")
        if reported_tree != source.local_adapter.tree_sha256:
            raise _error("adapter tree sha256 does not match evaluation identity")
        if not _SHA256.fullmatch(source.local_adapter.tree_sha256):
            raise _error("adapter tree sha256 is invalid")
        if not source.local_adapter.files:
            raise _error("local adapter contains no regular files")
        adapter_prefix = _run_path(identity.run_id, "train/adapter/")
        for artifact in source.local_adapter.files:
            _validate_artifact(artifact, context="adapter file")
            if not artifact.path.startswith(adapter_prefix):
                raise _error("adapter file is not under the v3 run adapter directory")
            if artifact.kind != "adapter_weights" or artifact.schema_version != "":
                raise _error("adapter file has the wrong artifact identity")
        if manifest_payload is None:
            raise _error("local adapter files are not bound by a train manifest")
        manifest_adapter_outputs = {
            _string(output.get("path"), context="adapter manifest output path"): output
            for output in (
                _mapping(item, context="train output")
                for item in _sequence(manifest_payload.get("outputs"), context="train outputs")
            )
            if output.get("kind") == "adapter_weights"
        }
        actual_files = {artifact.path: artifact for artifact in source.local_adapter.files}
        if set(manifest_adapter_outputs) != set(actual_files):
            raise _error("train manifest adapter file set does not match local adapter")
        for path, artifact in actual_files.items():
            _assert_ref(
                manifest_adapter_outputs[path],
                artifact,
                context=f"adapter manifest output {path}",
            )
        sources["adapter_files"] = [artifact.payload() for artifact in source.local_adapter.files]
        packaged_bytes = sum(artifact.bytes for artifact in source.local_adapter.files)
        tensor_files = [
            artifact
            for artifact in source.local_adapter.files
            if _ADAPTER_WEIGHT_FILE.fullmatch(artifact.path.rsplit("/", 1)[-1])
        ]
        if not tensor_files:
            raise _error("local adapter package has no recognized tensor weight file")
        adapter_section = {
            "available": True,
            "tree_sha256": source.local_adapter.tree_sha256,
            "packaged_adapter": {
                "bytes": packaged_bytes,
                "files": len(source.local_adapter.files),
                "boundary": (
                    "All regular files under the evaluated local adapter directory, "
                    "including configs and tokenizer assets."
                ),
            },
            "tensor_weights_only": {
                "bytes": sum(artifact.bytes for artifact in tensor_files),
                "files": len(tensor_files),
                "boundary": (
                    "Files named adapter_model.safetensors, adapter_model.bin, "
                    "or their numbered shards."
                ),
            },
            "evidence_kind": "observed_artifact_storage",
        }

    available = bool(
        runtime is not None or metrics is not None or adapter_section.get("available") is True
    )
    return (
        {
            "available": available,
            "train_stage_runtime": runtime_section,
            "source_code": source_code_section,
            "remote_execution": remote_execution_section,
            "tokens_seen": tokens_section,
            "end_to_end_token_throughput": throughput,
            "peak_gpu_memory": peak_section,
            "adapter_storage": adapter_section,
            "currency_cost": currency_cost,
            "full_finetune_savings": _unavailable("matched_full_finetune_evidence_not_supplied"),
        },
        sources,
    )


def _expected_efficiency_ratio(*, elapsed: float, gpu_count: int, successes: int) -> dict[str, Any]:
    common = {
        "unit": "gpu_seconds_per_full_call_exact_success",
        "full_call_exact_successes": successes,
        "basis": "generation_elapsed_seconds_x_configured_gpu_count",
    }
    if successes == 0:
        return {
            **common,
            "available": False,
            "value": None,
            "reason": "zero_full_call_exact_successes",
        }
    return {
        **common,
        "available": True,
        "value": round(elapsed * gpu_count / successes, 6),
        "reason": None,
    }


def _inference_arm_evidence(
    source: InferenceArmInput,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if source.model_kind not in {"base", "adapter"}:
        raise _error(f"{source.name} has an unsupported inference model kind")
    if not _SHA256.fullmatch(source.config_sha256):
        raise _error(f"{source.name} has an invalid evaluation config sha256")
    eval_prefix = f"eval/{source.model_kind}"
    eval_stage = f"eval-{source.model_kind}"
    _assert_artifact_identity(
        source.evaluation_report_artifact,
        expected_path=_run_path(source.run_id, f"{eval_prefix}/evaluation_report.json"),
        expected_kind="evaluation_report",
        expected_schema="sommelier.evaluation_report.v3",
        context=f"{source.name} evaluation report artifact",
    )
    sources: dict[str, Any] = {"evaluation_report": source.evaluation_report_artifact.payload()}
    if set(source.generation_artifacts) != {"en", "he"}:
        raise _error(f"{source.name} generation artifact set is incomplete")
    if set(source.actual_examples) != {"en", "he"}:
        raise _error(f"{source.name} observed example counts are incomplete")
    for language in ("en", "he"):
        _assert_artifact_identity(
            source.generation_artifacts[language],
            expected_path=_run_path(
                source.run_id,
                f"{eval_prefix}/generations.{language}.jsonl",
            ),
            expected_kind="generations",
            expected_schema="sommelier.generation.v2",
            context=f"{source.name} {language} generation artifact",
        )

    _assert_artifact_identity(
        source.evaluation_manifest_artifact,
        expected_path=_run_path(source.run_id, f"{eval_stage}_manifest.json"),
        expected_kind="manifest",
        expected_schema="sommelier.manifest.v1",
        context=f"{source.name} evaluation manifest artifact",
    )
    evaluation_manifest = dict(source.evaluation_manifest)
    if evaluation_manifest.get("schema_version") != "sommelier.manifest.v1":
        raise _error(f"{source.name} evaluation manifest has the wrong schema")
    if (
        evaluation_manifest.get("stage") != eval_stage
        or evaluation_manifest.get("status") != "succeeded"
    ):
        raise _error(f"{source.name} evaluation manifest is not a succeeded {eval_stage} stage")
    if evaluation_manifest.get("run_id") != source.run_id:
        raise _error(f"{source.name} evaluation manifest run_id does not match")
    if evaluation_manifest.get("config_sha256") != source.config_sha256:
        raise _error(f"{source.name} evaluation manifest config does not match")
    eval_git_commit = evaluation_manifest.get("git_commit")
    if eval_git_commit != "unknown" and (
        not isinstance(eval_git_commit, str) or not _GIT_COMMIT.fullmatch(eval_git_commit)
    ):
        raise _error(f"{source.name} evaluation manifest has no immutable source-code identity")

    if (source.telemetry is None) != (source.telemetry_artifact is None):
        raise _error(f"{source.name} inference telemetry evidence is incomplete")
    expected_outputs = {
        artifact.path: artifact
        for artifact in (
            source.evaluation_report_artifact,
            *source.generation_artifacts.values(),
            *((source.telemetry_artifact,) if source.telemetry_artifact is not None else ()),
        )
    }
    manifest_outputs = _sequence(
        evaluation_manifest.get("outputs"),
        context=f"{source.name} evaluation manifest outputs",
    )
    outputs_by_path = {
        _string(output.get("path"), context=f"{source.name} evaluation output path"): output
        for output in (
            _mapping(item, context=f"{source.name} evaluation output") for item in manifest_outputs
        )
    }
    if len(manifest_outputs) != len(expected_outputs) or set(outputs_by_path) != set(
        expected_outputs
    ):
        raise _error(f"{source.name} evaluation manifest output set does not match evidence")
    for path, artifact in expected_outputs.items():
        _assert_ref(
            outputs_by_path[path],
            artifact,
            context=f"{source.name} evaluation manifest output {path}",
        )

    _assert_artifact_identity(
        source.run_manifest_artifact,
        expected_path=_run_path(source.run_id, "manifest.json"),
        expected_kind="manifest",
        expected_schema="sommelier.manifest.v1",
        context=f"{source.name} root run manifest artifact",
    )
    run_manifest = dict(source.run_manifest)
    if run_manifest.get("schema_version") != "sommelier.manifest.v1":
        raise _error(f"{source.name} root run manifest has the wrong schema")
    if run_manifest.get("run_id") != source.run_id:
        raise _error(f"{source.name} root run manifest run_id does not match")
    if run_manifest.get("status") != "succeeded":
        raise _error(f"{source.name} root run manifest is not succeeded")
    stages = _mapping(run_manifest.get("stages"), context=f"{source.name} root run stages")
    if stages.get(eval_stage) != source.evaluation_manifest_artifact.path:
        raise _error(f"{source.name} root run manifest does not bind {eval_stage} evidence")
    config_ref = _mapping(run_manifest.get("config"), context=f"{source.name} root run config")
    _assert_artifact_identity(
        source.resolved_config_artifact,
        expected_path=_run_path(source.run_id, "config.resolved.yaml"),
        expected_kind="config",
        expected_schema="sommelier.config.v2",
        context=f"{source.name} resolved config artifact",
    )
    if source.resolved_config_artifact.sha256 != source.config_sha256:
        raise _error(f"{source.name} resolved config does not match evaluation")
    _assert_ref(
        config_ref,
        source.resolved_config_artifact,
        context=f"{source.name} root run config",
    )

    _assert_artifact_identity(
        source.runtime_artifact,
        expected_path=_run_path(source.run_id, "runtime_metadata.json"),
        expected_kind="runtime_metadata",
        expected_schema=RUNTIME_METADATA_SCHEMA,
        context=f"{source.name} runtime metadata artifact",
    )
    runtime = dict(source.runtime_metadata)
    if runtime.get("schema_version") != RUNTIME_METADATA_SCHEMA:
        raise _error(f"{source.name} runtime metadata has the wrong schema")
    if runtime.get("run_id") != source.run_id:
        raise _error(f"{source.name} runtime metadata run_id does not match")
    if runtime.get("config_sha256") != source.config_sha256:
        raise _error(f"{source.name} runtime metadata config does not match")
    runtime_hardware = _mapping(runtime.get("hardware"), context=f"{source.name} runtime hardware")
    if (
        runtime_hardware.get("source") != "config"
        or runtime_hardware.get("gpu") != source.configured_gpu_label
    ):
        raise _error(f"{source.name} runtime GPU does not match resolved config")
    runtime_source = _mapping(
        runtime.get("source_code"), context=f"{source.name} runtime source code"
    )
    runtime_git_commit = _string(
        runtime_source.get("git_commit"), context=f"{source.name} runtime git commit"
    )
    if not _GIT_COMMIT.fullmatch(runtime_git_commit):
        raise _error(f"{source.name} runtime source code is not immutable")
    if runtime_source.get("working_tree_clean") is not True:
        raise _error(f"{source.name} runtime source tree was not recorded clean")
    if eval_git_commit != runtime_git_commit:
        raise _error(f"{source.name} evaluation manifest git_commit does not match runtime")
    sources["evaluation_manifest"] = source.evaluation_manifest_artifact.payload()
    sources["run_manifest"] = source.run_manifest_artifact.payload()
    sources["resolved_config"] = source.resolved_config_artifact.payload()
    sources["runtime_metadata"] = source.runtime_artifact.payload()

    if source.efficiency is None or source.efficiency.get("available") is not True:
        reason = (
            "evaluation_report_has_no_inference_efficiency"
            if source.efficiency is None
            else str(source.efficiency.get("reason", "inference_efficiency_unavailable"))
        )
        return ({"available": False, "reason": reason}, sources)
    if source.telemetry is None or source.telemetry_artifact is None:
        raise _error(f"{source.name} inference efficiency is missing telemetry evidence")

    telemetry = dict(source.telemetry)
    if telemetry.get("schema_version") != INFERENCE_TELEMETRY_SCHEMA:
        raise _error(f"{source.name} inference telemetry has the wrong schema")
    if telemetry.get("run_id") != source.run_id:
        raise _error(f"{source.name} inference telemetry run_id does not match")
    if telemetry.get("model_kind") != source.model_kind:
        raise _error(f"{source.name} inference telemetry model_kind does not match")
    decoding = dict(source.decoding)
    if set(decoding) != {"temperature", "do_sample", "max_new_tokens"}:
        raise _error(f"{source.name} inference decoding contract is incomplete")
    temperature = decoding.get("temperature")
    max_new_tokens = decoding.get("max_new_tokens")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or float(temperature) != 0.0
        or decoding.get("do_sample") is not False
        or isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise _error(f"{source.name} inference decoding is not deterministic")
    if telemetry.get("decoding") != decoding:
        raise _error(f"{source.name} inference telemetry decoding does not match evaluation")
    _assert_artifact_identity(
        source.telemetry_artifact,
        expected_path=_run_path(source.run_id, f"{eval_prefix}/inference_telemetry.json"),
        expected_kind="inference_telemetry",
        expected_schema=INFERENCE_TELEMETRY_SCHEMA,
        context=f"{source.name} telemetry artifact",
    )
    _assert_ref(
        source.efficiency.get("telemetry_artifact"),
        source.telemetry_artifact,
        context=f"{source.name} telemetry artifact reference",
    )
    sources["inference_telemetry"] = source.telemetry_artifact.payload()

    expected_measurement = {
        "scope": GENERATION_TIMING_SCOPE,
        "aggregation": GENERATION_TIMING_AGGREGATION,
        "clock": "monotonic_seconds",
        "model_load_included": False,
        "parsing_and_artifact_io_included": False,
    }
    measurement = _mapping(telemetry.get("measurement"), context="telemetry measurement")
    if measurement != expected_measurement or source.efficiency.get("measurement") != measurement:
        raise _error(f"{source.name} inference measurement boundary does not match")
    timed_call_contract = _mapping(
        telemetry.get("timed_call_contract"),
        context="telemetry timed-call contract",
    )
    if (
        timed_call_contract != inference_timed_call_contract()
        or source.efficiency.get("timed_call_contract") != timed_call_contract
    ):
        raise _error(f"{source.name} inference timed-call contract does not match")
    warmup = _mapping(telemetry.get("warmup"), context="telemetry warmup contract")
    if warmup != inference_warmup_contract() or source.efficiency.get("warmup") != warmup:
        raise _error(f"{source.name} inference warmup contract does not match")
    sequential = _mapping(telemetry.get("sequential_run"), context="telemetry sequential run")
    expected_sequential = {
        "boundary": SEQUENTIAL_RUN_BOUNDARY,
        "concurrency": 1,
        "single_model_instance": True,
        "slice_order": ["en", "he"],
        "example_order": "formatted_test_order_within_slice",
    }
    if sequential != expected_sequential:
        raise _error(f"{source.name} inference sequential boundary does not match")
    if source.efficiency.get("sequential_run") != sequential:
        raise _error(f"{source.name} inference sequential metadata was altered")

    hardware = _mapping(telemetry.get("hardware"), context="telemetry hardware")
    gpu_label = _string(hardware.get("gpu_label"), context="telemetry GPU label")
    gpu_count = _integer(hardware.get("gpu_count"), context="telemetry GPU count", minimum=1)
    if gpu_count != gpu_count_from_label(gpu_label):
        raise _error(f"{source.name} inference GPU count does not match its label")
    if hardware.get("source") != "config.remote.gpu":
        raise _error(f"{source.name} inference GPU label is not config-bound")
    if gpu_label != source.configured_gpu_label:
        raise _error(f"{source.name} inference GPU does not match resolved config")
    if source.efficiency.get("hardware") != hardware:
        raise _error(f"{source.name} inference hardware metadata was altered")

    telemetry_slices = _mapping(telemetry.get("slices"), context="telemetry slices")
    efficiency_slices = _mapping(source.efficiency.get("slices"), context="efficiency slices")
    if set(telemetry_slices) != {"en", "he"} or set(efficiency_slices) != {"en", "he"}:
        raise _error(f"{source.name} inference evidence must contain en and he only")
    if set(source.exact_successes) != {"en", "he", "overall"}:
        raise _error(f"{source.name} exact-success evidence is incomplete")

    slices: dict[str, Any] = {}
    total_elapsed = 0.0
    total_examples = 0
    for language in ("en", "he"):
        raw = _mapping(telemetry_slices.get(language), context=f"telemetry {language}")
        reported = _mapping(efficiency_slices.get(language), context=f"efficiency {language}")
        artifact = source.generation_artifacts[language]
        _assert_artifact_identity(
            artifact,
            expected_path=_run_path(
                source.run_id,
                f"{eval_prefix}/generations.{language}.jsonl",
            ),
            expected_kind="generations",
            expected_schema="sommelier.generation.v2",
            context=f"{source.name} {language} generation artifact",
        )
        _assert_ref(
            raw.get("generation_artifact"),
            artifact,
            context=f"{source.name} {language} generation artifact",
        )
        for field in ("examples", "elapsed_seconds", "seconds_per_example"):
            if reported.get(field) != raw.get(field):
                raise _error(f"{source.name} {language} efficiency {field} was altered")
        examples = _integer(raw.get("examples"), context=f"{language} examples", minimum=1)
        observed_examples = _integer(
            source.actual_examples[language],
            context=f"{source.name} observed {language} examples",
            minimum=1,
        )
        if examples != observed_examples:
            raise _error(f"{source.name} {language} telemetry examples do not match generations")
        elapsed = _number(raw.get("elapsed_seconds"), context=f"{language} elapsed")
        per_example = _number(raw.get("seconds_per_example"), context=f"{language} seconds/example")
        if not math.isclose(per_example, elapsed / examples, rel_tol=0.0, abs_tol=1e-6):
            raise _error(f"{source.name} {language} seconds/example is inconsistent")
        successes = _integer(
            source.exact_successes[language],
            context=f"{source.name} {language} exact successes",
        )
        if successes > examples:
            raise _error(f"{source.name} {language} exact successes exceed examples")
        expected_ratio = _expected_efficiency_ratio(
            elapsed=elapsed,
            gpu_count=gpu_count,
            successes=successes,
        )
        if reported.get("gpu_seconds_per_full_call_exact_success") != expected_ratio:
            raise _error(f"{source.name} {language} GPU-seconds ratio is inconsistent")
        total_elapsed += elapsed
        total_examples += examples
        slices[language] = {
            "examples": examples,
            "generation_elapsed_seconds": elapsed,
            "seconds_per_example": per_example,
            "full_call_exact_successes": successes,
            "configured_gpu_seconds_per_full_call_exact_success": expected_ratio,
        }

    telemetry_total = _mapping(telemetry.get("total"), context="telemetry total")
    efficiency_overall = _mapping(source.efficiency.get("overall"), context="efficiency overall")
    if telemetry_total.get("examples") != total_examples:
        raise _error(f"{source.name} inference total examples is inconsistent")
    recorded_elapsed = _number(
        telemetry_total.get("elapsed_seconds"), context="telemetry total elapsed"
    )
    if not math.isclose(
        recorded_elapsed,
        total_elapsed,
        rel_tol=0.0,
        abs_tol=3e-6,
    ):
        raise _error(f"{source.name} inference total elapsed is inconsistent")
    recorded_seconds_per_example = _number(
        telemetry_total.get("seconds_per_example"),
        context="telemetry total seconds/example",
    )
    if not math.isclose(
        recorded_seconds_per_example,
        recorded_elapsed / total_examples,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise _error(f"{source.name} inference total seconds/example is inconsistent")
    for field in ("examples", "elapsed_seconds", "seconds_per_example"):
        if efficiency_overall.get(field) != telemetry_total.get(field):
            raise _error(f"{source.name} overall efficiency {field} was altered")
    successes = _integer(
        source.exact_successes["overall"], context=f"{source.name} overall successes"
    )
    if successes != source.exact_successes["en"] + source.exact_successes["he"]:
        raise _error(f"{source.name} overall exact successes is inconsistent")
    expected_overall_ratio = _expected_efficiency_ratio(
        elapsed=recorded_elapsed,
        gpu_count=gpu_count,
        successes=successes,
    )
    if efficiency_overall.get("gpu_seconds_per_full_call_exact_success") != expected_overall_ratio:
        raise _error(f"{source.name} overall GPU-seconds ratio is inconsistent")

    return (
        {
            "available": True,
            "measurement": measurement,
            "timed_call_contract": timed_call_contract,
            "warmup": warmup,
            "sequential_run": sequential,
            "decoding": decoding,
            "configured_gpu": {
                "label": gpu_label,
                "count": gpu_count,
                "source": "config.remote.gpu",
            },
            "slices": slices,
            "overall": {
                "examples": total_examples,
                "generation_elapsed_seconds": recorded_elapsed,
                "seconds_per_example": recorded_seconds_per_example,
                "full_call_exact_successes": successes,
                "configured_gpu_seconds_per_full_call_exact_success": (expected_overall_ratio),
            },
        },
        sources,
    )


def build_sovereign_tco_evidence(
    identity: TCOIdentity,
    *,
    tokenizer_tax: TokenizerTaxInput | None,
    training: TrainingInput | None,
    inference_arms: Sequence[InferenceArmInput],
    require_three_arm_matrix: bool = False,
) -> dict[str, Any]:
    """Builds v3 TCO evidence using only measured or deterministic quantities."""
    if not identity.run_id or not _SHA256.fullmatch(identity.config_sha256):
        raise _error("v3 TCO identity is incomplete")
    if not identity.tokenizer_id or not identity.tokenizer_revision:
        raise _error("v3 tokenizer identity is incomplete")
    if _GIT_COMMIT.fullmatch(identity.tokenizer_revision) is None:
        raise _error("v3 tokenizer revision is not immutable")
    if not _SHA256.fullmatch(identity.test_split_sha256):
        raise _error("v3 test split identity is invalid")
    if require_three_arm_matrix and identity.source_code_revision is None:
        raise _error("three-arm TCO evidence requires the v3 source-code revision")
    config_source: dict[str, object] | None = None
    if identity.resolved_config_artifact is not None:
        _assert_artifact_identity(
            identity.resolved_config_artifact,
            expected_path=_run_path(identity.run_id, "config.resolved.yaml"),
            expected_kind="config",
            expected_schema="sommelier.config.v2",
            context="resolved config artifact",
        )
        if identity.resolved_config_artifact.sha256 != identity.config_sha256:
            raise _error("resolved config artifact sha256 does not match v3 identity")
        config_source = identity.resolved_config_artifact.payload()

    tokenizer_section, tokenizer_sources = _tokenizer_evidence(identity, tokenizer_tax)
    training_section, training_sources = _training_evidence(identity, training, tokenizer_tax)
    inference: dict[str, Any] = {}
    inference_sources: dict[str, Any] = {}
    names: set[str] = set()
    for arm in inference_arms:
        if arm.name in names:
            raise _error(f"duplicate inference arm {arm.name}")
        names.add(arm.name)
        section, sources = _inference_arm_evidence(arm)
        inference[arm.name] = section
        inference_sources[arm.name] = sources
    required_arms = {"base", "v1_en", "v3_en_he"}
    if require_three_arm_matrix and names != required_arms:
        raise _error("sovereign TCO experiment requires base, v1_en, and v3_en_he arms")
    arms_by_name = {arm.name: arm for arm in inference_arms}
    package_identities: dict[str, dict[str, str]] = {}
    if names == required_arms:
        expected_kinds = {"base": "base", "v1_en": "adapter", "v3_en_he": "adapter"}
        for name, expected_kind in expected_kinds.items():
            if arms_by_name[name].model_kind != expected_kind:
                raise _error(f"{name} has the wrong model kind for the three-arm matrix")
        if arms_by_name["v3_en_he"].run_id != identity.run_id:
            raise _error("v3_en_he inference arm does not match the v3 TCO run")
        source_revisions: dict[str, str] = {}
        for name, arm in arms_by_name.items():
            source_code = _mapping(
                arm.runtime_metadata.get("source_code"),
                context=f"{name} runtime source code",
            )
            source_revisions[name] = _string(
                source_code.get("git_commit"),
                context=f"{name} runtime git commit",
            )
            packages = _mapping(
                arm.runtime_metadata.get("packages"),
                context=f"{name} runtime packages",
            )
            package_identities[name] = {
                _string(package, context=f"{name} runtime package name"): _string(
                    package_version,
                    context=f"{name} {package} package version",
                )
                for package, package_version in packages.items()
            }
            for required_package in _REQUIRED_INFERENCE_PACKAGES:
                version = package_identities[name].get(required_package)
                if version is None or version == "absent":
                    raise _error(
                        f"{name} runtime is missing required inference package {required_package}"
                    )
        if len(set(source_revisions.values())) != 1:
            raise _error("three-arm inference source-code revisions differ")
        if (
            identity.source_code_revision is not None
            and source_revisions["v3_en_he"] != identity.source_code_revision
        ):
            raise _error("v3 inference source code does not match the TCO identity")
    available_arms = [section for section in inference.values() if section.get("available") is True]
    if names != required_arms:
        cross_arm_comparability = {
            "available": False,
            "reason": "three_arm_matrix_not_supplied",
        }
    elif len(available_arms) != len(required_arms):
        cross_arm_comparability = {
            "available": False,
            "reason": "one_or_more_inference_arms_unavailable",
        }
    else:
        configurations = {
            (
                section["configured_gpu"]["label"],
                section["configured_gpu"]["count"],
            )
            for section in available_arms
        }
        reference = available_arms[0]
        if len(configurations) != 1:
            cross_arm_comparability = {
                "available": False,
                "reason": "configured_gpu_allocations_differ",
            }
        elif any(
            section["measurement"] != reference["measurement"]
            or section["timed_call_contract"] != reference["timed_call_contract"]
            or section["warmup"] != reference["warmup"]
            or section["sequential_run"] != reference["sequential_run"]
            for section in available_arms[1:]
        ):
            cross_arm_comparability = {
                "available": False,
                "reason": "inference_measurement_boundaries_differ",
            }
        elif any(section["decoding"] != reference["decoding"] for section in available_arms[1:]):
            cross_arm_comparability = {
                "available": False,
                "reason": "inference_decoding_configs_differ",
            }
        elif any(
            packages != package_identities["base"]
            for name, packages in package_identities.items()
            if name != "base"
        ):
            cross_arm_comparability = {
                "available": False,
                "reason": "runtime_package_identities_differ",
                "observed_packages_by_arm": package_identities,
            }
        else:
            label, count = next(iter(configurations))
            cross_arm_comparability = {
                "available": True,
                "configured_gpu": {"label": label, "count": count},
                "observed_packages": package_identities["base"],
                "boundary": (
                    "Identical sequential end-to-end generator-call measurement contract."
                ),
            }

    return {
        "schema_version": SOVEREIGN_TCO_EVIDENCE_SCHEMA,
        "subject": {
            "run_id": identity.run_id,
            "config_sha256": identity.config_sha256,
            "tokenizer": {
                "id": identity.tokenizer_id,
                "revision": identity.tokenizer_revision,
            },
        },
        "evidence_policy": {
            "scope": "bounded_observed_or_deterministically_projected_quantities",
            "currency_estimation": "forbidden_without_observed_billing",
            "full_finetune_savings_estimation": (
                "forbidden_without_matched_full_finetune_evidence"
            ),
        },
        "paired_tokenization": tokenizer_section,
        "qlora_training": training_section,
        "inference_efficiency": {
            "arms": inference,
            "cross_arm_comparability": cross_arm_comparability,
        },
        "explicitly_unavailable": {
            "currency_cost": training_section["currency_cost"],
            "full_finetune_savings": training_section["full_finetune_savings"],
        },
        "sources": {
            "resolved_config": config_source,
            "tokenization": tokenizer_sources,
            "training": training_sources,
            "inference": inference_sources,
        },
    }
