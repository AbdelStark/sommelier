from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, cast

from sommelier.artifacts import sha256_file
from sommelier.errors import EvaluationError, UserInputError
from sommelier.evaluation.generate import (
    GENERATION_SCHEMA,
    GENERATION_TIMING_AGGREGATION,
    GENERATION_TIMING_SCOPE,
    INFERENCE_TELEMETRY_FILENAME,
    INFERENCE_TELEMETRY_SCHEMA,
    SEQUENTIAL_RUN_BOUNDARY,
    ModelKind,
    evaluation_stage,
    gpu_count_from_label,
    inference_timed_call_contract,
    inference_warmup_contract,
)
from sommelier.evaluation.metrics import METRIC_NAMES, ScoredRecord, metric_components
from sommelier.evaluation.statistics import stable_bootstrap_seed
from sommelier.redaction import DuplicateJsonKeyError, loads_unique_json

EVALUATION_RELEASE_EVIDENCE_DIRNAME: Final = "evaluation_evidence"
EVALUATION_RELEASE_EVIDENCE_MANIFEST_SCHEMA: Final = (
    "sommelier.evaluation_release_evidence_manifest.v1"
)
EVALUATION_METRIC_COMPONENT_SCHEMA: Final = "sommelier.evaluation_metric_components.v1"
EXPERIMENT_REPORT_SCHEMA: Final = "sommelier.experiment_report.v2"
MANIFEST_FILENAME: Final = "manifest.json"
ARM_NAMES: Final = ("base", "v1_en", "v3_en_he")
LANGUAGES: Final = ("en", "he")

_PRIVACY_CONTRACT: Final = {
    "row_payload": "additive_metric_components_only",
    "pairing": "opaque_target_row_to_reference_row_index_map",
    "excluded_fields": [
        "example_id",
        "prompt_text",
        "target_text",
        "raw_text",
        "gold_call",
        "parsed_call",
    ],
}
_SHA256_CHARACTERS: Final = frozenset("0123456789abcdef")


def _payload_relative_files() -> frozenset[str]:
    return frozenset(
        {f"{arm}/correctness.{language}.jsonl" for arm in ARM_NAMES for language in LANGUAGES}
        | {f"{arm}/evaluation_manifest.json" for arm in ARM_NAMES}
        | {f"{arm}/{INFERENCE_TELEMETRY_FILENAME}" for arm in ARM_NAMES}
    )


EVALUATION_RELEASE_EVIDENCE_PAYLOAD_FILES: Final = _payload_relative_files()
EVALUATION_RELEASE_EVIDENCE_RELATIVE_FILES: Final = frozenset(
    {MANIFEST_FILENAME, *EVALUATION_RELEASE_EVIDENCE_PAYLOAD_FILES}
)
EVALUATION_RELEASE_EVIDENCE_BUNDLE_FILES: Final = frozenset(
    f"{EVALUATION_RELEASE_EVIDENCE_DIRNAME}/{name}"
    for name in EVALUATION_RELEASE_EVIDENCE_RELATIVE_FILES
)


@dataclass(frozen=True)
class EvaluationReleaseArm:
    """Validated arm inputs needed to derive the privacy-minimized release ledger."""

    name: str
    model_kind: ModelKind
    eval_dir: Path
    run_dir: Path
    artifact_root: Path
    scored: Mapping[str, Sequence[ScoredRecord]]
    example_ids: Mapping[str, Sequence[str]]
    prompt_set_sha256: Mapping[str, str]
    paired_reference_indices: Mapping[str, Sequence[int]]


def _writer_error(message: str) -> EvaluationError:
    return EvaluationError(
        f"evaluation release evidence {message}",
        hint="Use a fresh experiment output directory and rerun the experiment finalizer.",
    )


def _validation_error(message: str) -> UserInputError:
    return UserInputError(
        f"evaluation release evidence {message}",
        hint=(
            "Regenerate experiment_report.json and evaluation_evidence with "
            "sommelier report experiment; never edit release evidence by hand."
        ),
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _is_immutable_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _portable_path(path: Path, *, root: Path, context: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as error:
        raise _writer_error(f"{context} escapes the artifact root") from error
    if path.is_symlink() or not path.is_file():
        raise _writer_error(f"{context} is not a regular materialized file")
    return relative.as_posix()


def _canonical_json_line(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"


def _canonical_json_document(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_json_equal(observed: object, expected: object) -> bool:
    try:
        return _canonical_json_document(observed) == _canonical_json_document(expected)
    except (TypeError, ValueError):
        return False


def _ordered_index_digest(indices: Sequence[int]) -> str:
    encoded = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_rows(path: Path, records: Sequence[ScoredRecord]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        for index, record in enumerate(records):
            payload: dict[str, object] = {
                "schema_version": EVALUATION_METRIC_COMPONENT_SCHEMA,
                "row_index": index,
                "metrics": metric_components(record),
            }
            handle.write(_canonical_json_line(payload))


def _file_ref(path: Path, *, kind: str, schema_version: str) -> dict[str, object]:
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "kind": kind,
        "schema_version": schema_version,
    }


def _source_ref(path: Path, *, artifact_root: Path, context: str) -> dict[str, str]:
    return {
        "path": _portable_path(path, root=artifact_root, context=context),
        "sha256": sha256_file(path),
    }


def _mapping(value: object, *, context: str, writing: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        if writing:
            raise _writer_error(f"{context} must be a JSON object")
        raise _validation_error(f"{context} must be a JSON object")
    return cast("dict[str, Any]", value)


def ensure_fresh_evaluation_release_destination(out_dir: Path) -> None:
    """Reject a pre-existing evidence directory before any report bytes are written."""
    target = out_dir.resolve() / EVALUATION_RELEASE_EVIDENCE_DIRNAME
    if target.exists() or target.is_symlink():
        raise _writer_error(f"destination already exists: {target}")


def write_evaluation_release_evidence(
    *,
    out_dir: Path,
    experiment_report_path: Path,
    experiment_report: Mapping[str, Any],
    arms: Sequence[EvaluationReleaseArm],
) -> Path:
    """Atomically derive a deterministic, privacy-minimized row evidence directory."""
    if tuple(arm.name for arm in arms) != ARM_NAMES:
        raise _writer_error("requires ordered base, v1_en, and v3_en_he arms")
    if experiment_report.get("schema_version") != EXPERIMENT_REPORT_SCHEMA:
        raise _writer_error("requires the current experiment report schema")
    if experiment_report_path.is_symlink() or not experiment_report_path.is_file():
        raise _writer_error("requires a regular materialized experiment report")
    try:
        bound_report_value = loads_unique_json(experiment_report_path.read_text(encoding="utf-8"))
        if not isinstance(bound_report_value, dict):
            raise ValueError("experiment report must be an object")
        if _canonical_json_document(bound_report_value) != _canonical_json_document(
            experiment_report
        ):
            raise ValueError("experiment report mapping differs from bound file")
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        DuplicateJsonKeyError,
        TypeError,
        ValueError,
    ) as error:
        raise _writer_error("experiment report file and mapping differ or are invalid") from error
    target_parent = out_dir.resolve()
    target = target_parent / EVALUATION_RELEASE_EVIDENCE_DIRNAME
    ensure_fresh_evaluation_release_destination(target_parent)
    target_parent.mkdir(parents=True, exist_ok=True)

    report_arms = _mapping(experiment_report.get("arms"), context="report arms", writing=True)
    shared = _mapping(
        experiment_report.get("shared_evaluation_identity"),
        context="shared evaluation identity",
        writing=True,
    )
    shared_slices = _mapping(shared.get("slices"), context="shared slices", writing=True)
    shared_pairs = _mapping(
        shared.get("paired_cohorts"),
        context="shared paired cohorts",
        writing=True,
    )
    shared_hebrew_pair = _mapping(
        shared_pairs.get("he"),
        context="shared Hebrew pair",
        writing=True,
    )
    reference_indices = list(arms[0].paired_reference_indices.get("he", ()))
    english_examples = _mapping(
        shared_slices.get("en"), context="shared English slice", writing=True
    ).get("examples")
    hebrew_examples = _mapping(
        shared_slices.get("he"), context="shared Hebrew slice", writing=True
    ).get("examples")
    indices_sha256 = _ordered_index_digest(reference_indices)
    if (
        type(english_examples) is not int
        or type(hebrew_examples) is not int
        or english_examples <= 0
        or hebrew_examples <= 0
        or len(reference_indices) != hebrew_examples
        or len(set(reference_indices)) != len(reference_indices)
        or any(
            type(index) is not int or not 0 <= index < english_examples
            for index in reference_indices
        )
        or type(shared_hebrew_pair.get("pairs")) is not int
        or shared_hebrew_pair.get("pairs") != hebrew_examples
        or not _is_sha256(shared_hebrew_pair.get("pair_set_sha256"))
        or shared_hebrew_pair.get("reference_row_indices_sha256") != indices_sha256
        or any(
            set(arm.paired_reference_indices) != {"he"}
            or list(arm.paired_reference_indices["he"]) != reference_indices
            for arm in arms
        )
    ):
        raise _writer_error("shared Hebrew pair-index map drifted")
    pairing = {
        "he": {
            "reference_language": "en",
            "target_language": "he",
            "pairs": hebrew_examples,
            "pair_set_sha256": shared_hebrew_pair["pair_set_sha256"],
            "reference_row_indices": reference_indices,
            "reference_row_indices_sha256": indices_sha256,
        }
    }
    staging = Path(tempfile.mkdtemp(prefix=".evaluation-evidence.", dir=target_parent))
    staging.chmod(0o700)
    try:
        file_refs: dict[str, dict[str, object]] = {}
        arm_payloads: dict[str, dict[str, object]] = {}
        for arm in arms:
            if set(arm.scored) != set(LANGUAGES) or set(arm.example_ids) != set(LANGUAGES):
                raise _writer_error(f"{arm.name} does not contain exactly en/he rows")
            report_arm = _mapping(
                report_arms.get(arm.name),
                context=f"report arm {arm.name}",
                writing=True,
            )
            run_id = report_arm.get("run_id")
            config_sha256 = report_arm.get("config_sha256")
            if not isinstance(run_id, str) or not run_id or not _is_sha256(config_sha256):
                raise _writer_error(f"{arm.name} has an invalid report identity")
            if report_arm.get("model_kind") != arm.model_kind:
                raise _writer_error(f"{arm.name} model kind disagrees with its report")

            arm_dir = staging / arm.name
            arm_dir.mkdir(mode=0o700)
            slices: dict[str, dict[str, object]] = {}
            for language in LANGUAGES:
                records = arm.scored[language]
                ids = arm.example_ids[language]
                shared_slice = _mapping(
                    shared_slices.get(language),
                    context=f"shared {language} slice",
                    writing=True,
                )
                if len(records) != len(ids) or len(records) != shared_slice.get("examples"):
                    raise _writer_error(f"{arm.name} {language} row count drifted")
                rows_relative = f"{arm.name}/correctness.{language}.jsonl"
                rows_path = staging / rows_relative
                _write_rows(rows_path, records)
                file_refs[rows_relative] = _file_ref(
                    rows_path,
                    kind="metric_components",
                    schema_version=EVALUATION_METRIC_COMPONENT_SCHEMA,
                )
                slices[language] = {
                    "rows": len(records),
                    "ordered_example_ids_sha256": shared_slice.get("example_ids_sha256"),
                    "prompt_set_sha256": arm.prompt_set_sha256[language],
                    "correctness_file": rows_relative,
                }

            eval_manifest_source = arm.run_dir / f"{evaluation_stage(arm.model_kind)}_manifest.json"
            telemetry_source = arm.eval_dir / INFERENCE_TELEMETRY_FILENAME
            released_sources = (
                (
                    eval_manifest_source,
                    f"{arm.name}/evaluation_manifest.json",
                    "manifest",
                    "sommelier.manifest.v1",
                ),
                (
                    telemetry_source,
                    f"{arm.name}/{INFERENCE_TELEMETRY_FILENAME}",
                    "inference_telemetry",
                    INFERENCE_TELEMETRY_SCHEMA,
                ),
            )
            for source, relative, kind, schema in released_sources:
                _portable_path(
                    source,
                    root=arm.artifact_root,
                    context=f"{arm.name} {kind}",
                )
                destination = staging / relative
                shutil.copyfile(source, destination)
                file_refs[relative] = _file_ref(
                    destination,
                    kind=kind,
                    schema_version=schema,
                )

            artifacts = _mapping(
                report_arm.get("artifacts"),
                context=f"{arm.name} artifacts",
                writing=True,
            )
            source_artifacts = {
                name: dict(
                    _mapping(
                        artifacts.get(name),
                        context=f"{arm.name} artifact {name}",
                        writing=True,
                    )
                )
                for name in (
                    "evaluation_report",
                    "formatted_test",
                    "generations.en",
                    "generations.he",
                )
            }
            source_artifacts["evaluation_manifest"] = _source_ref(
                eval_manifest_source,
                artifact_root=arm.artifact_root,
                context=f"{arm.name} evaluation manifest",
            )
            source_artifacts["inference_telemetry"] = _source_ref(
                telemetry_source,
                artifact_root=arm.artifact_root,
                context=f"{arm.name} inference telemetry",
            )
            arm_payloads[arm.name] = {
                "run_id": run_id,
                "model_kind": arm.model_kind,
                "config_sha256": config_sha256,
                "source_artifacts": source_artifacts,
                "slices": slices,
            }

        manifest = {
            "schema_version": EVALUATION_RELEASE_EVIDENCE_MANIFEST_SCHEMA,
            "privacy": _PRIVACY_CONTRACT,
            "experiment_report": {
                "path": "experiment_report.json",
                "sha256": sha256_file(experiment_report_path),
                "schema_version": EXPERIMENT_REPORT_SCHEMA,
            },
            "pairing": pairing,
            "arms": arm_payloads,
            "files": {name: file_refs[name] for name in sorted(file_refs)},
        }
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = loads_unique_json(path.read_text(encoding="utf-8"))
    except (
        FileNotFoundError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        DuplicateJsonKeyError,
    ) as error:
        raise _validation_error(f"{context} is missing or invalid JSON") from error
    return _mapping(value, context=context)


def _closed(
    value: object,
    *,
    fields: frozenset[str],
    context: str,
) -> dict[str, Any]:
    payload = _mapping(value, context=context)
    if set(payload) != fields:
        raise _validation_error(f"{context} has unexpected or missing fields")
    return payload


def _validated_pairing(value: object, *, shared: Mapping[str, Any]) -> list[int]:
    pairing = _closed(value, fields=frozenset({"he"}), context="pairing")
    hebrew = _closed(
        pairing["he"],
        fields=frozenset(
            {
                "reference_language",
                "target_language",
                "pairs",
                "pair_set_sha256",
                "reference_row_indices",
                "reference_row_indices_sha256",
            }
        ),
        context="pairing.he",
    )
    shared_slices = _mapping(shared.get("slices"), context="shared slices")
    english_examples = _mapping(shared_slices.get("en"), context="shared English slice").get(
        "examples"
    )
    hebrew_examples = _mapping(shared_slices.get("he"), context="shared Hebrew slice").get(
        "examples"
    )
    shared_pairs = _closed(
        shared.get("paired_cohorts"),
        fields=frozenset({"he"}),
        context="shared paired cohorts",
    )
    shared_hebrew = _closed(
        shared_pairs["he"],
        fields=frozenset({"pairs", "pair_set_sha256", "reference_row_indices_sha256"}),
        context="shared paired cohorts.he",
    )
    raw_indices = hebrew["reference_row_indices"]
    if not isinstance(raw_indices, list):
        raise _validation_error("pairing.he reference_row_indices must be an array")
    indices = cast("list[object]", raw_indices)
    if (
        type(english_examples) is not int
        or type(hebrew_examples) is not int
        or english_examples <= 0
        or hebrew_examples <= 0
        or type(hebrew["pairs"]) is not int
        or hebrew["pairs"] != hebrew_examples
        or len(indices) != hebrew_examples
        or any(type(index) is not int for index in indices)
        or type(shared_hebrew["pairs"]) is not int
    ):
        raise _validation_error("pairing.he row counts or index types drifted")
    typed_indices = cast("list[int]", indices)
    indices_sha256 = _ordered_index_digest(typed_indices)
    if (
        hebrew["reference_language"] != "en"
        or hebrew["target_language"] != "he"
        or len(set(typed_indices)) != len(typed_indices)
        or any(not 0 <= index < english_examples for index in typed_indices)
        or not _is_sha256(hebrew["pair_set_sha256"])
        or not _is_sha256(hebrew["reference_row_indices_sha256"])
        or hebrew["reference_row_indices_sha256"] != indices_sha256
        or shared_hebrew
        != {
            "pairs": hebrew_examples,
            "pair_set_sha256": hebrew["pair_set_sha256"],
            "reference_row_indices_sha256": indices_sha256,
        }
    ):
        raise _validation_error("pairing.he identity or index map drifted")
    return typed_indices


def _validate_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise _validation_error("directory is missing or unsafe")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise _validation_error(f"contains a symlink: {relative}")
        if path.is_dir():
            observed_directories.add(relative)
        elif path.is_file():
            observed_files.add(relative)
        else:
            raise _validation_error(f"contains a non-regular entry: {relative}")
    missing = sorted(EVALUATION_RELEASE_EVIDENCE_RELATIVE_FILES - observed_files)
    extra = sorted(observed_files - EVALUATION_RELEASE_EVIDENCE_RELATIVE_FILES)
    extra_directories = sorted(observed_directories - set(ARM_NAMES))
    if missing or extra or extra_directories:
        raise _validation_error(
            "does not match its exact file allowlist "
            f"(missing={missing}, extra={extra}, extra_directories={extra_directories})"
        )


def _validate_file_map(root: Path, value: object) -> dict[str, dict[str, Any]]:
    files = _mapping(value, context="files")
    if set(files) != set(EVALUATION_RELEASE_EVIDENCE_PAYLOAD_FILES):
        raise _validation_error("files does not match the exact payload allowlist")
    expected_identity = {
        **{
            f"{arm}/correctness.{language}.jsonl": (
                "metric_components",
                EVALUATION_METRIC_COMPONENT_SCHEMA,
            )
            for arm in ARM_NAMES
            for language in LANGUAGES
        },
        **{
            f"{arm}/evaluation_manifest.json": ("manifest", "sommelier.manifest.v1")
            for arm in ARM_NAMES
        },
        **{
            f"{arm}/{INFERENCE_TELEMETRY_FILENAME}": (
                "inference_telemetry",
                INFERENCE_TELEMETRY_SCHEMA,
            )
            for arm in ARM_NAMES
        },
    }
    for name, (kind, schema) in expected_identity.items():
        reference = _closed(
            files[name],
            fields=frozenset({"sha256", "bytes", "kind", "schema_version"}),
            context=f"files.{name}",
        )
        path = root / name
        if (
            not _is_sha256(reference["sha256"])
            or reference["kind"] != kind
            or reference["schema_version"] != schema
            or isinstance(reference["bytes"], bool)
            or not isinstance(reference["bytes"], int)
            or reference["bytes"] < 0
            or reference["sha256"] != sha256_file(path)
            or reference["bytes"] != path.stat().st_size
        ):
            raise _validation_error(f"files.{name} does not bind its exact bytes")
    return {
        name: _mapping(files[name], context=f"files.{name}")
        for name in EVALUATION_RELEASE_EVIDENCE_PAYLOAD_FILES
    }


def _source_reference(value: object, *, context: str) -> dict[str, str]:
    payload = _closed(
        value,
        fields=frozenset({"path", "sha256"}),
        context=context,
    )
    path = payload["path"]
    if (
        not isinstance(path, str)
        or not path.startswith("runs/")
        or path.startswith("/")
        or ".." in path.split("/")
        or not _is_sha256(payload["sha256"])
    ):
        raise _validation_error(f"{context} is not a portable checksummed source")
    return cast("dict[str, str]", payload)


def _tco_source(report: Mapping[str, Any], *, arm: str, name: str) -> dict[str, Any]:
    tco = _mapping(report.get("sovereign_tco_evidence"), context="TCO evidence")
    sources = _mapping(tco.get("sources"), context="TCO sources")
    inference = _mapping(sources.get("inference"), context="TCO inference sources")
    arm_sources = _mapping(inference.get(arm), context=f"TCO {arm} sources")
    return _mapping(arm_sources.get(name), context=f"TCO {arm} {name}")


def _assert_source_matches_tco(
    reference: Mapping[str, str],
    report: Mapping[str, Any],
    *,
    arm: str,
    name: str,
    released_file: Mapping[str, Any],
    expected_kind: str,
    expected_schema: str,
) -> None:
    expected = _closed(
        _tco_source(report, arm=arm, name=name),
        fields=frozenset({"path", "sha256", "bytes", "kind", "schema_version"}),
        context=f"TCO {arm} {name}",
    )
    expected_bytes = expected["bytes"]
    if (
        reference["path"] != expected["path"]
        or reference["sha256"] != expected["sha256"]
        or type(expected_bytes) is not int
        or expected_bytes < 0
        or expected["kind"] != expected_kind
        or expected["schema_version"] != expected_schema
        or released_file.get("sha256") != expected["sha256"]
        or released_file.get("bytes") != expected_bytes
        or released_file.get("kind") != expected_kind
        or released_file.get("schema_version") != expected_schema
    ):
        raise _validation_error(f"{arm} {name} is stale or not TCO-bound")


def _validate_eval_manifest(
    path: Path,
    *,
    arm: str,
    run_id: str,
    model_kind: str,
    config_sha256: str,
    source_artifacts: Mapping[str, Mapping[str, str]],
    source_revision: object,
) -> None:
    payload = _load_json(path, context=f"{arm} evaluation manifest")
    if (
        payload.get("schema_version") != "sommelier.manifest.v1"
        or payload.get("stage") != f"eval-{model_kind}"
        or payload.get("status") != "succeeded"
        or payload.get("run_id") != run_id
        or payload.get("config_sha256") != config_sha256
        or payload.get("git_commit") != source_revision
    ):
        raise _validation_error(f"{arm} evaluation manifest identity drifted")
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not all(isinstance(item, dict) for item in outputs):
        raise _validation_error(f"{arm} evaluation manifest outputs are invalid")
    expected_outputs = {
        source_artifacts[name]["path"]: source_artifacts[name]["sha256"]
        for name in (
            "evaluation_report",
            "generations.en",
            "generations.he",
            "inference_telemetry",
        )
    }
    observed_outputs = {
        item.get("path"): item.get("sha256") for item in cast("list[dict[str, Any]]", outputs)
    }
    if observed_outputs != expected_outputs:
        raise _validation_error(f"{arm} evaluation manifest output bindings drifted")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not all(isinstance(item, dict) for item in inputs):
        raise _validation_error(f"{arm} evaluation manifest inputs are invalid")
    formatted = source_artifacts["formatted_test"]
    if (
        sum(
            item.get("path") == formatted["path"] and item.get("sha256") == formatted["sha256"]
            for item in cast("list[dict[str, Any]]", inputs)
        )
        != 1
    ):
        raise _validation_error(f"{arm} evaluation manifest does not bind the test split")


def _nonnegative_finite_number(value: object, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise _validation_error(f"{context} is not a non-negative finite number")
    return float(value)


def _validate_telemetry(
    path: Path,
    *,
    arm: str,
    run_id: str,
    model_kind: str,
    shared: Mapping[str, Any],
    source_artifacts: Mapping[str, Mapping[str, str]],
) -> None:
    payload = _load_json(path, context=f"{arm} inference telemetry")
    if (
        payload.get("schema_version") != INFERENCE_TELEMETRY_SCHEMA
        or payload.get("run_id") != run_id
        or payload.get("model_kind") != model_kind
        or payload.get("decoding") != shared.get("decoding")
    ):
        raise _validation_error(f"{arm} inference telemetry identity drifted")
    expected_measurement = {
        "scope": GENERATION_TIMING_SCOPE,
        "aggregation": GENERATION_TIMING_AGGREGATION,
        "clock": "monotonic_seconds",
        "model_load_included": False,
        "parsing_and_artifact_io_included": False,
    }
    if (
        payload.get("measurement") != expected_measurement
        or payload.get("timed_call_contract") != inference_timed_call_contract()
        or payload.get("warmup") != inference_warmup_contract()
        or payload.get("sequential_run")
        != {
            "boundary": SEQUENTIAL_RUN_BOUNDARY,
            "concurrency": 1,
            "single_model_instance": True,
            "slice_order": list(LANGUAGES),
            "example_order": "formatted_test_order_within_slice",
        }
    ):
        raise _validation_error(f"{arm} inference telemetry measurement contract drifted")
    hardware = _closed(
        payload.get("hardware"),
        fields=frozenset({"gpu_label", "gpu_count", "source"}),
        context=f"{arm} telemetry hardware",
    )
    gpu_label = hardware["gpu_label"]
    gpu_count = hardware["gpu_count"]
    if (
        not isinstance(gpu_label, str)
        or not gpu_label
        or type(gpu_count) is not int
        or gpu_count <= 0
        or gpu_count != gpu_count_from_label(gpu_label)
        or hardware["source"] != "config.remote.gpu"
    ):
        raise _validation_error(f"{arm} inference telemetry hardware contract drifted")
    slices = _mapping(payload.get("slices"), context=f"{arm} telemetry slices")
    if set(slices) != set(LANGUAGES):
        raise _validation_error(f"{arm} inference telemetry slice set drifted")
    shared_slices = _mapping(shared.get("slices"), context="shared slices")
    total_examples = 0
    total_elapsed_seconds = 0.0
    for language in LANGUAGES:
        slice_payload = _closed(
            slices[language],
            fields=frozenset(
                {"examples", "elapsed_seconds", "seconds_per_example", "generation_artifact"}
            ),
            context=f"{arm} telemetry {language}",
        )
        artifact = _closed(
            slice_payload.get("generation_artifact"),
            fields=frozenset({"path", "sha256", "bytes", "kind", "schema_version"}),
            context=f"{arm} telemetry {language} generation artifact",
        )
        source = source_artifacts[f"generations.{language}"]
        expected_examples = _mapping(
            shared_slices.get(language),
            context=f"shared {language} slice",
        ).get("examples")
        examples = slice_payload["examples"]
        artifact_bytes = artifact["bytes"]
        if (
            artifact.get("path") != source["path"]
            or artifact.get("sha256") != source["sha256"]
            or artifact.get("kind") != "generations"
            or artifact.get("schema_version") != GENERATION_SCHEMA
            or type(artifact_bytes) is not int
            or artifact_bytes < 0
            or type(expected_examples) is not int
            or expected_examples <= 0
            or type(examples) is not int
            or examples <= 0
            or examples != expected_examples
        ):
            raise _validation_error(f"{arm} telemetry {language} evidence drifted")
        elapsed_seconds = _nonnegative_finite_number(
            slice_payload["elapsed_seconds"],
            context=f"{arm} telemetry {language} elapsed_seconds",
        )
        seconds_per_example = _nonnegative_finite_number(
            slice_payload["seconds_per_example"],
            context=f"{arm} telemetry {language} seconds_per_example",
        )
        if not math.isclose(
            seconds_per_example,
            elapsed_seconds / examples,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise _validation_error(f"{arm} telemetry {language} timing is inconsistent")
        total_examples += examples
        total_elapsed_seconds += elapsed_seconds
    total = _closed(
        payload.get("total"),
        fields=frozenset({"examples", "elapsed_seconds", "seconds_per_example"}),
        context=f"{arm} telemetry total",
    )
    total_count = total["examples"]
    total_elapsed = _nonnegative_finite_number(
        total["elapsed_seconds"],
        context=f"{arm} telemetry total elapsed_seconds",
    )
    total_per_example = _nonnegative_finite_number(
        total["seconds_per_example"],
        context=f"{arm} telemetry total seconds_per_example",
    )
    if (
        type(total_count) is not int
        or total_count != total_examples
        or not math.isclose(
            total_elapsed,
            total_elapsed_seconds,
            rel_tol=0.0,
            abs_tol=(len(LANGUAGES) + 1) * 1e-6,
        )
        or not math.isclose(
            total_per_example,
            total_elapsed / total_examples,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise _validation_error(f"{arm} telemetry total timing is inconsistent")


ComponentRow = dict[str, dict[str, int]]


def _read_component_rows(path: Path, *, context: str) -> list[ComponentRow]:
    try:
        raw_lines = path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise _validation_error(f"{context} is unavailable") from error
    if not raw_lines:
        raise _validation_error(f"{context} is empty")
    rows: list[ComponentRow] = []
    for index, raw_line in enumerate(raw_lines):
        try:
            line = raw_line.decode("utf-8")
            value = loads_unique_json(line)
        except (UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKeyError) as error:
            raise _validation_error(f"{context} row {index} is invalid JSON") from error
        row = _closed(
            value,
            fields=frozenset({"schema_version", "row_index", "metrics"}),
            context=f"{context} row {index}",
        )
        if line != _canonical_json_line(row):
            raise _validation_error(f"{context} row {index} is not canonical JSONL")
        if (
            row["schema_version"] != EVALUATION_METRIC_COMPONENT_SCHEMA
            or type(row["row_index"]) is not int
            or row["row_index"] != index
        ):
            raise _validation_error(f"{context} row index or schema drifted")
        metrics = _mapping(row["metrics"], context=f"{context} row {index} metrics")
        if set(metrics) != set(METRIC_NAMES):
            raise _validation_error(f"{context} row {index} metric set drifted")
        typed_metrics: ComponentRow = {}
        for metric_name in METRIC_NAMES:
            component = _closed(
                metrics[metric_name],
                fields=frozenset({"numerator", "denominator"}),
                context=f"{context} row {index} {metric_name}",
            )
            numerator = component["numerator"]
            denominator = component["denominator"]
            if (
                isinstance(numerator, bool)
                or not isinstance(numerator, int)
                or isinstance(denominator, bool)
                or not isinstance(denominator, int)
                or denominator <= 0
                or numerator < 0
                or numerator > denominator
                or (metric_name != "argument_f1" and denominator != 1)
            ):
                raise _validation_error(f"{context} row {index} {metric_name} component is invalid")
            typed_metrics[metric_name] = {
                "numerator": numerator,
                "denominator": denominator,
            }
        valid = typed_metrics["valid_json_rate"]["numerator"]
        function = typed_metrics["function_name_accuracy"]["numerator"]
        arguments = typed_metrics["argument_exact_match"]["numerator"]
        full_call = typed_metrics["full_call_exact_match"]["numerator"]
        argument_f1 = typed_metrics["argument_f1"]
        if (
            function > valid
            or arguments > valid
            or full_call != function * arguments
            or (not valid and argument_f1["numerator"] != 0)
            or (arguments and argument_f1["numerator"] != argument_f1["denominator"])
        ):
            raise _validation_error(
                f"{context} row {index} metric components are mutually inconsistent"
            )
        rows.append(typed_metrics)
    return rows


def _totals(rows: Sequence[Mapping[str, Mapping[str, int]]]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "numerator": sum(row[name]["numerator"] for row in rows),
            "denominator": sum(row[name]["denominator"] for row in rows),
        }
        for name in METRIC_NAMES
    }


def _metric_values(
    rows: Sequence[Mapping[str, Mapping[str, int]]],
) -> dict[str, float]:
    totals = _totals(rows)
    return {
        name: component["numerator"] / component["denominator"]
        for name, component in totals.items()
    }


def _assert_metrics(
    rows: Sequence[Mapping[str, Mapping[str, int]]],
    reported: object,
    *,
    context: str,
) -> None:
    metrics = _mapping(reported, context=context)
    if set(metrics) != set(METRIC_NAMES):
        raise _validation_error(f"{context} metric set drifted")
    for name, total in _totals(rows).items():
        value = _mapping(metrics[name], context=f"{context}.{name}")
        numerator = value.get("numerator")
        denominator = value.get("denominator")
        observed = value.get("value")
        expected_value = total["numerator"] / total["denominator"]
        if (
            set(value) != {"value", "numerator", "denominator"}
            or type(numerator) is not int
            or type(denominator) is not int
            or isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(observed)
            or numerator != total["numerator"]
            or denominator != total["denominator"]
            or float(observed) != expected_value
        ):
            raise _validation_error(f"{context}.{name} is not backed by released rows")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_bootstrap_intervals(
    reference: Sequence[ComponentRow],
    candidate: Sequence[ComponentRow],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, dict[str, float]]:
    size = len(reference)
    samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    rng = random.Random(seed)
    for _ in range(resamples):
        counts = [0] * size
        for _ in range(size):
            counts[rng.randrange(size)] += 1
        for name in METRIC_NAMES:
            reference_numerator = sum(
                count * row[name]["numerator"] for count, row in zip(counts, reference, strict=True)
            )
            reference_denominator = sum(
                count * row[name]["denominator"]
                for count, row in zip(counts, reference, strict=True)
            )
            candidate_numerator = sum(
                count * row[name]["numerator"] for count, row in zip(counts, candidate, strict=True)
            )
            candidate_denominator = sum(
                count * row[name]["denominator"]
                for count, row in zip(counts, candidate, strict=True)
            )
            reference_value = reference_numerator / reference_denominator
            candidate_value = candidate_numerator / candidate_denominator
            samples[name].append(candidate_value - reference_value)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        name: {
            "lower": _percentile(values, alpha),
            "upper": _percentile(values, 1.0 - alpha),
        }
        for name, values in samples.items()
    }


def _mcnemar_p_value(first: int, second: int) -> float:
    discordant = first + second
    if discordant == 0:
        return 1.0
    lower = min(first, second)
    numerator = sum(math.comb(discordant, successes) for successes in range(lower + 1))
    return float(min(Fraction(2 * numerator, 1 << discordant), Fraction(1, 1)))


def _assert_mcnemar(
    reference: Sequence[ComponentRow],
    candidate: Sequence[ComponentRow],
    reported: object,
    *,
    language: str,
) -> None:
    if len(reference) != len(candidate):
        raise _validation_error(f"{language} paired row counts differ")
    reference_only = 0
    candidate_only = 0
    for reference_row, candidate_row in zip(reference, candidate, strict=True):
        reference_correct = bool(reference_row["full_call_exact_match"]["numerator"])
        candidate_correct = bool(candidate_row["full_call_exact_match"]["numerator"])
        reference_only += int(reference_correct and not candidate_correct)
        candidate_only += int(candidate_correct and not reference_correct)
    expected = {
        "method": "sommelier.exact_mcnemar.v1",
        "metric": "full_call_exact_match",
        "alternative": "two-sided",
        "pairs": len(reference),
        "discordant_pairs": reference_only + candidate_only,
        "discordant_counts": {
            "reference_correct_candidate_incorrect": reference_only,
            "reference_incorrect_candidate_correct": candidate_only,
        },
        "p_value": _mcnemar_p_value(reference_only, candidate_only),
    }
    if not _canonical_json_equal(reported, expected):
        raise _validation_error(f"{language} McNemar result is not backed by released rows")


def _assert_comparison(
    *,
    language: str,
    reference: Sequence[ComponentRow],
    candidate: Sequence[ComponentRow],
    reported: object,
    bootstrap: Mapping[str, Any],
    seed_offset: int,
) -> None:
    payload = _mapping(reported, context=f"{language} comparison")
    reference_values = _metric_values(reference)
    candidate_values = _metric_values(candidate)
    expected_deltas = {
        name: candidate_values[name] - reference_values[name] for name in METRIC_NAMES
    }
    if not _canonical_json_equal(payload.get("deltas"), expected_deltas):
        raise _validation_error(f"{language} deltas are not backed by released rows")
    ci = _mapping(payload.get("ci95"), context=f"{language} paired bootstrap")
    base_seed = bootstrap.get("seed")
    resamples = bootstrap.get("resamples")
    confidence_level = bootstrap.get("confidence_level")
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or base_seed != 42
        or isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples != 2000
        or isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or float(confidence_level) != 0.95
        or ci.get("method") != "sommelier.paired_bootstrap.v1"
        or ci.get("seed") != base_seed + seed_offset
        or ci.get("resamples") != resamples
        or ci.get("confidence_level") != confidence_level
    ):
        raise _validation_error(f"{language} paired-bootstrap identity drifted")
    intervals = _paired_bootstrap_intervals(
        reference,
        candidate,
        seed=base_seed + seed_offset,
        resamples=resamples,
        confidence_level=float(confidence_level),
    )
    if not _canonical_json_equal(ci.get("intervals"), intervals):
        raise _validation_error(
            f"{language} paired-bootstrap intervals are not backed by released rows"
        )
    _assert_mcnemar(reference, candidate, payload.get("mcnemar"), language=language)


def _assert_paired_slice(
    *,
    arm: str,
    reference: Sequence[ComponentRow],
    target: Sequence[ComponentRow],
    reference_slice_examples: int,
    pair_set_sha256: str,
    reported: object,
) -> None:
    if not reference or len(reference) != len(target):
        raise _validation_error(f"{arm} matched Hebrew cohort is empty or misaligned")
    payload = _closed(
        reported,
        fields=frozenset(
            {
                "reference_language",
                "target_language",
                "pairs",
                "coverage",
                "pair_set_sha256",
                "reference",
                "target",
                "gaps",
                "gap_ci95",
            }
        ),
        context=f"report {arm} paired_slices.he",
    )
    pairs = payload["pairs"]
    coverage = _closed(
        payload["coverage"],
        fields=frozenset(
            {
                "paired",
                "reference_slice_examples",
                "target_slice_examples",
                "reference_fraction",
            }
        ),
        context=f"report {arm} paired_slices.he.coverage",
    )
    expected_coverage = {
        "paired": len(target),
        "reference_slice_examples": reference_slice_examples,
        "target_slice_examples": len(target),
        "reference_fraction": len(target) / reference_slice_examples,
    }
    if (
        payload["reference_language"] != "en"
        or payload["target_language"] != "he"
        or type(pairs) is not int
        or pairs != len(target)
        or payload["pair_set_sha256"] != pair_set_sha256
        or type(coverage["paired"]) is not int
        or type(coverage["reference_slice_examples"]) is not int
        or type(coverage["target_slice_examples"]) is not int
        or isinstance(coverage["reference_fraction"], bool)
        or not isinstance(coverage["reference_fraction"], (int, float))
        or not math.isfinite(coverage["reference_fraction"])
        or coverage != expected_coverage
    ):
        raise _validation_error(f"{arm} matched Hebrew cohort identity drifted")
    reference_payload = _closed(
        payload["reference"],
        fields=frozenset({"metrics"}),
        context=f"report {arm} paired_slices.he.reference",
    )
    target_payload = _closed(
        payload["target"],
        fields=frozenset({"metrics"}),
        context=f"report {arm} paired_slices.he.target",
    )
    _assert_metrics(
        reference,
        reference_payload["metrics"],
        context=f"report {arm} matched English metrics",
    )
    _assert_metrics(
        target,
        target_payload["metrics"],
        context=f"report {arm} matched Hebrew metrics",
    )
    reference_values = _metric_values(reference)
    target_values = _metric_values(target)
    expected_gaps = {name: target_values[name] - reference_values[name] for name in METRIC_NAMES}
    if not _canonical_json_equal(payload["gaps"], expected_gaps):
        raise _validation_error(f"{arm} matched Hebrew gaps are not backed by released rows")
    interval = _closed(
        payload["gap_ci95"],
        fields=frozenset({"method", "seed", "confidence_level", "resamples", "intervals"}),
        context=f"report {arm} matched Hebrew paired bootstrap",
    )
    expected_seed = stable_bootstrap_seed(42, "language-gap:en:he")
    if (
        interval["method"] != "sommelier.paired_bootstrap.v1"
        or type(interval["seed"]) is not int
        or interval["seed"] != expected_seed
        or type(interval["resamples"]) is not int
        or interval["resamples"] != 2000
        or isinstance(interval["confidence_level"], bool)
        or not isinstance(interval["confidence_level"], (int, float))
        or float(interval["confidence_level"]) != 0.95
    ):
        raise _validation_error(f"{arm} matched Hebrew paired-bootstrap identity drifted")
    expected_intervals = _paired_bootstrap_intervals(
        reference,
        target,
        seed=expected_seed,
        resamples=2000,
        confidence_level=0.95,
    )
    if not _canonical_json_equal(interval["intervals"], expected_intervals):
        raise _validation_error(f"{arm} matched Hebrew intervals are not backed by released rows")


def validate_evaluation_release_evidence(
    *,
    bundle_dir: Path,
    experiment_report: Mapping[str, Any],
) -> None:
    """Validate the exact public evidence tree and recompute all outcome statistics."""
    root = bundle_dir / EVALUATION_RELEASE_EVIDENCE_DIRNAME
    _validate_tree(root)
    manifest = _closed(
        _load_json(root / MANIFEST_FILENAME, context="manifest"),
        fields=frozenset(
            {"schema_version", "privacy", "experiment_report", "pairing", "arms", "files"}
        ),
        context="manifest",
    )
    if manifest["schema_version"] != EVALUATION_RELEASE_EVIDENCE_MANIFEST_SCHEMA:
        raise _validation_error("manifest schema drifted")
    if manifest["privacy"] != _PRIVACY_CONTRACT:
        raise _validation_error("privacy contract drifted")
    if experiment_report.get("schema_version") != EXPERIMENT_REPORT_SCHEMA:
        raise _validation_error("experiment report schema drifted")
    report_ref = _closed(
        manifest["experiment_report"],
        fields=frozenset({"path", "sha256", "schema_version"}),
        context="experiment_report",
    )
    report_path = bundle_dir / "experiment_report.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise _validation_error("bound experiment report is not a regular file")
    bound_experiment_report = _load_json(report_path, context="bound experiment report")
    try:
        reports_match = _canonical_json_document(
            bound_experiment_report
        ) == _canonical_json_document(experiment_report)
    except (TypeError, ValueError) as error:
        raise _validation_error(
            "supplied experiment report mapping is not canonical JSON"
        ) from error
    if not reports_match:
        raise _validation_error("supplied experiment report differs from the bound file")
    experiment_report = bound_experiment_report
    if (
        report_ref["path"] != "experiment_report.json"
        or report_ref["schema_version"] != EXPERIMENT_REPORT_SCHEMA
        or report_ref["sha256"] != sha256_file(report_path)
    ):
        raise _validation_error("is stale relative to experiment_report.json")
    released_files = _validate_file_map(root, manifest["files"])

    report_arms = _mapping(experiment_report.get("arms"), context="report arms")
    shared = _mapping(
        experiment_report.get("shared_evaluation_identity"),
        context="shared evaluation identity",
    )
    shared_slices = _mapping(shared.get("slices"), context="shared slices")
    paired_reference_indices = _validated_pairing(manifest["pairing"], shared=shared)
    preregistration = _mapping(
        experiment_report.get("preregistration"),
        context="preregistration",
    )
    finalizer = _mapping(
        preregistration.get("finalizer_source_code"),
        context="finalizer source code",
    )
    source_revision = finalizer.get("git_commit")
    if not _is_immutable_revision(source_revision):
        raise _validation_error("finalizer source revision is not immutable")
    manifest_arms = _mapping(manifest["arms"], context="arms")
    if set(manifest_arms) != set(ARM_NAMES):
        raise _validation_error("arm set drifted")

    all_rows: dict[str, dict[str, list[ComponentRow]]] = {}
    for arm_name in ARM_NAMES:
        arm = _closed(
            manifest_arms[arm_name],
            fields=frozenset(
                {"run_id", "model_kind", "config_sha256", "source_artifacts", "slices"}
            ),
            context=f"arms.{arm_name}",
        )
        report_arm = _mapping(report_arms.get(arm_name), context=f"report arm {arm_name}")
        if any(
            arm[field] != report_arm.get(field)
            for field in ("run_id", "model_kind", "config_sha256")
        ):
            raise _validation_error(f"{arm_name} identity is stale")
        run_id_value = arm["run_id"]
        model_kind_value = arm["model_kind"]
        config_sha256_value = arm["config_sha256"]
        expected_kind = "base" if arm_name == "base" else "adapter"
        if (
            not isinstance(run_id_value, str)
            or not run_id_value
            or model_kind_value != expected_kind
            or not _is_sha256(config_sha256_value)
        ):
            raise _validation_error(f"{arm_name} identity is invalid")
        run_id = run_id_value
        model_kind = cast("str", model_kind_value)
        config_sha256 = cast("str", config_sha256_value)
        sources = _mapping(
            arm["source_artifacts"],
            context=f"{arm_name} source artifacts",
        )
        expected_source_names = {
            "evaluation_report",
            "formatted_test",
            "generations.en",
            "generations.he",
            "evaluation_manifest",
            "inference_telemetry",
        }
        if set(sources) != expected_source_names:
            raise _validation_error(f"{arm_name} source artifact set drifted")
        source_artifacts = {
            name: _source_reference(sources[name], context=f"{arm_name} {name}")
            for name in expected_source_names
        }
        report_artifacts = _mapping(
            report_arm.get("artifacts"),
            context=f"report {arm_name} artifacts",
        )
        for name in (
            "evaluation_report",
            "formatted_test",
            "generations.en",
            "generations.he",
        ):
            if source_artifacts[name] != report_artifacts.get(name):
                raise _validation_error(f"{arm_name} {name} is stale")
        released_source_identity = {
            "evaluation_manifest": (
                f"{arm_name}/evaluation_manifest.json",
                "manifest",
                "sommelier.manifest.v1",
            ),
            "inference_telemetry": (
                f"{arm_name}/{INFERENCE_TELEMETRY_FILENAME}",
                "inference_telemetry",
                INFERENCE_TELEMETRY_SCHEMA,
            ),
        }
        for name, (relative, kind, schema) in released_source_identity.items():
            _assert_source_matches_tco(
                source_artifacts[name],
                experiment_report,
                arm=arm_name,
                name=name,
                released_file=released_files[relative],
                expected_kind=kind,
                expected_schema=schema,
            )

        _validate_eval_manifest(
            root / arm_name / "evaluation_manifest.json",
            arm=arm_name,
            run_id=run_id,
            model_kind=model_kind,
            config_sha256=config_sha256,
            source_artifacts=source_artifacts,
            source_revision=source_revision,
        )
        _validate_telemetry(
            root / arm_name / INFERENCE_TELEMETRY_FILENAME,
            arm=arm_name,
            run_id=run_id,
            model_kind=model_kind,
            shared=shared,
            source_artifacts=source_artifacts,
        )

        slices = _mapping(arm["slices"], context=f"{arm_name} slices")
        if set(slices) != set(LANGUAGES):
            raise _validation_error(f"{arm_name} slice set drifted")
        metrics = _mapping(
            report_arm.get("metrics"),
            context=f"report {arm_name} metrics",
        )
        reported_slices = _mapping(
            metrics.get("slices"),
            context=f"report {arm_name} slices",
        )
        arm_rows: dict[str, list[ComponentRow]] = {}
        for language in LANGUAGES:
            slice_payload = _closed(
                slices[language],
                fields=frozenset(
                    {
                        "rows",
                        "ordered_example_ids_sha256",
                        "prompt_set_sha256",
                        "correctness_file",
                    }
                ),
                context=f"{arm_name} {language} slice",
            )
            shared_slice = _mapping(
                shared_slices.get(language),
                context=f"shared {language} slice",
            )
            expected_file = f"{arm_name}/correctness.{language}.jsonl"
            slice_rows = slice_payload["rows"]
            if (
                type(slice_rows) is not int
                or slice_rows <= 0
                or slice_rows != shared_slice.get("examples")
                or not _is_sha256(slice_payload["ordered_example_ids_sha256"])
                or not _is_sha256(slice_payload["prompt_set_sha256"])
                or slice_payload["ordered_example_ids_sha256"]
                != shared_slice.get("example_ids_sha256")
                or slice_payload["prompt_set_sha256"] != shared_slice.get("prompt_set_sha256")
                or slice_payload["correctness_file"] != expected_file
            ):
                raise _validation_error(f"{arm_name} {language} cohort binding drifted")
            rows = _read_component_rows(root / expected_file, context=expected_file)
            if len(rows) != slice_rows:
                raise _validation_error(f"{arm_name} {language} row count drifted")
            _assert_metrics(
                rows,
                reported_slices.get(language),
                context=f"report {arm_name} {language} metrics",
            )
            arm_rows[language] = rows
        _assert_metrics(
            [*arm_rows["en"], *arm_rows["he"]],
            metrics.get("overall"),
            context=f"report {arm_name} overall metrics",
        )
        all_rows[arm_name] = arm_rows

    comparisons = _mapping(experiment_report.get("comparisons"), context="comparisons")
    v3_vs_v1 = _mapping(
        comparisons.get("v3_vs_v1"),
        context="v3_vs_v1 comparison",
    )
    bootstrap = _mapping(experiment_report.get("bootstrap"), context="bootstrap")
    for offset, language in enumerate(LANGUAGES):
        _assert_comparison(
            language=language,
            reference=all_rows["v1_en"][language],
            candidate=all_rows["v3_en_he"][language],
            reported=v3_vs_v1.get(language),
            bootstrap=bootstrap,
            seed_offset=offset,
        )

    shared_pairs = _mapping(shared.get("paired_cohorts"), context="shared paired cohorts")
    shared_hebrew_pair = _mapping(shared_pairs.get("he"), context="shared paired cohorts.he")
    pair_set_sha256 = cast("str", shared_hebrew_pair["pair_set_sha256"])
    for arm_name in ARM_NAMES:
        report_arm = _mapping(report_arms[arm_name], context=f"report arm {arm_name}")
        paired_slices = _closed(
            report_arm.get("paired_slices"),
            fields=frozenset({"he"}),
            context=f"report {arm_name} paired_slices",
        )
        matched_english = [all_rows[arm_name]["en"][index] for index in paired_reference_indices]
        _assert_paired_slice(
            arm=arm_name,
            reference=matched_english,
            target=all_rows[arm_name]["he"],
            reference_slice_examples=len(all_rows[arm_name]["en"]),
            pair_set_sha256=pair_set_sha256,
            reported=paired_slices["he"],
        )
