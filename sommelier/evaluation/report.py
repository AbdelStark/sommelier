from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

from sommelier.artifacts import (
    ArtifactRef,
    make_artifact_ref,
    read_json_with_schema,
    sha256_file,
    write_artifact_atomic,
)
from sommelier.config import SommelierConfig, compute_config_digest, load_config
from sommelier.data.types import ToolCall
from sommelier.errors import EvaluationError, InvariantViolation, UserInputError
from sommelier.evaluation.generate import (
    GENERATION_SCHEMA,
    GENERATION_TIMING_AGGREGATION,
    GENERATION_TIMING_SCOPE,
    INFERENCE_TELEMETRY_FILENAME,
    INFERENCE_TELEMETRY_SCHEMA,
    SEQUENTIAL_RUN_BOUNDARY,
    AdapterRef,
    ModelKind,
    evaluation_stage,
    gpu_count_from_label,
    inference_timed_call_contract,
    inference_warmup_contract,
    read_test_slices,
    slice_filename,
)
from sommelier.evaluation.metrics import ScoredRecord, compute_metrics
from sommelier.evaluation.parse import ParseStatus
from sommelier.evaluation.statistics import (
    paired_bootstrap_intervals,
    stable_bootstrap_seed,
)
from sommelier.formatting.chat import FORMATTED_EXAMPLE_SCHEMA as FORMATTED_SCHEMA
from sommelier.formatting.chat import validate_assistant_target
from sommelier.manifests import (
    build_stage_manifest,
    update_run_manifest,
    write_stage_manifest,
)
from sommelier.redaction import redact_configured_fields
from sommelier.run_context import RunContext, read_jsonl_records, record_stage_success
from sommelier.runtime_metadata import runtime_section
from sommelier.security import validate_no_secrets
from sommelier.tracking import track_stage_metrics

EVALUATION_REPORT_SCHEMA: Final = "sommelier.evaluation_report.v3"
COMPARISON_REPORT_SCHEMA: Final = "sommelier.comparison_report.v3"

REPORT_FILENAME: Final = "evaluation_report.json"
COMPARISON_FILENAME: Final = "comparison_report.json"

_COMPARABILITY_FIELDS: Final = (
    "model_identity",
    "config_sha256",
    "split",
    "test_split_sha256",
    "parser_version",
    "decoding",
)


def prompt_set_digest(prompt_digests: list[str]) -> str:
    """Digest over the ordered per-example prompt digests of a test split."""
    return hashlib.sha256("\n".join(prompt_digests).encode("utf-8")).hexdigest()


def paired_set_digest(entries: list[dict[str, str]]) -> str:
    """Digest over ordered root/translation identities and both prompts."""
    payload = "\n".join(
        json.dumps(entry, separators=(",", ":"), sort_keys=True) for entry in entries
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_scored_records(
    formatted_examples: list[dict[str, object]],
    generations: list[dict[str, object]],
) -> list[ScoredRecord]:
    """Joins generations with formatted examples into scorable records.

    Every generation must reference a known example and carry the same
    prompt digest the formatter recorded; anything else breaks prompt
    identity (INV-ARCH-004) and fails instead of being skipped.

    v1 scores against the first gold call. Multi-call golds stay in the
    denominator; the single-call parser can never match them, and both
    model kinds face the identical contract, so comparisons remain fair.
    """
    formatted_by_id = {str(example["example_id"]): example for example in formatted_examples}
    if len(generations) != len(formatted_examples):
        raise EvaluationError(
            f"generation count {len(generations)} does not match "
            f"test split size {len(formatted_examples)}",
            hint="Re-run evaluation so every test prompt has exactly one generation.",
        )

    records: list[ScoredRecord] = []
    for generation in generations:
        example_id = str(generation["example_id"])
        example = formatted_by_id.get(example_id)
        if example is None:
            raise EvaluationError(
                f"generation references unknown example {example_id}",
                hint="Generations must come from the same formatted test split.",
            )
        if generation["prompt_sha256"] != example["prompt_sha256"]:
            raise InvariantViolation(
                f"prompt digest mismatch for example {example_id}",
                hint="Regenerate with the stored formatted split; prompts "
                "must be identical between formatting and evaluation.",
            )
        gold_calls = validate_assistant_target(
            str(example["target_text"]),
            context=f"example {example_id}",
        )
        records.append(
            ScoredRecord(
                example_id=example_id,
                parse_status=cast(ParseStatus, generation["parse_status"]),
                parsed_call=cast("ToolCall | None", generation["parsed_call"]),
                gold_call=gold_calls[0],
            )
        )
    return records


def _uniform_decoding(generations: list[dict[str, object]]) -> dict[str, object]:
    decodings = {json.dumps(generation["decoding"], sort_keys=True) for generation in generations}
    if len(decodings) != 1:
        raise InvariantViolation(
            "generations mix different decoding configs",
            hint="Regenerate the whole split with one deterministic decoding config.",
        )
    return cast("dict[str, object]", json.loads(decodings.pop()))


def _canonicalized_pair_record(record: ScoredRecord, root_id: str) -> ScoredRecord:
    return ScoredRecord(
        example_id=root_id,
        parse_status=record["parse_status"],
        parsed_call=record["parsed_call"],
        gold_call=record["gold_call"],
    )


def _paired_slice_reports(
    config: SommelierConfig,
    examples_by_slice: dict[str, list[dict[str, object]]],
    scored_by_slice: dict[str, list[ScoredRecord]],
) -> dict[str, Any]:
    """Matched language cohorts keyed by each translated target language.

    Marginal slice metrics remain useful operationally, but the primary
    language-tax estimate must compare Hebrew/French rows only against the
    exact English roots whose translations survived production.
    """
    reference_language = config.root_dataset.language
    if reference_language not in examples_by_slice:
        return {}
    reference_examples = examples_by_slice[reference_language]
    reference_examples_by_id = {
        str(example["example_id"]): example for example in reference_examples
    }
    reference_scored_by_id = {
        record["example_id"]: record for record in scored_by_slice[reference_language]
    }

    reports: dict[str, Any] = {}
    for target_language in config.eval.slices:
        if target_language == reference_language:
            continue
        target_examples = examples_by_slice[target_language]
        target_scored_by_id = {
            record["example_id"]: record for record in scored_by_slice[target_language]
        }
        seen_roots: set[str] = set()
        paired_reference: list[ScoredRecord] = []
        paired_target: list[ScoredRecord] = []
        digest_entries: list[dict[str, str]] = []
        for target_example in target_examples:
            target_id = str(target_example["example_id"])
            root_id_value = target_example.get("source_example_id")
            if not isinstance(root_id_value, str) or not root_id_value:
                raise EvaluationError(
                    f"slice {target_language} contains unpaired example {target_id}",
                    hint="Translated evaluation rows must carry source_example_id.",
                )
            root_id = root_id_value
            if root_id in seen_roots:
                raise EvaluationError(
                    f"slice {target_language} pairs root example {root_id} more than once",
                    hint="Keep exactly one translated evaluation row per root.",
                )
            seen_roots.add(root_id)
            root_example = reference_examples_by_id.get(root_id)
            root_scored = reference_scored_by_id.get(root_id)
            target_scored = target_scored_by_id.get(target_id)
            if root_example is None or root_scored is None or target_scored is None:
                raise EvaluationError(
                    f"slice {target_language} references missing evaluated root {root_id}",
                    hint="Evaluate the root and translated slices from the same paired split.",
                )
            if root_scored["gold_call"] != target_scored["gold_call"]:
                raise InvariantViolation(
                    f"paired gold call mismatch for {root_id} and {target_id}",
                    hint="Re-run data preparation; paired gold answers must be identical.",
                )
            paired_reference.append(root_scored)
            paired_target.append(_canonicalized_pair_record(target_scored, root_id))
            digest_entries.append(
                {
                    "reference_example_id": root_id,
                    "target_example_id": target_id,
                    "reference_prompt_sha256": str(root_example["prompt_sha256"]),
                    "target_prompt_sha256": str(target_example["prompt_sha256"]),
                }
            )

        reference_metrics = compute_metrics(paired_reference)
        target_metrics = compute_metrics(paired_target)
        seed = stable_bootstrap_seed(
            config.project.seed,
            f"language-gap:{reference_language}:{target_language}",
        )
        reports[target_language] = {
            "reference_language": reference_language,
            "target_language": target_language,
            "pairs": len(paired_target),
            "coverage": {
                "paired": len(paired_target),
                "reference_slice_examples": len(reference_examples),
                "target_slice_examples": len(target_examples),
                "reference_fraction": len(paired_target) / len(reference_examples),
            },
            "pair_set_sha256": paired_set_digest(digest_entries),
            "reference": {"metrics": reference_metrics},
            "target": {"metrics": target_metrics},
            "gaps": _metric_deltas(reference_metrics, target_metrics),
            "gap_ci95": paired_bootstrap_intervals(
                paired_reference,
                paired_target,
                seed=seed,
            ),
        }
    return reports


def _gpu_seconds_per_full_call_exact_success(
    *,
    elapsed_seconds: float,
    gpu_count: int,
    successes: int,
) -> dict[str, Any]:
    """Derives a hardware-time ratio without inventing a currency cost."""
    common: dict[str, Any] = {
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
        "value": round(elapsed_seconds * gpu_count / successes, 6),
        "reason": None,
    }


def _number(payload: dict[str, Any], field: str, *, context: str) -> float:
    value = payload.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise EvaluationError(
            f"{context}: {field} must be a non-negative number",
            hint="Re-run generation to rebuild inference telemetry.",
        )
    return float(value)


def _inference_efficiency_section(
    config: SommelierConfig,
    *,
    eval_dir: Path,
    model_kind: ModelKind,
    context: RunContext,
    decoding: dict[str, object],
    slices: dict[str, Any],
    overall_metrics: dict[str, Any],
    generation_refs: dict[str, ArtifactRef],
) -> tuple[dict[str, Any], ArtifactRef | None]:
    """Loads, validates, and derives TCO-oriented inference measurements.

    Older generation directories without the aggregate artifact remain
    reportable. Once the artifact exists, identity, counts, timing semantics,
    and generation hashes are fail-closed before any efficiency ratio is used.
    """
    telemetry_path = eval_dir / INFERENCE_TELEMETRY_FILENAME
    if not telemetry_path.exists():
        return (
            {
                "available": False,
                "reason": "inference_telemetry_artifact_missing",
            },
            None,
        )

    telemetry = read_json_with_schema(telemetry_path, expected_schema=INFERENCE_TELEMETRY_SCHEMA)
    if telemetry.get("run_id") != context.run_id:
        raise EvaluationError(
            "inference telemetry run_id does not match this evaluation run",
            hint="Use telemetry produced by the same run_generation invocation.",
        )
    if telemetry.get("model_kind") != model_kind:
        raise EvaluationError(
            "inference telemetry model_kind does not match the generations",
            hint="Use telemetry produced by the matching base or adapter evaluation.",
        )
    if telemetry.get("decoding") != decoding:
        raise EvaluationError(
            "inference telemetry decoding does not match the generations",
            hint="Re-run generation with one deterministic decoding configuration.",
        )

    measurement = telemetry.get("measurement")
    expected_measurement = {
        "scope": GENERATION_TIMING_SCOPE,
        "aggregation": GENERATION_TIMING_AGGREGATION,
        "clock": "monotonic_seconds",
        "model_load_included": False,
        "parsing_and_artifact_io_included": False,
    }
    if measurement != expected_measurement:
        raise EvaluationError(
            "inference telemetry uses an unsupported measurement boundary",
            hint="Re-run generation with the current sequential telemetry contract.",
        )
    timed_call_contract = telemetry.get("timed_call_contract")
    if timed_call_contract != inference_timed_call_contract():
        raise EvaluationError(
            "inference telemetry uses an unsupported timed-call contract",
            hint="Re-run generation to rebuild complete v2 inference telemetry.",
        )
    warmup = telemetry.get("warmup")
    if warmup != inference_warmup_contract():
        raise EvaluationError(
            "inference telemetry uses an unsupported warmup contract",
            hint="Re-run generation to rebuild complete v2 inference telemetry.",
        )
    sequential_run = telemetry.get("sequential_run")
    expected_sequential_run = {
        "boundary": SEQUENTIAL_RUN_BOUNDARY,
        "concurrency": 1,
        "single_model_instance": True,
        "slice_order": list(config.eval.slices),
        "example_order": "formatted_test_order_within_slice",
    }
    if sequential_run != expected_sequential_run:
        raise EvaluationError(
            "inference telemetry does not describe this sequential evaluation run",
            hint="Re-run generation so slice order and concurrency are recorded exactly.",
        )

    hardware = telemetry.get("hardware")
    expected_gpu_count = gpu_count_from_label(config.remote.gpu)
    if not isinstance(hardware, dict) or hardware != {
        "gpu_label": config.remote.gpu,
        "gpu_count": expected_gpu_count,
        "source": "config.remote.gpu",
    }:
        raise EvaluationError(
            "inference telemetry GPU allocation does not match remote.gpu",
            hint="Do not combine timing and GPU labels from different runs.",
        )

    telemetry_slices = telemetry.get("slices")
    if not isinstance(telemetry_slices, dict) or set(telemetry_slices) != set(slices):
        raise EvaluationError(
            "inference telemetry slice set does not match evaluated slices",
            hint="Re-run generation for every configured evaluation slice.",
        )

    efficiency_slices: dict[str, Any] = {}
    total_examples = 0
    total_elapsed_seconds = 0.0
    for slice_language in config.eval.slices:
        raw_slice = telemetry_slices.get(slice_language)
        if not isinstance(raw_slice, dict):
            raise EvaluationError(
                f"inference telemetry slice {slice_language} must be an object",
                hint="Re-run generation to rebuild inference telemetry.",
            )
        examples = raw_slice.get("examples")
        if isinstance(examples, bool) or not isinstance(examples, int) or examples <= 0:
            raise EvaluationError(
                f"inference telemetry slice {slice_language} has invalid examples",
                hint="Re-run generation to rebuild inference telemetry.",
            )
        if examples != slices[slice_language]["examples"]:
            raise EvaluationError(
                f"inference telemetry example count for {slice_language} does not "
                "match generations",
                hint="Do not combine telemetry and generations from different runs.",
            )
        if raw_slice.get("generation_artifact") != generation_refs[slice_language]:
            raise EvaluationError(
                f"inference telemetry generation artifact for {slice_language} does not match",
                hint="Do not alter generations after inference telemetry is recorded.",
            )
        elapsed_seconds = _number(
            raw_slice,
            "elapsed_seconds",
            context=f"inference telemetry slice {slice_language}",
        )
        seconds_per_example = _number(
            raw_slice,
            "seconds_per_example",
            context=f"inference telemetry slice {slice_language}",
        )
        if not math.isclose(
            seconds_per_example,
            elapsed_seconds / examples,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise EvaluationError(
                f"inference telemetry seconds_per_example for {slice_language} is inconsistent",
                hint="Re-run generation to rebuild inference telemetry.",
            )
        successes = slices[slice_language]["metrics"]["full_call_exact_match"]["numerator"]
        total_examples += examples
        total_elapsed_seconds += elapsed_seconds
        efficiency_slices[slice_language] = {
            **raw_slice,
            "gpu_seconds_per_full_call_exact_success": (
                _gpu_seconds_per_full_call_exact_success(
                    elapsed_seconds=elapsed_seconds,
                    gpu_count=expected_gpu_count,
                    successes=successes,
                )
            ),
        }

    raw_total = telemetry.get("total")
    if not isinstance(raw_total, dict) or raw_total.get("examples") != total_examples:
        raise EvaluationError(
            "inference telemetry total example count is inconsistent",
            hint="Re-run generation to rebuild inference telemetry.",
        )
    recorded_total_elapsed = _number(
        raw_total, "elapsed_seconds", context="inference telemetry total"
    )
    recorded_total_per_example = _number(
        raw_total, "seconds_per_example", context="inference telemetry total"
    )
    if not math.isclose(
        recorded_total_elapsed,
        total_elapsed_seconds,
        rel_tol=0.0,
        abs_tol=(len(slices) + 1) * 1e-6,
    ) or not math.isclose(
        recorded_total_per_example,
        recorded_total_elapsed / total_examples,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise EvaluationError(
            "inference telemetry total timing is inconsistent with its slices",
            hint="Re-run generation to rebuild inference telemetry.",
        )

    successes = overall_metrics["full_call_exact_match"]["numerator"]
    telemetry_ref = make_artifact_ref(
        telemetry_path,
        artifact_root=context.artifact_root,
        kind="inference_telemetry",
        schema_version=INFERENCE_TELEMETRY_SCHEMA,
    )
    return (
        {
            "available": True,
            "telemetry_artifact": telemetry_ref,
            "measurement": measurement,
            "timed_call_contract": timed_call_contract,
            "warmup": warmup,
            "sequential_run": sequential_run,
            "hardware": hardware,
            "slices": efficiency_slices,
            "overall": {
                **raw_total,
                "gpu_seconds_per_full_call_exact_success": (
                    _gpu_seconds_per_full_call_exact_success(
                        elapsed_seconds=recorded_total_elapsed,
                        gpu_count=expected_gpu_count,
                        successes=successes,
                    )
                ),
            },
        },
        telemetry_ref,
    )


def write_evaluation_report(
    config: SommelierConfig,
    *,
    formatted_dir: Path,
    eval_dir: Path,
    model_kind: ModelKind,
    context: RunContext,
    command: list[str],
    adapter: AdapterRef | None = None,
) -> ArtifactRef:
    """Writes evaluation_report.json next to the generations it scores.

    The report carries one metrics section per evaluated slice plus the
    overall block across all slices, and the identity digests the
    comparison gate checks: config digest, test split digest, per-slice
    ordered prompt set digests, parser version, and decoding config.
    Configured report fields are redacted before writing.
    """
    test_path = formatted_dir / "test.jsonl"
    by_slice = read_test_slices(config, formatted_dir)

    slices: dict[str, Any] = {}
    scored_by_slice: dict[str, list[ScoredRecord]] = {}
    all_scored: list[Any] = []
    all_decodings: list[dict[str, object]] = []
    generation_refs: list[ArtifactRef] = []
    generation_refs_by_slice: dict[str, ArtifactRef] = {}
    for slice_language in config.eval.slices:
        generations_path = eval_dir / slice_filename(slice_language)
        generations = read_jsonl_records(generations_path)
        for generation in generations:
            if generation.get("schema_version") != GENERATION_SCHEMA:
                raise EvaluationError(
                    f"{generations_path}: expected {GENERATION_SCHEMA} records",
                    hint="Re-run sommelier eval run to regenerate outputs.",
                )
            if generation.get("model_kind") != model_kind:
                raise EvaluationError(
                    f"{generations_path}: generations belong to "
                    f"{generation.get('model_kind')}, not {model_kind}",
                    hint="Point the report writer at the matching eval directory.",
                )
            if generation.get("language") != slice_language:
                raise EvaluationError(
                    f"{generations_path}: generation for example "
                    f"{generation.get('example_id')} carries language "
                    f"{generation.get('language')!r}, not {slice_language!r}",
                    hint="Re-run sommelier eval run to regenerate outputs.",
                )
        examples = by_slice[slice_language]
        scored = build_scored_records(examples, generations)
        scored_by_slice[slice_language] = scored
        all_scored.extend(scored)
        all_decodings.append(_uniform_decoding(generations))
        generations_ref = make_artifact_ref(
            generations_path,
            artifact_root=context.artifact_root,
            kind="generations",
            schema_version=GENERATION_SCHEMA,
        )
        generation_refs.append(generations_ref)
        generation_refs_by_slice[slice_language] = generations_ref
        slices[slice_language] = {
            "metrics": compute_metrics(scored),
            "examples": len(examples),
            "prompt_set_sha256": prompt_set_digest(
                [str(example["prompt_sha256"]) for example in examples]
            ),
            "generation_artifact": generations_ref["path"],
        }

    if len({json.dumps(decoding, sort_keys=True) for decoding in all_decodings}) != 1:
        raise InvariantViolation(
            "slices were generated with different decoding configs",
            hint="Regenerate every slice with one deterministic decoding config.",
        )

    overall_metrics = compute_metrics(all_scored)
    inference_efficiency, telemetry_ref = _inference_efficiency_section(
        config,
        eval_dir=eval_dir,
        model_kind=model_kind,
        context=context,
        decoding=all_decodings[0],
        slices=slices,
        overall_metrics=overall_metrics,
        generation_refs=generation_refs_by_slice,
    )
    report: dict[str, Any] = {
        "schema_version": EVALUATION_REPORT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": context.run_id,
        "model_kind": model_kind,
        "model_identity": {
            "base_model_id": config.model.base_model_id,
            "base_model_revision": config.model.base_model_revision,
            "tokenizer_id": config.model.base_model_id,
            "tokenizer_revision": config.model.tokenizer_revision,
        },
        "config_sha256": context.config_sha256,
        "split": "test",
        "slices": slices,
        "paired_slices": _paired_slice_reports(config, by_slice, scored_by_slice),
        "metrics": overall_metrics,
        "inference_efficiency": inference_efficiency,
        "adapter_source": adapter.describe() if adapter is not None else None,
        "parser_version": config.eval.parser_version,
        "test_split_sha256": sha256_file(test_path),
        "decoding": all_decodings[0],
    }
    report = redact_configured_fields(report, config.report.redact_fields)
    validate_no_secrets(report, context="evaluation report")

    report_path = eval_dir / REPORT_FILENAME

    def writer(temp_path: Path) -> None:
        temp_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    report_ref = write_artifact_atomic(
        report_path,
        writer,
        artifact_root=context.artifact_root,
        kind="evaluation_report",
        schema_version=EVALUATION_REPORT_SCHEMA,
    )

    test_ref = make_artifact_ref(
        test_path,
        artifact_root=context.artifact_root,
        kind="formatted_split",
        schema_version=FORMATTED_SCHEMA,
    )
    details: dict[str, Any] = {"eval_slices": list(config.eval.slices)}
    if adapter is not None:
        details["adapter_source"] = adapter.describe()
    telemetry_refs = [telemetry_ref] if telemetry_ref is not None else []
    record_stage_success(
        context,
        stage=evaluation_stage(model_kind),
        command=command,
        seed=config.project.seed,
        inputs=[test_ref, *generation_refs, *telemetry_refs],
        outputs=[*generation_refs, *telemetry_refs, report_ref],
        details=details,
    )
    track_stage_metrics(
        config,
        context,
        stage=f"eval-{model_kind}",
        records=[{name: value["value"] for name, value in report["metrics"].items()}],
    )
    return report_ref


def find_run_layout(path: Path) -> tuple[Path, Path, str]:
    """Locates (artifact_root, run_dir, run_id) for a path inside a run.

    Relies on the required artifact layout ``<root>/runs/<run_id>/...``.
    """
    resolved = path.resolve()
    parts = resolved.parts
    for index in range(len(parts) - 2, 0, -1):
        if parts[index] == "runs":
            artifact_root = Path(*parts[:index])
            run_id = parts[index + 1]
            return artifact_root, artifact_root / "runs" / run_id, run_id
    raise UserInputError(
        f"path is not inside a run directory: {path}",
        hint="Pass directories under <artifact_root>/runs/<run_id>/.",
    )


def _assert_comparable(base: dict[str, Any], adapter: dict[str, Any]) -> None:
    if base.get("model_kind") != "base" or adapter.get("model_kind") != "adapter":
        raise EvaluationError(
            "comparison requires a base report and an adapter report",
            hint="Pass --base and --adapter eval directories in that order.",
        )
    for field in _COMPARABILITY_FIELDS:
        if base.get(field) != adapter.get(field):
            raise EvaluationError(
                f"comparison rejected: mismatched {field}",
                hint="Base and adapter evaluations must share the same test "
                "split, prompts, parser, decoding, and config.",
            )
    if set(base["slices"].keys()) != set(adapter["slices"].keys()):
        raise EvaluationError(
            "comparison rejected: evaluated slices differ",
            hint="Base and adapter evaluations must cover the same language slices.",
        )
    for slice_language, base_slice in base["slices"].items():
        adapter_slice = adapter["slices"][slice_language]
        if base_slice.get("prompt_set_sha256") != adapter_slice.get("prompt_set_sha256"):
            raise EvaluationError(
                f"comparison rejected: mismatched prompt_set_sha256 for slice {slice_language}",
                hint="Base and adapter evaluations must score identical prompts in every slice.",
            )
    if set(base.get("paired_slices", {})) != set(adapter.get("paired_slices", {})):
        raise EvaluationError(
            "comparison rejected: paired slice sets differ",
            hint="Base and adapter reports must use the same matched language cohorts.",
        )
    for target_language, base_pair in base.get("paired_slices", {}).items():
        adapter_pair = adapter["paired_slices"][target_language]
        if base_pair.get("pair_set_sha256") != adapter_pair.get("pair_set_sha256"):
            raise EvaluationError(
                f"comparison rejected: mismatched pair_set_sha256 for slice {target_language}",
                hint="Base and adapter reports must use identical matched pairs.",
            )
    if set(base["metrics"].keys()) != set(adapter["metrics"].keys()):
        raise EvaluationError(
            "comparison rejected: metric names differ",
            hint="Regenerate both reports with the same pipeline version.",
        )


def _metric_deltas(base: dict[str, Any], adapter: dict[str, Any]) -> dict[str, float]:
    return {name: adapter[name]["value"] - base[name]["value"] for name in base}


def _language_gaps(
    report: dict[str, Any],
    *,
    reference: str,
) -> dict[str, dict[str, float]]:
    """Per-metric gap of each non-reference slice against the reference."""
    reference_metrics = report["slices"][reference]["metrics"]
    gaps: dict[str, dict[str, float]] = {}
    for slice_language, slice_report in report["slices"].items():
        if slice_language == reference:
            continue
        gaps[slice_language] = {
            name: slice_report["metrics"][name]["value"] - reference_metrics[name]["value"]
            for name in reference_metrics
        }
    return gaps


def _scored_slices_for_comparison(
    config: SommelierConfig,
    *,
    formatted_dir: Path,
    eval_dir: Path,
    model_kind: ModelKind,
) -> dict[str, list[ScoredRecord]]:
    examples_by_slice = read_test_slices(config, formatted_dir)
    scored: dict[str, list[ScoredRecord]] = {}
    for slice_language in config.eval.slices:
        generations_path = eval_dir / slice_filename(slice_language)
        generations = read_jsonl_records(generations_path)
        for generation in generations:
            if generation.get("schema_version") != GENERATION_SCHEMA:
                raise EvaluationError(
                    f"{generations_path}: expected {GENERATION_SCHEMA} records",
                    hint="Regenerate the evaluation before comparing it.",
                )
            if generation.get("model_kind") != model_kind:
                raise EvaluationError(
                    f"{generations_path}: expected {model_kind} generations",
                    hint="Point the comparison at the matching evaluation directory.",
                )
        scored[slice_language] = build_scored_records(
            examples_by_slice[slice_language], generations
        )
    return scored


def compare_evaluations(
    base_dir: Path,
    adapter_dir: Path,
    out_dir: Path,
    *,
    command: list[str] | None = None,
) -> ArtifactRef:
    """Writes comparison_report.json after enforcing the comparison gate.

    The gate rejects mismatched config, split, test split digest, prompt
    set digest, parser version, decoding config, or metric names
    (INV-DATA-006). Run identity, seed, and redaction settings come from
    the resolved config stored in the run directory that contains out_dir.
    """
    base_report = read_json_with_schema(
        base_dir / REPORT_FILENAME, expected_schema=EVALUATION_REPORT_SCHEMA
    )
    adapter_report = read_json_with_schema(
        adapter_dir / REPORT_FILENAME, expected_schema=EVALUATION_REPORT_SCHEMA
    )
    _assert_comparable(base_report, adapter_report)

    artifact_root, run_dir, run_id = find_run_layout(out_dir)
    resolved_config_path = run_dir / "config.resolved.yaml"
    if not resolved_config_path.exists():
        raise UserInputError(
            f"resolved config not found: {resolved_config_path}",
            hint="Write reports into the run directory that produced the evaluations.",
        )
    config = load_config(resolved_config_path)
    resolved_digest = compute_config_digest(resolved_config_path.read_text(encoding="utf-8"))
    if resolved_digest != base_report["config_sha256"]:
        raise EvaluationError(
            "comparison rejected: reports do not belong to this run's config",
            hint="Use the run directory whose config produced both evaluations.",
        )

    slice_languages = list(config.eval.slices)
    formatted_dir = run_dir / "formatted"
    base_scored = _scored_slices_for_comparison(
        config,
        formatted_dir=formatted_dir,
        eval_dir=base_dir,
        model_kind="base",
    )
    adapter_scored = _scored_slices_for_comparison(
        config,
        formatted_dir=formatted_dir,
        eval_dir=adapter_dir,
        model_kind="adapter",
    )
    slices: dict[str, Any] = {}
    for slice_language in slice_languages:
        base_slice = base_report["slices"][slice_language]
        adapter_slice = adapter_report["slices"][slice_language]
        slices[slice_language] = {
            "examples": base_slice["examples"],
            "prompt_set_sha256": base_slice["prompt_set_sha256"],
            "base": {"metrics": base_slice["metrics"]},
            "adapter": {"metrics": adapter_slice["metrics"]},
            "deltas": _metric_deltas(base_slice["metrics"], adapter_slice["metrics"]),
            "adapter_gain_ci95": paired_bootstrap_intervals(
                base_scored[slice_language],
                adapter_scored[slice_language],
                seed=stable_bootstrap_seed(
                    config.project.seed,
                    f"adapter-gain:{slice_language}",
                ),
            ),
            "generation_artifacts": {
                "base": base_slice["generation_artifact"],
                "adapter": adapter_slice["generation_artifact"],
            },
        }

    # The first configured slice is the reference the others are measured
    # against; with one slice the gaps section is empty.
    reference = config.root_dataset.language
    language_gaps: dict[str, Any] = {
        "reference": reference,
        "cohort": "marginal_full_slices",
        "base": _language_gaps(base_report, reference=reference),
        "adapter": _language_gaps(adapter_report, reference=reference),
    }
    paired_language_gaps = {
        target_language: {
            "pair_set_sha256": base_pair["pair_set_sha256"],
            "pairs": base_pair["pairs"],
            "coverage": base_pair["coverage"],
            "base": base_pair,
            "adapter": adapter_report["paired_slices"][target_language],
        }
        for target_language, base_pair in base_report.get("paired_slices", {}).items()
    }
    base_all = [record for language in slice_languages for record in base_scored[language]]
    adapter_all = [record for language in slice_languages for record in adapter_scored[language]]

    comparison: dict[str, Any] = {
        "schema_version": COMPARISON_REPORT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "shared": {field: base_report[field] for field in _COMPARABILITY_FIELDS},
        "slices": slices,
        "language_gaps": language_gaps,
        "paired_language_gaps": paired_language_gaps,
        "base": {
            "run_id": base_report["run_id"],
            "metrics": base_report["metrics"],
            "adapter_source": base_report.get("adapter_source"),
        },
        "adapter": {
            "run_id": adapter_report["run_id"],
            "metrics": adapter_report["metrics"],
            "adapter_source": adapter_report.get("adapter_source"),
        },
        "deltas": _metric_deltas(base_report["metrics"], adapter_report["metrics"]),
        "adapter_gain_ci95": paired_bootstrap_intervals(
            base_all,
            adapter_all,
            seed=stable_bootstrap_seed(config.project.seed, "adapter-gain:overall"),
        ),
        "runtime": runtime_section(run_dir),
    }
    comparison = redact_configured_fields(comparison, config.report.redact_fields)
    validate_no_secrets(comparison, context="comparison report")

    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / COMPARISON_FILENAME

    def writer(temp_path: Path) -> None:
        temp_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")

    comparison_ref = write_artifact_atomic(
        comparison_path,
        writer,
        artifact_root=artifact_root,
        kind="comparison_report",
        schema_version=COMPARISON_REPORT_SCHEMA,
    )

    from sommelier.evaluation.render import write_comparison_markdown

    markdown_ref = write_comparison_markdown(
        comparison_path,
        artifact_root=artifact_root,
    )

    base_ref = make_artifact_ref(
        base_dir / REPORT_FILENAME,
        artifact_root=artifact_root,
        kind="evaluation_report",
        schema_version=EVALUATION_REPORT_SCHEMA,
    )
    adapter_ref = make_artifact_ref(
        adapter_dir / REPORT_FILENAME,
        artifact_root=artifact_root,
        kind="evaluation_report",
        schema_version=EVALUATION_REPORT_SCHEMA,
    )
    manifest = build_stage_manifest(
        stage="report",
        run_id=run_id,
        config_sha256=resolved_digest,
        command=command or ["sommelier", "report", "compare"],
        seed=config.project.seed,
        inputs=[base_ref, adapter_ref],
        outputs=[comparison_ref, markdown_ref],
        status="succeeded",
    )
    stage_ref = write_stage_manifest(manifest, run_dir=run_dir, artifact_root=artifact_root)
    update_run_manifest(
        run_dir=run_dir,
        artifact_root=artifact_root,
        stage="report",
        stage_manifest_ref=stage_ref,
        status="running",
    )
    return comparison_ref
