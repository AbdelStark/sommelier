from __future__ import annotations

import hashlib
import json
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
    AdapterRef,
    ModelKind,
    read_test_slices,
    slice_filename,
)
from sommelier.evaluation.metrics import ScoredRecord, compute_metrics
from sommelier.evaluation.parse import ParseStatus
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

EVALUATION_REPORT_SCHEMA: Final = "sommelier.evaluation_report.v2"
COMPARISON_REPORT_SCHEMA: Final = "sommelier.comparison_report.v2"

REPORT_FILENAME: Final = "evaluation_report.json"
COMPARISON_FILENAME: Final = "comparison_report.json"

_COMPARABILITY_FIELDS: Final = (
    "config_sha256",
    "split",
    "test_split_sha256",
    "parser_version",
    "decoding",
)


def prompt_set_digest(prompt_digests: list[str]) -> str:
    """Digest over the ordered per-example prompt digests of a test split."""
    return hashlib.sha256("\n".join(prompt_digests).encode("utf-8")).hexdigest()


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
    all_scored: list[Any] = []
    all_decodings: list[dict[str, object]] = []
    generation_refs: list[ArtifactRef] = []
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
        all_scored.extend(scored)
        all_decodings.append(_uniform_decoding(generations))
        generations_ref = make_artifact_ref(
            generations_path,
            artifact_root=context.artifact_root,
            kind="generations",
            schema_version=GENERATION_SCHEMA,
        )
        generation_refs.append(generations_ref)
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

    report: dict[str, Any] = {
        "schema_version": EVALUATION_REPORT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": context.run_id,
        "model_kind": model_kind,
        "config_sha256": context.config_sha256,
        "split": "test",
        "slices": slices,
        "metrics": compute_metrics(all_scored),
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
    record_stage_success(
        context,
        stage="eval",
        command=command,
        seed=config.project.seed,
        inputs=[test_ref, *generation_refs],
        outputs=[*generation_refs, report_ref],
        details=details,
    )
    track_stage_metrics(
        config,
        context,
        stage=f"eval-{model_kind}",
        records=[
            {name: value["value"] for name, value in report["metrics"].items()}
        ],
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
            hint="Base and adapter evaluations must cover the same language "
            "slices.",
        )
    for slice_language, base_slice in base["slices"].items():
        adapter_slice = adapter["slices"][slice_language]
        if base_slice.get("prompt_set_sha256") != adapter_slice.get("prompt_set_sha256"):
            raise EvaluationError(
                f"comparison rejected: mismatched prompt_set_sha256 for slice "
                f"{slice_language}",
                hint="Base and adapter evaluations must score identical "
                "prompts in every slice.",
            )
    if set(base["metrics"].keys()) != set(adapter["metrics"].keys()):
        raise EvaluationError(
            "comparison rejected: metric names differ",
            hint="Regenerate both reports with the same pipeline version.",
        )


def _metric_deltas(base: dict[str, Any], adapter: dict[str, Any]) -> dict[str, float]:
    return {
        name: adapter[name]["value"] - base[name]["value"] for name in base
    }


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
    resolved_digest = compute_config_digest(
        resolved_config_path.read_text(encoding="utf-8")
    )
    if resolved_digest != base_report["config_sha256"]:
        raise EvaluationError(
            "comparison rejected: reports do not belong to this run's config",
            hint="Use the run directory whose config produced both evaluations.",
        )

    slice_languages = list(config.eval.slices)
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
            "generation_artifacts": {
                "base": base_slice["generation_artifact"],
                "adapter": adapter_slice["generation_artifact"],
            },
        }

    # The first configured slice is the reference the others are measured
    # against; with one slice the gaps section is empty.
    reference = slice_languages[0]
    language_gaps: dict[str, Any] = {
        "reference": reference,
        "base": _language_gaps(base_report, reference=reference),
        "adapter": _language_gaps(adapter_report, reference=reference),
    }

    comparison: dict[str, Any] = {
        "schema_version": COMPARISON_REPORT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "shared": {field: base_report[field] for field in _COMPARABILITY_FIELDS},
        "slices": slices,
        "language_gaps": language_gaps,
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
