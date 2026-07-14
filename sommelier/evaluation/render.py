from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from sommelier.artifacts import ArtifactRef, write_artifact_atomic

MARKDOWN_FILENAME: Final = "comparison_report.md"

LIMITATIONS: Final = (
    "- Metrics measure schema-valid single tool calls on the configured "
    "held-out test split only; multi-call plans are out of scope.\n"
    "- Non-English slices are machine-translated variants of the English "
    "test rows, not natively authored requests, and share their gold "
    "answers by construction.\n"
    "- Argument comparisons are exact canonical-JSON matches; semantically "
    "equivalent but differently formatted values count as mismatches.\n"
    "- Results describe the recorded run (hardware, dependencies, dataset "
    "revision) and do not claim production readiness, broad reliability, or "
    "generalization beyond the evaluated split.\n"
    "- Parse failures count against every metric; raw generations are "
    "retained for audit.\n"
)


def _evidence_class(run_id: str) -> str:
    return "smoke" if run_id.startswith("smoke-") else "full"


def _format_interval(interval: dict[str, Any]) -> str:
    return f"[{interval['lower']:+.4f}, {interval['upper']:+.4f}]"


def _metric_row(
    name: str,
    block: dict[str, Any],
    intervals: dict[str, Any] | None,
) -> str:
    base = block["base"]["metrics"][name]
    adapter = block["adapter"]["metrics"][name]
    delta = block["deltas"][name]
    interval_cell = f" | {_format_interval(intervals[name])}" if intervals is not None else ""
    return (
        f"| {name} | {base['value']:.4f} ({base['numerator']}/{base['denominator']}) "
        f"| {adapter['value']:.4f} ({adapter['numerator']}/{adapter['denominator']}) "
        f"| {delta:+.4f}{interval_cell} |"
    )


def _metrics_table(block: dict[str, Any]) -> list[str]:
    bootstrap = block.get("adapter_gain_ci95")
    intervals = bootstrap.get("intervals") if bootstrap is not None else None
    lines = (
        [
            "| Metric | Base | Adapter | Adapter - base | 95% CI |",
            "|--------|------|---------|----------------|--------|",
        ]
        if intervals is not None
        else [
            "| Metric | Base | Adapter | Adapter - base |",
            "|--------|------|---------|----------------|",
        ]
    )
    lines.extend(_metric_row(name, block, intervals) for name in block["deltas"])
    if bootstrap is not None:
        lines.extend(
            [
                "",
                "Adapter-gain intervals use "
                f"`{bootstrap['method']}` ({bootstrap['resamples']} paired "
                f"resamples; seed {bootstrap['seed']}; "
                f"confidence {bootstrap['confidence_level']:.0%}).",
            ]
        )
    return lines


def _marginal_language_gap_lines(comparison: dict[str, Any]) -> list[str]:
    gaps = comparison.get("language_gaps", {})
    reference = gaps.get("reference")
    base_gaps = gaps.get("base", {})
    if not base_gaps:
        return []
    lines = [
        "",
        "### Marginal full-slice gaps (descriptive)",
        "",
        "These values compare every surviving row in each complete slice. "
        "The cohorts can differ, so they are descriptive and are not the "
        "primary paired estimate. "
        f"Cohort label: `{gaps.get('cohort', 'unspecified')}`.",
        "",
        f"Each slice against the `{reference}` reference slice (positive means "
        "the slice scores higher):",
        "",
    ]
    for slice_language in sorted(base_gaps):
        lines.extend(
            [
                f"#### `{slice_language}` minus `{reference}`",
                "",
                "| Metric | Base gap | Adapter gap |",
                "|--------|----------|-------------|",
            ]
        )
        adapter_gaps = comparison["language_gaps"]["adapter"][slice_language]
        for name, base_gap in base_gaps[slice_language].items():
            lines.append(f"| {name} | {base_gap:+.4f} | {adapter_gaps[name]:+.4f} |")
        lines.append("")
    return lines


def _paired_language_gap_lines(comparison: dict[str, Any]) -> list[str]:
    paired = comparison.get("paired_language_gaps", {})
    if not paired:
        return []
    lines = [
        "",
        "## Language Gaps",
        "",
        "### Exact matched-pair gaps (primary)",
        "",
        "Each translated row is compared only with the exact root row named by "
        "`source_example_id`. Intervals resample those matched pairs; positive "
        "gaps mean the translated slice scores higher.",
        "",
    ]
    for target_language in sorted(paired):
        block = paired[target_language]
        base = block["base"]
        adapter = block["adapter"]
        reference = base["reference_language"]
        coverage = block["coverage"]
        base_intervals = base["gap_ci95"]["intervals"]
        adapter_intervals = adapter["gap_ci95"]["intervals"]
        lines.extend(
            [
                f"#### `{target_language}` minus `{reference}`",
                "",
                f"- Matched pairs: {block['pairs']}",
                f"- Reference coverage: {coverage['paired']}/"
                f"{coverage['reference_slice_examples']} "
                f"({coverage['reference_fraction']:.1%})",
                f"- Pair-set digest: `{block['pair_set_sha256']}`",
                "",
                "| Metric | Base gap | Base 95% CI | Adapter gap | Adapter 95% CI |",
                "|--------|----------|-------------|-------------|----------------|",
            ]
        )
        for name, base_gap in base["gaps"].items():
            lines.append(
                f"| {name} | {base_gap:+.4f} | "
                f"{_format_interval(base_intervals[name])} | "
                f"{adapter['gaps'][name]:+.4f} | "
                f"{_format_interval(adapter_intervals[name])} |"
            )
        lines.extend(
            [
                "",
                "Gap intervals use "
                f"`{base['gap_ci95']['method']}` "
                f"({base['gap_ci95']['resamples']} paired resamples; "
                f"base seed {base['gap_ci95']['seed']}; "
                f"adapter seed {adapter['gap_ci95']['seed']}).",
                "",
            ]
        )
    return lines


def _runtime_lines(runtime: dict[str, Any]) -> list[str]:
    if not runtime.get("available"):
        return ["Runtime metadata is unavailable for this run."]
    lines = [
        f"- Hardware: {runtime['hardware'].get('gpu', 'unknown')} "
        f"(source: {runtime['hardware'].get('source', 'unknown')})",
    ]
    peak = runtime.get("peak_gpu_memory_mb")
    lines.append(
        f"- Peak GPU memory: {peak} MiB" if peak is not None else "- Peak GPU memory: unavailable"
    )
    cost = runtime.get("observed_cost_usd")
    if cost is None:
        lines.append(f"- Observed cost: unavailable (source: {runtime.get('cost_source')})")
    else:
        lines.append(f"- Observed cost: {cost} USD (source: {runtime.get('cost_source')})")
    stages = runtime.get("stages", {})
    for stage_name in sorted(stages):
        lines.append(f"- {stage_name}: {stages[stage_name]['elapsed_seconds']} s elapsed")
    return lines


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Renders the human-readable report from the authoritative JSON.

    The Markdown is a rendering only; automation must consume
    comparison_report.json.
    """
    shared = comparison["shared"]
    run_id = str(comparison["run_id"])
    denominator = sum(int(block["examples"]) for block in comparison["slices"].values())
    slices = comparison["slices"]
    adapter_source = comparison["adapter"].get("adapter_source")

    lines: list[str] = [
        "# Sommelier Comparison Report",
        "",
        "The JSON report (`comparison_report.json`) is authoritative for "
        "automation; this document is a human rendering.",
        "",
        "## Run Identity",
        "",
        f"- Run ID: `{run_id}`",
        f"- Evidence class: {_evidence_class(run_id)} run",
        f"- Created at: {comparison['created_at']}",
        f"- Config digest: `{shared['config_sha256']}`",
        f"- Parser version: `{shared['parser_version']}`",
        f"- Decoding: `{json.dumps(shared['decoding'], sort_keys=True)}`",
    ]
    if adapter_source is not None:
        lines.append(
            f"- Adapter source: `{adapter_source['source']}` "
            f"({adapter_source['kind']}, revision {adapter_source['revision']})"
        )
    lines.extend(
        [
            "",
            "## Split Summary",
            "",
            f"- Split: {shared['split']}",
            f"- Slices: {', '.join(f'`{name}`' for name in slices)}",
            f"- Examples evaluated: {denominator} across all slices",
            f"- Test split digest: `{shared['test_split_sha256']}`",
            "",
            "## Metrics, all slices",
            "",
            *_metrics_table(comparison),
        ]
    )
    for slice_language, block in slices.items():
        lines.extend(
            [
                "",
                f"## Metrics, slice `{slice_language}`",
                "",
                f"- Examples: {block['examples']}",
                f"- Prompt set digest: `{block['prompt_set_sha256']}`",
                "",
                *_metrics_table(block),
            ]
        )
    paired_gap_lines = _paired_language_gap_lines(comparison)
    lines.extend(paired_gap_lines)
    if not paired_gap_lines and comparison.get("language_gaps", {}).get("base"):
        lines.extend(["", "## Language Gaps"])
    lines.extend(_marginal_language_gap_lines(comparison))
    lines.extend(
        [
            "",
            "## Runtime and Cost",
            "",
            *_runtime_lines(comparison.get("runtime", {"available": False})),
            "",
            "## Reproduction",
            "",
            "Using the resolved config stored in this run directory:",
            "",
            "```bash",
            f"sommelier eval run --config config.resolved.yaml --model base "
            f"--data formatted --out eval/base --run-id {run_id}",
            f"sommelier train run --config config.resolved.yaml "
            f"--data formatted --out train/adapter --run-id {run_id}",
            f"sommelier eval run --config config.resolved.yaml --model adapter "
            f"--adapter train/adapter --data formatted --out eval/adapter "
            f"--run-id {run_id}",
            "sommelier report compare --base eval/base --adapter eval/adapter --out report",
            "```",
            "",
            "Generation artifacts per slice: "
            + "; ".join(
                f"`{slice_language}`: "
                f"`{block['generation_artifacts']['base']}` (base), "
                f"`{block['generation_artifacts']['adapter']}` (adapter)"
                for slice_language, block in slices.items()
            )
            + ".",
            "",
            "## Limitations",
            "",
            LIMITATIONS.rstrip(),
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison_markdown(
    comparison_json_path: Path,
    *,
    artifact_root: Path,
) -> ArtifactRef:
    """Writes comparison_report.md next to the authoritative JSON report."""
    comparison = json.loads(comparison_json_path.read_text(encoding="utf-8"))
    markdown = render_comparison_markdown(comparison)
    markdown_path = comparison_json_path.parent / MARKDOWN_FILENAME

    def writer(temp_path: Path) -> None:
        temp_path.write_text(markdown, encoding="utf-8")

    return write_artifact_atomic(
        markdown_path,
        writer,
        artifact_root=artifact_root,
        kind="comparison_report_markdown",
        schema_version="",
    )
