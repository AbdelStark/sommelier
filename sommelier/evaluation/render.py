from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from sommelier.artifacts import ArtifactRef, write_artifact_atomic

MARKDOWN_FILENAME: Final = "comparison_report.md"

LIMITATIONS: Final = (
    "- Metrics measure schema-valid single tool calls on the configured "
    "held-out test split only; multi-call plans are out of scope for v1.0.\n"
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


def _metric_row(name: str, comparison: dict[str, Any]) -> str:
    base = comparison["base"]["metrics"][name]
    adapter = comparison["adapter"]["metrics"][name]
    delta = comparison["deltas"][name]
    return (
        f"| {name} | {base['value']:.4f} ({base['numerator']}/{base['denominator']}) "
        f"| {adapter['value']:.4f} ({adapter['numerator']}/{adapter['denominator']}) "
        f"| {delta:+.4f} |"
    )


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
        lines.append(
            f"- {stage_name}: {stages[stage_name]['elapsed_seconds']} s elapsed"
        )
    return lines


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Renders the human-readable report from the authoritative JSON.

    The Markdown is a rendering only; automation must consume
    comparison_report.json (RFC-0009).
    """
    shared = comparison["shared"]
    run_id = str(comparison["run_id"])
    metric_names = list(comparison["deltas"].keys())
    denominator = comparison["base"]["metrics"][metric_names[0]]["denominator"]

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
        "",
        "## Split Summary",
        "",
        f"- Split: {shared['split']}",
        f"- Examples evaluated: {denominator}",
        f"- Test split digest: `{shared['test_split_sha256']}`",
        f"- Prompt set digest: `{shared['prompt_set_sha256']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Base | Adapter | Delta |",
        "|--------|------|---------|-------|",
    ]
    lines.extend(_metric_row(name, comparison) for name in metric_names)
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
            "sommelier report compare --base eval/base --adapter eval/adapter "
            "--out report",
            "```",
            "",
            "Generation artifacts: "
            f"`{comparison['base']['generation_artifact']}` (base), "
            f"`{comparison['adapter']['generation_artifact']}` (adapter).",
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
