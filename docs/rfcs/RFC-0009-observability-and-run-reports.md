# RFC-0009: Observability and Run Reports

- Status: Accepted
- Authors: maintainers
- Created: 2026-07-01
- Target milestone: v1.0

## Summary

Sommelier writes structured logs, stage metrics, raw generation records, and a final comparison report for every reference run. The local artifacts are complete even when optional external experiment tracking is disabled.

## Motivation

The PRD requires a clear improvement report and reproducibility. The observability spec requires enough data to debug validation drops, parse failures, runtime, memory, and cost. Relying only on console output or hosted dashboards would make results hard to audit.

## Goals

- Emit structured JSONL logs.
- Persist training and evaluation metrics locally.
- Write both Markdown and JSON comparison reports.
- Include runtime, hardware, dependency, and cost metadata when available.
- Redact secrets before writing user-facing artifacts.

## Non-Goals

- Require a hosted observability backend.
- Implement distributed tracing.
- Store private production telemetry.

## Proposed Design

### Log API

```python
def log_event(
    level: Literal["debug", "info", "warning", "error"],
    stage: str,
    event: str,
    message: str,
    **fields: str | int | float | bool | None,
) -> None: ...
```

Logs are written as JSON lines and mirrored to concise console messages.

### Report API

```python
def write_comparison_report(
    base: EvaluationReport,
    adapter: EvaluationReport,
    out_dir: Path,
    context: ReportContext,
) -> StageManifest: ...
```

`comparison_report.json` is the authoritative machine-readable output. `comparison_report.md` is the human rendering.

### Required Report Sections

- Run identity and config digest.
- Dataset and split summary.
- Prompt, parser, decoding, and metric versions.
- Base metrics.
- Adapter metrics.
- Metric deltas.
- Runtime, hardware, and observed cost.
- Reproduction commands.
- Limitations and known failure modes.

### External Tracking

When enabled, training and evaluation may log to an external tracking service. The local report must include enough information to reproduce the result without accessing that service.

## Alternatives Considered

- Console-only progress reporting. Rejected because it is not durable.
- Hosted tracking as the source of truth. Rejected because it creates availability and access problems.
- Markdown-only reports. Rejected because automation needs JSON.

## Drawbacks

- More local artifacts to manage.
- Raw generation retention can expose adapted private schemas unless configured off.
- Cost metadata can be incomplete if the remote provider does not expose exact billing.

## Migration / Rollout

1. Add structured logger and redaction scanner.
2. Add training metrics JSONL.
3. Add generation records.
4. Add report writer.
5. Add optional external tracking integration.

## Testing Strategy

- Unit-test log redaction.
- Unit-test report rendering from fixture evaluation reports.
- Snapshot-test Markdown and JSON reports.
- Integration-test report rejection for mismatched parser or split digest.
- Verify external tracking disabled path still writes complete artifacts.

## Open Questions

None for v1.0.

## References

- [05-observability](../spec/05-observability.md)
- [06-security](../spec/06-security.md)
