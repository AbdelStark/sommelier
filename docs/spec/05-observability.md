# 05 Observability

- Status: Draft
- Target milestone: v1.0
- Primary RFC: [RFC-0009](../rfcs/RFC-0009-observability-and-run-reports.md)

## Goals

Observability exists to make a run reproducible and debuggable. It must answer:

- Which config, commit, dataset revision, tokenizer revision, and dependencies produced this artifact?
- Which examples failed validation or parsing, and why?
- How much time, memory, and remote GPU cost did each stage consume?
- Did base and adapter evaluations use identical prompts, parser, metrics, and decoding?

## Logs

Sommelier emits structured JSON lines to `runs/<run_id>/logs/<stage>.jsonl`:

```python
class LogEvent(TypedDict):
    schema_version: Literal["sommelier.log_event.v1"]
    timestamp: str
    level: Literal["debug", "info", "warning", "error"]
    run_id: str
    stage: str
    event: str
    message: str
    fields: dict[str, str | int | float | bool | None]
```

Log field values must be JSON-native scalars. Non-finite numbers are
rejected at write time. Messages and string fields are redacted before they
reach disk (see Redaction below).

Human-readable console output is a rendering of the structured events, not the source of truth.

## Metrics

Required stage metrics:

| Stage | Metrics |
|-------|---------|
| data | raw rows, dropped rows by reason, deduplicated rows, split counts, elapsed seconds |
| format | formatted rows, skipped rows, prompt token percentiles, target token percentiles |
| train | loss, eval loss, learning rate, tokens processed, peak GPU memory, elapsed seconds |
| eval | examples, parse statuses, metric numerators and denominators, tokens generated, elapsed seconds |
| report | metric deltas, artifact checksums, report path |

## Tracing

v1.0 does not require a distributed tracing backend. It requires a `run_id`, `stage`, and `example_id` correlation key so logs, generation records, and reports can be joined offline.

## Redaction

Logs and manifests must redact:

- API tokens.
- Remote service credentials.
- User home directory paths.
- Environment variable values whose names contain `TOKEN`, `KEY`, `SECRET`, or `PASSWORD`.

The redaction scanner runs before writing reports and failed manifests. It
covers JSONL logs, JSON manifests and artifacts, and Markdown reports under
an artifact tree, and it reports the file, location, and finding kind for
each hit. Release preflight fails closed (exit code 5) when any finding is
present. Configured report fields (`report.redact_fields`) are replaced with
`[redacted]` wherever the field name appears in a JSON tree.

## Reports

The release report includes:

- Config digest.
- Git commit.
- Dependency lock digest.
- Dataset revision.
- Model and tokenizer revision.
- Split counts.
- Hardware type.
- Runtime and observed cost.
- Base and adapter metrics.
- Raw generation artifact paths.
- Limitations and known failure modes.

The Markdown report is for humans. The JSON report is authoritative for automation.

## External Experiment Tracking

External tracking is optional. When enabled, the run manifest records the tracking project and run URL. The local artifacts remain complete even if external tracking is unavailable.
