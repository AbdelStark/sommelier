# 03 Data Model

- Status: Draft
- Target milestone: v1.0
- Primary RFCs: [RFC-0001](../rfcs/RFC-0001-project-configuration-and-run-manifest.md), [RFC-0002](../rfcs/RFC-0002-dataset-preparation-and-split-discipline.md), [RFC-0005](../rfcs/RFC-0005-evaluation-parser-and-metrics.md), [RFC-0006](../rfcs/RFC-0006-artifact-store-and-schema-versioning.md)

## Schema Versioning

All persisted JSON and JSONL records include a `schema_version` field. v1.0 accepts only these schema versions:

- `sommelier.config.v1`
- `sommelier.raw_tool_call_row.v1`
- `sommelier.prepared_example.v1`
- `sommelier.formatted_example.v1`
- `sommelier.generation.v1`
- `sommelier.evaluation_report.v1`
- `sommelier.comparison_report.v1`
- `sommelier.manifest.v1`
- `sommelier.log_event.v1`
- `sommelier.drop_summary.v1`
- `sommelier.training_metric.v1`

Readers fail closed on unknown schema versions.

## Raw Row

```python
class RawToolCallRow(TypedDict):
    schema_version: Literal["sommelier.raw_tool_call_row.v1"]
    source_id: str
    query: str
    tools: str
    answers: str
    source_revision: str
```

`tools` and `answers` are raw JSON strings from the source dataset until validation. They are not trusted as parsed objects.

## Prepared Example

```python
JsonObject = dict[str, Any]

class ToolSchema(TypedDict):
    name: str
    description: str
    parameters: JsonObject

class ToolCall(TypedDict):
    name: str
    arguments: JsonObject

class PreparedExample(TypedDict):
    schema_version: Literal["sommelier.prepared_example.v1"]
    example_id: str
    source_id: str
    query: str
    tools: list[ToolSchema]
    gold_calls: list[ToolCall]
    split: Literal["train", "validation", "test"]
    query_sha256: str
    source_revision: str
```

## Formatted Example

```python
class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class FormattedExample(TypedDict):
    schema_version: Literal["sommelier.formatted_example.v1"]
    example_id: str
    split: Literal["train", "validation", "test"]
    messages: list[ChatMessage]
    prompt_text: str
    target_text: str
    full_text: str
    prompt_sha256: str
    tokenizer_id: str
    tokenizer_revision: str
    template_policy: str
```

`prompt_text` excludes the assistant target. `full_text` includes the assistant target and is used for supervised training.

## Generation Record

```python
class GenerationRecord(TypedDict):
    schema_version: Literal["sommelier.generation.v1"]
    example_id: str
    model_kind: Literal["base", "adapter"]
    prompt_sha256: str
    raw_text: str
    parsed_call: ToolCall | None
    parse_status: Literal["ok", "no_json", "invalid_json", "invalid_shape"]
    decoding: dict[str, Any]
```

Raw generations must be retained for auditability even when parsing fails.

## Evaluation Report

```python
class MetricValue(TypedDict):
    value: float
    numerator: int
    denominator: int

class EvaluationReport(TypedDict):
    schema_version: Literal["sommelier.evaluation_report.v1"]
    created_at: str
    run_id: str
    model_kind: Literal["base", "adapter"]
    config_sha256: str
    split: Literal["test"]
    metrics: dict[
        Literal[
            "valid_json_rate",
            "function_name_accuracy",
            "argument_exact_match",
            "argument_f1",
            "full_call_exact_match",
        ],
        MetricValue,
    ]
    generation_artifact: str
    parser_version: str
    test_split_sha256: str
    prompt_set_sha256: str
    decoding: dict[str, Any]
```

`test_split_sha256` digests the formatted test split file, and
`prompt_set_sha256` digests the ordered per-example prompt digests; together
with `parser_version` and `decoding` they are the identity the comparison
gate checks (INV-DATA-006). The comparison report
(`sommelier.comparison_report.v1`) embeds both metric sets, the shared
identity fields, and per-metric deltas; it is written only when every
identity field matches.

## Artifact Reference

```python
class ArtifactRef(TypedDict):
    path: str
    kind: str
    schema_version: str
    sha256: str
    bytes: int
```

Paths are relative to the artifact root. Absolute paths are allowed only in process-local logs and must not appear in manifests.

## Required Artifact Layout

```text
artifacts/
  runs/<run_id>/
    config.resolved.yaml
    manifest.json
    data/
      train.jsonl
      validation.jsonl
      test.jsonl
      data_manifest.json
    formatted/
      train.jsonl
      validation.jsonl
      test.jsonl
      format_manifest.json
    train/
      adapter/
      train_manifest.json
      training_metrics.jsonl
    eval/
      base/
        generations.jsonl
        evaluation_report.json
      adapter/
        generations.jsonl
        evaluation_report.json
    report/
      comparison_report.md
      comparison_report.json
```

## Invariants

- `INV-DATA-001`: `example_id` is stable for the same normalized query, tools, answers, and source revision.
- `INV-DATA-002`: A `query_sha256` may appear in only one split.
- `INV-DATA-003`: Validation and test records are never used for gradient updates.
- `INV-DATA-004`: `gold_calls` contains at least one call in v1.0.
- `INV-DATA-005`: Metrics count parse failures as failures, not missing data.
- `INV-DATA-006`: A comparison report is invalid unless base and adapter reports share the same test split digest.
