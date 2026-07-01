# RFC-0002: Dataset Preparation and Split Discipline

- Status: Accepted
- Authors: maintainers
- Created: 2026-07-01
- Target milestone: v0.2

## Summary

Sommelier prepares the selected tool-calling dataset by validating required columns, parsing tool and answer JSON, filtering malformed rows, exact-deduplicating normalized requests before splitting, and writing deterministic train, validation, and test JSONL artifacts. The split manifest is the gate that prevents leakage into evaluation.

## Motivation

The PRD identifies train-test leakage as a primary risk and requires RAPIDS data preparation. The data model spec requires `query_sha256` to appear in only one split. A weak split discipline would make the base-versus-adapter comparison unreliable.

## Goals

- Validate raw rows before they become training examples.
- Deduplicate on normalized query before split assignment.
- Keep split assignment deterministic for a fixed dataset revision and seed.
- Record drop counts by reason.
- Produce JSONL files that downstream stages can consume without reparsing untrusted raw strings.

## Non-Goals

- Semantic deduplication.
- Multi-dataset balancing.
- Automatic dataset license inference.
- Multi-call or multi-turn examples in v1.0.

## Proposed Design

### Normalization

```python
def normalize_query(query: str) -> str:
    return " ".join(query.casefold().strip().split())
```

The dedupe key is `sha256(normalize_query(query))`. Exact normalized deduplication is intentionally conservative and inspectable.

### Validation

```python
def parse_tools(raw: str) -> list[ToolSchema]: ...
def parse_gold_calls(raw: str) -> list[ToolCall]: ...
def validate_raw_row(row: RawToolCallRow) -> PreparedExample | DropReason: ...
```

Drop reasons:

```python
DropReason = Literal[
    "missing_query",
    "missing_tools",
    "missing_answers",
    "query_too_short",
    "query_too_long",
    "invalid_tools_json",
    "invalid_answers_json",
    "invalid_tool_shape",
    "invalid_answer_shape",
    "duplicate_query",
]
```

Rows with parseable JSON but unsupported shape are dropped with explicit counts.

### GPU and CPU Boundary

GPU dataframe operations handle null filtering, length filtering, exact duplicate detection, and deterministic shuffling where supported. JSON parsing and shape validation run in Python after coarse filtering because nested JSON validation is easier to audit in typed Python code.

### Split Assignment

```python
def split_examples(
    examples: Sequence[PreparedExample],
    n_train: int,
    n_validation: int,
    n_test: int,
    seed: int,
) -> SplitResult: ...
```

The splitter shuffles deterministically and takes slices in train, validation, test order. It fails if fewer valid deduplicated examples exist than requested.

### Outputs

```text
data/train.jsonl
data/validation.jsonl
data/test.jsonl
data/data_manifest.json
data/drop_summary.json
```

Every JSONL row is a `PreparedExample`.

## Alternatives Considered

- Deduplicate after splitting. Rejected because it can leave related examples across train and test.
- Random percentage splits instead of fixed counts. Rejected because the PRD sets concrete default counts and fixed counts simplify cost estimates.
- Keep `tools` and `answers` as raw strings downstream. Rejected because every downstream stage would need to re-validate untrusted JSON.
- Add semantic deduplication in v1.0. Rejected because it would introduce embedding/model dependency and harder reproducibility.

## Drawbacks

- Exact query deduplication may miss paraphrase leakage.
- Python JSON validation may become the bottleneck after coarse GPU filtering.
- Fixed counts can fail if a future dataset revision has fewer valid examples.

## Migration / Rollout

1. Add fixture raw rows with valid and invalid cases.
2. Implement validators and split logic locally.
3. Add GPU dataframe implementation behind the same function contract.
4. Write data manifests and drop summaries.
5. Wire `sommelier data prepare`.

## Testing Strategy

- Unit-test every drop reason.
- Property-test that a normalized query appears in only one split.
- Snapshot-test `drop_summary.json`.
- Integration-test a tiny raw fixture through all three split outputs.
- Test that insufficient valid rows fails before writing partial outputs.

## Open Questions

None for v1.0.

## References

- [00-overview](../spec/00-overview.md)
- [03-data-model](../spec/03-data-model.md)
- [07-testing-strategy](../spec/07-testing-strategy.md)
