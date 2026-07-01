# RFC-0005: Evaluation Parser and Metrics

- Status: Accepted
- Authors: maintainers
- Created: 2026-07-01
- Target milestone: v0.3

## Summary

Sommelier evaluates the base model and adapter with deterministic decoding, the same formatted test prompts, a conservative JSON tool-call parser, and five metrics: valid-JSON rate, function-name accuracy, argument exact match, argument F1, and full-call exact match.

## Motivation

The PRD requires a clear base-versus-fine-tuned comparison on a held-out test set. The comparison is credible only if parsing failures are counted, raw generations are retained, and both model variants share prompts, decoding, parser, and metric definitions.

## Goals

- Use identical test prompts for base and adapter evaluation.
- Classify parse failures without aborting evaluation.
- Preserve raw generations for audit.
- Compute metric numerators and denominators, not only percentages.
- Fail comparison when base and adapter inputs differ.

## Non-Goals

- Judge semantic equivalence beyond JSON argument values.
- Support multi-turn tool plans.
- Hide invalid outputs with parser repair.
- Claim public benchmark performance.

## Proposed Design

### Evaluation Interface

```python
def evaluate_model(
    config: SommelierConfig,
    formatted_dir: Path,
    out_dir: Path,
    model_kind: Literal["base", "adapter"],
    adapter_dir: Path | None = None,
) -> StageManifest: ...
```

The evaluator reads `formatted/test.jsonl`, generates one output per prompt, writes `generations.jsonl`, and then writes `evaluation_report.json`.

### Decoding

```python
class DecodingConfig(TypedDict):
    temperature: Literal[0.0]
    do_sample: Literal[False]
    max_new_tokens: int
```

Any config that enables sampling is rejected for reference evaluation.

### Parser

```python
ParseStatus = Literal["ok", "no_json", "invalid_json", "invalid_shape"]

def parse_tool_call(text: str) -> tuple[ToolCall | None, ParseStatus]: ...
```

The parser extracts the first balanced JSON object or array. It then requires either a single object with `name` and `arguments`, or a one-element array containing such an object. It does not repair malformed JSON.

### Metrics

```python
def valid_json_rate(records: Sequence[ScoredRecord]) -> MetricValue: ...
def function_name_accuracy(records: Sequence[ScoredRecord]) -> MetricValue: ...
def argument_exact_match(records: Sequence[ScoredRecord]) -> MetricValue: ...
def argument_f1(records: Sequence[ScoredRecord]) -> MetricValue: ...
def full_call_exact_match(records: Sequence[ScoredRecord]) -> MetricValue: ...
```

Argument F1 flattens nested JSON objects into dotted key paths and canonical scalar JSON strings. Lists are compared by index for v1.0.

### Comparison

```python
def compare_evaluations(base_dir: Path, adapter_dir: Path, out_dir: Path) -> StageManifest: ...
```

Comparison fails unless both reports share config digest, test split digest, parser version, decoding config, and metric names.

## Alternatives Considered

- Use model-judged scoring. Rejected because it adds another model and weakens reproducibility.
- Repair invalid JSON before scoring. Rejected because valid structured output is a primary metric.
- Count only full-call exact match. Rejected because parse, name, and argument failures require separate diagnosis.
- Allow stochastic decoding. Rejected because it makes regression comparisons noisy.

## Drawbacks

- Exact argument comparison penalizes semantically equivalent but differently formatted values.
- Array-by-index F1 may be harsh for unordered arrays.
- Retaining raw generations increases artifact size and may require redaction for private adaptations.

## Migration / Rollout

1. Implement parser and metric unit tests before model generation.
2. Add generation fixtures for parse statuses.
3. Wire base evaluation.
4. Wire adapter evaluation.
5. Add comparison report validation.

## Testing Strategy

- Unit-test parser statuses.
- Unit-test metric numerator and denominator counts.
- Unit-test nested argument F1.
- Snapshot-test `evaluation_report.json`.
- Integration-test comparison rejection for mismatched split digests.

## Open Questions

None for v1.0.

## References

- [03-data-model](../spec/03-data-model.md)
- [07-testing-strategy](../spec/07-testing-strategy.md)
