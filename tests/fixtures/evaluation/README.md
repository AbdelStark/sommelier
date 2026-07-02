# Evaluation Golden Fixtures

- `raw_generations.jsonl`: synthetic raw model outputs paired with gold
  calls. The set covers every parser status (`ok`, `no_json`,
  `invalid_json`, `invalid_shape`) and the metric edge cases: wrong function
  name, missing and extra argument keys, nested value mismatch, and list
  order sensitivity. Keep rows synthetic.
- `golden_scored_records.jsonl`: committed snapshot of those outputs run
  through `parse_tool_call` into scored records. Any parser or scoring drift
  fails CI.
- `golden_metrics.json`: committed snapshot of `compute_metrics` over the
  scored records, pinning every metric's value, numerator, and denominator.

## Regenerating after intentional parser or metric changes

```bash
SOMMELIER_REGENERATE_GOLDEN=1 uv run pytest tests/evaluation
git diff tests/fixtures/evaluation/
```

Review the diff carefully: changed numbers here change reported evaluation
results and invalidate comparability with previously recorded runs.
