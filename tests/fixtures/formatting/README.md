# Formatting Golden Fixtures

- `prepared_examples.jsonl`: synthetic `sommelier.prepared_example.v1` rows,
  one per split, used as formatter input. Keep them synthetic; never commit
  private tool schemas.
- `golden_formatted_examples.jsonl`: committed snapshot of those rows rendered
  through the deterministic stub chat template in
  `tests/formatting/test_golden_prompts.py`. It pins rendered messages,
  prompt/full text, and `prompt_sha256` digests so unintended template drift
  fails CI.

Tests never download a tokenizer; the stub template stands in for the real
tokenizer chat template, which is exercised on remote runs.

## Regenerating after intentional template changes

```bash
SOMMELIER_REGENERATE_GOLDEN=1 uv run pytest tests/formatting
git diff tests/fixtures/formatting/golden_formatted_examples.jsonl
```

Review the diff carefully: every changed byte is a prompt-contract change
that invalidates comparability with previously recorded runs.
