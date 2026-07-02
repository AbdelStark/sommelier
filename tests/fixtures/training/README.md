# Training Smoke Fixture

`formatted/train.jsonl` and `formatted/validation.jsonl` are tiny,
synthetic `sommelier.formatted_example.v1` splits used by the one-step
smoke training tests (`tests/training/`). Records are internally coherent:
`full_text` starts with `prompt_text`, contains `target_text`, and
`prompt_sha256` matches the prompt bytes — the same contract the real
tokenizer path guarantees, so the fixture exercises the completion-only
collator end to end.

The local smoke runs the stubbed trainer path (no GPU packages, no model
download). The real one-training-step gate runs remotely via the smoke
pipeline (`sommelier pipeline run --mode smoke`), per
docs/spec/07-testing-strategy.md.
