# Testing

The numbers this site quotes come out of a pipeline whose machinery is tested deterministically on any machine, and whose GPU behavior is verified separately on real hardware. This page describes what runs where, what each layer proves, and where the local suite honestly stops.

## The local gate

Five commands, identical locally and in [CI](https://github.com/AbdelStark/sommelier/blob/main/.github/workflows/ci.yml), which runs them on every pull request and push to `main`:

```bash
uv run ruff check .
uv run mypy sommelier tests
uv run pytest
uv run sommelier config validate --config examples/config.smoke.yaml
uv run sommelier data validate-fixtures
```

mypy runs strict over both the package and the tests. The whole gate is designed for a clean machine: no GPU, no model downloads, no accounts. That property is enforced by a test, not by convention (import discipline, below).

## What the suite covers

330 tests at the time of writing, under [`tests/`](https://github.com/AbdelStark/sommelier/blob/main/tests). The themes that carry the most weight:

**Golden prompt fixtures.** Synthetic prepared examples are rendered through a deterministic stub chat template and compared byte for byte against a committed snapshot that pins the rendered messages, prompt text, full text, and `prompt_sha256` digests. Any drift in prompt construction fails CI. The stub stands in for the real tokenizer template, which is exercised on remote runs.

**Parser and metric regression fixtures.** Synthetic raw generations cover every parse status (`ok`, `no_json`, `invalid_json`, `invalid_shape`) and the metric edge cases: wrong function name, missing and extra argument keys, nested value mismatches, list order. Committed snapshots pin the scored records and every metric's value, numerator, and denominator, so parser or scoring drift fails CI with a diff.

**Comparison report snapshot.** The Markdown rendering of the comparison report is pinned against a golden file, so the human-facing report format cannot drift silently either.

**Split-leakage property tests.** Across multiple seeds, data preparation is asserted to place every query hash in exactly one split. This is the invariant the [comparison gate](../concepts/determinism.md) ultimately rests on: a leaked test example would poison every downstream number.

**Secret hygiene.** The config loader rejects secret-like keys and values, failed manifests redact before writing, and the artifact scanner is tested against tokens, environment values, and home paths planted in JSON, JSONL, and Markdown. The behaviors under test are described on the [security page](security.md).

**Import discipline.** `test_package_modules_never_import_heavy_dependencies` in [`tests/test_imports.py`](https://github.com/AbdelStark/sommelier/blob/main/tests/test_imports.py) imports every `sommelier` module in a clean subprocess interpreter and asserts that none of `torch`, `transformers`, `trl`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `vllm`, `modal`, `cudf`, or `wandb` got loaded. This turns "the core is GPU-free" from a claim into a property that fails CI when violated.

**Doc links.** Every relative Markdown link in the README, changelog, `docs/`, and `licenses/` must resolve, and key pages must retain required commands and warnings. Documentation rot fails CI like any other regression.

**Training mechanics.** The completion-only collator is tested one training step at a time against a synthetic formatted fixture with a stubbed trainer: prompt tokens masked, target tokens not. No GPU packages load. The real one-training-step gate runs on a GPU, through the pipeline's smoke mode (`uv run modal run remote_pipeline.py --mode smoke`).

## Regenerating golden fixtures

An intentional change to the prompt template, parser, or metrics requires regenerating the snapshots:

```bash
SOMMELIER_REGENERATE_GOLDEN=1 uv run pytest tests/formatting tests/evaluation
git diff tests/fixtures/
```

The diff is then reviewed as a prompt-contract change, because that is what it is: every changed byte invalidates comparability with previously recorded runs. Making regeneration an explicit, diffable event is the point; drift becomes a reviewed decision instead of an accident.

## The evidence ladder

Local tests prove the machinery. They do not prove that an 8B model trains, so evidence is graded:

| Rung | What runs | What it proves |
|------|-----------|----------------|
| Local tests and fixture mode | 330 tests, plus any stage against synthetic rows via `--fixture` | The machinery: schemas, parser, metrics, masking, splits, gates |
| Remote smoke | The full chain on a GPU, capped at 100/20/20 examples | The plumbing: real tokenizer, real training step, real generation |
| Full run | The reference configuration, 15,000/1,000/1,000 | The numbers. Only these are quotable |

Try the first rung yourself in the [quickstart](../getting-started/quickstart.md); the [reproduction guide](../guides/reproduction.md) covers the other two, and the [reference run](../results/reference-run.md) is the full-rung result the site quotes.

## What local tests do not prove

No local test loads the real model or tokenizer. Real chat template rendering, GPU memory behavior, training convergence, and generation quality all live behind the remote smoke and full runs. The stub template shares the formatter's contract, not the tokenizer's bytes, so the golden fixtures catch drift in Sommelier's code; drift in the upstream template is controlled by pinning `model.tokenizer_revision` in the config, not by tests. Read the local suite as proof that the pipeline measures correctly, and the [reference run](../results/reference-run.md) as proof of what it measured.
