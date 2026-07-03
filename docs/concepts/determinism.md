# Determinism and the comparison gate

A base-versus-adapter table is evidence only if both columns come from the same experiment: the same held-out examples, the same prompt bytes, the same parser, the same decoding. Sommelier makes each of those layers deterministic and digestible, then refuses to compare any two evaluations whose digests disagree. This page walks the stack from the bottom up and ends at the gate that enforces it.

One boundary first: the determinism claims here cover data preparation, formatting, and evaluation. Training on a GPU is not claimed to be bit-reproducible. What the pipeline guarantees is that whatever adapter a run produced is scored under exactly the conditions the gate checks, against a base model scored under the same conditions.

## The determinism stack

### Seeded, disjoint splits

Data preparation validates and deduplicates rows first, then shuffles the survivors with `random.Random(seed)` (seed from `project.seed`, default 42) and slices them in train, validation, test order. The same input rows and seed always produce the same splits. If fewer valid deduplicated rows exist than the requested split sizes, the stage fails before writing anything rather than producing smaller splits that would silently change the experiment. After splitting, disjointness is asserted: no `query_sha256` may appear in two splits. Deduplication policy and drop accounting are covered in [Data policy](data.md).

### Pinned revisions

`model.base_model_revision`, `model.tokenizer_revision`, and `dataset.dataset_revision` are required config fields with no defaults; a config without them does not validate. Every model and tokenizer load passes the pinned revision to `from_pretrained`, and the remote dataset export passes it to `load_dataset`. "Latest" is not a reproducible input: a Hub repo can gain a new tokenizer config or a re-uploaded dataset shard between your run and someone else's attempt to reproduce it.

### Canonical JSON

Whenever the pipeline serializes structured values that feed a digest or a comparison, it uses one canonical form: sorted keys, compact separators, ASCII escapes. Tool schemas in the system prompt, the assistant target (which must re-serialize to exactly the canonical string, with no fences or prose), and argument comparison in the [metrics](../reference/metrics.md) all use it. Without a canonical form, two semantically identical objects could serialize to different bytes, and every SHA-256 in the system would be measuring key order instead of content.

### A digest on every prompt

Formatting records `prompt_sha256`, the SHA-256 of each rendered prompt's UTF-8 bytes, on every formatted example. Evaluation copies that digest into every generation record, and scoring verifies that each generation's digest matches the formatted example it claims to answer. A mismatch means the model was shown something other than the recorded prompt, which breaks prompt identity (invariant INV-ARCH-004) and fails immediately instead of being skipped.

### Greedy decoding, enforced rather than coerced

Reference evaluation requires `eval.temperature` to be exactly `0.0`, `eval.do_sample` to be `false`, and `eval.max_new_tokens` to be positive. Any other value fails with an evaluation error before a single token is generated:

```text
sommelier: SOM005: reference evaluation requires temperature 0.0, got 0.7
hint: Set eval.temperature to 0.0 for deterministic decoding.
```

The values are never coerced. Coercion would change the experiment without changing the config digest, so the run's own record would lie about what was measured. Greedy decoding also needs no generation seed: with sampling disabled there is nothing random to seed. Every generation record stores the decoding dict it ran with, and a generations file that mixes decoding configs is rejected when the report is built.

### Evaluation replays stored prompts

`sommelier eval run` reads `prompt_text` directly from the stored formatted test split. It never re-renders prompts from prepared rows. If the tokenizer's chat template changed between formatting and evaluation, rebuilding would silently evaluate different bytes than training saw; replaying stored text makes that class of drift impossible, and the per-example digest check catches anything that slips past it. Base and adapter evaluation are the same code path reading the same file, as described in [The pipeline](pipeline.md).

## The comparison gate

Each evaluation report carries the identity of the experiment that produced it. `compare_evaluations` (in `sommelier/evaluation/report.py`, run by `sommelier report compare`) checks six fields for exact equality between the base and adapter reports:

| Field | What it pins |
|-------|--------------|
| `config_sha256` | SHA-256 of the run's `config.resolved.yaml`: model, revisions, prompt policy, decoding, everything |
| `split` | The evaluated split, always `test` |
| `test_split_sha256` | SHA-256 of the bytes of `formatted/test.jsonl`: prompts, targets, and metadata together |
| `prompt_set_sha256` | SHA-256 over the newline-joined, ordered per-example `prompt_sha256` values |
| `parser_version` | `sommelier.parser.v1`; a different parser would score generations differently |
| `decoding` | The exact decoding dict every generation ran with |

`test_split_sha256` and `prompt_set_sha256` overlap on purpose. The first changes if anything in the split file changes, including the gold targets used for scoring. The second isolates the prompt surface: exactly which prompts, in which order, the models saw. Both must match.

The gate checks more than the six fields. The two reports must have `model_kind` `base` and `adapter` respectively, their metric name sets must be identical, and the gate recomputes the digest of the resolved config in the run directory it is writing into and rejects reports whose `config_sha256` does not match it. That last check means you cannot point `report compare` at evaluation reports smuggled in from a different run's config.

Only when everything matches does the stage write `comparison_report.json` and its Markdown rendering, with the shared identity fields embedded in the report itself.

## What failure looks like

A mismatched field produces error code `SOM005` and a nonzero exit:

```text
sommelier: SOM005: comparison rejected: mismatched test_split_sha256
hint: Base and adapter evaluations must share the same test split, prompts, parser, decoding, and config.
```

Gate rejections are raised as `EvaluationError` (exit code 4). Violations of prompt identity caught earlier, such as a generation whose `prompt_sha256` does not match its formatted example, or a generations file mixing decoding configs, are raised as `InvariantViolation` (also `SOM005`, exit code 5), because they indicate corrupted or mismatched artifacts rather than a fixable input. The full code table is in [Errors and exit codes](../reference/errors.md).

## Why drift fails instead of warning

The gate enforces invariant INV-DATA-006: no comparison report exists unless base and adapter were scored under provably identical conditions. That contract only holds if failure is the response to drift.

A warning would be logged next to hundreds of routine lines and then forgotten, while the comparison report, the one artifact people actually share, would exist and look authoritative. Anyone downstream reads the report, not the logs. The design inverts that: the existence of `comparison_report.json` is itself the proof that the identity checks passed. A report that carries its own refutation in a log line is worse than no report, because it converts an honest mistake into a false claim.

Failing loudly also composes. `report compare` returns a nonzero exit code, so a CI job or a [remote pipeline](../guides/remote-execution.md) stops at the violation instead of publishing results built on mismatched inputs. And because every rejection names the exact field that differed, the failure is a diagnosis: mismatched `test_split_sha256` points at the data or formatting stage, mismatched `decoding` points at the eval config, mismatched `config_sha256` says the two evaluations never belonged together at all.
