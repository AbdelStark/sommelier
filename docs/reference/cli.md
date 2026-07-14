# CLI reference

The `sommelier` command is the entire public interface. The package deliberately exports nothing (`sommelier/__init__.py` holds only a version string), so everything below is the contract: each command, its flags, its defaults, what it reads, and what it writes, verified against [`sommelier/cli.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/cli.py).

| Command | Purpose |
|---------|---------|
| [`config validate`](#config-validate) | Check a config YAML, optionally write the resolved form |
| [`data prepare`](#data-prepare) | Validate, filter, dedupe, and split raw rows |
| [`data translate`](#data-translate) | Build an audited French or Hebrew paired dataset |
| [`data semantic-review-create`](#data-semantic-review-create) | Lock and back-translate the 200-row Hebrew review sample |
| [`data semantic-review-finalize`](#data-semantic-review-finalize) | Validate human decisions and bind the publication manifest |
| [`data validate-fixtures`](#data-validate-fixtures) | Schema-check the synthetic fixture files |
| [`format build`](#format-build) | Render prepared splits through the chat template |
| [`analyze tokenization`](#analyze-tokenization) | Measure exact per-language and matched-pair token cost |
| [`eval run`](#eval-run) | Generate and score the base model or the adapter |
| [`train run`](#train-run) | Train the QLoRA adapter |
| [`report compare`](#report-compare) | Gate and compare the two evaluation reports |
| [`report experiment`](#report-experiment) | Gate the base/v1/v3 English-Hebrew experiment |
| [`pipeline run`](#pipeline-run) | Run all seven stages end to end |
| [`serve adapter`](#serve-adapter) | Serve the adapter behind an OpenAI-compatible endpoint |
| [`release preflight`](#release-preflight) | Run the licensing and secret-scan release gates |
| [`release publish-dataset`](#release-publish-dataset) | Validate or explicitly publish the audited Hebrew dataset |
| [`release publish-adapter`](#release-publish-adapter) | Validate or explicitly publish the evidence-bound Hebrew adapter |

## Global behavior

`--debug` is the one global flag and it goes before the subcommand:

```bash
sommelier --debug train run --config examples/config.full.yaml ...
```

Without it, a failure prints a single line to stderr in the form `sommelier: <code>: <message>`, sometimes followed by a `hint:` line, and exits with a code that says whose fault it is. With `--debug`, the full Python traceback follows. The complete code and exit-code contract is in [Errors and exit codes](errors.md).

On success every command except `serve adapter` (which blocks, serving
requests) exits 0. Stage commands print one confirmation line (for example
`data prepare ok: run_id=demo out=...`); publication commands print their
machine-readable validation plan or verified receipt as JSON.

## Run directories and --run-id

Every stage command that takes `--config` resolves a run directory at `<config dir>/<artifact_root>/runs/<run_id>/`. Note the anchor: `artifact_root` is resolved relative to the directory containing the config file, not your working directory. Absolute paths, `..` traversal, and symlinks that resolve outside the config directory are rejected. The run directory receives `config.resolved.yaml`, the run manifest `manifest.json`, and one `<stage>_manifest.json` per stage. The `--out` flag controls only where a stage's data artifacts go; manifests always land in the run directory. `release preflight` follows the same default but accepts an explicit operator-selected `--artifact-root`, which may intentionally be elsewhere so a downloaded Modal tree can be scanned at its actual local path.

The run ID resolves in this order:

1. An explicit `--run-id` always wins.
2. `format build`, `eval run`, and `train run` infer it from the `--data` path with the regex `/runs/([^/]+)/` (first match, posix form). Keep stage outputs under one `runs/<id>/` layout and the ID follows automatically.
3. Otherwise a fresh ID is minted: a UTC timestamp plus eight hex characters, for example `20260702T101500Z-3fa2b9c1`.

Case 3 is fine for `data prepare`, which starts a run. For a mid-pipeline stage it silently starts a new run directory, which is rarely what you want, so either pass `--run-id` or keep the `runs/<id>/` layout. `report compare` takes no `--run-id` at all: it locates the run from its `--out` path.

## config validate

```text
sommelier config validate --config <yaml> [--write-resolved <dir>]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--config` | yes | | Path to the config YAML |
| `--write-resolved` | no | | Directory to write `config.resolved.yaml` into |

Loads and validates the config against `sommelier.config.v2` (unknown fields are rejected in every section) and runs the secret scan on both the raw YAML and the validated dump. A `sommelier.config.v1` file still loads: it is upgraded in memory with a deprecation warning, as described in [Configuration](configuration.md). With `--write-resolved`, writes `config.resolved.yaml` atomically to the given directory: the fully defaulted form whose SHA-256 digest identifies the config everywhere else in the pipeline. Field-by-field documentation is in [Configuration](configuration.md).

```bash
sommelier config validate --config examples/config.full.yaml
```

## data prepare

```text
sommelier data prepare --config <yaml> --out <dir> [--run-id <id>] [--input <jsonl>]
                       [--paired-input LANG=PATH]... [--fixture] [--gpu]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--config` | yes | | Config YAML |
| `--out` | yes | | Directory for the split files |
| `--run-id` | no | fresh ID | Run identifier |
| `--input` | no | `tests/fixtures/preparation_rows.jsonl` | Raw JSONL of `sommelier.raw_tool_call_row.v1` records for the root source |
| `--paired-input` | no | `<input stem>.<lang>.jsonl` next to `--input` | Raw JSONL for one paired source, as `LANG=PATH`; repeatable |
| `--fixture` | no | off | Use synthetic rows; skips real validation and splitting |
| `--gpu` | no | off | GPU dataframe coarse filter before Python validation |

Reads raw rows, validates each against the [data policy](../concepts/data.md), drops failures with a declared reason, deduplicates by normalized query, and writes seeded splits. When the config declares paired dataset sources, their rows are read from `--paired-input` (or the naming convention next to `--input`) and inherit split assignment from the root rows they name. Writes `train.jsonl`, `validation.jsonl`, and `test.jsonl` plus `drop_summary.json` into `--out`, and `data_manifest.json` into the run directory.

`--fixture` generates synthetic prepared examples instead, for GPU-free end-to-end testing; it writes the three split files but no drop summary, and `--input` and `--gpu` are ignored. `--gpu` runs a cudf coarse filter (null and query-length checks) before the exact Python validation, which cuts row count but never replaces the validation itself; it requires the `data-gpu` extra and fails with an install hint otherwise.

```bash
sommelier data prepare \
  --config examples/config.smoke.yaml \
  --out examples/artifacts/runs/demo/data \
  --run-id demo --fixture
```

## data translate

```text
sommelier data translate --input <jsonl> --out <dir> --model-id <hf-id>
                         [--target-language {fr,he}] [--model-revision <rev>]
                         [--max-new-tokens <n>] [--max-query-chars <n>]
                         [--output-decoder {standard,bytelevel_unicode}]
                         [--select-from <prepared-dir>] [--limit <n>]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--input` | yes | | Raw JSONL of the root source's `sommelier.raw_tool_call_row.v1` records |
| `--out` | yes | | Directory for `rows.<lang>.jsonl`, `translation_summary.json`, `translation_publication.json`, and the language-specific resume checkpoint |
| `--model-id` | yes | | Hugging Face id for this local CLI path; the interface selects vLLM chat or Transformers seq2seq |
| `--target-language` | no | `fr` | Target language, currently `fr` or `he` |
| `--model-revision` | no | `main` | Translator revision; recorded in the summary |
| `--max-new-tokens` | no | `1024` | Generation budget per query |
| `--max-query-chars` | no | `2000` | Reject longer translations, matching the preparation bound |
| `--output-decoder` | no | `standard` | Explicit completion decoder; `bytelevel_unicode` is only for a pinned model/tokenizer path whose runtime emits the reversible pre-decoder alphabet |
| `--select-from` | no | | Prepared data directory; only rows selected into its splits are translated |
| `--limit` | no | all | Translate only the first N rows (smoke runs) |

Builds a paired French or Hebrew dataset from the root source's raw rows. Only the query is translated: `tools` and `answers` are copied byte for byte. Protected literals include gold argument values found verbatim at identifier boundaries in the English query, present boundary-delimited components of comma-delimited gold strings, a gold function name written as explicit code/call syntax, and balanced list/dict spans that can be proven equivalent to a gold structure (including exact `HH:MM` to decimal-hour equivalence). Every output is audited for those exact source byte sequences. Chat translators mask only the same boundary-matched occurrences before generation; the raw seq2seq path relies on the same post-generation audit. Instruction-chat requests also receive deterministic semantic disambiguation metadata for the uniquely exact-name-matched gold-selected tool: its name and description plus sorted parameter names, types, and descriptions. Gold argument values, defaults, examples, enums, non-selected tools, and all other schema fields are excluded. The canonical ASCII-JSON context is delimiter-escaped and marked inert, non-output, and non-executable; a missing/ambiguous tool, malformed projection, or context over 8,192 characters is checkpointed as `invalid_row`, never truncated or guessed. TranslateGemma and MADLAD receive no semantic context, preserving their structured and raw-source request formats. The Hebrew target policy requires Hebrew coverage, permits conventional Latin technical text, rejects alphabetic text from unrelated scripts outside protected spans, and rejects unsafe bidi override controls. All targets reject the Unicode replacement character U+FFFD as corrupt output. Chat translators may retry up to twice. The retained MADLAD local path makes one deterministic attempt; the preregistered Hebrew v3 provider path makes exactly three audited attempts. A failed row is dropped with a counted reason, making the result a machine-translated survivor corpus. Output rows carry `source_example_id` naming the root row, ready for [`data prepare`](#data-prepare)'s paired-source path. Progress checkpoints bind the complete source row (including `tools` and `answers`), target, model and implementation revisions, request/context/postprocessing contracts, protected-placeholder schema, and audit version; reused output is re-audited. Provider progress also records `accepted_attempt` or `final_attempt`, while every raw v2 provider-journal event carries its source id and attempt. The initial publication manifest binds the accepted rows and summary; Hebrew full finalization regenerates it with the untouched machine-template and finalized-review digests.

This CLI command loads a local Hugging Face translator and therefore needs its
model runtime. The selected Hebrew v3 teacher is intentionally available only
through `remote_translate.py`'s explicit paid-provider boundary: exact snapshot
`gpt-5.5-2026-04-23`, interface `instruction_chat`, 512 maximum output tokens,
runtime backend `openai_responses`, Flex service, OpenAI SDK 2.45.0, a
900-second per-request timeout, zero SDK retries, eight workers, 32-row
checkpoints, and an explicit local public-list-price ceiling (`1.00` USD for
smoke or `50.00` USD for full) in a CPU-only Modal image. Three row attempts
cover semantic/audit rejection. Exact Flex HTTP 429 `resource_unavailable`
responses may retry the same row attempt after fixed 1/2/4/8/16-second delays;
each is a journaled `provider_call_attempt`, never switches tier, and does not
consume another row attempt. The ceiling is not an invoice or provider-side
account/project cap. The wrapper also
exports and selects the raw rows. It requires the named `openai-api-key` and
`huggingface-read-token` Modal secrets; model-name matching never selects the
provider. The exact command and cost/privacy boundaries are in the
[Hebrew v3 methodology](../results/hebrew-v3.md#reproduction-commands).

## data semantic-review-create

```text
sommelier data semantic-review-create --config <yaml>
  --root-input <rows.en.jsonl> --paired-input <rows.he.jsonl>
  --translation-summary <translation_summary.json>
  --out <translation_semantic_review_template.json>
```

Creates the immutable machine template for the Hebrew release gate. It
reconstructs the full root split assignment, selects exactly 200 accepted pairs
under the preregistered balancing/high-risk contract, and back-translates them
with `Helsinki-NLP/opus-mt-tc-big-he-en` at
`134c5a850dcaa763eec85bd1f4eb25112fecedbb`. The template binds the full paired
corpus, summary, ordered sample, requests, back-translations, model/decoding,
and empty rubric fields. It requires the model stack locally;
`remote_semantic_review.py --translation-run-id <full-id>` is the GPU producer
for a completed full translation run. Both paths require the exact producer
runtime (Python 3.13.3, torch 2.11.0, transformers 5.13.1, tokenizers 0.22.2,
accelerate 1.14.0, huggingface-hub 1.22.0, sentencepiece 0.2.2, and sacremoses
0.1.1), a clean immutable code revision identical to the full translation
revision, and a recorded hardware/allocation identity. The remote path pins
that runtime in its image; a mismatched local environment fails closed. It also
requires one safe run-id component and volume-commits an exclusively created,
empty, invalid final-path reservation before config/data/model work. Any later
failure removes it only when the producer can prove it is the exact unchanged
empty inode; replaced or nonempty markers stay fail-closed. A hard crash leaves
the empty marker for explicit operator inspection and recovery. This is a
mounted-filesystem no-replace boundary, not a claim of cross-container Modal
locking, so concurrent launches for one ID are unsupported. The pure local
command intentionally retains ordinary fixture output behavior. The
Marian request uses raw Hebrew input without a language prefix, disables
sampling, fixes one beam, caps batches at eight, and rejects rather than
truncates a source above 512 tokens.

## data semantic-review-finalize

```text
sommelier data semantic-review-finalize --config <yaml>
  --root-input <rows.en.jsonl> --paired-input <rows.he.jsonl>
  --translation-summary <translation_summary.json>
  --template <translation_semantic_review_template.json>
  --reviewed <reviewer-edited-copy.json>
  --out <translation_semantic_review.json> --reviewer-id <stable-id>
  [--publication-manifest <translation_publication.json>]
```

The reviewer may fill only the rubric, critical/pass decision, notes, and the
separately supplied reviewer id in a copy of the template. Finalization rejects
any changed sample, source/paired row, request, back-translation, or other
machine-locked field. All 200 decisions must be complete and internally
consistent; one critical error fails the entire publication rather than
dropping the row. A successful command writes the finalized
`sommelier.translation_semantic_review.v1` artifact and regenerates
`translation_publication.json` (or `--publication-manifest`) so it binds both
the untouched template and final review by SHA-256. `--template`, `--reviewed`,
and `--out` must name three distinct files; path aliases and hard links are
rejected so the machine template cannot be overwritten in place.

## data validate-fixtures

```text
sommelier data validate-fixtures [--fixtures-dir <dir>]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--fixtures-dir` | no | `tests/fixtures` | Directory of fixture JSONL files |

Schema-checks every `*.jsonl` file in the directory: each record must carry a supported `schema_version` and parse cleanly. Writes nothing. This is a repository hygiene command; it keeps the synthetic fixtures honest against the same reader the pipeline uses.

```bash
sommelier data validate-fixtures
```

## format build

```text
sommelier format build --config <yaml> --data <dir> --out <dir> [--run-id <id>] [--fixture]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--config` | yes | | Config YAML |
| `--data` | yes | | Directory with prepared splits |
| `--out` | yes | | Directory for the formatted splits |
| `--run-id` | no | inferred from `--data` | Run identifier |
| `--fixture` | no | off | Build without a tokenizer (fixture template policy) |

Reads the three prepared splits from `--data`, renders each example through the tokenizer's chat template into `prompt_text`, `target_text`, and `full_text` with a `prompt_sha256` digest, and writes formatted splits under the same names into `--out`, plus `format_manifest.json` into the run directory. The stage fails if `full_text` does not start with `prompt_text`, or the target does not appear after that prefix, because [training](../concepts/training.md) depends on a provable prompt boundary. `--fixture` substitutes a deterministic template so the stage runs without downloading a tokenizer.

```bash
sommelier format build \
  --config examples/config.smoke.yaml \
  --data examples/artifacts/runs/demo/data \
  --out examples/artifacts/runs/demo/formatted \
  --fixture
```

## analyze tokenization

```text
sommelier analyze tokenization --config <yaml> --data <formatted-dir> --out <dir>
                               [--run-id <id>]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--config` | yes | | Resolved experiment configuration |
| `--data` | yes | | Directory containing the three formatted splits |
| `--out` | yes | | Directory for `tokenizer_tax_records.jsonl` and `tokenizer_tax_report.json` |
| `--run-id` | no | inferred from `--data` | Run identifier |

Tokenizes the exact query, prompt, target, and full text consumed downstream. For each paired language it joins rows to exact roots, records coverage and per-example ratios, summarizes p50/p95/p99/max counts, and projects non-padding training tokens across configured epochs. The analysis is tied to the configured tokenizer id and revision. `pipeline run` additionally treats any over-budget configured training row as a hard gate before evaluation or training.

```bash
sommelier analyze tokenization \
  --config examples/config.full.yaml \
  --data examples/artifacts/runs/demo/formatted \
  --out examples/artifacts/runs/demo/analysis/tokenization
```

## eval run

```text
sommelier eval run --config <yaml> --model {base,adapter} --data <dir> --out <dir>
                   [--adapter <dir-or-hf-id>] [--adapter-revision <rev>] [--run-id <id>]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--config` | yes | | Config YAML |
| `--model` | yes | | `base` or `adapter` |
| `--data` | yes | | Directory with formatted splits |
| `--out` | yes | | Directory for generations and the report |
| `--adapter` | with `--model adapter` | | Trained adapter directory, or a published Hugging Face repo id |
| `--adapter-revision` | no | `main` | Revision when `--adapter` is a Hugging Face repo id |
| `--run-id` | no | inferred from `--data` | Run identifier |

`--adapter` is required with `--model adapter` and rejected with `--model base`; both mistakes fail immediately as user-input errors before any model loads. The command runs two steps: generation (one greedy completion per test prompt, read from the stored `prompt_text`, never rebuilt) and scoring. It runs once per configured [eval slice](configuration.md), writing `generations.<slice>.jsonl` per language, `inference_telemetry.json`, and one `evaluation_report.json` into `--out`. The run directory receives `eval-base_manifest.json` or `eval-adapter_manifest.json`, so the two model arms never overwrite each other's evidence. The manifest records the slices and, for adapters, the weights' immutable source/tree identity. Decoding must be deterministic (`temperature` 0.0, sampling off) or the command fails rather than coerce the settings. Metric definitions are in [Metrics](metrics.md); method and parser in [Evaluation](../concepts/evaluation.md).

```bash
sommelier eval run \
  --config examples/config.full.yaml \
  --model adapter \
  --data examples/artifacts/runs/demo/formatted \
  --out examples/artifacts/runs/demo/eval/adapter \
  --adapter examples/artifacts/runs/demo/train/adapter
```

## train run

```text
sommelier train run --config <yaml> --data <dir> --out <dir> [--run-id <id>]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--config` | yes | | Config YAML |
| `--data` | yes | | Directory with formatted splits |
| `--out` | yes | | Directory for the adapter weights |
| `--run-id` | no | inferred from `--data` | Run identifier |

Trains the QLoRA adapter on the formatted train split with completion-only loss and saves the adapter and tokenizer files into `--out`. Training metrics go to `training_metrics.jsonl` in the parent of `--out`, so the conventional layout is `--out .../train/adapter` with metrics at `.../train/training_metrics.jsonl`. The stage manifest is `train_manifest.json` in the run directory. Hyperparameters come from the config and are never adjusted to fit hardware: an out-of-memory failure surfaces as a [resource error](errors.md) whose hint names the exact fields to change.

```bash
sommelier train run \
  --config examples/config.full.yaml \
  --data examples/artifacts/runs/demo/formatted \
  --out examples/artifacts/runs/demo/train/adapter
```

## report compare

```text
sommelier report compare --base <dir> --adapter <dir> --out <dir>
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--base` | yes | | Base evaluation directory (contains `evaluation_report.json`) |
| `--adapter` | yes | | Adapter evaluation directory (contains `evaluation_report.json`) |
| `--out` | yes | | Directory for the comparison report, inside the run directory |

The comparison gate. Both reports must agree on model identity, config digest, split, test-split digest, prompt-set and pair-set digests, parser version, decoding settings, and metric names, and `--out` must sit inside a `runs/<id>/` layout whose `config.resolved.yaml` digest matches the reports. Any mismatch fails the command; there is no partial comparison. On success it writes `comparison_report.json` (authoritative) and `comparison_report.md` (human rendering) into `--out`, and `report_manifest.json` into the run directory. The gate's rationale is in [Determinism](../concepts/determinism.md).

```bash
sommelier report compare \
  --base examples/artifacts/runs/demo/eval/base \
  --adapter examples/artifacts/runs/demo/eval/adapter \
  --out examples/artifacts/runs/demo/report
```

## report experiment

```text
sommelier report experiment --base <eval-dir> --v1-en <eval-dir>
                            --v3-en-he <eval-dir> --out <dir>
                            --english-non-inferiority-margin <float>
                            --seed <int> --resamples <int>
```

Builds `sommelier.experiment_report.v1` from three independently checksummed
evaluation runs. Before reading outcomes, the finalizer requires a clean
immutable checkout at the exact source revision recorded by all three runs. It
then traverses the v3 root/publication/semantic-review inputs through succeeded
data, format, tokenization, train, and evaluation manifests to the exact scored
test split. It requires the fixed published v1 adapter and the canonical local
v3 adapter produced by that run's succeeded train manifest. It re-reads the
formatted splits and generation files, recomputes metrics, and requires
identical pinned model/tokenizer identity, full preregistered cohort counts,
English and Hebrew example order, prompt cohorts, pair-set digest, parser, and
decoding. V1 and v3 training config digests remain distinct arm provenance,
while the v3 config itself must match the committed Hebrew full-run contract.

The report compares v3 against v1 with paired-bootstrap intervals. It approves the Hebrew uplift statement only when the full-call interval's lower bound is above zero, and approves English non-inferiority only when the lower bound is at least the negative predeclared margin. Failed gates retain estimates and criteria but omit their claim statement.

The same report embeds `sommelier.sovereign_tco_evidence.v1`. It joins the v3
tokenizer report, training metrics/runtime and adapter tree, plus the
base/v1/v3 evaluation manifests and generation telemetry by exact path, digest,
config, hardware, decoding identity, clean source revision, and observed
runtime package versions. Observed QLoRA wall time, GPU allocation, peak memory
and storage remain distinct from the deterministic non-padding-token
projection. Sequential inference timing excludes model load, parsing, and
artifact I/O. Currency cost is unavailable without observed billing evidence,
and no full-fine-tuning saving is emitted without a matched full-parameter arm.

```bash
sommelier report experiment \
  --base artifacts/runs/he-v3-full/eval/base \
  --v1-en artifacts/runs/he-v3-v1-baseline/eval/adapter \
  --v3-en-he artifacts/runs/he-v3-full/eval/adapter \
  --english-non-inferiority-margin 0.01 \
  --seed 42 --resamples 2000 \
  --out artifacts/experiments/he-v3
```

## pipeline run

```text
sommelier pipeline run --config <yaml> --mode {smoke,full} [--input <jsonl>] [--run-id <id>]
                       [--adapter-id <dir-or-hf-id>] [--adapter-revision <rev>]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--config` | yes | | Config YAML |
| `--mode` | yes | | `smoke` or `full` |
| `--input` | no | `tests/fixtures/preparation_rows.jsonl` | Raw JSONL of `sommelier.raw_tool_call_row.v1` records |
| `--run-id` | no | fresh ID | Run identifier |
| `--adapter-id` | no | | Evaluate this published adapter (local dir or Hugging Face repo id); the train stage is skipped |
| `--adapter-revision` | no | `main` | Revision when `--adapter-id` is a Hugging Face repo id; full evidence must pass an immutable commit |

Runs the seven stages in order (data → format → tokenization → eval-base → train → eval-adapter → compare) inside one run directory, with per-stage wall clock recorded in `runtime_metadata.json` and, after training, peak GPU memory read from the training metrics. Base and adapter evaluation write `eval-base_manifest.json` and `eval-adapter_manifest.json` respectively, so neither arm overwrites the other's evidence. The tokenization stage is the pre-compute sequence-budget gate and writes the matched-pair cost evidence used by multilingual runs. Stage failures propagate with their exit codes; nothing is retried. The command fails up front if `--input` does not exist. Explicit run IDs must be 1–128 ASCII letters, digits, dots, underscores, or hyphens, beginning with an alphanumeric character; path separators, traversal, and absolute paths fail as user input before artifact mutation. Full mode also atomically rejects an existing run directory in the current filesystem view before writing its resolved config or manifest; a failed or completed full attempt must be followed by a fresh `--run-id`. Smoke mode keeps its diagnostic rerun behavior. With `--adapter-id` the run takes the baseline shape: nothing is trained, the adapter evaluation loads the referenced published adapter, and the comparison measures that adapter against the base model on the same prompts.

Smoke mode caps the splits at 100 train, 20 validation, and 20 test examples (taking the minimum with the configured counts) and prefixes the run ID with `smoke-` so a later full run can never overwrite smoke artifacts. The remote driver may stage a matching completed translation run in smoke mode. Full remote mode rejects that staging override: each paired source must be an immutable published dataset revision; Hebrew evidence carries rows plus the digest-bound translation summary, locked semantic-review template, finalized review, and publication manifest. See the [remote driver](../guides/remote-execution.md) for that publication boundary and the outer-timeout admission floor derived from non-enforced stage planning estimates.

```bash
sommelier pipeline run \
  --config examples/config.smoke.yaml \
  --mode smoke \
  --input data/raw/xlam_rows.jsonl
```

## serve adapter

```text
sommelier serve adapter --config <yaml> --adapter <dir> [--host <addr>] [--port <n>]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--config` | yes | | Config YAML |
| `--adapter` | yes | | Trained adapter directory |
| `--host` | no | `127.0.0.1` | Bind address |
| `--port` | no | `8000` | Bind port |

Loads the base model with the adapter and starts a blocking uvicorn server exposing `POST /v1/chat/completions` and `GET /health`. This exists to inspect the adapter, not to serve traffic; the request contract and its deliberate restrictions are in [Serving](../guides/serving.md).

```bash
sommelier serve adapter \
  --config examples/config.full.yaml \
  --adapter examples/artifacts/runs/demo/train/adapter \
  --port 8000
```

## release preflight

```text
sommelier release preflight --config <yaml> [--artifact-root <dir>]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--config` | yes | | Config YAML |
| `--artifact-root` | no | `<config dir>/<project.artifact_root>` | Intentional local artifact tree to scan and receive `release_preflight.json`; an explicit path may be outside the config directory |

Runs the release gates against the project and the exact selected artifact root: commit-certified project license and UTF-8 third-party notices covering the configured base model and root dataset, the "Built with Llama" derived-artifact notice, the commit-certified dependency lock, and a secret scan of the whole artifact tree. Git clean filters are honored, while assume-unchanged and skip-worktree entries prevent a clean source claim. It also requires the environment variable `SOMMELIER_ACK_BASE_MODEL_LICENSE` to equal `model.base_model_id`, a deliberate typing exercise that makes license acknowledgment explicit. The v2 report binds the normalized config; exact model, tokenizer, and ordered dataset revisions plus immutability decisions; producer Git commit and cleanliness decision; dependency-lock digest; and a digest, file count, and byte count for the selected tree. Every regular artifact is read again before certification to catch content mutation independently of timestamp metadata. Only the root-level `release_preflight.json` is excluded from its own tree identity. Run it from the exact producer repository root because project/source/license/lock discovery uses the current directory; absent or mismatched Git identity fails those file gates. A standalone pass records but does not itself require immutable revisions or a clean tree; strict adapter validation requires both.

When the artifact root is a safe writable directory, the report is written before a gate failure is raised so the evidence survives. If the root itself is a symlink, non-directory, uninspectable, or cannot be created safely, preflight writes nothing through it and says that no report was written. A failing secret scan exits 5; any other failing gate exits 3. Gate details are in [Licensing](../project/licensing.md) and [Security](../project/security.md).

```bash
SOMMELIER_ACK_BASE_MODEL_LICENSE="nvidia/Llama-3.1-Nemotron-Nano-8B-v1" \
  sommelier release preflight \
  --config artifacts/publication/hebrew-adapter/config.resolved.yaml \
  --artifact-root artifacts/publication/hebrew-adapter
```

## release publish-dataset

```text
sommelier release publish-dataset --config <yaml> --bundle <dir>
  --root-input <rows.en.jsonl> --repo-id <namespace/name>
  --commit-message <one-line-message> [--execute] [--create-repo]
  [--confirm-repo-id <namespace/name>] [--receipt <new-json-path>]
```

The command is validation-only unless `--execute` is present. This is the
preregistered Hebrew v3 publisher, not a generic dataset uploader:
`--repo-id` must be exactly
`abdelstark/sommelier-xlam-single-call-splits-he`, matching the configured
dataset identity. Any other well-formed repository ID is rejected before Hub
access. The bundle must contain exactly `README.md`, `rows.he.jsonl`,
`translation_summary.json`,
`translation_publication.json`, `translation_semantic_review_template.json`,
and `translation_semantic_review.json`. The tracked
[Hebrew dataset card](../release/hebrew-v3-dataset-card.md) is the release
template. Replace its pending-evidence block and remove
`REPLACE_FROM_VERIFIED_DATASET_BUNDLE` only after filling it from the audited
bundle; validation rejects the unresolved marker. The card must declare CC-BY-4.0,
Salesforce attribution, machine translation, and Hebrew. `--root-input` is not
uploaded: it lets the validator reconstruct the English/Hebrew pairing and run
the same full publication, semantic-review, provider-evidence, and immutable
producer checks consumed by the full pipeline. A raw
`openai_responses_provider.jsonl` journal, a symlink, an unexpected file, a
secret-like value, or an incomplete provenance chain fails before network I/O.

Execution additionally requires the `publish` extra, authenticated
`huggingface_hub`, and `--confirm-repo-id` exactly equal to `--repo-id`.
`--create-repo` is the only path that creates a repository; it creates one
public repository with `exist_ok=false`; in validation-only mode it records
that intent in the plan without creating anything. Omit it for an existing
dedicated repository that already has an immutable HEAD; a parentless initial
commit is allowed only for a repository created by this same transaction.
`--receipt` is mandatory on an executed publication, must not already exist,
and must be outside the source bundle so journaling cannot invalidate its
certified tree. Sommelier creates, preallocates, and fsyncs a pending
0600 receipt before any Hub mutation. It resolves and records the existing
immutable HEAD, inspects that exact snapshot, and submits the commit with HEAD
as its optimistic-concurrency parent. For the newly created empty repository,
it verifies the absent HEAD again immediately before its parentless initial
commit. The returned commit identity is durably journaled as
unverified before any download. Sommelier then enumerates that
exact revision, downloads every uploaded file, verifies every SHA-256, and
changes the receipt status to `verified`. Hugging Face's platform-managed
`.gitattributes` is the only extra remote file allowed.

```bash
# First pass: local validation only; no Hub import or mutation. This plan is
# for an absent first-publication destination.
uv run sommelier release publish-dataset \
  --config examples/config.v3-he-full.yaml \
  --bundle artifacts/publication/hebrew-dataset \
  --root-input artifacts/translation/he-v3-translate-full/rows.en.jsonl \
  --repo-id abdelstark/sommelier-xlam-single-call-splits-he \
  --commit-message "Publish audited Hebrew v3 paired rows" \
  --create-repo

# Deliberate first publication after reviewing the JSON plan.
uv run --extra publish sommelier release publish-dataset \
  --config examples/config.v3-he-full.yaml \
  --bundle artifacts/publication/hebrew-dataset \
  --root-input artifacts/translation/he-v3-translate-full/rows.en.jsonl \
  --repo-id abdelstark/sommelier-xlam-single-call-splits-he \
  --commit-message "Publish audited Hebrew v3 paired rows" \
  --execute --create-repo \
  --confirm-repo-id abdelstark/sommelier-xlam-single-call-splits-he \
  --receipt artifacts/publication/hebrew-dataset-receipt.json
```

If the dedicated repository already has an immutable HEAD, omit
`--create-repo` from both commands. Do not review a validation-only plan for an
existing repository and then add repository creation only at execution time.

## release publish-adapter

```text
sommelier release publish-adapter --bundle <dir>
  --repo-id <namespace/name> --commit-message <one-line-message>
  [--execute] [--create-repo] [--confirm-repo-id <namespace/name>]
  [--receipt <new-json-path>]
```

The mutation and round-trip rules are identical to dataset publication. Since
the adapter is Llama-derived, the repository basename (the part after the
namespace slash) must begin with the literal `Llama`; other well-formed IDs are
rejected before Hub access. The adapter bundle has a separate exact allowlist:
a public model card and
`THIRD_PARTY.md`; byte-exact reviewed `LICENSE-NVIDIA-OPEN-MODEL.txt`,
`LICENSE-LLAMA-3.1.txt`, and `NOTICE`; canonical PEFT LoRA config and
safetensors (plus only the declared tokenizer sidecars); the resolved config;
succeeded root/train manifests; final claim-gated `experiment_report.json`;
and a passing `release_preflight.json`. Validation rejects base-model tensors,
incomplete LoRA A/B pairs, mismatched file digests, an adapter/config/model
mismatch, a dirty or mutable producer identity, incomplete release gates,
invalid Hugging Face YAML license metadata, unresolved card markers, or a
model card that omits the base/data/source revisions, adapter tree digest,
experiment digest, license terms, or the prominent `Built with Llama` notice.
Start from the tracked
[adapter card template](../release/hebrew-v3-adapter-card-template.md), replace
every marker from the verified bundle, and let the publisher re-derive the
required hashes rather than trusting copied console values.

The order is part of the contract:

1. Assemble the complete curated bundle, except for `release_preflight.json`,
   from the exact clean producer revision. Pin every model, tokenizer, and
   dataset revision and finish the card, licenses, manifests, config, adapter,
   and experiment report.
2. Run `release preflight` with the bundle's `config.resolved.yaml` and the
   bundle directory as `--artifact-root`. The report excludes only itself from
   the certified tree.
3. Do not mutate the bundle. Run `publish-adapter` in validation-only mode,
   review the plan, and only then repeat it with the explicit execution flags.

For the canonical Hebrew run, the exact source mapping is documented in the
[adapter publication handoff](../results/hebrew-v3.md#adapter-publication-handoff):
`config.resolved.yaml`, `manifest.json`, and `train_manifest.json` come from
`artifacts/runs/he-v3-full/`; the required PEFT files and optional allowlisted
tokenizer sidecars come from its `train/adapter/`; and
`experiment_report.json` comes from
`artifacts/experiments/he-v3/experiment_report.json`. Copy the tracked card,
`THIRD_PARTY.md`, and exact license/notice files into the bundle, fill the card
from derived bundle identities, and publish before any tracked result edits.

```bash
# During bundle assembly, copy the reviewed terms and notice without editing.
cp licenses/LICENSE-NVIDIA-OPEN-MODEL.txt \
  artifacts/publication/hebrew-adapter/
cp licenses/LICENSE-LLAMA-3.1.txt licenses/NOTICE \
  artifacts/publication/hebrew-adapter/

# Certify this exact final bundle from the clean producer checkout.
SOMMELIER_ACK_BASE_MODEL_LICENSE="nvidia/Llama-3.1-Nemotron-Nano-8B-v1" \
  uv run sommelier release preflight \
  --config artifacts/publication/hebrew-adapter/config.resolved.yaml \
  --artifact-root artifacts/publication/hebrew-adapter

# Validation only: no Hub mutation. This plan assumes the destination is absent.
uv run sommelier release publish-adapter \
  --bundle artifacts/publication/hebrew-adapter \
  --repo-id abdelstark/Llama-3.1-Nemotron-Nano-8B-xlam-tool-calling-he-en-lora \
  --commit-message "Publish claim-gated Hebrew v3 QLoRA adapter" \
  --create-repo

# Deliberate first publication after reviewing that exact JSON plan.
uv sync --extra publish
uv run --extra publish sommelier release publish-adapter \
  --bundle artifacts/publication/hebrew-adapter \
  --repo-id abdelstark/Llama-3.1-Nemotron-Nano-8B-xlam-tool-calling-he-en-lora \
  --commit-message "Publish claim-gated Hebrew v3 QLoRA adapter" \
  --execute --create-repo \
  --confirm-repo-id abdelstark/Llama-3.1-Nemotron-Nano-8B-xlam-tool-calling-he-en-lora \
  --receipt artifacts/publication/hebrew-adapter-receipt.json
```

The receipt path must be fresh and outside the bundle. Require status
`verified` before recording the returned immutable commit or editing tracked
claims. If the dedicated repository already has an immutable HEAD, omit
`--create-repo` from both commands; never add it only after reviewing the
validation plan.

Validation and publication never delete remote files or overwrite a previous
local receipt. The publisher first copies every source into a private snapshot;
the same validated, scanned, hashed bytes are the only bytes handed to the Hub.
An executed attempt deliberately occupies its new outside-bundle receipt path,
even if it fails before the commit; use its `pending`,
`commit_submitting`, `commit_returned_unverified`, or `verified` state when
investigating. `commit_submitting` durably records the inspected parent before
the request begins. If a network error occurs after a commit request, inspect
both that journal and the Hub before retrying: the server may have accepted the
commit even when the client did not receive its identity.
