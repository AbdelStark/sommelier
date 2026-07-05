# CLI reference

The `sommelier` command is the entire public interface. The package deliberately exports nothing (`sommelier/__init__.py` holds only a version string), so everything below is the contract: each command, its flags, its defaults, what it reads, and what it writes, verified against [`sommelier/cli.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/cli.py).

| Command | Purpose |
|---------|---------|
| [`config validate`](#config-validate) | Check a config YAML, optionally write the resolved form |
| [`data prepare`](#data-prepare) | Validate, filter, dedupe, and split raw rows |
| [`data validate-fixtures`](#data-validate-fixtures) | Schema-check the synthetic fixture files |
| [`format build`](#format-build) | Render prepared splits through the chat template |
| [`eval run`](#eval-run) | Generate and score the base model or the adapter |
| [`train run`](#train-run) | Train the QLoRA adapter |
| [`report compare`](#report-compare) | Gate and compare the two evaluation reports |
| [`pipeline run`](#pipeline-run) | Run all six stages end to end |
| [`serve adapter`](#serve-adapter) | Serve the adapter behind an OpenAI-compatible endpoint |
| [`release preflight`](#release-preflight) | Run the licensing and secret-scan release gates |

## Global behavior

`--debug` is the one global flag and it goes before the subcommand:

```bash
sommelier --debug train run --config examples/config.full.yaml ...
```

Without it, a failure prints a single line to stderr in the form `sommelier: <code>: <message>`, sometimes followed by a `hint:` line, and exits with a code that says whose fault it is. With `--debug`, the full Python traceback follows. The complete code and exit-code contract is in [Errors and exit codes](errors.md).

On success every command except `serve adapter` (which blocks, serving requests) prints one confirmation line (for example `data prepare ok: run_id=demo out=...`) and exits 0.

## Run directories and --run-id

Every command that takes `--config` resolves a run directory at `<config dir>/<artifact_root>/runs/<run_id>/`. Note the anchor: `artifact_root` is resolved relative to the directory containing the config file, not your working directory, and the config schema rejects absolute paths for it. The run directory receives `config.resolved.yaml`, the run manifest `manifest.json`, and one `<stage>_manifest.json` per stage. The `--out` flag controls only where a stage's data artifacts go; manifests always land in the run directory.

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
                         [--model-revision <rev>] [--max-new-tokens <n>]
                         [--select-from <prepared-dir>] [--limit <n>]
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--input` | yes | | Raw JSONL of the root source's `sommelier.raw_tool_call_row.v1` records |
| `--out` | yes | | Directory for `rows.fr.jsonl`, `translation_summary.json`, and the resume checkpoint |
| `--model-id` | yes | | Hugging Face id of the translator model, served through vllm |
| `--model-revision` | no | `main` | Translator revision; recorded in the summary |
| `--max-new-tokens` | no | `512` | Generation budget per query |
| `--select-from` | no | | Prepared data directory; only rows selected into its splits are translated |
| `--limit` | no | all | Translate only the first N rows (smoke runs) |

Builds a paired French dataset from the root source's raw rows. Only the query is translated: `tools` and `answers` are copied byte for byte, and every output must reproduce its protected spans (gold argument values that appear verbatim in the English query) exactly. Failures retry up to twice with the rejection appended to the prompt, then drop with a counted reason in the summary. Output rows carry `source_example_id` naming the root row, ready for [`data prepare`](#data-prepare)'s paired-source path. Progress checkpoints per row, so an interrupted run resumes. This needs a GPU with vllm installed; `remote_translate.py` runs the same tool on Modal, including the raw-row export and split selection.

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

`--adapter` is required with `--model adapter` and rejected with `--model base`; both mistakes fail immediately as user-input errors before any model loads. The command runs two steps: generation (one greedy completion per test prompt, read from the stored `prompt_text`, never rebuilt) and scoring. It runs once per configured [eval slice](configuration.md), writing `generations.<slice>.jsonl` per language and one `evaluation_report.json` (per-slice and overall metrics) into `--out`, plus `eval_manifest.json` into the run directory; the manifest details record the slices and, for adapters, the weights' source and revision. Decoding must be deterministic (`temperature` 0.0, sampling off) or the command fails rather than coerce the settings. Metric definitions are in [Metrics](metrics.md); method and parser in [Evaluation](../concepts/evaluation.md).

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

The comparison gate. Both reports must agree on config digest, split, test-split digest, prompt-set digest, parser version, decoding settings, and metric names, and `--out` must sit inside a `runs/<id>/` layout whose `config.resolved.yaml` digest matches the reports. Any mismatch fails the command; there is no partial comparison. On success it writes `comparison_report.json` (authoritative) and `comparison_report.md` (human rendering) into `--out`, and `report_manifest.json` into the run directory. The gate's rationale is in [Determinism](../concepts/determinism.md).

```bash
sommelier report compare \
  --base examples/artifacts/runs/demo/eval/base \
  --adapter examples/artifacts/runs/demo/eval/adapter \
  --out examples/artifacts/runs/demo/report
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
| `--adapter-revision` | no | `main` | Revision when `--adapter-id` is a Hugging Face repo id |

Runs the six stages in order (data → format → eval-base → train → eval-adapter → compare) inside one run directory, with per-stage wall clock recorded in `runtime_metadata.json` and, after training, peak GPU memory read from the training metrics. Stage failures propagate with their exit codes; nothing is retried. The command fails up front if `--input` does not exist. With `--adapter-id` the run takes the baseline shape: nothing is trained, the adapter evaluation loads the referenced published adapter, and the comparison measures that adapter against the base model on the same prompts.

Smoke mode caps the splits at 100 train, 20 validation, and 20 test examples (taking the minimum with the configured counts) and prefixes the run ID with `smoke-` so a later full run can never overwrite smoke artifacts. Full mode uses the config as written; for the reference configuration that means a GPU and several hours, usually through the [remote driver](../guides/remote-execution.md).

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
sommelier release preflight --config <yaml>
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--config` | yes | | Config YAML |

Runs the release gates against the project and the artifact root (resolved relative to the config directory): license files present, third-party notices covering the configured base model and dataset, the "Built with Llama" derived-artifact notice, the dependency lock, and a secret scan of the whole artifact tree. It also requires the environment variable `SOMMELIER_ACK_BASE_MODEL_LICENSE` to equal `model.base_model_id`, a deliberate typing exercise that makes license acknowledgment explicit. The report `release_preflight.json` is written to the artifact root before any failure is raised, so the evidence survives. A failing secret scan exits 5; any other failing gate exits 3. Gate details are in [Licensing](../project/licensing.md) and [Security](../project/security.md).

```bash
SOMMELIER_ACK_BASE_MODEL_LICENSE="nvidia/Llama-3.1-Nemotron-Nano-8B-v1" \
  sommelier release preflight --config examples/config.full.yaml
```
