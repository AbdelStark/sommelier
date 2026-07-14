# Run the pipeline on Modal

[`remote_pipeline.py`](https://github.com/AbdelStark/sommelier/blob/main/remote_pipeline.py) runs all [seven stages](../concepts/pipeline.md) in one Modal container on a rented GPU. It is how the [reference run](../results/reference-run.md) was produced. The wrapper does not fork the pipeline: it chains the same stage code the CLI uses, and only adds what a multi-hour GPU run in someone else's datacenter actually needs.

## Prerequisites

1. A [Modal](https://modal.com) account with billing enabled, authenticated with `uv run modal token new`.
2. A Hugging Face token with access to the base model, as `HF_TOKEN=...` in a `.env` file at the repo root. Pipeline and semantic-review entrypoints ship it with `modal.Secret.from_dotenv`; it never appears in configs or artifacts.
3. For Hebrew v3 translation, an OpenAI API key with Responses access. The CPU producer requires the named Modal secrets `openai-api-key` (`OPENAI_API_KEY`) and `huggingface-read-token` (`HF_TOKEN`). Provision them without putting values in a config:

```bash
uv run modal secret create openai-api-key OPENAI_API_KEY="$OPENAI_API_KEY"
uv run modal secret create huggingface-read-token HF_TOKEN="$HF_TOKEN"
```

4. Acknowledgement of the base model license terms. The [release preflight](../project/licensing.md) gate passes only when the variable equals the configured base model id exactly:

```bash
export SOMMELIER_ACK_BASE_MODEL_LICENSE="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
uv run sommelier release preflight --config examples/config.full.yaml
```

!!! warning "These commands use paid compute"

    Pipeline and semantic-review commands start paid GPU containers. The Hebrew
    Responses producer uses a paid CPU container and bills the OpenAI API
    separately. Its 140-row diagnostic calculated $0.357667500 from usage and
    pinned public list prices; that is not an invoice. The runnable commands
    set local estimate ceilings of $1.00 for smoke and $50.00 for full, but
    those are not provider account/project caps. Always smoke and check
    provider-side spend controls first.

## The commands

### Hebrew v3 full-shape QLoRA diagnostic

Before committing to a full training allocation, the dedicated diagnostic
exercises the exact v3 resource shape on one L40S without reading either
experiment dataset or calling the translation provider:

```bash
uv run modal run --detach remote_qlora_preflight.py \
  --config examples/config.v3-he-full.yaml \
  --run-id he-v3-l40s-shape-001
```

It constructs paired English/Hebrew formatted rows whose full lengths are
between 4,080 and 4,096 tokens, runs four batch-4 training microbatches through
gradient accumulation for one real optimizer update, then runs one explicit
batch-4 evaluation forward. The model path is the full pinned Nemotron
checkpoint under NF4/bfloat16 QLoRA, rank 16/alpha 32/dropout 0.05, all seven
target module families, and non-reentrant gradient checkpointing. The Modal
function fixes `gpu=L40S`, a four-hour outer timeout, and zero retries. The
runtime gate requires exactly one visible L40S and rejects any `hf_device_map`
entry placed on CPU, disk, or a GPU other than CUDA device 0.

`preflight_report.json` records the exact config bytes/resolution, runtime,
hardware, attached LoRA modules, trainable parameters, observed batch/step
counts, finite losses, peak allocated/reserved memory, source commit and dirty-
state digest, plus a manifest of every regular file present when finalization
runs except the self-referential `preflight_report.json` and
`artifact_manifest.json`. After a safe run id reserves a new output directory,
config, provenance, runtime, and training failures—including OOM—write a
redacted terminal report before propagating. Output is non-resumable; use a new
run id rather than overwriting a prior attempt.

This is paid resource-fit evidence only. A dirty checkout is recorded rather
than hidden, but even a clean successful diagnostic is ineligible for dataset,
accuracy, full-training, cost-saving, or release claims. Run it from the clean
producer revision for the most useful comparison with the later full run.

### Pipeline and translation commands

```bash
# bounded smoke run (at most 100/20/20 examples, separate smoke- run namespace)
uv run modal run remote_pipeline.py \
  --config examples/config.smoke.yaml --mode smoke --max-rows 2500

# English-only v1/default full run (15,000/1,000/1,000)
SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=86400 \
uv run modal run --detach remote_pipeline.py \
  --config examples/config.full.yaml --mode full --max-rows 60000

# Hebrew paired smoke: explicit paid Responses/Flex producer on CPU
SOMMELIER_TIMEOUT_SECONDS=3600 \
uv run modal run --detach remote_translate.py \
  --config examples/config.v3-he-smoke.yaml \
  --run-id he-v3-translate-smoke --mode smoke --max-rows 2500 \
  --target-language he \
  --model-id gpt-5.5-2026-04-23 \
  --model-revision gpt-5.5-2026-04-23 \
  --max-new-tokens 512 --translator-interface instruction_chat \
  --max-model-len 0 --output-decoder standard \
  --runtime-backend openai_responses \
  --openai-service-tier flex --openai-max-workers 8 \
  --openai-list-price-limit-usd 1.00

SOMMELIER_GPU=A10G SOMMELIER_TIMEOUT_SECONDS=10800 \
uv run modal run --detach remote_pipeline.py \
  --config examples/config.v3-he-smoke.yaml --mode smoke --max-rows 2500 \
  --translation-run-id he-v3-translate-smoke

# Hebrew full evidence, only after publishing the audited paired dataset and
# replacing its `main` revision in config.v3-he-full.yaml with the exact commit
SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=86400 \
uv run modal run --detach remote_pipeline.py \
  --config examples/config.v3-he-full.yaml --mode full --max-rows 60000 \
  --run-id he-v3-full
```

Pass `--run-id <name>` to name the run; otherwise an id is generated, and smoke runs always get a `smoke-` prefix so a later full run cannot overwrite them. Run the full pipeline with `--detach` so a dropped local connection does not kill hour three of training; you pull the results off the volume afterwards. The default full config is English-only, so it has no translation artifact to stage or verify.

For a paired **smoke** run, `--translation-run-id` names the completed
`remote_translate.py` run staged next to the exported root rows. Translation
and pipeline must use the same config, mode, and `--max-rows`; a diagnostic
`--limit` prefix cannot feed the pipeline. The staging gate checks the selected
root identities, row bytes, summary, and publication manifest before data
preparation starts.

A **paired full** run rejects `--translation-run-id`. Publish the complete
audited survivor set first, together with `translation_summary.json`,
`translation_semantic_review_template.json`,
`translation_semantic_review.json`, and `translation_publication.json`, then
pin the exact 40- or 64-character Hugging Face dataset commit under
`datasets`. The driver downloads that commit and verifies that the publication
manifest binds the consumed rows, summary, locked machine template, and final
semantic review. The summary must also pin a clean source revision, the exact
dated provider snapshot, returned service tier, provider request/runtime
identity, content-free journal evidence, and implementation revision. The dated
API snapshot is not a provider-weight checksum or a promise of byte-identical
regeneration. The [Hebrew v3 methodology](../results/hebrew-v3.md) shows the
full translation and publication order.

Produce the locked semantic-review template from the completed full
translation run before any human edits:

```bash
SOMMELIER_GPU=A10G uv run modal run remote_semantic_review.py \
  --translation-run-id he-v3-translate-full
```

This producer fails unless it runs from the same clean immutable Git SHA
recorded by the full translation. Its evidence image is pinned to Python
3.13.3, torch 2.11.0, transformers 5.13.1, tokenizers 0.22.2, accelerate
1.14.0, huggingface-hub 1.22.0, sentencepiece 0.2.2, and sacremoses 0.1.1;
the local `semantic-review-create` path enforces those same versions and
records its local hardware. The remote artifact also records the explicitly
dispatched GPU and timeout rather than reading a container default. Keep
`translation_semantic_review_template.json` untouched, review a distinct copy,
and write `translation_semantic_review.json` to a third distinct file. The
finalizer rejects path aliases and hard links as well as identical path names.
Its exact flags are in the [CLI reference](../reference/cli.md#data-semantic-review-finalize).
The review is intentionally labeled non-native; no native-speaker review has
been completed.

The pipeline launch itself must also come from a clean Git worktree with an
immutable commit. The local Modal entrypoint measures that identity immediately
before dispatch; a dirty or unidentifiable source tree is rejected for full
evidence.

The Hebrew teacher is a separate provider and runtime boundary from the
fine-tuned base model. The full contract pins `gpt-5.5-2026-04-23` as both model
id and revision, the instruction-chat prompt family, 512 maximum output tokens,
explicit Flex service, a 900-second per-request timeout, three audited row
attempts, eight provider workers, 32-row durable checkpoints, and an explicit
local public-list-price ceiling (`1.00` USD for smoke; `50.00` USD for full).
Model-name matching
never authorizes paid inference: the launch must explicitly pass
`--runtime-backend openai_responses`.

The producer is a CPU-only Modal image pinned to Python 3.13.3,
`openai==2.45.0`, and `datasets==5.0.0`. Responses use strict JSON Schema,
`store=false`, background mode disabled, truncation disabled, reasoning effort
`none`, zero SDK retries, and a stable non-PII safety identifier. Row attempts
remain the semantic/audit retry boundary. Separately, an exact Flex HTTP 429
`resource_unavailable` response may retry the same row attempt at journaled
`provider_call_attempt` delays of 1, 2, 4, 8, and 16 seconds. Those calls never
switch tier or consume another row attempt. The returned model and service tier
must match the request. OpenAI's
[Flex processing guide](https://developers.openai.com/api/docs/guides/flex-processing)
recommends the pinned 15-minute timeout and notes slower service plus occasional,
uncharged `429 Resource Unavailable` responses; the producer never falls back to
another tier. Protected values are replaced
with deterministic ASCII placeholders before the provider request, restored
afterward, and audited byte for byte. The source query and bounded selected-tool
projection are still sent to OpenAI. `store=false` is not a Zero Data Retention
claim, and strict structured output does not establish semantic correctness.
Rows that exhaust the declared audits are counted drops, so the accepted rows
remain a machine-translated survivor corpus.

The raw v2 provider journal records a source id and attempt on every response,
error, and replay without adding those fields to the provider request. It is
fsynced before a response returns and committed with each 32-row Modal chunk.
This supports attributed replay but is not exactly once: a hard kill can lose
the current uncommitted chunk, and a death after provider acceptance but before
receipt/fsync can rebill a request. The raw journal stays in durable producer
artifacts. Only its content-free v2 aggregate and digest are published inside
`translation_summary.json`. The required ceiling is a local admission/stop
estimate, not an invoice, observed billing artifact, or provider
account/project cap.

To evaluate a published adapter instead of training one (the baseline shape),
pass `--adapter-id <hf-repo-or-dir>` and an immutable
`--adapter-revision`; the train stage is skipped. Hebrew v3 pins the v1 adapter
to `45a6e2fa3e29f8393ddf1e9bda51a9461b41ee0e`.

Remote entrypoints read the outer timeout at launch time; GPU-backed entrypoints
also read the allocation label. The CPU-only OpenAI translation producer does
not use a GPU allocation, so setting `SOMMELIER_GPU` does not change its runtime:

| Variable | Default | Meaning |
|----------|---------|---------|
| `SOMMELIER_GPU` | `A10G` | GPU type for GPU-backed functions; ignored by the OpenAI producer |
| `SOMMELIER_TIMEOUT_SECONDS` | `14400` (4 h) | Modal function timeout for the whole remote call |

For the GPU pipeline, Modal's outer timeout is the only enforced deadline; the
OpenAI producer separately pins its 900-second client timeout per request. The
legacy-named `remote.data_timeout_seconds`, `remote.train_timeout_seconds`, and
`remote.eval_timeout_seconds` fields are planning estimates; the current
single-function pipeline does not install per-stage watchdogs. The driver uses
their sequential sum as an outer-timeout admission floor:
`data_timeout_seconds + train_timeout_seconds + 2 * eval_timeout_seconds`
because both evaluation arms run sequentially; an external-adapter baseline
omits the training term. The English-only default full estimate is 45,000
seconds (1,800 + 28,800 + 2 × 7,200); its command uses an 86,400-second outer
timeout. The Hebrew full estimate is 81,000 seconds (1,800 + 43,200 + 2 ×
18,000), so the arithmetic planning headroom is 5,400 seconds. A stage may run
past its individual estimate and continue until the outer deadline. An outer
value below the applicable planning sum fails admission before the dataset or
model loads, and a value above 86,400 is rejected as exceeding the provider
maximum.

`SOMMELIER_GPU` selects the physical GPU, while `remote.gpu` is the recorded
allocation label. The driver rejects a mismatch before doing paid work; keep
the environment and config values identical.

## What the wrapper adds

Around the shared stages, the remote function adds six things:

- **In-container dataset export.** It loads the configured dataset at the pinned `dataset_revision`, and when `--max-rows` is below the dataset size it takes a seeded shuffle (`project.seed`) and selects that many rows. Each row is written as a `sommelier.raw_tool_call_row.v1` record, the same input shape `sommelier data prepare` accepts locally, so the remote run starts from the exact artifact contract described in [Data policy](../concepts/data.md).
- **Fail-closed paired-source provenance.** Smoke runs may stage a matching diagnostic translation. Full runs instead export the paired dataset at its immutable publication revision; Hebrew evidence requires the checksummed translation summary, locked review template, finalized semantic review, and publication manifest before the shared data stage can see those rows.
- **Persisted tokenization evidence before compute-heavy stages.** The shared `analyze tokenization` stage tokenizes every stored query, prompt, target, and `full_text`, writes p50/p95/p99/max summaries plus exact root↔translation ratios, and raises a `UserInputError` if a configured training row exceeds `train.max_sequence_length`. The [completion-only collator](../concepts/training.md) refuses by design to truncate target tokens, so catching the mistake here avoids spending on evaluation or training first.
- **GPU memory cleanup between stages.** The chained stages each load an 8B model inside a single process, so the wrapper runs `gc.collect()` and `torch.cuda.empty_cache()` after every stage to release the previous model's CUDA memory before the next load.
- **A volume commit after every stage.** Partial progress is visible on the volume as it happens. When a run fails at hour three, the earlier stages' checksummed artifacts are already committed and stay useful.
- **No automatic whole-run retries.** The chained function has no stage-resume contract, so a late retry would repeat baseline evaluation and QLoRA training under the same run id. A failed attempt stays failed; inspect its committed artifacts, fix the cause, and launch an explicit new run id.
- **Recorded package versions and execution boundary.** The Python patch version and installed versions of `torch`, `transformers`, `tokenizers`, `accelerate`, `peft`, `bitsandbytes`, `datasets`, and `huggingface_hub` go into the returned summary (`"absent"` if a package is missing). The pipeline image pins these to the probe-established stack (Python 3.13.3, torch 2.13.0, transformers 5.13.1, tokenizers 0.22.2, accelerate 1.14.0, peft 0.19.1, bitsandbytes 0.49.2, datasets 5.0.0, and huggingface_hub 1.23.0); a Hebrew full run rejects any drift before dataset export. Runtime metadata also records the clean launcher revision, provider-enforced outer timeout, configured GPU label, configured stage planning sum, arithmetic headroom, `per_stage_watchdogs_enforced: false`, and the Hugging Face download policy. Three-arm release evidence requires Python and the direct inference stack (`torch`, `transformers`, `tokenizers`, `accelerate`, `peft`, `datasets`, and `huggingface_hub`) to be present at identical versions in every arm.

The image and remote function both force `HF_HUB_DISABLE_XET=1` and `HF_HUB_DOWNLOAD_TIMEOUT=600` before Hugging Face access; runtime metadata records that boundary. The function also sets `HF_HOME=/hf-cache` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch initializes CUDA. Long-sequence batches fragment the allocator, and expandable segments avoid out-of-memory failures caused by reserved-but-unallocated blocks rather than real usage.

## Volumes and pulling artifacts

| Volume | Mount | Contents |
|--------|-------|----------|
| `sommelier-artifacts` | `/artifacts` | Run directories and the config copy for each mode |
| `sommelier-hf-cache` | `/hf-cache` | Model weights and dataset cache, reused across runs |

Both are created on first use. Runs land at `artifacts/runs/<run_id>/` on the artifacts volume, in the [standard run layout](../reference/artifacts.md). Pull them down with `modal volume get`:

```bash
# just the comparison report
mkdir -p artifacts/runs/<run_id>
uv run modal volume get sommelier-artifacts \
  artifacts/runs/<run_id>/report/ artifacts/runs/<run_id>/

# the whole run directory
mkdir -p artifacts/runs
uv run modal volume get sommelier-artifacts \
  artifacts/runs/<run_id>/ artifacts/runs/
```

The local destination must already exist and must be the **parent** directory.
Modal creates the remote directory's basename underneath it. Passing a
nonexistent run path as the destination can make the CLI treat that path as a
single output file; passing the run directory itself as an existing
destination creates an unwanted nested `<run_id>/<run_id>/` layout.

## The returned summary

An attached run prints a JSON summary when the pipeline finishes. A detached run cannot print it; the metrics and per-stage timings it contains are already on the volume, so read `comparison_report.json` and `runtime_metadata.json` there. The `versions` and `raw_rows` fields exist only in the printed summary.

| Key | Contents |
|-----|----------|
| `run_id` | The resolved run id |
| `gpu` | `remote.gpu` from the config (authoritative; see above) |
| `raw_rows` | Number of raw rows exported before preparation |
| `versions` | Python patch version plus the eight recorded distribution versions |
| `metrics` | `base` and `adapter` metric blocks plus `deltas`, read from `report/comparison_report.json` |
| `stage_seconds` | Per-stage wall clock, read from `runtime_metadata.json` |
| `report_path` | Path of `comparison_report.md` under the artifact root |

The metrics come from the digest-gated comparison, so they carry the same guarantees as any local run: see [Determinism and the comparison gate](../concepts/determinism.md) for what the gate checks and the [metrics reference](../reference/metrics.md) for what each number means. For the full walkthrough from clean checkout to published result, including the caveats, use the [reproduction guide](reproduction.md).
