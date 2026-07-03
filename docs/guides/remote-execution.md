# Run the pipeline on Modal

[`remote_pipeline.py`](https://github.com/AbdelStark/sommelier/blob/main/remote_pipeline.py) runs all [six stages](../concepts/pipeline.md) in one Modal container on a rented GPU. It is how the [reference run](../results/reference-run.md) was produced. The wrapper does not fork the pipeline: it chains the same stage code the CLI uses, and only adds what a multi-hour GPU run in someone else's datacenter actually needs.

## Prerequisites

1. A [Modal](https://modal.com) account with billing enabled, authenticated with `uv run modal token new`.
2. A Hugging Face token with access to the base model, as `HF_TOKEN=...` in a `.env` file at the repo root. The entrypoint ships it to the container with `modal.Secret.from_dotenv`; it never appears in configs or artifacts.
3. Acknowledgement of the base model license terms. The [release preflight](../project/licensing.md) gate passes only when the variable equals the configured base model id exactly:

```bash
export SOMMELIER_ACK_BASE_MODEL_LICENSE="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
uv run sommelier release preflight --config examples/config.full.yaml
```

!!! warning "This rents a GPU and bills your account"

    Every command on this page starts a paid GPU container. The reference full run held one L40S for about 3.7 hours end to end (10,996 s of that was training); a smoke run takes about half an hour. Always smoke before you commit to the full run.

## The commands

```bash
# bounded smoke run (at most 100/20/20 examples, separate smoke- run namespace)
uv run modal run remote_pipeline.py \
  --config examples/config.smoke.yaml --mode smoke --max-rows 2500

# full reference run
SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=28800 \
uv run modal run --detach remote_pipeline.py \
  --config examples/config.full.yaml --mode full --max-rows 60000
```

Pass `--run-id <name>` to name the run; otherwise an id is generated, and smoke runs always get a `smoke-` prefix so a later full run cannot overwrite them. Run the full pipeline with `--detach` so a dropped local connection does not kill hour three of training; you pull the results off the volume afterwards.

Two environment variables are read at launch time, before the container starts:

| Variable | Default | Meaning |
|----------|---------|---------|
| `SOMMELIER_GPU` | `A10G` | GPU type Modal attaches to the container |
| `SOMMELIER_TIMEOUT_SECONDS` | `14400` (4 h) | Modal function timeout for the whole run |

The default timeout is too short for the full run, hence `28800` in the reference command. One subtlety: `SOMMELIER_GPU` selects the physical GPU, but its value is not visible inside the container, so the `gpu` field in the returned summary is read from `remote.gpu` in the [config](../reference/configuration.md). Keep the two consistent, as `examples/config.full.yaml` does with `gpu: L40S`.

## What the wrapper adds

Around the shared stages, the remote function adds five things:

- **In-container dataset export.** It loads the configured dataset at the pinned `dataset_revision`, and when `--max-rows` is below the dataset size it takes a seeded shuffle (`project.seed`) and selects that many rows. Each row is written as a `sommelier.raw_tool_call_row.v1` record, the same input shape `sommelier data prepare` accepts locally, so the remote run starts from the exact artifact contract described in [Data policy](../concepts/data.md).
- **A sequence-length audit right after formatting.** It tokenizes the `full_text` of every train and validation record, prints `n/p50/p95/max/budget`, and raises a `UserInputError` naming the first offending example if anything exceeds `train.max_sequence_length`. The [completion-only collator](../concepts/training.md) refuses by design to truncate target tokens, so an over-budget example would otherwise kill training after base evaluation had already spent GPU time. Catching it here costs seconds. The fix is the one the error suggests: raise `train.max_sequence_length` above the longest rendered sequence.
- **GPU memory cleanup between stages.** The chained stages each load an 8B model inside a single process, so the wrapper runs `gc.collect()` and `torch.cuda.empty_cache()` after every stage to release the previous model's CUDA memory before the next load.
- **A volume commit after every stage.** Partial progress is visible on the volume as it happens. When a run fails at hour three, the earlier stages' checksummed artifacts are already committed and stay useful.
- **Recorded package versions.** The installed versions of `torch`, `transformers`, `peft`, `bitsandbytes`, `trl`, and `datasets` go into the returned summary (`"absent"` if a package is missing), so a result can always be tied to the stack that produced it.

The function also sets `HF_HOME=/hf-cache` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch initializes CUDA. Long-sequence batches fragment the allocator, and expandable segments avoid out-of-memory failures caused by reserved-but-unallocated blocks rather than real usage.

## Volumes and pulling artifacts

| Volume | Mount | Contents |
|--------|-------|----------|
| `sommelier-artifacts` | `/artifacts` | Run directories and the config copy for each mode |
| `sommelier-hf-cache` | `/hf-cache` | Model weights and dataset cache, reused across runs |

Both are created on first use. Runs land at `artifacts/runs/<run_id>/` on the artifacts volume, in the [standard run layout](../reference/artifacts.md). Pull them down with `modal volume get`:

```bash
# just the comparison report
uv run modal volume get sommelier-artifacts artifacts/runs/<run_id>/report/ report/

# the whole run directory
uv run modal volume get sommelier-artifacts artifacts/runs/<run_id>/ <run_id>/
```

The trailing slash matters: it tells Modal the remote path is a directory.

## The returned summary

An attached run prints a JSON summary when the pipeline finishes. A detached run cannot print it; the metrics and per-stage timings it contains are already on the volume, so read `comparison_report.json` and `runtime_metadata.json` there. The `versions` and `raw_rows` fields exist only in the printed summary.

| Key | Contents |
|-----|----------|
| `run_id` | The resolved run id |
| `gpu` | `remote.gpu` from the config (authoritative; see above) |
| `raw_rows` | Number of raw rows exported before preparation |
| `versions` | Installed versions of the six training-stack packages |
| `metrics` | `base` and `adapter` metric blocks plus `deltas`, read from `report/comparison_report.json` |
| `stage_seconds` | Per-stage wall clock, read from `runtime_metadata.json` |
| `report_path` | Path of `comparison_report.md` under the artifact root |

The metrics come from the digest-gated comparison, so they carry the same guarantees as any local run: see [Determinism and the comparison gate](../concepts/determinism.md) for what the gate checks and the [metrics reference](../reference/metrics.md) for what each number means. For the full walkthrough from clean checkout to published result, including the caveats, use the [reproduction guide](reproduction.md).
