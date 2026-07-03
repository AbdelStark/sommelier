# Reproduce the reference run

This guide walks from a clean checkout to a full reproduction of the published result and explains how to read the report it produces. Commands are exact. Steps that need external accounts or hardware say so up front, because renting the GPU costs money and you should know before you start.

## 1. Install

Prerequisites: Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AbdelStark/sommelier
cd sommelier
uv sync --extra dev
```

The base install never pulls GPU, remote execution, or tracking packages; those live behind optional extras and remote images. [Installation](../getting-started/installation.md) has the details.

## 2. Local validation (no GPU, no accounts)

```bash
uv run ruff check .
uv run mypy sommelier tests
uv run pytest
uv run sommelier config validate --config examples/config.smoke.yaml
uv run sommelier data validate-fixtures
```

A fixture pipeline slice runs entirely locally:

```bash
uv run sommelier data prepare --config examples/config.smoke.yaml --fixture \
  --out examples/artifacts/runs/local/data --run-id local
uv run sommelier format build --config examples/config.smoke.yaml \
  --data examples/artifacts/runs/local/data \
  --out examples/artifacts/runs/local/formatted --run-id local --fixture
```

To prepare real rows instead of fixtures, pass `--input <raw_rows.jsonl>` with `sommelier.raw_tool_call_row.v1` records; the input must contain at least as many valid deduplicated rows as the configured split sizes require, or preparation exits 2 with the shortfall.

Evaluation, training, and serving need the model stack (torch, transformers, peft); locally they fail with an actionable `ExternalDependencyError` (exit 3) by design.

## 3. Remote prerequisites

Reproducing the reference result requires:

- A [Modal](https://modal.com) account with billing enabled (`uv run modal token new`). Remote GPU time costs money; you are responsible for the spend.
- A Hugging Face token with access to the base model. Put `HF_TOKEN=...` in a `.env` file at the repo root: the remote entrypoint ships it to the container through Modal's secret mechanism (`modal.Secret.from_dotenv`). Tokens never go in config files, and the redaction scanner keeps them out of artifacts.
- Acknowledgement of the base model license terms:

```bash
export SOMMELIER_ACK_BASE_MODEL_LICENSE="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
uv run sommelier release preflight --config examples/config.full.yaml
```

The preflight verifies the project license, third-party notices ([licenses/THIRD_PARTY.md](https://github.com/AbdelStark/sommelier/blob/main/licenses/THIRD_PARTY.md)), the recorded model and dataset obligations, and scans artifacts for secrets before anything is published.

## 4. Smoke run

Always smoke before you spend hours. The smoke mode bounds split sizes (at most 100 train / 20 validation / 20 test examples) and writes into a separate `smoke-` run namespace, so a later full run can never overwrite it:

```bash
uv run modal run remote_pipeline.py \
  --config examples/config.smoke.yaml --mode smoke --max-rows 2500
```

[`remote_pipeline.py`](https://github.com/AbdelStark/sommelier/blob/main/remote_pipeline.py) exports the pinned dataset revision to raw rows inside the container, audits every rendered train and validation sequence against the training token budget right after formatting (so a budget mistake costs seconds, not GPU hours), then chains all six stages and commits artifacts to the `sommelier-artifacts` volume after each stage. [Run the pipeline on Modal](remote-execution.md) explains the moving parts.

If you have your own GPU machine with the training stack installed, the same chain runs locally through the CLI: `uv run sommelier pipeline run --config examples/config.smoke.yaml --mode smoke --input <raw_rows.jsonl>`.

## 5. Full run

```bash
SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=28800 \
uv run modal run --detach remote_pipeline.py \
  --config examples/config.full.yaml --mode full --max-rows 60000
```

Run it detached: the reference run took about 3.7 hours end to end on one L40S (10,996 s of training, about 29 minutes of evaluation). `examples/config.full.yaml` records the exact settings that produced the published result: 15,000/1,000/1,000 splits, batch 4 with gradient accumulation 4, a 4,096-token sequence budget, LoRA rank 16.

When it finishes, pull the report down:

```bash
uv run modal volume get sommelier-artifacts artifacts/runs/<run_id>/report/ report/
```

(Trailing slash matters: it tells Modal to download a directory.)

The local CLI variant, for a machine that already has the GPU stack: `uv run sommelier pipeline run --config examples/config.full.yaml --mode full --input <raw_rows.jsonl>`.

## 6. Reading the report

Each run directory (`artifacts/runs/<run_id>/`) contains:

- `report/comparison_report.json`: authoritative, machine-readable.
- `report/comparison_report.md`: human rendering with run identity, split summary, metric deltas, runtime and cost, and limitations.

Interpretation:

- **valid_json_rate**: share of generations that parsed into a schema-valid tool call. Parse failures count against every metric.
- **function_name_accuracy**: the parsed call names the gold function.
- **argument_exact_match**: arguments equal the gold arguments exactly (canonical JSON).
- **argument_f1**: micro-F1 over flattened argument key/value pairs.
- **full_call_exact_match**: name and arguments both match.
- **Deltas** are adapter minus base. The comparison is only written when base and adapter evaluations share the same config digest, test split digest, prompt set digest, parser version, and decoding settings; a mismatch fails instead of producing a misleading report.
- The **evidence class** line distinguishes smoke runs from full runs. Do not quote smoke numbers as reference results.

Exact metric definitions live in the [metrics reference](../reference/metrics.md), and the published numbers with their run identity live in [the reference run](../results/reference-run.md).

## 7. Caveats

- **License**: the base model carries NVIDIA Open Model License and Llama 3.1 Community License obligations, and the dataset is CC-BY-4.0; see [Licensing](../project/licensing.md). Derived adapters are not published by default.
- **Cost**: remote runs bill against your provider account. Observed cost in reports is evidence from the run, not a guarantee; when the provider exposes no billing data the report says `unavailable` explicitly.
- **Determinism**: the pipeline pins revisions, seeds, and decoding, and the comparison gate enforces identity between the two evaluation sides. Bit-identical generations across different GPUs and driver stacks are still not guaranteed; expect metrics within noise of the published run, verified against the same splits.
- **Limitations**: metrics cover schema-valid single tool calls on the configured held-out split. Exact-match scoring penalizes semantically equivalent but differently formatted arguments. Results do not imply production readiness or generalization beyond the evaluated split.
