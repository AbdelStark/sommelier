# Reproduce the reference run

This guide walks from a clean checkout to a full reproduction of the published result and explains how to read the report it produces. Commands are exact. Steps that need external accounts or hardware say so up front, because renting the GPU costs money and you should know before you start.

## 1. Install

Prerequisites: Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AbdelStark/sommelier
cd sommelier
uv sync --extra dev
```

The base install includes the lightweight Modal client used to launch remote work,
but it does not install GPU training/inference libraries or tracking SDKs; those
live in remote images or optional extras. [Installation](../getting-started/installation.md)
has the details.

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

[`remote_pipeline.py`](https://github.com/AbdelStark/sommelier/blob/main/remote_pipeline.py) exports the pinned dataset revision to raw rows inside the container, then chains all seven stages and commits artifacts to the `sommelier-artifacts` volume after each stage. The tokenization stage audits every rendered train and validation sequence against the training budget and persists exact per-language and matched-pair token counts, so a budget mistake costs seconds rather than GPU hours. [Run the pipeline on Modal](remote-execution.md) explains the moving parts.

If you have your own GPU machine with the training stack installed, the same chain runs locally through the CLI: `uv run sommelier pipeline run --config examples/config.smoke.yaml --mode smoke --input <raw_rows.jsonl>`.

## 5. Full run

`examples/config.full.yaml` is the runnable English-only v1/default full
configuration. It pins the base model, tokenizer, and root dataset revisions,
uses the 15,000/1,000/1,000 split scale, and contains no paired source. Its
configured sequential planning estimate is 45,000 seconds (1,800 data +
28,800 train + two 7,200-second evaluation arms), so the launch below satisfies
the outer-timeout admission gate. These legacy-named stage values are not
enforced watchdogs; only the Modal function's outer timeout is enforced:

```bash
SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=86400 \
uv run modal run --detach remote_pipeline.py \
  --config examples/config.full.yaml --mode full --max-rows 60000
```

The historical v1 run took about 3.7 hours end to end on one L40S (10,996 s
of training, about 29 minutes of evaluation). A new launch writes a new run
identity and evidence; the historical run's resolved config and checksummed
split/report artifacts remain authoritative for the published numbers.

The [French v2 result](../results/french-run.md) is a separate historical
evidence record and does **not** currently have a strict reproduction path.
Its published paired-dataset commit contains a v1 translation summary and no
current `translation_publication.json`, so the full driver rejects that source.
`config.full.yaml` intentionally excludes it. A new strict French full run
requires a provenance-complete immutable republish and a separate config
pinned to that commit; direct `--translation-run-id` staging remains
smoke-only. No such migrated French revision is claimed here.

The pending paired Hebrew contract and its publication sequence are in the
[Hebrew v3 methodology](../results/hebrew-v3.md). Its provisional dataset
revision must likewise be replaced by a provenance-complete immutable commit
before that full command is runnable. Building that dataset is a separate paid
CPU/provider step: it requires the named Modal secrets `openai-api-key` and
`huggingface-read-token` and explicit `--runtime-backend openai_responses`; the
dated model name alone never authorizes the call. The selected Flex contract
uses a 900-second request timeout, zero SDK retries, and an explicit local
public-list-price ceiling (`1.00` USD for smoke or `50.00` USD for full). Three
row attempts remain the semantic/audit retry boundary. Exact Flex HTTP 429
`resource_unavailable` responses instead retry the same row attempt as
journaled provider-call attempts after fixed 1/2/4/8/16-second delays, without
switching tier. The ceiling is not an invoice or provider account/project cap.
The semantic-review machine template
must be produced from the same clean Git SHA as the full translation under
exactly Python 3.13.3, torch 2.11.0, transformers 5.13.1, accelerate 1.14.0,
and sentencepiece 0.2.2. Reviewers edit a separate copy; the untouched
template, reviewed copy, and finalized output must be three distinct files
(including distinct inodes). The [remote guide](remote-execution.md) gives the
producer command and the [CLI reference](../reference/cli.md#data-semantic-review-finalize)
gives the finalization flags.

When it finishes, pull the report down:

```bash
uv run modal volume get sommelier-artifacts artifacts/runs/<run_id>/report/ report/
```

(Trailing slash matters: it tells Modal to download a directory.)

On a machine with the training stack installed, the English-only chain can run
locally as
`uv run sommelier pipeline run --config examples/config.full.yaml --mode full --input <raw_rows.jsonl>`.
The local CLI enforces the same paired-source provenance boundary; supplying
raw root rows does not turn the legacy French publication into current full
evidence.

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
