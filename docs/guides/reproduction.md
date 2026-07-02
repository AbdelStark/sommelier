# Reproduction Guide

This guide walks from a clean checkout to a full reproduction run and
explains how to read the resulting report. Commands are exact; steps that
need external accounts or hardware say so up front.

## 1. Install

Prerequisites: Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AbdelStark/sommelier
cd sommelier
uv sync --extra dev
```

The base install never pulls GPU, remote execution, or tracking packages;
those live behind optional extras and remote images.

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

To prepare real rows instead of fixtures, pass
`--input <raw_rows.jsonl>` with `sommelier.raw_tool_call_row.v1` records;
the input must contain at least as many valid deduplicated rows as the
configured split sizes require, or preparation exits 2 with the shortfall.

Evaluation, training, and serving need the model stack (torch,
transformers, peft); locally they fail with an actionable
`ExternalDependencyError` (exit 3) by design.

## 3. Remote prerequisites

Reproducing the reference result requires:

- A [Modal](https://modal.com) account with billing enabled
  (`uv run modal token new`). Remote GPU time costs money; you are
  responsible for the spend.
- A Hugging Face token (`HF_TOKEN`) with access to the configured base
  model, provided through the environment or the Modal secret store —
  never through config files.
- Acknowledgement of the base model license terms:

```bash
export SOMMELIER_ACK_BASE_MODEL_LICENSE="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
uv run sommelier release preflight --config examples/config.full.yaml
```

The preflight verifies the project license, third-party notices
([licenses/THIRD_PARTY.md](../../licenses/THIRD_PARTY.md)), the recorded
model and dataset obligations, and scans artifacts for secrets before
anything is published.

## 4. Smoke run

The smoke mode bounds split sizes (at most 100 train / 20 validation /
20 test examples) and writes into a separate `smoke-` run namespace, so a
later full run can never overwrite it:

```bash
uv run sommelier pipeline run --config examples/config.full.yaml \
  --mode smoke --input <raw_rows.jsonl>
```

`--input` takes `sommelier.raw_tool_call_row.v1` JSONL rows. The pipeline
chains data preparation, formatting, baseline evaluation, adapter
training, adapter evaluation, and the comparison report.

## 5. Full run

```bash
uv run sommelier pipeline run --config examples/config.full.yaml \
  --mode full --input <raw_rows.jsonl>
```

The full run uses the configured split sizes (15000/1000/1000 by default)
and the configured GPU (`remote.gpu`). Runtime and cost depend on the
provider; the report records observed evidence only.

## 6. Reading the report

Each run directory (`artifacts/runs/<run_id>/`) contains:

- `report/comparison_report.json` — authoritative, machine-readable.
- `report/comparison_report.md` — human rendering with run identity,
  split summary, metric deltas, runtime/cost, and limitations.

Interpretation:

- **valid_json_rate** — share of generations that parsed into a
  schema-valid tool call. Parse failures count against every metric.
- **function_name_accuracy** — parsed call names the gold function.
- **argument_exact_match** — arguments equal the gold arguments exactly
  (canonical JSON).
- **argument_f1** — micro-F1 over flattened argument key/value pairs.
- **full_call_exact_match** — name and arguments both match.
- **Deltas** are adapter minus base. The comparison is only written when
  base and adapter evaluations share the same config digest, test split
  digest, prompt set digest, parser version, and decoding settings; a
  mismatch fails instead of producing a misleading report.
- The **evidence class** line distinguishes smoke runs from full runs.
  Do not quote smoke numbers as reference results.

## 7. Caveats

- **License**: the base model carries NVIDIA Open Model License and
  Llama 3.1 Community License obligations, and the dataset is CC-BY-4.0;
  see [licenses/THIRD_PARTY.md](../../licenses/THIRD_PARTY.md). Derived
  adapters are not published by default.
- **Cost**: remote runs bill against your provider account. Observed cost
  in reports is evidence from the run, not a guarantee; when the provider
  exposes no billing data the report says `unavailable` explicitly.
- **Limitations**: metrics cover schema-valid single tool calls on the
  configured held-out split. Exact-match scoring penalizes semantically
  equivalent but differently formatted arguments. Results do not imply
  production readiness or generalization beyond the evaluated split.
