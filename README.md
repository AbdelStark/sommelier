<div align="center">

# 🍷 Sommelier

**Fine-tune small open models into reliable JSON tool callers — reproducibly, end to end, on a single GPU.**

[![CI](https://img.shields.io/github/actions/workflow/status/AbdelStark/sommelier/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/AbdelStark/sommelier/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?style=for-the-badge)](pyproject.toml)
[![Mypy](https://img.shields.io/badge/typed-mypy%20strict-2A6DB2?style=for-the-badge)](pyproject.toml)

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-LoRA%20adapter-FFD21E?style=for-the-badge)](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-dataset-FFD21E?style=for-the-badge)](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits)
[![Compute](https://img.shields.io/badge/compute-Modal-7C3AED?style=for-the-badge)](https://modal.com)

</div>

Sommelier is a reference implementation for adapting a small open language
model to emit **exactly one schema-valid JSON tool call** per request. It is
built around one idea: *a fine-tuning claim is only as good as its evidence.*
Every stage writes schema-versioned, checksummed artifacts; prompts carry
digests; evaluation is deterministic; and the final base-vs-adapter
comparison is **refused** unless both sides provably used identical data,
prompts, parser, and decoding.

## 📊 Reference result

QLoRA on [nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1),
15,000 training examples, 2 epochs, one L40S (~3 h training). Evaluated
greedily on 1,000 held-out prompts with a conservative parser that counts
every parse failure as a failure:

| Metric | Base | Adapter | Δ |
|--------|------|---------|---|
| Valid JSON rate | 0.916 | **1.000** | +0.084 |
| Function name accuracy | 0.911 | **0.996** | +0.085 |
| Argument exact match | 0.707 | **0.876** | +0.169 |
| Argument F1 | 0.757 | **0.929** | +0.172 |
| Full-call exact match | 0.705 | **0.874** | +0.169 |

Everything needed to verify or reproduce this row-for-row is public:

| Artifact | Where |
|----------|-------|
| 🤗 Adapter + evaluation evidence | [`abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora`](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora) |
| 🤗 Exact train/val/test splits | [`abdelstark/sommelier-xlam-single-call-splits`](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits) |
| Reproduction commands | [docs/guides/reproduction.md](docs/guides/reproduction.md) |

Claim boundaries: these numbers hold for the recorded dataset revision,
prompt policy, parser, and decoding config — they are not claims of
production readiness or general agent reliability.

## 🏗️ Pipeline architecture

One CLI, six stages, no hidden state — stages communicate only through
schema-versioned files under an artifact root, and every transition writes a
manifest with input/output checksums.

```text
                         Salesforce/xlam-function-calling-60k
                                        │
                                        ▼ raw_tool_call_row.v1
                       ┌────────────────────────────────┐
                       │  1 · data prepare              │  validate rows · drop multi-call
                       │                                │  dedupe queries · split 15k/1k/1k
                       └───────────────┬────────────────┘  (a query lives in exactly one split)
                                       │ prepared_example.v1 + drop_summary
                                       ▼
                       ┌────────────────────────────────┐
                       │  2 · format build              │  tokenizer chat template
                       │                                │  canonical tools JSON · prompt_sha256
                       └───────────────┬────────────────┘  sequence-length audit
                                       │ formatted_example.v1
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
      ┌──────────────────────────┐        ┌──────────────────────────┐
      │  3 · eval run (base)     │        │  4 · train run           │
      │  greedy · temp 0         │        │  QLoRA · NF4 4-bit       │
      │  conservative parser     │        │  completion-only loss    │
      └────────────┬─────────────┘        │  provable prompt boundary│
                   │                      └────────────┬─────────────┘
                   │ generations.jsonl                 │ adapter/ + training_metrics.jsonl
                   │ evaluation_report.json            ▼
                   │                      ┌──────────────────────────┐
                   │                      │  5 · eval run (adapter)  │
                   │                      │  same prompts · parser   │
                   │                      │  and decoding, by digest │
                   │                      └────────────┬─────────────┘
                   ▼                                   ▼
      ┌─────────────────────────────────────────────────────────────┐
      │  6 · report compare ····································    │
      │  THE COMPARISON GATE: refuses mismatched config, test-split,│
      │  prompt-set, parser, or decoding digests (INV-DATA-006)     │
      └─────────────────────────────┬───────────────────────────────┘
                                    ▼
                     comparison_report.json (authoritative)
                     comparison_report.md   (human rendering)
```

```text
  artifacts/runs/<run_id>/            every stage → a manifest with
  ├── config.resolved.yaml            sha256-checksummed inputs/outputs;
  ├── manifest.json                   failures land as explicit failed
  ├── data/ formatted/ train/ eval/   manifests, never silent partial
  ├── report/                         state; logs and reports pass a
  └── runtime_metadata.json           secret-redaction scanner
```

## ✨ What makes it strict

- **Determinism as a contract** — seeded splits, greedy decoding, pinned
  revisions, and digest-gated comparisons; drift fails loudly.
- **Completion-only training** — prompt tokens are masked with a
  *proven* prompt/target token boundary; if a tokenizer merges across it,
  training refuses to fall back to full-sequence loss.
- **Conservative evaluation** — the parser extracts the first balanced
  JSON span and never repairs output; `no_json`, `invalid_json`, and
  `invalid_shape` all count against every metric.
- **Honest data policy** — ~52% of xlam rows are multi-call; they are
  dropped via a declared filter (recorded in the drop summary) because the
  v1 contract trains and scores exactly one call.
- **Security by default** — secrets live in the environment only; logs,
  manifests, and reports are redaction-scanned; releases are gated by
  `sommelier release preflight`.
- **GPU-free core** — `import sommelier` never touches torch/CUDA; heavy
  stacks stay behind optional extras and remote images (enforced in CI).

## 🚀 Install and Quickstart

Prerequisites: Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AbdelStark/sommelier
cd sommelier
uv sync --extra dev
uv run pytest
uv run sommelier config validate --config examples/config.smoke.yaml
uv run sommelier data prepare --config examples/config.smoke.yaml --fixture \
  --out examples/artifacts/runs/local/data --run-id local
uv run sommelier format build --config examples/config.smoke.yaml \
  --data examples/artifacts/runs/local/data \
  --out examples/artifacts/runs/local/formatted --run-id local --fixture
```

Everything above runs on a clean machine — no GPU, no accounts. The
[reproduction guide](docs/guides/reproduction.md) covers the remote
prerequisites (Modal account, `HF_TOKEN`, license acknowledgement), the
smoke and full runs, and how to read the report.

### Run the whole pipeline on a GPU

[`remote_pipeline.py`](remote_pipeline.py) executes all six stages on
[Modal](https://modal.com), exporting the dataset in-container, caching
weights on a volume, and committing artifacts stage by stage:

```bash
# bounded smoke run (≤100/20/20 examples, separate smoke- run namespace)
uv run modal run remote_pipeline.py \
  --config examples/config.smoke.yaml --mode smoke --max-rows 2500

# full reference run
SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=28800 \
uv run modal run --detach remote_pipeline.py \
  --config examples/config.full.yaml --mode full --max-rows 60000
```

## 🧰 CLI

| Command | Purpose |
|---------|---------|
| `sommelier config validate` | Strict config validation (unknown keys rejected) |
| `sommelier data prepare` | Validate, filter, dedupe, and split raw rows |
| `sommelier data validate-fixtures` | Check the synthetic test fixtures |
| `sommelier format build` | Render chat templates + prompt digests (`--fixture` for no-tokenizer builds) |
| `sommelier eval run` | Deterministic generation + `evaluation_report.json` |
| `sommelier train run` | QLoRA adapter training with completion-only loss |
| `sommelier report compare` | Digest-gated comparison → JSON + Markdown reports |
| `sommelier pipeline run` | Chain all stages (`--mode smoke\|full`) |
| `sommelier serve adapter` | Optional single-adapter inference endpoint |
| `sommelier release preflight` | License, notices, acknowledgement, lock, and secret gates |

Command names and flags follow the
[public API spec](docs/spec/02-public-api.md#cli-contract). Expected errors
map to documented exit codes (2 input · 3 dependency/license · 4 resource ·
5 invariant).

## 🖥️ Serving (optional and illustrative)

`sommelier serve adapter` starts a single-adapter endpoint for manual
inspection. It is deliberately not a production serving system:
no production readiness, no autoscaling, no multi-tenant isolation, no
streaming ([RFC-0010](docs/rfcs/RFC-0010-optional-inference-service.md)).
It reuses the evaluation prompt policy and parser, so responses report
`parse_status` instead of repairing invalid output:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in Paris today?"}],
    "tools": [{"name": "lookup_weather", "description": "Look up weather for a city.",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
    "temperature": 0.0,
    "max_tokens": 256
  }'
```

The core evaluation claim never depends on serving.

## 📚 Documentation

| Document | Contents |
|----------|----------|
| [Product requirements](prd.md) | What v1.0 is and is not |
| [Specification index](SPEC.md) | The full spec corpus + [RFC index](SPEC.md#rfc-index) |
| [Detailed specification](docs/spec/00-overview.md) | Architecture, data model, error model, security, testing |
| [Reproduction guide](docs/guides/reproduction.md) | Clean checkout → full reproduction, with caveats |
| [v1.0 release checklist](docs/release/v1.0-checklist.md) | Every release blocker mapped to machine-checkable evidence |
| [Changelog](CHANGELOG.md) | Categorized, migration-noted history |

## 🛠️ Development

```bash
uv run ruff check .
uv run mypy sommelier tests
uv run pytest
uv run sommelier data validate-fixtures
```

The suite (300+ tests) includes golden prompt fixtures that fail on any
template drift, parser/metric regression snapshots, split-leakage property
tests, secret-hygiene checks, and an import-discipline gate that walks every
module in a clean interpreter. Optional GPU coarse filtering is available
with `uv sync --extra data-gpu` and the `--gpu` flag on
`sommelier data prepare`; preparing real rows with `--input` requires at
least as many valid deduplicated rows as the configured split sizes.

## ⚖️ License

Code is [MIT](LICENSE). Third-party model, dataset, and package obligations
are recorded in [licenses/THIRD_PARTY.md](licenses/THIRD_PARTY.md) and
enforced by `sommelier release preflight`. The published adapter is **Built
with Llama** and subject to the NVIDIA Open Model License and the Llama 3.1
Community License; the published dataset is CC-BY-4.0 (derivative of
Salesforce/xlam-function-calling-60k).

## 📖 Citation

```bibtex
@software{sommelier2026,
  author = {Bakhta, Abdelhamid},
  title = {Sommelier: reproducible tool-calling fine-tuning for small open models},
  year = {2026},
  url = {https://github.com/AbdelStark/sommelier}
}
```
