<div align="center">

# 🍷 Sommelier

**Fine-tune small open models into reliable JSON tool callers. Reproducible, end to end, on a single GPU.**

[![CI](https://img.shields.io/github/actions/workflow/status/AbdelStark/sommelier/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/AbdelStark/sommelier/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/AbdelStark/sommelier/docs.yml?branch=main&style=for-the-badge&label=docs)](https://abdelstark.github.io/sommelier/)
[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?style=for-the-badge)](pyproject.toml)
[![Mypy](https://img.shields.io/badge/typed-mypy%20strict-2A6DB2?style=for-the-badge)](pyproject.toml)

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-LoRA%20adapter%20(en)-FFD21E?style=for-the-badge)](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-LoRA%20adapter%20(en%2Bfr)-FFD21E?style=for-the-badge)](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-fr-en-lora)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-dataset-FFD21E?style=for-the-badge)](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits)
[![Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-interactive%20tour-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/abdelstark/sommelier)
[![Compute](https://img.shields.io/badge/compute-Modal-7C3AED?style=for-the-badge)](https://modal.com)

**[Documentation](https://abdelstark.github.io/sommelier/)** · **[Interactive tour](https://huggingface.co/spaces/abdelstark/sommelier)** · **[Quickstart](https://abdelstark.github.io/sommelier/getting-started/quickstart/)** · **[The reference run](https://abdelstark.github.io/sommelier/results/reference-run/)** · **[The French run](https://abdelstark.github.io/sommelier/results/french-run/)**

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
prompt policy, parser, and decoding config. They are not claims of
production readiness or general agent reliability.

## 🌍 Multilingual result: closing the French gap

Tool calling should work as well in French as in English, and that is a
claim worth measuring rather than assuming. The v2 experiment built a
French paired variant of every selected row by **constrained
translation**: only the query is translated, tool schemas and gold
answers stay byte identical, every output is audited against the gold
argument values it must preserve, and each French row inherits its
English source's train/validation/test split, so a translation can
never leak across a split boundary.

Full-call exact match on the held-out test slices (en n=1,000,
fr n=879, same gold answers by construction, same parser and decoding):

| Model | en | fr | Gap |
|-------|----|----|-----|
| Base Nemotron-Nano-8B | 0.705 | 0.663 | -4.2 pts |
| v1 adapter (English-only training) | 0.874 | 0.851 | -2.3 pts |
| **v2 adapter (mixed en+fr training)** | 0.870 | **0.873** | **+0.3 pts** |

Three findings, measured rather than assumed:

1. **The base model pays about 4 points on French input.** Same tools,
   same gold calls, only the query language changed.
2. **English-only fine-tuning transfers most of its gains for free.**
   Without seeing a single French example, the v1 adapter lifts French
   full-call accuracy by 18.8 points and halves the language gap.
3. **A machine-translated training slice closes the gap to noise.**
   French now matches English on every metric to within a third of a
   point, at a cost of at most 0.8 points on English (within one
   standard error at n=1,000; the regression is reported, not hidden).

The whole comparison is anchored by digests: the English prompt set of
the v2 run is byte-identical to the reference run's, so the two results
are measured on exactly the same prompts. Setup, per-slice tables with
counts, runtime, and the boundaries the machine-translated slice adds
are on [the French run](https://abdelstark.github.io/sommelier/results/french-run/) page.

| Artifact | Where |
|----------|-------|
| 🤗 Bilingual adapter + evaluation evidence | [`abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-fr-en-lora`](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-fr-en-lora) |
| 🤗 French paired rows + translation provenance | [`abdelstark/sommelier-xlam-single-call-splits-fr`](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits-fr) |
| Translation method and drop accounting | [`sommelier data translate`](https://abdelstark.github.io/sommelier/reference/cli/#data-translate) |

## 🏗️ Pipeline architecture

One CLI, six stages, no hidden state. Stages communicate only through
schema-versioned files under an artifact root, and every transition writes a
manifest with input/output checksums.

```text
                         Salesforce/xlam-function-calling-60k
                                        │  (+ optional paired-language rows,
                                        ▼ raw_tool_call_row.v1   e.g. the French translation set)
                       ┌────────────────────────────────┐
                       │  1 · data prepare              │  validate rows · drop multi-call
                       │                                │  dedupe queries · split 15k/1k/1k
                       └───────────────┬────────────────┘  (a query lives in exactly one split;
                                       │                    paired rows inherit their root's split)
                                       │ prepared_example.v2 + drop_summary
                                       ▼
                       ┌────────────────────────────────┐
                       │  2 · format build              │  tokenizer chat template
                       │                                │  canonical tools JSON · prompt_sha256
                       └───────────────┬────────────────┘  sequence-length audit
                                       │ formatted_example.v2
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

- **Determinism as a contract.** Seeded splits, greedy decoding, pinned
  revisions, and digest-gated comparisons; drift fails loudly.
- **Completion-only training.** Prompt tokens are masked with a
  *proven* prompt/target token boundary; if a tokenizer merges across it,
  training refuses to fall back to full-sequence loss.
- **Conservative evaluation.** The parser extracts the first balanced
  JSON span and never repairs output; `no_json`, `invalid_json`, and
  `invalid_shape` all count against every metric.
- **Honest data policy.** 52.6% of xlam rows answer with more than one
  call; they are dropped via a declared filter (recorded in the drop
  summary) because the contract trains and scores exactly one call.
- **Byte-identical gold across languages.** A translated row must carry
  the exact tool schemas and gold answers of its English source, checked
  at preparation time against the exact source row, and it inherits that
  row's split; rows whose gold arguments cannot survive translation are
  dropped with counted reasons, not bent to fit.
- **Security by default.** Secrets live in the environment only; logs,
  manifests, and reports are redaction-scanned; releases are gated by
  `sommelier release preflight`.
- **GPU-free core.** `import sommelier` never touches torch/CUDA; heavy
  stacks stay behind optional extras and remote images (enforced in CI).

Each of these is a deliberate trade, and the docs record what was
rejected and why: [design decisions](https://abdelstark.github.io/sommelier/concepts/design-decisions/).

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

Everything above runs on a clean machine: no GPU, no accounts. The
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

# full bilingual run: build the French pairs first, then train on both
SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=14400 \
uv run modal run --detach remote_translate.py \
  --config examples/config.full.yaml --run-id fr-translate-2

SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=36000 \
uv run modal run --detach remote_pipeline.py \
  --config examples/config.full.yaml --mode full --max-rows 60000 \
  --translation-run-id fr-translate-2
```

## 🧰 CLI

| Command | Purpose |
|---------|---------|
| `sommelier config validate` | Strict config validation (unknown keys rejected) |
| `sommelier data prepare` | Validate, filter, dedupe, and split raw rows |
| `sommelier data translate` | Build a French paired dataset by constrained, audited translation |
| `sommelier data validate-fixtures` | Check the synthetic test fixtures |
| `sommelier format build` | Render chat templates + prompt digests (`--fixture` for no-tokenizer builds) |
| `sommelier eval run` | Deterministic generation + `evaluation_report.json` |
| `sommelier train run` | QLoRA adapter training with completion-only loss |
| `sommelier report compare` | Digest-gated comparison → JSON + Markdown reports |
| `sommelier pipeline run` | Chain all stages (`--mode smoke\|full`) |
| `sommelier serve adapter` | Optional single-adapter inference endpoint |
| `sommelier release preflight` | License, notices, acknowledgement, lock, and secret gates |

Expected errors map to documented
[exit codes](https://abdelstark.github.io/sommelier/reference/errors/)
(2 input · 3 dependency/license · 4 resource · 5 invariant); the full
command reference is in the
[docs](https://abdelstark.github.io/sommelier/reference/cli/).

## 🖥️ Serving (optional and illustrative)

`sommelier serve adapter` starts a single-adapter endpoint for manual
inspection. It is deliberately not a production serving system:
no production readiness, no autoscaling, no multi-tenant isolation, no
streaming.
It reuses the evaluation prompt policy and parser, so responses report
`parse_status` instead of repairing invalid output:

```bash
uv run sommelier serve adapter --config examples/config.full.yaml \
  --adapter <path-to-adapter-dir>

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in Paris today?"}],
    "tools": [{"name": "lookup_weather",
               "description": "Look up the current weather for a city.",
               "parameters": {"city": {"description": "Name of the city.", "type": "str"}}}],
    "temperature": 0.0,
    "max_tokens": 256
  }'
```

```json
{"raw_text": "[{\"arguments\":{\"city\":\"Paris\"},\"name\":\"lookup_weather\"}]",
 "parsed_call": {"name": "lookup_weather", "arguments": {"city": "Paris"}},
 "parse_status": "ok", "model_kind": "adapter"}
```

Tool schemas should use the xlam-style flat parameter map shown above
(`"parameters": {"<param>": {"description": …, "type": …}}`), the shape
the published adapter was trained on; JSON-Schema-style
`{"type": "object", "properties": …}` tools are out of distribution and
typically yield `invalid_json`. Requests are logged with their parse
status to `<adapter-dir>/../logs/serve.jsonl`. The core evaluation claim
never depends on serving.

### High-throughput serving on Modal (vLLM)

[`remote_serving.py`](remote_serving.py) deploys the published adapter
behind vLLM's OpenAI-compatible server on a Modal GPU (scales to zero
when idle). One deployment registers two models, the base
(`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`) and the LoRA
(`sommelier-tool-caller`), so base-vs-adapter A/B requests hit the same
endpoint:

```bash
uv run modal deploy remote_serving.py   # deploy (URL printed)
uv run modal run remote_serving.py      # smoke: canonical request, parsed
                                        # with sommelier's own parser
```

The adapter loads from the published Hugging Face repo by default;
`SOMMELIER_ADAPTER_VOLUME_PATH` serves one straight from a pipeline run
on the artifacts volume instead. Set `SOMMELIER_SERVE_API_KEY` in `.env`
to require a Bearer token; without it the URL is open, so treat it as
the illustrative endpoint it is. Cold starts take a few minutes; the
smoke entrypoint polls readiness before asserting.

## 📚 Documentation

Full documentation lives at
**[abdelstark.github.io/sommelier](https://abdelstark.github.io/sommelier/)**:
concepts, guides, and a complete reference for every command, config
field, artifact schema, error code, and metric.

Key entry points:

| Document | Contents |
|----------|----------|
| [Quickstart](https://abdelstark.github.io/sommelier/getting-started/quickstart/) | The fixture pipeline on a laptop, no GPU or accounts |
| [Reproduction guide](docs/guides/reproduction.md) | Clean checkout to full reproduction, with caveats |
| [The reference run](https://abdelstark.github.io/sommelier/results/reference-run/) | The published result, its evidence, and its boundaries |
| [The French run](https://abdelstark.github.io/sommelier/results/french-run/) | The multilingual result: the language gap measured and closed |
| [v1.0 release checklist](docs/release/v1.0-checklist.md) | Every release blocker mapped to machine-checkable evidence |
| [Changelog](CHANGELOG.md) | Categorized, migration-noted history |

## 🛠️ Development

```bash
uv run ruff check .
uv run mypy sommelier tests
uv run pytest
uv run sommelier data validate-fixtures
```

The suite (380+ tests) includes golden prompt fixtures that fail on any
template drift, parser/metric regression snapshots, split-leakage property
tests, secret-hygiene checks, and an import-discipline gate that walks every
module in a clean interpreter. Optional GPU coarse filtering is available
with `uv sync --extra data-gpu` and the `--gpu` flag on
`sommelier data prepare`; preparing real rows with `--input` requires at
least as many valid deduplicated rows as the configured split sizes.

The documentation site builds locally with
`uv sync --extra docs && uv run mkdocs serve`; CI builds it strict on
every pull request and deploys it to GitHub Pages on merge.

## ⚖️ License

Code is [MIT](LICENSE). Third-party model, dataset, and package obligations
are recorded in [licenses/THIRD_PARTY.md](licenses/THIRD_PARTY.md) and
enforced by `sommelier release preflight`. The published adapters are **Built
with Llama** and subject to the NVIDIA Open Model License and the Llama 3.1
Community License; the published datasets are CC-BY-4.0 (derivatives of
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
