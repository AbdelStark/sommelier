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
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-LoRA%20adapter%20(en%2Bhe)-FFD21E?style=for-the-badge)](https://huggingface.co/abdelstark/Llama-3.1-Nemotron-Nano-8B-xlam-tool-calling-he-en-hymt-lora)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-dataset-FFD21E?style=for-the-badge)](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits)
[![Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-interactive%20tour-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/abdelstark/sommelier)
[![Compute](https://img.shields.io/badge/compute-Modal-7C3AED?style=for-the-badge)](https://modal.com)

**[Documentation](https://abdelstark.github.io/sommelier/)** · **[Interactive tour](https://huggingface.co/spaces/abdelstark/sommelier)** · **[Quickstart](https://abdelstark.github.io/sommelier/getting-started/quickstart/)** · **[The reference run](https://abdelstark.github.io/sommelier/results/reference-run/)** · **[The French run](https://abdelstark.github.io/sommelier/results/french-run/)** · **[The Hebrew Hy-MT2 run](https://abdelstark.github.io/sommelier/results/hebrew-hymt/)**

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

The public artifacts support checking the reported aggregate metrics and
reproducing the recorded protocol. They do not support independent row-by-row
rescoring because the raw generation files retained by the original runs are
not present in the published v1/v2 repositories:

| Artifact | Where |
|----------|-------|
| 🤗 Adapter + evaluation evidence | [`abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora`](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora) |
| 🤗 Exact train/val/test splits | [`abdelstark/sommelier-xlam-single-call-splits`](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits) |
| Reproduction commands | [docs/guides/reproduction.md](docs/guides/reproduction.md) |

The published reports expose metric numerators, denominators, configuration,
and identity digests, so the displayed aggregate rates can be checked without
trusting rounded floats. Reproducing the run can regenerate row-level evidence;
the existing public artifacts alone cannot independently rescore each row.

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

These are the historical v2 **marginal full-slice** figures: all 1,000
English rows versus the 879 translations that survived the translation
contract. Because those cohorts differ, v3 reports exact root-matched gaps
and paired-bootstrap confidence intervals as the primary estimate, while
retaining full-slice gaps under an explicit descriptive label.

Three descriptive findings from those marginal slices:

1. **The base model's marginal French slice is about 4 points lower.**
   Exact pairs keep tools and gold calls fixed, but this published table also
   includes unmatched English roots.
2. **English-only fine-tuning transfers much of its gain.**
   Without seeing a single French example, the v1 adapter lifts French
   full-call accuracy by 18.8 points and halves the language gap.
3. **The mixed adapter's marginal slice values nearly coincide.**
   French and English differ by at most a third of a point in this table,
   while English is up to 0.8 points below v1. The v2 artifact does not carry
   the paired interval needed to call either difference noise.

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

This v2 result remains historical evidence, not a claim of strict
reproducibility under today's stronger publication gate. Its French dataset
commit has a v1 translation summary and no current publication manifest; a
new strict French run requires a provenance-complete republish. The default
`examples/config.full.yaml` is therefore English-only.

## 🇮🇱 Hebrew v3: teacher selected, full results pending

The v3 experiment extends the paired design to Hebrew and adds the cost
measurement that multilingual fine-tuning discussions often omit. It will
report the observed English↔Hebrew token inflation under the pinned base
tokenizer, the English-only workload counterfactual, additive Hebrew data/token
tax, combined-vs-English multipliers, and exact matched-pair accuracy gaps. The
counterfactual is derived from the same formatted rows and epochs, not a
separately trained runtime or accuracy arm. A gated three-arm comparison—base
model, published v1 English adapter, and v3 English+Hebrew QLoRA adapter—will
test Hebrew uplift while enforcing a predeclared English non-inferiority margin.

The one-time dataset teacher is the exact dated OpenAI Responses snapshot
`gpt-5.5-2026-04-23`, under the `instruction_chat` contract with 512 maximum
output tokens, explicit Flex service, OpenAI SDK 2.45.0, a 900-second
per-request timeout, three audited row attempts, eight provider workers, and
32-row durable checkpoints. SDK retries stay at zero. Exact Flex HTTP 429
`resource_unavailable` responses use separately attributed same-row
`provider_call_attempt` retries after fixed 1/2/4/8/16-second delays, with no
tier switch. Paid launches also require a local public-list-price ceiling:
`1.00` USD for smoke or `50.00` USD for full. It is not an invoice or provider
account/project cap. The producer is a CPU-only Modal image. Protected
literals are replaced with deterministic ASCII placeholders before the
provider request, restored afterward, and audited byte for byte; tools and gold
answers never enter the translation output and remain
byte-identical. The API snapshot is an identity, not a public weight digest or
a promise of byte-identical provider regeneration. The external teacher is
limited to dataset creation; training, evaluation, adapter weights, and
deployment remain on the pinned open Nemotron/Llama-derived stack.

Before that full provider run, the Phase-A config must commit one named human
reviewer's stable id, canonical comment-free Ed25519 public key, and matching
OpenSSH SHA-256 fingerprint. The private key remains solely with the reviewer:
it must never enter the repository, Modal, Sommelier, or Codex. The producer
exclusively reserves `translation_run_identity.json` before dataset or provider
access, and the published paired dataset includes that identity plus the exact
Phase-A config bytes as `translation_config.yaml`.

A 140-row Flex smoke accepted 140/140 rows after 143 provider requests. Its
model-assisted, non-native diagnostic inspection—not independent human
review—labeled 127 clean, 13 minor, and zero hard semantic errors. Provider
usage was 73,359 input tokens, zero cached input tokens, 11,618 output tokens,
zero reasoning tokens, and 84,977 total tokens. Applying the pinned public list
prices gives **$0.357667500**; this is a calculated estimate, not an invoice or
observed billing. That smoke used a 256-token output limit and the historical
v1 journal/evidence contract, so it selected the provider but did not validate
the final 512-token/v2 producer above. It also came from a dirty worktree and is
diagnostic only: it is not the full corpus, native-speaker validation, a Hebrew
accuracy result, or QLoRA/TCO evidence.

Full evidence does not consume a mutable translation run. The audited
machine-translated survivor corpus and provenance files must first be published
together, then loaded from an immutable Hugging Face dataset commit. An
independent pinned Helsinki-NLP model back-translates the locked review sample,
but no native-speaker review has been completed. The run page documents the
commands, claim gates, provider evidence, and placeholders that must be filled
from checksummed artifacts: [Hebrew v3 methodology](docs/results/hebrew-v3.md).
Pipeline TCO keeps observed billing separate from deterministic projections,
and no full-fine-tuning saving is claimed without a matched full-parameter arm.

## 🇮🇱 Hebrew local-MT result: Hy-MT2

A separate, honestly scoped experiment used Tencent Hy-MT2 locally through
Ollama to translate the selected xLAM queries into Hebrew. It accepted 16,272
of 17,000 roots. In the exact training snapshot, tools and gold answers remained
byte-identical; the public release sanitizes 15 credential-shaped synthetic
tool literals and is not byte-identical. The shared prepare gate retained 14,286
Hebrew training examples and 945 held-out pairs. This is a machine-translated
survivor corpus with no human semantic review, so it is not the preregistered v3
result above.

QLoRA on one Modal L40S raised held-out full-call exact match on the Hebrew
slice from 410/945 (43.39%) to 786/945 (83.17%), a +39.79-point paired gain
(95% CI +36.61 to +42.96). On the 1,000 English prompts, exact match rose from
70.50% to 88.40%. The adapter's exact matched Hebrew-minus-English gap was
-5.19 points, compared with -27.62 points for the base model.

| Artifact | Where |
|----------|-------|
| 🤗 English+Hebrew adapter + benchmark evidence | [`abdelstark/Llama-3.1-Nemotron-Nano-8B-xlam-tool-calling-he-en-hymt-lora`](https://huggingface.co/abdelstark/Llama-3.1-Nemotron-Nano-8B-xlam-tool-calling-he-en-hymt-lora) |
| 🤗 Hy-MT2 Hebrew paired rows + sanitization accounting | [`abdelstark/sommelier-xlam-single-call-splits-he-hymt-sanitized`](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits-he-hymt-sanitized) |
| Full setup, tokenization tax, runtime, metrics, and limitations | [Hebrew Hy-MT2 run](docs/results/hebrew-hymt.md) |

## 🏗️ Pipeline architecture

One CLI, seven stages, no hidden state. Stages communicate only through
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
                       └───────────────┬────────────────┘
                                       │ formatted_example.v2
                                       ▼
                       ┌──────────────────────────────┐
                       │  3 · analyze tokenization      │  exact root↔translation ratios
                       │                                │  workload · sequence-budget gate
                       └───────────────┬────────────────┘
                                       │ tokenizer_tax_report.v1
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
      ┌──────────────────────────┐        ┌──────────────────────────┐
      │  4 · eval run (base)     │        │  5 · train run           │
      │  greedy · temp 0         │        │  QLoRA · NF4 4-bit       │
      │  conservative parser     │        │  completion-only loss    │
      └────────────┬─────────────┘        │  provable prompt boundary│
                   │                      └────────────┬─────────────┘
                   │ generations.jsonl                 │ adapter/ + training_metrics.jsonl
                   │ evaluation_report.json            ▼
                   │                      ┌──────────────────────────┐
                   │                      │  6 · eval run (adapter)  │
                   │                      │  same prompts · parser   │
                   │                      │  and decoding, by digest │
                   │                      └────────────┬─────────────┘
                   ▼                                   ▼
      ┌─────────────────────────────────────────────────────────────┐
      │  7 · report compare ····································    │
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
  ├── data/ formatted/ analysis/       manifests, never silent partial
  ├── train/ eval/
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
  manifests, and reports are redaction-scanned. Adapter releases require an
  identity-bound preflight, and both publishers enforce exact artifact-specific
  validation before any explicit mutation.
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
smoke and full runs, and how to read the report. Hebrew v3 translation adds
named OpenAI/Hugging Face Modal secrets and an explicit paid-provider launch;
those requirements are documented separately on its methodology page.

### Run the whole pipeline on a GPU

[`remote_pipeline.py`](remote_pipeline.py) executes all seven stages on
[Modal](https://modal.com), exporting the dataset in-container, caching
weights on a volume, and committing artifacts stage by stage:

```bash
# bounded smoke run (≤100/20/20 examples, separate smoke- run namespace)
uv run modal run remote_pipeline.py \
  --config examples/config.smoke.yaml --mode smoke --max-rows 2500

# English-only v1/default full run (15,000/1,000/1,000)
SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=86400 \
uv run modal run --detach remote_pipeline.py \
  --config examples/config.full.yaml --mode full --max-rows 60000
```

The default full config has no paired source and needs no translation staging.
Paired smoke runs may stage a completed diagnostic translation with
`--translation-run-id`; paired full runs reject that override and verify the
translation summary, pre-provider run identity, exact Phase-A config, locked
semantic-review template, finalized signed review, and publication manifest at
the pinned dataset revision. The exact paired
translation and three-arm commands are in the
[Hebrew v3 methodology](docs/results/hebrew-v3.md).

## 🧰 CLI

| Command | Purpose |
|---------|---------|
| `sommelier config validate` | Strict config validation (unknown keys rejected) |
| `sommelier data prepare` | Validate, filter, dedupe, and split raw rows |
| `sommelier data translate` | Build a French or Hebrew paired dataset by constrained, audited translation |
| `sommelier data semantic-review-create` | Lock and back-translate the preregistered Hebrew review sample |
| `sommelier data semantic-review-attestation-create` | Recompute the sample and create the canonical payload for the named human to sign |
| `sommelier data semantic-review-finalize` | Validate reviewer decisions and bind the publication manifest |
| `sommelier data validate-fixtures` | Check the synthetic test fixtures |
| `sommelier format build` | Render chat templates + prompt digests (`--fixture` for no-tokenizer builds) |
| `sommelier analyze tokenization` | Measure exact matched-pair tokenizer cost and enforce the sequence budget |
| `sommelier eval run` | Deterministic generation + `evaluation_report.json` |
| `sommelier train run` | QLoRA adapter training with completion-only loss |
| `sommelier report compare` | Digest-gated comparison → JSON + Markdown reports |
| `sommelier report experiment` | Gate base/v1/v3 Hebrew uplift and English non-inferiority claims |
| `sommelier pipeline run` | Chain all stages (`--mode smoke\|full`) |
| `sommelier serve adapter` | Optional single-adapter inference endpoint |
| `sommelier release preflight` | License/secret gates plus config, revision, source, lock, and tree identity |
| `sommelier release publish-dataset` | Validate by default; explicitly publish and round-trip-verify the audited Hebrew dataset |
| `sommelier release publish-adapter` | Validate by default; explicitly publish and round-trip-verify the evidence-bound adapter |

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
are recorded in [licenses/THIRD_PARTY.md](licenses/THIRD_PARTY.md). The
preflight checks the project-level model/root-dataset subset; the publishers
enforce the remaining artifact-specific contracts. Published adapters are **Built
with Llama** and subject to the NVIDIA Open Model License and the Llama 3.1
Community License. Reviewed agreement copies and the lineage attribution are
tracked in `licenses/LICENSE-NVIDIA-OPEN-MODEL.txt`,
`licenses/LICENSE-LLAMA-3.1.txt`, and `licenses/NOTICE`; the adapter publisher
requires their exact bytes. Published datasets are CC-BY-4.0 (derivatives of
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
