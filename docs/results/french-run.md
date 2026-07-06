# The French run

This page states the v2 result the way the [reference run](reference-run.md) page does: the exact setup, the numbers with their numerators and denominators, the run identity that makes the comparison verifiable, and the boundaries of the claim. The run is `nemotron-8b-fr-full-4`, executed 2026-07-06 on a single L40S. Everything below comes from the run's own comparison report and runtime metadata.

The question this run answers is the one the reference run left open: tool calling should work as well in French as in English, and that is a claim worth measuring rather than assuming.

## Run identity

| Field | Value |
|-------|-------|
| Run ID | `nemotron-8b-fr-full-4` |
| Evidence class | full run |
| Created at | `2026-07-06T09:06:44.927671+00:00` |
| Config digest | `87a6c067167d801d85cbf3105d08b145c62955f4298de0559a0c44608e3fca89` |
| Test split digest | `11267f2e2e6293b6132a1a955b28a84caa12b15c980a2a39653d4e4ee33d80e9` |
| Prompt set digest, `en` slice | `a0da8fa28835a329dba5c5314ada3aff21f950939af2e1ae186155d3b494f39a` |
| Prompt set digest, `fr` slice | `b6111339b6dd6d6a911aeea4e0bf960bd3d78cd68ed14a1da05403aeacb76783` |
| Parser version | `sommelier.parser.v1` |
| Decoding | `{"do_sample": false, "max_new_tokens": 512, "temperature": 0.0}` |

One digest deserves a sentence: the `en` prompt set digest is identical to the reference run's, so the English numbers below were measured on byte-for-byte the same 1,000 prompts as the v1 result they are compared against.

## Setup

**One changed variable.** Same base model, same hyperparameters, same pipeline, same seed as the [reference run](reference-run.md). The training data adds a French paired variant of every selected row: [abdelstark/sommelier-xlam-single-call-splits-fr](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits-fr), built by constrained translation (only the query is translated; tool schemas and gold answers stay byte identical, enforced at preparation time as described in [Data policy](../concepts/data.md)). Training saw 15,000 English plus 13,113 French rows; the French side runs short where the gold contract rejects translation, and the drop accounting is per reason in the dataset card and the run's drop summary. The system prompt stays English for both languages, so the query language is the only moving variable ([why](../concepts/design-decisions.md)).

**Evaluation.** Two slices of the held-out test split: 1,000 English prompts and 879 French pairs, each evaluated for the base model and the adapter under identical decoding and the same conservative parser. The [comparison gate](../concepts/determinism.md) additionally requires the slice sets and every per-slice prompt digest to match.

## Results

### English slice (n=1000)

| Metric | Base | Adapter | Delta |
|--------|------|---------|-------|
| Valid JSON rate | 0.9160 (916/1000) | **0.9970** (997/1000) | +0.0810 |
| Function name accuracy | 0.9110 (911/1000) | **0.9930** (993/1000) | +0.0820 |
| Argument exact match | 0.7070 (707/1000) | **0.8730** (873/1000) | +0.1660 |
| Argument F1 | 0.7569 (3858/5097) | **0.9211** (5112/5550) | +0.1642 |
| Full-call exact match | 0.7050 (705/1000) | **0.8700** (870/1000) | +0.1650 |

### French slice (n=879)

| Metric | Base | Adapter | Delta |
|--------|------|---------|-------|
| Valid JSON rate | 0.9044 (795/879) | **0.9954** (875/879) | +0.0910 |
| Function name accuracy | 0.8976 (789/879) | **0.9898** (870/879) | +0.0922 |
| Argument exact match | 0.6655 (585/879) | **0.8760** (770/879) | +0.2105 |
| Argument F1 | 0.7091 (3000/4231) | **0.9208** (4288/4657) | +0.2117 |
| Full-call exact match | 0.6633 (583/879) | **0.8726** (767/879) | +0.2093 |

## The language gap, measured three ways

Full-call exact match, French minus English, across the three models this project has measured on these slices:

| Model | en | fr | Gap |
|-------|----|----|-----|
| Base model | 0.7050 | 0.6633 | -0.0417 |
| v1 adapter, English-only training ([M1 baseline](https://github.com/AbdelStark/sommelier/issues/108)) | 0.8740 | 0.8510 | -0.0230 |
| v2 adapter, mixed en+fr training (this run) | 0.8700 | 0.8726 | +0.0026 |

Three findings, stated plainly. The base model pays about 4 points of full-call exactness on French input. English-only fine-tuning transfers most of its gains to French without seeing a single French example, narrowing the gap to 2.3 points. Adding the French training data closes the gap to measurement noise: French now matches English on every metric to within a third of a point.

The cost, reported rather than hidden: the v2 adapter's English slice sits 0.3 to 0.8 points below the v1 reference (full-call 0.8700 against 0.8740, argument F1 0.9211 against 0.9291, valid JSON 0.9970 against 1.0000). At n=1000 this is within one standard error, and whether it is noise or a small capacity trade for bilingual coverage cannot be decided from one seed.

## Runtime and cost

Per-stage wall clock from the run's `runtime_metadata.json`:

| Stage | Elapsed |
|-------|---------|
| data prepare | 8.1 s |
| format build | 30.6 s |
| eval run (base, both slices) | 1,805.2 s |
| train run | 20,540.3 s |
| eval run (adapter, both slices) | 2,835.1 s |
| report compare | 1.1 s |

Training took 5 hours 42 minutes (3,516 optimizer steps over 28,113 rows, roughly twice the reference run's data); each evaluation covered 1,879 prompts across the two slices. Peak GPU memory was 26,369 MiB. Observed cost is recorded as unavailable, as in the reference run, because Modal exposed no billing data to the run itself.

The per-language sequence audit that gates training measured the French rows at nearly identical token lengths to English (max 2,174 against 2,166 tokens, p95 711 against 716, budget 4,096), so the French data changed the compute bill only by its row count.

## Where the evidence lives

- The [v2 adapter repository](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-fr-en-lora) carries the per-slice evaluation reports, the gated comparison with its language gaps section, and the runtime metadata under `reports/`. The adapter is a Llama 3.1 derivative ("Built with Llama"); obligations are in [Licensing](../project/licensing.md).
- The [French dataset](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits-fr) carries the 14,936 paired rows and the translation summary with the translator pin, prompt digest, and per reason drop counts (CC-BY-4.0).
- The [English splits dataset](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits) is unchanged from the reference run.
- The [reproduction guide](../guides/reproduction.md) and [remote execution guide](../guides/remote-execution.md) cover the two-step bilingual flow: the translation run, then the pipeline run that stages its rows.

## Claim boundaries

Everything from the [reference run's boundaries](reference-run.md) applies, plus the boundaries the French slice adds:

- The French test rows are machine translations (Mistral-Nemo-Instruct-2407, greedy, constrained by protected spans), reviewed on samples, not natively authored requests.
- The French slice excludes the 12.1 percent of pairs whose gold arguments embed English text lifted from the query, because faithful translation would break the byte-identical gold contract. The slice is therefore slightly biased toward rows with language-neutral arguments.
- Instruction-language effects are unmeasured: the system prompt is English for both slices by design.
- One run, one seed, two languages. Nothing here ranks the model against any public benchmark.
