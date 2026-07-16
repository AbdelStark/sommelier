# The reference run

This page states the published Sommelier result the way a paper would: the exact setup, the numbers with their numerators and denominators, the run identity that makes the comparison verifiable, and the boundaries of the claim. The run is `nemotron-8b-full-3`, executed 2026-07-02 on a single L40S. Everything below comes from the run's own comparison report and runtime metadata, not from memory. The question this run leaves open, whether the same result holds in French, is answered by [the French run](french-run.md).

## Run identity

The digests are the point of the whole system: the [comparison gate](../concepts/determinism.md) refused to write this report until base and adapter evaluations agreed on every one of them. Anyone reproducing the run can check their digests against these.

| Field | Value |
|-------|-------|
| Run ID | `nemotron-8b-full-3` |
| Evidence class | full run |
| Created at | `2026-07-02T16:44:06.278982+00:00` |
| Config digest | `3a8fb0d8b4f5b4c6bcde370a51b0111598814e28df78ba8e6c2c25a907e5958b` |
| Test split digest | `db8dd82f2a29577426b7b9598e2c0a296257d7535542e3117c4044f506305ae8` |
| Prompt set digest | `a0da8fa28835a329dba5c5314ada3aff21f950939af2e1ae186155d3b494f39a` |
| Parser version | `sommelier.parser.v1` |
| Decoding | `{"do_sample": false, "max_new_tokens": 512, "temperature": 0.0}` |

## Setup

**Model and data.** Base model [nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1). The historical run exported [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) at the recorded `main` revision. The repository's current `examples/config.full.yaml` is a runnable English-only v1/default full config with immutable base, tokenizer, and root-dataset pins. It is a recipe for a new run, not the historical run's identity: the exact rows scored by this result remain the published [abdelstark/sommelier-xlam-single-call-splits](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits), and the recorded resolved config and checksummed artifacts remain authoritative. After the single-call filter and deduplication described in [Data policy](../concepts/data.md), the recorded run used seeded splits of 15,000 train, 1,000 validation, and 1,000 test examples (seed 42).

**Training.** The current [`examples/config.full.yaml`](https://github.com/AbdelStark/sommelier/blob/main/examples/config.full.yaml) matches the recorded v1 QLoRA hyperparameters below:

| Setting | Value |
|---------|-------|
| Quantization | NF4 4-bit, bfloat16 compute |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Target modules | q_proj · k_proj · v_proj · o_proj · gate_proj · up_proj · down_proj |
| Schedule | 2 epochs, learning rate 2e-4, cosine, warmup ratio 0.03 |
| Batch | per-device 4, gradient accumulation 4 (effective 16) |
| Max sequence length | 4096 |
| Loss | completion-only, prompt tokens masked ([why](../concepts/training.md)) |

**Evaluation.** Both models generated one completion per test prompt with greedy decoding (temperature 0.0, sampling disabled, at most 512 new tokens), parsed by `sommelier.parser.v1`, which extracts a single JSON tool call and never repairs malformed output. Parse failures count against every metric. The adapter evaluation read the same stored prompts as the base evaluation, matched by digest. Method details in [Evaluation method](../concepts/evaluation.md).

**Hardware.** One L40S, recorded from the config (the runtime metadata notes its hardware source explicitly).

## Results

All five metrics, base versus adapter, on the same 1,000 held-out test prompts. Values are quoted with their counts, exactly as the report records them:

| Metric | Base | Adapter | Delta |
|--------|------|---------|-------|
| Valid JSON rate | 0.9160 (916/1000) | **1.0000** (1000/1000) | +0.0840 |
| Function name accuracy | 0.9110 (911/1000) | **0.9960** (996/1000) | +0.0850 |
| Argument exact match | 0.7070 (707/1000) | **0.8760** (876/1000) | +0.1690 |
| Argument F1 | 0.7569 (3858/5097) | **0.9291** (5122/5513) | +0.1722 |
| Full-call exact match | 0.7050 (705/1000) | **0.8740** (874/1000) | +0.1690 |

Argument F1 is micro-averaged over flattened argument key/value pairs, so its counts are not per-example: the numerator is two times the matched pair count and the denominator is the pooled count of predicted plus gold pairs, which is why it differs between the two models. Definitions for all five metrics are in the [metrics reference](../reference/metrics.md).

## What improved, and what it means

JSON validity saturated: the adapter produced a parseable, schema-valid single tool call on all 1,000 prompts, against 916 for the base model. Function naming is nearly solved (996/1000). The remaining adapter errors are almost entirely argument mismatches: 996 correct names but only 876 exact argument matches. Some of those mismatches are genuine errors; others are values that a human would accept but canonical-JSON equality does not, because exact-match scoring counts semantically equivalent, differently formatted values as wrong. That scoring choice is deliberate and conservative, and it penalizes both models the same way. The [blog post](../blogposts/sommelier_blog_post.md) walks through examples from the raw generations.

## Runtime and cost

Per-stage wall clock from the run's `runtime_metadata.json`:

| Stage | Elapsed |
|-------|---------|
| data prepare | 5.6 s |
| format build | 16.9 s |
| eval run (base) | 823.5 s |
| train run | 10,996.0 s |
| eval run (adapter) | 923.4 s |
| report compare | 1.3 s |

Training took just over three hours; the two 1,000-prompt evaluations took 13.7 and 15.4 minutes. Peak GPU memory was 26,306 MiB. Observed cost is recorded as **unavailable**: cost in Sommelier is observed evidence, not an estimate, and Modal exposed no billing data to this run, so the report says so instead of guessing. From the maintainer's billing console, outside the run record, the GPU bill for this run was about eight dollars; that figure and its context are in the [blog post](../blogposts/sommelier_blog_post.md).

## Where the evidence lives

- The [adapter repository](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora) carries the comparison report and the evaluation evidence under its `reports/` directory. The adapter is a Llama 3.1 derivative ("Built with Llama"); obligations are in [Licensing](../project/licensing.md).
- The [splits dataset](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits) carries the exact train, validation, and test rows (CC-BY-4.0).
- The [reproduction guide](../guides/reproduction.md) goes from a clean checkout to this comparison report on a rented GPU.
- What each artifact contains is specified in the [artifacts reference](../reference/artifacts.md).

## Claim boundaries

The report states its own limitations, and they are part of the result:

- Metrics measure schema-valid single tool calls on this held-out test split only. Multi-call plans are out of scope for v1.0.
- Argument comparisons are exact canonical-JSON matches; semantically equivalent but differently formatted values count as mismatches.
- The numbers describe the recorded run: this hardware, these pinned dependencies, this dataset revision, this parser, this decoding config. They are not claims of production readiness, broad reliability, or generalization beyond the evaluated split, and they rank nothing against any public benchmark or hosted model.
- Parse failures count against every metric, and the raw generations are retained for audit.

## Smoke runs are not reference results

Sommelier also produces smoke runs: the same pipeline bounded to at most 100 train, 20 validation, and 20 test examples, under a `smoke-` run ID prefix. Their reports carry an explicit "Evidence class: smoke run" line. A 20-prompt evaluation is a plumbing check, not a measurement; smoke numbers must never be quoted as reference results, and the report format exists so they cannot be mistaken for one.
