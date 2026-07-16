---
license: llama3.1
base_model: nvidia/Llama-3.1-Nemotron-Nano-8B-v1
library_name: peft
pipeline_tag: text-generation
datasets:
  - abdelstark/sommelier-xlam-single-call-splits-he-hymt-sanitized
  - Salesforce/xlam-function-calling-60k
language:
  - en
  - he
tags:
  - lora
  - qlora
  - tool-calling
  - function-calling
  - hebrew
---

# Llama 3.1 Nemotron English-Hebrew tool-calling QLoRA adapter (Hy-MT2 slice)

**Built with Llama**

This is an unmerged PEFT QLoRA adapter for
[`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1)
at base and tokenizer revision `54641c1611fcff44fa4865626462445e0a153fc7`. It was
trained on paired English and Hebrew single-tool-call examples, where the Hebrew
queries were machine-translated locally with the official `Q8_0` build from
[`tencent/Hy-MT2-1.8B-GGUF`](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)
(see the
[sanitized public dataset](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits-he-hymt-sanitized)).

Use and redistribution are subject to the NVIDIA Open Model License and the
Llama 3.1 Community License. Review `THIRD_PARTY.md`,
`LICENSE-NVIDIA-OPEN-MODEL.txt`, `LICENSE-LLAMA-3.1.txt`, and `NOTICE` before
use. This repository distributes adapter weights, not merged base weights.

## What this is (and is not)

This adapter is a **local, open-model experiment**: does adding locally
machine-translated Hebrew tool-calling data to the English training set improve
Hebrew single-tool-call accuracy without materially hurting English? It is
deliberately distinct from the preregistered Sommelier Hebrew v3 adapter, which
uses a different forward teacher and a human-signed semantic review. This slice
carries **no human semantic review** of the training data, so its results should
be read as an engineering benchmark, not a validated Hebrew capability claim.

## Training

- Base: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` @ `54641c1611fcff44fa4865626462445e0a153fc7`
- Method: QLoRA (nf4 4-bit, bf16 compute), LoRA rank 16 / alpha 32 / dropout 0.05
  on `q,k,v,o,gate,up,down` projections
- Data: English xLAM single-call rows + their Hy-MT2 Hebrew translations
- Exact paired training-dataset revision (privately quarantined):
  `c9598f1855ca88e11a4aa31c53ad387faf467eb2`
- Public dataset: a release-sanitized copy with 15 credential-shaped upstream
  literals replaced and no rows removed; see its `public_sanitization.json`
- Prepared training examples: 15,000 English + 14,286 Hebrew; validation:
  1,000 English + 955 Hebrew
- Epochs 2, effective batch 16 (per-device 4 x grad-accum 4), cosine schedule,
  lr 2e-4, max sequence length 4096
- Hardware: one L40S on Modal; 3,662 optimizer steps, 33,955,808 input tokens,
  22,453 seconds training time, 26,608 MiB peak GPU memory

Exact resolved config, all stage manifests, training metrics, evaluation
reports and telemetry, tokenizer-tax records/report, and the comparison report
are in the `sommelier/` directory of this repository.

## Benchmark (deterministic, temperature 0)

Base model vs this adapter on the held-out test split, per language slice.
Metrics come from `report/comparison_report.json`
(`sommelier.comparison_report.v3`); the Hebrew slice pairs each Hebrew test row
to its exact English root.

| Slice | Metric | Base | Adapter | Delta (95% CI) |
|-------|--------|------|---------|----------------|
| Hebrew (n=945) | Full-call exact match | 410/945 (43.39%) | 786/945 (83.17%) | +39.79 pts (+36.61, +42.96) |
| Hebrew (n=945) | Function name accuracy | 706/945 (74.71%) | 937/945 (99.15%) | +24.44 pts (+21.69, +27.20) |
| Hebrew (n=945) | Argument exact match | 415/945 (43.92%) | 787/945 (83.28%) | +39.37 pts (+36.19, +42.54) |
| Hebrew (n=945) | Argument F1 | 2254/4207 (53.58%) | 4562/5112 (89.24%) | +35.66 pts (+30.64, +40.27) |
| Hebrew (n=945) | Valid JSON rate | 765/945 (80.95%) | 941/945 (99.58%) | +18.62 pts (+16.08, +21.16) |
| English (n=1,000) | Full-call exact match | 705/1000 (70.50%) | 884/1000 (88.40%) | +17.90 pts (+15.10, +20.80) |
| English (n=1,000) | Function name accuracy | 911/1000 (91.10%) | 994/1000 (99.40%) | +8.30 pts (+6.60, +10.10) |

Intervals are paired-bootstrap 95% confidence intervals with 2,000 resamples.
On the exact 945 matched roots, the adapter's Hebrew-minus-English full-call
gap is -5.19 points (95% CI -6.98 to -3.49); the base gap is -27.62 points.

## Translation and workload accounting

The local translator accepted 16,272 of 17,000 selected roots (95.72%). The
shared preparation gate removed 86 duplicate Hebrew queries, leaving 14,286 / 955 /
945 train/validation/test examples. Under the pinned tokenizer, matched Hebrew
queries used 2.932 times the English query tokens; the combined two-epoch
non-padding full-token workload was 2.068 times the English-only counterfactual.

The translation used a 1,908,528,192-byte local Ollama GGUF blob with SHA-256
`5c3fe0b1408a5ceb0143184ef247b11b579c525f4b02b060e6c851bb76fef1a4`.
That blob matches `Hy-MT2-1.8B-Q8_0.gguf` at immutable Tencent repository
revision `b27182d810fa3ceb6ed04e7c324c54e35c0d209c` byte for byte.
It did not directly load the separate Tencent FP8 checkpoint.

## Artifact identities

- Run: `he-hymt-full-002`; Modal call: `fc-01KXJBMN4CMJ68BZDSFFWAKM9F`
- Adapter weights: 167,832,240 bytes; SHA-256
  `78babe658312dc9b7983c8823a7969fe6fd6fb9993a515148f1d60264746a07f`
- Original Modal six-file training-output adapter-tree SHA-256 (the Hub root is
  re-laid out for direct PEFT loading and has its own `SHA256SUMS`):
  `dccadbba86a35a6eb6df38ac7da5f737b01fbc2b87d3ae2cce0b436040245c00`
- Comparison report SHA-256:
  `1c0640062188edabd520b595671895d569eb5a1fe529e6d07690e75db11ced9d`

Runtime metadata records repository HEAD
`705fe9df273144039c3b53b381c04761b41c0205` with unknown worktree
cleanliness. The Hy-MT wrapper was mounted from local source and was not in
that commit, and stage manifests record a null dependency-lock digest. This
release therefore does not claim byte-for-byte source reconstruction from the
recorded SHA alone. `translator_identity.json` is an independently verified
post-run binding, not metadata captured by the original translation process.

## Public evidence boundary

The public bundle deliberately omits the verbatim formatted test prompts and
raw generation JSONL files. Sommelier's fail-closed publication scan found
credential-shaped synthetic benchmark content in them (including a `password`
argument name and API-key-shaped example text). Those exact files remain bound
by the published stage-manifest hashes and in the durable run storage, but are
not copied to the Hub. The aggregate evaluation reports, inference telemetry,
comparison report, manifests, training metrics, and tokenizer analysis contain
no redaction findings and are published with `SHA256SUMS`.

## Limitations

The Hebrew training and evaluation queries are a machine-translated survivor
corpus with no human semantic review; translation error and audit-driven
selection are limitations of both the training data and the Hebrew evaluation
slice. Metrics are single-tool-call, single-turn, and specific to this base
model, tokenizer, parser, and decoding settings. This adapter is not a native
Hebrew assistant or production safety artifact.
