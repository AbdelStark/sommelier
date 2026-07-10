---
title: Sommelier
emoji: 🍷
colorFrom: green
colorTo: gray
sdk: static
pinned: true
license: mit
short_description: Reproducible tool-calling fine-tuning, with evidence
models:
  - abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora
  - abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-fr-en-lora
  - nvidia/Llama-3.1-Nemotron-Nano-8B-v1
datasets:
  - abdelstark/sommelier-xlam-single-call-splits
  - abdelstark/sommelier-xlam-single-call-splits-fr
  - Salesforce/xlam-function-calling-60k
tags:
  - tool-calling
  - function-calling
  - lora
  - qlora
  - peft
  - nemotron
  - reproducibility
  - sovereign-ai
  - multilingual
---

# Sommelier

An interactive, educational front for [Sommelier](https://github.com/AbdelStark/sommelier): a reference implementation for adapting a small open language model (nvidia/Llama-3.1-Nemotron-Nano-8B-v1) to emit exactly one schema-valid JSON tool call per request, built on one premise: a fine-tuning claim is only as good as its evidence.

The page covers the whole recipe with real, verifiable data:

- Why post-training small open models matters (the sovereign AI argument, with the French language gap as a measured example: -4.2 points, closed to +0.3).
- The NVIDIA Nemotron open-model effort the recipe builds on.
- The data policy, the six-stage pipeline, QLoRA training with the actual 1,876-step loss curve, and the conservative evaluation parser (with an in-browser playground that runs a faithful JS port of it).
- Base-versus-adapter results with numerators and denominators, an outcome transition matrix computed from the published generations, and a browser over verbatim model outputs.
- How to run the adapters (transformers + PEFT, vLLM, Modal), how to reproduce every number, and where the claims stop.

Everything is static: no GPU behind this Space, no simulated numbers. Every figure traces to a run artifact in the linked model repos, the linked datasets, or a cited source.

## Deploying

This directory is the source of truth for the Space. After changing it, deploy with:

```bash
hf upload abdelstark/sommelier space . --type space
```

The page is a single self contained `index.html` (inline CSS and JS, Google Fonts and MathJax from CDN, no build step). The interactive components are driven by data extracted from the published run artifacts: the training curve from the adapter repo's `reports/training_metrics.jsonl`, the outcome matrices and sample browser from the `generations.jsonl` files of runs `nemotron-8b-full-3` and `nemotron-8b-fr-full-4`, and the metric tables from the published comparison reports. If a new run supersedes those, regenerate the inlined data rather than editing numbers by hand.
