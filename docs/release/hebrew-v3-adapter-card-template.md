---
license: llama3.1
base_model: nvidia/Llama-3.1-Nemotron-Nano-8B-v1
library_name: peft
tags:
  - lora
  - qlora
  - tool-calling
  - hebrew
---

# Llama 3.1 Nemotron English-Hebrew tool-calling QLoRA adapter

> Replace every `REPLACE_FROM_VERIFIED_BUNDLE` marker from the final local
> evidence before validation or publication. Do not hand-copy metric values
> from console output.

**Built with Llama**

This is an unmerged PEFT QLoRA adapter for
`nvidia/Llama-3.1-Nemotron-Nano-8B-v1` at base and tokenizer revision
`54641c1611fcff44fa4865626462445e0a153fc7`. It was trained on paired English
and Hebrew single-tool-call examples under Sommelier's recorded v3 contract.

Use and redistribution are subject to the NVIDIA Open Model License and the
Llama 3.1 Community License. Review `THIRD_PARTY.md`,
`LICENSE-NVIDIA-OPEN-MODEL.txt`, `LICENSE-LLAMA-3.1.txt`, and `NOTICE` before
use. The publisher requires exact copies of the reviewed project files and
rejects abbreviated or edited terms. This repository distributes adapter
weights, not merged base weights.

## Immutable evidence identity

- Adapter tree SHA-256: `REPLACE_FROM_VERIFIED_BUNDLE`
- Final `experiment_report.json` SHA-256: `REPLACE_FROM_VERIFIED_BUNDLE`
- Producer source revision: `REPLACE_FROM_VERIFIED_BUNDLE`
- Hebrew paired dataset revision: `REPLACE_FROM_VERIFIED_BUNDLE`

The `sommelier/` directory contains the resolved config, succeeded run/train
manifests, passing release preflight, and final claim-gated
`sommelier.experiment_report.v2`. The experiment report is authoritative for
cohort sizes, marginal and exact matched-pair metrics, ordered pairing identity,
paired-bootstrap intervals, claim decisions, tokenizer tax, training runtime,
memory, storage, and inference telemetry.

The immutable Hebrew dataset revision named above is separately validated to
contain the exact Phase-A config, its pre-provider translation run identity,
and the finalized semantic review carrying the preregistered human reviewer's
signed attestation. Signature verification establishes possession of that
configured public key and integrity of the attested decisions; it does not
establish native fluency or semantic correctness.

Its training-tax evidence separates the observed combined en+he formatted-row
workload into an English-only arithmetic counterfactual and an additive Hebrew
increment, with data/token ratios and combined-vs-English multipliers. The
counterfactual is not a separately trained English-only runtime or accuracy arm.

REPLACE_FROM_VERIFIED_BUNDLE_WITH_RENDERED_CLAIM_SECTION

## Limitations

The Hebrew data is machine-translated and selection-conditioned. Its bounded
semantic review is non-native and does not establish native fluency or complete
semantic correctness. Results apply only to the pinned dataset, prompts,
parser, decoding, model, adapter tree, and hardware/runtime evidence. Sequential
inference telemetry is not serving throughput, projected tokens are not billed
tokens, the Hebrew increment is selection-conditioned on accepted translated
rows, and no full-fine-tuning or currency saving is claimed without a matched
measurement.
