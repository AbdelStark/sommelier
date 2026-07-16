# Hebrew slice, translated locally with Hy-MT2

This page records a second, independent Hebrew experiment that is deliberately
separate from the preregistered [Hebrew v3](hebrew-v3.md) evidence. It asks a
narrower, engineering question: if the Hebrew training data is produced by a
small open-source translation model running locally, does adding it to the
English training set improve Hebrew single-tool-call accuracy without materially
reducing English accuracy? Because the forward translator is a local open model
and there is no human semantic review, this slice never claims to be the
preregistered v3 result, and it publishes under its own dataset and adapter
repositories.

## Why a separate slice

The preregistered Hebrew v3 pipeline requires a paid provider teacher and a
named human reviewer who signs a back-translation audit with an Ed25519 key.
Neither is available for this run, and the run does not fabricate them. Instead
it uses a distinct project name and a distinct Hebrew dataset repository, so the
preregistered-contract and human-review admission gates do not apply, and it
runs the standard training and evaluation stages through
`remote_hy_pipeline.py`. Every stage after data preparation is the exact shared
pipeline code: the same base model, tokenizer, formatter, QLoRA trainer,
deterministic generator, parser, and comparison gate.

## Translation

The forward translator is
[tencent/Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)
(Hunyuan-MT), run locally through ollama from the official `Q8_0` GGUF build.
The exact local Ollama blob was 1,908,528,192 bytes with SHA-256
`5c3fe0b1408a5ceb0143184ef247b11b579c525f4b02b060e6c851bb76fef1a4`
(Ollama model id prefix `ad45125201fa`). It matches
`Hy-MT2-1.8B-Q8_0.gguf` at immutable repository revision
`b27182d810fa3ceb6ed04e7c324c54e35c0d209c` byte for byte. This is Tencent's Q8 GGUF variant,
not a direct load of the separate `Hy-MT2-1.8B-FP8` checkpoint; the model
family is the same, but the quantized artifact is not.
The `local_hy_translate.py` driver reproduces the exact root-cohort selection a
full pipeline run makes, then translates only each natural-language query into
Hebrew with greedy decoding (temperature 0, fixed per-attempt seed). Tool schemas
and gold answers stay byte-identical to the English root, and every accepted row
names its English root with `source_example_id`.

Hy-MT2 is a pure translator, so it would otherwise translate gold-bearing
argument values. Each protected span in the query is masked with a short ASCII
sentinel before translation and restored afterward; a row is retried with two
alternate sentinels and then an unmasked pass. Every accepted row passes the
same mechanical audit the French and Hebrew pipelines use: every protected span
present at token boundaries, Hebrew as the dominant script (at least half of
unprotected letters), no unsafe bidirectional control or Unicode
control/format/surrogate/replacement code points, no reproduction of the
translation instruction, and length within the preparation budget. Rows that
fail are absent and counted, so the release is a machine-translated survivor
corpus.

## Setup

**Model and data.** Base model
[nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1)
at base and tokenizer revision `54641c1611fcff44fa4865626462445e0a153fc7`. The
English root is
[Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
at revision `26d14ebfe18b1f7b524bd39b404b50af5dc97866`. After the single-call
filter and deduplication, the run uses seeded splits of 15,000 train, 1,000
validation, and 1,000 test roots (seed 42). The exact Hebrew training rows came
from revision `c9598f1855ca88e11a4aa31c53ad387faf467eb2` and pair to those
roots by `source_example_id`. That snapshot is now privately quarantined
because it inherited credential-shaped synthetic strings from APIGen. The
[public sanitized dataset](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits-he-hymt-sanitized)
keeps every row but replaces 15 GitHub-PAT-shaped substrings in `tools`; its
manifest binds both the exact training hash and the public output hash.

**Training.** The QLoRA hyperparameters match the reference run: NF4 4-bit with
bfloat16 compute, LoRA rank 16, alpha 32, dropout 0.05 on q_proj · k_proj ·
v_proj · o_proj · gate_proj · up_proj · down_proj, 2 epochs, learning rate 2e-4,
cosine schedule, warmup ratio 0.03, per-device batch 4 with gradient
accumulation 4 (effective 16), max sequence length 4096, completion-only loss.
The training set is the English train split plus the accepted Hebrew rows paired
to English train roots. Hardware is one L40S on Modal.

**Evaluation.** Base model and adapter each generate one completion per test
prompt with greedy decoding (temperature 0.0, sampling disabled, at most 512 new
tokens), parsed by `sommelier.parser.v1`. The test set is evaluated as two
slices, English (`en`) and Hebrew (`he`); the Hebrew slice pairs each accepted
Hebrew test row to its exact English root. Parse failures count against every
metric.

## Translation accounting

The published `translation_summary.json` matches the local producer bytes
exactly and binds the quarantined training JSONL. The public `rows.he.jsonl`
contains the same 16,272 rows after the separately manifested 15-literal
credential-pattern sanitization; it is intentionally not producer-byte-identical.

| Field | Value |
|-------|-------|
| Selected roots | 17,000 |
| Accepted Hebrew rows | 16,272 |
| Yield | 95.72% |
| Accepted by split (train/validation/test) | 14,364 / 961 / 947 |
| Mechanical drops (by reason) | 626 missing protected span; 48 malformed spacing; 54 wrong script |

The shared prepare stage then removed 86 duplicate Hebrew queries, leaving
14,286 training, 955 validation, and 945 held-out Hebrew examples. It found no
pairing, tool-schema, gold-answer, or cross-split mismatch.

## Tokenizer and workload

On the 16,186 prepared exact pairs, Hebrew query text used 2.932 times as many
tokens as its matched English roots under the pinned Nemotron tokenizer
(median per-pair ratio 3.00; p95 4.64). For the actual two-epoch training split,
English plus Hebrew contained 2.068 times the non-padding full-sequence tokens
of the English-only counterfactual. These are deterministic token counts; they
exclude dynamic padding and are not a measured runtime or billing ratio.

## Results

Base model versus this adapter on the same held-out test prompts, per language
slice, quoted with counts exactly as `report/comparison_report.json` records
them. Delta intervals are paired-bootstrap 95% confidence intervals with 2,000
resamples.

| Slice | Metric | Base | Adapter | Delta (95% CI) |
|-------|--------|------|---------|----------------|
| Hebrew (n=945) | Full-call exact match | 410/945 (43.39%) | 786/945 (83.17%) | +39.79 pts (+36.61, +42.96) |
| Hebrew (n=945) | Function name accuracy | 706/945 (74.71%) | 937/945 (99.15%) | +24.44 pts (+21.69, +27.20) |
| Hebrew (n=945) | Argument F1 | 2254/4207 (53.58%) | 4562/5112 (89.24%) | +35.66 pts (+30.64, +40.27) |
| Hebrew (n=945) | Valid JSON rate | 765/945 (80.95%) | 941/945 (99.58%) | +18.62 pts (+16.08, +21.16) |
| English (n=1,000) | Full-call exact match | 705/1000 (70.50%) | 884/1000 (88.40%) | +17.90 pts (+15.10, +20.80) |
| English (n=1,000) | Function name accuracy | 911/1000 (91.10%) | 994/1000 (99.40%) | +8.30 pts (+6.60, +10.10) |

On the exact 945 matched roots, the adapter's Hebrew-minus-English full-call
gap was -5.19 points (95% CI -6.98 to -3.49), compared with -27.62 points for
the base model. This is a within-run base-versus-adapter comparison, not the
preregistered v1-versus-v3 three-arm claim.

## Runtime

Modal function call `fc-01KXJBMN4CMJ68BZDSFFWAKM9F` completed successfully on
one L40S. Training took 22,453 seconds (6 h 14 min), base evaluation 1,957
seconds, and adapter evaluation 1,560 seconds. Peak recorded GPU memory was
26,608 MiB. Modal billing was not exposed to the run, so observed cost is
reported as unavailable rather than estimated after the fact.

## Where the evidence lives

- The immutable model tag `he-hymt-full-002` resolves to Hub commit
  `f09e5d31ab29dca49e7c2df7113e810cf3dfb43a`; every one of its 33 curated
  files matches the local publication bundle by Git-blob or LFS SHA-256.
- The immutable dataset tag `he-hymt-sanitized-v1` resolves to Hub commit
  `f7159c08823e2f986375927f998263f969738f43`; all six curated files match the
  local publication bundle, including the 30,099,996-byte sanitized JSONL.
- The [adapter repository](https://huggingface.co/abdelstark/Llama-3.1-Nemotron-Nano-8B-xlam-tool-calling-he-en-hymt-lora)
  carries the PEFT weights, tokenizer sidecars, resolved config, every stage
  manifest, training metrics, evaluation telemetry and aggregate reports,
  tokenizer-tax records/report, and comparison report. Verbatim formatted test
  prompts and raw generations are not copied to the Hub because the
  fail-closed release scan found credential-shaped synthetic benchmark fields;
  their exact identities remain bound by the published manifest hashes. The
  adapter is a Llama 3.1 derivative
  ("Built with Llama"); obligations are in [Licensing](../project/licensing.md).
- The public Hebrew dataset repository carries sanitized `rows.he.jsonl`, the
  exact producer `translation_summary.json` (CC-BY-4.0), and a closed
  sanitization manifest; its later
  `translator_identity.json` binds the locally observed blob to the immutable
  official Tencent file without changing the rows used for training.
- The English rows are the published
  [splits dataset](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits).

## Claim boundaries

- The Hebrew training and evaluation queries are a machine-translated survivor
  corpus with no human semantic review. Translation error and audit-driven
  selection are limitations of both the training data and the Hebrew evaluation
  slice.
- Metrics measure schema-valid single tool calls on this held-out test split
  only, for this base model, tokenizer, parser, and decoding config. They are not
  claims of production readiness or generalization, and they rank nothing against
  any public benchmark or hosted model.
- This slice is not the preregistered Hebrew v3 result and does not substitute
  for it. It is an open-model engineering benchmark.
- Runtime metadata records repository HEAD `705fe9df273144039c3b53b381c04761b41c0205`
  with unknown worktree cleanliness. The Hy-MT wrapper was mounted from local
  source and was not contained in that commit, and stage manifests have no
  dependency-lock digest, so the run does not claim byte-for-byte source
  reconstruction from the recorded SHA alone.
