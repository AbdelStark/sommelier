# Training

`sommelier train run` trains a LoRA adapter on a 4-bit quantized base model with completion-only loss. The stage has a strict contract: hyperparameters come from the config and are never adjusted at runtime, the loss mask boundary is proven at the token level before any gradient step, and failures name the exact config fields to change. This page explains each part of that contract and what it looked like in the [reference run](../results/reference-run.md).

## The QLoRA setup

The base model loads in 4-bit NF4 quantization with double quantization enabled and bfloat16 compute, then a LoRA adapter is attached on top. Only the adapter weights train; the quantized base stays frozen. This is what lets an 8B model fine-tune on a single mid-range GPU. The values below are the reference configuration from [`examples/config.full.yaml`](https://github.com/AbdelStark/sommelier/blob/main/examples/config.full.yaml):

| Config field | Reference value |
|---|---|
| `train.quantization` | `nf4-4bit` |
| `train.compute_dtype` | `bfloat16` |
| `train.lora_rank` | 16 |
| `train.lora_alpha` | 32 |
| `train.lora_dropout` | 0.05 |
| `train.target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| `train.epochs` | 2 |
| `train.per_device_batch_size` | 4 |
| `train.gradient_accumulation_steps` | 4 |
| `train.learning_rate` | 0.0002 |
| `train.scheduler` | `cosine` |
| `train.warmup_ratio` | 0.03 |
| `train.max_sequence_length` | 4096 |

Two implementation choices in [`qlora.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/training/qlora.py) are worth knowing. Gradient checkpointing is enabled explicitly rather than left to library defaults, because an 8B model at batch size 8 without checkpointing exceeded a 44 GiB GPU in the first full run. And logging runs every step (`logging_steps=1`) with input-token counting on, so the metrics file records the whole loss curve, not a sampled sketch. The full field list is in the [configuration reference](../reference/configuration.md).

## Completion-only loss

Each formatted example renders as `prompt_text` followed by `target_text` (the canonical JSON of the gold tool call) plus whatever closing tokens the chat template adds (see [Data policy](data.md) and [the format stage](pipeline.md)). Training computes loss only on the tokens after the proven prompt boundary. Every prompt token gets the label `-100` (`IGNORE_INDEX` in [`collators.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/training/collators.py)), which the loss function skips; padding positions get attention 0 and the same ignore label.

The reasoning: the model's job at inference time is to emit a tool call given a prompt it did not write. Spending gradient on reproducing the system message and the tool schemas teaches the wrong thing and dilutes the signal from the part that matters. The mask makes the training objective match the evaluation objective exactly: given this prompt, produce this JSON.

## The provable prompt boundary

Masking prompt tokens requires knowing exactly where the prompt ends in token space, and tokenizers do not guarantee that text concatenation survives tokenization. `find_prompt_token_count` in `collators.py` proves the boundary per example with two checks:

1. `full_text` starts with `prompt_text` as a string.
2. Tokenizing `full_text` yields a sequence whose prefix equals the tokenization of `prompt_text` on its own.

If a tokenizer merges tokens across the boundary, the second check fails and training stops with a `SchemaValidationError`. There is no fallback to full-sequence loss.

The refusal is the point. A silent fallback would keep the run alive but quietly swap the objective from "learn to emit the tool call" to "learn to reproduce the prompt too", for exactly the examples where the boundary was ambiguous. The resulting adapter would still produce a metrics table, and nothing in that table would tell you the objective changed mid-dataset. The same principle covers truncation: if cutting a sequence to `max_sequence_length` would remove every target token, the collator fails instead of training on a sequence with nothing left to learn.

The format stage sets this proof up: it fails during formatting if the chat template does not render `full_text` as `prompt_text` followed by the target, so the string-prefix property holds before training ever starts.

## Failures name their fix

Training never adapts hyperparameters to fit hardware. An auto-tuned run is not the run you configured: if the trainer silently halved the batch size to survive an OOM, the effective batch, the learning-rate schedule, and the cost would all differ from what `config.resolved.yaml` records, and the [comparison gate](determinism.md) would be certifying a run that never happened as configured. So resource failures surface as errors instead:

| Failure | Error | What the hint names |
|---|---|---|
| GPU out of memory | `ResourceError` (SOM401, exit 4) | `train.per_device_batch_size`, `train.gradient_accumulation_steps`, `train.max_sequence_length`, `remote.gpu`, with their current values |
| Time budget exceeded | `ResourceError` (SOM401, exit 4) | `remote.train_timeout_seconds`, with suggestions to raise it, reduce epochs, or shrink splits |

You change the config, and the next run is a different run with a different digest, which is exactly what it should be. Exit codes and error classes are cataloged in the [error reference](../reference/errors.md).

## What training writes

Under `artifacts/runs/<run_id>/`:

| Path | Contents |
|---|---|
| `train/adapter/` | LoRA adapter weights and tokenizer files, saved with `save_pretrained` |
| `train/training_metrics.jsonl` | One `sommelier.training_metric.v1` record per logged step |
| `train_manifest.json` | The train stage manifest |

Each training metric record carries `step`, `epoch`, `train_loss`, `eval_loss`, `learning_rate`, `tokens_seen`, and `peak_gpu_memory_mb`. Train loss appears on optimizer steps; eval loss appears on the per-epoch validation passes. Peak GPU memory is measured once after training, so it appears on the final record only. Non-finite values fail closed: a NaN loss stops the run rather than being persisted, because a divergent run should not leave artifacts that parse cleanly.

The manifest records the formatted train and validation splits as inputs, with their digests. The test split is never read by this stage (invariant INV-DATA-003 in `qlora.py`): validation exists for eval loss during training, and the test split stays unseen until [evaluation](evaluation.md). Manifest structure and schemas are in the [artifact reference](../reference/artifacts.md).

## What it cost in practice

The reference run `nemotron-8b-full-3` trained on 15,000 examples for 2 epochs on one L40S. Batch size 4 with gradient accumulation 4 gives an effective batch of 16, which works out to 938 optimizer steps per epoch, 1,876 total, at about 5.9 seconds per step. The train stage took 10,996 seconds, just over three hours, with peak GPU memory of 26,306 MiB.

One environment detail matters at 4096-token sequences: the [remote wrapper](../guides/remote-execution.md) sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch initializes CUDA. Long-sequence batches fragment the CUDA allocator, and expandable segments prevent OOM from memory that is reserved but not allocatable. If you run training on your own GPU outside the wrapper, set it yourself.

For the full run record, including both evaluation passes and the final comparison, see [the reference run](../results/reference-run.md) and the [reproduction guide](../guides/reproduction.md).
