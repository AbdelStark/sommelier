# The adapter and the base model

The pipeline never modifies the base model. What it trains, evaluates, and publishes is an adapter: a small set of extra weight matrices that sit beside the frozen base weights and add a correction to them. This page explains that relationship precisely: where the adapter attaches inside the transformer, what its weights are, how the two compose at training time and at inference time, and why this structure is what makes the project's evidence story possible. Every number below comes from the published artifacts themselves: the base model's `config.json`, the adapter's `adapter_config.json`, the tensor shapes inside `adapter_model.safetensors`, and the recorded runs. The operational training contract (loss masking, failure behavior, artifacts) is on the [training page](training.md); this page is about the mathematics and the architecture.

## The base model, as the adapter sees it

[nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1) is a `LlamaForCausalLM`: token embeddings, a stack of 32 identical transformer blocks, a final norm, and an output projection, about 8.03 billion parameters in bfloat16. Each block contains exactly seven linear projections, and these seven are the only places the adapter touches:

| Projection | Role | Weight shape (out x in) |
|------------|------|------------------------|
| `q_proj` | attention queries, 32 heads of dimension 128 | 4096 x 4096 |
| `k_proj` | attention keys, 8 shared key/value heads (grouped-query attention) | 1024 x 4096 |
| `v_proj` | attention values, same 8 shared heads | 1024 x 4096 |
| `o_proj` | attention output, back to the residual stream | 4096 x 4096 |
| `gate_proj` | MLP gate half of the SwiGLU activation | 14336 x 4096 |
| `up_proj` | MLP up projection | 14336 x 4096 |
| `down_proj` | MLP down projection, back to the residual stream | 4096 x 14336 |

Everything else stays untouched and frozen: embeddings, the output head, every norm, the attention mechanism itself. The adapter changes what the linear maps compute, never the wiring between them.

## What a LoRA adapter is

A trained weight update to a linear layer is itself a matrix: fine-tuning classically computes `W' = W + ΔW` with `ΔW` the same shape as `W`. LoRA (low-rank adaptation) constrains that update to a product of two thin matrices:

```text
h = W x  +  (α / r) · B A x
        └──────────┬────────┘
             the adapter branch
```

- `A` has shape `r x d_in`: it projects the layer input down to `r` dimensions.
- `B` has shape `d_out x r`: it projects those `r` dimensions back up.
- `ΔW = (α / r) · B A` therefore has rank at most `r`, however large the layer is.

The bet this encodes: the correction a fine-tune needs is low-rank. Adapting an 8B model to emit one strict JSON tool call does not require independently moving all 16.8 million entries of a `q_proj`; it requires a coordinated adjustment that can be expressed in a handful of directions. The result on the [reference run](../results/reference-run.md) and the [French run](../results/french-run.md) is the empirical support: rank 16, against layer dimensions of 1024 to 14336, carries the entire measured improvement.

Three details of the configuration, all recorded in the published `adapter_config.json`:

- **Rank and scaling.** `r=16`, `lora_alpha=32`, so the branch is scaled by `α / r = 2`. The scale decouples the learning rate from the rank: `α` fixes the magnitude the branch contributes at a given weight size, so changing `r` alone does not silently change how strongly the adapter speaks.
- **Initialization.** `A` starts random (Kaiming uniform, the peft default recorded as `init_lora_weights: true`) and `B` starts at zero. Their product is therefore exactly zero at step 0: training begins at precisely the behavior of the base it wraps and moves away from it only as gradients accumulate in `A` and `B`. Up to the 4-bit quantization of the frozen path at training time (the asymmetry noted below), the base-model evaluation the comparison is gated against is the training starting point.
- **Dropout and bias.** `lora_dropout=0.05` randomly zeroes the branch input during training only; at inference the branch is deterministic. `bias: "none"` means no bias terms train anywhere: the adapter is purely the 448 matrices described next.

## The weights, counted

`adapter_model.safetensors` in the published repositories contains 448 tensors: 32 layers times 7 projections times the pair `(lora_A, lora_B)`, stored in float32. Per layer:

| Projection | `lora_A` shape | `lora_B` shape | Parameters |
|------------|----------------|----------------|------------|
| `q_proj` | 16 x 4096 | 4096 x 16 | 131,072 |
| `k_proj` | 16 x 4096 | 1024 x 16 | 81,920 |
| `v_proj` | 16 x 4096 | 1024 x 16 | 81,920 |
| `o_proj` | 16 x 4096 | 4096 x 16 | 131,072 |
| `gate_proj` | 16 x 4096 | 14336 x 16 | 294,912 |
| `up_proj` | 16 x 4096 | 14336 x 16 | 294,912 |
| `down_proj` | 16 x 14336 | 4096 x 16 | 294,912 |
| per layer | | | 1,310,720 |

32 layers give 41,943,040 trainable parameters, 0.52 percent of the base model's 8.03 billion, 168 MB on disk in float32 against roughly 16 GB of bfloat16 base weights. One way to feel the size: the whole adapter holds exactly as many parameters as the four attention projections of a single transformer block (16,777,216 + 16,777,216 + 4,194,304 + 4,194,304 = 41,943,040). The correction that turns the base model into a reliable tool caller is, in parameter count, one thirty-second of one component of the network it corrects.

Note where the capacity went: the three MLP projections carry 68 percent of the adapter (294,912 each) because their frozen counterparts are the widest matrices in the block. Targeting all seven projections rather than attention only follows the QLoRA paper's finding that adapting every linear layer is what recovers full fine-tuning quality; the pipeline treats `train.target_modules` as part of the recorded configuration rather than a tunable default.

## Composition at training time

Training composes three precisions deliberately ([`qlora.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/training/qlora.py)):

1. **The frozen base is quantized to NF4.** Every linear layer's weight matrix is stored in 4-bit NormalFloat: weights are grouped in blocks of 64, each block keeps one absmax scale, and each weight is snapped to one of 16 levels placed at the quantiles of a standard normal distribution, which is close to how trained weights are actually distributed. Double quantization compresses the block scales themselves. The embeddings and the output head are not linear-layer weights in this sense and stay in bfloat16, so the 16 GB base fits in roughly 5.7 GB rather than the naive 4-bit arithmetic's 4 GB.
2. **Computation runs in bfloat16.** For every matmul, the touched blocks are dequantized on the fly to bfloat16 (`bnb_4bit_compute_dtype`). The quantization is a storage format, not a compute format.
3. **The adapter lives in float32.** peft creates and keeps the trainable `A` and `B` matrices in full precision over a 4-bit base, which is why the published safetensors are float32; `prepare_model_for_kbit_training` additionally casts the remaining non-quantized base parameters (the norms, embeddings, and output head) to float32 for numerical stability.

Gradients flow through the frozen dequantized weights but are only stored for `A` and `B`. That asymmetry is the whole memory story: the AdamW optimizer keeps its two moment tensors only for 41.9 million parameters instead of 8 billion, activations are recomputed through gradient checkpointing, and the measured peak for the [French run](../results/french-run.md) was 26,369 MiB on one L40S, training 28,113 examples at sequence length 4096.

## Composition at inference time

The pipeline never merges the adapter into the base. At evaluation, [`generate.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/evaluation/generate.py) loads the base model unquantized in bfloat16, then attaches the adapter with `PeftModel.from_pretrained`; every forward pass computes the frozen path and the adapter branch and adds them, exactly as during training. The optional [vLLM serving endpoint](../guides/serving.md) does the same at higher throughput: one deployment registers the base and the LoRA (`--enable-lora --max-lora-rank 16`), so base and adapter answer A/B requests from the same process.

Two consequences are worth stating precisely:

- **One honest asymmetry.** The adapter is optimized against the NF4-quantized base but evaluated and served against the bfloat16 base. Training against the quantized base is the original QLoRA recipe; evaluating and serving against the unquantized one is this pipeline's own deployment choice, matching how the adapter is actually used. Both sides of every comparison run on the same bfloat16 base, so the base-vs-adapter delta stays internally consistent; the small train-versus-inference difference in the frozen path is stated here rather than glossed over.
- **Merging is possible and deliberately unused.** `(α / r) · B A` could be added into `W` once, yielding a standalone checkpoint with zero inference overhead and no visible adapter. The pipeline keeps the two factored apart because the separation is the evidence: the base stays pinned by `model.base_model_revision`, the adapter is the entire diff, and anyone can reproduce the composition from the two published pieces. A merged model would bury a 168 MB claim inside a 16 GB artifact.

## Why this structure carries the evidence

The project's claim is always a comparison: the same base model, with and without the correction. The adapter structure makes that claim unusually clean.

- The base model is never written to, so "base" in every report means the pinned upstream artifact, byte for byte, not a locally mutated copy.
- The published adapter is the complete difference between the two systems being compared. The 168 MB file, its `adapter_config.json`, and the base revision pin fully determine the fine-tuned model.
- Zero-initialized `B` means the untrained adapter is a no-op, so the improvement measured by the [comparison gate](determinism.md) is attributable to training and nothing else.
- Several corrections can share one base. The [v1 English adapter](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora) and the [v2 bilingual adapter](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-fr-en-lora) are two 168 MB deltas on the same frozen 8B foundation, which is what made the [French run's](../results/french-run.md) three-way gap measurement cheap to produce and easy to verify.

## Reading the weights yourself

The tensor inventory above is checkable in a few lines against either published adapter, without loading a model:

```python
import json, struct
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-fr-en-lora",
    "adapter_model.safetensors",
)
with open(path, "rb") as f:
    header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
header.pop("__metadata__", None)

total = sum(
    s[0] * s[1] for s in (t["shape"] for t in header.values())
)
print(len(header), "tensors,", total, "parameters")
# 448 tensors, 41943040 parameters
```

The names follow the module path they attach to (`base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`), so the mapping from every tensor to its place in the architecture is explicit in the file itself.
