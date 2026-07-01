# RFC-0004: Adapter Training Contract

- Status: Accepted
- Authors: maintainers
- Created: 2026-07-01
- Target milestone: v0.4

## Summary

Sommelier trains a LoRA adapter with 4-bit base-model loading, bfloat16 compute where supported, assistant-token-only loss, and manifest-backed checkpoint artifacts. The training stage is configurable but must not silently change hyperparameters to fit hardware.

## Motivation

The PRD requires QLoRA on a single GPU with configurable hyperparameters. The performance spec requires bounded runtime and memory reporting. The architecture spec requires training to write an adapter artifact without owning evaluation metrics.

## Goals

- Train a parameter-efficient adapter on formatted examples.
- Compute loss only on assistant tool-call target tokens.
- Save adapter weights, tokenizer metadata, training metrics, and manifest.
- Fail with actionable resource errors instead of silently changing batch size or sequence length.
- Keep validation split separate from test split.

## Non-Goals

- Full-parameter fine-tuning.
- Multi-GPU or distributed training.
- Preference optimization.
- Automatic hyperparameter search.

## Proposed Design

### Training Interface

```python
def train_adapter(
    config: SommelierConfig,
    formatted_dir: Path,
    out_dir: Path,
) -> StageManifest: ...
```

The function reads `formatted/train.jsonl` and `formatted/validation.jsonl`, validates their manifests, loads the configured base model, applies LoRA, and writes:

```text
train/adapter/
train/training_metrics.jsonl
train/train_manifest.json
```

### Default Adapter Settings

```yaml
train:
  quantization: nf4-4bit
  compute_dtype: bfloat16
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### Label Masking

```python
def build_completion_labels(
    input_ids: list[int],
    prompt_token_count: int,
    ignore_index: int = -100,
) -> list[int]: ...
```

All prompt tokens receive `ignore_index`; assistant target tokens keep their token IDs. The stage fails if the formatter cannot provide a reliable prompt boundary.

### Metrics

Training writes JSONL records:

```python
class TrainingMetric(TypedDict):
    step: int
    epoch: float
    train_loss: float | None
    eval_loss: float | None
    learning_rate: float
    tokens_seen: int
    peak_gpu_memory_mb: int | None
```

### Resource Handling

Out-of-memory errors map to `ResourceError` with the current batch size, sequence length, gradient accumulation, and suggested config fields to reduce. The command does not retry with altered values.

## Alternatives Considered

- Full-parameter fine-tuning. Rejected because it violates the single-GPU and low-cost constraints.
- Prompt-token loss. Rejected because the target behavior is only the assistant JSON tool call.
- Automatic batch-size search. Rejected because it makes runs harder to compare and can hide cost changes.
- Merge adapter into base model during training. Rejected because evaluation and optional serving should control merge behavior explicitly.

## Drawbacks

- LoRA target modules are model-family-specific.
- Assistant-token-only masking adds implementation complexity.
- Strict resource handling may require users to tune config manually.

## Migration / Rollout

1. Add local tests for label masking.
2. Add training config validation.
3. Implement a one-step smoke training fixture behind optional training dependencies.
4. Implement remote training function.
5. Add full adapter artifact manifest and metrics output.

## Testing Strategy

- Unit-test label masking on synthetic token sequences.
- Unit-test config validation for LoRA rank, dropout, and target modules.
- Smoke-test one training step on a tiny formatted dataset.
- Verify `train_manifest.json` records input split and prompt digests.
- Verify OOM mapping in a mocked training exception.

## Open Questions

None for v1.0.

## References

- [01-architecture](../spec/01-architecture.md)
- [08-performance-budget](../spec/08-performance-budget.md)
