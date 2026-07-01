# 08 Performance Budget

- Status: Draft
- Target milestone: v1.0
- Primary RFCs: [RFC-0002](../rfcs/RFC-0002-dataset-preparation-and-split-discipline.md), [RFC-0004](../rfcs/RFC-0004-adapter-training-contract.md), [RFC-0007](../rfcs/RFC-0007-remote-gpu-orchestration.md)

## Reference Budget

The default full run targets:

| Stage | Budget |
|-------|--------|
| data preparation | under 20 minutes |
| formatting | under 15 minutes |
| base evaluation | under 2 GPU hours |
| adapter training | under 4 GPU hours |
| adapter evaluation | under 2 GPU hours |
| total observed cost | under 30 USD for the reference run when market pricing allows |

Cost is reported, not guaranteed. Runtime and cost vary by GPU type, queue time, package cache state, and remote provider pricing.

## Data Sizes

Default split sizes:

```yaml
data:
  n_train: 15000
  n_validation: 1000
  n_test: 1000
```

The smaller experiment note used 10000 training rows. The v1.0 default uses 15000 from the PRD spec, but config may reduce this for smoke runs.

## Memory Budget

| Component | Target |
|-----------|--------|
| sequence length | 2048 tokens default |
| quantization | 4-bit NF4 for training |
| compute dtype | bfloat16 where supported |
| LoRA rank | 16 default |
| per-device batch | 8 default, reduced on 24 GB GPUs |
| gradient accumulation | 2 default |

If the configured GPU cannot fit defaults, the training command fails with a resource hint rather than silently changing hyperparameters.

## Profiling

Each remote stage records:

- wall-clock seconds,
- peak GPU memory if available,
- input and output row counts,
- tokens processed,
- hardware type,
- package install/cache time when measurable.

Profiling data is written to the stage manifest and report JSON.

## Regression Thresholds

The local CI suite enforces only deterministic, non-GPU performance checks. Full GPU performance thresholds are advisory until enough reference runs establish variance.

## Optimization Rules

- Preserve correctness and auditability over throughput.
- Do not drop examples for speed except through declared filters.
- Do not change evaluation decoding or parser behavior for speed after a baseline has been recorded.
- Document any performance workaround in the run report.
