# Sommelier Comparison Report

The JSON report (`comparison_report.json`) is authoritative for automation; this document is a human rendering.

## Run Identity

- Run ID: `smoke-fixture-1`
- Evidence class: smoke run
- Created at: 2026-07-02T12:00:00+00:00
- Config digest: `cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc`
- Parser version: `sommelier.parser.v1`
- Decoding: `{"do_sample": false, "max_new_tokens": 512, "temperature": 0.0}`

## Split Summary

- Split: test
- Examples evaluated: 20
- Test split digest: `tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt`
- Prompt set digest: `pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp`

## Metrics

| Metric | Base | Adapter | Delta |
|--------|------|---------|-------|
| valid_json_rate | 0.5000 (10/20) | 0.9500 (19/20) | +0.4500 |
| function_name_accuracy | 0.4500 (9/20) | 0.9000 (18/20) | +0.4500 |
| argument_exact_match | 0.2500 (5/20) | 0.7500 (15/20) | +0.5000 |
| argument_f1 | 0.6000 (48/80) | 0.9000 (72/80) | +0.3000 |
| full_call_exact_match | 0.2000 (4/20) | 0.7000 (14/20) | +0.5000 |

## Runtime and Cost

- Hardware: A10G (source: config)
- Peak GPU memory: 15872 MiB
- Observed cost: unavailable (source: unavailable)
- eval-base: 60.0 s elapsed
- train: 180.5 s elapsed

## Reproduction

Using the resolved config stored in this run directory:

```bash
sommelier eval run --config config.resolved.yaml --model base --data formatted --out eval/base --run-id smoke-fixture-1
sommelier train run --config config.resolved.yaml --data formatted --out train/adapter --run-id smoke-fixture-1
sommelier eval run --config config.resolved.yaml --model adapter --adapter train/adapter --data formatted --out eval/adapter --run-id smoke-fixture-1
sommelier report compare --base eval/base --adapter eval/adapter --out report
```

Generation artifacts: `runs/smoke-fixture-1/eval/base/generations.jsonl` (base), `runs/smoke-fixture-1/eval/adapter/generations.jsonl` (adapter).

## Limitations

- Metrics measure schema-valid single tool calls on the configured held-out test split only; multi-call plans are out of scope for v1.0.
- Argument comparisons are exact canonical-JSON matches; semantically equivalent but differently formatted values count as mismatches.
- Results describe the recorded run (hardware, dependencies, dataset revision) and do not claim production readiness, broad reliability, or generalization beyond the evaluated split.
- Parse failures count against every metric; raw generations are retained for audit.
