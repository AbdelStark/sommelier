# Sommelier

Sommelier turns a small open language model into a reliable JSON tool caller and proves the improvement. One CLI runs the whole path: prepare data, render prompts, measure tokenizer cost, evaluate the base model, train a QLoRA adapter, evaluate the adapter, and compare both sides under a gate that refuses to compare anything that is not provably identical.

The project exists because most fine-tuning writeups ask you to trust them. Sommelier is built on the opposite premise: a fine-tuning claim is only as good as its evidence. Every stage writes schema-versioned, checksummed artifacts. Prompts carry digests. Decoding is deterministic. The final report cannot be produced unless base and adapter were scored on the same test split, the same prompts, the same parser, and the same decoding settings.

## The reference result

QLoRA on [nvidia/Llama-3.1-Nemotron-Nano-8B-v1](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1), 15,000 training examples, 2 epochs, one L40S GPU, about 3 hours of training. Both models evaluated greedily on the same 1,000 held-out prompts with a conservative parser that counts every parse failure as a failure:

| Metric | Base | Adapter | Delta |
|--------|------|---------|-------|
| Valid JSON rate | 0.916 | **1.000** | +0.084 |
| Function name accuracy | 0.911 | **0.996** | +0.085 |
| Argument exact match | 0.707 | **0.876** | +0.169 |
| Argument F1 | 0.757 | **0.929** | +0.172 |
| Full-call exact match | 0.705 | **0.874** | +0.169 |

Everything needed to verify or reproduce this is public: the [adapter and its evaluation evidence](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora), the [exact train/validation/test splits](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits), and the [reproduction guide](guides/reproduction.md).

These numbers hold for the recorded dataset revision, prompt policy, parser, and decoding config. They are not claims of production readiness or general agent reliability. [The reference run](results/reference-run.md) states the full claim and its boundaries.

## Where to go

<div class="grid cards" markdown>

- **[Quickstart](getting-started/quickstart.md)**

    Run the first two pipeline stages on your laptop in fixture mode. No GPU, no accounts, no downloads.

- **[Concepts](concepts/pipeline.md)**

    How the seven stages fit together, and why the pipeline refuses to do certain things.

- **[Reproduce the reference run](guides/reproduction.md)**

    From a clean checkout to the comparison report on a rented GPU, with costs and caveats stated up front.

- **[Reference](reference/cli.md)**

    Every command, config field, artifact schema, error code, and metric definition.

</div>

## What Sommelier is not

Scope discipline is part of the method, so the boundaries are explicit:

- It trains and scores exactly one tool call per request. Multi-call answers are filtered out of the data, and the filter is recorded in the drop summary.
- It is not a production serving system. The bundled endpoints exist to inspect the adapter, and their limits are [documented](guides/serving.md).
- It is not an agent framework. There is no multi-turn planning, no tool execution, no retry logic.
- It makes no claims against public benchmarks or frontier hosted models. The only claim is base versus adapter, under identical conditions, on this task.

If you want the reasoning behind these boundaries, read [Design decisions](concepts/design-decisions.md).
