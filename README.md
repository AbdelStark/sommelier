# Sommelier

Sommelier is a reference implementation for fine-tuning a small open language model to emit schema-valid tool calls. The project is currently in specification bootstrap: the repository contains the canonical product requirements, specification corpus, and RFCs that define the v1.0 implementation work.

## Canonical Documents

- [Product requirements](prd.md)
- [Specification index](SPEC.md)
- [Detailed specification](docs/spec/00-overview.md)
- [RFC index](SPEC.md#rfc-index)

## Scope

The v1.0 target is a single-GPU pipeline that prepares one tool-calling dataset, formats examples through the selected chat template, evaluates the base model, trains a parameter-efficient adapter, evaluates the adapter with the same prompts and parser, and writes a comparison report.

The project does not claim production serving readiness, broad agent reliability, or superiority over larger hosted models.

## Current Code

The existing Python entrypoint is a remote execution smoke test. Implementation issues filed from the specification corpus track the package, CLI, data, formatting, training, evaluation, reporting, and release work required for v1.0.

```bash
uv run python sommelier_entrypoint.py
```

## Diagram

![AI Agent lifecycle](./docs/img/gtcdc25-nemo-diagram.png)
