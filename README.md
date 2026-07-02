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

The package exposes configuration validation, dataset preparation with deterministic splits, fixture-mode stage stubs, and a Modal smoke app. The remote app lives in `sommelier.remote.app` (modal imported lazily, per the optional extras boundary); `sommelier_entrypoint.py` is a thin compatibility wrapper that materializes it.

### Commands

| Command | Status |
|---------|--------|
| `sommelier config validate` | Implemented |
| `sommelier data prepare` | Implemented (raw JSONL input or `--fixture`) |
| `sommelier data validate-fixtures` | Implemented |
| `sommelier format build` | Implemented (tokenizer template; `--fixture` for no-tokenizer builds) |
| `sommelier eval run` | Pending (#26) — fails with an explicit not-implemented error |
| `sommelier train run` | Pending (#31) — fails with an explicit not-implemented error |
| `sommelier report compare` | Pending (#27) — fails with an explicit not-implemented error |
| `sommelier pipeline run` | Pending (#35) — fails with an explicit not-implemented error |
| `sommelier serve adapter` | Pending (#40) — fails with an explicit not-implemented error |

Command names and flags follow [docs/spec/02-public-api.md](docs/spec/02-public-api.md#cli-contract).

```bash
uv sync --extra dev
uv run ruff check .
uv run mypy sommelier tests
uv run pytest
uv run sommelier config validate --config examples/config.smoke.yaml
uv run sommelier data validate-fixtures
uv run sommelier data prepare --config examples/config.smoke.yaml --input tests/fixtures/preparation_rows.jsonl --out artifacts/runs/local/data --run-id local
uv run sommelier data prepare --config examples/config.smoke.yaml --fixture --out artifacts/runs/local/data --run-id local
uv run python sommelier_entrypoint.py
```

Optional GPU coarse filtering is available with `uv sync --extra data-gpu` and the `--gpu` flag on `sommelier data prepare`.

### Optional extras boundary

`import sommelier` never imports GPU, remote execution, or tracking
packages. Heavy dependencies stay behind optional extras (for example
`data-gpu`) and are imported inside stage functions only when a command
needs them, so contributors on non-GPU machines can run the full local
suite. `tests/test_imports.py` enforces this boundary in CI.

## Diagram

![AI Agent lifecycle](./docs/img/gtcdc25-nemo-diagram.png)

## License

Sommelier is released under the [MIT License](LICENSE). Third-party model,
dataset, and package obligations are recorded in
[licenses/THIRD_PARTY.md](licenses/THIRD_PARTY.md).
