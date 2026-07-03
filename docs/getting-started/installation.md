# Installation

Sommelier installs in seconds on any machine because the package deliberately contains no ML stack. The local install gives you the CLI, the config and schema validation, the data and formatting stages, and the launcher for remote GPU runs. Everything heavy lives in container images that only ever exist on the GPU host.

## Prerequisites

- Python 3.13 or newer ([`pyproject.toml`](https://github.com/AbdelStark/sommelier/blob/main/pyproject.toml) declares `requires-python = ">=3.13"`).
- [uv](https://docs.astral.sh/uv/), which manages the virtual environment and the lockfile.
- git.

A Modal account is not needed for anything on this page or the [quickstart](quickstart.md). It becomes relevant only when you run stages on a rented GPU; see [Remote execution](../guides/remote-execution.md).

## Install

```bash
git clone https://github.com/AbdelStark/sommelier
cd sommelier
uv sync --extra dev
```

`uv sync` creates a project-local `.venv` and installs the exact versions recorded in the committed `uv.lock`. The lockfile is part of the provenance story: every stage manifest records `dependency_lock_sha256`, the SHA-256 of `uv.lock` at the time the stage ran, so any artifact can be traced back to the dependency set that produced it. See [Artifacts and schemas](../reference/artifacts.md).

## What the base install contains

Three runtime dependencies:

| Package | Why it is a base dependency |
|---|---|
| `modal>=1.5.1` | Launching remote GPU runs from your machine |
| `pydantic>=2.0` | Config validation and every artifact schema |
| `pyyaml>=6.0` | Reading config files |

torch, transformers, peft, trl, and bitsandbytes are not dependencies at any level, not even as optional extras. They are installed only inside the remote Modal images defined in [`sommelier/remote/images.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/remote/images.py). The split keeps the laptop, CI, and the GPU host running the same package: CI can gate every change without provisioning a GPU, and mypy deliberately treats the heavy stacks as untyped even where they are installed, so type checking gives the same result everywhere.

## Extras

| Extra | Installs | When you need it |
|---|---|---|
| `dev` | pytest, ruff, mypy, types-pyyaml, python-dotenv | The test suite and linters. Install it by default; the verification below assumes it. |
| `data-gpu` | cudf-cu12 | GPU dataframe coarse filtering via `sommelier data prepare --gpu`. Only meaningful on a CUDA host; the remote data image installs it for you. |
| `docs` | mkdocs-material | Building this documentation site locally |

## Verify the install

These are the same checks CI runs on every pull request:

```bash
uv run pytest
uv run sommelier config validate --config examples/config.smoke.yaml
uv run sommelier data validate-fixtures
```

The whole suite runs on a GPU-free machine in a few seconds. The two CLI checks print:

```text
config ok: examples/config.smoke.yaml
fixtures ok: tests/fixtures
```

`config validate` parses and strictly validates the example config (unknown keys are rejected; see [Configuration](../reference/configuration.md)). `data validate-fixtures` re-reads the synthetic JSONL fixtures under `tests/fixtures/` and checks every record against its declared schema. For full CI parity, add `uv run ruff check .` and `uv run mypy sommelier tests`.

## The import discipline

`import sommelier` never touches torch or CUDA, and a test keeps it that way: [`tests/test_imports.py`](https://github.com/AbdelStark/sommelier/blob/main/tests/test_imports.py) imports every module of the package in a clean interpreter and asserts that none of modal, cudf, torch, transformers, trl, peft, bitsandbytes, accelerate, datasets, vllm, or wandb was loaded as a side effect. Heavy imports happen inside stage functions, at call time. When a stage needs a stack that is not installed, it raises `ExternalDependencyError` (exit code 3) naming the missing packages, rather than an `ImportError` traceback; the [quickstart](quickstart.md) shows this on real commands, and [Errors and exit codes](../reference/errors.md) lists the full taxonomy.

## Next

Run the [quickstart](quickstart.md): the first two pipeline stages on your laptop, followed by a guided read of the artifacts they write.
