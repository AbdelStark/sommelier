# Installation

Sommelier installs in seconds on any machine because the package deliberately contains no ML stack. The local install gives you the CLI, config and schema validation, data and formatting stages, and launchers for remote work. Model training, evaluation, local-model translation, and semantic review live in GPU images; the OpenAI Responses translation producer lives in a separate CPU-only image.

## Prerequisites

- Python 3.13 or newer ([`pyproject.toml`](https://github.com/AbdelStark/sommelier/blob/main/pyproject.toml) declares `requires-python = ">=3.13"`).
- [uv](https://docs.astral.sh/uv/), which manages the virtual environment and the lockfile.
- git.

A Modal account is not needed for anything on this page or the [quickstart](quickstart.md). It becomes relevant when you run a rented GPU stage or the CPU-only provider translation producer; see [Remote execution](../guides/remote-execution.md).

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
| `modal>=1.5.1` | Launching remote CPU/GPU runs from your machine |
| `pydantic>=2.0` | Config validation and every artifact schema |
| `pyyaml>=6.0` | Reading config files |

torch, transformers, tokenizers, accelerate, peft, bitsandbytes, datasets, and
the OpenAI SDK are not local project dependencies at any level. They are
installed only inside the remote Modal images defined in
[`sommelier/remote/images.py`](https://github.com/AbdelStark/sommelier/blob/main/sommelier/remote/images.py).
`huggingface_hub` is isolated in the `publish` extra: publication validation
does not import it, and an authenticated Hub mutation imports it only after the
operator passes `--execute`. The selected Hebrew producer's CPU image contains only
Python 3.13.3, `openai==2.45.0`, and `datasets==5.0.0`; it does not inherit the
CUDA/model stack. This split keeps the laptop and CI lean while each remote
boundary runs the same Sommelier package source. CI can gate the core without a
GPU or provider credential, and mypy deliberately treats optional heavy stacks
as untyped where installed so type checking is stable.

## Extras

| Extra | Installs | When you need it |
|---|---|---|
| `dev` | pytest, ruff, mypy, types-pyyaml, python-dotenv | The test suite and linters. Install it by default; the verification below assumes it. |
| `data-gpu` | cudf-cu12 | GPU dataframe coarse filtering via `sommelier data prepare --gpu`. Only meaningful on a CUDA host; the remote data image installs it for you. |
| `docs` | mkdocs-material | Building this documentation site locally |
| `publish` | huggingface-hub 1.23.0 | Executing the explicit, round-trip-verified Hugging Face dataset or adapter publication commands. Validation-only publication does not need it. |

Install the publication boundary only on the authenticated release host:

```bash
uv sync --extra publish
```

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

`import sommelier` never touches torch or CUDA, and a test keeps it that way: [`tests/test_imports.py`](https://github.com/AbdelStark/sommelier/blob/main/tests/test_imports.py) imports every module of the package in a clean interpreter and asserts that none of modal, cudf, torch, transformers, peft, bitsandbytes, accelerate, datasets, huggingface_hub, vllm, or wandb was loaded as a side effect. The OpenAI adapter also imports its SDK only inside the explicit provider factory, so importing the package does not require provider credentials or the SDK. Heavy/provider imports happen inside stage functions, at call time. When a stage needs a stack that is not installed, it raises `ExternalDependencyError` (exit code 3) naming the missing packages, rather than an `ImportError` traceback; the [quickstart](quickstart.md) shows this on real commands, and [Errors and exit codes](../reference/errors.md) lists the full taxonomy.

## Next

Run the [quickstart](quickstart.md): the first two pipeline stages on your laptop, followed by a guided read of the artifacts they write.
