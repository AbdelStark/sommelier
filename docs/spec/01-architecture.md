# 01 Architecture

- Status: Draft
- Target milestone: v1.0
- Primary RFCs: [RFC-0001](../rfcs/RFC-0001-project-configuration-and-run-manifest.md), [RFC-0006](../rfcs/RFC-0006-artifact-store-and-schema-versioning.md), [RFC-0007](../rfcs/RFC-0007-remote-gpu-orchestration.md)

## System Shape

Sommelier is a command-line and Python package that coordinates a staged pipeline:

1. Validate configuration.
2. Prepare raw tool-calling rows into deterministic train, validation, and test splits.
3. Render rows through the selected chat template.
4. Evaluate the base model on the held-out test split.
5. Train a parameter-efficient adapter.
6. Evaluate the adapter with the same prompt and parser.
7. Write a comparison report and machine-readable manifests.
8. Optionally start a single-adapter inference service.

The stages communicate through schema-versioned files under an artifact root. No stage depends on hidden in-memory state from a previous stage.

## Package Layout

The v1.0 package layout is:

```text
sommelier/
  __init__.py
  cli.py
  config.py
  errors.py
  manifests.py
  artifacts.py
  data/
    load.py
    prepare.py
    validate.py
  formatting/
    chat.py
    templates.py
  training/
    qlora.py
    collators.py
  evaluation/
    generate.py
    parse.py
    metrics.py
    report.py
  remote/
    app.py
    images.py
    secrets.py
  serving/
    openai_compat.py
tests/
```

## Module Boundaries

| Module | Responsibility | Forbidden responsibility |
|--------|----------------|--------------------------|
| `config` | Load, validate, freeze, and hash configuration. | Reading datasets or launching training. |
| `data` | Download, validate, clean, deduplicate, split, and write JSONL splits. | Rendering chat prompts or scoring model outputs. |
| `formatting` | Convert validated examples into model-specific prompt/target records. | Loading model weights or changing splits. |
| `training` | Load the base model, apply adapter config, train, and save adapter artifacts. | Computing evaluation metrics. |
| `evaluation` | Generate outputs, parse tool calls, score metrics, and write reports. | Updating model weights. |
| `remote` | Define remote images, volumes, secrets, and execution entrypoints. | Encoding business logic that cannot run locally. |
| `serving` | Run the optional adapter-backed inference endpoint. | Advertising production deployment guarantees. |

## Data Flow

```text
SommelierConfig
  -> RawToolCallRow[]
  -> PreparedExample train/validation/test JSONL
  -> FormattedExample train/validation/test JSONL
  -> BaseModelEvaluation
  -> AdapterArtifact
  -> AdapterEvaluation
  -> EvaluationComparisonReport
```

Every transition writes a manifest containing:

```python
class StageManifest(TypedDict):
    schema_version: Literal["sommelier.manifest.v1"]
    stage: str
    run_id: str
    created_at: str
    git_commit: str
    config_sha256: str
    inputs: list[ArtifactRef]
    outputs: list[ArtifactRef]
    command: list[str]
    seed: int
    status: Literal["succeeded", "failed"]
```

## Execution Modes

- `local`: validates configuration, fixtures, schemas, formatting, parser, metrics, and report generation without GPU access.
- `remote-smoke`: runs a bounded sample, default `n=100`, through remote data, formatting, baseline generation, and one short training step.
- `remote-full`: runs the configured reference split sizes and writes the release report.

The local mode must not require GPU libraries at import time. GPU-only dependencies stay behind optional extras and remote entrypoints.

## Invariants

- `INV-ARCH-001`: Stages read only declared inputs and write only declared outputs.
- `INV-ARCH-002`: Every artifact path is relative to the configured artifact root.
- `INV-ARCH-003`: Every persisted artifact has a manifest entry and checksum.
- `INV-ARCH-004`: Base and adapter evaluation share formatter, parser, metric, decoding, and test split digests.
- `INV-ARCH-005`: Remote orchestration may wrap a stage but must not fork the stage contract.

## Dependency Strategy

The base package contains only local validation and CLI dependencies. GPU, training, evaluation, and serving stacks are optional extras:

```toml
[project.optional-dependencies]
data-gpu = ["cudf-cu12"]
train = ["torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate", "datasets"]
eval = ["vllm", "transformers", "datasets"]
remote = ["modal"]
tracking = ["wandb"]
dev = ["pytest", "ruff", "mypy", "hypothesis"]
```

Version pins are locked after the first green remote-smoke run and recorded in the release report.

## Failure Isolation

Each stage can fail independently and leaves a failed manifest with `status="failed"` when the process reaches Sommelier error handling. A failed stage must not overwrite a previous successful artifact directory. Partial writes go to a temporary directory and are atomically moved into place after validation.
