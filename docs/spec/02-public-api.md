# 02 Public API

- Status: Draft
- Target milestone: v1.0
- Primary RFC: [RFC-0008](../rfcs/RFC-0008-cli-and-python-public-api.md)

## Public Surface

Sommelier exposes:

- A CLI named `sommelier`.
- Typed Python functions under the `sommelier` package.
- Schema-versioned JSON and JSONL artifacts.
- Markdown and JSON evaluation reports.

Anything outside the documented package modules is internal and may change before v1.0.

## CLI Contract

```text
sommelier config validate --config config.yaml
sommelier data prepare --config config.yaml --out artifacts/data
sommelier data validate-fixtures
sommelier format build --config config.yaml --data artifacts/data --out artifacts/formatted
sommelier eval run --config config.yaml --model base --data artifacts/formatted --out artifacts/eval/base
sommelier train run --config config.yaml --data artifacts/formatted --out artifacts/train/adapter
sommelier eval run --config config.yaml --model adapter --adapter artifacts/train/adapter --data artifacts/formatted --out artifacts/eval/adapter
sommelier report compare --base artifacts/eval/base --adapter artifacts/eval/adapter --out artifacts/report
sommelier pipeline run --config config.yaml --mode smoke|full
sommelier serve adapter --config config.yaml --adapter artifacts/train/adapter
```

CLI commands return:

- `0`: success.
- `2`: invalid user input, configuration, schema, or missing file.
- `3`: external dependency unavailable or unauthenticated.
- `4`: runtime failure that can be retried after resource changes.
- `5`: invariant violation or suspected code defect.

## Python API Contract

```python
from pathlib import Path
from sommelier.config import SommelierConfig, load_config
from sommelier.data.prepare import prepare_dataset
from sommelier.formatting.chat import build_formatted_splits
from sommelier.training.qlora import train_adapter
from sommelier.evaluation.generate import evaluate_model
from sommelier.evaluation.report import compare_evaluations

def load_config(path: Path) -> SommelierConfig: ...

def prepare_dataset(config: SommelierConfig, out_dir: Path) -> StageManifest: ...

def build_formatted_splits(
    config: SommelierConfig,
    data_dir: Path,
    out_dir: Path,
) -> StageManifest: ...

def train_adapter(
    config: SommelierConfig,
    formatted_dir: Path,
    out_dir: Path,
) -> StageManifest: ...

def evaluate_model(
    config: SommelierConfig,
    formatted_dir: Path,
    out_dir: Path,
    model_kind: Literal["base", "adapter"],
    adapter_dir: Path | None = None,
) -> StageManifest: ...

def compare_evaluations(base_dir: Path, adapter_dir: Path, out_dir: Path) -> StageManifest: ...
```

The API accepts `pathlib.Path` for filesystem paths and returns manifests instead of printing-only results. Functions may log progress, but their return values are the authoritative structured outputs.

## Configuration API

```python
class SommelierConfig(BaseModel):
    schema_version: Literal["sommelier.config.v1"]
    project: ProjectConfig
    model: ModelConfig
    dataset: DatasetConfig
    data: DataConfig
    formatting: FormattingConfig
    train: TrainConfig
    eval: EvalConfig
    remote: RemoteConfig
    report: ReportConfig
```

Validation is strict. Unknown fields are rejected to avoid silently misspelled hyperparameters.

## Versioning Policy

- Before `1.0.0`, documented Python APIs may change only with changelog entries and migration notes.
- At `1.0.0`, documented modules, CLI command names, artifact schemas, and metric names become stable.
- Artifact schema changes require a new `schema_version` and migration notes.
- The CLI may add flags in minor releases but must not change default semantics without a deprecation window.

## Deprecation Policy

Deprecations require:

1. A warning in the CLI and Python API.
2. A changelog entry.
3. A replacement path.
4. At least one minor release before removal after `1.0.0`.

## Public Documentation Requirements

The README must link to this corpus, show local validation commands, state GPU prerequisites for remote stages, and avoid implying that the reference result is production-ready.
