# RFC-0001: Project Configuration and Run Manifest

- Status: Accepted
- Authors: maintainers
- Created: 2026-07-01
- Target milestone: v0.1

## Summary

Sommelier uses a single strict YAML configuration file and a schema-versioned run manifest as the authoritative record for every pipeline execution. This locks reproducibility around one resolved config digest, one run identifier, and one manifest chain rather than relying on ad hoc command arguments or remote service state.

## Motivation

The PRD requires a single configuration file, fixed random seed, pinned dependencies, and reproducible evaluation. The overview and data model specs define artifact provenance as a v1.0 success criterion. Without a strict config and manifest contract, two runs can differ silently in split sizes, prompt policy, model revision, decoding, or adapter hyperparameters.

## Goals

- Define `SommelierConfig` as the only source of tunable pipeline settings.
- Reject unknown config fields and invalid combinations early.
- Write a resolved config and manifest for every run.
- Make config, command, commit, dependency lock, input artifacts, output artifacts, and seed auditable.
- Keep secrets out of config and manifests.

## Non-Goals

- Build a general experiment manager.
- Support multiple concurrent config formats.
- Infer missing model, dataset, or training settings from remote service defaults.

## Proposed Design

### Configuration Schema

```python
class ProjectConfig(BaseModel):
    name: str
    artifact_root: Path
    seed: int = 42

class ModelConfig(BaseModel):
    base_model_id: str
    base_model_revision: str
    tokenizer_revision: str
    allow_remote_code: bool = False
    remote_code_reason: str | None = None

class DatasetConfig(BaseModel):
    dataset_id: str
    dataset_revision: str
    query_column: str = "query"
    tools_column: str = "tools"
    answers_column: str = "answers"

class DataConfig(BaseModel):
    n_train: int = 15000
    n_validation: int = 1000
    n_test: int = 1000
    min_query_chars: int = 10
    max_query_chars: int = 2000
    dedupe_key: Literal["normalized_query"] = "normalized_query"

class FormattingConfig(BaseModel):
    system_prompt: str
    template_policy: Literal["tokenizer_chat_template"] = "tokenizer_chat_template"
    target_format: Literal["json_tool_call"] = "json_tool_call"

class TrainConfig(BaseModel):
    epochs: int = 2
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    scheduler: Literal["cosine"] = "cosine"
    warmup_ratio: float = 0.03
    max_sequence_length: int = 2048
    quantization: Literal["nf4-4bit"] = "nf4-4bit"
    compute_dtype: Literal["bfloat16"] = "bfloat16"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str]

class EvalConfig(BaseModel):
    split: Literal["test"] = "test"
    temperature: float = 0.0
    do_sample: bool = False
    max_new_tokens: int = 512
    parser_version: Literal["sommelier.parser.v1"] = "sommelier.parser.v1"

class RemoteConfig(BaseModel):
    enabled: bool = True
    gpu: str
    data_timeout_seconds: int = 1800
    train_timeout_seconds: int = 14400
    eval_timeout_seconds: int = 7200

class ReportConfig(BaseModel):
    retain_raw_generations: bool = True
    redact_fields: list[str] = []

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

The loader resolves relative paths against the config file directory, normalizes defaults, rejects unknown fields, and writes `config.resolved.yaml`. The SHA-256 digest of this resolved file is the config digest used in manifests.

### Manifest Schema

```python
class ArtifactRef(TypedDict):
    path: str
    kind: str
    schema_version: str
    sha256: str
    bytes: int

class StageManifest(TypedDict):
    schema_version: Literal["sommelier.manifest.v1"]
    stage: Literal["data", "format", "train", "eval", "report", "serve"]
    run_id: str
    created_at: str
    git_commit: str
    config_sha256: str
    dependency_lock_sha256: str | None
    command: list[str]
    seed: int
    inputs: list[ArtifactRef]
    outputs: list[ArtifactRef]
    status: Literal["succeeded", "failed"]
```

Every stage writes to a temporary directory, validates outputs, computes checksums, and then atomically publishes outputs and the manifest.

### Determinism

The config seed is passed to data shuffling, training framework seed setup, and any deterministic fixture generation. Evaluation uses greedy decoding and does not rely on a random seed.

### Secret Exclusion

Config fields must not contain token-like keys or values. Secrets are provided through environment variables or remote secret stores. Manifest writers scan values before writing and fail on suspected secrets.

## Alternatives Considered

- Command-line flags as the primary configuration surface. Rejected because flags are harder to hash, audit, and reuse across remote stages.
- Multiple config files per stage. Rejected because cross-stage drift would be easier.
- Relying on external experiment tracking for provenance. Rejected because local artifacts must remain complete without third-party availability.

## Drawbacks

- Strict validation makes early experimentation less forgiving.
- The resolved config can become verbose.
- Any schema change requires migration discipline.

## Migration / Rollout

1. Add `sommelier.config` and `sommelier.manifests`.
2. Add `examples/config.smoke.yaml`.
3. Implement `sommelier config validate`.
4. Update every stage to accept `SommelierConfig` rather than individual loose arguments.
5. Add manifest writing after each stage.

## Testing Strategy

- Unit-test unknown-field rejection and path resolution.
- Unit-test config digest stability.
- Unit-test secret scanner failures.
- Round-trip a successful and failed manifest fixture.
- Verify absolute artifact paths are rejected.
- Verify all CLI stage commands write a manifest in fixture mode.

## Open Questions

None for v1.0.

## References

- [SPEC.md](../../SPEC.md)
- [03-data-model](../spec/03-data-model.md)
- [05-observability](../spec/05-observability.md)
