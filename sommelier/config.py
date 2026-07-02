from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sommelier.artifacts import write_artifact_atomic
from sommelier.errors import ConfigError, SecurityPolicyError
from sommelier.security import validate_no_secrets

RESOLVED_CONFIG_NAME = "config.resolved.yaml"


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    artifact_root: Path
    seed: int = 42

    @field_validator("artifact_root", mode="before")
    @classmethod
    def normalize_artifact_root(cls, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("artifact_root must be relative to the config file directory")
        return path


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model_id: str
    base_model_revision: str
    tokenizer_revision: str
    allow_remote_code: bool = False
    remote_code_reason: str | None = None


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_revision: str
    query_column: str = "query"
    tools_column: str = "tools"
    answers_column: str = "answers"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_train: int = 15000
    n_validation: int = 1000
    n_test: int = 1000
    min_query_chars: int = 10
    max_query_chars: int = 2000
    dedupe_key: Literal["normalized_query"] = "normalized_query"


class FormattingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    template_policy: Literal["tokenizer_chat_template"] = "tokenizer_chat_template"
    target_format: Literal["json_tool_call"] = "json_tool_call"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    split: Literal["test"] = "test"
    temperature: float = 0.0
    do_sample: bool = False
    max_new_tokens: int = 512
    parser_version: Literal["sommelier.parser.v1"] = "sommelier.parser.v1"


class RemoteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    gpu: str
    data_timeout_seconds: int = 1800
    train_timeout_seconds: int = 14400
    eval_timeout_seconds: int = 7200


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retain_raw_generations: bool = True
    redact_fields: list[str] = Field(default_factory=list)


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: Literal["wandb"] = "wandb"
    project: str = "sommelier"


class SommelierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @model_validator(mode="after")
    def validate_remote_code_reason(self) -> SommelierConfig:
        if self.model.allow_remote_code and not self.model.remote_code_reason:
            raise ValueError("remote_code_reason is required when allow_remote_code is true")
        return self

    @model_validator(mode="after")
    def validate_split_sizes(self) -> SommelierConfig:
        if self.data.n_train <= 0 or self.data.n_validation <= 0 or self.data.n_test <= 0:
            raise ValueError("n_train, n_validation, and n_test must be positive")
        if self.data.min_query_chars <= 0 or self.data.max_query_chars <= self.data.min_query_chars:
            raise ValueError("max_query_chars must be greater than min_query_chars")
        return self


def _dump_resolved_config(config: SommelierConfig) -> str:
    payload = config.model_dump(mode="json")
    payload["project"]["artifact_root"] = config.project.artifact_root.as_posix()
    return yaml.safe_dump(payload, sort_keys=False)


def compute_config_digest(resolved_yaml: str) -> str:
    return hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()


def load_config(path: Path) -> SommelierConfig:
    if not path.exists():
        raise ConfigError(
            f"config file not found: {path}",
            hint="Pass --config with the path to a Sommelier YAML config.",
        )

    raw_text = path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as error:
        raise ConfigError(
            f"invalid YAML in {path}: {error}",
            hint="Fix the YAML syntax and rerun validation.",
        ) from error

    if not isinstance(raw, dict):
        raise ConfigError(
            f"config file must contain a mapping: {path}",
            hint="Use the example configs under examples/ as a template.",
        )

    try:
        validate_no_secrets(raw, context="config")
    except SecurityPolicyError:
        raise

    try:
        config = SommelierConfig.model_validate(raw)
    except SecurityPolicyError:
        raise
    except Exception as error:
        raise ConfigError(
            f"invalid config: {error}",
            hint="Compare your config against examples/config.smoke.yaml.",
        ) from error

    validate_no_secrets(config.model_dump(mode="json"), context="config")
    return config


def write_resolved_config(
    config: SommelierConfig,
    out_dir: Path,
) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = out_dir / RESOLVED_CONFIG_NAME
    resolved_yaml = _dump_resolved_config(config)
    validate_no_secrets(yaml.safe_load(resolved_yaml), context="resolved config")

    def writer(temp_path: Path) -> None:
        temp_path.write_text(resolved_yaml, encoding="utf-8")

    write_artifact_atomic(resolved_path, writer)
    return resolved_path, compute_config_digest(resolved_yaml)
