from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sommelier.artifacts import write_artifact_atomic
from sommelier.errors import ConfigError, SecurityPolicyError
from sommelier.security import validate_no_secrets

RESOLVED_CONFIG_NAME = "config.resolved.yaml"
CONFIG_SCHEMA_VERSION = "sommelier.config.v2"
V1_CONFIG_SCHEMA_VERSION = "sommelier.config.v1"

LANGUAGE_PATTERN = re.compile(r"^[a-z]{2}$")


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


class DatasetSourceConfig(BaseModel):
    """One language's source dataset.

    The source without ``source_id_column`` is the root source: it gets
    independent split assignment. A source with ``source_id_column`` is
    paired: each of its rows names a root row and inherits that row's split.
    """

    model_config = ConfigDict(extra="forbid")

    language: str
    dataset_id: str
    dataset_revision: str
    query_column: str = "query"
    tools_column: str = "tools"
    answers_column: str = "answers"
    source_id_column: str | None = None

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if not LANGUAGE_PATTERN.fullmatch(value):
            raise ValueError("language must be a two letter lowercase ISO 639-1 code")
        return value


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
    languages: list[str] = Field(default_factory=list)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: Literal["test"] = "test"
    slices: list[str] = Field(default_factory=lambda: ["en"])
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

    schema_version: Literal["sommelier.config.v2"]
    project: ProjectConfig
    model: ModelConfig
    datasets: list[DatasetSourceConfig]
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

    @model_validator(mode="after")
    def validate_dataset_sources(self) -> SommelierConfig:
        if not self.datasets:
            raise ValueError("datasets must list at least one source")
        languages = [source.language for source in self.datasets]
        duplicates = sorted({lang for lang in languages if languages.count(lang) > 1})
        if duplicates:
            raise ValueError(f"duplicate dataset language: {', '.join(duplicates)}")
        roots = [source for source in self.datasets if source.source_id_column is None]
        if len(roots) != 1:
            raise ValueError(
                "exactly one dataset source must omit source_id_column (the root source)"
            )
        return self

    @model_validator(mode="after")
    def resolve_language_references(self) -> SommelierConfig:
        configured = [source.language for source in self.datasets]
        if not self.train.languages:
            self.train.languages = list(configured)
        for field_name, values in (
            ("train.languages", self.train.languages),
            ("eval.slices", self.eval.slices),
        ):
            duplicates = sorted({value for value in values if values.count(value) > 1})
            if duplicates:
                raise ValueError(f"duplicate entry in {field_name}: {', '.join(duplicates)}")
            unknown = [value for value in values if value not in configured]
            if unknown:
                raise ValueError(
                    f"{field_name} references a language with no dataset source: "
                    f"{', '.join(unknown)}"
                )
        return self

    @property
    def root_dataset(self) -> DatasetSourceConfig:
        for source in self.datasets:
            if source.source_id_column is None:
                return source
        raise ValueError("no root dataset source configured")

    def dataset_for(self, language: str) -> DatasetSourceConfig:
        for source in self.datasets:
            if source.language == language:
                return source
        raise ValueError(f"no dataset source configured for language {language!r}")


def upgrade_v1_document(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrades a sommelier.config.v1 mapping to the v2 shape in memory.

    The single ``dataset`` section becomes the one ``en`` entry under
    ``datasets``; ``train.languages`` and ``eval.slices`` take their v2
    defaults, which resolve to English only.
    """
    upgraded = dict(raw)
    upgraded["schema_version"] = CONFIG_SCHEMA_VERSION
    if "dataset" in upgraded and "datasets" not in upgraded:
        source = dict(upgraded.pop("dataset"))
        source["language"] = "en"
        upgraded["datasets"] = [source]
    return upgraded


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

    if raw.get("schema_version") == V1_CONFIG_SCHEMA_VERSION:
        warnings.warn(
            f"{path}: {V1_CONFIG_SCHEMA_VERSION} is deprecated; migrate the file to "
            f"{CONFIG_SCHEMA_VERSION} (the dataset section becomes the one en entry "
            "under datasets)",
            DeprecationWarning,
            stacklevel=2,
        )
        raw = upgrade_v1_document(raw)

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
