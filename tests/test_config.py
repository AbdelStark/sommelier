from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from sommelier.config import (
    load_config,
    write_resolved_config,
)
from sommelier.errors import ConfigError

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_smoke_example_loads() -> None:
    config = load_config(EXAMPLES_DIR / "config.smoke.yaml")
    assert config.schema_version == "sommelier.config.v2"
    assert config.data.n_train == 100


def test_full_example_loads() -> None:
    config = load_config(EXAMPLES_DIR / "config.full.yaml")
    assert config.data.n_train == 15000


def test_language_defaults_resolve_to_configured_languages() -> None:
    config = load_config(EXAMPLES_DIR / "config.smoke.yaml")
    assert [source.language for source in config.datasets] == ["en"]
    assert config.train.languages == ["en"]
    assert config.eval.slices == ["en"]
    assert config.root_dataset.dataset_id == "Salesforce/xlam-function-calling-60k"
    assert config.dataset_for("en") is config.root_dataset


def _write_v1_document(tmp_path: Path) -> Path:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    raw["schema_version"] = "sommelier.config.v1"
    source = dict(raw.pop("datasets")[0])
    source.pop("language")
    raw["dataset"] = source
    config_path = tmp_path / "config-v1.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def test_v1_document_upgrades_with_deprecation_warning(tmp_path: Path) -> None:
    config_path = _write_v1_document(tmp_path)
    with pytest.warns(DeprecationWarning, match="sommelier.config.v1 is deprecated"):
        config = load_config(config_path)
    assert config.schema_version == "sommelier.config.v2"
    assert [source.language for source in config.datasets] == ["en"]
    assert config.datasets[0].source_id_column is None
    assert config.train.languages == ["en"]
    assert config.eval.slices == ["en"]


def test_v1_resolved_form_persists_as_v2(tmp_path: Path) -> None:
    config_path = _write_v1_document(tmp_path)
    with pytest.warns(DeprecationWarning):
        config = load_config(config_path)
    resolved_path, _ = write_resolved_config(config, tmp_path / "run")
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert resolved["schema_version"] == "sommelier.config.v2"
    assert resolved["datasets"][0]["language"] == "en"
    assert "dataset" not in resolved


def _load_modified(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    mutate(raw)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    load_config(config_path)


def _french_source(raw: dict[str, Any]) -> dict[str, Any]:
    source = dict(raw["datasets"][0])
    source["language"] = "fr"
    source["source_id_column"] = "source_example_id"
    return source


def test_duplicate_dataset_language_rejected(tmp_path: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["datasets"].append(dict(raw["datasets"][0]))

    with pytest.raises(ConfigError, match="duplicate dataset language"):
        _load_modified(tmp_path, mutate)


def test_empty_datasets_rejected(tmp_path: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["datasets"] = []

    with pytest.raises(ConfigError, match="at least one source"):
        _load_modified(tmp_path, mutate)


def test_second_root_source_rejected(tmp_path: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        second = _french_source(raw)
        second.pop("source_id_column")
        raw["datasets"].append(second)

    with pytest.raises(ConfigError, match="exactly one dataset source"):
        _load_modified(tmp_path, mutate)


def test_invalid_language_code_rejected(tmp_path: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["datasets"][0]["language"] = "FR"

    with pytest.raises(ConfigError, match="ISO 639-1"):
        _load_modified(tmp_path, mutate)


def test_unknown_train_language_rejected(tmp_path: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["train"]["languages"] = ["en", "fr"]

    with pytest.raises(ConfigError, match="train.languages references a language"):
        _load_modified(tmp_path, mutate)


def test_unknown_eval_slice_rejected(tmp_path: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["eval"]["slices"] = ["fr"]

    with pytest.raises(ConfigError, match="eval.slices references a language"):
        _load_modified(tmp_path, mutate)


def test_duplicate_eval_slice_rejected(tmp_path: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["eval"]["slices"] = ["en", "en"]

    with pytest.raises(ConfigError, match="duplicate entry in eval.slices"):
        _load_modified(tmp_path, mutate)


def test_bilingual_config_loads(tmp_path: Path) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    raw["datasets"].append(_french_source(raw))
    raw["eval"]["slices"] = ["en", "fr"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    assert config.train.languages == ["en", "fr"]
    assert config.eval.slices == ["en", "fr"]
    assert config.root_dataset.language == "en"
    assert config.dataset_for("fr").source_id_column == "source_example_id"


def test_unknown_field_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (EXAMPLES_DIR / "config.smoke.yaml").read_text() + "\nunexpected_field: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_duplicate_yaml_key_is_rejected_before_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (EXAMPLES_DIR / "config.smoke.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  seed: 42",
            "  seed: 1\n  seed: 2",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate key.*seed"):
        load_config(config_path)


def test_absolute_artifact_root_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    raw["project"]["artifact_root"] = "/tmp/artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path)


@pytest.mark.parametrize("artifact_root", ["../outside", "nested/../../outside", "."])
def test_artifact_root_cannot_escape_or_alias_config_directory(
    tmp_path: Path,
    artifact_root: str,
) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    raw["project"]["artifact_root"] = artifact_root
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="artifact_root"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("train", "learning_rate"),
        ("train", "warmup_ratio"),
        ("train", "lora_dropout"),
        ("eval", "temperature"),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_config_values_are_rejected(
    tmp_path: Path,
    section: str,
    field: str,
    value: float,
) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    raw[section][field] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match=field):
        load_config(config_path)


@pytest.mark.parametrize(
    "field",
    ["data_timeout_seconds", "train_timeout_seconds", "eval_timeout_seconds"],
)
@pytest.mark.parametrize("value", [0, -1])
def test_remote_planning_estimates_must_be_positive(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    raw["remote"][field] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match=field):
        load_config(config_path)


def test_eval_slices_cannot_be_empty(tmp_path: Path) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    raw["eval"]["slices"] = []
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="eval.slices"):
        load_config(config_path)


def test_digest_stability(tmp_path: Path) -> None:
    config = load_config(EXAMPLES_DIR / "config.smoke.yaml")
    _, digest_one = write_resolved_config(config, tmp_path / "run-a")
    _, digest_two = write_resolved_config(config, tmp_path / "run-b")
    assert digest_one == digest_two


def test_resolved_config_contains_relative_artifact_root(tmp_path: Path) -> None:
    config = load_config(EXAMPLES_DIR / "config.smoke.yaml")
    resolved_path, _ = write_resolved_config(config, tmp_path)
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert resolved["project"]["artifact_root"] == "artifacts"


def test_remote_code_requires_reason(tmp_path: Path) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    raw["model"]["allow_remote_code"] = True
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path)
