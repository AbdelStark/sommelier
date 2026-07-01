from pathlib import Path

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
    assert config.schema_version == "sommelier.config.v1"
    assert config.data.n_train == 100


def test_full_example_loads() -> None:
    config = load_config(EXAMPLES_DIR / "config.full.yaml")
    assert config.data.n_train == 15000


def test_unknown_field_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (EXAMPLES_DIR / "config.smoke.yaml").read_text()
        + "\nunexpected_field: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_absolute_artifact_root_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    raw["project"]["artifact_root"] = "/tmp/artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
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
