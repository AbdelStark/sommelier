import json
import subprocess
import sys
from pathlib import Path

import yaml

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _write_tiny_config(config_path: Path) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 2
    raw["data"]["n_validation"] = 1
    raw["data"]["n_test"] = 1
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")


def test_data_prepare_writes_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "config.smoke.yaml"
    _write_tiny_config(config_path)
    out_dir = tmp_path / "artifacts" / "runs" / "fixture-run" / "data"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sommelier.cli",
            "data",
            "prepare",
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
            "--run-id",
            "fixture-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert "data prepare ok" in result.stdout
    assert (out_dir / "train.jsonl").exists()
    manifest_path = tmp_path / "artifacts" / "runs" / "fixture-run" / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "sommelier.manifest.v1"
    assert manifest["status"] == "succeeded"


def test_format_build_writes_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "config.smoke.yaml"
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 1
    raw["data"]["n_validation"] = 1
    raw["data"]["n_test"] = 1
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    data_dir = tmp_path / "artifacts" / "runs" / "fixture-run" / "data"
    formatted_dir = tmp_path / "artifacts" / "runs" / "fixture-run" / "formatted"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sommelier.cli",
            "data",
            "prepare",
            "--config",
            str(config_path),
            "--out",
            str(data_dir),
            "--run-id",
            "fixture-run",
        ],
        check=True,
        cwd=tmp_path,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sommelier.cli",
            "format",
            "build",
            "--config",
            str(config_path),
            "--data",
            str(data_dir),
            "--out",
            str(formatted_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert "format build ok" in result.stdout
    assert (formatted_dir / "train.jsonl").exists()
    manifest = json.loads(
        (tmp_path / "artifacts" / "runs" / "fixture-run" / "format_manifest.json").read_text()
    )
    assert manifest["stage"] == "format"
    assert manifest["status"] == "succeeded"


def test_validate_fixtures_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sommelier.cli", "data", "validate-fixtures"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "fixtures ok" in result.stdout
