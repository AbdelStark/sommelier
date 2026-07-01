from pathlib import Path
from typing import cast

import pytest
import yaml

from sommelier.config import load_config
from sommelier.errors import SecurityPolicyError
from sommelier.manifests import FailedStageManifest, build_stage_manifest
from sommelier.security import scan_mapping_for_secrets, validate_no_secrets

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_rejects_secret_like_config_key(tmp_path: Path) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    raw["hf_token"] = "not-a-real-token"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SecurityPolicyError):
        load_config(config_path)


def test_rejects_secret_like_config_value(tmp_path: Path) -> None:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text())
    raw["project"]["name"] = "hf_abcdefghijklmnopqrstuvwxyz1234567890"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SecurityPolicyError):
        load_config(config_path)


def test_allows_safe_key_names() -> None:
    violations = scan_mapping_for_secrets(
        {
            "data": {"dedupe_key": "normalized_query"},
            "dataset": {"query_column": "query"},
        }
    )
    assert violations == []


def test_manifest_secret_scan() -> None:
    with pytest.raises(SecurityPolicyError):
        validate_no_secrets(
            {
                "schema_version": "sommelier.manifest.v1",
                "error_message": "sk-abcdefghijklmnopqrstuvwxyz1234567890",
            },
            context="manifest",
        )


def test_failed_manifest_redacts_before_write() -> None:
    manifest = build_stage_manifest(
        stage="eval",
        run_id="run-1",
        config_sha256="abc",
        command=["sommelier", "eval", "run"],
        seed=42,
        inputs=[],
        outputs=[],
        status="failed",
        error_code="SOM006",
        error_message="value hf_abcdefghijklmnopqrstuvwxyz1234567890 leaked",
    )
    failed = cast(FailedStageManifest, manifest)
    assert failed["error_message"] == "stage failed; details redacted"
