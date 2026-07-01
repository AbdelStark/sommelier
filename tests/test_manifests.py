import json
from pathlib import Path
from typing import cast

import pytest

from sommelier.config import load_config, write_resolved_config
from sommelier.errors import InvariantViolation
from sommelier.manifests import (
    FailedStageManifest,
    artifact_root_for,
    build_stage_manifest,
    config_artifact_ref,
    initialize_run_manifest,
    run_dir_for,
    update_run_manifest,
    write_stage_manifest,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_stage_manifest_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.smoke.yaml"
    config_path.write_text(
        (EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = load_config(config_path)
    config_dir = tmp_path
    run_id = "test-run"
    run_dir = run_dir_for(config, run_id, config_dir=config_dir)
    artifact_root = artifact_root_for(config, config_dir=config_dir)
    run_dir.mkdir(parents=True)

    resolved_path, config_sha256 = write_resolved_config(config, run_dir)
    config_ref = config_artifact_ref(resolved_path, artifact_root=artifact_root)
    initialize_run_manifest(run_dir=run_dir, run_id=run_id, config_ref=config_ref)

    manifest = build_stage_manifest(
        stage="data",
        run_id=run_id,
        config_sha256=config_sha256,
        command=["sommelier", "data", "prepare"],
        seed=config.project.seed,
        inputs=[config_ref],
        outputs=[],
        status="succeeded",
    )
    stage_ref = write_stage_manifest(
        manifest,
        run_dir=run_dir,
        artifact_root=artifact_root,
    )
    root = update_run_manifest(
        run_dir=run_dir,
        artifact_root=artifact_root,
        stage="data",
        stage_manifest_ref=stage_ref,
        status="running",
    )
    assert root["stages"]["data"] == stage_ref["path"]
    assert root["config"]["sha256"] == config_ref["sha256"]


def test_failed_manifest_redacts_secrets() -> None:
    manifest = build_stage_manifest(
        stage="train",
        run_id="run-1",
        config_sha256="abc",
        command=["sommelier", "train", "run"],
        seed=42,
        inputs=[],
        outputs=[],
        status="failed",
        error_code="SOM004",
        error_message="authentication failed: hf_abcdefghijklmnopqrstuvwxyz123456",
    )
    failed = cast(FailedStageManifest, manifest)
    assert failed["error_message"] == "stage failed; details redacted"


def test_failed_manifest_requires_error_fields() -> None:
    with pytest.raises(InvariantViolation):
        build_stage_manifest(
            stage="eval",
            run_id="run-1",
            config_sha256="abc",
            command=["sommelier", "eval", "run"],
            seed=42,
            inputs=[],
            outputs=[],
            status="failed",
        )


def test_manifest_round_trip_from_disk(tmp_path: Path) -> None:
    manifest = build_stage_manifest(
        stage="format",
        run_id="run-2",
        config_sha256="digest",
        command=["sommelier", "format", "build"],
        seed=7,
        inputs=[],
        outputs=[],
        status="succeeded",
    )
    path = tmp_path / "format_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "sommelier.manifest.v1"
    assert loaded["status"] == "succeeded"
