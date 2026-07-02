from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from sommelier.artifacts import ArtifactRef, make_artifact_ref, write_artifact_atomic
from sommelier.config import SommelierConfig
from sommelier.errors import InvariantViolation
from sommelier.security import validate_no_secrets

StageName = Literal["data", "format", "train", "eval", "report", "serve"]


class StageManifest(TypedDict):
    schema_version: Literal["sommelier.manifest.v1"]
    stage: StageName
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


class FailedStageManifest(StageManifest):
    error_code: str
    error_message: str


class RunManifest(TypedDict):
    schema_version: Literal["sommelier.manifest.v1"]
    run_id: str
    stages: dict[str, str]
    config: ArtifactRef
    status: Literal["running", "succeeded", "failed"]
    tracking: NotRequired[dict[str, str]]


def create_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def get_dependency_lock_sha256(project_root: Path | None = None) -> str | None:
    root = project_root or Path.cwd()
    lock_path = root / "uv.lock"
    if not lock_path.exists():
        return None
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def run_dir_for(config: SommelierConfig, run_id: str, *, config_dir: Path) -> Path:
    return (config_dir / config.project.artifact_root / "runs" / run_id).resolve()


def artifact_root_for(config: SommelierConfig, *, config_dir: Path) -> Path:
    return (config_dir / config.project.artifact_root).resolve()


def _manifest_filename(stage: StageName) -> str:
    return f"{stage}_manifest.json"


def _root_manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def _redact_failed_message(message: str) -> str:
    redacted = message
    for pattern in ("hf_", "sk-", "ghp_", "token", "secret", "password"):
        if pattern.lower() in redacted.lower():
            return "stage failed; details redacted"
    return redacted[:500]


def build_stage_manifest(
    *,
    stage: StageName,
    run_id: str,
    config_sha256: str,
    command: list[str],
    seed: int,
    inputs: list[ArtifactRef],
    outputs: list[ArtifactRef],
    status: Literal["succeeded", "failed"],
    dependency_lock_sha256: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> StageManifest | FailedStageManifest:
    manifest: StageManifest | FailedStageManifest = {
        "schema_version": "sommelier.manifest.v1",
        "stage": stage,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "config_sha256": config_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
        "command": command,
        "seed": seed,
        "inputs": inputs,
        "outputs": outputs,
        "status": status,
    }
    if status == "failed":
        if error_code is None or error_message is None:
            raise InvariantViolation(
                "failed stage manifests require error_code and error_message",
                hint="Provide sanitized failure metadata when writing failed manifests.",
            )
        failed: FailedStageManifest = {
            **manifest,
            "error_code": error_code,
            "error_message": _redact_failed_message(error_message),
        }
        validate_no_secrets(failed, context="failed manifest")
        return failed
    validate_no_secrets(manifest, context="stage manifest")
    return manifest


def write_stage_manifest(
    manifest: StageManifest | FailedStageManifest,
    *,
    run_dir: Path,
    artifact_root: Path,
) -> ArtifactRef:
    stage = manifest["stage"]
    manifest_path = run_dir / _manifest_filename(stage)

    def writer(temp_path: Path) -> None:
        temp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return write_artifact_atomic(
        manifest_path,
        writer,
        artifact_root=artifact_root,
        kind="manifest",
        schema_version="sommelier.manifest.v1",
    )


def initialize_run_manifest(
    *,
    run_dir: Path,
    run_id: str,
    config_ref: ArtifactRef,
) -> RunManifest:
    manifest: RunManifest = {
        "schema_version": "sommelier.manifest.v1",
        "run_id": run_id,
        "stages": {},
        "config": config_ref,
        "status": "running",
    }
    validate_no_secrets(manifest, context="run manifest")
    _write_run_manifest(run_dir, manifest)
    return manifest


def update_run_manifest(
    *,
    run_dir: Path,
    artifact_root: Path,
    stage: StageName,
    stage_manifest_ref: ArtifactRef,
    status: Literal["running", "succeeded", "failed"],
) -> RunManifest:
    root_path = _root_manifest_path(run_dir)
    if root_path.exists():
        current = json.loads(root_path.read_text(encoding="utf-8"))
    else:
        raise InvariantViolation(
            f"run manifest not found at {root_path}",
            hint="Initialize the run directory before updating stage manifests.",
        )

    stages = dict(current.get("stages", {}))
    stages[stage] = stage_manifest_ref["path"]
    updated: RunManifest = {
        "schema_version": "sommelier.manifest.v1",
        "run_id": current["run_id"],
        "stages": stages,
        "config": current["config"],
        "status": status,
    }
    if "tracking" in current:
        updated["tracking"] = current["tracking"]
    validate_no_secrets(updated, context="run manifest")
    _write_run_manifest(run_dir, updated)
    return updated


def record_tracking_in_run_manifest(
    *,
    run_dir: Path,
    tracking: dict[str, str],
) -> RunManifest:
    """Records the external tracker project and run URL in the run manifest."""
    root_path = _root_manifest_path(run_dir)
    if not root_path.exists():
        raise InvariantViolation(
            f"run manifest not found at {root_path}",
            hint="Initialize the run directory before recording tracking info.",
        )
    current = json.loads(root_path.read_text(encoding="utf-8"))
    updated: RunManifest = {
        "schema_version": "sommelier.manifest.v1",
        "run_id": current["run_id"],
        "stages": dict(current.get("stages", {})),
        "config": current["config"],
        "status": current["status"],
        "tracking": tracking,
    }
    validate_no_secrets(updated, context="run manifest")
    _write_run_manifest(run_dir, updated)
    return updated


def _write_run_manifest(run_dir: Path, manifest: RunManifest) -> None:
    root_path = _root_manifest_path(run_dir)

    def writer(temp_path: Path) -> None:
        temp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_artifact_atomic(root_path, writer)


def config_artifact_ref(resolved_config_path: Path, *, artifact_root: Path) -> ArtifactRef:
    return make_artifact_ref(
        resolved_config_path,
        artifact_root=artifact_root,
        kind="config",
        schema_version="sommelier.config.v1",
    )
