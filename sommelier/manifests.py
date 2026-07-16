from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

from sommelier.artifacts import ArtifactRef, make_artifact_ref, write_artifact_atomic
from sommelier.config import SommelierConfig, resolve_config_artifact_root
from sommelier.errors import ConfigError, InvariantViolation
from sommelier.security import validate_no_secrets

StageName = Literal[
    "data",
    "format",
    "tokenization",
    "train",
    "eval",
    "eval-base",
    "eval-adapter",
    "report",
    "serve",
]


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
    # Optional stage-specific evidence (e.g. training example counts per
    # language); absent unless a stage has something structured to record.
    details: NotRequired[dict[str, Any]]


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
    mounted_revision = os.environ.get("SOMMELIER_GIT_COMMIT")
    if mounted_revision is not None:
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", mounted_revision):
            raise InvariantViolation(
                "SOMMELIER_GIT_COMMIT must be an immutable hexadecimal Git object ID",
                hint="Pass the local git revision into the remote entrypoint unchanged.",
            )
        return mounted_revision
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


def get_git_worktree_clean() -> bool | None:
    """Whether tracked and untracked local source matches the recorded commit."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return not bool(result.stdout.strip())


def get_dependency_lock_sha256(project_root: Path | None = None) -> str | None:
    root = project_root or Path.cwd()
    lock_path = root / "uv.lock"
    if not lock_path.exists():
        return None
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def run_dir_for(config: SommelierConfig, run_id: str, *, config_dir: Path) -> Path:
    artifact_root = artifact_root_for(config, config_dir=config_dir)
    run_dir = (artifact_root / "runs" / run_id).resolve()
    try:
        relative = run_dir.relative_to(artifact_root)
    except ValueError as error:
        raise ConfigError(
            f"run directory escapes project.artifact_root: {run_dir}",
            hint="Remove escaping symlinks under the configured artifact root.",
        ) from error
    if not relative.parts:
        raise ConfigError("run directory resolves to project.artifact_root")
    return run_dir


def artifact_root_for(config: SommelierConfig, *, config_dir: Path) -> Path:
    return resolve_config_artifact_root(config, config_dir=config_dir)


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
    details: dict[str, Any] | None = None,
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
    if details is not None:
        manifest["details"] = details
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


def set_run_manifest_status(
    *,
    run_dir: Path,
    status: Literal["succeeded", "failed"],
) -> RunManifest:
    """Closes a run after orchestration succeeds or propagates a failure."""
    root_path = _root_manifest_path(run_dir)
    if not root_path.exists():
        raise InvariantViolation(
            f"run manifest not found at {root_path}",
            hint="Initialize the run directory before finalizing it.",
        )
    current = json.loads(root_path.read_text(encoding="utf-8"))
    updated: RunManifest = {
        "schema_version": "sommelier.manifest.v1",
        "run_id": current["run_id"],
        "stages": dict(current.get("stages", {})),
        "config": current["config"],
        "status": status,
    }
    if "tracking" in current:
        updated["tracking"] = current["tracking"]
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
        schema_version="sommelier.config.v2",
    )
