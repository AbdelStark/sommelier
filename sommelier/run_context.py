from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sommelier.artifacts import ArtifactRef
from sommelier.config import SommelierConfig, write_resolved_config
from sommelier.errors import ArtifactNotFoundError, UserInputError
from sommelier.manifests import (
    StageName,
    artifact_root_for,
    build_stage_manifest,
    config_artifact_ref,
    create_run_id,
    get_dependency_lock_sha256,
    initialize_run_manifest,
    run_dir_for,
    update_run_manifest,
    write_stage_manifest,
)

RUN_DATA_SUFFIX = Path("data")
RUN_ID_PATTERN = re.compile(r"/runs/([^/]+)/")


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    artifact_root: Path
    config_dir: Path
    config_sha256: str
    config_ref: ArtifactRef
    dependency_lock_sha256: str | None


def infer_run_id_from_path(path: Path) -> str | None:
    match = RUN_ID_PATTERN.search(path.as_posix())
    if match is None:
        return None
    return match.group(1)


def ensure_run_context(
    config: SommelierConfig,
    *,
    config_path: Path,
    run_id: str | None = None,
    project_root: Path | None = None,
    reject_existing_run: bool = False,
) -> RunContext:
    resolved_run_id = run_id or create_run_id()
    config_dir = config_path.parent.resolve()
    run_dir = run_dir_for(config, resolved_run_id, config_dir=config_dir)
    artifact_root = artifact_root_for(config, config_dir=config_dir)
    if reject_existing_run:
        try:
            # mkdir(2) is the reservation boundary: exactly one concurrent
            # full attempt can claim this final path.  Do not use an exists()
            # check here; that would leave a race before evidence is written.
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise UserInputError(
                f"full pipeline run directory already exists: {run_dir}",
                hint=(
                    "Choose a fresh --run-id. Full pipeline attempts are "
                    "non-resumable and never overwrite prior evidence."
                ),
            ) from error
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    resolved_path, config_sha256 = write_resolved_config(config, run_dir)
    config_ref = config_artifact_ref(resolved_path, artifact_root=artifact_root)
    root_manifest_path = run_dir / "manifest.json"
    if not root_manifest_path.exists():
        initialize_run_manifest(
            run_dir=run_dir,
            run_id=resolved_run_id,
            config_ref=config_ref,
        )

    return RunContext(
        run_id=resolved_run_id,
        run_dir=run_dir,
        artifact_root=artifact_root,
        config_dir=config_dir,
        config_sha256=config_sha256,
        config_ref=config_ref,
        dependency_lock_sha256=get_dependency_lock_sha256(project_root),
    )


def record_stage_success(
    context: RunContext,
    *,
    stage: StageName,
    command: list[str],
    seed: int,
    inputs: list[ArtifactRef],
    outputs: list[ArtifactRef],
    details: dict[str, Any] | None = None,
) -> ArtifactRef:
    manifest = build_stage_manifest(
        stage=stage,
        run_id=context.run_id,
        config_sha256=context.config_sha256,
        dependency_lock_sha256=context.dependency_lock_sha256,
        command=command,
        seed=seed,
        inputs=inputs,
        outputs=outputs,
        status="succeeded",
        details=details,
    )
    stage_ref = write_stage_manifest(
        manifest,
        run_dir=context.run_dir,
        artifact_root=context.artifact_root,
    )
    update_run_manifest(
        run_dir=context.run_dir,
        artifact_root=context.artifact_root,
        stage=stage,
        stage_manifest_ref=stage_ref,
        status="running",
    )
    return stage_ref


def write_jsonl_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def read_jsonl_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise ArtifactNotFoundError(
            f"artifact not found: {path}",
            hint="Run the preceding stage or pass the correct input directory.",
        )
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise UserInputError(
                    f"{path}:{line_number} must contain a JSON object",
                    hint="Use schema-versioned JSON objects in JSONL artifacts.",
                )
            records.append(payload)
    return records
