from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from sommelier.errors import SchemaValidationError

SUPPORTED_SCHEMAS = frozenset(
    {
        "sommelier.config.v1",
        "sommelier.manifest.v1",
        "sommelier.raw_tool_call_row.v1",
        "sommelier.prepared_example.v1",
        "sommelier.formatted_example.v1",
        "sommelier.generation.v1",
        "sommelier.evaluation_report.v1",
        "sommelier.drop_summary.v1",
        "sommelier.log_event.v1",
    }
)


class ArtifactRef(TypedDict):
    path: str
    kind: str
    schema_version: str
    sha256: str
    bytes: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_supported_schema(schema_version: str | None, *, context: str) -> None:
    if schema_version is None:
        raise SchemaValidationError(
            f"{context} is missing schema_version",
            hint="Add a supported schema_version field to the record.",
        )
    if schema_version not in SUPPORTED_SCHEMAS:
        raise SchemaValidationError(
            f"{context} uses unsupported schema_version {schema_version!r}",
            hint=f"Supported schema versions: {', '.join(sorted(SUPPORTED_SCHEMAS))}",
        )


def read_json_with_schema(path: Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SchemaValidationError(
            f"{path} must contain a JSON object",
            hint="Use a schema-versioned JSON object for this artifact.",
        )
    schema_version = payload.get("schema_version")
    assert_supported_schema(schema_version, context=str(path))
    if expected_schema is not None and schema_version != expected_schema:
        raise SchemaValidationError(
            f"{path} expected schema_version {expected_schema!r}, got {schema_version!r}",
            hint="Regenerate the artifact with the current pipeline version.",
        )
    return payload


def read_jsonl_with_schema(
    path: Path,
    *,
    expected_schema: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise SchemaValidationError(
                    f"{path}:{line_number} must contain a JSON object",
                    hint="Use one schema-versioned JSON object per JSONL line.",
                )
            schema_version = payload.get("schema_version")
            assert_supported_schema(schema_version, context=f"{path}:{line_number}")
            if expected_schema is not None and schema_version != expected_schema:
                raise SchemaValidationError(
                    f"{path}:{line_number} expected schema_version "
                    f"{expected_schema!r}, got {schema_version!r}",
                    hint="Regenerate the artifact with the current pipeline version.",
                )
            records.append(payload)
    return records


def make_artifact_ref(
    path: Path,
    *,
    artifact_root: Path,
    kind: str,
    schema_version: str,
) -> ArtifactRef:
    resolved_path = path.resolve()
    resolved_root = artifact_root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise SchemaValidationError(
            f"artifact path escapes artifact root: {path}",
            hint="Keep all artifact paths inside the configured artifact_root.",
        ) from error
    if ".." in relative.parts:
        raise SchemaValidationError(
            f"artifact path escapes artifact root: {path}",
            hint="Keep all artifact paths inside the configured artifact_root.",
        )
    data = resolved_path.read_bytes()
    return ArtifactRef(
        path=relative.as_posix(),
        kind=kind,
        schema_version=schema_version,
        sha256=sha256_bytes(data),
        bytes=len(data),
    )


def write_artifact_atomic(
    path: Path,
    writer: Callable[[Path], None],
    *,
    artifact_root: Path | None = None,
    kind: str = "file",
    schema_version: str = "",
) -> ArtifactRef:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        writer(temp_path)
        if not temp_path.exists():
            raise SchemaValidationError(
                f"writer did not create {temp_path}",
                hint="Ensure the artifact writer persists output before returning.",
            )
        shutil.move(str(temp_path), str(path))
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    if artifact_root is None:
        data = path.read_bytes()
        return ArtifactRef(
            path=path.name,
            kind=kind,
            schema_version=schema_version,
            sha256=sha256_bytes(data),
            bytes=len(data),
        )

    return make_artifact_ref(
        path,
        artifact_root=artifact_root,
        kind=kind,
        schema_version=schema_version,
    )
