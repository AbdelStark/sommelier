from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from sommelier.errors import SchemaValidationError
from sommelier.redaction import DuplicateJsonKeyError, loads_unique_json

SUPPORTED_SCHEMAS = frozenset(
    {
        "sommelier.config.v1",
        "sommelier.config.v2",
        "sommelier.manifest.v1",
        "sommelier.raw_tool_call_row.v1",
        "sommelier.prepared_example.v1",
        "sommelier.prepared_example.v2",
        "sommelier.formatted_example.v1",
        "sommelier.formatted_example.v2",
        "sommelier.generation.v1",
        "sommelier.generation.v2",
        "sommelier.inference_telemetry.v1",
        "sommelier.inference_telemetry.v2",
        "sommelier.evaluation_report.v1",
        "sommelier.evaluation_report.v2",
        "sommelier.evaluation_report.v3",
        "sommelier.drop_summary.v1",
        "sommelier.drop_summary.v2",
        "sommelier.log_event.v1",
        "sommelier.comparison_report.v1",
        "sommelier.comparison_report.v2",
        "sommelier.comparison_report.v3",
        # v1 remains readable for historical inspection. The current
        # finalizer and release/publication validators require v2.
        "sommelier.experiment_report.v1",
        "sommelier.experiment_report.v2",
        "sommelier.training_metric.v1",
        "sommelier.translation_summary.v1",
        "sommelier.translation_summary.v2",
        "sommelier.translation_publication_manifest.v1",
        "sommelier.translation_run_identity.v1",
        "sommelier.translation_semantic_review.v1",
        "sommelier.translation_semantic_review_template.v1",
        "sommelier.translation_semantic_review_attestation.v1",
        "sommelier.tokenizer_tax_record.v1",
        "sommelier.tokenizer_tax_report.v1",
        "sommelier.sovereign_tco_evidence.v1",
        # Historical reports remain readable for inspection. Release/publication
        # validation imports the current v2 contract and rejects v1 evidence.
        "sommelier.release_preflight.v1",
        "sommelier.release_preflight.v2",
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
            try:
                payload = loads_unique_json(stripped)
            except (json.JSONDecodeError, DuplicateJsonKeyError) as error:
                raise SchemaValidationError(
                    f"{path}:{line_number} contains invalid JSON",
                    hint=(
                        "Use one valid JSON object per line; duplicate object keys are forbidden."
                    ),
                ) from error
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


def _private_staging_directory(parent: Path, *, target_name: str) -> Path:
    for _ in range(128):
        candidate = parent / f".{target_name}.writer.{secrets.token_hex(16)}"
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate
    raise SchemaValidationError(
        f"could not reserve a private staging directory for {parent / target_name}",
        hint="Remove stale artifact staging entries and retry the write.",
    )


def _open_exclusive_regular_temp(parent: Path, *, target_name: str) -> tuple[Path, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for flag_name in ("O_NOFOLLOW", "O_CLOEXEC"):
        flags |= getattr(os, flag_name, 0)
    for _ in range(128):
        candidate = parent / f".{target_name}.tmp.{secrets.token_hex(16)}"
        try:
            descriptor = os.open(candidate, flags, 0o600)
        except FileExistsError:
            continue
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            candidate.unlink(missing_ok=True)
            raise SchemaValidationError(f"artifact staging path is not a regular file: {candidate}")
        return candidate, descriptor
    raise SchemaValidationError(
        f"could not reserve an exclusive artifact staging file for {parent / target_name}",
        hint="Remove stale artifact staging entries and retry the write.",
    )


def _open_writer_output(path: Path) -> tuple[int, os.stat_result]:
    try:
        path_metadata = path.lstat()
    except FileNotFoundError as error:
        raise SchemaValidationError(
            f"writer did not create {path}",
            hint="Ensure the artifact writer persists output before returning.",
        ) from error
    if not stat.S_ISREG(path_metadata.st_mode):
        raise SchemaValidationError(
            f"artifact writer output is not a regular file: {path}",
            hint=(
                "Write bytes to the provided path; symbolic links and special files are forbidden."
            ),
        )
    flags = os.O_RDONLY
    for flag_name in ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC"):
        flags |= getattr(os, flag_name, 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SchemaValidationError(
            f"artifact writer output could not be opened safely as a regular file: {path}"
        ) from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_dev != path_metadata.st_dev
        or opened.st_ino != path_metadata.st_ino
    ):
        os.close(descriptor)
        raise SchemaValidationError(
            f"artifact writer output changed before it could be copied: {path}"
        )
    return descriptor, opened


def _same_file_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
    )


def _hash_open_descriptor(descriptor: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    byte_count = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
        byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _artifact_relative_path(path: Path, *, artifact_root: Path) -> str:
    try:
        root_metadata = artifact_root.lstat()
    except FileNotFoundError:
        root_metadata = None
    if root_metadata is not None and (
        stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode)
    ):
        raise SchemaValidationError(
            f"artifact root is not a safe directory: {artifact_root}",
            hint="Use a real directory rather than a symbolic link or special file.",
        )
    resolved_root = artifact_root.resolve()
    # The final component is replaced rather than followed. Resolve only its
    # parent so an existing in-root symlink at the target is safely replaced.
    resolved_path = path.parent.resolve() / path.name
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise SchemaValidationError(
            f"artifact path escapes artifact root: {path}",
            hint="Keep all artifact paths inside the configured artifact_root.",
        ) from error
    if not relative.parts or ".." in relative.parts:
        raise SchemaValidationError(
            f"artifact path escapes artifact root: {path}",
            hint="Keep all artifact paths inside the configured artifact_root.",
        )
    return relative.as_posix()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_artifact_atomic(
    path: Path,
    writer: Callable[[Path], None],
    *,
    artifact_root: Path | None = None,
    kind: str = "file",
    schema_version: str = "",
    replace_existing: bool = True,
) -> ArtifactRef:
    artifact_path = (
        path.name
        if artifact_root is None
        else _artifact_relative_path(path, artifact_root=artifact_root)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = _private_staging_directory(path.parent, target_name=path.name)
    writer_path = staging_dir / path.name
    temp_path: Path | None = None
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        writer(writer_path)
        source_descriptor, source_before = _open_writer_output(writer_path)
        temp_path, destination_descriptor = _open_exclusive_regular_temp(
            staging_dir,
            target_name=path.name,
        )
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(destination_descriptor, chunk[offset:])
                if written <= 0:
                    raise OSError("short write while staging artifact")
                offset += written
        source_after = os.fstat(source_descriptor)
        try:
            source_path_after = writer_path.lstat()
        except FileNotFoundError as error:
            raise SchemaValidationError(
                f"artifact writer output disappeared while it was being copied: {writer_path}"
            ) from error
        if (
            byte_count != source_before.st_size
            or not _same_file_snapshot(source_before, source_after)
            or not _same_file_snapshot(source_after, source_path_after)
        ):
            raise SchemaValidationError(
                f"artifact writer output changed while it was being copied: {writer_path}",
                hint="Stop concurrent writers and retry from an immutable input snapshot.",
            )
        verified_sha256, verified_bytes = _hash_open_descriptor(source_descriptor)
        if verified_sha256 != digest.hexdigest() or verified_bytes != byte_count:
            raise SchemaValidationError(
                f"artifact writer output changed while it was being copied: {writer_path}",
                hint="Stop concurrent writers and retry from an immutable input snapshot.",
            )
        os.fsync(destination_descriptor)
        os.close(destination_descriptor)
        destination_descriptor = None
        if replace_existing:
            os.replace(temp_path, path)
            temp_path = None
        else:
            try:
                # The staged file and destination share a filesystem. A hard
                # link publishes the fully fsynced inode without following or
                # replacing an existing final component; unlike a preflight
                # exists() check, this remains exclusive at the mutation.
                os.link(temp_path, path)
            except FileExistsError as error:
                raise SchemaValidationError(
                    f"artifact already exists and is immutable: {path}",
                    hint="Choose a fresh output path or verify the existing evidence.",
                ) from error
            except OSError as error:
                raise SchemaValidationError(
                    f"could not publish a new immutable artifact: {path}",
                    hint="Use a local filesystem that supports same-filesystem hard links.",
                ) from error
            temp_path.unlink()
            temp_path = None
        _fsync_directory(path.parent)
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        shutil.rmtree(staging_dir, ignore_errors=True)

    return ArtifactRef(
        path=artifact_path,
        kind=kind,
        schema_version=schema_version,
        sha256=digest.hexdigest(),
        bytes=byte_count,
    )
