from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final, Literal, TypedDict, cast

from sommelier.artifacts import write_artifact_atomic
from sommelier.config import SommelierConfig
from sommelier.errors import ExternalDependencyError, InvariantViolation, SecurityPolicyError
from sommelier.redaction import (
    SCANNABLE_SUFFIXES,
    RedactionFinding,
    scan_artifact_text,
)
from sommelier.security import redact_text

PREFLIGHT_SCHEMA: Final = "sommelier.release_preflight.v2"
PREFLIGHT_FILENAME: Final = "release_preflight.json"

PREFLIGHT_CONFIG_NORMALIZATION: Final = "pydantic-json-canonical-v1"
PREFLIGHT_ARTIFACT_TREE_ALGORITHM: Final = "sha256-canonical-file-manifest-v1"
PREFLIGHT_TREE_EXCLUDED_PATHS: Final = (PREFLIGHT_FILENAME,)

ACK_ENV_NAME: Final = "SOMMELIER_ACK_BASE_MODEL_LICENSE"

# The Llama 3.1 Community License requires this notice on derived
# artifacts; licenses/THIRD_PARTY.md records it for the configured base
# model.
REQUIRED_DERIVED_NOTICE: Final = "Built with Llama"

REQUIRED_RELEASE_GATES: Final = (
    "project_license",
    "third_party_notices",
    "base_model_obligations",
    "dataset_license",
    "derived_artifact_notice",
    "base_model_license_ack",
    "dependency_lock",
    "artifact_secret_scan",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()
_GIT_TIMEOUT_SECONDS: Final = 10
_REQUIRED_TRACKED_PROJECT_PATHS: Final = (
    "LICENSE",
    "licenses/THIRD_PARTY.md",
    "uv.lock",
)

GateStatus = Literal["pass", "fail", "skip"]


class ReleaseGate(TypedDict):
    name: str
    status: GateStatus
    evidence: str


class ConfigIdentity(TypedDict):
    normalization: str
    sha256: str
    bytes: int


class ModelIdentity(TypedDict):
    base_model_id: str
    base_model_revision: str
    base_model_revision_is_immutable: bool
    tokenizer_revision: str
    tokenizer_revision_is_immutable: bool


class DatasetIdentity(TypedDict):
    language: str
    role: Literal["root", "paired"]
    dataset_id: str
    dataset_revision: str
    revision_is_immutable: bool


class SourceCodeIdentity(TypedDict):
    discovery: Literal["git-project-root-v1", "git-project-root-partial-v1", "unavailable"]
    git_commit: str
    working_tree_clean: bool | None
    git_status_sha256: str | None


class DependencyLockIdentity(TypedDict):
    path: str
    sha256: str | None
    bytes: int | None


class ArtifactTreeIdentity(TypedDict):
    algorithm: str
    excluded_paths: list[str]
    certified: bool
    sha256: str | None
    file_count: int | None
    total_bytes: int | None
    error: str | None


class ReleaseIdentity(TypedDict):
    config: ConfigIdentity
    model: ModelIdentity
    datasets: list[DatasetIdentity]
    source_code: SourceCodeIdentity
    dependency_lock: DependencyLockIdentity
    artifact_tree: ArtifactTreeIdentity


class PreflightReport(TypedDict):
    schema_version: str
    created_at: str
    status: Literal["pass", "fail"]
    identity: ReleaseIdentity
    gates: list[ReleaseGate]


def _gate(name: str, status: GateStatus, evidence: str) -> ReleaseGate:
    # Evidence strings can contain filesystem paths; redact home
    # directories so the written report passes its own secret scan.
    return ReleaseGate(name=name, status=status, evidence=redact_text(evidence))


def normalized_config_bytes(config: SommelierConfig) -> bytes:
    """Returns the stable normalized bytes certified by a v2 preflight."""
    try:
        normalized = json.dumps(
            config.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ValueError as error:
        raise InvariantViolation("release config contains a non-finite numeric value") from error
    return normalized.encode("utf-8")


def _config_identity(config: SommelierConfig) -> ConfigIdentity:
    normalized = normalized_config_bytes(config)
    return ConfigIdentity(
        normalization=PREFLIGHT_CONFIG_NORMALIZATION,
        sha256=hashlib.sha256(normalized).hexdigest(),
        bytes=len(normalized),
    )


def _model_identity(config: SommelierConfig) -> ModelIdentity:
    model = config.model
    return ModelIdentity(
        base_model_id=model.base_model_id,
        base_model_revision=model.base_model_revision,
        base_model_revision_is_immutable=_IMMUTABLE_REVISION.fullmatch(model.base_model_revision)
        is not None,
        tokenizer_revision=model.tokenizer_revision,
        tokenizer_revision_is_immutable=_IMMUTABLE_REVISION.fullmatch(model.tokenizer_revision)
        is not None,
    )


def _dataset_identities(config: SommelierConfig) -> list[DatasetIdentity]:
    root = config.root_dataset
    return [
        DatasetIdentity(
            language=source.language,
            role="root" if source is root else "paired",
            dataset_id=source.dataset_id,
            dataset_revision=source.dataset_revision,
            revision_is_immutable=_IMMUTABLE_REVISION.fullmatch(source.dataset_revision)
            is not None,
        )
        for source in config.datasets
    ]


def _git_environment() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}


def _run_git(
    project_root: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=True,
        capture_output=True,
        text=False,
        input=input_bytes,
        env=_git_environment(),
        timeout=_GIT_TIMEOUT_SECONDS,
    )


def _unavailable_source_identity() -> SourceCodeIdentity:
    return SourceCodeIdentity(
        discovery="unavailable",
        git_commit="unknown",
        working_tree_clean=None,
        git_status_sha256=None,
    )


def _partial_source_identity(revision: str) -> SourceCodeIdentity:
    return SourceCodeIdentity(
        discovery="git-project-root-partial-v1",
        git_commit=revision,
        working_tree_clean=None,
        git_status_sha256=None,
    )


def _exact_git_root_and_revision(project_root: Path) -> tuple[bool, str | None]:
    try:
        result = _run_git(project_root, "rev-parse", "--show-toplevel", "HEAD")
        lines = result.stdout.splitlines()
        if len(lines) != 2:
            return True, None
        top_level = Path(os.fsdecode(lines[0])).resolve()
        revision = lines[1].decode("ascii")
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeDecodeError,
    ):
        return False, None
    if top_level != project_root.resolve() or _IMMUTABLE_REVISION.fullmatch(revision) is None:
        return True, None
    return True, revision


@dataclass(frozen=True)
class _GitBlobIdentity:
    mode: str
    object_id: str


def _git_blob_identity_at_revision(
    project_root: Path,
    revision: str,
    relative_path: str,
) -> _GitBlobIdentity | None:
    try:
        tree = _run_git(project_root, "ls-tree", "-z", revision, "--", relative_path).stdout
        records = [record for record in tree.split(b"\0") if record]
        if len(records) != 1:
            return None
        metadata, separator, encoded_path = records[0].partition(b"\t")
        fields = metadata.split()
        if (
            separator != b"\t"
            or encoded_path.decode("utf-8") != relative_path
            or len(fields) != 3
            or fields[0] not in {b"100644", b"100755"}
            or fields[1] != b"blob"
        ):
            return None
        object_id = fields[2].decode("ascii")
        if _IMMUTABLE_REVISION.fullmatch(object_id) is None:
            return None
        return _GitBlobIdentity(mode=fields[0].decode("ascii"), object_id=object_id)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeDecodeError,
    ):
        return None


def _read_open_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _read_regular_file_bytes(path: Path) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError(f"not a regular file: {path}")
    flags = os.O_RDONLY
    for flag_name in ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC"):
        flags |= getattr(os, flag_name, 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != metadata.st_dev
            or opened.st_ino != metadata.st_ino
        ):
            raise OSError(f"regular file identity changed before read: {path}")
        data = _read_open_descriptor(descriptor)
        after = os.fstat(descriptor)
        path_after = path.lstat()
        verified_data = _read_open_descriptor(descriptor)
        verified = os.fstat(descriptor)
        if (
            not _same_stat_snapshot(opened, after)
            or not _same_stat_snapshot(after, path_after)
            or not _same_stat_snapshot(path_after, verified)
            or data != verified_data
        ):
            raise OSError(f"regular file changed while being read: {path}")
        return data
    finally:
        os.close(descriptor)


def _project_file_matches_revision(
    project_root: Path,
    revision: str,
    relative_path: str,
    data: bytes,
) -> bool:
    expected = _git_blob_identity_at_revision(project_root, revision, relative_path)
    if expected is None:
        return False
    try:
        observed = (
            _run_git(
                project_root,
                "hash-object",
                f"--path={relative_path}",
                "--stdin",
                input_bytes=data,
            )
            .stdout.decode("ascii")
            .strip()
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeDecodeError,
    ):
        return False
    return observed == expected.object_id


def _required_project_file_bytes(project_root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for relative_path in _REQUIRED_TRACKED_PROJECT_PATHS:
        try:
            files[relative_path] = _read_regular_file_bytes(project_root / relative_path)
        except OSError:
            continue
    return files


def _tracked_project_files_match(project_root: Path, revision: str) -> bool:
    files = _required_project_file_bytes(project_root)
    return all(
        (data := files.get(relative_path)) is not None
        and _project_file_matches_revision(project_root, revision, relative_path, data)
        for relative_path in _REQUIRED_TRACKED_PROJECT_PATHS
    )


def _git_index_has_hidden_paths(project_root: Path) -> bool:
    try:
        records = _run_git(project_root, "ls-files", "-v", "-z").stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return True
    return any(record and (record[:1] == b"S" or chr(record[0]).islower()) for record in records)


def _discover_project_source(project_root: Path) -> SourceCodeIdentity:
    discovered, revision = _exact_git_root_and_revision(project_root)
    if not discovered or revision is None:
        return _unavailable_source_identity()

    try:
        status = _run_git(
            project_root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=normal",
            "--ignore-submodules=none",
        ).stdout
        if _git_index_has_hidden_paths(project_root) or not _tracked_project_files_match(
            project_root,
            revision,
        ):
            return _partial_source_identity(revision)
        revision_after = _run_git(project_root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
        status_after = _run_git(
            project_root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=normal",
            "--ignore-submodules=none",
        ).stdout
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        UnicodeDecodeError,
    ):
        return _partial_source_identity(revision)
    if (
        revision_after != revision
        or status_after != status
        or not _tracked_project_files_match(
            project_root,
            revision,
        )
    ):
        return _partial_source_identity(revision)

    return SourceCodeIdentity(
        discovery="git-project-root-v1",
        git_commit=revision,
        working_tree_clean=not bool(status),
        git_status_sha256=hashlib.sha256(status).hexdigest(),
    )


@dataclass(frozen=True)
class _ReadLockFromDisk:
    pass


_READ_LOCK_FROM_DISK: Final = _ReadLockFromDisk()


def _dependency_lock_identity(
    project_root: Path,
    *,
    expected_git_commit: str | None = None,
    lock_bytes: bytes | None | _ReadLockFromDisk = _READ_LOCK_FROM_DISK,
) -> DependencyLockIdentity:
    if isinstance(lock_bytes, _ReadLockFromDisk):
        try:
            data = _read_regular_file_bytes(project_root / "uv.lock")
        except OSError:
            return DependencyLockIdentity(path="uv.lock", sha256=None, bytes=None)
    elif lock_bytes is not None:
        data = lock_bytes
    else:
        return DependencyLockIdentity(path="uv.lock", sha256=None, bytes=None)

    discovered, observed_revision = _exact_git_root_and_revision(project_root)
    revision = expected_git_commit or observed_revision
    if (
        not discovered
        or observed_revision is None
        or revision is None
        or revision != observed_revision
        or not _project_file_matches_revision(project_root, revision, "uv.lock", data)
    ):
        return DependencyLockIdentity(path="uv.lock", sha256=None, bytes=None)

    return DependencyLockIdentity(
        path="uv.lock",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )


@dataclass(frozen=True)
class _ArtifactEntry:
    path: Path
    relative_path: str
    metadata: os.stat_result
    is_regular_file: bool


def _same_stat_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
    )


def _canonical_artifact_relative_path(path: Path, artifact_root: Path) -> str:
    relative = path.relative_to(artifact_root).as_posix()
    posix_path = PurePosixPath(relative)
    if (
        posix_path.is_absolute()
        or relative != posix_path.as_posix()
        or not posix_path.parts
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise InvariantViolation(
            f"release preflight artifact tree has a non-canonical path: {relative!r}"
        )
    return relative


def _artifact_tree_snapshot(
    artifact_root: Path,
) -> tuple[bool, os.stat_result | None, list[_ArtifactEntry]]:
    try:
        root_metadata = artifact_root.lstat()
    except FileNotFoundError:
        return False, None, []
    if stat.S_ISLNK(root_metadata.st_mode):
        raise InvariantViolation("release preflight artifact root cannot be a symbolic link")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise InvariantViolation("release preflight artifact root must be a directory")

    entries: list[_ArtifactEntry] = []
    for path in artifact_root.rglob("*"):
        relative = _canonical_artifact_relative_path(path, artifact_root)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise InvariantViolation(
                f"release preflight artifact tree contains a symbolic link: {relative}"
            )
        if stat.S_ISREG(metadata.st_mode):
            is_regular_file = True
        elif stat.S_ISDIR(metadata.st_mode):
            is_regular_file = False
        else:
            raise InvariantViolation(
                f"release preflight artifact tree contains a non-regular entry: {relative}"
            )
        entries.append(
            _ArtifactEntry(
                path=path,
                relative_path=relative,
                metadata=metadata,
                is_regular_file=is_regular_file,
            )
        )
    entries.sort(key=lambda entry: entry.relative_path)
    return True, root_metadata, entries


def _artifact_snapshot_signature(
    root_metadata: os.stat_result | None,
    entries: list[_ArtifactEntry],
) -> tuple[object, ...]:
    root_signature: tuple[object, ...] | None = None
    if root_metadata is not None:
        root_signature = (
            stat.S_IFMT(root_metadata.st_mode),
            root_metadata.st_dev,
            root_metadata.st_ino,
            root_metadata.st_size,
            root_metadata.st_mtime_ns,
            root_metadata.st_ctime_ns,
        )
    return (
        root_signature,
        tuple(
            (
                entry.relative_path,
                entry.is_regular_file,
                stat.S_IFMT(entry.metadata.st_mode),
                entry.metadata.st_dev,
                entry.metadata.st_ino,
                entry.metadata.st_size,
                entry.metadata.st_mtime_ns,
                entry.metadata.st_ctime_ns,
            )
            for entry in entries
        ),
    )


def _scan_artifact_bytes(data: bytes, *, relative_path: str) -> list[RedactionFinding]:
    suffix = PurePosixPath(relative_path).suffix.lower()
    if suffix not in SCANNABLE_SUFFIXES:
        return []
    text = data.decode("utf-8")
    return scan_artifact_text(text, file=relative_path, suffix=suffix)


def _read_artifact_entry(
    entry: _ArtifactEntry,
    *,
    capture_scannable: bool = True,
) -> tuple[str, int, bytes | None]:
    flags = os.O_RDONLY
    for flag_name in ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC"):
        flags |= getattr(os, flag_name, 0)
    descriptor = os.open(entry.path, flags)
    try:
        opened = os.fstat(descriptor)
        if not _same_stat_snapshot(entry.metadata, opened):
            raise InvariantViolation(
                f"release preflight artifact changed before read: {entry.relative_path}"
            )
        digest = hashlib.sha256()
        byte_count = 0
        scannable = (
            capture_scannable
            and PurePosixPath(entry.relative_path).suffix.lower() in SCANNABLE_SUFFIXES
        )
        scanned_bytes: bytearray | None = bytearray() if scannable else None
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
            if scanned_bytes is not None:
                scanned_bytes.extend(chunk)
        after = os.fstat(descriptor)
        try:
            path_after = entry.path.lstat()
        except FileNotFoundError as error:
            raise InvariantViolation(
                f"release preflight artifact disappeared during read: {entry.relative_path}"
            ) from error
        if (
            byte_count != opened.st_size
            or not _same_stat_snapshot(opened, after)
            or not _same_stat_snapshot(after, path_after)
        ):
            raise InvariantViolation(
                f"release preflight artifact changed during read: {entry.relative_path}"
            )
        return (
            digest.hexdigest(),
            byte_count,
            None if scanned_bytes is None else bytes(scanned_bytes),
        )
    finally:
        os.close(descriptor)


def _inspect_artifact_tree(
    artifact_root: Path,
) -> tuple[ArtifactTreeIdentity, list[RedactionFinding], bool]:
    existed, root_before, snapshot_before = _artifact_tree_snapshot(artifact_root)
    manifest_entries: list[dict[str, object]] = []
    observed_files: dict[str, tuple[str, int]] = {}
    findings: list[RedactionFinding] = []
    total_bytes = 0
    for entry in snapshot_before:
        if not entry.is_regular_file:
            continue
        file_sha256, byte_count, scanned_bytes = _read_artifact_entry(entry)
        observed_files[entry.relative_path] = (file_sha256, byte_count)
        if scanned_bytes is not None:
            findings.extend(_scan_artifact_bytes(scanned_bytes, relative_path=entry.relative_path))
        if entry.relative_path in PREFLIGHT_TREE_EXCLUDED_PATHS:
            continue
        total_bytes += byte_count
        manifest_entries.append(
            {"path": entry.relative_path, "bytes": byte_count, "sha256": file_sha256}
        )

    existed_after, root_after, snapshot_after = _artifact_tree_snapshot(artifact_root)
    if existed_after != existed or _artifact_snapshot_signature(
        root_before, snapshot_before
    ) != _artifact_snapshot_signature(root_after, snapshot_after):
        raise InvariantViolation("release preflight artifact tree changed during certification")

    for entry in snapshot_after:
        if not entry.is_regular_file:
            continue
        verified_sha256, verified_bytes, _ = _read_artifact_entry(
            entry,
            capture_scannable=False,
        )
        if observed_files.get(entry.relative_path) != (verified_sha256, verified_bytes):
            raise InvariantViolation(
                f"release preflight artifact content changed during certification: "
                f"{entry.relative_path}"
            )

    existed_verified, root_verified, snapshot_verified = _artifact_tree_snapshot(artifact_root)
    if existed_verified != existed_after or _artifact_snapshot_signature(
        root_after, snapshot_after
    ) != _artifact_snapshot_signature(root_verified, snapshot_verified):
        raise InvariantViolation("release preflight artifact tree changed during certification")

    manifest = json.dumps(
        manifest_entries,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    identity = ArtifactTreeIdentity(
        algorithm=PREFLIGHT_ARTIFACT_TREE_ALGORITHM,
        excluded_paths=list(PREFLIGHT_TREE_EXCLUDED_PATHS),
        certified=True,
        sha256=hashlib.sha256(manifest).hexdigest(),
        file_count=len(manifest_entries),
        total_bytes=total_bytes,
        error=None,
    )
    return identity, findings, existed


def certify_artifact_tree(artifact_root: Path) -> ArtifactTreeIdentity:
    """Certifies and secret-scans one stable byte snapshot of an artifact tree."""
    identity, _, _ = _inspect_artifact_tree(artifact_root)
    return identity


def _failed_artifact_tree_identity(error: Exception) -> ArtifactTreeIdentity:
    message = redact_text(str(error))[:500] or "artifact tree certification failed"
    return ArtifactTreeIdentity(
        algorithm=PREFLIGHT_ARTIFACT_TREE_ALGORITHM,
        excluded_paths=list(PREFLIGHT_TREE_EXCLUDED_PATHS),
        certified=False,
        sha256=None,
        file_count=None,
        total_bytes=None,
        error=message,
    )


def _evaluate_release_inputs(
    config: SommelierConfig,
    *,
    project_root: Path,
    artifact_root: Path,
    environ: Mapping[str, str] | None = None,
) -> tuple[
    list[ReleaseGate],
    SourceCodeIdentity,
    DependencyLockIdentity,
    ArtifactTreeIdentity,
]:
    """Evaluates gates and identities from one coherent release-input pass."""
    env = environ if environ is not None else os.environ
    gates: list[ReleaseGate] = []
    project_files = _required_project_file_bytes(project_root)
    source_code = _discover_project_source(project_root)
    expected_commit = (
        source_code["git_commit"] if source_code["discovery"] != "unavailable" else None
    )

    def certified_project_file(relative_path: str) -> bytes | None:
        data = project_files.get(relative_path)
        if (
            data is None
            or expected_commit is None
            or not _project_file_matches_revision(
                project_root,
                expected_commit,
                relative_path,
                data,
            )
        ):
            return None
        return data

    license_path = project_root / "LICENSE"
    license_bytes = certified_project_file("LICENSE")
    gates.append(
        _gate(
            "project_license",
            "pass" if license_bytes is not None else "fail",
            f"certified tracked regular project license {license_path}"
            if license_bytes is not None
            else f"could not certify tracked regular project license {license_path}",
        )
    )

    notices_path = project_root / "licenses" / "THIRD_PARTY.md"
    notices_text = ""
    notices_bytes = certified_project_file("licenses/THIRD_PARTY.md")
    if notices_bytes is not None:
        try:
            notices_text = notices_bytes.decode("utf-8")
        except UnicodeDecodeError:
            notices_bytes = None
    gates.append(
        _gate(
            "third_party_notices",
            "pass" if notices_bytes is not None else "fail",
            f"certified tracked UTF-8 third-party notices {notices_path}"
            if notices_bytes is not None
            else f"could not certify tracked UTF-8 third-party notices {notices_path}",
        )
    )

    base_model = config.model.base_model_id
    gates.append(
        _gate(
            "base_model_obligations",
            "pass" if base_model in notices_text else "fail",
            f"looked for {base_model!r} in third-party notices",
        )
    )
    dataset = config.root_dataset.dataset_id
    gates.append(
        _gate(
            "dataset_license",
            "pass" if dataset in notices_text else "fail",
            f"looked for {dataset!r} in third-party notices",
        )
    )
    gates.append(
        _gate(
            "derived_artifact_notice",
            "pass" if REQUIRED_DERIVED_NOTICE in notices_text else "fail",
            f"looked for required notice {REQUIRED_DERIVED_NOTICE!r} in third-party notices",
        )
    )

    acknowledged = env.get(ACK_ENV_NAME, "")
    gates.append(
        _gate(
            "base_model_license_ack",
            "pass" if acknowledged == base_model else "fail",
            f"{ACK_ENV_NAME} must be set to {base_model!r} to acknowledge "
            "the base model license terms",
        )
    )

    lock_bytes = certified_project_file("uv.lock")
    dependency_lock = _dependency_lock_identity(
        project_root,
        expected_git_commit=expected_commit,
        lock_bytes=lock_bytes,
    )
    lock_path = project_root / dependency_lock["path"]
    gates.append(
        _gate(
            "dependency_lock",
            "pass" if dependency_lock["sha256"] is not None else "fail",
            f"checked tracked regular dependency lock {lock_path}"
            if dependency_lock["sha256"] is not None
            else f"could not certify tracked regular dependency lock {lock_path}",
        )
    )

    try:
        artifact_tree, findings, artifact_root_existed = _inspect_artifact_tree(artifact_root)
    except (InvariantViolation, OSError, UnicodeDecodeError) as error:
        artifact_tree = _failed_artifact_tree_identity(error)
        gates.append(
            _gate(
                "artifact_secret_scan",
                "fail",
                f"artifact tree could not be safely inspected: {error}",
            )
        )
    else:
        if not artifact_root_existed:
            gates.append(
                _gate(
                    "artifact_secret_scan",
                    "skip",
                    f"no artifacts present under {artifact_root}",
                )
            )
        elif findings:
            first = findings[0]
            gates.append(
                _gate(
                    "artifact_secret_scan",
                    "fail",
                    f"{len(findings)} finding(s); first: {first['kind']} in "
                    f"{first['file']} at {first['location']}",
                )
            )
        else:
            gates.append(
                _gate(
                    "artifact_secret_scan",
                    "pass",
                    f"inspected and certified artifacts under {artifact_root}",
                )
            )

    return gates, source_code, dependency_lock, artifact_tree


def build_release_gates(
    config: SommelierConfig,
    *,
    project_root: Path,
    artifact_root: Path,
    environ: Mapping[str, str] | None = None,
) -> list[ReleaseGate]:
    """Evaluates every release gate without raising."""
    gates, _, _, _ = _evaluate_release_inputs(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=environ,
    )
    return gates


def _closed_mapping(
    value: object,
    *,
    required: frozenset[str],
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise InvariantViolation(f"{context} must be a JSON object with string keys")
    payload = cast(Mapping[str, object], value)
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - required)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise InvariantViolation(f"{context} contract drift: {'; '.join(details)}")
    return payload


def _validate_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise InvariantViolation(f"{context} is not a SHA-256")
    return value


def _validate_config_identity(value: object, *, config: SommelierConfig) -> None:
    payload = _closed_mapping(
        value,
        required=frozenset({"normalization", "sha256", "bytes"}),
        context="release preflight config identity",
    )
    _validate_sha256(payload["sha256"], context="release preflight config digest")
    byte_count = payload["bytes"]
    if (
        payload["normalization"] != PREFLIGHT_CONFIG_NORMALIZATION
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise InvariantViolation("release preflight config identity is invalid")
    expected = _config_identity(config)
    if dict(payload) != expected:
        raise InvariantViolation("release preflight config identity does not match the config")


def _validate_model_identity(value: object, *, config: SommelierConfig) -> ModelIdentity:
    required = frozenset(
        {
            "base_model_id",
            "base_model_revision",
            "base_model_revision_is_immutable",
            "tokenizer_revision",
            "tokenizer_revision_is_immutable",
        }
    )
    payload = _closed_mapping(value, required=required, context="release preflight model identity")
    for field in (
        "base_model_id",
        "base_model_revision",
        "tokenizer_revision",
    ):
        if not isinstance(payload[field], str):
            raise InvariantViolation(f"release preflight model identity {field} is invalid")
    for field in (
        "base_model_revision_is_immutable",
        "tokenizer_revision_is_immutable",
    ):
        if not isinstance(payload[field], bool):
            raise InvariantViolation(f"release preflight model identity {field} is invalid")
    expected = _model_identity(config)
    if dict(payload) != expected:
        raise InvariantViolation("release preflight model identity does not match the config")
    return expected


def _validate_dataset_identities(
    value: object,
    *,
    config: SommelierConfig,
) -> list[DatasetIdentity]:
    if not isinstance(value, list):
        raise InvariantViolation("release preflight datasets identity must be a JSON array")
    required = frozenset(
        {"language", "role", "dataset_id", "dataset_revision", "revision_is_immutable"}
    )
    for index, raw_dataset in enumerate(value):
        dataset = _closed_mapping(
            raw_dataset,
            required=required,
            context=f"release preflight dataset identity[{index}]",
        )
        for field in ("language", "dataset_id", "dataset_revision"):
            if not isinstance(dataset[field], str):
                raise InvariantViolation(
                    f"release preflight dataset identity[{index}].{field} is invalid"
                )
        role = dataset["role"]
        if (
            not isinstance(role, str)
            or role not in {"root", "paired"}
            or not isinstance(dataset["revision_is_immutable"], bool)
        ):
            raise InvariantViolation(f"release preflight dataset identity[{index}] is invalid")
    expected = _dataset_identities(config)
    if value != expected:
        raise InvariantViolation("release preflight dataset identities do not match the config")
    return expected


def _validate_source_identity(value: object) -> SourceCodeIdentity:
    payload = _closed_mapping(
        value,
        required=frozenset({"discovery", "git_commit", "working_tree_clean", "git_status_sha256"}),
        context="release preflight source-code identity",
    )
    discovery = payload["discovery"]
    revision = payload["git_commit"]
    clean = payload["working_tree_clean"]
    status_sha256 = payload["git_status_sha256"]
    if not isinstance(discovery, str) or discovery not in {
        "git-project-root-v1",
        "git-project-root-partial-v1",
        "unavailable",
    }:
        raise InvariantViolation("release preflight source-code discovery method is invalid")
    if not isinstance(revision, str):
        raise InvariantViolation("release preflight source-code revision must be a string")
    if clean is not None and not isinstance(clean, bool):
        raise InvariantViolation("release preflight source-code clean flag is invalid")
    if discovery == "unavailable":
        if revision != "unknown" or clean is not None or status_sha256 is not None:
            raise InvariantViolation("unavailable release source identity contains Git claims")
    else:
        if _IMMUTABLE_REVISION.fullmatch(revision) is None:
            raise InvariantViolation("release preflight source revision is not immutable")
        if discovery == "git-project-root-partial-v1":
            if clean is not None or status_sha256 is not None:
                raise InvariantViolation("partial release source identity contains status claims")
        else:
            digest = _validate_sha256(
                status_sha256,
                context="release preflight Git status digest",
            )
            if clean is None or clean != (digest == _EMPTY_SHA256):
                raise InvariantViolation(
                    "release preflight source clean flag disagrees with its status digest"
                )
    return cast(SourceCodeIdentity, dict(payload))


def _validate_dependency_lock_identity(
    value: object,
    *,
    dependency_gate_status: GateStatus,
) -> DependencyLockIdentity:
    payload = _closed_mapping(
        value,
        required=frozenset({"path", "sha256", "bytes"}),
        context="release preflight dependency-lock identity",
    )
    path = payload["path"]
    digest = payload["sha256"]
    byte_count = payload["bytes"]
    if path != "uv.lock":
        raise InvariantViolation("release preflight dependency-lock path drift")
    if digest is None or byte_count is None:
        if digest is not None or byte_count is not None or dependency_gate_status != "fail":
            raise InvariantViolation("release preflight dependency-lock identity is incomplete")
    else:
        _validate_sha256(digest, context="release preflight dependency-lock digest")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise InvariantViolation("release preflight dependency-lock byte count is invalid")
        if dependency_gate_status != "pass":
            raise InvariantViolation(
                "release preflight dependency-lock identity disagrees with its gate"
            )
    return cast(DependencyLockIdentity, dict(payload))


def _validate_artifact_tree_identity(
    value: object,
    *,
    artifact_gate_status: GateStatus,
) -> ArtifactTreeIdentity:
    payload = _closed_mapping(
        value,
        required=frozenset(
            {
                "algorithm",
                "excluded_paths",
                "certified",
                "sha256",
                "file_count",
                "total_bytes",
                "error",
            }
        ),
        context="release preflight artifact-tree identity",
    )
    if payload["algorithm"] != PREFLIGHT_ARTIFACT_TREE_ALGORITHM:
        raise InvariantViolation("release preflight artifact-tree algorithm drift")
    if payload["excluded_paths"] != list(PREFLIGHT_TREE_EXCLUDED_PATHS):
        raise InvariantViolation("release preflight artifact-tree exclusion drift")
    certified = payload["certified"]
    if not isinstance(certified, bool):
        raise InvariantViolation("release preflight artifact-tree certified flag is invalid")
    if certified:
        _validate_sha256(payload["sha256"], context="release preflight artifact-tree digest")
        for field in ("file_count", "total_bytes"):
            measurement = payload[field]
            if isinstance(measurement, bool) or not isinstance(measurement, int) or measurement < 0:
                raise InvariantViolation(f"release preflight artifact-tree {field} is invalid")
        if payload["error"] is not None:
            raise InvariantViolation("certified release artifact tree contains an error")
    else:
        if any(payload[field] is not None for field in ("sha256", "file_count", "total_bytes")):
            raise InvariantViolation("uncertified release artifact tree contains digest claims")
        if not isinstance(payload["error"], str) or not payload["error"]:
            raise InvariantViolation("uncertified release artifact tree has no error evidence")
        if artifact_gate_status != "fail":
            raise InvariantViolation(
                "uncertified release artifact tree disagrees with the artifact gate"
            )
    return cast(ArtifactTreeIdentity, dict(payload))


def validate_release_preflight_report(
    report: object,
    *,
    config: SommelierConfig,
    artifact_root: Path | None = None,
    project_root: Path | None = None,
    expected_source_git_commit: str | None = None,
    expected_dependency_lock_sha256: str | None = None,
    require_pass: bool = False,
    require_all_gates_pass: bool = False,
    require_clean_source: bool = False,
    require_immutable_revisions: bool = False,
) -> PreflightReport:
    """Validates the closed v2 contract and optionally re-certifies inputs.

    Adapter publication must run preflight against the final curated bundle,
    after every dataset revision is pinned; the report itself is the only
    excluded path. Publication should then provide the resolved config, that
    bundle as ``artifact_root``, the training source revision, and the training
    manifest's dependency-lock digest, while requiring
    pass/all-pass/clean/immutable. This makes a passing report non-replayable
    across configs, source snapshots, or bundle trees.
    """
    payload = _closed_mapping(
        report,
        required=frozenset({"schema_version", "created_at", "status", "identity", "gates"}),
        context="release preflight report",
    )
    if payload["schema_version"] != PREFLIGHT_SCHEMA:
        raise InvariantViolation("release preflight report schema drift")
    created_at = payload["created_at"]
    if not isinstance(created_at, str):
        raise InvariantViolation("release preflight created_at must be a string")
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError as error:
        raise InvariantViolation("release preflight created_at is invalid") from error
    if created.tzinfo is None:
        raise InvariantViolation("release preflight created_at must include a timezone")

    gates_value = payload["gates"]
    if not isinstance(gates_value, list):
        raise InvariantViolation("release preflight gates must be a JSON array")
    gates: list[ReleaseGate] = []
    for index, raw_gate in enumerate(gates_value):
        gate = _closed_mapping(
            raw_gate,
            required=frozenset({"name", "status", "evidence"}),
            context=f"release preflight gate[{index}]",
        )
        name = gate["name"]
        status = gate["status"]
        evidence = gate["evidence"]
        if (
            not isinstance(name, str)
            or not isinstance(status, str)
            or status not in {"pass", "fail", "skip"}
        ):
            raise InvariantViolation(f"release preflight gate[{index}] is invalid")
        if not isinstance(evidence, str) or not evidence:
            raise InvariantViolation(f"release preflight gate[{index}] has no evidence")
        gates.append(cast(ReleaseGate, dict(gate)))
    if [gate["name"] for gate in gates] != list(REQUIRED_RELEASE_GATES):
        raise InvariantViolation("release preflight gate set or order drift")
    by_name = {gate["name"]: gate["status"] for gate in gates}
    expected_status = "fail" if "fail" in by_name.values() else "pass"
    if payload["status"] != expected_status:
        raise InvariantViolation("release preflight aggregate status disagrees with its gates")
    if require_pass and expected_status != "pass":
        raise InvariantViolation("release preflight report is not passing")
    if require_all_gates_pass and any(status != "pass" for status in by_name.values()):
        raise InvariantViolation("release preflight did not pass every required gate")

    identity = _closed_mapping(
        payload["identity"],
        required=frozenset(
            {"config", "model", "datasets", "source_code", "dependency_lock", "artifact_tree"}
        ),
        context="release preflight identity",
    )
    _validate_config_identity(identity["config"], config=config)
    model = _validate_model_identity(identity["model"], config=config)
    datasets = _validate_dataset_identities(identity["datasets"], config=config)
    source = _validate_source_identity(identity["source_code"])
    dependency_lock = _validate_dependency_lock_identity(
        identity["dependency_lock"],
        dependency_gate_status=by_name["dependency_lock"],
    )
    artifact_tree = _validate_artifact_tree_identity(
        identity["artifact_tree"],
        artifact_gate_status=by_name["artifact_secret_scan"],
    )

    if require_immutable_revisions and (
        not model["base_model_revision_is_immutable"]
        or not model["tokenizer_revision_is_immutable"]
        or any(not dataset["revision_is_immutable"] for dataset in datasets)
    ):
        raise InvariantViolation("release preflight contains a mutable input revision")
    if expected_source_git_commit is not None:
        if (
            not isinstance(expected_source_git_commit, str)
            or _IMMUTABLE_REVISION.fullmatch(expected_source_git_commit) is None
        ):
            raise InvariantViolation("expected release source revision is not immutable")
        if source["git_commit"] != expected_source_git_commit:
            raise InvariantViolation("release preflight source revision does not match the run")
    if require_clean_source and source["working_tree_clean"] is not True:
        raise InvariantViolation("release preflight source snapshot is not clean")
    if expected_dependency_lock_sha256 is not None:
        _validate_sha256(
            expected_dependency_lock_sha256,
            context="expected release dependency-lock digest",
        )
        if dependency_lock["sha256"] != expected_dependency_lock_sha256:
            raise InvariantViolation("release preflight dependency lock does not match the run")
    if project_root is not None:
        expected_commit = source["git_commit"] if source["discovery"] != "unavailable" else None
        observed_lock = _dependency_lock_identity(
            project_root,
            expected_git_commit=expected_commit,
        )
        if dependency_lock != observed_lock:
            raise InvariantViolation(
                "release preflight dependency lock does not match the project tree"
            )
    if artifact_root is not None:
        observed_tree, observed_findings, _ = _inspect_artifact_tree(artifact_root)
        if artifact_tree != observed_tree:
            raise InvariantViolation(
                "release preflight artifact-tree identity does not match the on-disk tree"
            )
        if observed_findings and by_name["artifact_secret_scan"] != "fail":
            raise InvariantViolation(
                "release preflight artifact scan gate does not match the on-disk tree"
            )

    return cast(PreflightReport, dict(payload))


def _ensure_safe_report_directory(artifact_root: Path) -> None:
    try:
        metadata = artifact_root.lstat()
    except FileNotFoundError:
        try:
            artifact_root.mkdir(parents=True)
            metadata = artifact_root.lstat()
        except OSError as error:
            raise SecurityPolicyError(
                "release preflight report cannot be written safely",
                hint="No report was written because the artifact root could not be created safely.",
            ) from error
    except OSError as error:
        raise SecurityPolicyError(
            "release preflight report cannot be written safely",
            hint="No report was written because the artifact root could not be inspected safely.",
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SecurityPolicyError(
            "release preflight report cannot be written safely",
            hint=(
                "No report was written because the artifact root is a symbolic link or is not "
                "a directory."
            ),
        )


def run_release_preflight(
    config: SommelierConfig,
    *,
    project_root: Path,
    artifact_root: Path,
    environ: Mapping[str, str] | None = None,
) -> PreflightReport:
    """Writes release_preflight.json and fails closed on failing gates.

    A failing secret scan raises SecurityPolicyError (exit 5); any other
    failing gate raises ExternalDependencyError (exit 3), matching the
    license-gate contract. The report is written before raising so the evidence
    survives the failure. The v2 identity excludes only the report itself from
    its certified artifact-tree digest.
    """
    gates, source_code, dependency_lock, artifact_tree = _evaluate_release_inputs(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=environ,
    )

    failed = [gate for gate in gates if gate["status"] == "fail"]
    report = PreflightReport(
        schema_version=PREFLIGHT_SCHEMA,
        created_at=datetime.now(UTC).isoformat(),
        status="fail" if failed else "pass",
        identity=ReleaseIdentity(
            config=_config_identity(config),
            model=_model_identity(config),
            datasets=_dataset_identities(config),
            source_code=source_code,
            dependency_lock=dependency_lock,
            artifact_tree=artifact_tree,
        ),
        gates=gates,
    )
    validate_release_preflight_report(report, config=config)

    _ensure_safe_report_directory(artifact_root)
    report_path = artifact_root / PREFLIGHT_FILENAME

    def writer(temp_path: Path) -> None:
        temp_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    write_artifact_atomic(report_path, writer, artifact_root=artifact_root)

    if failed:
        failed_names = ", ".join(gate["name"] for gate in failed)
        if any(gate["name"] == "artifact_secret_scan" for gate in failed):
            raise SecurityPolicyError(
                f"release preflight failed: {failed_names}",
                hint=f"See {report_path} for gate evidence.",
            )
        raise ExternalDependencyError(
            f"release preflight failed: {failed_names}",
            hint=f"See {report_path} for gate evidence; acknowledge the base "
            f"model license by setting {ACK_ENV_NAME}.",
        )
    return report
