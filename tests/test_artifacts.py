from pathlib import Path

import pytest

from sommelier.artifacts import (
    make_artifact_ref,
    sha256_bytes,
    sha256_file,
    write_artifact_atomic,
)
from sommelier.errors import SchemaValidationError


def test_sha256_bytes_and_file(tmp_path: Path) -> None:
    payload = b"artifact-bytes"
    path = tmp_path / "artifact.json"
    path.write_bytes(payload)
    assert sha256_bytes(payload) == sha256_file(path)


def test_atomic_write_creates_checksum(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "artifact.json"

    def writer(path: Path) -> None:
        path.write_text('{"schema_version":"sommelier.manifest.v1"}', encoding="utf-8")

    ref = write_artifact_atomic(
        target,
        writer,
        kind="manifest",
        schema_version="sommelier.manifest.v1",
    )
    assert target.exists()
    assert ref["bytes"] == target.stat().st_size
    assert ref["sha256"] == sha256_file(target)


def test_atomic_write_cleans_up_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"

    def writer(_path: Path) -> None:
        raise RuntimeError("write failed")

    with pytest.raises(RuntimeError):
        write_artifact_atomic(target, writer)
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_make_artifact_ref_rejects_paths_outside_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(SchemaValidationError):
        make_artifact_ref(
            outside,
            artifact_root=artifact_root,
            kind="file",
            schema_version="sommelier.manifest.v1",
        )


def test_partial_write_not_published(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    seen: list[Path] = []

    def writer(path: Path) -> None:
        seen.append(path)
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError):
        write_artifact_atomic(target, writer)
    assert not target.exists()
    assert seen
    assert not seen[0].exists()
