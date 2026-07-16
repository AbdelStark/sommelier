import os
from pathlib import Path

import pytest

import sommelier.artifacts as artifacts
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


@pytest.mark.parametrize("occupied_kind", ["file", "symlink"])
def test_atomic_write_can_publish_new_evidence_without_replacing_target(
    tmp_path: Path,
    occupied_kind: str,
) -> None:
    target = tmp_path / "artifact.json"
    victim = tmp_path / "victim.json"
    if occupied_kind == "file":
        target.write_text("existing\n", encoding="utf-8")
    else:
        victim.write_text("victim\n", encoding="utf-8")
        target.symlink_to(victim)

    def writer(path: Path) -> None:
        path.write_text("replacement\n", encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="already exists and is immutable"):
        write_artifact_atomic(target, writer, replace_existing=False)

    expected = "existing\n" if occupied_kind == "file" else "victim\n"
    observed = target.read_text(encoding="utf-8")
    assert observed == expected
    if occupied_kind == "symlink":
        assert target.is_symlink()


def test_atomic_write_exclusively_publishes_absent_evidence(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"

    def writer(path: Path) -> None:
        path.write_text("new evidence\n", encoding="utf-8")

    ref = write_artifact_atomic(target, writer, replace_existing=False)

    assert target.read_text(encoding="utf-8") == "new evidence\n"
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


def test_atomic_write_does_not_follow_predictable_temp_symlink(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me\n", encoding="utf-8")
    predictable = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    predictable.symlink_to(victim)

    def writer(path: Path) -> None:
        path.write_text('{"safe": true}\n', encoding="utf-8")

    write_artifact_atomic(target, writer)

    assert victim.read_text(encoding="utf-8") == "keep me\n"
    assert target.read_text(encoding="utf-8") == '{"safe": true}\n'
    assert target.is_file()
    assert not target.is_symlink()
    assert predictable.is_symlink()


def test_atomic_write_rejects_writer_symlink_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me\n", encoding="utf-8")

    def writer(path: Path) -> None:
        path.symlink_to(victim)

    with pytest.raises(SchemaValidationError, match="regular file"):
        write_artifact_atomic(target, writer)

    assert victim.read_text(encoding="utf-8") == "keep me\n"
    assert not target.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["victim.txt"]


def test_atomic_write_rejects_artifact_root_escape_before_calling_writer(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("keep me\n", encoding="utf-8")
    writer_called = False

    def writer(path: Path) -> None:
        nonlocal writer_called
        writer_called = True
        path.write_text("clobbered\n", encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="escapes artifact root"):
        write_artifact_atomic(outside, writer, artifact_root=artifact_root)

    assert writer_called is False
    assert outside.read_text(encoding="utf-8") == "keep me\n"


def test_atomic_write_detects_content_change_without_timestamp_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("previous\n", encoding="utf-8")
    writer_path: Path | None = None

    def writer(path: Path) -> None:
        nonlocal writer_path
        writer_path = path
        path.write_bytes(b"first")

    original_hash = artifacts._hash_open_descriptor

    def mutate_then_hash(descriptor: int) -> tuple[str, int]:
        assert writer_path is not None
        metadata = writer_path.stat()
        writer_path.write_bytes(b"other")
        os.utime(
            writer_path,
            ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
        )
        return original_hash(descriptor)

    monkeypatch.setattr(artifacts, "_same_file_snapshot", lambda _left, _right: True)
    monkeypatch.setattr(artifacts, "_hash_open_descriptor", mutate_then_hash)

    with pytest.raises(SchemaValidationError, match="changed while it was being copied"):
        write_artifact_atomic(target, writer)

    assert target.read_text(encoding="utf-8") == "previous\n"
