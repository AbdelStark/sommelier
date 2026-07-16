from __future__ import annotations

from pathlib import Path

import pytest

from sommelier.errors import UserInputError
from sommelier.evaluation.generate import AdapterRef, adapter_tree_sha256


def test_local_adapter_tree_digest_is_stable_and_content_bound(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    nested = adapter / "nested"
    nested.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r":16}', encoding="utf-8")
    weights = nested / "adapter_model.safetensors"
    weights.write_bytes(b"weights-v1")

    first = adapter_tree_sha256(adapter)
    second = adapter_tree_sha256(adapter)
    assert first == second
    assert AdapterRef(source=str(adapter)).describe()["tree_sha256"] == first

    weights.write_bytes(b"weights-v2")
    assert adapter_tree_sha256(adapter) != first


def test_local_run_adapter_records_portable_artifact_path(tmp_path: Path) -> None:
    adapter = tmp_path / "artifacts" / "runs" / "v3-run" / "train" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    identity = AdapterRef(source=str(adapter)).describe()

    assert identity["artifact_path"] == "runs/v3-run/train/adapter"
    assert identity["tree_sha256"] == adapter_tree_sha256(adapter)


def test_hugging_face_adapter_identity_marks_mutable_revisions() -> None:
    mutable = AdapterRef(source="org/adapter", revision="main").describe()
    immutable = AdapterRef(source="org/adapter", revision="a" * 40).describe()
    assert mutable["revision_is_immutable"] is False
    assert immutable["revision_is_immutable"] is True


def test_local_adapter_tree_rejects_symlinks(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    target = tmp_path / "weights"
    target.write_bytes(b"weights")
    (adapter / "adapter_model.safetensors").symlink_to(target)
    with pytest.raises(UserInputError, match="contains a symlink"):
        adapter_tree_sha256(adapter)
