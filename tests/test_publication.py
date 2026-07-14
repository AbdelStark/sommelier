from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

import sommelier.publication as publication
import sommelier.release as release
from sommelier.artifacts import sha256_file
from sommelier.config import SommelierConfig, load_config
from sommelier.data.openai_evidence import OPENAI_PROVIDER_JOURNAL_FILENAME
from sommelier.data.prepare import paired_input_path
from sommelier.data.semantic_review import (
    SEMANTIC_REVIEW_FILENAME,
    SEMANTIC_REVIEW_TEMPLATE_FILENAME,
)
from sommelier.data.translate import (
    PUBLICATION_MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    translation_provenance_sidecar_path,
)
from sommelier.errors import (
    ExternalDependencyError,
    SecurityPolicyError,
    UserInputError,
)
from sommelier.evaluation.data_provenance import HEBREW_V3_PAIRED_DATASET_ID
from sommelier.evaluation.experiment import EXPERIMENT_REPORT_SCHEMA
from sommelier.publication import (
    ADAPTER_LICENSE_FILE_SHA256,
    ADAPTER_UPLOAD_OPTIONAL_FILES,
    ADAPTER_UPLOAD_REQUIRED_FILES,
    DATASET_REQUIRED_FILES,
    PreparedPublication,
    prepare_hebrew_adapter_publication,
    prepare_hebrew_dataset_publication,
)
from sommelier.publication import (
    _publish_prepared_bundle as publish_prepared_bundle,
)
from sommelier.release import ACK_ENV_NAME, REQUIRED_DERIVED_NOTICE, run_release_preflight


class FakeHubClient:
    def __init__(
        self,
        root: Path,
        *,
        repository_exists: bool = True,
        existing_files: set[str] | None = None,
    ) -> None:
        self.root = root
        self.repository_exists = repository_exists
        self.existing_files = (
            set(existing_files) if existing_files is not None else {".gitattributes"}
        )
        self.revision = "a" * 40
        self.head_revision: str | None = "b" * 40
        self.parent_commits: list[str | None] = []
        self.post_commit_files: set[str] | None = None
        self.corrupt_file: str | None = None
        self.symlink_download: str | None = None
        self.fail_commit = False
        self.events: list[str] = []
        self.committed: dict[str, bytes] = {}

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: publication.PublicationRepoType,
    ) -> None:
        self.events.append(f"create:{repo_type}:{repo_id}")
        if self.repository_exists:
            raise RuntimeError("repository already exists")
        self.repository_exists = True
        self.existing_files = {".gitattributes"} if self.head_revision is not None else set()

    def list_files(
        self,
        *,
        repo_id: str,
        repo_type: publication.PublicationRepoType,
        revision: str | None,
    ) -> Sequence[str]:
        del repo_id, repo_type
        self.events.append(f"list:{revision or 'head'}")
        if not self.repository_exists:
            raise RuntimeError("repository not found")
        if revision is None or not self.committed:
            return sorted(self.existing_files)
        if self.post_commit_files is not None:
            return sorted(self.post_commit_files)
        return sorted(set(self.committed) | {".gitattributes"})

    def resolve_revision(
        self,
        *,
        repo_id: str,
        repo_type: publication.PublicationRepoType,
    ) -> str | None:
        del repo_id, repo_type
        self.events.append("resolve")
        if not self.repository_exists:
            raise RuntimeError("repository not found")
        return self.head_revision

    def create_commit(
        self,
        *,
        repo_id: str,
        repo_type: publication.PublicationRepoType,
        files: Mapping[str, Path],
        commit_message: str,
        parent_commit: str | None,
    ) -> str:
        del repo_id, repo_type, commit_message
        self.events.append("commit")
        if self.fail_commit:
            raise RuntimeError("commit failed")
        self.parent_commits.append(parent_commit)
        self.committed = {name: path.read_bytes() for name, path in files.items()}
        return self.revision

    def download_file(
        self,
        *,
        repo_id: str,
        repo_type: publication.PublicationRepoType,
        filename: str,
        revision: str,
    ) -> Path:
        del repo_id, repo_type, revision
        self.events.append(f"download:{filename}")
        destination = self.root / "downloads" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if filename == self.symlink_download:
            destination.symlink_to(self.root / "missing-target")
            return destination
        data = self.committed[filename]
        if filename == self.corrupt_file:
            data += b"corrupt"
        destination.write_bytes(data)
        return destination


def _write_files(root: Path, names: Sequence[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {name}\n", encoding="utf-8")
        files[name] = path
    return files


def _prepared_dataset(tmp_path: Path) -> PreparedPublication:
    return PreparedPublication(
        repo_type="dataset",
        files=_write_files(tmp_path / "dataset", sorted(DATASET_REQUIRED_FILES)),
    )


def _prepared_adapter(tmp_path: Path, *, optional: bool = False) -> PreparedPublication:
    names = set(ADAPTER_UPLOAD_REQUIRED_FILES)
    if optional:
        names.update(ADAPTER_UPLOAD_OPTIONAL_FILES)
    return PreparedPublication(
        repo_type="model",
        files=_write_files(tmp_path / "adapter", sorted(names)),
    )


def _immutable_hebrew_config(tmp_path: Path) -> tuple[SommelierConfig, Path]:
    text = Path("examples/config.v3-he-full.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        text.replace("dataset_revision: main", f"dataset_revision: {'e' * 40}"),
        encoding="utf-8",
    )
    return load_config(config_path), config_path


def _dataset_bundle(root: Path) -> Path:
    root.mkdir()
    for name in DATASET_REQUIRED_FILES:
        path = root / name
        if name == "README.md":
            path.write_text(
                "---\nlicense: cc-by-4.0\n---\n"
                "# Hebrew machine-translated data\n"
                "Derived from Salesforce/xlam-function-calling-60k.\n",
                encoding="utf-8",
            )
        elif name.endswith(".jsonl"):
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text("{}\n", encoding="utf-8")
    return root


def _adapter_bundle(root: Path, config_path: Path, *, optional: bool = True) -> Path:
    root.mkdir()
    for name in publication.ADAPTER_REQUIRED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in ADAPTER_LICENSE_FILE_SHA256:
            shutil.copy2(Path("licenses") / name, path)
        elif name == "config.resolved.yaml":
            shutil.copy2(config_path, path)
        elif name.endswith(".json"):
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text(f"fixture for {name}\n", encoding="utf-8")
    if optional:
        tokenizer = root / "adapter" / "tokenizer.json"
        tokenizer.write_text("{}\n", encoding="utf-8")
    return root


def _valid_adapter_card(
    config: SommelierConfig,
    *,
    experiment_sha256: str,
    tree_sha256: str,
    source_revision: str,
    dataset_revision: str,
) -> str:
    return "\n".join(
        (
            "---",
            "license: llama3.1",
            f"base_model: {config.model.base_model_id}",
            "---",
            "# Llama Hebrew tool-calling adapter",
            REQUIRED_DERIVED_NOTICE,
            "NVIDIA Open Model License",
            "Llama 3.1 Community License",
            config.model.base_model_id,
            config.model.base_model_revision,
            experiment_sha256,
            tree_sha256,
            source_revision,
            dataset_revision,
        )
    )


def test_validate_only_is_default_and_performs_no_hub_calls(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)

    plan = publish_prepared_bundle(
        prepared,
        repo_id="owner/hebrew-data",
        commit_message="Publish audited Hebrew dataset",
        create_repo=True,
        client=client,
    )

    assert plan["status"] == "validated"
    assert plan["executed"] is False
    repository = cast("dict[str, object]", plan["repository"])
    assert repository["create_repo"] is True
    assert repository["commit_sha"] is None
    assert client.events == []


def test_validate_only_rejects_receipt_path(tmp_path: Path) -> None:
    with pytest.raises(UserInputError, match="only valid with execute"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Validate",
            receipt_path=tmp_path / "receipt.json",
        )


@pytest.mark.parametrize("repo_id", ["missing-namespace", "/owner/name", "owner/name/extra"])
def test_invalid_repo_ids_fail_before_hub_access(tmp_path: Path, repo_id: str) -> None:
    client = FakeHubClient(tmp_path)
    with pytest.raises(UserInputError, match="invalid Hugging Face repo ID"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id=repo_id,
            commit_message="Validate",
            client=client,
        )
    assert client.events == []


@pytest.mark.parametrize("message", ["", "   ", "line one\nline two", "line one\rline two"])
def test_invalid_commit_messages_fail_closed(tmp_path: Path, message: str) -> None:
    with pytest.raises(UserInputError, match="commit_message"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message=message,
        )


def test_execute_requires_exact_confirmation_and_receipt(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)
    with pytest.raises(UserInputError, match="does not exactly match"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/other",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )
    with pytest.raises(UserInputError, match="requires an explicit receipt"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            client=client,
        )
    assert client.events == []


def test_existing_repository_commit_is_round_trip_verified(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(
        tmp_path,
        existing_files={".gitattributes", "README.md"},
    )
    receipt_path = tmp_path / "receipts" / "dataset.json"

    result = publish_prepared_bundle(
        prepared,
        repo_id="owner/hebrew-data",
        commit_message="Publish audited Hebrew dataset",
        execute=True,
        confirmed_repo_id="owner/hebrew-data",
        receipt_path=receipt_path,
        client=client,
    )

    assert result["status"] == "verified"
    assert result["platform_files"] == [".gitattributes"]
    repository = cast("dict[str, object]", result["repository"])
    assert repository["commit_sha"] == "a" * 40
    assert repository["create_repo"] is False
    assert client.events[:4] == ["resolve", f"list:{'b' * 40}", "commit", f"list:{'a' * 40}"]
    assert client.parent_commits == ["b" * 40]
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == result
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    recorded = {
        cast("str", item["path"]): cast("str", item["sha256"])
        for item in cast("list[dict[str, object]]", result["files"])
    }
    assert recorded == prepared.sha256


def test_public_dataset_upload_uses_validated_private_snapshot_when_source_mutates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _dataset_bundle(tmp_path / "bundle")
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text('{"source": "en"}\n', encoding="utf-8")
    original_readme = (bundle / "README.md").read_bytes()
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    receipt_path = tmp_path / "receipt.json"

    def fake_validate(
        config: SommelierConfig,
        staged_root: Path,
    ) -> dict[str, dict[str, Path]]:
        del config
        return {"he": {"rows": paired_input_path(staged_root, "he")}}

    monkeypatch.setattr(publication, "validate_full_paired_input_contract", fake_validate)

    class SourceMutatingClient(FakeHubClient):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.upload_paths: dict[str, Path] = {}
            self.upload_modes: dict[str, int] = {}
            self.upload_sha256: dict[str, str] = {}

        def resolve_revision(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
        ) -> str | None:
            revision = super().resolve_revision(repo_id=repo_id, repo_type=repo_type)
            assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "pending"
            (bundle / "README.md").write_text(secret, encoding="utf-8")
            return revision

        def create_commit(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
            files: Mapping[str, Path],
            commit_message: str,
            parent_commit: str | None,
        ) -> str:
            self.upload_paths = dict(files)
            self.upload_modes = {name: path.stat().st_mode & 0o777 for name, path in files.items()}
            self.upload_sha256 = {name: sha256_file(path) for name, path in files.items()}
            return super().create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                files=files,
                commit_message=commit_message,
                parent_commit=parent_commit,
            )

    client = SourceMutatingClient(tmp_path)
    receipt = publication.publish_hebrew_dataset_bundle(
        config_path=Path("examples/config.v3-he-full.yaml"),
        bundle_dir=bundle,
        root_rows_path=root_rows,
        repo_id=HEBREW_V3_PAIRED_DATASET_ID,
        commit_message="Publish immutable Hebrew dataset snapshot",
        execute=True,
        confirmed_repo_id=HEBREW_V3_PAIRED_DATASET_ID,
        receipt_path=receipt_path,
        client=client,
    )

    assert receipt["status"] == "verified"
    assert (bundle / "README.md").read_text(encoding="utf-8") == secret
    assert client.committed["README.md"] == original_readme
    assert all(secret.encode() not in payload for payload in client.committed.values())
    assert all(not path.is_relative_to(bundle) for path in client.upload_paths.values())
    assert set(client.upload_modes.values()) == {0o400}
    readme_evidence = next(
        item
        for item in cast("list[dict[str, object]]", receipt["files"])
        if item["path"] == "README.md"
    )
    assert readme_evidence == {
        "path": "README.md",
        "sha256": client.upload_sha256["README.md"],
        "bytes": len(original_readme),
    }


def test_absent_repository_is_created_only_with_explicit_flag(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path, repository_exists=False)

    result = publish_prepared_bundle(
        prepared,
        repo_id="owner/hebrew-data",
        commit_message="Create audited Hebrew dataset",
        execute=True,
        create_repo=True,
        confirmed_repo_id="owner/hebrew-data",
        receipt_path=tmp_path / "receipt.json",
        client=client,
    )

    assert result["status"] == "verified"
    assert client.events[:3] == [
        "create:dataset:owner/hebrew-data",
        "resolve",
        f"list:{'b' * 40}",
    ]


def test_first_commit_to_new_empty_repository_allows_no_parent(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path, repository_exists=False)
    client.head_revision = None

    result = publish_prepared_bundle(
        prepared,
        repo_id="owner/hebrew-data",
        commit_message="Create audited Hebrew dataset",
        execute=True,
        create_repo=True,
        confirmed_repo_id="owner/hebrew-data",
        receipt_path=tmp_path / "receipt.json",
        client=client,
    )

    assert result["status"] == "verified"
    assert client.events[:5] == [
        "create:dataset:owner/hebrew-data",
        "resolve",
        "list:head",
        "resolve",
        "commit",
    ]
    assert client.parent_commits == [None]


def test_preexisting_empty_repository_cannot_receive_parentless_commit(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path, repository_exists=True, existing_files=set())
    client.head_revision = None

    with pytest.raises(ExternalDependencyError, match="not created by this publication"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Refuse ambiguous initial commit",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )

    assert client.events == ["resolve", "list:head"]
    assert "commit" not in client.events


def test_absent_repository_without_create_flag_fails_before_commit(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path, repository_exists=False)
    with pytest.raises(ExternalDependencyError, match="could not inspect"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )
    assert client.events == ["resolve"]


def test_create_flag_refuses_to_adopt_existing_repository(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path, repository_exists=True)
    with pytest.raises(ExternalDependencyError, match="could not create new public"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            create_repo=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )
    assert client.events == ["create:dataset:owner/hebrew-data"]


def test_production_create_repo_is_public_and_never_exist_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeApi:
        def create_repo(self, **kwargs: object) -> None:
            calls.update(kwargs)

    class FakeCommitOperationAdd:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    def fake_download(**kwargs: object) -> str:
        del kwargs
        return str(tmp_path / "download")

    module = ModuleType("huggingface_hub")
    setattr(module, "HfApi", FakeApi)
    setattr(module, "CommitOperationAdd", FakeCommitOperationAdd)
    setattr(module, "hf_hub_download", fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    client = publication._HuggingFaceHubClient()
    client.create_repo(repo_id="owner/hebrew-data", repo_type="dataset")

    assert calls == {
        "repo_id": "owner/hebrew-data",
        "repo_type": "dataset",
        "private": False,
        "exist_ok": False,
    }


def test_production_commit_is_bound_to_observed_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeInfo:
        sha = "b" * 40
        oid = "a" * 40

    class FakeApi:
        def repo_info(self, **kwargs: object) -> FakeInfo:
            calls["repo_info"] = kwargs
            return FakeInfo()

        def create_commit(self, **kwargs: object) -> FakeInfo:
            calls["create_commit"] = kwargs
            return FakeInfo()

    class FakeCommitOperationAdd:
        def __init__(self, **kwargs: object) -> None:
            calls["operation"] = kwargs

    def fake_download(**kwargs: object) -> str:
        del kwargs
        return str(tmp_path / "download")

    module = ModuleType("huggingface_hub")
    setattr(module, "HfApi", FakeApi)
    setattr(module, "CommitOperationAdd", FakeCommitOperationAdd)
    setattr(module, "hf_hub_download", fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    source = tmp_path / "README.md"
    source.write_text("release\n", encoding="utf-8")

    client = publication._HuggingFaceHubClient()
    parent = client.resolve_revision(repo_id="owner/repo", repo_type="model")
    revision = client.create_commit(
        repo_id="owner/repo",
        repo_type="model",
        files={"README.md": source},
        commit_message="Publish",
        parent_commit=parent,
    )

    assert parent == "b" * 40
    assert revision == "a" * 40
    commit_call = cast("dict[str, object]", calls["create_commit"])
    assert commit_call["parent_commit"] == "b" * 40


def test_production_download_materializes_hub_cache_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = tmp_path / "cache" / "blobs" / "sha"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"verified bytes")
    cache_pointer = tmp_path / "cache" / "snapshots" / ("a" * 40) / "README.md"
    cache_pointer.parent.mkdir(parents=True)
    cache_pointer.symlink_to(blob)

    class FakeApi:
        pass

    class FakeCommitOperationAdd:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    def fake_download(**kwargs: object) -> str:
        del kwargs
        return str(cache_pointer)

    module = ModuleType("huggingface_hub")
    setattr(module, "HfApi", FakeApi)
    setattr(module, "CommitOperationAdd", FakeCommitOperationAdd)
    setattr(module, "hf_hub_download", fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    client = publication._HuggingFaceHubClient()
    materialized = client.download_file(
        repo_id="owner/repo",
        repo_type="model",
        filename="README.md",
        revision="a" * 40,
    )

    assert materialized.read_bytes() == b"verified bytes"
    assert materialized.is_file()
    assert not materialized.is_symlink()


def test_public_api_exposes_only_fully_validated_publication_boundaries() -> None:
    assert publication.__all__ == [
        "publish_hebrew_adapter_bundle",
        "publish_hebrew_dataset_bundle",
    ]
    assert not hasattr(publication, "publish_prepared_bundle")


def test_unexpected_existing_remote_file_blocks_upload(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path, existing_files={"unrelated.bin"})
    with pytest.raises(UserInputError, match="outside the publication allowlist"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )
    assert "commit" not in client.events


def test_noncanonical_remote_path_fails_closed(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path, existing_files={"../escape"})
    with pytest.raises(ExternalDependencyError, match="non-canonical"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )


def test_nonimmutable_commit_identity_is_journaled_but_not_verified(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path)
    client.revision = "main"
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ExternalDependencyError, match="non-immutable"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )
    journal = json.loads(receipt.read_text(encoding="utf-8"))
    assert journal["status"] == "commit_returned_unverified"
    assert journal["repository"]["commit_sha"] == "main"


@pytest.mark.parametrize("extra", [False, True])
def test_round_trip_file_tree_must_match_allowlist(tmp_path: Path, extra: bool) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)
    files = set(prepared.files)
    if extra:
        files.add("surprise.txt")
    else:
        files.remove(next(iter(files)))
    client.post_commit_files = files

    with pytest.raises(ExternalDependencyError, match="filename verification failed"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )


def test_round_trip_sha_mismatch_retains_unverified_commit_journal(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)
    client.corrupt_file = "README.md"
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ExternalDependencyError, match="SHA256 mismatch"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )
    journal = json.loads(receipt.read_text(encoding="utf-8"))
    assert journal["status"] == "commit_returned_unverified"
    assert journal["repository"]["commit_sha"] == "a" * 40


def test_round_trip_symlink_is_rejected(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)
    client.symlink_download = "README.md"
    with pytest.raises(ExternalDependencyError, match="regular file"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )


def test_existing_receipt_fails_before_remote_mutation_and_is_unchanged(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("original\n", encoding="utf-8")
    client = FakeHubClient(tmp_path)
    with pytest.raises(UserInputError, match="receipt already exists"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )
    assert receipt.read_text(encoding="utf-8") == "original\n"
    assert client.events == []


def test_broken_symlink_receipt_target_fails_before_remote_mutation(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.symlink_to(tmp_path / "missing")
    client = FakeHubClient(tmp_path)
    with pytest.raises(UserInputError, match="receipt already exists"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )
    assert receipt.is_symlink()
    assert client.events == []


def test_unwritable_receipt_destination_fails_before_remote_mutation(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied\n", encoding="utf-8")
    client = FakeHubClient(tmp_path)

    with pytest.raises(UserInputError, match="could not reserve publication receipt"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=blocked_parent / "receipt.json",
            client=client,
        )

    assert client.events == []


def test_commit_failure_leaves_durable_submission_journal(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    client = FakeHubClient(tmp_path)
    client.fail_commit = True

    with pytest.raises(ExternalDependencyError, match="commit failed"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )

    journal = json.loads(receipt.read_text(encoding="utf-8"))
    assert journal["status"] == "commit_submitting"
    assert journal["repository"]["commit_sha"] is None
    assert journal["repository"]["parent_commit"] == "b" * 40
    assert receipt.stat().st_mode & 0o777 == 0o600


def test_receipt_content_continuity_rejects_same_inode_tampering_without_leak(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    reservation = publication._reserve_receipt(
        receipt,
        {"status": "pending"},
        source_roots=(),
    )
    secret = "hf_" + "z" * 30
    replacement = f'{{"replacement":"{secret}"}}\n'.encode()
    replacement += b" " * (reservation.capacity - len(replacement))
    try:
        receipt.write_bytes(replacement)

        with pytest.raises(
            ExternalDependencyError,
            match="durably update publication receipt",
        ) as captured:
            publication._write_receipt(
                reservation,
                {"status": "commit_submitting"},
            )

        assert secret not in str(captured.value)
        assert receipt.read_bytes() == replacement
    finally:
        publication._close_receipt_reservation(reservation)


def test_receipt_content_identity_advances_across_every_durable_stage(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    reservation = publication._reserve_receipt(
        receipt,
        {"status": "pending"},
        source_roots=(),
    )
    observed_hashes = [reservation.expected_sha256]
    try:
        for status in ("commit_submitting", "commit_returned_unverified", "verified"):
            reservation = publication._write_receipt(reservation, {"status": status})
            assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == status
            assert reservation.expected_sha256 == sha256_file(receipt)
            observed_hashes.append(reservation.expected_sha256)
    finally:
        publication._close_receipt_reservation(reservation)

    assert len(set(observed_hashes)) == len(observed_hashes)


def test_receipt_update_verifies_written_content_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    reservation = publication._reserve_receipt(
        receipt,
        {"status": "pending"},
        source_roots=(),
    )
    real_write_all = publication._write_all

    def corrupt_after_write(descriptor: int, data: bytes) -> None:
        real_write_all(descriptor, data)
        os.lseek(descriptor, 0, os.SEEK_SET)
        assert os.write(descriptor, b"x") == 1

    monkeypatch.setattr(publication, "_write_all", corrupt_after_write)
    try:
        with pytest.raises(
            ExternalDependencyError,
            match="durably update publication receipt",
        ):
            publication._write_receipt(
                reservation,
                {"status": "commit_submitting"},
            )
    finally:
        publication._close_receipt_reservation(reservation)


@pytest.mark.parametrize("outcome", ("success", "inspect-error", "commit-error"))
def test_receipt_handle_closes_on_every_publication_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    captured: dict[str, int] = {}
    real_reserve = publication._reserve_receipt
    real_close = os.close
    closed: list[int] = []

    def tracking_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    def capture_reservation(
        path: Path,
        payload: Mapping[str, object],
        *,
        source_roots: Sequence[Path],
    ) -> publication._ReceiptReservation:
        reservation = real_reserve(path, payload, source_roots=source_roots)
        captured["descriptor"] = reservation.descriptor
        captured["prior_closes"] = closed.count(reservation.descriptor)
        return reservation

    monkeypatch.setattr(os, "close", tracking_close)
    monkeypatch.setattr(publication, "_reserve_receipt", capture_reservation)

    class InspectFailureClient(FakeHubClient):
        def resolve_revision(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
        ) -> str | None:
            super().resolve_revision(repo_id=repo_id, repo_type=repo_type)
            raise RuntimeError("inspection failed")

    if outcome == "inspect-error":
        client: FakeHubClient = InspectFailureClient(tmp_path)
    else:
        client = FakeHubClient(tmp_path)
        client.fail_commit = outcome == "commit-error"

    if outcome == "success":
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )
    else:
        with pytest.raises(ExternalDependencyError):
            publish_prepared_bundle(
                _prepared_dataset(tmp_path),
                repo_id="owner/hebrew-data",
                commit_message="Publish",
                execute=True,
                confirmed_repo_id="owner/hebrew-data",
                receipt_path=tmp_path / "receipt.json",
                client=client,
            )

    descriptor = captured["descriptor"]
    assert closed.count(descriptor) == captured["prior_closes"] + 1
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_receipt_inode_replacement_is_detected_before_commit(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"

    class ReplacingClient(FakeHubClient):
        def resolve_revision(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
        ) -> str | None:
            revision = super().resolve_revision(repo_id=repo_id, repo_type=repo_type)
            receipt.unlink()
            receipt.write_text("replacement\n", encoding="utf-8")
            return revision

    client = ReplacingClient(tmp_path)
    with pytest.raises(ExternalDependencyError, match="durably update publication receipt"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )

    assert receipt.read_text(encoding="utf-8") == "replacement\n"
    assert "commit" not in client.events


@pytest.mark.skipif(os.name != "posix", reason="POSIX dirfd receipt hardening")
def test_receipt_parent_replacement_is_detected_before_commit(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    source_root = prepared.files["README.md"].parent
    receipt_parent = tmp_path / "receipts"
    receipt = receipt_parent / "attempt.json"
    moved_parent = tmp_path / "receipts-moved"

    class ReplacingParentClient(FakeHubClient):
        def resolve_revision(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
        ) -> str | None:
            revision = super().resolve_revision(repo_id=repo_id, repo_type=repo_type)
            receipt_parent.rename(moved_parent)
            receipt_parent.symlink_to(source_root, target_is_directory=True)
            return revision

    client = ReplacingParentClient(tmp_path)
    with pytest.raises(ExternalDependencyError, match="update reserved publication receipt"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )

    assert (moved_parent / "attempt.json").is_file()
    assert not (source_root / "attempt.json").exists()
    assert "commit" not in client.events


def test_receipt_cannot_be_an_upload_source(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    with pytest.raises(UserInputError, match="outside the source bundle"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=prepared.files["README.md"],
            client=FakeHubClient(tmp_path),
        )


def test_receipt_cannot_be_nested_anywhere_in_source_bundle(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)
    source_root = prepared.files["README.md"].parent

    with pytest.raises(UserInputError, match="outside the source bundle"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=source_root / "receipts" / "attempt.json",
            client=client,
        )

    assert client.events == []


def test_receipt_cannot_use_case_alias_of_source_bundle(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path / "Bundle")
    source_root = prepared.files["README.md"].parent
    case_alias = tmp_path / "bundle" / "dataset"
    try:
        aliases_source = case_alias.exists() and os.path.samefile(case_alias, source_root)
    except OSError:
        aliases_source = False
    if not aliases_source:
        pytest.skip("filesystem is case-sensitive")
    client = FakeHubClient(tmp_path)

    with pytest.raises(UserInputError, match="outside the source bundle"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=case_alias / "receipts" / "attempt.json",
            client=client,
        )

    assert client.events == []


def test_receipt_parent_swap_into_source_is_rejected_before_hub_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_dataset(tmp_path)
    source_root = prepared.files["README.md"].parent
    outside = tmp_path / "outside-receipts"
    outside.mkdir()
    alias = tmp_path / "receipt-parent"
    alias.symlink_to(outside, target_is_directory=True)
    receipt = alias / "attempt.json"
    real_reserve = publication._reserve_receipt

    def swap_then_reserve(
        path: Path,
        payload: Mapping[str, object],
        *,
        source_roots: Sequence[Path],
    ) -> object:
        alias.unlink()
        alias.symlink_to(source_root, target_is_directory=True)
        return real_reserve(path, payload, source_roots=source_roots)

    monkeypatch.setattr(publication, "_reserve_receipt", swap_then_reserve)
    client = FakeHubClient(tmp_path)

    with pytest.raises(UserInputError, match="could not reserve publication receipt"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )

    assert not (source_root / "attempt.json").exists()
    assert client.events == []


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_constructed_dataset_preparation_cannot_bypass_upload_allowlist(
    tmp_path: Path,
    mutation: str,
) -> None:
    prepared = _prepared_dataset(tmp_path)
    files = dict(prepared.files)
    if mutation == "missing":
        files.pop("README.md")
    else:
        unexpected = tmp_path / "unexpected.txt"
        unexpected.write_text("no\n", encoding="utf-8")
        files["unexpected.txt"] = unexpected
    with pytest.raises(UserInputError, match="exact upload allowlist"):
        publish_prepared_bundle(
            PreparedPublication(repo_type="dataset", files=files),
            repo_id="owner/hebrew-data",
            commit_message="Validate",
        )


def test_constructed_preparation_rejects_symlink_source(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    target = prepared.files["README.md"]
    link = tmp_path / "README-link.md"
    link.symlink_to(target)
    files = dict(prepared.files)
    files["README.md"] = link
    with pytest.raises(UserInputError, match="not a regular file"):
        publish_prepared_bundle(
            PreparedPublication(repo_type="dataset", files=files),
            repo_id="owner/hebrew-data",
            commit_message="Validate",
        )


def test_model_upload_allowlist_accepts_only_explicit_peft_and_evidence_files(
    tmp_path: Path,
) -> None:
    plan = publish_prepared_bundle(
        _prepared_adapter(tmp_path, optional=True),
        repo_id="owner/Llama-hebrew-adapter",
        commit_message="Validate adapter",
    )
    paths = {cast("str", item["path"]) for item in cast("list[dict[str, object]]", plan["files"])}
    assert paths == ADAPTER_UPLOAD_REQUIRED_FILES | ADAPTER_UPLOAD_OPTIONAL_FILES


@pytest.mark.parametrize("repo_id", ["owner/hebrew-adapter", "owner/llama-hebrew-adapter"])
def test_llama_derived_model_repo_name_is_enforced(tmp_path: Path, repo_id: str) -> None:
    with pytest.raises(UserInputError, match="must begin with 'Llama'"):
        publish_prepared_bundle(
            _prepared_adapter(tmp_path),
            repo_id=repo_id,
            commit_message="Validate adapter",
        )


def test_dataset_preparation_validates_full_paired_contract_and_exact_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _dataset_bundle(tmp_path / "bundle")
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text('{"source": "en"}\n', encoding="utf-8")
    observed: dict[str, str] = {}

    def fake_validate(
        config: SommelierConfig,
        staged_root: Path,
    ) -> dict[str, dict[str, Path]]:
        assert config.dataset_for("he").dataset_revision == "0" * 40
        observed["root"] = staged_root.read_text(encoding="utf-8")
        observed["paired"] = paired_input_path(staged_root, "he").read_text(encoding="utf-8")
        for filename in (
            SUMMARY_FILENAME,
            PUBLICATION_MANIFEST_FILENAME,
            SEMANTIC_REVIEW_FILENAME,
            SEMANTIC_REVIEW_TEMPLATE_FILENAME,
        ):
            assert translation_provenance_sidecar_path(staged_root, filename, "he").is_file()
        return {"he": {"rows": paired_input_path(staged_root, "he")}}

    monkeypatch.setattr(publication, "validate_full_paired_input_contract", fake_validate)
    prepared = prepare_hebrew_dataset_publication(
        config_path=Path("examples/config.v3-he-full.yaml"),
        bundle_dir=bundle,
        root_rows_path=root_rows,
    )

    assert set(prepared.files) == DATASET_REQUIRED_FILES
    assert prepared.expected_repo_id == HEBREW_V3_PAIRED_DATASET_ID
    assert observed["root"] == '{"source": "en"}\n'
    assert observed["paired"] == "{}\n"


def test_prepared_repository_identity_cannot_be_redirected(tmp_path: Path) -> None:
    prepared = PreparedPublication(
        repo_type="dataset",
        files=_write_files(tmp_path / "dataset", sorted(DATASET_REQUIRED_FILES)),
        expected_repo_id=HEBREW_V3_PAIRED_DATASET_ID,
    )
    client = FakeHubClient(tmp_path)

    with pytest.raises(UserInputError, match="does not match the prepared artifact identity"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/lookalike-hebrew-dataset",
            commit_message="Validate",
            client=client,
        )

    assert client.events == []


def test_dataset_card_requires_cc_by_attribution_and_translation_disclosure(
    tmp_path: Path,
) -> None:
    card = tmp_path / "README.md"
    card.write_text("---\nlicense: mit\n---\nHebrew data\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="cc-by-4.0"):
        publication._validate_dataset_card(card)


def test_dataset_card_does_not_accept_license_prose_outside_frontmatter(tmp_path: Path) -> None:
    card = tmp_path / "README.md"
    card.write_text(
        "---\nlicense: mit\n---\n"
        "license: cc-by-4.0\nSalesforce/xlam-function-calling-60k machine-translated Hebrew\n",
        encoding="utf-8",
    )
    with pytest.raises(UserInputError, match="frontmatter license"):
        publication._validate_dataset_card(card)


def test_dataset_card_rejects_unresolved_verified_bundle_marker(tmp_path: Path) -> None:
    card = tmp_path / "README.md"
    card.write_text(
        "---\nlicense: cc-by-4.0\n---\n"
        "Salesforce/xlam-function-calling-60k machine-translated Hebrew\n"
        "REPLACE_FROM_VERIFIED_DATASET_BUNDLE\n",
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="unresolved release template markers"):
        publication._validate_dataset_card(card)


def test_dataset_card_rejects_duplicate_yaml_frontmatter_keys(tmp_path: Path) -> None:
    card = tmp_path / "README.md"
    card.write_text(
        "---\nlicense: mit\nlicense: cc-by-4.0\n---\n"
        "Salesforce/xlam-function-calling-60k machine-translated Hebrew\n",
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="invalid YAML frontmatter"):
        publication._validate_dataset_card(card)


def test_publication_json_objects_reject_duplicate_keys_without_exposing_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    path.write_text(
        f'{{"status":"{secret}","status":"safe"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="missing or invalid JSON") as captured:
        publication._load_json_object(path, context="run manifest")

    assert secret not in str(captured.value)


def test_dataset_bundle_rejects_unexpected_files_before_contract_validation(
    tmp_path: Path,
) -> None:
    bundle = _dataset_bundle(tmp_path / "bundle")
    (bundle / "notes.txt").write_text("not curated\n", encoding="utf-8")
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text("{}\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="unexpected files: notes.txt"):
        prepare_hebrew_dataset_publication(
            config_path=Path("examples/config.v3-he-full.yaml"),
            bundle_dir=bundle,
            root_rows_path=root_rows,
        )


def test_dataset_bundle_rejects_secrets(tmp_path: Path) -> None:
    bundle = _dataset_bundle(tmp_path / "bundle")
    (bundle / SUMMARY_FILENAME).write_text(
        f'{{"token": "hf_{"a" * 30}"}}\n',
        encoding="utf-8",
    )
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SecurityPolicyError, match="secret-like"):
        prepare_hebrew_dataset_publication(
            config_path=Path("examples/config.v3-he-full.yaml"),
            bundle_dir=bundle,
            root_rows_path=root_rows,
        )


def test_dataset_bundle_rejects_duplicate_json_keys_before_contract_validation(
    tmp_path: Path,
) -> None:
    bundle = _dataset_bundle(tmp_path / "bundle")
    (bundle / SUMMARY_FILENAME).write_text(
        '{"status":"pending","status":"accepted"}\n',
        encoding="utf-8",
    )
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SecurityPolicyError, match="duplicate_key"):
        prepare_hebrew_dataset_publication(
            config_path=Path("examples/config.v3-he-full.yaml"),
            bundle_dir=bundle,
            root_rows_path=root_rows,
        )


def test_raw_provider_journal_is_never_publishable(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / OPENAI_PROVIDER_JOURNAL_FILENAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(SecurityPolicyError, match="raw OpenAI provider journal"):
        publication._assert_no_raw_provider_journal(bundle)


def test_adapter_preparation_maps_peft_root_and_namespaces_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path = _immutable_hebrew_config(tmp_path)
    bundle = _adapter_bundle(tmp_path / "bundle", config_path)

    def fake_manifests(
        *,
        bundle_dir: Path,
        config: SommelierConfig,
        adapter_files: Sequence[str],
    ) -> tuple[str, str, dict[str, Any]]:
        del bundle_dir, config
        assert "adapter/tokenizer.json" in adapter_files
        return (
            "run-1",
            "f" * 40,
            {
                "config_sha256": "0" * 64,
                "dependency_lock_sha256": "d" * 64,
            },
        )

    def fake_experiment(**kwargs: object) -> dict[str, Any]:
        del kwargs
        return {}

    monkeypatch.setattr(publication, "_validate_adapter_config", lambda *_: None)
    monkeypatch.setattr(publication, "_validate_safetensors", lambda *_: None)
    monkeypatch.setattr(publication, "_validate_adapter_manifests", fake_manifests)
    monkeypatch.setattr(publication, "_validate_experiment_identity", fake_experiment)
    monkeypatch.setattr(publication, "_validate_release_evidence", lambda *_, **__: None)
    monkeypatch.setattr(publication, "_validate_adapter_card", lambda *_, **__: None)

    prepared = prepare_hebrew_adapter_publication(bundle_dir=bundle)

    assert prepared.repo_type == "model"
    assert set(prepared.files) == ADAPTER_UPLOAD_REQUIRED_FILES | {"tokenizer.json"}
    assert prepared.files["adapter_model.safetensors"] == (
        bundle / "adapter" / "adapter_model.safetensors"
    )
    assert prepared.files["sommelier/config.resolved.yaml"] == bundle / "config.resolved.yaml"
    assert config.model.base_model_id in config_path.read_text(encoding="utf-8")


def test_adapter_license_files_must_match_reviewed_project_copies(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for filename in ADAPTER_LICENSE_FILE_SHA256:
        shutil.copy2(Path("licenses") / filename, bundle / filename)
    publication._validate_adapter_license_files(bundle)

    (bundle / "LICENSE-LLAMA-3.1.txt").write_text("abbreviated terms\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="does not match the reviewed project copy"):
        publication._validate_adapter_license_files(bundle)


def test_experiment_identity_requires_clean_finalizer_source(tmp_path: Path) -> None:
    report = tmp_path / "experiment_report.json"
    source_revision = "f" * 40
    tree_sha256 = "2" * 64
    finalizer: dict[str, object] = {
        "git_commit": source_revision,
        "working_tree_clean": False,
    }
    payload = {
        "schema_version": EXPERIMENT_REPORT_SCHEMA,
        "arms": {
            "v3_en_he": {
                "run_id": "run-1",
                "config_sha256": "0" * 64,
                "adapter_source": {
                    "source": "runs/run-1/train/adapter",
                    "revision": None,
                    "kind": "local_directory",
                    "tree_sha256": tree_sha256,
                    "artifact_path": "runs/run-1/train/adapter",
                    "revision_is_immutable": True,
                },
            }
        },
        "preregistration": {"finalizer_source_code": finalizer},
        "data_provenance": {"contract": {"source_code_revision": source_revision}},
        "all_claims_passed": False,
        "approved_claims": [],
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match="finalizer source"):
        publication._validate_experiment_identity(
            path=report,
            run_id="run-1",
            source_revision=source_revision,
            config_sha256="0" * 64,
            tree_sha256=tree_sha256,
        )

    finalizer["working_tree_clean"] = True
    report.write_text(json.dumps(payload), encoding="utf-8")
    publication._validate_experiment_identity(
        path=report,
        run_id="run-1",
        source_revision=source_revision,
        config_sha256="0" * 64,
        tree_sha256=tree_sha256,
    )


def test_adapter_card_requires_llama_and_nvidia_obligations(tmp_path: Path) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    card = tmp_path / "README.md"
    experiment_sha256 = "1" * 64
    tree_sha256 = "2" * 64
    card.write_text(
        _valid_adapter_card(
            config,
            experiment_sha256=experiment_sha256,
            tree_sha256=tree_sha256,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
        ),
        encoding="utf-8",
    )
    publication._validate_adapter_card(
        card,
        config=config,
        experiment_sha256=experiment_sha256,
        tree_sha256=tree_sha256,
        source_revision="f" * 40,
        dataset_revision=config.dataset_for("he").dataset_revision,
    )

    card.write_text(
        card.read_text(encoding="utf-8").replace("NVIDIA Open Model License", ""),
        encoding="utf-8",
    )
    with pytest.raises(UserInputError, match="NVIDIA Open Model License"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256=experiment_sha256,
            tree_sha256=tree_sha256,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
        )


def test_adapter_card_rejects_wrong_frontmatter_and_unresolved_markers(tmp_path: Path) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    card = tmp_path / "README.md"
    text = _valid_adapter_card(
        config,
        experiment_sha256="1" * 64,
        tree_sha256="2" * 64,
        source_revision="f" * 40,
        dataset_revision=config.dataset_for("he").dataset_revision,
    )
    card.write_text(text.replace("license: llama3.1", "license: mit"), encoding="utf-8")
    with pytest.raises(UserInputError, match="frontmatter license"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
        )

    card.write_text(text + "\nREPLACE_FROM_VERIFIED_BUNDLE\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="unresolved release template markers"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
        )


@pytest.mark.parametrize(
    "replacement",
    (
        "# Hebrew tool-calling adapter",
        "## Llama Hebrew tool-calling adapter",
        "# Hebrew adapter\n# Llama appears too late",
    ),
)
def test_adapter_card_first_markdown_h1_must_begin_with_llama(
    tmp_path: Path,
    replacement: str,
) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    card = tmp_path / "README.md"
    text = _valid_adapter_card(
        config,
        experiment_sha256="1" * 64,
        tree_sha256="2" * 64,
        source_revision="f" * 40,
        dataset_revision=config.dataset_for("he").dataset_revision,
    )
    card.write_text(
        text.replace("# Llama Hebrew tool-calling adapter", replacement),
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="first Markdown H1 must begin with 'Llama'"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
        )


def test_release_evidence_is_identity_bound_and_requires_notices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source_revision = "f" * 40
    dependency_lock_sha256 = sha256_file(Path("uv.lock"))
    source_identity = release.SourceCodeIdentity(
        discovery="git-project-root-v1",
        git_commit=source_revision,
        working_tree_clean=True,
        git_status_sha256=release._EMPTY_SHA256,
    )
    lock_identity = release.DependencyLockIdentity(
        path="uv.lock",
        sha256=dependency_lock_sha256,
        bytes=Path("uv.lock").stat().st_size,
    )
    monkeypatch.setattr(release, "_discover_project_source", lambda _: source_identity)
    monkeypatch.setattr(
        release,
        "_project_file_matches_revision",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        release,
        "_dependency_lock_identity",
        lambda *_args, **_kwargs: lock_identity,
    )

    (bundle / "release_preflight.json").write_text(
        json.dumps(
            {
                "schema_version": release.PREFLIGHT_SCHEMA,
                "status": "pass",
                "gates": [],
            }
        ),
        encoding="utf-8",
    )
    notices = "\n".join(
        (
            config.model.base_model_id,
            "NVIDIA Open Model License",
            "Llama 3.1 Community License",
            REQUIRED_DERIVED_NOTICE,
            config.root_dataset.dataset_id,
            "CC-BY-4.0",
        )
    )
    (bundle / "THIRD_PARTY.md").write_text(notices, encoding="utf-8")
    with pytest.raises(UserInputError, match="identity is invalid"):
        publication._validate_release_evidence(
            bundle,
            config,
            source_revision=source_revision,
            dependency_lock_sha256=dependency_lock_sha256,
        )

    run_release_preflight(
        config,
        project_root=Path.cwd(),
        artifact_root=bundle,
        environ={ACK_ENV_NAME: config.model.base_model_id},
    )
    publication._validate_release_evidence(
        bundle,
        config,
        source_revision=source_revision,
        dependency_lock_sha256=dependency_lock_sha256,
    )

    (bundle / "THIRD_PARTY.md").write_text(
        notices.replace("NVIDIA Open Model License", ""),
        encoding="utf-8",
    )
    run_release_preflight(
        config,
        project_root=Path.cwd(),
        artifact_root=bundle,
        environ={ACK_ENV_NAME: config.model.base_model_id},
    )
    with pytest.raises(UserInputError, match="NVIDIA Open Model License"):
        publication._validate_release_evidence(
            bundle,
            config,
            source_revision=source_revision,
            dependency_lock_sha256=dependency_lock_sha256,
        )


def _write_safetensors(path: Path, tensors: dict[str, object]) -> None:
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    tensor_metadata = [
        cast("dict[str, object]", metadata)
        for name, metadata in tensors.items()
        if name != "__metadata__"
    ]
    payload_size = max(
        cast("list[int]", metadata["data_offsets"])[1] for metadata in tensor_metadata
    )
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(payload_size))


def _write_raw_safetensors_header(path: Path, header: str, *, payload_bytes: int) -> None:
    encoded = header.encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(payload_bytes))


def test_safetensors_gate_accepts_only_complete_lora_pairs(tmp_path: Path) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    tensors: dict[str, object] = {
        "__metadata__": {"format": "pt"},
        f"{prefix}A.default.weight": {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [0, 2],
        },
        f"{prefix}B.default.weight": {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [2, 4],
        },
    }
    _write_safetensors(path, tensors)
    publication._validate_safetensors(path)

    tensors.pop(f"{prefix}B.default.weight")
    _write_safetensors(path, tensors)
    with pytest.raises(UserInputError, match="incomplete LoRA"):
        publication._validate_safetensors(path)


def test_safetensors_gate_rejects_base_model_tensor(tmp_path: Path) -> None:
    path = tmp_path / "adapter_model.safetensors"
    _write_safetensors(
        path,
        {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        },
    )
    with pytest.raises(UserInputError, match="non-LoRA tensor"):
        publication._validate_safetensors(path)


def test_safetensors_gate_scans_metadata_for_secrets_without_exposing_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    _write_safetensors(
        path,
        {
            "__metadata__": {"training_note": secret},
            f"{prefix}A.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            },
            f"{prefix}B.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [2, 4],
            },
        },
    )

    with pytest.raises(SecurityPolicyError, match="__metadata__ contains") as captured:
        publication._validate_safetensors(path)

    assert secret not in str(captured.value)


@pytest.mark.parametrize("duplicate_location", ("metadata", "tensor"))
def test_safetensors_gate_rejects_duplicate_json_keys_without_exposing_values(
    tmp_path: Path,
    duplicate_location: str,
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    metadata = (
        f'"__metadata__":{{"note":"{secret}"}},"__metadata__":{{"format":"pt"}},'
        if duplicate_location == "metadata"
        else '"__metadata__":{"format":"pt"},'
    )
    tensor_a = (
        f'"{prefix}A.default.weight":'
        '{"dtype":"F16","dtype":"F32","shape":[1],"data_offsets":[0,2]},'
        if duplicate_location == "tensor"
        else f'"{prefix}A.default.weight":{{"dtype":"F16","shape":[1],"data_offsets":[0,2]}},'
    )
    header = (
        "{" + metadata + tensor_a + f'"{prefix}B.default.weight":'
        '{"dtype":"F16","shape":[1],"data_offsets":[2,4]}}'
    )
    _write_raw_safetensors_header(path, header, payload_bytes=4)

    with pytest.raises(UserInputError, match="invalid JSON header") as captured:
        publication._validate_safetensors(path)

    assert secret not in str(captured.value)


@pytest.mark.parametrize("mutation", ("leading-space", "trailing-tab"))
def test_safetensors_gate_rejects_noncanonical_header_padding(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    header = json.dumps(
        {
            f"{prefix}A.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            },
            f"{prefix}B.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [2, 4],
            },
        },
        separators=(",", ":"),
    )
    malformed = f" {header}" if mutation == "leading-space" else f"{header}\t"
    _write_raw_safetensors_header(path, malformed, payload_bytes=4)

    with pytest.raises(UserInputError, match="header envelope"):
        publication._validate_safetensors(path)


@pytest.mark.parametrize(
    ("dtype", "shape", "offsets", "error"),
    (
        ("NOT_A_DTYPE", [1], ([0, 2], [2, 4]), "incomplete"),
        ("F16", [-1], ([0, 2], [2, 4]), "incomplete"),
        ("F16", [True], ([0, 2], [2, 4]), "incomplete"),
        ("F16", [2], ([0, 2], [2, 4]), "does not match its offsets"),
        ("F4", [1], ([0, 0], [0, 0]), "does not match its offsets"),
    ),
)
def test_safetensors_gate_rejects_invalid_dtype_shape_and_byte_span(
    tmp_path: Path,
    dtype: str,
    shape: list[object],
    offsets: tuple[list[int], list[int]],
    error: str,
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    _write_safetensors(
        path,
        {
            f"{prefix}A.default.weight": {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": offsets[0],
            },
            f"{prefix}B.default.weight": {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": offsets[1],
            },
        },
    )

    with pytest.raises(UserInputError, match=error):
        publication._validate_safetensors(path)


@pytest.mark.parametrize(
    ("shape", "offsets"),
    (
        ([0, 4], ([0, 0], [0, 0])),
        ([], ([0, 2], [2, 4])),
    ),
)
def test_safetensors_gate_preserves_empty_and_scalar_tensors(
    tmp_path: Path,
    shape: list[int],
    offsets: tuple[list[int], list[int]],
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    _write_safetensors(
        path,
        {
            f"{prefix}A.default.weight": {
                "dtype": "F16",
                "shape": shape,
                "data_offsets": offsets[0],
            },
            f"{prefix}B.default.weight": {
                "dtype": "F16",
                "shape": shape,
                "data_offsets": offsets[1],
            },
        },
    )

    publication._validate_safetensors(path)


def test_safetensors_gate_rejects_unrecognized_tensor_metadata_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    _write_safetensors(
        path,
        {
            f"{prefix}A.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
                "note": secret,
            },
            f"{prefix}B.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [2, 4],
            },
        },
    )

    with pytest.raises(UserInputError, match="unexpected fields") as captured:
        publication._validate_safetensors(path)

    assert secret not in str(captured.value)


def test_adapter_config_must_match_bound_base_and_qlora_contract(tmp_path: Path) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    path = tmp_path / "adapter_config.json"
    payload: dict[str, object] = {
        "base_model_name_or_path": config.model.base_model_id,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": config.train.lora_rank,
        "lora_alpha": config.train.lora_alpha,
        "lora_dropout": config.train.lora_dropout,
        "bias": "none",
        "target_modules": list(config.train.target_modules),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    publication._validate_adapter_config(path, config)

    payload["base_model_name_or_path"] = "other/model"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UserInputError, match="base_model_name_or_path"):
        publication._validate_adapter_config(path, config)


def test_prepared_hashes_are_deterministic_and_bound_to_bytes(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    assert prepared.sha256 == {
        name: sha256_file(path) for name, path in sorted(prepared.files.items())
    }
    readme = prepared.files["README.md"]
    before = prepared.sha256["README.md"]
    readme.write_text("changed\n", encoding="utf-8")
    assert prepared.sha256["README.md"] != before
