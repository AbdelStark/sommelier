from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, cast

import yaml

from sommelier.artifacts import sha256_file
from sommelier.config import SommelierConfig, compute_config_digest, load_config
from sommelier.data.openai_evidence import OPENAI_PROVIDER_JOURNAL_FILENAME
from sommelier.data.prepare import paired_input_path
from sommelier.data.semantic_review import (
    SEMANTIC_REVIEW_FILENAME,
    SEMANTIC_REVIEW_TEMPLATE_FILENAME,
)
from sommelier.data.translate import (
    PUBLICATION_MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    rows_filename,
    translation_provenance_sidecar_path,
    validate_full_paired_input_contract,
)
from sommelier.errors import (
    ExternalDependencyError,
    InvariantViolation,
    SecurityPolicyError,
    UserInputError,
)
from sommelier.evaluation.data_provenance import (
    HEBREW_V3_PAIRED_DATASET_ID,
    validate_hebrew_v3_preregistered_config,
)
from sommelier.evaluation.experiment import EXPERIMENT_REPORT_SCHEMA
from sommelier.evaluation.generate import IMMUTABLE_HF_REVISION, adapter_tree_sha256
from sommelier.redaction import (
    DuplicateJsonKeyError,
    assert_artifacts_publishable,
    loads_unique_json,
    reject_duplicate_json_keys,
    scan_artifact_file,
    scan_json_payload,
)
from sommelier.release import REQUIRED_DERIVED_NOTICE, validate_release_preflight_report

PublicationRepoType = Literal["dataset", "model"]

PUBLICATION_RECEIPT_SCHEMA: Final = "sommelier.huggingface_publication_receipt.v1"
PLATFORM_MANAGED_FILES: Final = frozenset({".gitattributes"})
ADAPTER_LICENSE_FILE_SHA256: Final = {
    "LICENSE-NVIDIA-OPEN-MODEL.txt": (
        "2fcf3939d6a883791efa216a18970af88394658c2bb86b6c3e3d20d8d8801f41"
    ),
    "LICENSE-LLAMA-3.1.txt": "6d6cc0314fdc084f456328422b3a69dd1a6367029f7e44c07bc3eb97cfb79f94",
    "NOTICE": "6b245d4aa77226052e851e73e6ff4cc4cb89f02353f5d0bec11ce7ca314b7a3a",
}
_UNRESOLVED_CARD_MARKER: Final = "REPLACE_FROM_VERIFIED_BUNDLE"
_UNRESOLVED_DATASET_CARD_MARKER: Final = "REPLACE_FROM_VERIFIED_DATASET_BUNDLE"

DATASET_REQUIRED_FILES: Final = frozenset(
    {
        "README.md",
        rows_filename("he"),
        SUMMARY_FILENAME,
        PUBLICATION_MANIFEST_FILENAME,
        SEMANTIC_REVIEW_FILENAME,
        SEMANTIC_REVIEW_TEMPLATE_FILENAME,
    }
)

ADAPTER_REQUIRED_FILES: Final = frozenset(
    {
        "README.md",
        "THIRD_PARTY.md",
        *ADAPTER_LICENSE_FILE_SHA256,
        "adapter/README.md",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "config.resolved.yaml",
        "experiment_report.json",
        "manifest.json",
        "release_preflight.json",
        "train_manifest.json",
    }
)

# Tokenizer files are emitted by the current trainer beside the PEFT adapter.
# They are not required to load a LoRA, but each name is explicit so a future
# trainer cannot silently add an arbitrary file to a release.
ADAPTER_OPTIONAL_FILES: Final = frozenset(
    {
        "adapter/added_tokens.json",
        "adapter/chat_template.jinja",
        "adapter/special_tokens_map.json",
        "adapter/tokenizer.json",
        "adapter/tokenizer.model",
        "adapter/tokenizer_config.json",
    }
)

# The adapter bundle mirrors the completed run on disk, while the Hub model
# repository puts PEFT loader files at its root and keeps Sommelier evidence in
# one explicit namespace. Keep this second allowlist independent so a caller
# cannot bypass ``prepare_hebrew_adapter_publication`` by constructing a
# ``PreparedPublication`` with arbitrary upload paths.
ADAPTER_UPLOAD_REQUIRED_FILES: Final = frozenset(
    {
        "README.md",
        "THIRD_PARTY.md",
        *ADAPTER_LICENSE_FILE_SHA256,
        "adapter_config.json",
        "adapter_model.safetensors",
        "sommelier/config.resolved.yaml",
        "sommelier/experiment_report.json",
        "sommelier/manifest.json",
        "sommelier/release_preflight.json",
        "sommelier/train_manifest.json",
        "sommelier/training_adapter_README.md",
    }
)
ADAPTER_UPLOAD_OPTIONAL_FILES: Final = frozenset(
    Path(name).relative_to("adapter").as_posix() for name in ADAPTER_OPTIONAL_FILES
)

_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_LORA_TENSOR = re.compile(r"^(?P<prefix>.+\.lora_)(?P<side>[AB])(?:\.[^.]+)?\.weight$")
_MARKDOWN_H1 = re.compile(r"^#[ \t]+(?P<title>\S.*?)[ \t]*$")
_SAFETENSORS_DTYPE_BITS: Final = {
    "BOOL": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}


class HubPublicationClient(Protocol):
    """Narrow Hub boundary used by the publication transaction.

    Tests inject a network-free fake; the production implementation imports
    ``huggingface_hub`` only when execution is explicitly requested. Repository
    creation is a separate operation so the caller must opt into it rather than
    relying on an upload helper with implicit create/overwrite behavior.
    """

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
    ) -> None: ...

    def list_files(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
        revision: str | None,
    ) -> Sequence[str]: ...

    def resolve_revision(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
    ) -> str | None: ...

    def create_commit(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
        files: Mapping[str, Path],
        commit_message: str,
        parent_commit: str | None,
    ) -> str: ...

    def download_file(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
        filename: str,
        revision: str,
    ) -> Path: ...


class _HuggingFaceHubClient:
    def __init__(self) -> None:
        try:
            from huggingface_hub import (
                CommitOperationAdd,
                HfApi,
                hf_hub_download,
            )
        except ImportError as error:
            raise ExternalDependencyError(
                "Hugging Face publication requires huggingface_hub",
                hint=(
                    "Install the pinned publication runtime, authenticate with HF_TOKEN, "
                    "then rerun the command with --execute."
                ),
            ) from error
        self._api = HfApi()
        self._operation_type = CommitOperationAdd
        self._download = hf_hub_download
        self._verification_staging = tempfile.TemporaryDirectory(prefix="sommelier-hf-roundtrip-")

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
    ) -> None:
        # Never adopt an existing repository through the create path. A caller
        # that intends to publish to an existing repo must leave create_repo
        # disabled so its current file tree is inspected first.
        self._api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=False,
            exist_ok=False,
        )

    def list_files(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
        revision: str | None,
    ) -> Sequence[str]:
        return cast(
            "Sequence[str]",
            self._api.list_repo_files(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
            ),
        )

    def resolve_revision(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
    ) -> str | None:
        info = self._api.repo_info(repo_id=repo_id, repo_type=repo_type)
        revision = getattr(info, "sha", None)
        if revision is not None and not isinstance(revision, str):
            raise ExternalDependencyError(
                "Hugging Face returned an invalid repository HEAD identity",
                hint="No upload was attempted; inspect the repository before retrying.",
            )
        return revision

    def create_commit(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
        files: Mapping[str, Path],
        commit_message: str,
        parent_commit: str | None,
    ) -> str:
        operations = [
            self._operation_type(path_in_repo=name, path_or_fileobj=str(path))
            for name, path in sorted(files.items())
        ]
        info = self._api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=commit_message,
            parent_commit=parent_commit,
        )
        oid = getattr(info, "oid", None)
        if not isinstance(oid, str):
            raise ExternalDependencyError(
                "Hugging Face did not return the created commit identity",
                hint="Inspect the repository before retrying; do not assume the upload failed.",
            )
        return oid

    def download_file(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
        filename: str,
        revision: str,
    ) -> Path:
        cached = Path(
            self._download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                revision=revision,
            )
        )
        if not cached.is_file():
            raise ExternalDependencyError(
                f"Hugging Face cache did not materialize a regular file for {filename!r}"
            )
        destination = Path(self._verification_staging.name) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)
        return destination


@dataclass(frozen=True)
class PreparedPublication:
    repo_type: PublicationRepoType
    files: Mapping[str, Path]
    expected_repo_id: str | None = None
    source_roots: tuple[Path, ...] = ()

    @property
    def sha256(self) -> dict[str, str]:
        return {name: sha256_file(path) for name, path in sorted(self.files.items())}


@dataclass(frozen=True)
class _SnapshotFile:
    path: Path
    sha256: str
    bytes: int
    device: int
    inode: int


@dataclass(frozen=True)
class _PrivateSnapshot:
    root: Path
    files: Mapping[str, Path]
    identities: Mapping[str, _SnapshotFile]


def _canonical_snapshot_name(name: str) -> Path:
    path = Path(name)
    if not name or name.startswith("/") or ".." in path.parts or path.as_posix() != name:
        raise UserInputError(f"publication snapshot has a non-canonical path: {name!r}")
    return path


def _copy_regular_file_to_snapshot(source: Path, destination: Path) -> _SnapshotFile:
    """Copy one source through no-follow descriptors into a private read-only file."""
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    try:
        source_descriptor = os.open(source, source_flags)
    except OSError as error:
        raise UserInputError(
            f"publication snapshot source is unavailable or unsafe: {source}",
            hint="Materialize every source as a regular file and retry validation.",
        ) from error
    destination_descriptor: int | None = None
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise UserInputError(f"publication snapshot source is not a regular file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.parent.chmod(0o700)
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            offset = 0
            while offset < len(chunk):
                written = os.write(destination_descriptor, chunk[offset:])
                if written <= 0:
                    raise OSError("short write while materializing publication snapshot")
                offset += written
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        source_identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        source_identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if source_identity_before != source_identity_after:
            raise UserInputError(
                f"publication snapshot source changed while it was copied: {source}",
                hint="Stop concurrent writers and rebuild the curated publication bundle.",
            )
    except UserInputError:
        destination.unlink(missing_ok=True)
        raise
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise UserInputError(
            f"could not materialize private publication snapshot from {source}",
            hint="Use a local filesystem with enough space, then retry validation.",
        ) from error
    finally:
        os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)

    destination.chmod(0o400)
    metadata = destination.stat(follow_symlinks=False)
    return _SnapshotFile(
        path=destination,
        sha256=sha256_file(destination),
        bytes=metadata.st_size,
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


@contextmanager
def _private_snapshot(
    files: Mapping[str, Path],
    *,
    prefix: str,
) -> Iterator[_PrivateSnapshot]:
    """Materialize exact source bytes below an unguessable mode-0700 directory."""
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        root = Path(temporary)
        root.chmod(0o700)
        staged: dict[str, Path] = {}
        identities: dict[str, _SnapshotFile] = {}
        for name, source in sorted(files.items()):
            relative = _canonical_snapshot_name(name)
            destination = root / relative
            identity = _copy_regular_file_to_snapshot(source, destination)
            staged[name] = destination
            identities[name] = identity
        yield _PrivateSnapshot(
            root=root,
            files=MappingProxyType(staged),
            identities=MappingProxyType(identities),
        )


def _assert_snapshot_unchanged(snapshot: _PrivateSnapshot) -> None:
    for name, expected in snapshot.identities.items():
        path = snapshot.files[name]
        try:
            observed = path.stat(follow_symlinks=False)
        except OSError as error:
            raise SecurityPolicyError(
                f"private publication snapshot changed before upload: {name}"
            ) from error
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_dev != expected.device
            or observed.st_ino != expected.inode
            or observed.st_size != expected.bytes
            or sha256_file(path) != expected.sha256
        ):
            raise SecurityPolicyError(
                f"private publication snapshot changed before upload: {name}",
                hint="Treat the attempt as unsafe and rebuild a fresh curated bundle.",
            )


@contextmanager
def _snapshot_dataset_sources(
    *,
    config_path: Path,
    bundle_dir: Path,
    root_rows_path: Path,
) -> Iterator[tuple[Path, Path, Path, Path]]:
    observed = _validate_exact_tree(bundle_dir, required=DATASET_REQUIRED_FILES)
    sources = {f"bundle/{name}": bundle_dir / name for name in observed}
    sources["inputs/config.yaml"] = config_path
    sources["inputs/root_rows.jsonl"] = root_rows_path
    with _private_snapshot(sources, prefix="sommelier-hf-dataset-source-") as snapshot:
        yield (
            snapshot.root / "inputs/config.yaml",
            snapshot.root / "bundle",
            snapshot.root / "inputs/root_rows.jsonl",
            snapshot.root,
        )


@contextmanager
def _snapshot_adapter_sources(bundle_dir: Path) -> Iterator[tuple[Path, Path]]:
    observed = _validate_exact_tree(
        bundle_dir,
        required=ADAPTER_REQUIRED_FILES,
        optional=ADAPTER_OPTIONAL_FILES,
    )
    sources = {f"bundle/{name}": bundle_dir / name for name in observed}
    with _private_snapshot(sources, prefix="sommelier-hf-adapter-source-") as snapshot:
        yield snapshot.root / "bundle", snapshot.root


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_yaml_mapping(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    payload: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in payload
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "YAML mapping key is not hashable",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "duplicate YAML mapping key",
                key_node.start_mark,
            )
        payload[key] = loader.construct_object(value_node, deep=deep)
    return payload


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _load_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = loads_unique_json(path.read_text(encoding="utf-8"))
    except (
        FileNotFoundError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        DuplicateJsonKeyError,
    ) as error:
        raise UserInputError(f"{context} is missing or invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise UserInputError(f"{context} must be a JSON object: {path}")
    return cast("dict[str, Any]", payload)


def _load_huggingface_frontmatter(path: Path, *, context: str) -> dict[str, Any]:
    """Parse the leading Hugging Face YAML block instead of trusting prose substrings."""
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError) as error:
        raise UserInputError(f"{context} is missing or is not UTF-8: {path}") from error
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise UserInputError(f"{context} must start with YAML frontmatter")
    try:
        closing = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration as error:
        raise UserInputError(f"{context} has no closing YAML frontmatter delimiter") from error
    try:
        payload = yaml.load("\n".join(lines[1:closing]), Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise UserInputError(f"{context} has invalid YAML frontmatter") from error
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise UserInputError(f"{context} YAML frontmatter must be a string-keyed object")
    return cast("dict[str, Any]", payload)


def _validate_exact_tree(
    root: Path,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> frozenset[str]:
    if not root.is_dir() or root.is_symlink():
        raise UserInputError(
            f"publication bundle is not a materialized directory: {root}",
            hint="Create a regular directory containing only the documented release files.",
        )
    allowed_files = required | optional
    allowed_directories = {
        Path(name).parent.as_posix() for name in allowed_files if Path(name).parent != Path(".")
    }
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise UserInputError(
                f"publication bundle contains a symlink: {relative}",
                hint="Materialize every release file before validation.",
            )
        if path.is_dir():
            observed_directories.add(relative)
        elif path.is_file():
            observed_files.add(relative)
        else:
            raise UserInputError(f"publication bundle contains a non-regular entry: {relative}")
    missing = sorted(required - observed_files)
    unexpected = sorted(observed_files - allowed_files)
    unexpected_directories = sorted(observed_directories - allowed_directories)
    if missing or unexpected or unexpected_directories:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected files: " + ", ".join(unexpected))
        if unexpected_directories:
            details.append("unexpected directories: " + ", ".join(unexpected_directories))
        raise UserInputError(
            "publication bundle does not match its exact allowlist (" + "; ".join(details) + ")",
            hint="Build a fresh curated bundle; never publish an artifact/run directory directly.",
        )
    return frozenset(observed_files)


def _assert_no_raw_provider_journal(bundle_dir: Path) -> None:
    matches = [
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob(OPENAI_PROVIDER_JOURNAL_FILENAME)
    ]
    if matches:
        raise SecurityPolicyError(
            "publication bundle contains the raw OpenAI provider journal",
            hint=(
                f"Remove {', '.join(matches)}. Publish the validated aggregate provenance "
                "in translation_summary.json, never the durable request journal."
            ),
        )


def _assert_no_secret_text(bundle_dir: Path, filenames: Sequence[str]) -> None:
    findings = []
    for filename in filenames:
        path = bundle_dir / filename
        if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}:
            continue
        findings.extend(scan_artifact_file(path, base_dir=bundle_dir))
    if findings:
        first = findings[0]
        raise SecurityPolicyError(
            f"publication bundle contains {len(findings)} unsafe or secret-like finding(s); "
            f"first is {first['kind']} in {first['file']} at {first['location']}",
            hint=(
                "Remove credentials, local home paths, and duplicate JSON keys before publication."
            ),
        )


def _prepublication_hebrew_config(config: SommelierConfig) -> SommelierConfig:
    hebrew = config.dataset_for("he")
    if hebrew.dataset_id != HEBREW_V3_PAIRED_DATASET_ID:
        raise UserInputError(
            "Hebrew v3 dataset repo does not match the preregistered dataset identity",
            hint=f"Use {HEBREW_V3_PAIRED_DATASET_ID} or amend the preregistration first.",
        )
    datasets = [
        source.model_copy(update={"dataset_revision": "0" * 40})
        if source.language == "he"
        else source
        for source in config.datasets
    ]
    pinned = config.model_copy(update={"datasets": datasets})
    validate_hebrew_v3_preregistered_config(pinned)
    return pinned


def _validate_dataset_card(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    metadata = _load_huggingface_frontmatter(path, context="Hebrew dataset README")
    if metadata.get("license") != "cc-by-4.0":
        raise UserInputError("Hebrew dataset README frontmatter license must be exactly cc-by-4.0")
    if _UNRESOLVED_DATASET_CARD_MARKER in text:
        raise UserInputError("Hebrew dataset README contains unresolved release template markers")
    required = (
        "Salesforce/xlam-function-calling-60k",
        "machine-translated",
        "Hebrew",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise UserInputError(
            "Hebrew dataset README is missing required license/provenance text: "
            + ", ".join(missing),
            hint="Use a dataset card that states CC-BY-4.0, source attribution, and translation.",
        )


def prepare_hebrew_dataset_publication(
    *,
    config_path: Path,
    bundle_dir: Path,
    root_rows_path: Path,
) -> PreparedPublication:
    """Validate and prepare the exact Hebrew dataset commit file map."""
    observed = _validate_exact_tree(bundle_dir, required=DATASET_REQUIRED_FILES)
    _assert_no_raw_provider_journal(bundle_dir)
    _assert_no_secret_text(bundle_dir, sorted(observed))
    _validate_dataset_card(bundle_dir / "README.md")

    config = _prepublication_hebrew_config(load_config(config_path))
    if not root_rows_path.is_file() or root_rows_path.is_symlink():
        raise UserInputError(f"root English rows are unavailable: {root_rows_path}")

    # Reuse the same complete selection/publication/semantic/provenance gate
    # consumed by a full training run. The temporary staging only adapts the
    # flat Hub bundle names to the local paired-input sidecar convention.
    with tempfile.TemporaryDirectory(prefix="sommelier-hf-dataset-") as temporary:
        staging = Path(temporary)
        staged_root = staging / "rows.jsonl"
        shutil.copy2(root_rows_path, staged_root)
        shutil.copy2(bundle_dir / rows_filename("he"), paired_input_path(staged_root, "he"))
        for filename in (
            SUMMARY_FILENAME,
            PUBLICATION_MANIFEST_FILENAME,
            SEMANTIC_REVIEW_FILENAME,
            SEMANTIC_REVIEW_TEMPLATE_FILENAME,
        ):
            shutil.copy2(
                bundle_dir / filename,
                translation_provenance_sidecar_path(staged_root, filename, "he"),
            )
        validated = validate_full_paired_input_contract(config, staged_root)
        if set(validated) != {"he"}:
            raise UserInputError(
                "Hebrew publication validation did not produce the exact Hebrew contract"
            )

    return PreparedPublication(
        repo_type="dataset",
        files={name: bundle_dir / name for name in sorted(DATASET_REQUIRED_FILES)},
        expected_repo_id=HEBREW_V3_PAIRED_DATASET_ID,
        source_roots=(bundle_dir,),
    )


def _validate_safetensors(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 10:
        raise UserInputError("adapter_model.safetensors is truncated")
    header_bytes = int.from_bytes(data[:8], byteorder="little", signed=False)
    if header_bytes <= 1 or header_bytes > len(data) - 8:
        raise UserInputError("adapter_model.safetensors has an invalid header length")
    encoded_header = data[8 : 8 + header_bytes]
    if not encoded_header.startswith(b"{"):
        raise UserInputError("adapter_model.safetensors has an invalid JSON header envelope")
    try:
        header_text = encoded_header.decode("utf-8")
        decoder = json.JSONDecoder(object_pairs_hook=reject_duplicate_json_keys)
        header, end = decoder.raw_decode(header_text)
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKeyError) as error:
        raise UserInputError("adapter_model.safetensors has an invalid JSON header") from error
    if any(character != " " for character in header_text[end:]):
        raise UserInputError("adapter_model.safetensors has an invalid JSON header envelope")
    if not isinstance(header, dict):
        raise UserInputError("adapter_model.safetensors header must be an object")
    safetensors_metadata = header.get("__metadata__")
    if safetensors_metadata is not None:
        if not isinstance(safetensors_metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in safetensors_metadata.items()
        ):
            raise UserInputError(
                "adapter_model.safetensors __metadata__ must be a string-to-string object"
            )
        findings = scan_json_payload(
            safetensors_metadata,
            file="adapter_model.safetensors::__metadata__",
        )
        if findings:
            first = findings[0]
            raise SecurityPolicyError(
                "adapter_model.safetensors __metadata__ contains "
                f"{len(findings)} secret-like value(s); first at {first['location']}",
                hint="Remove credentials and local paths from safetensors metadata.",
            )
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise UserInputError("adapter_model.safetensors contains no tensors")
    pairs: dict[str, set[str]] = {}
    intervals: list[tuple[int, int]] = []
    payload_bytes = len(data) - 8 - header_bytes
    for name, raw in tensors.items():
        if not isinstance(name, str) or (match := _LORA_TENSOR.fullmatch(name)) is None:
            raise UserInputError(
                f"adapter_model.safetensors contains non-LoRA tensor {name!r}",
                hint="Publish an unmerged PEFT LoRA, never base-model weights.",
            )
        if not isinstance(raw, dict):
            raise UserInputError(f"safetensors metadata for {name!r} is invalid")
        unexpected_fields = sorted(set(raw) - {"dtype", "shape", "data_offsets"})
        if unexpected_fields:
            raise UserInputError(
                f"safetensors metadata for {name!r} has unexpected fields",
                hint=(
                    "Publish canonical tensor metadata containing only dtype, shape, "
                    "and data_offsets."
                ),
            )
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        dtype = raw.get("dtype")
        if (
            not isinstance(dtype, str)
            or dtype not in _SAFETENSORS_DTYPE_BITS
            or not isinstance(shape, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in shape
            )
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise UserInputError(f"safetensors metadata for {name!r} is incomplete")
        start, end = cast("list[int]", offsets)
        if start < 0 or end < start or end > payload_bytes:
            raise UserInputError(f"safetensors offsets for {name!r} are outside the payload")
        elements = 1
        for dimension in cast("list[int]", shape):
            elements *= dimension
        expected_bits = elements * _SAFETENSORS_DTYPE_BITS[dtype]
        if expected_bits % 8 != 0 or end - start != expected_bits // 8:
            raise UserInputError(
                f"safetensors shape/dtype byte size for {name!r} does not match its offsets"
            )
        intervals.append((start, end))
        pair_key = match.group("prefix") + ".weight"
        pairs.setdefault(pair_key, set()).add(match.group("side"))
    if any(sides != {"A", "B"} for sides in pairs.values()):
        raise UserInputError("adapter_model.safetensors has an incomplete LoRA A/B tensor pair")
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise UserInputError("adapter_model.safetensors offsets overlap or leave gaps")
        cursor = end
    if cursor != payload_bytes:
        raise UserInputError("adapter_model.safetensors payload is not fully indexed")


def _validate_adapter_config(path: Path, config: SommelierConfig) -> None:
    payload = _load_json_object(path, context="PEFT adapter config")
    expected: dict[str, object] = {
        "base_model_name_or_path": config.model.base_model_id,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": config.train.lora_rank,
        "lora_alpha": config.train.lora_alpha,
        "lora_dropout": config.train.lora_dropout,
        "bias": "none",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise UserInputError(
                f"adapter_config.json {field}={payload.get(field)!r} does not match {value!r}",
                hint="Publish the canonical PEFT adapter produced by the bound training run.",
            )
    targets = payload.get("target_modules")
    if not isinstance(targets, list) or set(targets) != set(config.train.target_modules):
        raise UserInputError("adapter_config.json target_modules do not match the run config")


def _validate_adapter_manifests(
    *,
    bundle_dir: Path,
    config: SommelierConfig,
    adapter_files: Sequence[str],
) -> tuple[str, str, dict[str, Any]]:
    run = _load_json_object(bundle_dir / "manifest.json", context="run manifest")
    train = _load_json_object(bundle_dir / "train_manifest.json", context="train manifest")
    if (
        run.get("schema_version") != "sommelier.manifest.v1"
        or train.get("schema_version") != "sommelier.manifest.v1"
    ):
        raise UserInputError("adapter bundle manifests use an unsupported schema")
    run_id = train.get("run_id")
    if not isinstance(run_id, str) or not run_id or run.get("run_id") != run_id:
        raise UserInputError("run and train manifests do not share one run_id")
    if (
        run.get("status") != "succeeded"
        or train.get("stage") != "train"
        or train.get("status") != "succeeded"
    ):
        raise UserInputError("adapter publication requires a succeeded run and train stage")
    revision = train.get("git_commit")
    if not isinstance(revision, str) or IMMUTABLE_HF_REVISION.fullmatch(revision) is None:
        raise UserInputError("train manifest does not record an immutable source revision")
    resolved = bundle_dir / "config.resolved.yaml"
    config_digest = compute_config_digest(resolved.read_text(encoding="utf-8"))
    if train.get("config_sha256") != config_digest:
        raise UserInputError("train manifest does not bind config.resolved.yaml")
    config_ref = run.get("config")
    if not isinstance(config_ref, dict) or config_ref != {
        "path": f"runs/{run_id}/config.resolved.yaml",
        "kind": "config",
        "schema_version": "sommelier.config.v2",
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }:
        raise UserInputError("run manifest does not bind config.resolved.yaml")
    stages = run.get("stages")
    if not isinstance(stages, dict) or stages.get("train") != f"runs/{run_id}/train_manifest.json":
        raise UserInputError("run manifest does not bind the train manifest path")
    if train.get("seed") != config.project.seed:
        raise UserInputError("train manifest seed does not match the run config")
    dependency_lock_sha256 = train.get("dependency_lock_sha256")
    if (
        not isinstance(dependency_lock_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", dependency_lock_sha256) is None
    ):
        raise UserInputError("train manifest does not bind an immutable dependency lock digest")

    outputs = train.get("outputs")
    if not isinstance(outputs, list) or not all(isinstance(item, dict) for item in outputs):
        raise UserInputError("train manifest outputs are missing or invalid")
    adapter_outputs = {
        cast(str, output.get("path")): cast("dict[str, Any]", output)
        for output in outputs
        if output.get("kind") == "adapter_weights"
    }
    expected_paths = {
        f"runs/{run_id}/train/adapter/{Path(name).relative_to('adapter').as_posix()}"
        for name in adapter_files
    }
    if set(adapter_outputs) != expected_paths:
        raise UserInputError("train manifest does not bind the exact adapter file allowlist")
    for bundle_name in adapter_files:
        relative = Path(bundle_name).relative_to("adapter").as_posix()
        manifest_path = f"runs/{run_id}/train/adapter/{relative}"
        source = bundle_dir / bundle_name
        expected = {
            "path": manifest_path,
            "kind": "adapter_weights",
            "schema_version": "",
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
        }
        if adapter_outputs[manifest_path] != expected:
            raise UserInputError(f"train manifest digest does not match adapter/{relative}")
    details = train.get("details")
    if not isinstance(details, dict) or details.get("train_languages") != list(
        config.train.languages
    ):
        raise UserInputError("train manifest does not bind the configured training languages")
    return run_id, revision, train


def _validate_experiment_identity(
    *,
    path: Path,
    run_id: str,
    source_revision: str,
    config_sha256: str,
    tree_sha256: str,
) -> dict[str, Any]:
    report = _load_json_object(path, context="final experiment report")
    if report.get("schema_version") != EXPERIMENT_REPORT_SCHEMA:
        raise UserInputError("adapter publication requires the final experiment report schema")
    arms = report.get("arms")
    v3 = arms.get("v3_en_he") if isinstance(arms, dict) else None
    if not isinstance(v3, dict):
        raise UserInputError("final experiment report has no v3_en_he arm")
    source = v3.get("adapter_source")
    expected_source = {
        "source": f"runs/{run_id}/train/adapter",
        "revision": None,
        "kind": "local_directory",
        "tree_sha256": tree_sha256,
        "artifact_path": f"runs/{run_id}/train/adapter",
        "revision_is_immutable": True,
    }
    if v3.get("run_id") != run_id or v3.get("config_sha256") != config_sha256:
        raise UserInputError("final experiment report does not bind the training run/config")
    if source != expected_source:
        raise UserInputError("final experiment report does not bind the exact local adapter tree")
    preregistration = report.get("preregistration")
    finalizer = (
        preregistration.get("finalizer_source_code") if isinstance(preregistration, dict) else None
    )
    if (
        not isinstance(finalizer, dict)
        or finalizer.get("git_commit") != source_revision
        or finalizer.get("working_tree_clean") is not True
    ):
        raise UserInputError("experiment finalizer source does not match the training source")
    data_provenance = report.get("data_provenance")
    contract = data_provenance.get("contract") if isinstance(data_provenance, dict) else None
    if not isinstance(contract, dict) or contract.get("source_code_revision") != source_revision:
        raise UserInputError("experiment data provenance does not match the training source")
    if not isinstance(report.get("all_claims_passed"), bool) or not isinstance(
        report.get("approved_claims"), list
    ):
        raise UserInputError("final experiment report has incomplete claim-gate results")
    return report


def _validate_release_evidence(
    bundle_dir: Path,
    config: SommelierConfig,
    *,
    source_revision: str,
    dependency_lock_sha256: str,
) -> None:
    report = _load_json_object(bundle_dir / "release_preflight.json", context="release preflight")
    try:
        validate_release_preflight_report(
            report,
            config=config,
            artifact_root=bundle_dir,
            expected_source_git_commit=source_revision,
            expected_dependency_lock_sha256=dependency_lock_sha256,
            require_pass=True,
            require_all_gates_pass=True,
            require_clean_source=True,
            require_immutable_revisions=True,
        )
    except InvariantViolation as error:
        raise UserInputError(
            f"adapter publication release preflight identity is invalid: {error}",
            hint=(
                "Run release preflight against the final curated adapter bundle from the clean "
                "training source revision, then publish that unchanged bundle."
            ),
        ) from None
    notices = (bundle_dir / "THIRD_PARTY.md").read_text(encoding="utf-8")
    for text in (
        config.model.base_model_id,
        "NVIDIA Open Model License",
        "Llama 3.1 Community License",
        REQUIRED_DERIVED_NOTICE,
        config.root_dataset.dataset_id,
        "CC-BY-4.0",
    ):
        if text not in notices:
            raise UserInputError(f"THIRD_PARTY.md is missing required notice {text!r}")


def _validate_adapter_license_files(bundle_dir: Path) -> None:
    for filename, expected_sha256 in ADAPTER_LICENSE_FILE_SHA256.items():
        observed_sha256 = sha256_file(bundle_dir / filename)
        if observed_sha256 != expected_sha256:
            raise UserInputError(
                f"adapter license file {filename} does not match the reviewed project copy",
                hint=(
                    "Copy the exact tracked file from licenses/ into the curated adapter bundle; "
                    "do not abbreviate or edit license terms or NOTICE text."
                ),
            )


def _validate_adapter_card(
    path: Path,
    *,
    config: SommelierConfig,
    experiment_sha256: str,
    tree_sha256: str,
    source_revision: str,
    dataset_revision: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    metadata = _load_huggingface_frontmatter(path, context="adapter model card")
    if metadata.get("license") != "llama3.1":
        raise UserInputError("adapter model card frontmatter license must be exactly llama3.1")
    if metadata.get("base_model") != config.model.base_model_id:
        raise UserInputError("adapter model card frontmatter base_model does not match the run")
    if _UNRESOLVED_CARD_MARKER in text:
        raise UserInputError("adapter model card contains unresolved release template markers")
    lines = text.splitlines()
    closing = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    first_h1 = next(
        (
            match.group("title")
            for line in lines[closing + 1 :]
            if (match := _MARKDOWN_H1.fullmatch(line)) is not None
        ),
        None,
    )
    if first_h1 is None or not first_h1.startswith("Llama"):
        raise UserInputError(
            "adapter model card first Markdown H1 must begin with 'Llama'",
            hint="Name the derived-model card with a top-level '# Llama...' heading.",
        )
    required = (
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
    missing = [value for value in required if value not in text]
    if missing:
        raise UserInputError(
            "adapter model card is missing required license/identity text: " + ", ".join(missing),
            hint=(
                "Record Built with Llama, the base identity, adapter tree SHA256, and "
                "final experiment_report.json SHA256 in README.md."
            ),
        )


def prepare_hebrew_adapter_publication(*, bundle_dir: Path) -> PreparedPublication:
    """Validate and prepare the exact Hebrew v3 adapter commit file map."""
    observed = _validate_exact_tree(
        bundle_dir,
        required=ADAPTER_REQUIRED_FILES,
        optional=ADAPTER_OPTIONAL_FILES,
    )
    _assert_no_raw_provider_journal(bundle_dir)
    assert_artifacts_publishable(bundle_dir)
    _validate_adapter_license_files(bundle_dir)

    config_path = bundle_dir / "config.resolved.yaml"
    config = load_config(config_path)
    validate_hebrew_v3_preregistered_config(config)
    adapter_names = sorted(name for name in observed if name.startswith("adapter/"))
    adapter_dir = bundle_dir / "adapter"
    _validate_adapter_config(adapter_dir / "adapter_config.json", config)
    _validate_safetensors(adapter_dir / "adapter_model.safetensors")
    tree_sha256 = adapter_tree_sha256(adapter_dir)
    run_id, revision, train = _validate_adapter_manifests(
        bundle_dir=bundle_dir,
        config=config,
        adapter_files=adapter_names,
    )
    config_sha256 = cast(str, train["config_sha256"])
    experiment_path = bundle_dir / "experiment_report.json"
    _validate_experiment_identity(
        path=experiment_path,
        run_id=run_id,
        source_revision=revision,
        config_sha256=config_sha256,
        tree_sha256=tree_sha256,
    )
    dependency_lock_sha256 = cast(str, train["dependency_lock_sha256"])
    _validate_release_evidence(
        bundle_dir,
        config,
        source_revision=revision,
        dependency_lock_sha256=dependency_lock_sha256,
    )
    _validate_adapter_card(
        bundle_dir / "README.md",
        config=config,
        experiment_sha256=sha256_file(experiment_path),
        tree_sha256=tree_sha256,
        source_revision=revision,
        dataset_revision=config.dataset_for("he").dataset_revision,
    )

    files: dict[str, Path] = {
        "README.md": bundle_dir / "README.md",
        "THIRD_PARTY.md": bundle_dir / "THIRD_PARTY.md",
        "sommelier/config.resolved.yaml": config_path,
        "sommelier/experiment_report.json": experiment_path,
        "sommelier/manifest.json": bundle_dir / "manifest.json",
        "sommelier/release_preflight.json": bundle_dir / "release_preflight.json",
        "sommelier/train_manifest.json": bundle_dir / "train_manifest.json",
        "sommelier/training_adapter_README.md": adapter_dir / "README.md",
    }
    for filename in ADAPTER_LICENSE_FILE_SHA256:
        files[filename] = bundle_dir / filename
    for bundle_name in adapter_names:
        if bundle_name == "adapter/README.md":
            continue
        files[Path(bundle_name).relative_to("adapter").as_posix()] = bundle_dir / bundle_name
    return PreparedPublication(repo_type="model", files=files, source_roots=(bundle_dir,))


def _validate_repo_id(repo_id: str) -> None:
    if _REPO_ID.fullmatch(repo_id) is None:
        raise UserInputError(
            f"invalid Hugging Face repo ID: {repo_id!r}",
            hint="Pass one explicit namespace/name ID, without a URL or revision suffix.",
        )


def _normalize_remote_files(files: Sequence[str]) -> frozenset[str]:
    normalized: set[str] = set()
    for value in files:
        if not isinstance(value, str) or not value or value.startswith("/"):
            raise ExternalDependencyError("Hugging Face returned an invalid repository filename")
        path = Path(value)
        if ".." in path.parts or path.as_posix() != value:
            raise ExternalDependencyError("Hugging Face returned a non-canonical repository path")
        normalized.add(value)
    return frozenset(normalized)


def _validate_prepared_publication(prepared: PreparedPublication) -> None:
    names = frozenset(prepared.files)
    if prepared.repo_type == "dataset":
        missing = sorted(DATASET_REQUIRED_FILES - names)
        unexpected = sorted(names - DATASET_REQUIRED_FILES)
    else:
        allowed = ADAPTER_UPLOAD_REQUIRED_FILES | ADAPTER_UPLOAD_OPTIONAL_FILES
        missing = sorted(ADAPTER_UPLOAD_REQUIRED_FILES - names)
        unexpected = sorted(names - allowed)
    if missing or unexpected:
        raise UserInputError(
            "prepared publication does not match its exact upload allowlist "
            f"(missing={missing}, unexpected={unexpected})",
            hint="Prepare a fresh curated Hebrew dataset or adapter publication bundle.",
        )
    for name, path in prepared.files.items():
        remote_path = Path(name)
        if (
            not name
            or name.startswith("/")
            or ".." in remote_path.parts
            or remote_path.as_posix() != name
        ):
            raise UserInputError(f"prepared publication has a non-canonical path: {name!r}")
        if not path.is_file() or path.is_symlink():
            raise UserInputError(
                f"prepared publication source is not a regular file: {path}",
                hint="Materialize every upload source and rerun validation.",
            )


def _publication_source_roots(prepared: PreparedPublication) -> tuple[Path, ...]:
    resolved = [path.resolve() for path in prepared.files.values()]
    try:
        common = Path(os.path.commonpath(resolved))
    except ValueError as error:
        raise UserInputError(
            "prepared publication files do not share one source bundle root",
            hint="Prepare every upload from one curated publication bundle.",
        ) from error
    if common in resolved:
        common = common.parent
    roots = [*prepared.source_roots]
    if common not in roots:
        roots.append(common)
    return tuple(roots)


def _assert_receipt_outside_publication_sources(
    receipt_path: Path,
    *,
    source_roots: Sequence[Path],
) -> None:
    receipt = receipt_path.resolve(strict=False)
    for source_root in source_roots:
        root = source_root.resolve(strict=False)
        current = Path(os.path.abspath(receipt_path))
        aliases_source_root = False
        while True:
            try:
                if (current.exists() or current.is_symlink()) and os.path.samefile(
                    current, source_root
                ):
                    aliases_source_root = True
                    break
            except OSError:
                pass
            parent = current.parent
            if parent == current:
                break
            current = parent
        if receipt == root or receipt.is_relative_to(root) or aliases_source_root:
            raise UserInputError(
                "publication receipt must be outside the source bundle and private snapshot",
                hint="Choose a new receipt path in a separate directory.",
            )


def _assert_new_receipt_path(path: Path) -> None:
    # Path.exists() is false for a broken symlink, so check both. An existing
    # directory is also deliberately reported as an occupied receipt target.
    if path.exists() or path.is_symlink():
        raise UserInputError(
            f"publication receipt already exists: {path}",
            hint="Choose a new receipt path so an earlier verified release is never overwritten.",
        )


@dataclass(frozen=True)
class _ReceiptReservation:
    path: Path
    capacity: int
    device: int
    inode: int
    parent_device: int
    parent_inode: int
    source_root_identities: frozenset[tuple[int, int]]


def _receipt_bytes(payload: Mapping[str, object], *, capacity: int | None = None) -> bytes:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if capacity is None:
        return encoded
    if len(encoded) > capacity:
        raise ExternalDependencyError(
            "publication receipt exceeded its preallocated durable reservation",
            hint=(
                "The pending receipt still records the attempted publication; "
                "inspect it and the Hub."
            ),
        )
    return encoded + (b" " * (capacity - len(encoded)))


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("short write while updating publication receipt")
        offset += written


def _source_root_identities(source_roots: Sequence[Path]) -> frozenset[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    for source_root in source_roots:
        metadata = source_root.resolve(strict=True).stat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError("publication source root is not a directory")
        identities.add((metadata.st_dev, metadata.st_ino))
    return frozenset(identities)


def _posix_directory_is_within_sources(
    descriptor: int,
    *,
    source_root_identities: frozenset[tuple[int, int]],
) -> bool:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    current = os.dup(descriptor)
    seen: set[tuple[int, int]] = set()
    try:
        while True:
            metadata = os.fstat(current)
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in source_root_identities:
                return True
            if identity in seen:
                raise OSError("directory ancestry loop while reserving publication receipt")
            seen.add(identity)
            parent = os.open("..", flags, dir_fd=current)
            try:
                parent_metadata = os.fstat(parent)
            except Exception:
                os.close(parent)
                raise
            parent_identity = (parent_metadata.st_dev, parent_metadata.st_ino)
            if parent_identity == identity:
                os.close(parent)
                return False
            os.close(current)
            current = parent
    finally:
        os.close(current)


def _open_posix_receipt_parent(
    path: Path,
    *,
    source_root_identities: frozenset[tuple[int, int]],
    expected_identity: tuple[int, int] | None = None,
) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path.parent, flags)
    try:
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError("publication receipt parent is not a directory")
        if expected_identity is not None and identity != expected_identity:
            raise OSError("publication receipt parent identity changed")
        if _posix_directory_is_within_sources(
            descriptor,
            source_root_identities=source_root_identities,
        ):
            raise OSError("publication receipt parent aliases a publication source")
        path_metadata = path.parent.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or path_metadata.st_dev != metadata.st_dev
            or path_metadata.st_ino != metadata.st_ino
        ):
            raise OSError("publication receipt parent path identity changed")
        return descriptor, metadata
    except Exception:
        os.close(descriptor)
        raise


def _reserve_receipt(
    path: Path,
    payload: Mapping[str, object],
    *,
    source_roots: Sequence[Path],
) -> _ReceiptReservation:
    """Create and fsync a generously preallocated pending journal before Hub mutation."""
    _assert_new_receipt_path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise UserInputError(
            f"could not reserve publication receipt before network access: {path}",
            hint="Choose a writable local path with sufficient durable storage, then retry.",
        ) from error
    _assert_new_receipt_path(path)
    pending = _receipt_bytes(payload)
    capacity = len(pending) + 16_384
    descriptor: int | None = None
    parent_descriptor: int | None = None
    metadata: os.stat_result | None = None
    parent_metadata: os.stat_result | None = None
    try:
        root_identities = _source_root_identities(source_roots)
        if os.name == "posix":
            parent_descriptor, parent_metadata = _open_posix_receipt_parent(
                path,
                source_root_identities=root_identities,
            )
            descriptor = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
        else:
            _assert_receipt_outside_publication_sources(path, source_roots=source_roots)
            parent_metadata = path.parent.stat()
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise UserInputError(
            f"publication receipt already exists: {path}",
            hint="Choose a new receipt path so an earlier verified release is never overwritten.",
        ) from error
    except OSError as error:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise UserInputError(
            f"could not reserve publication receipt before network access: {path}",
            hint="Choose a writable local path with sufficient durable storage, then retry.",
        ) from error
    try:
        if descriptor is None or parent_metadata is None:
            raise OSError("publication receipt descriptors were not initialized")
        _write_all(descriptor, _receipt_bytes(payload, capacity=capacity))
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("reserved publication receipt is not a regular file")
        if parent_descriptor is not None:
            entry = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
            if entry.st_dev != metadata.st_dev or entry.st_ino != metadata.st_ino:
                raise OSError("reserved publication receipt path identity changed")
            if _posix_directory_is_within_sources(
                parent_descriptor,
                source_root_identities=root_identities,
            ):
                raise OSError("publication receipt parent moved into a publication source")
            os.fsync(parent_descriptor)
    except Exception as error:
        if descriptor is not None and parent_descriptor is not None:
            try:
                entry = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
                current = os.fstat(descriptor)
                if entry.st_dev == current.st_dev and entry.st_ino == current.st_ino:
                    os.unlink(path.name, dir_fd=parent_descriptor)
            except OSError:
                pass
        elif descriptor is not None:
            try:
                current = os.fstat(descriptor)
                entry = path.stat(follow_symlinks=False)
                if entry.st_dev == current.st_dev and entry.st_ino == current.st_ino:
                    path.unlink()
            except OSError:
                pass
        raise UserInputError(
            f"could not durably reserve publication receipt before network access: {path}",
            hint="Choose a writable local path with sufficient durable storage, then retry.",
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    if metadata is None or parent_metadata is None:
        raise UserInputError(f"could not reserve publication receipt before network access: {path}")
    return _ReceiptReservation(
        path=path,
        capacity=capacity,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        parent_device=parent_metadata.st_dev,
        parent_inode=parent_metadata.st_ino,
        source_root_identities=root_identities,
    )


def _write_receipt(reservation: _ReceiptReservation, payload: Mapping[str, object]) -> None:
    flags = os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor: int | None = None
    parent_descriptor: int | None = None
    try:
        if os.name == "posix":
            parent_descriptor, _ = _open_posix_receipt_parent(
                reservation.path,
                source_root_identities=reservation.source_root_identities,
                expected_identity=(reservation.parent_device, reservation.parent_inode),
            )
            descriptor = os.open(
                reservation.path.name,
                flags,
                dir_fd=parent_descriptor,
            )
        else:
            descriptor = os.open(reservation.path, flags)
    except OSError as error:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise ExternalDependencyError(
            f"could not update reserved publication receipt: {reservation.path}",
            hint="Inspect the Hub and the pending receipt before any retry.",
        ) from error
    try:
        if descriptor is None:
            raise OSError("reserved publication receipt descriptor was not initialized")
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_dev != reservation.device
            or metadata.st_ino != reservation.inode
        ):
            raise OSError("reserved publication receipt path identity changed")
        data = _receipt_bytes(payload, capacity=reservation.capacity)
        _write_all(descriptor, data)
        os.fsync(descriptor)
        if parent_descriptor is not None:
            entry = os.stat(
                reservation.path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if entry.st_dev != metadata.st_dev or entry.st_ino != metadata.st_ino:
                raise OSError("reserved publication receipt path identity changed")
            if _posix_directory_is_within_sources(
                parent_descriptor,
                source_root_identities=reservation.source_root_identities,
            ):
                raise OSError("publication receipt parent moved into a publication source")
    except Exception as error:
        raise ExternalDependencyError(
            f"could not durably update publication receipt: {reservation.path}",
            hint="Inspect the Hub and the pending receipt before any retry.",
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _publish_prepared_bundle(
    prepared: PreparedPublication,
    *,
    repo_id: str,
    commit_message: str,
    execute: bool = False,
    create_repo: bool = False,
    confirmed_repo_id: str | None = None,
    receipt_path: Path | None = None,
    client: HubPublicationClient | None = None,
    _snapshot: _PrivateSnapshot | None = None,
) -> dict[str, object]:
    """Validate-only by default; explicitly execute and round-trip one commit."""
    _validate_repo_id(repo_id)
    _validate_prepared_publication(prepared)
    if prepared.expected_repo_id is not None and repo_id != prepared.expected_repo_id:
        raise UserInputError(
            "publication repository does not match the prepared artifact identity",
            hint=f"Publish this bundle only to {prepared.expected_repo_id}.",
        )
    if prepared.repo_type == "model" and not repo_id.partition("/")[2].startswith("Llama"):
        raise UserInputError(
            "Llama-derived adapter repository names must begin with 'Llama'",
            hint="Choose an explicit namespace/Llama... repository ID required by the license.",
        )
    if not commit_message.strip() or "\n" in commit_message or "\r" in commit_message:
        raise UserInputError("commit_message must be one non-empty line")
    if _snapshot is None:
        source_roots = _publication_source_roots(prepared)
        with _private_snapshot(
            prepared.files,
            prefix="sommelier-hf-upload-snapshot-",
        ) as snapshot:
            staged = replace(
                prepared,
                files=snapshot.files,
                source_roots=(*source_roots, snapshot.root),
            )
            return _publish_prepared_bundle(
                staged,
                repo_id=repo_id,
                commit_message=commit_message,
                execute=execute,
                create_repo=create_repo,
                confirmed_repo_id=confirmed_repo_id,
                receipt_path=receipt_path,
                client=client,
                _snapshot=snapshot,
            )
    _assert_snapshot_unchanged(_snapshot)
    file_hashes = {name: identity.sha256 for name, identity in sorted(_snapshot.identities.items())}
    plan: dict[str, object] = {
        "schema_version": PUBLICATION_RECEIPT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "validated" if not execute else "pending",
        "executed": execute,
        "repository": {
            "repo_id": repo_id,
            "repo_type": prepared.repo_type,
            "commit_message": commit_message,
            "commit_sha": None,
            "parent_commit": None,
            "create_repo": create_repo,
        },
        "files": [
            {
                "path": name,
                "sha256": file_hashes[name],
                "bytes": _snapshot.identities[name].bytes,
            }
            for name in sorted(prepared.files)
        ],
        "platform_files": [],
    }
    if not execute:
        if receipt_path is not None:
            raise UserInputError("receipt_path is only valid with execute=True")
        return plan
    if confirmed_repo_id != repo_id:
        raise UserInputError(
            "confirmed repo ID does not exactly match repo_id",
            hint=f"Pass --confirm-repo-id {repo_id} together with --execute.",
        )
    if receipt_path is None:
        raise UserInputError("executed publication requires an explicit receipt_path")
    _assert_receipt_outside_publication_sources(
        receipt_path,
        source_roots=prepared.source_roots,
    )
    reservation = _reserve_receipt(
        receipt_path,
        plan,
        source_roots=prepared.source_roots,
    )

    active = client if client is not None else _HuggingFaceHubClient()
    allowed_remote = frozenset(prepared.files) | PLATFORM_MANAGED_FILES
    created_by_transaction = False
    if create_repo:
        try:
            active.create_repo(repo_id=repo_id, repo_type=prepared.repo_type)
            created_by_transaction = True
        except Exception as error:
            raise ExternalDependencyError(
                f"could not create new public Hugging Face repository {repo_id!r}",
                hint=(
                    "No upload was attempted. Repository creation uses private=False and "
                    "exist_ok=False; disable create_repo to target an existing repository."
                ),
            ) from error
    try:
        parent_commit = active.resolve_revision(repo_id=repo_id, repo_type=prepared.repo_type)
        if parent_commit is not None and IMMUTABLE_HF_REVISION.fullmatch(parent_commit) is None:
            raise ExternalDependencyError(
                f"Hugging Face returned a non-immutable repository HEAD: {parent_commit!r}"
            )
        existing = _normalize_remote_files(
            active.list_files(
                repo_id=repo_id,
                repo_type=prepared.repo_type,
                revision=parent_commit,
            )
        )
    except ExternalDependencyError:
        raise
    except Exception as error:
        raise ExternalDependencyError(
            f"could not inspect existing Hugging Face repository {repo_id!r}",
            hint=(
                "Authorize an existing repository or explicitly enable create_repo for a new "
                "public repository, then rerun."
            ),
        ) from error
    unexpected_existing = sorted(existing - allowed_remote)
    if unexpected_existing:
        raise UserInputError(
            "Hugging Face repository contains files outside the publication allowlist: "
            + ", ".join(unexpected_existing),
            hint="Use a dedicated empty repository or remove unrelated files before publishing.",
        )
    if parent_commit is None:
        if existing:
            raise ExternalDependencyError(
                "Hugging Face reported files for a repository with no commit identity",
                hint="No upload was attempted; inspect the repository before retrying.",
            )
        if not created_by_transaction:
            raise ExternalDependencyError(
                "Hugging Face reported a parentless empty repository that was not created "
                "by this publication transaction",
                hint=(
                    "No upload was attempted. Use --create-repo only for an absent repository; "
                    "pre-existing or ambiguous empty repositories fail closed."
                ),
            )
        latest = active.resolve_revision(repo_id=repo_id, repo_type=prepared.repo_type)
        if latest is not None:
            raise ExternalDependencyError(
                "Hugging Face repository changed while preparing its first commit",
                hint="No upload was attempted; rerun against the now-immutable repository HEAD.",
            )

    commit_submitting: dict[str, object] = {
        **plan,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "commit_submitting",
        "repository": {
            "repo_id": repo_id,
            "repo_type": prepared.repo_type,
            "commit_message": commit_message,
            "commit_sha": None,
            "parent_commit": parent_commit,
            "create_repo": create_repo,
        },
        "platform_files": sorted(existing - frozenset(prepared.files)),
    }
    _write_receipt(reservation, commit_submitting)
    _assert_snapshot_unchanged(_snapshot)

    try:
        revision = active.create_commit(
            repo_id=repo_id,
            repo_type=prepared.repo_type,
            files=prepared.files,
            commit_message=commit_message,
            parent_commit=parent_commit,
        )
    except Exception as error:
        raise ExternalDependencyError(
            f"Hugging Face commit failed for {repo_id!r}",
            hint="Inspect the repository before retrying; the server may have accepted the commit.",
        ) from error
    commit_returned: dict[str, object] = {
        **plan,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "commit_returned_unverified",
        "repository": {
            "repo_id": repo_id,
            "repo_type": prepared.repo_type,
            "commit_message": commit_message,
            "commit_sha": revision,
            "parent_commit": parent_commit,
            "create_repo": create_repo,
        },
    }
    _write_receipt(reservation, commit_returned)
    _assert_snapshot_unchanged(_snapshot)
    if IMMUTABLE_HF_REVISION.fullmatch(revision) is None:
        raise ExternalDependencyError(
            f"Hugging Face returned a non-immutable commit identity: {revision!r}",
            hint="Inspect the repository before retrying; do not record a branch name as evidence.",
        )
    try:
        remote_files = _normalize_remote_files(
            active.list_files(
                repo_id=repo_id,
                repo_type=prepared.repo_type,
                revision=revision,
            )
        )
    except Exception as error:
        raise ExternalDependencyError(
            f"could not enumerate {repo_id!r} at returned commit {revision}",
            hint="Treat the publication as unverified and inspect the immutable commit manually.",
        ) from error
    if (
        not frozenset(prepared.files).issubset(remote_files)
        or remote_files - frozenset(prepared.files) - PLATFORM_MANAGED_FILES
    ):
        missing = sorted(frozenset(prepared.files) - remote_files)
        unexpected = sorted(remote_files - frozenset(prepared.files) - PLATFORM_MANAGED_FILES)
        raise ExternalDependencyError(
            "Hugging Face round-trip filename verification failed "
            f"(missing={missing}, unexpected={unexpected})",
            hint="Treat the publication as unverified; do not pin this commit in a config.",
        )
    for filename, expected_sha256 in file_hashes.items():
        try:
            downloaded = active.download_file(
                repo_id=repo_id,
                repo_type=prepared.repo_type,
                filename=filename,
                revision=revision,
            )
        except Exception as error:
            raise ExternalDependencyError(
                f"could not round-trip {filename!r} from {repo_id}@{revision}",
                hint="Treat the publication as unverified; do not pin this commit.",
            ) from error
        if not downloaded.is_file() or downloaded.is_symlink():
            raise ExternalDependencyError(
                f"Hugging Face round-trip for {filename!r} did not return a regular file"
            )
        observed_sha256 = sha256_file(downloaded)
        if observed_sha256 != expected_sha256:
            raise ExternalDependencyError(
                f"Hugging Face round-trip SHA256 mismatch for {filename!r}",
                hint="Treat the publication as corrupted/unverified; do not pin this commit.",
            )

    receipt: dict[str, object] = {
        **plan,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "verified",
        "repository": {
            "repo_id": repo_id,
            "repo_type": prepared.repo_type,
            "commit_message": commit_message,
            "commit_sha": revision,
            "parent_commit": parent_commit,
            "create_repo": create_repo,
        },
        "platform_files": sorted(remote_files - frozenset(prepared.files)),
    }
    _write_receipt(reservation, receipt)
    return receipt


def publish_hebrew_dataset_bundle(
    *,
    config_path: Path,
    bundle_dir: Path,
    root_rows_path: Path,
    repo_id: str,
    commit_message: str,
    execute: bool = False,
    create_repo: bool = False,
    confirmed_repo_id: str | None = None,
    receipt_path: Path | None = None,
    client: HubPublicationClient | None = None,
) -> dict[str, object]:
    """Validate the complete Hebrew dataset contract, then optionally publish it."""
    with _snapshot_dataset_sources(
        config_path=config_path,
        bundle_dir=bundle_dir,
        root_rows_path=root_rows_path,
    ) as (staged_config, staged_bundle, staged_root_rows, source_snapshot_root):
        prepared = prepare_hebrew_dataset_publication(
            config_path=staged_config,
            bundle_dir=staged_bundle,
            root_rows_path=staged_root_rows,
        )
        prepared = replace(
            prepared,
            source_roots=(bundle_dir, source_snapshot_root),
        )
        return _publish_prepared_bundle(
            prepared,
            repo_id=repo_id,
            commit_message=commit_message,
            execute=execute,
            create_repo=create_repo,
            confirmed_repo_id=confirmed_repo_id,
            receipt_path=receipt_path,
            client=client,
        )


def publish_hebrew_adapter_bundle(
    *,
    bundle_dir: Path,
    repo_id: str,
    commit_message: str,
    execute: bool = False,
    create_repo: bool = False,
    confirmed_repo_id: str | None = None,
    receipt_path: Path | None = None,
    client: HubPublicationClient | None = None,
) -> dict[str, object]:
    """Validate the complete Hebrew adapter contract, then optionally publish it."""
    with _snapshot_adapter_sources(bundle_dir) as (staged_bundle, source_snapshot_root):
        prepared = prepare_hebrew_adapter_publication(bundle_dir=staged_bundle)
        prepared = replace(
            prepared,
            source_roots=(bundle_dir, source_snapshot_root),
        )
        return _publish_prepared_bundle(
            prepared,
            repo_id=repo_id,
            commit_message=commit_message,
            execute=execute,
            create_repo=create_repo,
            confirmed_repo_id=confirmed_repo_id,
            receipt_path=receipt_path,
            client=client,
        )


__all__ = [
    "publish_hebrew_adapter_bundle",
    "publish_hebrew_dataset_bundle",
]
