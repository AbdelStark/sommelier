from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
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
    SEMANTIC_REVIEW_SCHEMA,
    SEMANTIC_REVIEW_TEMPLATE_FILENAME,
    SEMANTIC_REVIEW_TEMPLATE_SCHEMA,
)
from sommelier.data.translate import (
    PUBLICATION_MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    TRANSLATION_CONFIG_FILENAME,
    TRANSLATION_PUBLICATION_SCHEMA,
    TRANSLATION_RUN_IDENTITY_FILENAME,
    TRANSLATION_RUN_IDENTITY_SCHEMA,
    TRANSLATION_SUMMARY_SCHEMA,
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
    DATA_PROVENANCE_SCHEMA,
    HEBREW_V3_BASE_MODEL_ID,
    HEBREW_V3_BASE_MODEL_REVISION,
    HEBREW_V3_PAIRED_DATASET_ID,
    HEBREW_V3_ROOT_DATASET_ID,
    HEBREW_V3_ROOT_DATASET_REVISION,
    HEBREW_V3_V1_ADAPTER_ID,
    HEBREW_V3_V1_ADAPTER_REVISION,
    validate_hebrew_v3_preregistered_config,
)
from sommelier.evaluation.experiment import (
    EXPERIMENT_REPORT_SCHEMA,
    HEBREW_V3_BOOTSTRAP_RESAMPLES,
    HEBREW_V3_BOOTSTRAP_SEED,
    HEBREW_V3_ENGLISH_NON_INFERIORITY_MARGIN,
    HEBREW_V3_PREREGISTRATION_SCHEMA,
)
from sommelier.evaluation.generate import (
    GENERATION_TIMING_AGGREGATION,
    GENERATION_TIMING_SCOPE,
    IMMUTABLE_HF_REVISION,
    SEQUENTIAL_RUN_BOUNDARY,
    adapter_tree_sha256,
    inference_timed_call_contract,
    inference_warmup_contract,
)
from sommelier.evaluation.metrics import METRIC_NAMES
from sommelier.evaluation.release_evidence import (
    EVALUATION_RELEASE_EVIDENCE_BUNDLE_FILES,
    validate_evaluation_release_evidence,
)
from sommelier.evaluation.statistics import (
    EXACT_MCNEMAR_VERSION,
    PAIRED_BOOTSTRAP_VERSION,
    stable_bootstrap_seed,
)
from sommelier.evaluation.tco import SOVEREIGN_TCO_EVIDENCE_SCHEMA
from sommelier.hebrew_v3_preregistration import require_committed_config_bytes
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
_EXPERIMENT_REPORT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "created_at",
        "preregistration",
        "shared_evaluation_identity",
        "data_provenance",
        "bootstrap",
        "arms",
        "comparisons",
        "sovereign_tco_evidence",
        "claims",
        "all_claims_passed",
        "approved_claims",
    }
)
_HEBREW_UPLIFT_STATEMENT: Final = (
    "The v3 en+he adapter improves Hebrew full-call exact match "
    "over the v1 English adapter on the gated cohort."
)
_ENGLISH_NON_INFERIORITY_STATEMENT: Final = (
    "The v3 en+he adapter is non-inferior to the v1 English adapter "
    "on English full-call exact match at the declared margin."
)
_CLAIM_ORDER: Final = (
    "hebrew_full_call_uplift",
    "english_full_call_non_inferiority",
)
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_HEBREW_V3_MAX_TEST_EXAMPLES: Final = 1_000
_MAX_PUBLICATION_RECEIPT_BYTES: Final = 1024 * 1024
_CLAIM_HEADING: Final = re.compile(r"##[ \t]+Claim-gated result[ \t]*")
_UNAPPROVED_RESULT_ASSERTION: Final = re.compile(
    r"(?:"
    r"\b(?:accuracy|exact[ -]match|full[ -]call|performance|score|rate|uplift|result)\w*\b"
    r"[^\n.!?]{0,160}"
    r"\b(?:improv\w*|outperform\w*|beat\w*|better|worse|superior|"
    r"non[ -]inferior\w*|regress\w*|gain\w*|increas\w*|decreas\w*|"
    r"(?:un)?compromis\w*|preserv\w*|maintain\w*|retain\w*|unchang\w*|stable|intact)\b"
    r"|"
    r"\b(?:improv\w*|outperform\w*|beat\w*|better|worse|superior|"
    r"non[ -]inferior\w*|regress\w*|gain\w*|increas\w*|decreas\w*|"
    r"(?:un)?compromis\w*|preserv\w*|maintain\w*|retain\w*|unchang\w*|stable|intact)\b"
    r"[^\n.!?]{0,160}"
    r"\b(?:accuracy|exact[ -]match|full[ -]call|performance|score|rate|uplift|result)\w*\b"
    r"|"
    r"\b(?:estimate|confidence interval|paired[ -]bootstrap CI)\b"
    r"\s*[:=`\[]*\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"|"
    r"\b(?:accuracy|exact[ -]match|full[ -]call|performance|score|rate|uplift)\w*\b"
    r"[^\n.!?]{0,100}"
    r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*%|[+-]?(?:0?\.\d+))"
    r")",
    flags=re.IGNORECASE,
)

DATASET_REQUIRED_FILES: Final = frozenset(
    {
        "README.md",
        rows_filename("he"),
        SUMMARY_FILENAME,
        PUBLICATION_MANIFEST_FILENAME,
        SEMANTIC_REVIEW_FILENAME,
        SEMANTIC_REVIEW_TEMPLATE_FILENAME,
        TRANSLATION_CONFIG_FILENAME,
        TRANSLATION_RUN_IDENTITY_FILENAME,
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
        *EVALUATION_RELEASE_EVIDENCE_BUNDLE_FILES,
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
        *{f"sommelier/{name}" for name in EVALUATION_RELEASE_EVIDENCE_BUNDLE_FILES},
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

    def inspect_commit(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
        revision: str,
    ) -> _HubCommitMetadata: ...

    def download_file(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
        filename: str,
        revision: str,
    ) -> Path: ...


@dataclass(frozen=True)
class _HubCommitMetadata:
    parent_commit: str | None
    title: str


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

    def inspect_commit(
        self,
        *,
        repo_id: str,
        repo_type: PublicationRepoType,
        revision: str,
    ) -> _HubCommitMetadata:
        commits = self._api.list_repo_commits(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
        )
        if not isinstance(commits, list) or not commits:
            raise ExternalDependencyError(
                f"Hugging Face did not return commit metadata for {repo_id}@{revision}"
            )
        commit_id = getattr(commits[0], "commit_id", None)
        title = getattr(commits[0], "title", None)
        parent_commit = getattr(commits[1], "commit_id", None) if len(commits) > 1 else None
        if (
            commit_id != revision
            or not isinstance(title, str)
            or (
                parent_commit is not None
                and (
                    not isinstance(parent_commit, str)
                    or IMMUTABLE_HF_REVISION.fullmatch(parent_commit) is None
                )
            )
        ):
            raise ExternalDependencyError(
                f"Hugging Face returned invalid commit metadata for {repo_id}@{revision}"
            )
        return _HubCommitMetadata(parent_commit=parent_commit, title=title)

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


def _phase_a_source_revision(bundle_dir: Path) -> str:
    summary = _load_json_object(
        bundle_dir / SUMMARY_FILENAME,
        context="Hebrew v3 translation summary",
    )
    source_code = summary.get("source_code")
    source_revision = source_code.get("git_commit") if isinstance(source_code, dict) else None
    if (
        not isinstance(source_revision, str)
        or IMMUTABLE_HF_REVISION.fullmatch(source_revision) is None
    ):
        raise UserInputError(
            "Hebrew v3 translation summary has no immutable Phase-A source revision"
        )
    return source_revision


def prepare_hebrew_dataset_publication(
    *,
    config_path: Path,
    bundle_dir: Path,
    root_rows_path: Path,
) -> PreparedPublication:
    """Validate and prepare the exact Hebrew dataset commit file map."""
    return _prepare_hebrew_dataset_publication(
        config_path=config_path,
        bundle_dir=bundle_dir,
        root_rows_path=root_rows_path,
        committed_config_bytes=None,
    )


def _prepare_hebrew_dataset_publication(
    *,
    config_path: Path,
    bundle_dir: Path,
    root_rows_path: Path,
    committed_config_bytes: bytes | None,
) -> PreparedPublication:
    """Prepare a bundle, optionally using proof obtained before snapshotting."""
    observed = _validate_exact_tree(bundle_dir, required=DATASET_REQUIRED_FILES)
    _assert_no_raw_provider_journal(bundle_dir)
    _assert_no_secret_text(bundle_dir, sorted(observed))
    _validate_dataset_card(bundle_dir / "README.md")

    phase_a_config_path = bundle_dir / TRANSLATION_CONFIG_FILENAME
    if phase_a_config_path.read_bytes() != config_path.read_bytes():
        raise UserInputError(
            "translation_config.yaml is not the exact Phase-A config supplied for publication",
            hint="Copy the immutable translation run's config.yaml bytes without reserializing.",
        )
    phase_a_config = load_config(config_path)
    from sommelier.evaluation.data_provenance import validate_hebrew_v3_translation_config

    validate_hebrew_v3_translation_config(phase_a_config)
    source_revision = _phase_a_source_revision(bundle_dir)
    if committed_config_bytes is None:
        committed_config_bytes = require_committed_config_bytes(
            config_path,
            code_revision=source_revision,
            context="Hebrew v3 dataset publication",
        )
    if committed_config_bytes != config_path.read_bytes():
        raise UserInputError(
            "Hebrew v3 publication snapshot does not match the committed Phase-A config"
        )
    config = _prepublication_hebrew_config(phase_a_config)
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
            TRANSLATION_CONFIG_FILENAME,
            TRANSLATION_RUN_IDENTITY_FILENAME,
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


def _experiment_report_error(message: str) -> UserInputError:
    return UserInputError(
        f"final experiment report {message}",
        hint=(
            "Regenerate experiment_report.json with `sommelier report experiment`; "
            "do not hand-edit release evidence or claim decisions."
        ),
    )


def _closed_experiment_mapping(
    value: object,
    *,
    fields: frozenset[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _experiment_report_error(f"{context} must be a JSON object")
    payload = cast("dict[str, Any]", value)
    if set(payload) != fields:
        raise _experiment_report_error(f"{context} has unexpected or missing fields")
    return payload


def _experiment_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _experiment_report_error(f"{context} must be a finite number")
    return float(value)


def _experiment_integer(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _experiment_report_error(f"{context} must be an integer >= {minimum}")
    return value


def _experiment_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _experiment_report_error(f"{context} must be a lowercase SHA-256 digest")
    return value


def _matches_v3_adapter_source(
    value: object,
    *,
    run_id: str,
    tree_sha256: str,
) -> bool:
    if not isinstance(value, dict):
        return False
    expected_path = f"runs/{run_id}/train/adapter"
    source = value.get("source")
    portable_source = source.replace("\\", "/").rstrip("/") if isinstance(source, str) else ""
    return (
        set(value)
        == {
            "source",
            "revision",
            "kind",
            "tree_sha256",
            "artifact_path",
            "revision_is_immutable",
        }
        and (portable_source == expected_path or portable_source.endswith(f"/{expected_path}"))
        and value.get("revision") is None
        and value.get("kind") == "local_directory"
        and value.get("tree_sha256") == tree_sha256
        and value.get("artifact_path") == expected_path
        and value.get("revision_is_immutable") is True
    )


def _validate_metric_map(value: object, *, context: str) -> dict[str, float]:
    metrics = _closed_experiment_mapping(
        value,
        fields=frozenset(METRIC_NAMES),
        context=context,
    )
    values: dict[str, float] = {}
    for name in METRIC_NAMES:
        metric = _closed_experiment_mapping(
            metrics[name],
            fields=frozenset({"value", "numerator", "denominator"}),
            context=f"{context}.{name}",
        )
        numerator = _experiment_integer(metric["numerator"], context=f"{context}.{name}.numerator")
        denominator = _experiment_integer(
            metric["denominator"], context=f"{context}.{name}.denominator"
        )
        if numerator > denominator:
            raise _experiment_report_error(f"{context}.{name} numerator exceeds denominator")
        observed = _experiment_number(metric["value"], context=f"{context}.{name}.value")
        expected = numerator / denominator if denominator else 0.0
        if observed != expected:
            raise _experiment_report_error(
                f"{context}.{name} value does not match numerator/denominator"
            )
        values[name] = observed
    if metrics["full_call_exact_match"]["denominator"] == 0:
        raise _experiment_report_error(f"{context} has an empty full-call cohort")
    return values


def _validate_paired_slice_payload(
    value: object,
    *,
    arm_name: str,
) -> dict[str, Any]:
    context = f"arms.{arm_name}.paired_slices.he"
    payload = _closed_experiment_mapping(
        value,
        fields=frozenset(
            {
                "reference_language",
                "target_language",
                "pairs",
                "coverage",
                "pair_set_sha256",
                "reference",
                "target",
                "gaps",
                "gap_ci95",
            }
        ),
        context=context,
    )
    pairs = _experiment_integer(payload["pairs"], context=f"{context}.pairs", minimum=1)
    if pairs > _HEBREW_V3_MAX_TEST_EXAMPLES:
        raise _experiment_report_error(f"{context}.pairs exceeds the release cohort ceiling")
    _experiment_sha256(payload["pair_set_sha256"], context=f"{context}.pair_set_sha256")
    coverage = _closed_experiment_mapping(
        payload["coverage"],
        fields=frozenset(
            {
                "paired",
                "reference_slice_examples",
                "target_slice_examples",
                "reference_fraction",
            }
        ),
        context=f"{context}.coverage",
    )
    paired = _experiment_integer(
        coverage["paired"], context=f"{context}.coverage.paired", minimum=1
    )
    reference_examples = _experiment_integer(
        coverage["reference_slice_examples"],
        context=f"{context}.coverage.reference_slice_examples",
        minimum=1,
    )
    target_examples = _experiment_integer(
        coverage["target_slice_examples"],
        context=f"{context}.coverage.target_slice_examples",
        minimum=1,
    )
    reference_fraction = _experiment_number(
        coverage["reference_fraction"],
        context=f"{context}.coverage.reference_fraction",
    )
    if (
        payload["reference_language"] != "en"
        or payload["target_language"] != "he"
        or pairs != paired
        or pairs != target_examples
        or pairs > reference_examples
        or reference_fraction != pairs / reference_examples
    ):
        raise _experiment_report_error(f"{context} cohort identity or coverage is inconsistent")
    reference = _closed_experiment_mapping(
        payload["reference"],
        fields=frozenset({"metrics"}),
        context=f"{context}.reference",
    )
    target = _closed_experiment_mapping(
        payload["target"],
        fields=frozenset({"metrics"}),
        context=f"{context}.target",
    )
    reference_metrics = _validate_metric_map(
        reference["metrics"], context=f"{context}.reference.metrics"
    )
    target_metrics = _validate_metric_map(target["metrics"], context=f"{context}.target.metrics")
    for cohort_name, cohort in (("reference", reference), ("target", target)):
        cohort_metrics = cast("dict[str, Any]", cohort["metrics"])
        if any(
            cast("dict[str, Any]", cohort_metrics[metric_name])["denominator"] != pairs
            for metric_name in (
                "valid_json_rate",
                "function_name_accuracy",
                "argument_exact_match",
                "full_call_exact_match",
            )
        ):
            raise _experiment_report_error(
                f"{context}.{cohort_name}.metrics per-example denominators "
                "do not match the pair count"
            )
    gaps = _closed_experiment_mapping(
        payload["gaps"],
        fields=frozenset(METRIC_NAMES),
        context=f"{context}.gaps",
    )
    for metric_name in METRIC_NAMES:
        gap = _experiment_number(gaps[metric_name], context=f"{context}.gaps.{metric_name}")
        if gap != target_metrics[metric_name] - reference_metrics[metric_name]:
            raise _experiment_report_error(
                f"{context}.gaps.{metric_name} is not Hebrew minus matched English"
            )
    interval = _closed_experiment_mapping(
        payload["gap_ci95"],
        fields=frozenset({"method", "seed", "confidence_level", "resamples", "intervals"}),
        context=f"{context}.gap_ci95",
    )
    expected_seed = stable_bootstrap_seed(HEBREW_V3_BOOTSTRAP_SEED, "language-gap:en:he")
    if (
        interval["method"] != PAIRED_BOOTSTRAP_VERSION
        or _experiment_integer(interval["seed"], context=f"{context}.gap_ci95.seed")
        != expected_seed
        or _experiment_integer(
            interval["resamples"], context=f"{context}.gap_ci95.resamples", minimum=1
        )
        != HEBREW_V3_BOOTSTRAP_RESAMPLES
        or _experiment_number(
            interval["confidence_level"], context=f"{context}.gap_ci95.confidence_level"
        )
        != 0.95
    ):
        raise _experiment_report_error(f"{context}.gap_ci95 bootstrap contract drifted")
    intervals = _closed_experiment_mapping(
        interval["intervals"],
        fields=frozenset(METRIC_NAMES),
        context=f"{context}.gap_ci95.intervals",
    )
    for metric_name in METRIC_NAMES:
        bounds = _closed_experiment_mapping(
            intervals[metric_name],
            fields=frozenset({"lower", "upper"}),
            context=f"{context}.gap_ci95.intervals.{metric_name}",
        )
        lower = _experiment_number(
            bounds["lower"], context=f"{context}.gap_ci95.intervals.{metric_name}.lower"
        )
        upper = _experiment_number(
            bounds["upper"], context=f"{context}.gap_ci95.intervals.{metric_name}.upper"
        )
        if lower > upper or lower < -1.0 or upper > 1.0:
            raise _experiment_report_error(
                f"{context}.gap_ci95.intervals.{metric_name} is outside the metric-delta range"
            )
    return payload


def _validate_arm_payload(
    value: object,
    *,
    name: str,
    expected_kind: str,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    arm = _closed_experiment_mapping(
        value,
        fields=frozenset(
            {
                "run_id",
                "model_kind",
                "config_sha256",
                "adapter_source",
                "metrics",
                "paired_slices",
                "artifacts",
            }
        ),
        context=f"arms.{name}",
    )
    if not isinstance(arm["run_id"], str) or not arm["run_id"]:
        raise _experiment_report_error(f"arms.{name}.run_id must be a non-empty string")
    if arm["model_kind"] != expected_kind:
        raise _experiment_report_error(f"arms.{name}.model_kind is not {expected_kind}")
    _experiment_sha256(arm["config_sha256"], context=f"arms.{name}.config_sha256")
    metrics = _closed_experiment_mapping(
        arm["metrics"],
        fields=frozenset({"overall", "slices"}),
        context=f"arms.{name}.metrics",
    )
    _validate_metric_map(metrics["overall"], context=f"arms.{name}.metrics.overall")
    slices = _closed_experiment_mapping(
        metrics["slices"],
        fields=frozenset({"en", "he"}),
        context=f"arms.{name}.metrics.slices",
    )
    slice_metrics = {
        language: _validate_metric_map(
            slices[language], context=f"arms.{name}.metrics.slices.{language}"
        )
        for language in ("en", "he")
    }
    paired_slices = _closed_experiment_mapping(
        arm["paired_slices"],
        fields=frozenset({"he"}),
        context=f"arms.{name}.paired_slices",
    )
    paired_hebrew = _validate_paired_slice_payload(paired_slices["he"], arm_name=name)
    paired_target = cast("dict[str, Any]", paired_hebrew["target"])
    if paired_target["metrics"] != slices["he"]:
        raise _experiment_report_error(
            f"arms.{name}.paired_slices.he target metrics differ from the Hebrew slice"
        )
    artifacts = _closed_experiment_mapping(
        arm["artifacts"],
        fields=frozenset(
            {"evaluation_report", "formatted_test", "generations.en", "generations.he"}
        ),
        context=f"arms.{name}.artifacts",
    )
    for artifact_name, raw_artifact in artifacts.items():
        artifact = _closed_experiment_mapping(
            raw_artifact,
            fields=frozenset({"path", "sha256"}),
            context=f"arms.{name}.artifacts.{artifact_name}",
        )
        path = artifact["path"]
        if (
            not isinstance(path, str)
            or not path.startswith("runs/")
            or path.startswith("/")
            or ".." in path.split("/")
        ):
            raise _experiment_report_error(
                f"arms.{name}.artifacts.{artifact_name}.path is not a portable run path"
            )
        _experiment_sha256(
            artifact["sha256"], context=f"arms.{name}.artifacts.{artifact_name}.sha256"
        )
    return arm, slice_metrics


def _validate_experiment_arms(
    report: Mapping[str, Any],
    *,
    run_id: str,
    config_sha256: str,
    tree_sha256: str,
) -> dict[str, dict[str, dict[str, float]]]:
    arms = _closed_experiment_mapping(
        report.get("arms"),
        fields=frozenset({"base", "v1_en", "v3_en_he"}),
        context="arms",
    )
    loaded: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name, kind in (("base", "base"), ("v1_en", "adapter"), ("v3_en_he", "adapter")):
        loaded[name], metrics[name] = _validate_arm_payload(
            arms[name], name=name, expected_kind=kind
        )

    shared = cast("dict[str, Any]", report["shared_evaluation_identity"])
    shared_slices = cast("dict[str, Any]", shared["slices"])
    shared_hebrew_pair = cast("dict[str, Any]", shared["paired_cohorts"])["he"]
    expected_english_examples = cast("dict[str, Any]", shared_slices["en"])["examples"]
    expected_hebrew_examples = cast("dict[str, Any]", shared_slices["he"])["examples"]
    for name, arm in loaded.items():
        paired_hebrew = cast("dict[str, Any]", arm["paired_slices"])["he"]
        coverage = cast("dict[str, Any]", paired_hebrew["coverage"])
        if (
            paired_hebrew["pairs"] != expected_hebrew_examples
            or paired_hebrew["pair_set_sha256"] != shared_hebrew_pair["pair_set_sha256"]
            or coverage["paired"] != expected_hebrew_examples
            or coverage["target_slice_examples"] != expected_hebrew_examples
            or coverage["reference_slice_examples"] != expected_english_examples
        ):
            raise _experiment_report_error(
                f"arms.{name}.paired_slices.he disagrees with shared evaluation identity"
            )

    if loaded["base"]["adapter_source"] is not None:
        raise _experiment_report_error("arms.base.adapter_source must be null")
    expected_v1_source = {
        "source": HEBREW_V3_V1_ADAPTER_ID,
        "revision": HEBREW_V3_V1_ADAPTER_REVISION,
        "kind": "huggingface_repo",
        "tree_sha256": None,
        "artifact_path": None,
        "revision_is_immutable": True,
    }
    if loaded["v1_en"]["adapter_source"] != expected_v1_source:
        raise _experiment_report_error("arms.v1_en does not bind the immutable v1 adapter")
    if (
        loaded["v3_en_he"]["run_id"] != run_id
        or loaded["v3_en_he"]["config_sha256"] != config_sha256
        or not _matches_v3_adapter_source(
            loaded["v3_en_he"]["adapter_source"],
            run_id=run_id,
            tree_sha256=tree_sha256,
        )
    ):
        raise _experiment_report_error("arms.v3_en_he does not bind the training run/config/tree")
    return metrics


def _validate_mcnemar(value: object, *, language: str) -> None:
    payload = _closed_experiment_mapping(
        value,
        fields=frozenset(
            {
                "method",
                "metric",
                "alternative",
                "pairs",
                "discordant_pairs",
                "discordant_counts",
                "p_value",
            }
        ),
        context=f"comparisons.v3_vs_v1.{language}.mcnemar",
    )
    if (
        payload["method"] != EXACT_MCNEMAR_VERSION
        or payload["metric"] != "full_call_exact_match"
        or payload["alternative"] != "two-sided"
    ):
        raise _experiment_report_error(f"{language} McNemar contract drifted")
    pairs = _experiment_integer(payload["pairs"], context=f"{language} McNemar pairs", minimum=1)
    if pairs > _HEBREW_V3_MAX_TEST_EXAMPLES:
        raise _experiment_report_error(
            f"{language} McNemar pairs exceeds the "
            f"{_HEBREW_V3_MAX_TEST_EXAMPLES}-example release cohort ceiling"
        )
    discordant = _experiment_integer(
        payload["discordant_pairs"], context=f"{language} McNemar discordant_pairs"
    )
    counts = _closed_experiment_mapping(
        payload["discordant_counts"],
        fields=frozenset(
            {
                "reference_correct_candidate_incorrect",
                "reference_incorrect_candidate_correct",
            }
        ),
        context=f"comparisons.v3_vs_v1.{language}.mcnemar.discordant_counts",
    )
    first = _experiment_integer(
        counts["reference_correct_candidate_incorrect"],
        context=f"{language} McNemar reference-only count",
    )
    second = _experiment_integer(
        counts["reference_incorrect_candidate_correct"],
        context=f"{language} McNemar candidate-only count",
    )
    p_value = _experiment_number(payload["p_value"], context=f"{language} McNemar p_value")
    if discordant != first + second or discordant > pairs or not 0.0 <= p_value <= 1.0:
        raise _experiment_report_error(f"{language} McNemar counts or p-value are inconsistent")
    expected_p_value = (
        1.0
        if discordant == 0
        else min(
            1.0,
            2.0
            * sum(math.comb(discordant, successes) for successes in range(min(first, second) + 1))
            / (2**discordant),
        )
    )
    if p_value != expected_p_value:
        raise _experiment_report_error(f"{language} McNemar p-value does not match its counts")


def _validate_experiment_comparisons(
    report: Mapping[str, Any],
    *,
    arm_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    comparisons = _closed_experiment_mapping(
        report.get("comparisons"),
        fields=frozenset({"v3_vs_v1"}),
        context="comparisons",
    )
    by_language = _closed_experiment_mapping(
        comparisons["v3_vs_v1"],
        fields=frozenset({"en", "he"}),
        context="comparisons.v3_vs_v1",
    )
    primary: dict[str, dict[str, float]] = {}
    for offset, language in enumerate(("en", "he")):
        comparison = _closed_experiment_mapping(
            by_language[language],
            fields=frozenset({"deltas", "ci95", "mcnemar"}),
            context=f"comparisons.v3_vs_v1.{language}",
        )
        deltas = _closed_experiment_mapping(
            comparison["deltas"],
            fields=frozenset(METRIC_NAMES),
            context=f"comparisons.v3_vs_v1.{language}.deltas",
        )
        for metric_name in METRIC_NAMES:
            observed = _experiment_number(
                deltas[metric_name], context=f"{language} {metric_name} delta"
            )
            expected = (
                arm_metrics["v3_en_he"][language][metric_name]
                - arm_metrics["v1_en"][language][metric_name]
            )
            if observed != expected:
                raise _experiment_report_error(
                    f"{language} {metric_name} delta does not match arm metrics"
                )
        bootstrap = _closed_experiment_mapping(
            comparison["ci95"],
            fields=frozenset({"method", "seed", "confidence_level", "resamples", "intervals"}),
            context=f"comparisons.v3_vs_v1.{language}.ci95",
        )
        expected_bootstrap = {
            "method": PAIRED_BOOTSTRAP_VERSION,
            "seed": HEBREW_V3_BOOTSTRAP_SEED + offset,
            "confidence_level": 0.95,
            "resamples": HEBREW_V3_BOOTSTRAP_RESAMPLES,
        }
        if any(bootstrap[field] != expected for field, expected in expected_bootstrap.items()):
            raise _experiment_report_error(f"{language} paired-bootstrap contract drifted")
        intervals = _closed_experiment_mapping(
            bootstrap["intervals"],
            fields=frozenset(METRIC_NAMES),
            context=f"comparisons.v3_vs_v1.{language}.ci95.intervals",
        )
        for metric_name in METRIC_NAMES:
            interval = _closed_experiment_mapping(
                intervals[metric_name],
                fields=frozenset({"lower", "upper"}),
                context=f"{language} {metric_name} interval",
            )
            lower = _experiment_number(
                interval["lower"], context=f"{language} {metric_name} interval lower"
            )
            upper = _experiment_number(
                interval["upper"], context=f"{language} {metric_name} interval upper"
            )
            if lower > upper or lower < -1.0 or upper > 1.0:
                raise _experiment_report_error(
                    f"{language} {metric_name} interval is outside the metric-delta range"
                )
            if metric_name == "full_call_exact_match":
                primary[language] = {
                    "estimate": cast(float, deltas[metric_name]),
                    "lower": lower,
                    "upper": upper,
                }
        _validate_mcnemar(comparison["mcnemar"], language=language)
    return primary


def _validate_shared_evaluation_identity(report: Mapping[str, Any]) -> None:
    shared = _closed_experiment_mapping(
        report.get("shared_evaluation_identity"),
        fields=frozenset(
            {
                "model_identity",
                "split",
                "parser_version",
                "decoding",
                "test_split_sha256",
                "slices",
                "paired_cohorts",
            }
        ),
        context="shared_evaluation_identity",
    )
    model = _closed_experiment_mapping(
        shared["model_identity"],
        fields=frozenset(
            {"base_model_id", "base_model_revision", "tokenizer_id", "tokenizer_revision"}
        ),
        context="shared_evaluation_identity.model_identity",
    )
    if model != {
        "base_model_id": HEBREW_V3_BASE_MODEL_ID,
        "base_model_revision": HEBREW_V3_BASE_MODEL_REVISION,
        "tokenizer_id": HEBREW_V3_BASE_MODEL_ID,
        "tokenizer_revision": HEBREW_V3_BASE_MODEL_REVISION,
    }:
        raise _experiment_report_error("shared model/tokenizer identity drifted")
    if shared["split"] != "test" or shared["parser_version"] != "sommelier.parser.v1":
        raise _experiment_report_error("shared split or parser identity drifted")
    decoding = _closed_experiment_mapping(
        shared["decoding"],
        fields=frozenset({"temperature", "do_sample", "max_new_tokens"}),
        context="shared_evaluation_identity.decoding",
    )
    if (
        decoding["temperature"] != 0.0
        or decoding["do_sample"] is not False
        or _experiment_integer(
            decoding["max_new_tokens"], context="shared max_new_tokens", minimum=1
        )
        <= 0
    ):
        raise _experiment_report_error("shared deterministic decoding contract drifted")
    _experiment_sha256(
        shared["test_split_sha256"], context="shared_evaluation_identity.test_split_sha256"
    )
    slices = _closed_experiment_mapping(
        shared["slices"],
        fields=frozenset({"en", "he"}),
        context="shared_evaluation_identity.slices",
    )
    counts: dict[str, int] = {}
    for language in ("en", "he"):
        payload = _closed_experiment_mapping(
            slices[language],
            fields=frozenset({"examples", "example_ids_sha256", "prompt_set_sha256"}),
            context=f"shared_evaluation_identity.slices.{language}",
        )
        counts[language] = _experiment_integer(
            payload["examples"], context=f"shared {language} examples", minimum=1
        )
        if counts[language] > _HEBREW_V3_MAX_TEST_EXAMPLES:
            raise _experiment_report_error(
                f"shared {language} examples exceeds the "
                f"{_HEBREW_V3_MAX_TEST_EXAMPLES}-example release cohort ceiling"
            )
        _experiment_sha256(payload["example_ids_sha256"], context=f"shared {language} example IDs")
        _experiment_sha256(payload["prompt_set_sha256"], context=f"shared {language} prompt set")
    paired = _closed_experiment_mapping(
        shared["paired_cohorts"],
        fields=frozenset({"he"}),
        context="shared_evaluation_identity.paired_cohorts",
    )
    hebrew = _closed_experiment_mapping(
        paired["he"],
        fields=frozenset({"pairs", "pair_set_sha256", "reference_row_indices_sha256"}),
        context="shared_evaluation_identity.paired_cohorts.he",
    )
    if (
        _experiment_integer(hebrew["pairs"], context="shared Hebrew pairs", minimum=1)
        != counts["he"]
    ):
        raise _experiment_report_error("shared Hebrew pair count disagrees with its slice")
    _experiment_sha256(hebrew["pair_set_sha256"], context="shared Hebrew pair set")
    _experiment_sha256(
        hebrew["reference_row_indices_sha256"],
        context="shared Hebrew reference row indices",
    )


def _validate_experiment_data_provenance(
    report: Mapping[str, Any],
    *,
    run_id: str,
    config_sha256: str,
    source_revision: str,
    dataset_revision: str,
) -> dict[str, Any]:
    provenance = _closed_experiment_mapping(
        report.get("data_provenance"),
        fields=frozenset({"schema_version", "contract", "sources"}),
        context="data_provenance",
    )
    if provenance["schema_version"] != DATA_PROVENANCE_SCHEMA:
        raise _experiment_report_error("data_provenance schema drifted")
    contract = _closed_experiment_mapping(
        provenance["contract"],
        fields=frozenset(
            {
                "seed",
                "root_dataset",
                "paired_dataset",
                "requested_splits",
                "observed_cohorts",
                "semantic_review",
                "source_code_revision",
            }
        ),
        context="data_provenance.contract",
    )
    root_dataset = _closed_experiment_mapping(
        contract["root_dataset"],
        fields=frozenset({"dataset_id", "dataset_revision"}),
        context="data_provenance.contract.root_dataset",
    )
    paired_dataset = _closed_experiment_mapping(
        contract["paired_dataset"],
        fields=frozenset({"dataset_id", "dataset_revision"}),
        context="data_provenance.contract.paired_dataset",
    )
    seed = _experiment_integer(
        contract["seed"],
        context="data_provenance seed",
    )
    if (
        seed != HEBREW_V3_BOOTSTRAP_SEED
        or root_dataset
        != {
            "dataset_id": HEBREW_V3_ROOT_DATASET_ID,
            "dataset_revision": HEBREW_V3_ROOT_DATASET_REVISION,
        }
        or paired_dataset
        != {"dataset_id": HEBREW_V3_PAIRED_DATASET_ID, "dataset_revision": dataset_revision}
        or contract["source_code_revision"] != source_revision
    ):
        raise _experiment_report_error("data_provenance contract identity drifted")
    requested = _closed_experiment_mapping(
        contract["requested_splits"],
        fields=frozenset({"train", "validation", "test"}),
        context="data_provenance.contract.requested_splits",
    )
    expected_requested = {"train": 15_000, "validation": 1_000, "test": 1_000}
    for split, expected_count in expected_requested.items():
        observed_count = _experiment_integer(
            requested[split],
            context=f"data_provenance requested {split} rows",
            minimum=1,
        )
        if observed_count != expected_count:
            raise _experiment_report_error("data_provenance requested cohorts drifted")
    observed = _closed_experiment_mapping(
        contract["observed_cohorts"],
        fields=frozenset({"train", "validation", "test"}),
        context="data_provenance.contract.observed_cohorts",
    )
    for split, expected_english in expected_requested.items():
        cohort = _closed_experiment_mapping(
            observed[split],
            fields=frozenset({"en", "he", "total"}),
            context=f"data_provenance.contract.observed_cohorts.{split}",
        )
        english = _experiment_integer(cohort["en"], context=f"{split} English rows", minimum=1)
        hebrew = _experiment_integer(cohort["he"], context=f"{split} Hebrew rows", minimum=1)
        total = _experiment_integer(cohort["total"], context=f"{split} total rows", minimum=1)
        if english != expected_english or total != english + hebrew:
            raise _experiment_report_error(f"data_provenance {split} cohort is inconsistent")
    review = _closed_experiment_mapping(
        contract["semantic_review"],
        fields=frozenset({"sample_size", "required_critical_errors", "status"}),
        context="data_provenance.contract.semantic_review",
    )
    sample_size = _experiment_integer(
        review["sample_size"],
        context="data_provenance semantic-review sample_size",
        minimum=1,
    )
    required_critical_errors = _experiment_integer(
        review["required_critical_errors"],
        context="data_provenance semantic-review required_critical_errors",
    )
    if sample_size != 200 or required_critical_errors != 0 or review["status"] != "validated":
        raise _experiment_report_error("data_provenance semantic review gate drifted")
    run = f"runs/{run_id}"
    source_specs: dict[str, tuple[str, str, str, str | None]] = {
        "resolved_config": (
            f"{run}/config.resolved.yaml",
            "config",
            "sommelier.config.v2",
            config_sha256,
        ),
        "runtime_metadata": (
            f"{run}/runtime_metadata.json",
            "runtime_metadata",
            "sommelier.runtime_metadata.v1",
            None,
        ),
        "root_manifest": (
            f"{run}/manifest.json",
            "manifest",
            "sommelier.manifest.v1",
            None,
        ),
        "data_manifest": (
            f"{run}/data_manifest.json",
            "manifest",
            "sommelier.manifest.v1",
            None,
        ),
        "format_manifest": (
            f"{run}/format_manifest.json",
            "manifest",
            "sommelier.manifest.v1",
            None,
        ),
        "tokenization_manifest": (
            f"{run}/tokenization_manifest.json",
            "manifest",
            "sommelier.manifest.v1",
            None,
        ),
        "eval_adapter_manifest": (
            f"{run}/eval-adapter_manifest.json",
            "manifest",
            "sommelier.manifest.v1",
            None,
        ),
        "root_rows": (
            f"{run}/data/source_inputs/rows.en.jsonl",
            "raw_dataset",
            "sommelier.raw_tool_call_row.v1",
            None,
        ),
        "he_paired_rows": (
            f"{run}/data/source_inputs/rows.en.he.jsonl",
            "raw_paired_dataset",
            "sommelier.raw_tool_call_row.v1",
            None,
        ),
        "he_translation_summary": (
            f"{run}/data/source_inputs/translation_summary.he.json",
            "translation_summary",
            TRANSLATION_SUMMARY_SCHEMA,
            None,
        ),
        "he_translation_publication": (
            f"{run}/data/source_inputs/translation_publication.he.json",
            "translation_publication_manifest",
            TRANSLATION_PUBLICATION_SCHEMA,
            None,
        ),
        "he_semantic_review_template": (
            f"{run}/data/source_inputs/translation_semantic_review_template.he.json",
            "translation_semantic_review_template",
            SEMANTIC_REVIEW_TEMPLATE_SCHEMA,
            None,
        ),
        "he_semantic_review": (
            f"{run}/data/source_inputs/translation_semantic_review.he.json",
            "translation_semantic_review",
            SEMANTIC_REVIEW_SCHEMA,
            None,
        ),
        "he_translation_config": (
            f"{run}/data/source_inputs/translation_config.he.yaml",
            "config",
            "sommelier.config.v2",
            None,
        ),
        "he_translation_run_identity": (
            f"{run}/data/source_inputs/translation_run_identity.he.json",
            "translation_run_identity",
            TRANSLATION_RUN_IDENTITY_SCHEMA,
            None,
        ),
        "prepared_train": (
            f"{run}/data/train.jsonl",
            "dataset_split",
            "sommelier.prepared_example.v2",
            None,
        ),
        "prepared_validation": (
            f"{run}/data/validation.jsonl",
            "dataset_split",
            "sommelier.prepared_example.v2",
            None,
        ),
        "prepared_test": (
            f"{run}/data/test.jsonl",
            "dataset_split",
            "sommelier.prepared_example.v2",
            None,
        ),
        "prepared_drop_summary": (
            f"{run}/data/drop_summary.json",
            "drop_summary",
            "sommelier.drop_summary.v2",
            None,
        ),
        "formatted_train": (
            f"{run}/formatted/train.jsonl",
            "formatted_split",
            "sommelier.formatted_example.v2",
            None,
        ),
        "formatted_validation": (
            f"{run}/formatted/validation.jsonl",
            "formatted_split",
            "sommelier.formatted_example.v2",
            None,
        ),
        "formatted_test": (
            f"{run}/formatted/test.jsonl",
            "formatted_split",
            "sommelier.formatted_example.v2",
            None,
        ),
        "tokenization_tokenizer_tax_records": (
            f"{run}/analysis/tokenization/tokenizer_tax_records.jsonl",
            "tokenizer_tax_records",
            "sommelier.tokenizer_tax_record.v1",
            None,
        ),
        "tokenization_tokenizer_tax_report": (
            f"{run}/analysis/tokenization/tokenizer_tax_report.json",
            "tokenizer_tax_report",
            "sommelier.tokenizer_tax_report.v1",
            None,
        ),
        "v3_inference_telemetry": (
            f"{run}/eval/adapter/inference_telemetry.json",
            "inference_telemetry",
            "sommelier.inference_telemetry.v2",
            None,
        ),
    }
    sources = _closed_experiment_mapping(
        provenance["sources"],
        fields=frozenset(source_specs),
        context="data_provenance.sources",
    )
    validated_sources: dict[str, Any] = {}
    for name, (path, kind, schema, digest) in source_specs.items():
        validated_sources[name] = _validate_tco_artifact_source(
            sources[name],
            context=f"data_provenance.sources.{name}",
            expected_path=path,
            expected_kind=kind,
            expected_schema=schema,
            expected_sha256=digest,
        )
    return validated_sources


def _validate_experiment_preregistration(
    report: Mapping[str, Any], *, source_revision: str
) -> None:
    created_at = report.get("created_at")
    if not isinstance(created_at, str):
        raise _experiment_report_error("created_at must be a timestamp string")
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError as error:
        raise _experiment_report_error("created_at is not an ISO-8601 timestamp") from error
    if created.tzinfo is None:
        raise _experiment_report_error("created_at must include a timezone")

    preregistration = _closed_experiment_mapping(
        report.get("preregistration"),
        fields=frozenset(
            {
                "schema_version",
                "status",
                "english_non_inferiority_margin",
                "bootstrap",
                "primary_claim_rules",
                "finalizer_source_code",
            }
        ),
        context="preregistration",
    )
    if (
        preregistration["schema_version"] != HEBREW_V3_PREREGISTRATION_SCHEMA
        or preregistration["status"] != "committed_in_source_before_full_results"
        or preregistration["english_non_inferiority_margin"]
        != HEBREW_V3_ENGLISH_NON_INFERIORITY_MARGIN
    ):
        raise _experiment_report_error("preregistration identity or margin drifted")
    prereg_bootstrap = _closed_experiment_mapping(
        preregistration["bootstrap"],
        fields=frozenset({"seed", "resamples", "confidence_level", "method"}),
        context="preregistration.bootstrap",
    )
    if prereg_bootstrap != {
        "seed": HEBREW_V3_BOOTSTRAP_SEED,
        "resamples": HEBREW_V3_BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
        "method": PAIRED_BOOTSTRAP_VERSION,
    }:
        raise _experiment_report_error("preregistered bootstrap contract drifted")
    rules = _closed_experiment_mapping(
        preregistration["primary_claim_rules"],
        fields=frozenset(_CLAIM_ORDER),
        context="preregistration.primary_claim_rules",
    )
    if rules != {
        "hebrew_full_call_uplift": "95% lower bound > 0",
        "english_full_call_non_inferiority": "95% lower bound >= -0.01",
    }:
        raise _experiment_report_error("primary claim rules drifted")
    finalizer = _closed_experiment_mapping(
        preregistration["finalizer_source_code"],
        fields=frozenset({"git_commit", "working_tree_clean", "boundary"}),
        context="preregistration.finalizer_source_code",
    )
    if finalizer != {
        "git_commit": source_revision,
        "working_tree_clean": True,
        "boundary": "Observed before loading experiment outcome artifacts.",
    }:
        raise _experiment_report_error("finalizer source does not match the clean training source")

    bootstrap = _closed_experiment_mapping(
        report.get("bootstrap"),
        fields=frozenset({"seed", "resamples", "confidence_level"}),
        context="bootstrap",
    )
    if bootstrap != {
        "seed": HEBREW_V3_BOOTSTRAP_SEED,
        "resamples": HEBREW_V3_BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
    }:
        raise _experiment_report_error("bootstrap summary drifted from preregistration")


def _validate_experiment_claims(
    report: Mapping[str, Any],
    *,
    primary: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    claims = _closed_experiment_mapping(
        report.get("claims"),
        fields=frozenset(_CLAIM_ORDER),
        context="claims",
    )
    definitions = {
        "hebrew_full_call_uplift": {
            "language": "he",
            "criterion": "95% paired-bootstrap lower bound > 0",
            "statement": _HEBREW_UPLIFT_STATEMENT,
            "passed": primary["he"]["lower"] > 0.0,
            "margin": None,
        },
        "english_full_call_non_inferiority": {
            "language": "en",
            "criterion": "95% paired-bootstrap lower bound >= -margin",
            "statement": _ENGLISH_NON_INFERIORITY_STATEMENT,
            "passed": (primary["en"]["lower"] >= -HEBREW_V3_ENGLISH_NON_INFERIORITY_MARGIN),
            "margin": HEBREW_V3_ENGLISH_NON_INFERIORITY_MARGIN,
        },
    }
    validated: dict[str, dict[str, Any]] = {}
    approved: list[str] = []
    for name in _CLAIM_ORDER:
        definition = definitions[name]
        passed = cast(bool, definition["passed"])
        fields = {"passed", "metric", "estimate", "ci95", "criterion"}
        if definition["margin"] is not None:
            fields.add("margin")
        if passed:
            fields.add("statement")
        claim = _closed_experiment_mapping(
            claims[name], fields=frozenset(fields), context=f"claims.{name}"
        )
        language = cast(str, definition["language"])
        interval = _closed_experiment_mapping(
            claim["ci95"],
            fields=frozenset({"lower", "upper"}),
            context=f"claims.{name}.ci95",
        )
        expected_interval = {
            "lower": primary[language]["lower"],
            "upper": primary[language]["upper"],
        }
        if (
            claim["passed"] is not passed
            or claim["metric"] != "full_call_exact_match"
            or claim["estimate"] != primary[language]["estimate"]
            or interval != expected_interval
            or claim["criterion"] != definition["criterion"]
        ):
            raise _experiment_report_error(f"claims.{name} disagrees with the derived gate")
        margin = definition["margin"]
        if margin is not None and claim["margin"] != margin:
            raise _experiment_report_error(f"claims.{name}.margin drifted")
        statement = cast(str, definition["statement"])
        if passed:
            if claim["statement"] != statement:
                raise _experiment_report_error(f"claims.{name}.statement drifted")
            approved.append(statement)
        validated[name] = claim

    expected_all_passed = all(cast(bool, definitions[name]["passed"]) for name in _CLAIM_ORDER)
    if report.get("all_claims_passed") is not expected_all_passed:
        raise _experiment_report_error("all_claims_passed disagrees with the derived gates")
    if report.get("approved_claims") != approved:
        raise _experiment_report_error("approved_claims disagrees with the derived gates")
    return validated


def _experiment_number_at_least(
    value: object,
    *,
    context: str,
    minimum: float,
    strictly: bool = False,
) -> float:
    number = _experiment_number(value, context=context)
    invalid = number <= minimum if strictly else number < minimum
    if invalid:
        operator = ">" if strictly else ">="
        raise _experiment_report_error(f"{context} must be {operator} {minimum}")
    return number


def _validate_tco_artifact_source(
    value: object,
    *,
    context: str,
    expected_path: str,
    expected_kind: str,
    expected_schema: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    artifact = _closed_experiment_mapping(
        value,
        fields=frozenset({"path", "kind", "schema_version", "sha256", "bytes"}),
        context=context,
    )
    digest = _experiment_sha256(artifact["sha256"], context=f"{context}.sha256")
    _experiment_integer(artifact["bytes"], context=f"{context}.bytes", minimum=1)
    if (
        artifact["path"] != expected_path
        or artifact["kind"] != expected_kind
        or artifact["schema_version"] != expected_schema
        or (expected_sha256 is not None and digest != expected_sha256)
    ):
        raise _experiment_report_error(f"{context} identity drifted")
    return artifact


def _validate_tco_paired_scopes(
    value: object,
    *,
    observed_cohorts: Mapping[str, Any],
) -> None:
    scopes = _closed_experiment_mapping(
        value,
        fields=frozenset({"all", "train", "validation", "test"}),
        context="TCO paired tokenization scopes",
    )
    split_names = ("train", "validation", "test")
    expected_counts: dict[str, tuple[int, int]] = {}
    for split in split_names:
        cohort = cast("Mapping[str, Any]", observed_cohorts[split])
        expected_counts[split] = (cast(int, cohort["he"]), cast(int, cohort["en"]))
    expected_counts["all"] = (
        sum(expected_counts[split][0] for split in split_names),
        sum(expected_counts[split][1] for split in split_names),
    )

    validated_tokens: dict[str, dict[str, tuple[int, int]]] = {}
    for scope_name in ("all", *split_names):
        scope = _closed_experiment_mapping(
            scopes[scope_name],
            fields=frozenset({"coverage", "token_ratios"}),
            context=f"TCO paired {scope_name}",
        )
        coverage = _closed_experiment_mapping(
            scope["coverage"],
            fields=frozenset({"paired", "roots", "ratio"}),
            context=f"TCO paired {scope_name} coverage",
        )
        paired_count = _experiment_integer(
            coverage["paired"],
            context=f"TCO paired {scope_name} coverage.paired",
            minimum=1,
        )
        root_count = _experiment_integer(
            coverage["roots"],
            context=f"TCO paired {scope_name} coverage.roots",
            minimum=1,
        )
        coverage_ratio = _experiment_number_at_least(
            coverage["ratio"],
            context=f"TCO paired {scope_name} coverage.ratio",
            minimum=0.0,
        )
        expected_paired, expected_roots = expected_counts[scope_name]
        if (
            paired_count != expected_paired
            or root_count != expected_roots
            or paired_count > root_count
            or not math.isclose(
                coverage_ratio,
                paired_count / root_count,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise _experiment_report_error(f"TCO paired {scope_name} coverage is inconsistent")

        ratios = _closed_experiment_mapping(
            scope["token_ratios"],
            fields=frozenset({"query_tokens", "prompt_tokens", "full_tokens"}),
            context=f"TCO paired {scope_name} token ratios",
        )
        scope_tokens: dict[str, tuple[int, int]] = {}
        for field in ("query_tokens", "prompt_tokens", "full_tokens"):
            metric = _closed_experiment_mapping(
                ratios[field],
                fields=frozenset(
                    {
                        "paired_hebrew_tokens",
                        "matched_english_tokens",
                        "hebrew_to_english_ratio",
                    }
                ),
                context=f"TCO paired {scope_name} {field}",
            )
            hebrew_tokens = _experiment_integer(
                metric["paired_hebrew_tokens"],
                context=f"TCO paired {scope_name} {field} Hebrew tokens",
                minimum=1,
            )
            english_tokens = _experiment_integer(
                metric["matched_english_tokens"],
                context=f"TCO paired {scope_name} {field} English tokens",
                minimum=1,
            )
            ratio = _experiment_number_at_least(
                metric["hebrew_to_english_ratio"],
                context=f"TCO paired {scope_name} {field} ratio",
                minimum=0.0,
            )
            if not math.isclose(
                ratio,
                hebrew_tokens / english_tokens,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise _experiment_report_error(
                    f"TCO paired {scope_name} {field} ratio is inconsistent"
                )
            scope_tokens[field] = (hebrew_tokens, english_tokens)
        validated_tokens[scope_name] = scope_tokens

    for field in ("query_tokens", "prompt_tokens", "full_tokens"):
        all_hebrew, all_english = validated_tokens["all"][field]
        if all_hebrew != sum(validated_tokens[split][field][0] for split in split_names) or (
            all_english != sum(validated_tokens[split][field][1] for split in split_names)
        ):
            raise _experiment_report_error(f"TCO paired all {field} does not sum its splits")


def _validate_tco_training(
    value: object,
    *,
    source_revision: str,
    adapter_tree_sha256: str,
) -> dict[str, Any]:
    training = _closed_experiment_mapping(
        value,
        fields=frozenset(
            {
                "available",
                "train_stage_runtime",
                "source_code",
                "remote_execution",
                "tokens_seen",
                "end_to_end_token_throughput",
                "peak_gpu_memory",
                "adapter_storage",
                "currency_cost",
                "full_finetune_savings",
            }
        ),
        context="TCO QLoRA training",
    )
    if training["available"] is not True:
        raise _experiment_report_error("TCO QLoRA training is unavailable")

    runtime = _closed_experiment_mapping(
        training["train_stage_runtime"],
        fields=frozenset(
            {
                "available",
                "elapsed_seconds",
                "boundary",
                "configured_gpu",
                "configured_gpu_hours",
                "configured_gpu_hours_kind",
            }
        ),
        context="TCO train-stage runtime",
    )
    elapsed = _experiment_number_at_least(
        runtime["elapsed_seconds"],
        context="TCO train elapsed_seconds",
        minimum=0.0,
        strictly=True,
    )
    configured_gpu = _closed_experiment_mapping(
        runtime["configured_gpu"],
        fields=frozenset({"label", "count", "source"}),
        context="TCO train configured GPU",
    )
    gpu_count = _experiment_integer(
        configured_gpu["count"], context="TCO train configured GPU count", minimum=1
    )
    gpu_label = configured_gpu["label"]
    gpu_hours = _experiment_number_at_least(
        runtime["configured_gpu_hours"],
        context="TCO configured GPU-hours",
        minimum=0.0,
        strictly=True,
    )
    if (
        runtime["available"] is not True
        or not isinstance(runtime["boundary"], str)
        or not runtime["boundary"]
        or not isinstance(gpu_label, str)
        or not gpu_label
        or configured_gpu["source"] != "resolved config via runtime metadata"
        or runtime["configured_gpu_hours_kind"] != "observed_wall_time_x_configured_gpu_count"
        or not math.isclose(
            gpu_hours,
            elapsed * gpu_count / 3600.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise _experiment_report_error("TCO train-stage runtime is inconsistent")

    source_code = _closed_experiment_mapping(
        training["source_code"],
        fields=frozenset({"available", "git_commit", "working_tree_clean", "boundary", "reason"}),
        context="TCO training source code",
    )
    if (
        source_code["available"] is not True
        or source_code["git_commit"] != source_revision
        or source_code["working_tree_clean"] is not True
        or not isinstance(source_code["boundary"], str)
        or not source_code["boundary"]
        or source_code["reason"] is not None
    ):
        raise _experiment_report_error("TCO training source-code provenance drifted")

    remote = _closed_experiment_mapping(
        training["remote_execution"],
        fields=frozenset(
            {
                "available",
                "provider",
                "function_timeout_seconds",
                "gpu_allocation_label",
                "boundary",
            }
        ),
        context="TCO remote execution",
    )
    _experiment_integer(
        remote["function_timeout_seconds"],
        context="TCO remote function timeout",
        minimum=1,
    )
    if (
        remote["available"] is not True
        or not isinstance(remote["provider"], str)
        or not remote["provider"]
        or remote["gpu_allocation_label"] != gpu_label
        or not isinstance(remote["boundary"], str)
        or not remote["boundary"]
    ):
        raise _experiment_report_error("TCO remote-execution boundary drifted")

    tokens = _closed_experiment_mapping(
        training["tokens_seen"],
        fields=frozenset({"available", "value", "unit", "source_label", "boundary"}),
        context="TCO training tokens seen",
    )
    tokens_seen = _experiment_integer(
        tokens["value"], context="TCO training tokens_seen", minimum=1
    )
    if (
        tokens["available"] is not True
        or tokens["unit"] != "trainer_reported_input_tokens"
        or tokens["source_label"] != "maximum_positive_transformers.num_input_tokens_seen"
        or not isinstance(tokens["boundary"], str)
        or not tokens["boundary"]
    ):
        raise _experiment_report_error("TCO training token evidence drifted")

    throughput = _closed_experiment_mapping(
        training["end_to_end_token_throughput"],
        fields=frozenset({"available", "value", "unit", "boundary"}),
        context="TCO training throughput",
    )
    throughput_value = _experiment_number_at_least(
        throughput["value"],
        context="TCO training throughput value",
        minimum=0.0,
        strictly=True,
    )
    if (
        throughput["available"] is not True
        or throughput["unit"] != "trainer_reported_input_tokens_per_train_stage_second"
        or not isinstance(throughput["boundary"], str)
        or not throughput["boundary"]
        or not math.isclose(
            throughput_value,
            tokens_seen / elapsed,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise _experiment_report_error("TCO training throughput is inconsistent")

    peak = _closed_experiment_mapping(
        training["peak_gpu_memory"],
        fields=frozenset({"available", "value", "unit", "evidence_kind"}),
        context="TCO peak GPU memory",
    )
    _experiment_integer(peak["value"], context="TCO peak GPU memory value", minimum=1)
    if (
        peak["available"] is not True
        or peak["unit"] != "MiB"
        or peak["evidence_kind"] != "observed_peak_allocated_gpu_memory"
    ):
        raise _experiment_report_error("TCO peak GPU memory evidence drifted")

    storage = _closed_experiment_mapping(
        training["adapter_storage"],
        fields=frozenset(
            {
                "available",
                "tree_sha256",
                "packaged_adapter",
                "tensor_weights_only",
                "evidence_kind",
            }
        ),
        context="TCO adapter storage",
    )
    packaged = _closed_experiment_mapping(
        storage["packaged_adapter"],
        fields=frozenset({"bytes", "files", "boundary"}),
        context="TCO packaged adapter",
    )
    tensor = _closed_experiment_mapping(
        storage["tensor_weights_only"],
        fields=frozenset({"bytes", "files", "boundary"}),
        context="TCO tensor-only adapter",
    )
    packaged_bytes = _experiment_integer(
        packaged["bytes"], context="TCO packaged adapter bytes", minimum=1
    )
    packaged_files = _experiment_integer(
        packaged["files"], context="TCO packaged adapter files", minimum=1
    )
    tensor_bytes = _experiment_integer(
        tensor["bytes"], context="TCO tensor-only adapter bytes", minimum=1
    )
    tensor_files = _experiment_integer(
        tensor["files"], context="TCO tensor-only adapter files", minimum=1
    )
    if (
        storage["available"] is not True
        or storage["tree_sha256"] != adapter_tree_sha256
        or storage["evidence_kind"] != "observed_artifact_storage"
        or not isinstance(packaged["boundary"], str)
        or not packaged["boundary"]
        or not isinstance(tensor["boundary"], str)
        or not tensor["boundary"]
        or tensor_bytes > packaged_bytes
        or tensor_files > packaged_files
    ):
        raise _experiment_report_error("TCO adapter storage is inconsistent")

    return {
        "payload": training,
        "gpu_label": gpu_label,
        "gpu_count": gpu_count,
        "packaged_bytes": packaged_bytes,
        "packaged_files": packaged_files,
        "tensor_bytes": tensor_bytes,
        "tensor_files": tensor_files,
    }


def _validate_tco_efficiency_ratio(
    value: object,
    *,
    context: str,
    elapsed_seconds: float,
    gpu_count: int,
    successes: int,
) -> None:
    ratio = _closed_experiment_mapping(
        value,
        fields=frozenset(
            {
                "available",
                "value",
                "reason",
                "unit",
                "full_call_exact_successes",
                "basis",
            }
        ),
        context=context,
    )
    reported_successes = _experiment_integer(
        ratio["full_call_exact_successes"],
        context=f"{context} successes",
    )
    if (
        reported_successes != successes
        or ratio["unit"] != "gpu_seconds_per_full_call_exact_success"
        or ratio["basis"] != "generation_elapsed_seconds_x_configured_gpu_count"
    ):
        raise _experiment_report_error(f"{context} is inconsistent")
    if successes == 0:
        if (
            ratio["available"] is not False
            or ratio["value"] is not None
            or ratio["reason"] != "zero_full_call_exact_successes"
        ):
            raise _experiment_report_error(f"{context} is inconsistent")
        return
    observed = _experiment_number_at_least(
        ratio["value"],
        context=f"{context} value",
        minimum=0.0,
    )
    expected = round(elapsed_seconds * gpu_count / successes, 6)
    if (
        ratio["available"] is not True
        or ratio["reason"] is not None
        or not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise _experiment_report_error(f"{context} is inconsistent")


def _validate_tco_inference_arm(
    value: object,
    *,
    name: str,
    expected_decoding: Mapping[str, Any],
    expected_slices: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    arm = _closed_experiment_mapping(
        value,
        fields=frozenset(
            {
                "available",
                "measurement",
                "timed_call_contract",
                "warmup",
                "sequential_run",
                "decoding",
                "configured_gpu",
                "slices",
                "overall",
            }
        ),
        context=f"TCO inference arm {name}",
    )
    if arm["available"] is not True:
        raise _experiment_report_error(f"TCO inference arm {name} is unavailable")
    expected_measurement = {
        "scope": GENERATION_TIMING_SCOPE,
        "aggregation": GENERATION_TIMING_AGGREGATION,
        "clock": "monotonic_seconds",
        "model_load_included": False,
        "parsing_and_artifact_io_included": False,
    }
    if arm["measurement"] != expected_measurement:
        raise _experiment_report_error(f"TCO inference arm {name} measurement contract drifted")
    if arm["timed_call_contract"] != inference_timed_call_contract():
        raise _experiment_report_error(f"TCO inference arm {name} timed-call contract drifted")
    if arm["warmup"] != inference_warmup_contract():
        raise _experiment_report_error(f"TCO inference arm {name} warmup contract drifted")
    expected_sequential = {
        "boundary": SEQUENTIAL_RUN_BOUNDARY,
        "concurrency": 1,
        "single_model_instance": True,
        "slice_order": ["en", "he"],
        "example_order": "formatted_test_order_within_slice",
    }
    if arm["sequential_run"] != expected_sequential:
        raise _experiment_report_error(f"TCO inference arm {name} sequential-run contract drifted")
    if arm["decoding"] != expected_decoding:
        raise _experiment_report_error(f"TCO inference arm {name} decoding drifted")

    configured_gpu = _closed_experiment_mapping(
        arm["configured_gpu"],
        fields=frozenset({"label", "count", "source"}),
        context=f"TCO inference arm {name} configured GPU",
    )
    gpu_label = configured_gpu["label"]
    gpu_count = _experiment_integer(
        configured_gpu["count"],
        context=f"TCO inference arm {name} GPU count",
        minimum=1,
    )
    if (
        not isinstance(gpu_label, str)
        or not gpu_label
        or configured_gpu["source"] != "config.remote.gpu"
    ):
        raise _experiment_report_error(f"TCO inference arm {name} GPU identity drifted")

    slices = _closed_experiment_mapping(
        arm["slices"],
        fields=frozenset({"en", "he"}),
        context=f"TCO inference arm {name} slices",
    )
    total_examples = 0
    total_elapsed = 0.0
    total_successes = 0
    for language in ("en", "he"):
        payload = _closed_experiment_mapping(
            slices[language],
            fields=frozenset(
                {
                    "examples",
                    "generation_elapsed_seconds",
                    "seconds_per_example",
                    "full_call_exact_successes",
                    "configured_gpu_seconds_per_full_call_exact_success",
                }
            ),
            context=f"TCO {name} {language} inference slice",
        )
        examples = _experiment_integer(
            payload["examples"],
            context=f"TCO {name} {language} examples",
            minimum=1,
        )
        elapsed = _experiment_number_at_least(
            payload["generation_elapsed_seconds"],
            context=f"TCO {name} {language} elapsed seconds",
            minimum=0.0,
        )
        seconds_per_example = _experiment_number_at_least(
            payload["seconds_per_example"],
            context=f"TCO {name} {language} seconds per example",
            minimum=0.0,
        )
        successes = _experiment_integer(
            payload["full_call_exact_successes"],
            context=f"TCO {name} {language} full-call successes",
        )
        expected_examples, expected_successes = expected_slices[language]
        if (
            examples != expected_examples
            or successes != expected_successes
            or successes > examples
            or not math.isclose(
                seconds_per_example,
                elapsed / examples,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise _experiment_report_error(
                f"TCO {name} {language} inference counts or timing are inconsistent"
            )
        _validate_tco_efficiency_ratio(
            payload["configured_gpu_seconds_per_full_call_exact_success"],
            context=f"TCO {name} {language} GPU-seconds ratio",
            elapsed_seconds=elapsed,
            gpu_count=gpu_count,
            successes=successes,
        )
        total_examples += examples
        total_elapsed += elapsed
        total_successes += successes

    overall = _closed_experiment_mapping(
        arm["overall"],
        fields=frozenset(
            {
                "examples",
                "generation_elapsed_seconds",
                "seconds_per_example",
                "full_call_exact_successes",
                "configured_gpu_seconds_per_full_call_exact_success",
            }
        ),
        context=f"TCO inference arm {name} overall",
    )
    overall_examples = _experiment_integer(
        overall["examples"], context=f"TCO {name} overall examples", minimum=1
    )
    overall_elapsed = _experiment_number_at_least(
        overall["generation_elapsed_seconds"],
        context=f"TCO {name} overall elapsed seconds",
        minimum=0.0,
    )
    overall_seconds_per_example = _experiment_number_at_least(
        overall["seconds_per_example"],
        context=f"TCO {name} overall seconds per example",
        minimum=0.0,
    )
    overall_successes = _experiment_integer(
        overall["full_call_exact_successes"],
        context=f"TCO {name} overall full-call successes",
    )
    if (
        overall_examples != total_examples
        or overall_successes != total_successes
        or not math.isclose(overall_elapsed, total_elapsed, rel_tol=0.0, abs_tol=3e-6)
        or not math.isclose(
            overall_seconds_per_example,
            overall_elapsed / overall_examples,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise _experiment_report_error(f"TCO inference arm {name} overall is inconsistent")
    _validate_tco_efficiency_ratio(
        overall["configured_gpu_seconds_per_full_call_exact_success"],
        context=f"TCO {name} overall GPU-seconds ratio",
        elapsed_seconds=overall_elapsed,
        gpu_count=gpu_count,
        successes=overall_successes,
    )
    return {
        "gpu_label": gpu_label,
        "gpu_count": gpu_count,
        "measurement": expected_measurement,
        "timed_call_contract": inference_timed_call_contract(),
        "warmup": inference_warmup_contract(),
        "sequential_run": expected_sequential,
        "decoding": dict(expected_decoding),
    }


def _validate_tco_sources(
    value: object,
    *,
    run_id: str,
    config_sha256: str,
    report_arms: Mapping[str, Any],
    provenance_sources: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> None:
    sources = _closed_experiment_mapping(
        value,
        fields=frozenset(
            {
                "resolved_config",
                "tokenization",
                "training",
                "inference",
                "run_manifest",
                "data_provenance",
            }
        ),
        context="TCO sources",
    )
    resolved_config = _validate_tco_artifact_source(
        sources["resolved_config"],
        context="TCO resolved-config source",
        expected_path=f"runs/{run_id}/config.resolved.yaml",
        expected_kind="config",
        expected_schema="sommelier.config.v2",
        expected_sha256=config_sha256,
    )
    run_manifest = _validate_tco_artifact_source(
        sources["run_manifest"],
        context="TCO root-manifest source",
        expected_path=f"runs/{run_id}/manifest.json",
        expected_kind="manifest",
        expected_schema="sommelier.manifest.v1",
    )

    tokenization = _closed_experiment_mapping(
        sources["tokenization"],
        fields=frozenset(
            {
                "tokenizer_tax_report",
                "tokenizer_tax_records",
                "tokenization_manifest",
                "formatted_inputs",
            }
        ),
        context="TCO tokenization sources",
    )
    _validate_tco_artifact_source(
        tokenization["tokenizer_tax_report"],
        context="TCO tokenizer-tax report source",
        expected_path=f"runs/{run_id}/analysis/tokenization/tokenizer_tax_report.json",
        expected_kind="tokenizer_tax_report",
        expected_schema="sommelier.tokenizer_tax_report.v1",
    )
    _validate_tco_artifact_source(
        tokenization["tokenizer_tax_records"],
        context="TCO tokenizer-tax records source",
        expected_path=f"runs/{run_id}/analysis/tokenization/tokenizer_tax_records.jsonl",
        expected_kind="tokenizer_tax_records",
        expected_schema="sommelier.tokenizer_tax_record.v1",
    )
    tokenization_manifest = _validate_tco_artifact_source(
        tokenization["tokenization_manifest"],
        context="TCO tokenization-manifest source",
        expected_path=f"runs/{run_id}/tokenization_manifest.json",
        expected_kind="manifest",
        expected_schema="sommelier.manifest.v1",
    )
    formatted = _closed_experiment_mapping(
        tokenization["formatted_inputs"],
        fields=frozenset({"train", "validation", "test"}),
        context="TCO formatted-input sources",
    )
    for split in ("train", "validation", "test"):
        _validate_tco_artifact_source(
            formatted[split],
            context=f"TCO formatted {split} source",
            expected_path=f"runs/{run_id}/formatted/{split}.jsonl",
            expected_kind="formatted_split",
            expected_schema="sommelier.formatted_example.v2",
        )

    training = _closed_experiment_mapping(
        sources["training"],
        fields=frozenset(
            {"train_manifest", "runtime_metadata", "training_metrics", "adapter_files"}
        ),
        context="TCO training sources",
    )
    _validate_tco_artifact_source(
        training["train_manifest"],
        context="TCO train-manifest source",
        expected_path=f"runs/{run_id}/train_manifest.json",
        expected_kind="manifest",
        expected_schema="sommelier.manifest.v1",
    )
    runtime_source = _validate_tco_artifact_source(
        training["runtime_metadata"],
        context="TCO training runtime source",
        expected_path=f"runs/{run_id}/runtime_metadata.json",
        expected_kind="runtime_metadata",
        expected_schema="sommelier.runtime_metadata.v1",
    )
    _validate_tco_artifact_source(
        training["training_metrics"],
        context="TCO training-metrics source",
        expected_path=f"runs/{run_id}/train/training_metrics.jsonl",
        expected_kind="training_metrics",
        expected_schema="sommelier.training_metric.v1",
    )
    adapter_files = training["adapter_files"]
    if not isinstance(adapter_files, list) or not adapter_files:
        raise _experiment_report_error("TCO adapter-file sources must be a non-empty array")
    adapter_prefix = f"runs/{run_id}/train/adapter/"
    seen_paths: set[str] = set()
    packaged_bytes = 0
    tensor_bytes = 0
    tensor_files = 0
    for index, raw_file in enumerate(adapter_files):
        candidate = _closed_experiment_mapping(
            raw_file,
            fields=frozenset({"path", "kind", "schema_version", "sha256", "bytes"}),
            context=f"TCO adapter-file source {index}",
        )
        path = candidate["path"]
        if not isinstance(path, str) or not path.startswith(adapter_prefix):
            raise _experiment_report_error(
                f"TCO adapter-file source {index} is outside the adapter directory"
            )
        artifact = _validate_tco_artifact_source(
            raw_file,
            context=f"TCO adapter-file source {index}",
            expected_path=path,
            expected_kind="adapter_weights",
            expected_schema="",
        )
        if path in seen_paths:
            raise _experiment_report_error("TCO adapter-file sources repeat a path")
        seen_paths.add(path)
        file_bytes = cast(int, artifact["bytes"])
        packaged_bytes += file_bytes
        filename = path.rsplit("/", 1)[-1]
        if re.fullmatch(
            r"adapter_model(?:-[0-9]+-of-[0-9]+)?\.(?:safetensors|bin)",
            filename,
        ):
            tensor_bytes += file_bytes
            tensor_files += 1
    if (
        packaged_bytes != training_summary["packaged_bytes"]
        or len(adapter_files) != training_summary["packaged_files"]
        or tensor_bytes != training_summary["tensor_bytes"]
        or tensor_files != training_summary["tensor_files"]
    ):
        raise _experiment_report_error("TCO adapter storage disagrees with its file sources")

    inference = _closed_experiment_mapping(
        sources["inference"],
        fields=frozenset({"base", "v1_en", "v3_en_he"}),
        context="TCO inference sources",
    )
    validated_inference: dict[str, dict[str, dict[str, Any]]] = {}
    for name, model_kind in (
        ("base", "base"),
        ("v1_en", "adapter"),
        ("v3_en_he", "adapter"),
    ):
        arm = cast("Mapping[str, Any]", report_arms[name])
        arm_run_id = cast(str, arm["run_id"])
        arm_config_sha256 = cast(str, arm["config_sha256"])
        arm_sources = _closed_experiment_mapping(
            inference[name],
            fields=frozenset(
                {
                    "evaluation_report",
                    "evaluation_manifest",
                    "run_manifest",
                    "resolved_config",
                    "runtime_metadata",
                    "inference_telemetry",
                }
            ),
            context=f"TCO {name} inference sources",
        )
        artifacts = cast("Mapping[str, Any]", arm["artifacts"])
        evaluation_report = cast("Mapping[str, Any]", artifacts["evaluation_report"])
        validated_inference[name] = {
            "evaluation_report": _validate_tco_artifact_source(
                arm_sources["evaluation_report"],
                context=f"TCO {name} evaluation-report source",
                expected_path=(f"runs/{arm_run_id}/eval/{model_kind}/evaluation_report.json"),
                expected_kind="evaluation_report",
                expected_schema="sommelier.evaluation_report.v3",
                expected_sha256=cast(str, evaluation_report["sha256"]),
            ),
            "evaluation_manifest": _validate_tco_artifact_source(
                arm_sources["evaluation_manifest"],
                context=f"TCO {name} evaluation-manifest source",
                expected_path=f"runs/{arm_run_id}/eval-{model_kind}_manifest.json",
                expected_kind="manifest",
                expected_schema="sommelier.manifest.v1",
            ),
            "run_manifest": _validate_tco_artifact_source(
                arm_sources["run_manifest"],
                context=f"TCO {name} run-manifest source",
                expected_path=f"runs/{arm_run_id}/manifest.json",
                expected_kind="manifest",
                expected_schema="sommelier.manifest.v1",
            ),
            "resolved_config": _validate_tco_artifact_source(
                arm_sources["resolved_config"],
                context=f"TCO {name} resolved-config source",
                expected_path=f"runs/{arm_run_id}/config.resolved.yaml",
                expected_kind="config",
                expected_schema="sommelier.config.v2",
                expected_sha256=arm_config_sha256,
            ),
            "runtime_metadata": _validate_tco_artifact_source(
                arm_sources["runtime_metadata"],
                context=f"TCO {name} runtime source",
                expected_path=f"runs/{arm_run_id}/runtime_metadata.json",
                expected_kind="runtime_metadata",
                expected_schema="sommelier.runtime_metadata.v1",
            ),
            "inference_telemetry": _validate_tco_artifact_source(
                arm_sources["inference_telemetry"],
                context=f"TCO {name} inference-telemetry source",
                expected_path=(f"runs/{arm_run_id}/eval/{model_kind}/inference_telemetry.json"),
                expected_kind="inference_telemetry",
                expected_schema="sommelier.inference_telemetry.v2",
            ),
        }

    if sources["data_provenance"] != provenance_sources:
        raise _experiment_report_error("TCO data_provenance sources disagree with the report")
    required_cross_bindings = (
        (resolved_config, provenance_sources.get("resolved_config")),
        (run_manifest, provenance_sources.get("root_manifest")),
        (tokenization_manifest, provenance_sources.get("tokenization_manifest")),
        (runtime_source, provenance_sources.get("runtime_metadata")),
        (
            validated_inference["v3_en_he"]["inference_telemetry"],
            provenance_sources.get("v3_inference_telemetry"),
        ),
    )
    if any(observed != expected for observed, expected in required_cross_bindings):
        raise _experiment_report_error("TCO sources disagree with data_provenance bindings")


def _validate_experiment_tco(
    report: Mapping[str, Any],
    *,
    run_id: str,
    config_sha256: str,
    provenance_sources: Mapping[str, Any],
) -> None:
    tco = _closed_experiment_mapping(
        report.get("sovereign_tco_evidence"),
        fields=frozenset(
            {
                "schema_version",
                "subject",
                "evidence_policy",
                "paired_tokenization",
                "qlora_training",
                "inference_efficiency",
                "explicitly_unavailable",
                "sources",
            }
        ),
        context="TCO evidence",
    )
    if tco["schema_version"] != SOVEREIGN_TCO_EVIDENCE_SCHEMA:
        raise _experiment_report_error("TCO evidence schema drifted")
    subject = _closed_experiment_mapping(
        tco["subject"],
        fields=frozenset({"run_id", "config_sha256", "tokenizer"}),
        context="TCO evidence subject",
    )
    tokenizer = _closed_experiment_mapping(
        subject["tokenizer"],
        fields=frozenset({"id", "revision"}),
        context="TCO evidence subject tokenizer",
    )
    if (
        subject["run_id"] != run_id
        or subject["config_sha256"] != config_sha256
        or not isinstance(tokenizer["id"], str)
        or not tokenizer["id"]
        or not isinstance(tokenizer["revision"], str)
        or IMMUTABLE_HF_REVISION.fullmatch(tokenizer["revision"]) is None
    ):
        raise _experiment_report_error("TCO evidence subject does not bind the v3 run")
    policy = _closed_experiment_mapping(
        tco["evidence_policy"],
        fields=frozenset({"scope", "currency_estimation", "full_finetune_savings_estimation"}),
        context="TCO evidence policy",
    )
    if policy != {
        "scope": "bounded_observed_or_deterministically_projected_quantities",
        "currency_estimation": "forbidden_without_observed_billing",
        "full_finetune_savings_estimation": ("forbidden_without_matched_full_finetune_evidence"),
    }:
        raise _experiment_report_error("TCO evidence policy drifted")

    paired = _closed_experiment_mapping(
        tco["paired_tokenization"],
        fields=frozenset(
            {
                "available",
                "evidence_kind",
                "reference_language",
                "target_language",
                "paired_scopes",
                "projected_training_workload",
            }
        ),
        context="TCO paired tokenization",
    )
    if (
        paired["available"] is not True
        or paired["evidence_kind"] != "deterministic_measurement_from_pinned_tokenizer"
        or paired["reference_language"] != "en"
        or paired["target_language"] != "he"
    ):
        raise _experiment_report_error("TCO paired tokenization is unavailable or drifted")
    data_contract = cast(
        "Mapping[str, Any]",
        cast("Mapping[str, Any]", report["data_provenance"])["contract"],
    )
    observed_cohorts = cast("Mapping[str, Any]", data_contract["observed_cohorts"])
    _validate_tco_paired_scopes(
        paired["paired_scopes"],
        observed_cohorts=observed_cohorts,
    )
    workload = _closed_experiment_mapping(
        paired["projected_training_workload"],
        fields=frozenset(
            {
                "languages",
                "examples_per_epoch",
                "non_padding_full_tokens_per_epoch",
                "epochs",
                "projected_non_padding_full_tokens",
                "english_only_counterfactual",
                "hebrew_increment",
                "combined_vs_english_only",
                "evidence_kind",
                "boundary",
            }
        ),
        context="TCO projected training workload",
    )
    if (
        workload["languages"] != ["en", "he"]
        or workload["evidence_kind"] != "deterministic_projection"
        or not isinstance(workload["boundary"], str)
        or not workload["boundary"]
    ):
        raise _experiment_report_error("TCO projected training workload drifted")
    examples = _experiment_integer(
        workload["examples_per_epoch"], context="TCO examples_per_epoch", minimum=1
    )
    tokens_per_epoch = _experiment_integer(
        workload["non_padding_full_tokens_per_epoch"],
        context="TCO non_padding_full_tokens_per_epoch",
        minimum=1,
    )
    epochs = _experiment_integer(workload["epochs"], context="TCO epochs", minimum=1)
    projected = _experiment_integer(
        workload["projected_non_padding_full_tokens"],
        context="TCO projected_non_padding_full_tokens",
        minimum=1,
    )
    if projected != tokens_per_epoch * epochs:
        raise _experiment_report_error("TCO projected training tokens are inconsistent")

    english = _closed_experiment_mapping(
        workload["english_only_counterfactual"],
        fields=frozenset(
            {
                "language",
                "examples_per_epoch",
                "non_padding_full_tokens_per_epoch",
                "epochs",
                "projected_non_padding_full_tokens",
            }
        ),
        context="TCO English-only counterfactual",
    )
    english_examples = _experiment_integer(
        english["examples_per_epoch"],
        context="TCO English-only examples_per_epoch",
        minimum=1,
    )
    english_tokens = _experiment_integer(
        english["non_padding_full_tokens_per_epoch"],
        context="TCO English-only non_padding_full_tokens_per_epoch",
        minimum=1,
    )
    english_epochs = _experiment_integer(
        english["epochs"],
        context="TCO English-only epochs",
        minimum=1,
    )
    english_projected = _experiment_integer(
        english["projected_non_padding_full_tokens"],
        context="TCO English-only projected_non_padding_full_tokens",
        minimum=1,
    )
    if (
        english["language"] != "en"
        or english_epochs != epochs
        or english_projected != english_tokens * english_epochs
    ):
        raise _experiment_report_error("TCO English-only counterfactual is inconsistent")

    hebrew = _closed_experiment_mapping(
        workload["hebrew_increment"],
        fields=frozenset(
            {
                "language",
                "examples_per_epoch",
                "examples_per_epoch_ratio_to_english_only",
                "non_padding_full_tokens_per_epoch",
                "non_padding_full_tokens_per_epoch_ratio_to_english_only",
                "epochs",
                "projected_non_padding_full_tokens",
                "projected_non_padding_full_tokens_ratio_to_english_only",
            }
        ),
        context="TCO Hebrew increment",
    )
    hebrew_examples = _experiment_integer(
        hebrew["examples_per_epoch"],
        context="TCO Hebrew incremental examples_per_epoch",
        minimum=1,
    )
    hebrew_tokens = _experiment_integer(
        hebrew["non_padding_full_tokens_per_epoch"],
        context="TCO Hebrew incremental non_padding_full_tokens_per_epoch",
        minimum=1,
    )
    hebrew_epochs = _experiment_integer(
        hebrew["epochs"],
        context="TCO Hebrew incremental epochs",
        minimum=1,
    )
    hebrew_projected = _experiment_integer(
        hebrew["projected_non_padding_full_tokens"],
        context="TCO Hebrew incremental projected_non_padding_full_tokens",
        minimum=1,
    )
    hebrew_example_ratio = _experiment_number(
        hebrew["examples_per_epoch_ratio_to_english_only"],
        context="TCO Hebrew incremental example ratio",
    )
    hebrew_token_ratio = _experiment_number(
        hebrew["non_padding_full_tokens_per_epoch_ratio_to_english_only"],
        context="TCO Hebrew incremental token ratio",
    )
    hebrew_projected_ratio = _experiment_number(
        hebrew["projected_non_padding_full_tokens_ratio_to_english_only"],
        context="TCO Hebrew incremental projected-token ratio",
    )
    if (
        hebrew["language"] != "he"
        or hebrew_epochs != epochs
        or hebrew_projected != hebrew_tokens * hebrew_epochs
        or not math.isclose(
            hebrew_example_ratio,
            hebrew_examples / english_examples,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            hebrew_token_ratio,
            hebrew_tokens / english_tokens,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            hebrew_projected_ratio,
            hebrew_projected / english_projected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise _experiment_report_error("TCO Hebrew increment is inconsistent")

    if (
        examples != english_examples + hebrew_examples
        or tokens_per_epoch != english_tokens + hebrew_tokens
        or projected != english_projected + hebrew_projected
    ):
        raise _experiment_report_error("TCO combined workload is not English plus Hebrew")
    train_cohort = cast("Mapping[str, Any]", observed_cohorts["train"])
    if (
        english_examples != train_cohort["en"]
        or hebrew_examples != train_cohort["he"]
        or examples != train_cohort["total"]
    ):
        raise _experiment_report_error("TCO training workload disagrees with data provenance")

    multipliers = _closed_experiment_mapping(
        workload["combined_vs_english_only"],
        fields=frozenset(
            {
                "examples_per_epoch_multiplier",
                "non_padding_full_tokens_per_epoch_multiplier",
                "projected_non_padding_full_tokens_multiplier",
            }
        ),
        context="TCO combined-vs-English multipliers",
    )
    example_multiplier = _experiment_number(
        multipliers["examples_per_epoch_multiplier"],
        context="TCO combined-vs-English example multiplier",
    )
    token_multiplier = _experiment_number(
        multipliers["non_padding_full_tokens_per_epoch_multiplier"],
        context="TCO combined-vs-English token multiplier",
    )
    projected_multiplier = _experiment_number(
        multipliers["projected_non_padding_full_tokens_multiplier"],
        context="TCO combined-vs-English projected-token multiplier",
    )
    if (
        not math.isclose(
            example_multiplier,
            examples / english_examples,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            token_multiplier,
            tokens_per_epoch / english_tokens,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            projected_multiplier,
            projected / english_projected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise _experiment_report_error("TCO combined-vs-English multiplier is inconsistent")

    report_arms = cast("Mapping[str, Any]", report["arms"])
    v3_arm = cast("Mapping[str, Any]", report_arms["v3_en_he"])
    v3_adapter = cast("Mapping[str, Any]", v3_arm["adapter_source"])
    training_summary = _validate_tco_training(
        tco["qlora_training"],
        source_revision=cast(str, data_contract["source_code_revision"]),
        adapter_tree_sha256=_experiment_sha256(
            v3_adapter["tree_sha256"], context="TCO v3 adapter tree"
        ),
    )
    training = cast("Mapping[str, Any]", training_summary["payload"])

    unavailable = _closed_experiment_mapping(
        tco["explicitly_unavailable"],
        fields=frozenset({"currency_cost", "full_finetune_savings"}),
        context="TCO explicitly unavailable claims",
    )
    expected_unavailable = {
        "currency_cost": {
            "available": False,
            "value": None,
            "reason": "provider_billing_evidence_not_supplied",
        },
        "full_finetune_savings": {
            "available": False,
            "value": None,
            "reason": "matched_full_finetune_evidence_not_supplied",
        },
    }
    if unavailable != expected_unavailable:
        raise _experiment_report_error("TCO unavailable cost claims drifted")
    if (
        training["currency_cost"] != expected_unavailable["currency_cost"]
        or training["full_finetune_savings"] != expected_unavailable["full_finetune_savings"]
    ):
        raise _experiment_report_error("TCO training cost claims disagree with policy")

    inference = _closed_experiment_mapping(
        tco["inference_efficiency"],
        fields=frozenset({"arms", "cross_arm_comparability"}),
        context="TCO inference efficiency",
    )
    inference_arms = _closed_experiment_mapping(
        inference["arms"],
        fields=frozenset({"base", "v1_en", "v3_en_he"}),
        context="TCO inference arms",
    )
    shared = cast("Mapping[str, Any]", report["shared_evaluation_identity"])
    shared_decoding = cast("Mapping[str, Any]", shared["decoding"])
    inference_summaries: dict[str, dict[str, Any]] = {}
    for name in ("base", "v1_en", "v3_en_he"):
        report_arm = cast("Mapping[str, Any]", report_arms[name])
        metrics = cast("Mapping[str, Any]", report_arm["metrics"])
        slices = cast("Mapping[str, Any]", metrics["slices"])
        expected_slices: dict[str, tuple[int, int]] = {}
        for language in ("en", "he"):
            language_metrics = cast("Mapping[str, Any]", slices[language])
            full_call = cast("Mapping[str, Any]", language_metrics["full_call_exact_match"])
            expected_slices[language] = (
                cast(int, full_call["denominator"]),
                cast(int, full_call["numerator"]),
            )
        inference_summaries[name] = _validate_tco_inference_arm(
            inference_arms[name],
            name=name,
            expected_decoding=shared_decoding,
            expected_slices=expected_slices,
        )

    comparability = _closed_experiment_mapping(
        inference["cross_arm_comparability"],
        fields=frozenset({"available", "configured_gpu", "observed_packages", "boundary"}),
        context="TCO cross-arm inference comparability",
    )
    comparable_gpu = _closed_experiment_mapping(
        comparability["configured_gpu"],
        fields=frozenset({"label", "count"}),
        context="TCO comparable inference GPU",
    )
    comparable_gpu_count = _experiment_integer(
        comparable_gpu["count"], context="TCO comparable inference GPU count", minimum=1
    )
    packages = comparability["observed_packages"]
    required_packages = {
        "python",
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "peft",
        "datasets",
        "huggingface_hub",
    }
    if not isinstance(packages, dict):
        raise _experiment_report_error("TCO observed inference packages must be an object")
    package_payload = cast("dict[object, object]", packages)
    if (
        comparability["available"] is not True
        or not isinstance(comparable_gpu["label"], str)
        or not comparable_gpu["label"]
        or not required_packages.issubset(package_payload)
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            or version == "absent"
            for name, version in package_payload.items()
        )
        or comparability["boundary"]
        != "Identical sequential end-to-end generator-call measurement contract."
        or any(
            summary["gpu_label"] != comparable_gpu["label"]
            or summary["gpu_count"] != comparable_gpu_count
            for summary in inference_summaries.values()
        )
        or training_summary["gpu_label"] != comparable_gpu["label"]
        or training_summary["gpu_count"] != comparable_gpu_count
    ):
        raise _experiment_report_error("TCO cross-arm inference comparability is inconsistent")

    _validate_tco_sources(
        tco["sources"],
        run_id=run_id,
        config_sha256=config_sha256,
        report_arms=report_arms,
        provenance_sources=provenance_sources,
        training_summary=training_summary,
    )


def _validate_experiment_identity(
    *,
    path: Path,
    run_id: str,
    source_revision: str,
    config_sha256: str,
    tree_sha256: str,
    dataset_revision: str,
) -> dict[str, Any]:
    report = _load_json_object(path, context="final experiment report")
    if report.get("schema_version") != EXPERIMENT_REPORT_SCHEMA:
        raise UserInputError("adapter publication requires the final experiment report schema")
    arms = report.get("arms")
    v3 = arms.get("v3_en_he") if isinstance(arms, dict) else None
    if not isinstance(v3, dict):
        raise UserInputError("final experiment report has no v3_en_he arm")
    source = v3.get("adapter_source")
    if v3.get("run_id") != run_id or v3.get("config_sha256") != config_sha256:
        raise UserInputError("final experiment report does not bind the training run/config")
    if not _matches_v3_adapter_source(source, run_id=run_id, tree_sha256=tree_sha256):
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
    if set(report) != _EXPERIMENT_REPORT_FIELDS:
        raise UserInputError("final experiment report has unexpected or missing top-level fields")
    _validate_experiment_preregistration(report, source_revision=source_revision)
    _validate_shared_evaluation_identity(report)
    provenance_sources = _validate_experiment_data_provenance(
        report,
        run_id=run_id,
        config_sha256=config_sha256,
        source_revision=source_revision,
        dataset_revision=dataset_revision,
    )
    arm_metrics = _validate_experiment_arms(
        report,
        run_id=run_id,
        config_sha256=config_sha256,
        tree_sha256=tree_sha256,
    )
    primary = _validate_experiment_comparisons(report, arm_metrics=arm_metrics)
    _validate_experiment_claims(report, primary=primary)
    _validate_experiment_tco(
        report,
        run_id=run_id,
        config_sha256=config_sha256,
        provenance_sources=provenance_sources,
    )
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


def _claim_contract_for_render(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    arms = _closed_experiment_mapping(
        report.get("arms"),
        fields=frozenset({"base", "v1_en", "v3_en_he"}),
        context="arms",
    )
    arm_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name, kind in (("base", "base"), ("v1_en", "adapter"), ("v3_en_he", "adapter")):
        _, arm_metrics[name] = _validate_arm_payload(arms[name], name=name, expected_kind=kind)
    primary = _validate_experiment_comparisons(report, arm_metrics=arm_metrics)
    return _validate_experiment_claims(report, primary=primary)


def _claim_number(value: object) -> str:
    number = _experiment_number(value, context="claim-section number")
    return json.dumps(number, allow_nan=False)


def render_hebrew_v3_claim_section(report: Mapping[str, Any]) -> str:
    """Render the exact release-card claim section from validated gate decisions."""
    claims = _claim_contract_for_render(report)
    labels = {
        "hebrew_full_call_uplift": "Hebrew full-call uplift versus immutable v1",
        "english_full_call_non_inferiority": (
            "English full-call non-inferiority at the 0.01 absolute margin"
        ),
    }
    lines = ["## Claim-gated result", ""]
    for name in _CLAIM_ORDER:
        claim = claims[name]
        interval = cast("dict[str, Any]", claim["ci95"])
        status = "passed" if claim["passed"] is True else "withheld"
        line = (
            f"- **{labels[name]} — {status}.** "
            f"Estimate `{_claim_number(claim['estimate'])}`; 95% paired-bootstrap CI "
            f"`[{_claim_number(interval['lower'])}, {_claim_number(interval['upper'])}]`."
        )
        if claim["passed"] is True:
            line += f" {claim['statement']}"
        else:
            line += f" Criterion: `{claim['criterion']}`. No release claim is approved."
        lines.append(line)
    return "\n".join(lines)


def _markdown_h2_section(text: str, *, heading: str) -> str | None:
    lines = text.splitlines()
    try:
        start = lines.index(f"## {heading}")
    except ValueError:
        return None
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end]).strip()


def _validate_adapter_card(
    path: Path,
    *,
    config: SommelierConfig,
    experiment_sha256: str,
    tree_sha256: str,
    source_revision: str,
    dataset_revision: str,
    experiment_report: Mapping[str, Any],
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
    claim_heading = "## Claim-gated result"
    if (
        sum(_CLAIM_HEADING.fullmatch(line) is not None for line in lines) != 1
        or claim_heading not in lines
    ):
        raise UserInputError(
            "adapter model card claim-gated result section must contain exactly one "
            "Claim-gated result heading",
            hint="Keep one deterministic section rendered from the final experiment report.",
        )
    claims = _claim_contract_for_render(experiment_report)
    expected_claim_section = render_hebrew_v3_claim_section(experiment_report)
    if _markdown_h2_section(text, heading="Claim-gated result") != expected_claim_section:
        raise UserInputError(
            "adapter model card claim-gated result section does not match experiment_report.json",
            hint=(
                "Replace the complete Claim-gated result section with "
                "render_hebrew_v3_claim_section(report); do not paraphrase release claims."
            ),
        )
    for name, statement in (
        ("hebrew_full_call_uplift", _HEBREW_UPLIFT_STATEMENT),
        ("english_full_call_non_inferiority", _ENGLISH_NON_INFERIORITY_STATEMENT),
    ):
        expected_count = 1 if claims[name]["passed"] is True else 0
        if text.count(statement) != expected_count:
            raise UserInputError(
                "adapter model card contains a missing, duplicated, or unapproved claim statement"
            )
    claim_start = lines.index(claim_heading)
    claim_end = next(
        (index for index in range(claim_start + 1, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    prose_outside_claim_section = "\n".join([*lines[:claim_start], *lines[claim_end:]])
    if _UNAPPROVED_RESULT_ASSERTION.search(prose_outside_claim_section) is not None:
        raise UserInputError(
            "adapter model card contains unapproved result or claim prose outside the "
            "deterministic Claim-gated result section",
            hint=(
                "Remove performance assertions and numeric result prose outside the section "
                "rendered from experiment_report.json."
            ),
        )
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
    experiment_report = _validate_experiment_identity(
        path=experiment_path,
        run_id=run_id,
        source_revision=revision,
        config_sha256=config_sha256,
        tree_sha256=tree_sha256,
        dataset_revision=config.dataset_for("he").dataset_revision,
    )
    validate_evaluation_release_evidence(
        bundle_dir=bundle_dir,
        experiment_report=experiment_report,
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
        experiment_report=experiment_report,
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
    for bundle_name in sorted(EVALUATION_RELEASE_EVIDENCE_BUNDLE_FILES):
        files[f"sommelier/{bundle_name}"] = bundle_dir / bundle_name
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
    descriptor: int
    capacity: int
    device: int
    inode: int
    parent_device: int
    parent_inode: int
    source_root_identities: frozenset[tuple[int, int]]
    expected_sha256: str


def _recovery_receipt_error(path: Path) -> UserInputError:
    return UserInputError(
        f"publication receipt already exists and is not a recoverable "
        f"commit_submitting journal: {path}",
        hint=(
            "Keep the existing journal unchanged. Retry only the exact publication whose "
            "commit response may have been lost, or choose a new receipt for a new release."
        ),
    )


def _read_exact_descriptor(descriptor: int, *, expected_bytes: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    observed_bytes = 0
    while observed_bytes < expected_bytes:
        chunk = os.read(descriptor, min(1024 * 1024, expected_bytes - observed_bytes))
        if not chunk:
            break
        chunks.append(chunk)
        observed_bytes += len(chunk)
    if observed_bytes != expected_bytes or os.read(descriptor, 1):
        raise OSError("publication receipt size changed while reopening")
    return b"".join(chunks)


def _open_recovery_receipt(
    path: Path,
    *,
    source_roots: Sequence[Path],
) -> tuple[_ReceiptReservation, dict[str, Any]]:
    """Securely reopen one durable uncertain-submission journal for recovery."""
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    descriptor: int | None = None
    parent_descriptor: int | None = None
    try:
        root_identities = _source_root_identities(source_roots)
        if os.name == "posix":
            parent_descriptor, parent_metadata = _open_posix_receipt_parent(
                path,
                source_root_identities=root_identities,
            )
            descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
            entry = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        else:
            _assert_receipt_outside_publication_sources(path, source_roots=source_roots)
            parent_metadata = path.parent.stat()
            descriptor = os.open(path, flags)
            entry = path.stat(follow_symlinks=False)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not stat.S_ISREG(entry.st_mode)
            or metadata.st_dev != entry.st_dev
            or metadata.st_ino != entry.st_ino
            or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600)
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_PUBLICATION_RECEIPT_BYTES
        ):
            raise OSError("existing publication receipt identity is unsafe")
        data = _read_exact_descriptor(descriptor, expected_bytes=metadata.st_size)
        try:
            loaded = loads_unique_json(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKeyError) as error:
            raise OSError("existing publication receipt is invalid JSON") from error
        if not isinstance(loaded, dict):
            raise OSError("existing publication receipt is not an object")
        reservation = _ReceiptReservation(
            path=path,
            descriptor=descriptor,
            capacity=metadata.st_size,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            parent_device=parent_metadata.st_dev,
            parent_inode=parent_metadata.st_ino,
            source_root_identities=root_identities,
            expected_sha256=hashlib.sha256(data).hexdigest(),
        )
        descriptor = None
        return reservation, cast("dict[str, Any]", loaded)
    except (OSError, UserInputError):
        raise _recovery_receipt_error(path) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _validate_recovery_receipt(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, object],
    repo_id: str,
    repo_type: PublicationRepoType,
    commit_message: str,
    create_repo: bool,
    path: Path,
) -> tuple[str | None, frozenset[str]]:
    expected_top_level = {
        "schema_version",
        "created_at",
        "status",
        "executed",
        "repository",
        "files",
        "platform_files",
    }
    repository = payload.get("repository")
    platform_files = payload.get("platform_files")
    if (
        set(payload) != expected_top_level
        or payload.get("schema_version") != PUBLICATION_RECEIPT_SCHEMA
        or not isinstance(payload.get("created_at"), str)
        or not payload.get("created_at")
        or payload.get("status") != "commit_submitting"
        or payload.get("executed") is not True
        or payload.get("files") != plan.get("files")
        or not isinstance(repository, dict)
        or set(repository)
        != {
            "repo_id",
            "repo_type",
            "commit_message",
            "commit_sha",
            "parent_commit",
            "create_repo",
        }
        or repository.get("repo_id") != repo_id
        or repository.get("repo_type") != repo_type
        or repository.get("commit_message") != commit_message
        or repository.get("commit_sha") is not None
        or repository.get("create_repo") is not create_repo
        or not isinstance(platform_files, list)
        or any(not isinstance(name, str) for name in platform_files)
    ):
        raise _recovery_receipt_error(path)
    try:
        normalized_platform_files = _normalize_remote_files(cast("list[str]", platform_files))
    except ExternalDependencyError:
        raise _recovery_receipt_error(path) from None
    if (
        list(platform_files) != sorted(normalized_platform_files)
        or normalized_platform_files - PLATFORM_MANAGED_FILES
    ):
        raise _recovery_receipt_error(path)
    parent_commit = repository.get("parent_commit")
    if (
        parent_commit is not None
        and (
            not isinstance(parent_commit, str)
            or IMMUTABLE_HF_REVISION.fullmatch(parent_commit) is None
        )
    ) or (parent_commit is None and not create_repo):
        raise _recovery_receipt_error(path)
    return parent_commit, normalized_platform_files


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


def _descriptor_sha256(descriptor: int, *, expected_bytes: int) -> str:
    """Hash the exact current receipt without retaining or reporting its payload."""
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    observed_bytes = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        observed_bytes += len(chunk)
        if observed_bytes > expected_bytes:
            raise OSError("reserved publication receipt size changed")
        digest.update(chunk)
    if observed_bytes != expected_bytes:
        raise OSError("reserved publication receipt size changed")
    return digest.hexdigest()


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
    reserved_data = _receipt_bytes(payload, capacity=capacity)
    descriptor: int | None = None
    parent_descriptor: int | None = None
    metadata: os.stat_result | None = None
    parent_metadata: os.stat_result | None = None
    keep_descriptor_open = False
    try:
        root_identities = _source_root_identities(source_roots)
        if os.name == "posix":
            parent_descriptor, parent_metadata = _open_posix_receipt_parent(
                path,
                source_root_identities=root_identities,
            )
            descriptor = os.open(
                path.name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_BINARY", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
        else:
            _assert_receipt_outside_publication_sources(path, source_roots=source_roots)
            parent_metadata = path.parent.stat()
            descriptor = os.open(
                path,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                0o600,
            )
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
        _write_all(descriptor, reserved_data)
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
        keep_descriptor_open = True
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
        if descriptor is not None and not keep_descriptor_open:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    if descriptor is None or metadata is None or parent_metadata is None:
        raise UserInputError(f"could not reserve publication receipt before network access: {path}")
    return _ReceiptReservation(
        path=path,
        descriptor=descriptor,
        capacity=capacity,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        parent_device=parent_metadata.st_dev,
        parent_inode=parent_metadata.st_ino,
        source_root_identities=root_identities,
        expected_sha256=hashlib.sha256(reserved_data).hexdigest(),
    )


def _write_receipt(
    reservation: _ReceiptReservation,
    payload: Mapping[str, object],
) -> _ReceiptReservation:
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
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
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise ExternalDependencyError(
            f"could not update reserved publication receipt: {reservation.path}",
            hint="Inspect the Hub and the pending receipt before any retry.",
        ) from error
    try:
        if descriptor is None:
            raise OSError("reserved publication receipt descriptor was not initialized")
        held_metadata = os.fstat(reservation.descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(held_metadata.st_mode)
            or held_metadata.st_dev != reservation.device
            or held_metadata.st_ino != reservation.inode
            or held_metadata.st_size != reservation.capacity
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_dev != reservation.device
            or metadata.st_ino != reservation.inode
            or metadata.st_size != reservation.capacity
        ):
            raise OSError("reserved publication receipt path identity changed")
        if parent_descriptor is not None:
            entry = os.stat(
                reservation.path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        else:
            entry = reservation.path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_dev != metadata.st_dev
            or entry.st_ino != metadata.st_ino
        ):
            raise OSError("reserved publication receipt path identity changed")
        observed_sha256 = _descriptor_sha256(
            descriptor,
            expected_bytes=reservation.capacity,
        )
        if observed_sha256 != reservation.expected_sha256:
            raise OSError("reserved publication receipt content changed")
        data = _receipt_bytes(payload, capacity=reservation.capacity)
        expected_sha256 = hashlib.sha256(data).hexdigest()
        os.lseek(descriptor, 0, os.SEEK_SET)
        _write_all(descriptor, data)
        os.fsync(descriptor)
        if (
            _descriptor_sha256(
                descriptor,
                expected_bytes=reservation.capacity,
            )
            != expected_sha256
        ):
            raise OSError("reserved publication receipt content changed during update")
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_dev != reservation.device
            or metadata.st_ino != reservation.inode
            or metadata.st_size != reservation.capacity
        ):
            raise OSError("reserved publication receipt path identity changed")
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
        else:
            entry = reservation.path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(entry.st_mode)
                or entry.st_dev != metadata.st_dev
                or entry.st_ino != metadata.st_ino
            ):
                raise OSError("reserved publication receipt path identity changed")
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
    return replace(
        reservation,
        expected_sha256=expected_sha256,
    )


def _close_receipt_reservation(reservation: _ReceiptReservation) -> None:
    try:
        os.close(reservation.descriptor)
    except OSError:
        # Closing is best effort and must never hide the publication outcome.
        pass


def _verify_remote_commit(
    client: HubPublicationClient,
    *,
    prepared: PreparedPublication,
    repo_id: str,
    revision: str,
    file_hashes: Mapping[str, str],
) -> frozenset[str]:
    try:
        remote_files = _normalize_remote_files(
            client.list_files(
                repo_id=repo_id,
                repo_type=prepared.repo_type,
                revision=revision,
            )
        )
    except ExternalDependencyError:
        raise
    except Exception as error:
        raise ExternalDependencyError(
            f"could not enumerate {repo_id!r} at returned commit {revision}",
            hint="Treat the publication as unverified and inspect the immutable commit manually.",
        ) from error
    intended_files = frozenset(prepared.files)
    if (
        not intended_files.issubset(remote_files)
        or remote_files - intended_files - PLATFORM_MANAGED_FILES
    ):
        missing = sorted(intended_files - remote_files)
        unexpected = sorted(remote_files - intended_files - PLATFORM_MANAGED_FILES)
        raise ExternalDependencyError(
            "Hugging Face round-trip filename verification failed "
            f"(missing={missing}, unexpected={unexpected})",
            hint="Treat the publication as unverified; do not pin this commit in a config.",
        )
    for filename, expected_sha256 in file_hashes.items():
        try:
            downloaded = client.download_file(
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
    return remote_files


def _recover_uncertain_commit(
    client: HubPublicationClient,
    *,
    prepared: PreparedPublication,
    repo_id: str,
    commit_message: str,
    create_repo: bool,
    parent_commit: str | None,
    prior_platform_files: frozenset[str],
    plan: Mapping[str, object],
    file_hashes: Mapping[str, str],
    reservation: _ReceiptReservation,
    snapshot: _PrivateSnapshot,
) -> dict[str, object]:
    """Adopt an exact accepted commit without ever resubmitting the mutation."""
    _assert_snapshot_unchanged(snapshot)
    try:
        revision = client.resolve_revision(repo_id=repo_id, repo_type=prepared.repo_type)
    except Exception as error:
        raise ExternalDependencyError(
            f"could not inspect uncertain Hugging Face commit for {repo_id!r}",
            hint="No second commit was submitted; preserve the receipt and inspect the Hub.",
        ) from error
    if (
        not isinstance(revision, str)
        or IMMUTABLE_HF_REVISION.fullmatch(revision) is None
        or revision == parent_commit
    ):
        raise ExternalDependencyError(
            "Hugging Face does not expose one recoverable immutable commit after the "
            "journaled parent",
            hint="No second commit was submitted; preserve the receipt and inspect the Hub.",
        )
    try:
        metadata = client.inspect_commit(
            repo_id=repo_id,
            repo_type=prepared.repo_type,
            revision=revision,
        )
    except ExternalDependencyError:
        raise
    except Exception as error:
        raise ExternalDependencyError(
            f"could not inspect uncertain Hugging Face commit {repo_id}@{revision}",
            hint="No second commit was submitted; preserve the receipt and inspect the Hub.",
        ) from error
    if metadata.parent_commit != parent_commit or metadata.title != commit_message:
        raise ExternalDependencyError(
            "Hugging Face HEAD is not the exact direct commit recorded by the uncertain "
            "publication journal",
            hint="No second commit was submitted; preserve the receipt and inspect the Hub.",
        )
    remote_files = _verify_remote_commit(
        client,
        prepared=prepared,
        repo_id=repo_id,
        revision=revision,
        file_hashes=file_hashes,
    )
    observed_platform_files = remote_files - frozenset(prepared.files)
    if parent_commit is not None and observed_platform_files != prior_platform_files:
        raise ExternalDependencyError(
            "Hugging Face platform-managed file set changed in the uncertain commit",
            hint="No second commit was submitted; preserve the receipt and inspect the Hub.",
        )
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
        "platform_files": sorted(observed_platform_files),
    }
    reservation = _write_receipt(reservation, commit_returned)
    _assert_snapshot_unchanged(snapshot)
    receipt: dict[str, object] = {
        **commit_returned,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "verified",
    }
    _write_receipt(reservation, receipt)
    return receipt


def _publish_prepared_bundle_impl(
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
    _exit_stack: ExitStack,
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
    recovery_parent: str | None = None
    recovery_platform_files: frozenset[str] = frozenset()
    recovering = receipt_path.exists() or receipt_path.is_symlink()
    if recovering:
        reservation, recovery_payload = _open_recovery_receipt(
            receipt_path,
            source_roots=prepared.source_roots,
        )
        _exit_stack.callback(_close_receipt_reservation, reservation)
        recovery_parent, recovery_platform_files = _validate_recovery_receipt(
            recovery_payload,
            plan=plan,
            repo_id=repo_id,
            repo_type=prepared.repo_type,
            commit_message=commit_message,
            create_repo=create_repo,
            path=receipt_path,
        )
    else:
        reservation = _reserve_receipt(
            receipt_path,
            plan,
            source_roots=prepared.source_roots,
        )
        _exit_stack.callback(_close_receipt_reservation, reservation)

    active = client if client is not None else _HuggingFaceHubClient()
    if recovering:
        return _recover_uncertain_commit(
            active,
            prepared=prepared,
            repo_id=repo_id,
            commit_message=commit_message,
            create_repo=create_repo,
            parent_commit=recovery_parent,
            prior_platform_files=recovery_platform_files,
            plan=plan,
            file_hashes=file_hashes,
            reservation=reservation,
            snapshot=_snapshot,
        )
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
        if parent_commit is not None and (
            not isinstance(parent_commit, str)
            or IMMUTABLE_HF_REVISION.fullmatch(parent_commit) is None
        ):
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
    reservation = _write_receipt(reservation, commit_submitting)
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
    reservation = _write_receipt(reservation, commit_returned)
    _assert_snapshot_unchanged(_snapshot)
    if not isinstance(revision, str) or IMMUTABLE_HF_REVISION.fullmatch(revision) is None:
        raise ExternalDependencyError(
            f"Hugging Face returned a non-immutable commit identity: {revision!r}",
            hint="Inspect the repository before retrying; do not record a branch name as evidence.",
        )
    remote_files = _verify_remote_commit(
        active,
        prepared=prepared,
        repo_id=repo_id,
        revision=revision,
        file_hashes=file_hashes,
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
    with ExitStack() as exit_stack:
        return _publish_prepared_bundle_impl(
            prepared,
            repo_id=repo_id,
            commit_message=commit_message,
            execute=execute,
            create_repo=create_repo,
            confirmed_repo_id=confirmed_repo_id,
            receipt_path=receipt_path,
            client=client,
            _snapshot=_snapshot,
            _exit_stack=exit_stack,
        )


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
        source_revision = _phase_a_source_revision(staged_bundle)
        committed_config_bytes = require_committed_config_bytes(
            config_path,
            code_revision=source_revision,
            context="Hebrew v3 dataset publication",
        )
        if committed_config_bytes != staged_config.read_bytes():
            raise UserInputError(
                "Hebrew v3 publication snapshot does not match the committed Phase-A config"
            )
        prepared = _prepare_hebrew_dataset_publication(
            config_path=staged_config,
            bundle_dir=staged_bundle,
            root_rows_path=staged_root_rows,
            committed_config_bytes=committed_config_bytes,
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
