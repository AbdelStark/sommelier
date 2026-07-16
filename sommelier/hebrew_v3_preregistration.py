"""Phase-boundary helpers for the Hebrew v3 human-review preregistration."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from sommelier.config import SommelierConfig
from sommelier.errors import UserInputError
from sommelier.reviewer import (
    ReviewerRequirement,
    reviewer_preregistration_payload,
    reviewer_preregistration_sha256,
)

HEBREW_V3_PROVISIONAL_DATASET_REVISION: Final = "main"
_IMMUTABLE_REVISION: Final = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def require_preregistered_reviewer(
    config: SommelierConfig,
    *,
    context: str = "Hebrew v3 full launch",
) -> ReviewerRequirement:
    """Fail admission unless a real canonical reviewer is configured."""
    if config.semantic_review is None:
        raise UserInputError(
            f"{context} is missing its preregistered named human reviewer",
            hint=(
                "Add the reviewer's stable public ID, canonical OpenSSH Ed25519 public key, "
                "and matching SHA256 fingerprint to semantic_review.reviewer in "
                "examples/config.v3-he-full.yaml, then commit it before translation. "
                "Do not use a placeholder or an automation-owned signing key."
            ),
        )
    return config.semantic_review.reviewer.to_requirement()


def reviewer_anchor_payload(
    config: SommelierConfig,
    *,
    context: str = "Hebrew v3 full launch",
) -> dict[str, str]:
    """Canonical payload suitable for a pre-provider run identity."""
    return reviewer_preregistration_payload(require_preregistered_reviewer(config, context=context))


def reviewer_anchor_sha256(
    config: SommelierConfig,
    *,
    context: str = "Hebrew v3 full launch",
) -> str:
    """Stable reviewer digest shared by Phase-A and Phase-B configs."""
    return reviewer_preregistration_sha256(require_preregistered_reviewer(config, context=context))


def validate_reviewer_anchor_evidence(
    config: SommelierConfig,
    evidence: Mapping[str, object],
    *,
    context: str,
) -> ReviewerRequirement:
    """Bind stored evidence to the named reviewer in the committed config."""
    requirement = require_preregistered_reviewer(config, context=context)
    expected_payload = reviewer_preregistration_payload(requirement)
    expected_sha256 = reviewer_preregistration_sha256(requirement)
    if evidence.get("reviewer_preregistration") != expected_payload:
        raise UserInputError(
            f"{context} reviewer preregistration does not match its config",
            hint=(
                "Use the exact Phase-A config and translation evidence. A reviewer change "
                "requires a fresh committed Phase-A translation run."
            ),
        )
    if evidence.get("reviewer_preregistration_sha256") != expected_sha256:
        raise UserInputError(f"{context} reviewer preregistration digest is invalid")
    return requirement


def require_committed_config_bytes(
    config_path: Path,
    *,
    code_revision: str,
    context: str,
) -> bytes:
    """Return config bytes only when they are the tracked blob at the launch SHA."""
    if config_path.is_symlink() or not config_path.is_file():
        raise UserInputError(f"{context} config is not a regular file: {config_path}")
    try:
        root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
                cwd=config_path.resolve().parent,
            ).stdout.strip()
        ).resolve()
        resolved = config_path.resolve()
        relative = resolved.relative_to(root)
        committed = subprocess.run(
            ["git", "show", f"{code_revision}:{relative.as_posix()}"],
            check=True,
            capture_output=True,
            cwd=root,
        ).stdout
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise UserInputError(
            f"{context} config is not tracked at the immutable launch revision",
            hint=(
                "Put the reviewer-anchored config inside the repository, commit it, and "
                "launch the exact tracked file from a clean worktree."
            ),
        ) from error
    observed = config_path.read_bytes()
    if observed != committed:
        raise UserInputError(
            f"{context} config bytes differ from the immutable launch revision",
            hint="Commit the exact config bytes before dispatching any paid or remote work.",
        )
    return observed


def _hebrew_dataset_index(config: SommelierConfig, *, context: str) -> int:
    indexes = [index for index, source in enumerate(config.datasets) if source.language == "he"]
    if len(indexes) != 1:
        raise UserInputError(f"{context} must contain exactly one Hebrew dataset source")
    return indexes[0]


def validate_hebrew_v3_phase_transition(
    phase_a: SommelierConfig,
    phase_b: SommelierConfig,
) -> ReviewerRequirement:
    """Prove Phase B changed only the provisional Hebrew dataset revision.

    Experiment-specific validators still own the fixed model, dataset, training,
    and evaluation values.  This helper establishes the two-commit transition:
    the named reviewer remains byte-for-byte identical while ``main`` becomes
    one immutable published dataset revision and no other resolved setting moves.
    """
    phase_a_reviewer = require_preregistered_reviewer(
        phase_a,
        context="Hebrew v3 Phase-A config",
    )
    phase_b_reviewer = require_preregistered_reviewer(
        phase_b,
        context="Hebrew v3 Phase-B config",
    )
    if phase_b_reviewer != phase_a_reviewer:
        raise UserInputError(
            "Hebrew v3 reviewer preregistration changed between Phase A and Phase B",
            hint="A reviewer change requires a fresh Phase-A commit and translation run.",
        )

    phase_a_index = _hebrew_dataset_index(phase_a, context="Hebrew v3 Phase-A config")
    phase_b_index = _hebrew_dataset_index(phase_b, context="Hebrew v3 Phase-B config")
    phase_a_revision = phase_a.datasets[phase_a_index].dataset_revision
    phase_b_revision = phase_b.datasets[phase_b_index].dataset_revision
    if phase_a_revision != HEBREW_V3_PROVISIONAL_DATASET_REVISION:
        raise UserInputError(
            "Hebrew v3 Phase-A config must use the provisional Hebrew revision 'main'"
        )
    if _IMMUTABLE_REVISION.fullmatch(phase_b_revision) is None:
        raise UserInputError("Hebrew v3 Phase-B config must use an immutable Hebrew dataset commit")

    expected_phase_b = phase_a.model_dump(mode="json")
    expected_datasets = expected_phase_b.get("datasets")
    if not isinstance(expected_datasets, list):  # pragma: no cover - guaranteed by config type
        raise AssertionError("validated config did not serialize datasets as a list")
    expected_hebrew = expected_datasets[phase_a_index]
    if not isinstance(expected_hebrew, dict):  # pragma: no cover - guaranteed by config type
        raise AssertionError("validated Hebrew dataset did not serialize as an object")
    expected_hebrew["dataset_revision"] = phase_b_revision
    if expected_phase_b != phase_b.model_dump(mode="json"):
        raise UserInputError(
            "Hebrew v3 Phase-B config differs from Phase A by more than the published "
            "Hebrew dataset revision",
            hint=(
                "Restore every Phase-A field, including the reviewer anchor, and change only "
                "datasets[he].dataset_revision to the verified immutable commit."
            ),
        )
    return phase_a_reviewer
