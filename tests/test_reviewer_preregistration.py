from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from sommelier.config import SommelierConfig, load_config
from sommelier.errors import UserInputError
from sommelier.hebrew_v3_preregistration import (
    require_committed_config_bytes,
    require_preregistered_reviewer,
    reviewer_anchor_payload,
    reviewer_anchor_sha256,
    validate_hebrew_v3_phase_transition,
)
from sommelier.reviewer import (
    REVIEWER_PREREGISTRATION_SCHEMA,
    canonical_reviewer_requirement,
    reviewer_preregistration_payload,
    reviewer_preregistration_sha256,
    validated_reviewer_requirement,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
REVIEWER_PUBLIC_KEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f"
)
REVIEWER = canonical_reviewer_requirement("fixture-reviewer", REVIEWER_PUBLIC_KEY)


def _reviewer_section(*, reviewer_id: str = "fixture-reviewer") -> dict[str, object]:
    return {
        "reviewer": {
            "reviewer_id": reviewer_id,
            "ssh_public_key": REVIEWER.ssh_public_key,
            "public_key_fingerprint": REVIEWER.public_key_fingerprint,
        }
    }


def _write_config(path: Path, payload: dict[str, Any]) -> SommelierConfig:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_config(path)


def _phase_configs(tmp_path: Path) -> tuple[SommelierConfig, SommelierConfig]:
    phase_a_payload = yaml.safe_load(
        (EXAMPLES_DIR / "config.v3-he-full.yaml").read_text(encoding="utf-8")
    )
    phase_a_payload["semantic_review"] = _reviewer_section()
    phase_b_payload = copy.deepcopy(phase_a_payload)
    phase_b_payload["datasets"][1]["dataset_revision"] = "a" * 40
    return (
        _write_config(tmp_path / "phase-a.yaml", phase_a_payload),
        _write_config(tmp_path / "phase-b.yaml", phase_b_payload),
    )


def test_reviewer_requirement_has_stable_closed_payload_and_digest() -> None:
    config_payload = {
        "schema_version": REVIEWER_PREREGISTRATION_SCHEMA,
        "reviewer_id": "fixture-reviewer",
        "ssh_public_key": REVIEWER_PUBLIC_KEY,
        "public_key_fingerprint": REVIEWER.public_key_fingerprint,
    }

    assert reviewer_preregistration_payload(REVIEWER) == config_payload
    assert reviewer_preregistration_sha256(REVIEWER) == reviewer_preregistration_sha256(
        canonical_reviewer_requirement("fixture-reviewer", REVIEWER_PUBLIC_KEY)
    )
    assert len(reviewer_preregistration_sha256(REVIEWER)) == 64
    assert config_payload["public_key_fingerprint"].startswith("SHA256:")


def test_operational_reviewer_validator_uses_typed_user_input_error() -> None:
    with pytest.raises(UserInputError, match="OpenSSH Ed25519 public key"):
        validated_reviewer_requirement("fixture-reviewer", "not-a-public-key")


def test_missing_real_reviewer_is_explicit_admission_failure() -> None:
    config = load_config(EXAMPLES_DIR / "config.v3-he-full.yaml")

    with pytest.raises(
        UserInputError,
        match="missing its preregistered named human reviewer",
    ) as captured:
        require_preregistered_reviewer(config)

    assert captured.value.hint is not None
    assert "commit it before translation" in captured.value.hint
    assert "placeholder" in captured.value.hint


def test_phase_transition_accepts_only_dataset_pin_and_preserves_anchor(
    tmp_path: Path,
) -> None:
    phase_a, phase_b = _phase_configs(tmp_path)

    assert validate_hebrew_v3_phase_transition(phase_a, phase_b) == REVIEWER
    assert reviewer_anchor_payload(phase_a) == reviewer_anchor_payload(phase_b)
    assert reviewer_anchor_sha256(phase_a) == reviewer_anchor_sha256(phase_b)


def test_phase_transition_rejects_reviewer_identity_change(tmp_path: Path) -> None:
    phase_a, phase_b = _phase_configs(tmp_path)
    assert phase_b.semantic_review is not None
    phase_b_payload = phase_b.model_dump(mode="json")
    phase_b_payload["semantic_review"]["reviewer"]["reviewer_id"] = "replacement-reviewer"
    changed = _write_config(tmp_path / "phase-b-changed.yaml", phase_b_payload)

    with pytest.raises(UserInputError, match="reviewer preregistration changed"):
        validate_hebrew_v3_phase_transition(phase_a, changed)


def test_phase_transition_rejects_any_other_config_change(tmp_path: Path) -> None:
    phase_a, phase_b = _phase_configs(tmp_path)
    phase_b_payload = phase_b.model_dump(mode="json")
    phase_b_payload["train"]["epochs"] = 3
    changed = _write_config(tmp_path / "phase-b-changed.yaml", phase_b_payload)

    with pytest.raises(UserInputError, match="differs from Phase A by more than"):
        validate_hebrew_v3_phase_transition(phase_a, changed)


def test_phase_transition_requires_provisional_phase_a_revision(tmp_path: Path) -> None:
    phase_a, phase_b = _phase_configs(tmp_path)
    phase_a_payload = phase_a.model_dump(mode="json")
    phase_a_payload["datasets"][1]["dataset_revision"] = "b" * 40
    changed = _write_config(tmp_path / "phase-a-changed.yaml", phase_a_payload)

    with pytest.raises(UserInputError, match="Phase-A config must use.*'main'"):
        validate_hebrew_v3_phase_transition(changed, phase_b)


@pytest.mark.parametrize("revision", ["main", "release-tag", "A" * 40, "a" * 39])
def test_phase_transition_requires_immutable_lowercase_phase_b_revision(
    tmp_path: Path,
    revision: str,
) -> None:
    phase_a, phase_b = _phase_configs(tmp_path)
    phase_b_payload = phase_b.model_dump(mode="json")
    phase_b_payload["datasets"][1]["dataset_revision"] = revision
    changed = _write_config(tmp_path / f"phase-b-{len(revision)}.yaml", phase_b_payload)

    with pytest.raises(UserInputError, match="immutable Hebrew dataset commit"):
        validate_hebrew_v3_phase_transition(phase_a, changed)


def _committed_config_fixture(tmp_path: Path) -> tuple[Path, str, bytes]:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=tmp_path,
        check=True,
    )
    config_path = tmp_path / "config.yaml"
    encoded = b"schema_version: sommelier.config.v2\n"
    config_path.write_bytes(encoded)
    subprocess.run(["git", "add", "config.yaml"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "config"], cwd=tmp_path, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return config_path, revision, encoded


def test_committed_config_boundary_accepts_exact_tracked_blob(tmp_path: Path) -> None:
    config_path, revision, encoded = _committed_config_fixture(tmp_path)

    assert (
        require_committed_config_bytes(
            config_path,
            code_revision=revision,
            context="test launch",
        )
        == encoded
    )


def test_committed_config_boundary_rejects_modified_or_untracked_bytes(tmp_path: Path) -> None:
    config_path, revision, _encoded = _committed_config_fixture(tmp_path)
    config_path.write_text("schema_version: changed\n", encoding="utf-8")

    with pytest.raises(UserInputError, match="bytes differ"):
        require_committed_config_bytes(
            config_path,
            code_revision=revision,
            context="test launch",
        )

    untracked = tmp_path / "untracked.yaml"
    untracked.write_text("schema_version: sommelier.config.v2\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="not tracked"):
        require_committed_config_bytes(
            untracked,
            code_revision=revision,
            context="test launch",
        )
