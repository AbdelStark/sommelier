"""Canonical identity contract for a named human reviewer.

Public reviewer keys are configuration and provenance, not credentials.  This
module deliberately accepts only OpenSSH Ed25519 public keys and reduces them
to one comment-free representation so every downstream digest sees identical
bytes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Final

from sommelier.errors import UserInputError

REVIEWER_PREREGISTRATION_SCHEMA: Final = "sommelier.human_reviewer_preregistration.v1"

_REVIEWER_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,127}$")
_SSH_ED25519_PREFIX: Final = "ssh-ed25519"


@dataclass(frozen=True)
class ReviewerRequirement:
    """One stable public identity authorized to sign a human review."""

    reviewer_id: str
    ssh_public_key: str
    public_key_fingerprint: str


class ReviewerRequirementValidationError(ValueError):
    """A reviewer identity or public key is not canonical and usable."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


def canonical_reviewer_requirement(
    reviewer_id: str,
    ssh_public_key: str,
) -> ReviewerRequirement:
    """Validate and canonicalize one public Ed25519 reviewer identity."""
    if reviewer_id != reviewer_id.strip() or _REVIEWER_ID_PATTERN.fullmatch(reviewer_id) is None:
        raise ReviewerRequirementValidationError(
            "semantic review requires a safe stable reviewer id",
            hint=(
                "Use 1-128 ASCII letters, digits, dots, underscores, colons, at signs, "
                "plus signs, or hyphens; do not use a name containing spaces or secrets."
            ),
        )
    parts = ssh_public_key.strip().split()
    if len(parts) not in {2, 3} or parts[0] != _SSH_ED25519_PREFIX:
        raise ReviewerRequirementValidationError(
            "semantic review requires one OpenSSH Ed25519 public key",
            hint="Provide the reviewer's .pub value; private keys must never enter Sommelier.",
        )
    try:
        decoded = base64.b64decode(parts[1], validate=True)
    except (binascii.Error, ValueError) as error:
        raise ReviewerRequirementValidationError(
            "semantic-review public key is not valid base64"
        ) from error

    def field(offset: int) -> tuple[bytes, int]:
        if offset + 4 > len(decoded):
            raise ReviewerRequirementValidationError(
                "semantic-review public key has an invalid SSH wire format"
            )
        size = int.from_bytes(decoded[offset : offset + 4], "big")
        start = offset + 4
        end = start + size
        if end > len(decoded):
            raise ReviewerRequirementValidationError(
                "semantic-review public key has an invalid SSH wire format"
            )
        return decoded[start:end], end

    algorithm, offset = field(0)
    key_bytes, offset = field(offset)
    if (
        algorithm != _SSH_ED25519_PREFIX.encode("ascii")
        or len(key_bytes) != 32
        or offset != len(decoded)
    ):
        raise ReviewerRequirementValidationError(
            "semantic-review public key is not a canonical Ed25519 key"
        )
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(decoded).digest()).decode(
        "ascii"
    ).rstrip("=")
    return ReviewerRequirement(
        reviewer_id=reviewer_id,
        ssh_public_key=f"{_SSH_ED25519_PREFIX} {parts[1]}",
        public_key_fingerprint=fingerprint,
    )


def validated_reviewer_requirement(
    reviewer_id: str,
    ssh_public_key: str,
) -> ReviewerRequirement:
    """Public boundary wrapper using Sommelier's typed input error."""
    try:
        return canonical_reviewer_requirement(reviewer_id, ssh_public_key)
    except ReviewerRequirementValidationError as error:
        raise UserInputError(str(error), hint=error.hint) from error


def reviewer_preregistration_payload(
    requirement: ReviewerRequirement,
) -> dict[str, str]:
    """Return the closed canonical payload stored in launch identities."""
    canonical = canonical_reviewer_requirement(
        requirement.reviewer_id,
        requirement.ssh_public_key,
    )
    if requirement != canonical:
        raise ReviewerRequirementValidationError(
            "semantic-review reviewer requirement is not canonical"
        )
    return {
        "schema_version": REVIEWER_PREREGISTRATION_SCHEMA,
        "reviewer_id": requirement.reviewer_id,
        "ssh_public_key": requirement.ssh_public_key,
        "public_key_fingerprint": requirement.public_key_fingerprint,
    }


def reviewer_preregistration_sha256(requirement: ReviewerRequirement) -> str:
    """Digest the exact reviewer identity independently of YAML formatting."""
    encoded = json.dumps(
        reviewer_preregistration_payload(requirement),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
