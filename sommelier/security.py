from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from sommelier.errors import SecurityPolicyError

ALLOWED_SECRET_ENV_NAMES = frozenset({"HF_TOKEN", "WANDB_API_KEY"})

SAFE_KEY_SUFFIXES = frozenset(
    {
        "dedupe_key",
        "query_column",
        "tools_column",
        "answers_column",
        "redact_fields",
        "target_modules",
    }
)

SENSITIVE_KEY_PATTERN = re.compile(
    r"(^|_)(token|secret|password|api_key|apikey)(_|$)",
    re.IGNORECASE,
)

SECRET_VALUE_PATTERNS = (
    re.compile(r"^hf_[A-Za-z0-9]{20,}$"),
    re.compile(r"^sk-[A-Za-z0-9\-_]{20,}$"),
    re.compile(r"^ghp_[A-Za-z0-9]{20,}$"),
    re.compile(r"^xox[baprs]-[A-Za-z0-9\-]{10,}$"),
)

SECRET_TEXT_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
)

SENSITIVE_ENV_NAME_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD")

REDACTED_PLACEHOLDER = "[redacted]"

MIN_ENV_SECRET_LENGTH = 8


def redact_text(text: str) -> str:
    """Redacts token-like values, sensitive env values, and home paths.

    This is the write-time redaction applied to log messages and other
    security-sensitive text artifacts per docs/spec/05-observability.md.
    """
    redacted = text
    for pattern in SECRET_TEXT_PATTERNS:
        redacted = pattern.sub(REDACTED_PLACEHOLDER, redacted)
    for name, value in os.environ.items():
        if len(value) < MIN_ENV_SECRET_LENGTH:
            continue
        upper_name = name.upper()
        if any(marker in upper_name for marker in SENSITIVE_ENV_NAME_MARKERS):
            redacted = redacted.replace(value, REDACTED_PLACEHOLDER)
    home = Path.home().as_posix()
    if len(home) > 1:
        redacted = redacted.replace(home, "~")
    return redacted


def _is_sensitive_key(key: str) -> bool:
    if key in SAFE_KEY_SUFFIXES:
        return False
    return bool(SENSITIVE_KEY_PATTERN.search(key))


def _is_secret_value(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    return any(pattern.match(stripped) for pattern in SECRET_VALUE_PATTERNS)


def scan_mapping_for_secrets(
    payload: Any,
    *,
    path: str = "",
) -> list[str]:
    violations: list[str] = []

    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}" if path else key
            if _is_sensitive_key(key):
                violations.append(f"sensitive key at {child_path}")
            violations.extend(scan_mapping_for_secrets(value, path=child_path))
        return violations

    if isinstance(payload, list):
        for index, item in enumerate(payload):
            violations.extend(scan_mapping_for_secrets(item, path=f"{path}[{index}]"))
        return violations

    if isinstance(payload, str) and _is_secret_value(payload):
        violations.append(f"suspected secret value at {path or '<root>'}")

    return violations


def validate_no_secrets(payload: Any, *, context: str) -> None:
    violations = scan_mapping_for_secrets(payload)
    if not violations:
        return
    raise SecurityPolicyError(
        f"{context} contains suspected secrets: {violations[0]}",
        hint="Provide secrets through environment variables or the remote secret store.",
    )
