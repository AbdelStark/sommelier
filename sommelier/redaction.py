from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, TypedDict

from sommelier.errors import SecurityPolicyError
from sommelier.security import (
    MIN_ENV_SECRET_LENGTH,
    REDACTED_PLACEHOLDER,
    SECRET_TEXT_PATTERNS,
    SENSITIVE_ENV_NAME_MARKERS,
    scan_mapping_for_secrets,
)

FindingKind = Literal[
    "sensitive_key",
    "secret_value",
    "sensitive_env_value",
    "home_path",
    "duplicate_key",
]

SCANNABLE_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"})


class RedactionFinding(TypedDict):
    file: str
    location: str
    kind: FindingKind
    detail: str


class DuplicateJsonKeyError(ValueError):
    """Raised when a JSON object would otherwise silently shadow an earlier key."""


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            # Never include the key: a token-shaped key is itself sensitive.
            raise DuplicateJsonKeyError("duplicate JSON object key")
        payload[key] = value
    return payload


def loads_unique_json(text: str) -> Any:
    """Decode JSON while rejecting duplicate keys at every object depth."""
    return json.loads(text, object_pairs_hook=reject_duplicate_json_keys)


def _sensitive_env_values() -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for name, value in os.environ.items():
        if len(value) < MIN_ENV_SECRET_LENGTH:
            continue
        upper_name = name.upper()
        if any(marker in upper_name for marker in SENSITIVE_ENV_NAME_MARKERS):
            values.append((name, value))
    return values


def _finding(file: str, location: str, kind: FindingKind, detail: str) -> RedactionFinding:
    return RedactionFinding(file=file, location=location, kind=kind, detail=detail)


def scan_text_for_secrets(text: str, *, file: str) -> list[RedactionFinding]:
    findings: list[RedactionFinding] = []
    env_values = _sensitive_env_values()
    home = Path.home().as_posix()
    for line_number, line in enumerate(text.splitlines(), start=1):
        location = f"line {line_number}"
        for pattern in SECRET_TEXT_PATTERNS:
            if pattern.search(line):
                findings.append(
                    _finding(file, location, "secret_value", "token-like value in text")
                )
                break
        for name, value in env_values:
            if value in line:
                findings.append(
                    _finding(
                        file,
                        location,
                        "sensitive_env_value",
                        f"value of environment variable {name} in text",
                    )
                )
        if len(home) > 1 and home in line:
            findings.append(_finding(file, location, "home_path", "home directory path in text"))
    return findings


def scan_json_payload(payload: Any, *, file: str) -> list[RedactionFinding]:
    findings: list[RedactionFinding] = []
    for violation in scan_mapping_for_secrets(payload):
        if not violation.startswith("sensitive key"):
            # Anchored secret-value matches are a subset of the substring
            # scan below; only key findings are unique to this pass.
            continue
        findings.append(_finding(file, violation, "sensitive_key", violation))
    findings.extend(_scan_json_strings(payload, file=file, path=""))
    return findings


def _scan_json_strings(payload: Any, *, file: str, path: str) -> list[RedactionFinding]:
    findings: list[RedactionFinding] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{path}.{key}" if path else str(key)
            findings.extend(_scan_json_strings(str(key), file=file, path=f"{child}::<key>"))
            findings.extend(_scan_json_strings(value, file=file, path=child))
        return findings
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            findings.extend(_scan_json_strings(item, file=file, path=f"{path}[{index}]"))
        return findings
    if isinstance(payload, str):
        location = path or "<root>"
        for pattern in SECRET_TEXT_PATTERNS:
            if pattern.search(payload):
                findings.append(
                    _finding(file, location, "secret_value", "token-like value in string")
                )
                break
        for name, value in _sensitive_env_values():
            if value in payload:
                findings.append(
                    _finding(
                        file,
                        location,
                        "sensitive_env_value",
                        f"value of environment variable {name} in string",
                    )
                )
        home = Path.home().as_posix()
        if len(home) > 1 and home in payload:
            findings.append(_finding(file, location, "home_path", "home directory path in string"))
    return findings


def _scan_json_document(text: str, *, file: str) -> list[RedactionFinding]:
    try:
        payload = loads_unique_json(text)
    except DuplicateJsonKeyError:
        # Duplicate keys are invalid for publication. Also scan the raw bytes so
        # an earlier value shadowed by the final key cannot hide a credential.
        return [
            _finding(
                file,
                "<json-object>",
                "duplicate_key",
                "duplicate JSON object key",
            ),
            *scan_text_for_secrets(text, file=file),
        ]
    except json.JSONDecodeError:
        return scan_text_for_secrets(text, file=file)
    return scan_json_payload(payload, file=file)


def scan_artifact_text(text: str, *, file: str, suffix: str) -> list[RedactionFinding]:
    """Scan one decoded artifact without allowing duplicate JSON keys to shadow data."""
    if suffix == ".json":
        return _scan_json_document(text, file=file)

    if suffix == ".jsonl":
        findings: list[RedactionFinding] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            findings.extend(_scan_json_document(stripped, file=f"{file}:{line_number}"))
        return findings

    return scan_text_for_secrets(text, file=file)


def scan_artifact_file(path: Path, *, base_dir: Path | None = None) -> list[RedactionFinding]:
    file = path.relative_to(base_dir).as_posix() if base_dir is not None else path.as_posix()
    text = path.read_text(encoding="utf-8")
    return scan_artifact_text(text, file=file, suffix=path.suffix.lower())


def scan_artifact_tree(root: Path) -> list[RedactionFinding]:
    """Scans logs, manifests, JSON/JSONL artifacts, and Markdown reports."""
    findings: list[RedactionFinding] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SCANNABLE_SUFFIXES:
            findings.extend(scan_artifact_file(path, base_dir=root))
    return findings


def redact_configured_fields(payload: Any, field_names: Iterable[str]) -> Any:
    """Replaces values of configured field names anywhere in a JSON tree."""
    names = set(field_names)
    if not names:
        return payload

    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: REDACTED_PLACEHOLDER if key in names else visit(value)
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [visit(item) for item in node]
        return node

    return visit(payload)


def assert_artifacts_publishable(root: Path) -> None:
    """Fails closed when any scannable artifact under root contains secrets.

    Raises SecurityPolicyError (exit code 5) listing the first findings so
    release preflight can refuse to publish.
    """
    findings = scan_artifact_tree(root)
    if not findings:
        return
    first = findings[0]
    raise SecurityPolicyError(
        f"artifacts under {root} contain {len(findings)} redaction finding(s); "
        f"first: {first['kind']} in {first['file']} at {first['location']}",
        hint="Remove or redact the flagged values, then rerun the scan.",
    )
