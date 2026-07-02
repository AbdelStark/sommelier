from __future__ import annotations

import json
from pathlib import Path

import pytest

from sommelier.errors import SecurityPolicyError
from sommelier.redaction import (
    assert_artifacts_publishable,
    redact_configured_fields,
    scan_artifact_file,
    scan_artifact_tree,
)

FAKE_TOKEN = "hf_" + "a" * 30


def test_scanner_flags_token_in_markdown_report(tmp_path: Path) -> None:
    report = tmp_path / "comparison_report.md"
    report.write_text(f"# Report\n\nAuth used {FAKE_TOKEN}.\n", encoding="utf-8")

    findings = scan_artifact_file(report, base_dir=tmp_path)
    assert len(findings) == 1
    assert findings[0]["kind"] == "secret_value"
    assert findings[0]["file"] == "comparison_report.md"
    assert findings[0]["location"] == "line 3"


def test_scanner_flags_secret_in_json_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"run_id": "r1", "notes": f"token {FAKE_TOKEN}"}),
        encoding="utf-8",
    )

    findings = scan_artifact_file(manifest, base_dir=tmp_path)
    assert [finding["kind"] for finding in findings] == ["secret_value"]
    assert findings[0]["location"] == "notes"


def test_scanner_flags_sensitive_key_in_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"api_key": "value"}), encoding="utf-8")

    findings = scan_artifact_file(manifest, base_dir=tmp_path)
    assert findings[0]["kind"] == "sensitive_key"


def test_scanner_flags_jsonl_line_numbers(tmp_path: Path) -> None:
    log = tmp_path / "data.jsonl"
    log.write_text(
        json.dumps({"message": "clean"}) + "\n" + json.dumps({"message": FAKE_TOKEN}) + "\n",
        encoding="utf-8",
    )

    findings = scan_artifact_file(log, base_dir=tmp_path)
    assert len(findings) == 1
    assert findings[0]["file"] == "data.jsonl:2"


def test_scanner_flags_sensitive_env_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOMMELIER_TEST_TOKEN", "super-secret-value-123")
    report = tmp_path / "report.md"
    report.write_text("cost: super-secret-value-123\n", encoding="utf-8")

    findings = scan_artifact_file(report, base_dir=tmp_path)
    assert any(finding["kind"] == "sensitive_env_value" for finding in findings)


def test_scanner_flags_home_directory_paths(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text(f"artifact at {Path.home().as_posix()}/artifacts\n", encoding="utf-8")

    findings = scan_artifact_file(report, base_dir=tmp_path)
    assert any(finding["kind"] == "home_path" for finding in findings)


def test_scan_tree_covers_logs_manifests_and_reports(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "train.jsonl").write_text(
        json.dumps({"message": FAKE_TOKEN}) + "\n", encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"password_hint": "x"}), encoding="utf-8")
    (tmp_path / "report.md").write_text(f"secret {FAKE_TOKEN}\n", encoding="utf-8")
    (tmp_path / "weights.bin").write_bytes(b"\x00\x01")

    findings = scan_artifact_tree(tmp_path)
    files = {finding["file"].split(":")[0] for finding in findings}
    assert files == {"logs/train.jsonl", "manifest.json", "report.md"}


def test_clean_tree_passes(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "r1"}), encoding="utf-8")
    (tmp_path / "report.md").write_text("# Clean report\n", encoding="utf-8")

    assert scan_artifact_tree(tmp_path) == []
    assert_artifacts_publishable(tmp_path)


def test_publish_gate_fails_on_findings(tmp_path: Path) -> None:
    (tmp_path / "report.md").write_text(f"leak {FAKE_TOKEN}\n", encoding="utf-8")

    with pytest.raises(SecurityPolicyError) as excinfo:
        assert_artifacts_publishable(tmp_path)
    assert excinfo.value.exit_code == 5
    assert "report.md" in str(excinfo.value)


def test_redact_configured_fields_replaces_values() -> None:
    payload = {
        "metrics": {"argument_f1": 0.5},
        "internal_notes": "customer schema",
        "rows": [{"internal_notes": "row detail", "keep": 1}],
    }

    redacted = redact_configured_fields(payload, ["internal_notes"])

    assert redacted["internal_notes"] == "[redacted]"
    assert redacted["rows"][0]["internal_notes"] == "[redacted]"
    assert redacted["rows"][0]["keep"] == 1
    assert redacted["metrics"]["argument_f1"] == 0.5
    assert payload["internal_notes"] == "customer schema"


def test_redact_configured_fields_noop_without_names() -> None:
    payload = {"a": 1}
    assert redact_configured_fields(payload, []) == {"a": 1}
