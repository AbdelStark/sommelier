from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, TypedDict

from sommelier.artifacts import write_artifact_atomic
from sommelier.config import SommelierConfig
from sommelier.errors import ExternalDependencyError, SecurityPolicyError
from sommelier.redaction import scan_artifact_tree
from sommelier.security import redact_text

PREFLIGHT_SCHEMA: Final = "sommelier.release_preflight.v1"
PREFLIGHT_FILENAME: Final = "release_preflight.json"

ACK_ENV_NAME: Final = "SOMMELIER_ACK_BASE_MODEL_LICENSE"

# The Llama 3.1 Community License requires this notice on derived
# artifacts; licenses/THIRD_PARTY.md records it for the configured base
# model.
REQUIRED_DERIVED_NOTICE: Final = "Built with Llama"

GateStatus = Literal["pass", "fail", "skip"]


class ReleaseGate(TypedDict):
    name: str
    status: GateStatus
    evidence: str


class PreflightReport(TypedDict):
    schema_version: str
    created_at: str
    status: Literal["pass", "fail"]
    gates: list[ReleaseGate]


def _gate(name: str, status: GateStatus, evidence: str) -> ReleaseGate:
    # Evidence strings can contain filesystem paths; redact home
    # directories so the written report passes its own secret scan.
    return ReleaseGate(name=name, status=status, evidence=redact_text(evidence))


def build_release_gates(
    config: SommelierConfig,
    *,
    project_root: Path,
    artifact_root: Path,
    environ: Mapping[str, str] | None = None,
) -> list[ReleaseGate]:
    """Evaluates every release gate without raising (RFC-0011)."""
    env = environ if environ is not None else os.environ
    gates: list[ReleaseGate] = []

    license_path = project_root / "LICENSE"
    gates.append(
        _gate(
            "project_license",
            "pass" if license_path.exists() else "fail",
            f"checked {license_path}",
        )
    )

    notices_path = project_root / "licenses" / "THIRD_PARTY.md"
    notices_text = ""
    if notices_path.exists():
        notices_text = notices_path.read_text(encoding="utf-8")
        gates.append(_gate("third_party_notices", "pass", f"checked {notices_path}"))
    else:
        gates.append(_gate("third_party_notices", "fail", f"missing {notices_path}"))

    base_model = config.model.base_model_id
    gates.append(
        _gate(
            "base_model_obligations",
            "pass" if base_model in notices_text else "fail",
            f"looked for {base_model!r} in third-party notices",
        )
    )
    dataset = config.dataset.dataset_id
    gates.append(
        _gate(
            "dataset_license",
            "pass" if dataset in notices_text else "fail",
            f"looked for {dataset!r} in third-party notices",
        )
    )
    gates.append(
        _gate(
            "derived_artifact_notice",
            "pass" if REQUIRED_DERIVED_NOTICE in notices_text else "fail",
            f"looked for required notice {REQUIRED_DERIVED_NOTICE!r} "
            "in third-party notices",
        )
    )

    acknowledged = env.get(ACK_ENV_NAME, "")
    gates.append(
        _gate(
            "base_model_license_ack",
            "pass" if acknowledged == base_model else "fail",
            f"{ACK_ENV_NAME} must be set to {base_model!r} to acknowledge "
            "the base model license terms",
        )
    )

    lock_path = project_root / "uv.lock"
    gates.append(
        _gate(
            "dependency_lock",
            "pass" if lock_path.exists() else "fail",
            f"checked {lock_path}",
        )
    )

    if not artifact_root.exists():
        gates.append(
            _gate(
                "artifact_secret_scan",
                "skip",
                f"no artifacts present under {artifact_root}",
            )
        )
    else:
        findings = scan_artifact_tree(artifact_root)
        if findings:
            first = findings[0]
            gates.append(
                _gate(
                    "artifact_secret_scan",
                    "fail",
                    f"{len(findings)} finding(s); first: {first['kind']} in "
                    f"{first['file']} at {first['location']}",
                )
            )
        else:
            gates.append(
                _gate(
                    "artifact_secret_scan",
                    "pass",
                    f"scanned artifacts under {artifact_root}",
                )
            )

    return gates


def run_release_preflight(
    config: SommelierConfig,
    *,
    project_root: Path,
    artifact_root: Path,
    environ: Mapping[str, str] | None = None,
) -> PreflightReport:
    """Writes release_preflight.json and fails closed on failing gates.

    A failing secret scan raises SecurityPolicyError (exit 5); any other
    failing gate raises ExternalDependencyError (exit 3), matching the
    license-gate contract in docs/spec/06-security.md. The report is
    written before raising so the evidence survives the failure.
    """
    gates = build_release_gates(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=environ,
    )
    failed = [gate for gate in gates if gate["status"] == "fail"]
    report = PreflightReport(
        schema_version=PREFLIGHT_SCHEMA,
        created_at=datetime.now(UTC).isoformat(),
        status="fail" if failed else "pass",
        gates=gates,
    )

    artifact_root.mkdir(parents=True, exist_ok=True)
    report_path = artifact_root / PREFLIGHT_FILENAME

    def writer(temp_path: Path) -> None:
        temp_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    write_artifact_atomic(report_path, writer)

    if failed:
        failed_names = ", ".join(gate["name"] for gate in failed)
        if any(gate["name"] == "artifact_secret_scan" for gate in failed):
            raise SecurityPolicyError(
                f"release preflight failed: {failed_names}",
                hint=f"See {report_path} for gate evidence.",
            )
        raise ExternalDependencyError(
            f"release preflight failed: {failed_names}",
            hint=f"See {report_path} for gate evidence; acknowledge the base "
            f"model license by setting {ACK_ENV_NAME}.",
        )
    return report
