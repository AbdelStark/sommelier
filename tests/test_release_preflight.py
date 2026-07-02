from __future__ import annotations

import json
from pathlib import Path

import pytest

from sommelier.config import SommelierConfig, load_config
from sommelier.errors import ExternalDependencyError, SecurityPolicyError
from sommelier.release import (
    ACK_ENV_NAME,
    PREFLIGHT_FILENAME,
    build_release_gates,
    run_release_preflight,
)

SMOKE_CONFIG = Path("examples/config.smoke.yaml")

FAKE_TOKEN = "hf_" + "a" * 30


def make_project(tmp_path: Path) -> tuple[SommelierConfig, Path, Path]:
    config = load_config(SMOKE_CONFIG)
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    licenses_dir = project_root / "licenses"
    licenses_dir.mkdir()
    (licenses_dir / "THIRD_PARTY.md").write_text(
        f"# Third-Party Notices\n\n"
        f"- Base model: {config.model.base_model_id}\n"
        f"- Required notice: Built with Llama\n"
        f"- Dataset: {config.dataset.dataset_id}\n",
        encoding="utf-8",
    )
    (project_root / "uv.lock").write_text("lock\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    return config, project_root, artifact_root


def ack_env(config: SommelierConfig) -> dict[str, str]:
    return {ACK_ENV_NAME: config.model.base_model_id}


def test_preflight_passes_and_writes_report(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    artifact_root.mkdir()
    (artifact_root / "report.md").write_text("clean\n", encoding="utf-8")

    report = run_release_preflight(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=ack_env(config),
    )

    assert report["status"] == "pass"
    on_disk = json.loads((artifact_root / PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "sommelier.release_preflight.v1"
    gate_names = [gate["name"] for gate in on_disk["gates"]]
    assert gate_names == [
        "project_license",
        "third_party_notices",
        "base_model_obligations",
        "dataset_license",
        "derived_artifact_notice",
        "base_model_license_ack",
        "dependency_lock",
        "artifact_secret_scan",
    ]
    assert all(gate["status"] == "pass" for gate in on_disk["gates"])


def test_missing_license_fails_with_exit_three(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    (project_root / "LICENSE").unlink()

    with pytest.raises(ExternalDependencyError) as excinfo:
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )
    assert excinfo.value.exit_code == 3
    assert "project_license" in str(excinfo.value)

    on_disk = json.loads((artifact_root / PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
    assert on_disk["status"] == "fail"


def test_missing_notices_fail_dependent_gates(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    (project_root / "licenses" / "THIRD_PARTY.md").unlink()

    gates = build_release_gates(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=ack_env(config),
    )
    by_name = {gate["name"]: gate["status"] for gate in gates}
    assert by_name["third_party_notices"] == "fail"
    assert by_name["base_model_obligations"] == "fail"
    assert by_name["dataset_license"] == "fail"
    assert by_name["derived_artifact_notice"] == "fail"


def test_missing_acknowledgement_fails_with_exit_three(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)

    with pytest.raises(ExternalDependencyError) as excinfo:
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ={},
        )
    assert excinfo.value.exit_code == 3
    assert "base_model_license_ack" in str(excinfo.value)


def test_wrong_acknowledgement_value_fails(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    gates = build_release_gates(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ={ACK_ENV_NAME: "some/other-model"},
    )
    by_name = {gate["name"]: gate["status"] for gate in gates}
    assert by_name["base_model_license_ack"] == "fail"


def test_detected_secret_fails_with_exit_five(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    artifact_root.mkdir()
    (artifact_root / "report.md").write_text(f"leak {FAKE_TOKEN}\n", encoding="utf-8")

    with pytest.raises(SecurityPolicyError) as excinfo:
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )
    assert excinfo.value.exit_code == 5

    on_disk = json.loads((artifact_root / PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
    scan_gate = next(
        gate for gate in on_disk["gates"] if gate["name"] == "artifact_secret_scan"
    )
    assert scan_gate["status"] == "fail"
    assert FAKE_TOKEN not in json.dumps(on_disk)


def test_absent_artifacts_skip_secret_scan(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)

    report = run_release_preflight(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=ack_env(config),
    )
    scan_gate = next(
        gate for gate in report["gates"] if gate["name"] == "artifact_secret_scan"
    )
    assert scan_gate["status"] == "skip"
    assert report["status"] == "pass"


def test_repo_preflight_against_real_files(tmp_path: Path) -> None:
    # The actual repository must satisfy every file-based gate.
    config = load_config(Path("examples/config.full.yaml"))
    gates = build_release_gates(
        config,
        project_root=Path.cwd(),
        artifact_root=tmp_path / "no-artifacts",
        environ=ack_env(config),
    )
    by_name = {gate["name"]: gate["status"] for gate in gates}
    assert by_name["project_license"] == "pass"
    assert by_name["third_party_notices"] == "pass"
    assert by_name["base_model_obligations"] == "pass"
    assert by_name["dataset_license"] == "pass"
    assert by_name["derived_artifact_notice"] == "pass"
    assert by_name["dependency_lock"] == "pass"
