from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import sommelier.release as release
from sommelier.config import DatasetSourceConfig, SommelierConfig, load_config
from sommelier.errors import ExternalDependencyError, InvariantViolation, SecurityPolicyError
from sommelier.redaction import RedactionFinding
from sommelier.release import (
    ACK_ENV_NAME,
    PREFLIGHT_FILENAME,
    build_release_gates,
    certify_artifact_tree,
    normalized_config_bytes,
    run_release_preflight,
    validate_release_preflight_report,
)

SMOKE_CONFIG = Path("examples/config.smoke.yaml")

FAKE_TOKEN = "hf_" + "a" * 30


def make_project(
    tmp_path: Path,
    *,
    initialize_git: bool = True,
) -> tuple[SommelierConfig, Path, Path]:
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
        f"- Dataset: {config.root_dataset.dataset_id}\n",
        encoding="utf-8",
    )
    (project_root / "uv.lock").write_text("lock\n", encoding="utf-8")
    if initialize_git:
        initialize_git_project(project_root)
    artifact_root = tmp_path / "artifacts"
    return config, project_root, artifact_root


def ack_env(config: SommelierConfig) -> dict[str, str]:
    return {ACK_ENV_NAME: config.model.base_model_id}


def initialize_git_project(project_root: Path, *, add: tuple[str, ...] = (".",)) -> str:
    subprocess.run(["git", "init", "--quiet", str(project_root)], check=True)
    subprocess.run(
        ["git", "-C", str(project_root), "config", "user.email", "release@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "config", "user.name", "Release Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(project_root), "add", "--", *add], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
    assert on_disk["schema_version"] == "sommelier.release_preflight.v2"
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

    identity = on_disk["identity"]
    normalized = normalized_config_bytes(config)
    assert identity["config"] == {
        "normalization": "pydantic-json-canonical-v1",
        "sha256": hashlib.sha256(normalized).hexdigest(),
        "bytes": len(normalized),
    }
    assert identity["model"] == {
        "base_model_id": config.model.base_model_id,
        "base_model_revision": config.model.base_model_revision,
        "base_model_revision_is_immutable": True,
        "tokenizer_revision": config.model.tokenizer_revision,
        "tokenizer_revision_is_immutable": True,
    }
    assert identity["datasets"] == [
        {
            "language": "en",
            "role": "root",
            "dataset_id": config.root_dataset.dataset_id,
            "dataset_revision": config.root_dataset.dataset_revision,
            "revision_is_immutable": True,
        }
    ]
    assert identity["dependency_lock"] == {
        "path": "uv.lock",
        "sha256": hashlib.sha256(b"lock\n").hexdigest(),
        "bytes": 5,
    }
    assert identity["artifact_tree"] == certify_artifact_tree(artifact_root)
    assert identity["artifact_tree"]["file_count"] == 1
    assert identity["artifact_tree"]["total_bytes"] == len(b"clean\n")
    validate_release_preflight_report(
        on_disk,
        config=config,
        artifact_root=artifact_root,
        project_root=project_root,
        require_pass=True,
    )


def test_v2_report_cannot_be_replayed_across_config_or_artifact_tree(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    artifact_root.mkdir()
    (artifact_root / "result.json").write_text('{"value": 1}\n', encoding="utf-8")
    report = run_release_preflight(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=ack_env(config),
    )

    other_config = config.model_copy(deep=True)
    other_config.project.seed += 1
    with pytest.raises(InvariantViolation, match="config identity"):
        validate_release_preflight_report(report, config=other_config)

    other_root = tmp_path / "other-artifacts"
    other_root.mkdir()
    (other_root / "result.json").write_text('{"value": 2}\n', encoding="utf-8")
    (other_root / PREFLIGHT_FILENAME).write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(InvariantViolation, match="artifact-tree identity"):
        validate_release_preflight_report(
            report,
            config=config,
            artifact_root=other_root,
            require_pass=True,
        )


def test_strict_publication_profile_binds_clean_source_lock_and_revisions(
    tmp_path: Path,
) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    revision = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    artifact_root.mkdir()
    (artifact_root / "adapter.safetensors").write_bytes(b"adapter")

    report = run_release_preflight(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=ack_env(config),
    )

    source = report["identity"]["source_code"]
    assert source == {
        "discovery": "git-project-root-v1",
        "git_commit": revision,
        "working_tree_clean": True,
        "git_status_sha256": hashlib.sha256(b"").hexdigest(),
    }
    validate_release_preflight_report(
        report,
        config=config,
        artifact_root=artifact_root,
        project_root=project_root,
        expected_source_git_commit=revision,
        expected_dependency_lock_sha256=hashlib.sha256(b"lock\n").hexdigest(),
        require_pass=True,
        require_all_gates_pass=True,
        require_clean_source=True,
        require_immutable_revisions=True,
    )

    with pytest.raises(InvariantViolation, match="source revision"):
        validate_release_preflight_report(
            report,
            config=config,
            expected_source_git_commit="f" * 40,
        )
    with pytest.raises(InvariantViolation, match="dependency lock does not match"):
        validate_release_preflight_report(
            report,
            config=config,
            expected_dependency_lock_sha256="f" * 64,
        )


def test_source_discovery_rejects_ignored_project_nested_in_parent_repo(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / ".gitignore").write_text("producer/\n", encoding="utf-8")
    initialize_git_project(outer, add=(".gitignore",))
    producer = outer / "producer"
    producer.mkdir()
    (producer / "source.py").write_text("not present in the parent commit\n", encoding="utf-8")

    identity = release._discover_project_source(producer)

    assert identity == {
        "discovery": "unavailable",
        "git_commit": "unknown",
        "working_tree_clean": None,
        "git_status_sha256": None,
    }


def test_untracked_dependency_lock_fails_and_persists_report(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path, initialize_git=False)
    initialize_git_project(project_root, add=("LICENSE", "licenses/THIRD_PARTY.md"))
    artifact_root.mkdir()

    with pytest.raises(ExternalDependencyError, match="dependency_lock"):
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )

    report = json.loads((artifact_root / PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
    lock_gate = next(gate for gate in report["gates"] if gate["name"] == "dependency_lock")
    assert lock_gate["status"] == "fail"
    assert report["identity"]["dependency_lock"] == {
        "path": "uv.lock",
        "sha256": None,
        "bytes": None,
    }


def test_dependency_lock_symlink_is_rejected_without_reading_target(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    external_lock = tmp_path / "external.lock"
    external_lock.write_text("external dependency bytes\n", encoding="utf-8")
    (project_root / "uv.lock").unlink()
    (project_root / "uv.lock").symlink_to(external_lock)
    artifact_root.mkdir()

    with pytest.raises(ExternalDependencyError, match="dependency_lock"):
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )

    report = json.loads((artifact_root / PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
    assert report["identity"]["dependency_lock"]["sha256"] is None


def test_dependency_lock_must_match_tracked_commit_even_when_status_hides_change(
    tmp_path: Path,
) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    subprocess.run(
        ["git", "-C", str(project_root), "update-index", "--assume-unchanged", "uv.lock"],
        check=True,
    )
    (project_root / "uv.lock").write_text("locally substituted lock\n", encoding="utf-8")
    artifact_root.mkdir()

    with pytest.raises(ExternalDependencyError, match="dependency_lock"):
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )


def test_skip_worktree_source_change_cannot_claim_clean_commit(tmp_path: Path) -> None:
    _, project_root, _ = make_project(tmp_path)
    source_path = project_root / "producer.py"
    source_path.write_text("ORIGINAL = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project_root), "add", "producer.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "producer",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "update-index", "--skip-worktree", "producer.py"],
        check=True,
    )
    source_path.write_text("ORIGINAL = False\n", encoding="utf-8")

    identity = release._discover_project_source(project_root)

    assert identity["discovery"] == "git-project-root-partial-v1"
    assert identity["working_tree_clean"] is None


def test_dependency_lock_fails_closed_outside_certifiable_git_repo(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path, initialize_git=False)

    with pytest.raises(ExternalDependencyError, match="dependency_lock"):
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )

    report = json.loads((artifact_root / PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
    dependency_gate = next(gate for gate in report["gates"] if gate["name"] == "dependency_lock")
    assert dependency_gate["status"] == "fail"


def test_clean_crlf_filtered_checkout_is_certifiable(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path, initialize_git=False)
    subprocess.run(["git", "init", "--quiet", str(project_root)], check=True)
    subprocess.run(
        ["git", "-C", str(project_root), "config", "user.email", "release@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "config", "user.name", "Release Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project_root), "config", "core.autocrlf", "true"],
        check=True,
    )
    subprocess.run(["git", "-C", str(project_root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )
    for relative_path in ("LICENSE", "licenses/THIRD_PARTY.md", "uv.lock"):
        (project_root / relative_path).unlink()
    subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "checkout",
            "--quiet",
            "--",
            "LICENSE",
            "licenses/THIRD_PARTY.md",
            "uv.lock",
        ],
        check=True,
    )

    report = run_release_preflight(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=ack_env(config),
    )

    assert report["status"] == "pass"
    assert report["identity"]["source_code"]["discovery"] == "git-project-root-v1"
    assert (
        report["identity"]["dependency_lock"]["sha256"]
        == hashlib.sha256((project_root / "uv.lock").read_bytes()).hexdigest()
    )


def test_notice_gate_bytes_must_match_certified_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    notices_path = project_root / "licenses" / "THIRD_PARTY.md"
    committed_notices = "# Notices intentionally missing configured obligations\n"
    notices_path.write_text(committed_notices, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(project_root), "add", "licenses/THIRD_PARTY.md"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "incomplete notices",
        ],
        check=True,
    )
    notices_path.write_text(
        f"{config.model.base_model_id}\n{config.root_dataset.dataset_id}\nBuilt with Llama\n",
        encoding="utf-8",
    )
    original_discovery = release._discover_project_source

    def restore_then_discover(project: Path) -> release.SourceCodeIdentity:
        notices_path.write_text(committed_notices, encoding="utf-8")
        return original_discovery(project)

    monkeypatch.setattr(release, "_discover_project_source", restore_then_discover)

    with pytest.raises(ExternalDependencyError, match="third_party_notices"):
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )

    report = json.loads((artifact_root / PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
    by_name = {gate["name"]: gate["status"] for gate in report["gates"]}
    assert by_name["third_party_notices"] == "fail"
    assert by_name["base_model_obligations"] == "fail"
    assert by_name["dataset_license"] == "fail"
    assert by_name["derived_artifact_notice"] == "fail"


def test_v1_report_is_not_accepted_by_strict_validator(tmp_path: Path) -> None:
    config, _, _ = make_project(tmp_path)
    with pytest.raises(InvariantViolation, match="contract drift|schema drift"):
        validate_release_preflight_report(
            {
                "schema_version": "sommelier.release_preflight.v1",
                "created_at": "2026-01-01T00:00:00+00:00",
                "status": "pass",
                "gates": [],
            },
            config=config,
        )


def test_all_dataset_sources_are_bound_and_mutable_revision_is_rejected(
    tmp_path: Path,
) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    config.datasets.append(
        DatasetSourceConfig(
            language="he",
            dataset_id="owner/paired-hebrew",
            dataset_revision="e" * 40,
            source_id_column="source_example_id",
        )
    )
    artifact_root.mkdir()
    report = run_release_preflight(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=ack_env(config),
    )

    assert report["identity"]["datasets"] == [
        {
            "language": "en",
            "role": "root",
            "dataset_id": config.datasets[0].dataset_id,
            "dataset_revision": config.datasets[0].dataset_revision,
            "revision_is_immutable": True,
        },
        {
            "language": "he",
            "role": "paired",
            "dataset_id": "owner/paired-hebrew",
            "dataset_revision": "e" * 40,
            "revision_is_immutable": True,
        },
    ]
    validate_release_preflight_report(
        report,
        config=config,
        require_immutable_revisions=True,
    )

    mutable = config.model_copy(deep=True)
    mutable.datasets[1].dataset_revision = "main"
    mutable_report = run_release_preflight(
        mutable,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=ack_env(mutable),
    )
    with pytest.raises(InvariantViolation, match="mutable input revision"):
        validate_release_preflight_report(
            mutable_report,
            config=mutable,
            require_immutable_revisions=True,
        )


def test_unsafe_artifact_tree_writes_failed_report_before_raising(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    artifact_root.mkdir()
    target = artifact_root / "target.txt"
    target.write_text("content\n", encoding="utf-8")
    (artifact_root / "alias.txt").symlink_to(target)

    with pytest.raises(SecurityPolicyError, match="artifact_secret_scan"):
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )

    report = json.loads((artifact_root / PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["identity"]["artifact_tree"]["certified"] is False
    assert report["identity"]["artifact_tree"]["sha256"] is None
    scan = next(gate for gate in report["gates"] if gate["name"] == "artifact_secret_scan")
    assert scan["status"] == "fail"


def test_preflight_rejects_duplicate_json_that_shadows_a_secret(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    artifact_root.mkdir()
    (artifact_root / "manifest.json").write_text(
        f'{{"note":"{FAKE_TOKEN}","note":"clean"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(SecurityPolicyError, match="artifact_secret_scan"):
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )

    report = json.loads((artifact_root / PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
    scan = next(gate for gate in report["gates"] if gate["name"] == "artifact_secret_scan")
    assert scan["status"] == "fail"


def test_symlink_artifact_root_never_redirects_failed_report(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_report = outside / PREFLIGHT_FILENAME
    outside_report.write_text("do not replace\n", encoding="utf-8")
    artifact_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SecurityPolicyError, match="not written|cannot be written safely"):
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )

    assert artifact_root.is_symlink()
    assert outside_report.read_text(encoding="utf-8") == "do not replace\n"


def test_non_directory_artifact_root_is_preserved_without_report(tmp_path: Path) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    artifact_root.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(SecurityPolicyError, match="not written|cannot be written safely"):
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )

    assert artifact_root.read_text(encoding="utf-8") == "not a directory\n"


def test_artifact_mutation_between_scan_and_certification_fails_with_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    artifact_root.mkdir()
    artifact = artifact_root / "report.md"
    artifact.write_text("clean\n", encoding="utf-8")
    original_scan = release._scan_artifact_bytes

    def mutate_after_scan(data: bytes, *, relative_path: str) -> list[RedactionFinding]:
        findings = original_scan(data, relative_path=relative_path)
        artifact.write_text(f"leak {FAKE_TOKEN}\n", encoding="utf-8")
        return findings

    monkeypatch.setattr(release, "_scan_artifact_bytes", mutate_after_scan)
    monkeypatch.setattr(
        release,
        "_artifact_snapshot_signature",
        lambda _root, _entries: (),
    )

    with pytest.raises(SecurityPolicyError, match="artifact_secret_scan"):
        run_release_preflight(
            config,
            project_root=project_root,
            artifact_root=artifact_root,
            environ=ack_env(config),
        )

    report = json.loads((artifact_root / PREFLIGHT_FILENAME).read_text(encoding="utf-8"))
    scan = next(gate for gate in report["gates"] if gate["name"] == "artifact_secret_scan")
    assert scan["status"] == "fail"
    assert report["identity"]["artifact_tree"]["certified"] is False


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
    scan_gate = next(gate for gate in on_disk["gates"] if gate["name"] == "artifact_secret_scan")
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
    scan_gate = next(gate for gate in report["gates"] if gate["name"] == "artifact_secret_scan")
    assert scan_gate["status"] == "skip"
    assert report["status"] == "pass"


@pytest.mark.parametrize("malformed_field", ["gate_status", "dataset_role", "source_discovery"])
def test_closed_validator_maps_unhashable_values_to_invariant_violation(
    tmp_path: Path,
    malformed_field: str,
) -> None:
    config, project_root, artifact_root = make_project(tmp_path)
    artifact_root.mkdir()
    report = run_release_preflight(
        config,
        project_root=project_root,
        artifact_root=artifact_root,
        environ=ack_env(config),
    )
    malformed = copy.deepcopy(report)
    if malformed_field == "gate_status":
        malformed["gates"][0]["status"] = []  # type: ignore[typeddict-item]
    elif malformed_field == "dataset_role":
        malformed["identity"]["datasets"][0]["role"] = []  # type: ignore[typeddict-item]
    else:
        malformed["identity"]["source_code"]["discovery"] = []  # type: ignore[typeddict-item]

    with pytest.raises(InvariantViolation):
        validate_release_preflight_report(malformed, config=config)


def test_config_canonicalization_rejects_non_finite_mutation() -> None:
    config = load_config(SMOKE_CONFIG)
    config.train.learning_rate = float("nan")

    with pytest.raises(InvariantViolation, match="finite"):
        normalized_config_bytes(config)


def test_repo_preflight_against_real_files(tmp_path: Path) -> None:
    # Live-checkout gates must reflect whether each required file is the
    # filtered working-tree form tracked by the currently certified commit.
    config = load_config(Path("examples/config.full.yaml"))
    gates = build_release_gates(
        config,
        project_root=Path.cwd(),
        artifact_root=tmp_path / "no-artifacts",
        environ=ack_env(config),
    )
    by_name = {gate["name"]: gate["status"] for gate in gates}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def matches_head(relative_path: str) -> bool:
        path = Path(relative_path)
        return (
            not path.is_symlink()
            and path.is_file()
            and release._project_file_matches_revision(
                Path.cwd(),
                revision,
                relative_path,
                path.read_bytes(),
            )
        )

    license_matches_head = matches_head("LICENSE")
    notices_match_head = matches_head("licenses/THIRD_PARTY.md")
    lock_matches_head = matches_head("uv.lock")
    assert by_name["project_license"] == ("pass" if license_matches_head else "fail")
    assert by_name["third_party_notices"] == ("pass" if notices_match_head else "fail")
    for gate_name in ("base_model_obligations", "dataset_license", "derived_artifact_notice"):
        assert by_name[gate_name] == ("pass" if notices_match_head else "fail")
    assert by_name["dependency_lock"] == ("pass" if lock_matches_head else "fail")
