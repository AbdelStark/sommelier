from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import remote_qlora_preflight as remote
from sommelier.config import SommelierConfig
from sommelier.errors import UserInputError
from sommelier.remote.images import PIPELINE_HF_ENV
from sommelier.training.qlora_preflight import (
    GPU_ALLOCATION,
    SourceProvenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG_PATH = REPO_ROOT / "examples/config.v3-he-full.yaml"


def _source() -> SourceProvenance:
    return SourceProvenance(
        git_commit="a" * 40,
        working_tree_clean=True,
        git_status_sha256=hashlib.sha256(b"").hexdigest(),
        boundary="test launcher provenance",
    )


def test_modal_contract_is_fixed_l40s_no_retry_and_diagnostic_only() -> None:
    assert remote.APP_NAME == "sommelier-qlora-shape-preflight"
    assert remote.PREFLIGHT_MAX_RETRIES == 0
    assert remote.PREFLIGHT_TIMEOUT_SECONDS == 4 * 60 * 60
    assert GPU_ALLOCATION == "L40S"
    assert remote.HF_READ_SECRET_NAME == "huggingface-read-token"
    assert remote.HF_TOKEN_ENV == "HF_TOKEN"
    assert remote.ARTIFACTS_ROOT == Path("/artifacts/diagnostics/qlora-shape-preflight")


def test_source_provenance_hashes_status_without_publishing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = b" M private-name.txt\0?? another-name.txt\0"
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        assert kwargs == {"check": True, "capture_output": True, "timeout": 10}
        stdout = b"a" * 40 + b"\n" if command[1:3] == ["rev-parse", "HEAD"] else status
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    source = remote.local_source_provenance()

    assert source["git_commit"] == "a" * 40
    assert source["working_tree_clean"] is False
    assert source["git_status_sha256"] == hashlib.sha256(status).hexdigest()
    assert "private-name" not in source["boundary"]
    assert "path/status set" in source["boundary"]
    assert commands == [
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=normal"],
    ]


def test_git_measurement_failure_is_not_misclassified_as_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail_run)

    with pytest.raises(UserInputError, match="could not measure local Git provenance"):
        remote.local_source_provenance()


def test_remote_function_forces_hf_policy_commits_both_volumes_and_returns_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_yaml = FULL_CONFIG_PATH.read_text(encoding="utf-8")
    source = _source()
    captured: dict[str, object] = {}
    commits: list[str] = []

    def fake_runner(
        config: SommelierConfig,
        *,
        config_yaml: str,
        output_dir: Path,
        run_id: str,
        source: SourceProvenance,
    ) -> dict[str, object]:
        captured.update(
            config=config,
            config_yaml=config_yaml,
            output_dir=output_dir,
            run_id=run_id,
            source=source,
        )
        return {
            "status": "succeeded",
            "diagnostic_only": True,
            "provider_accessed": False,
            "dataset_accessed": False,
        }

    monkeypatch.setattr(remote, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(remote, "run_qlora_shape_preflight", fake_runner)
    monkeypatch.setattr(
        remote,
        "artifacts_volume",
        SimpleNamespace(commit=lambda: commits.append("artifacts")),
    )
    monkeypatch.setattr(
        remote,
        "hf_cache_volume",
        SimpleNamespace(commit=lambda: commits.append("hf-cache")),
    )
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "1")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    result = cast(
        dict[str, object],
        remote.run_remote_qlora_preflight.get_raw_f()(
            config_yaml,
            "he-v3-remote-test",
            source,
        ),
    )

    assert result == {
        "status": "succeeded",
        "diagnostic_only": True,
        "provider_accessed": False,
        "dataset_accessed": False,
        "report_path": (
            "diagnostics/qlora-shape-preflight/he-v3-remote-test/preflight_report.json"
        ),
    }
    assert cast(SommelierConfig, captured["config"]).model.base_model_id.startswith(
        "nvidia/Llama-3.1-Nemotron"
    )
    assert captured["config_yaml"] == config_yaml
    assert captured["output_dir"] == tmp_path / "he-v3-remote-test"
    assert captured["run_id"] == "he-v3-remote-test"
    assert captured["source"] == source
    assert commits == ["artifacts", "hf-cache"]
    for name, value in PIPELINE_HF_ENV:
        assert os.environ[name] == value
    assert os.environ["HF_HOME"] == "/hf-cache"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_remote_function_commits_failure_report_before_propagating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commits: list[str] = []

    def failing_runner(
        _config: SommelierConfig,
        **_kwargs: object,
    ) -> dict[str, object]:
        raise RuntimeError("diagnostic failed")

    monkeypatch.setattr(remote, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(remote, "run_qlora_shape_preflight", failing_runner)
    monkeypatch.setattr(
        remote,
        "artifacts_volume",
        SimpleNamespace(commit=lambda: commits.append("artifacts")),
    )
    monkeypatch.setattr(
        remote,
        "hf_cache_volume",
        SimpleNamespace(commit=lambda: commits.append("hf-cache")),
    )

    with pytest.raises(RuntimeError, match="diagnostic failed"):
        remote.run_remote_qlora_preflight.get_raw_f()(
            FULL_CONFIG_PATH.read_text(encoding="utf-8"),
            "he-v3-failed-remote-test",
            _source(),
        )

    assert commits == ["artifacts", "hf-cache"]


def test_local_entrypoint_validates_then_dispatches_exact_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source()
    captured: list[object] = []

    def fake_remote(*args: object) -> dict[str, object]:
        captured.extend(args)
        return {"status": "succeeded", "diagnostic_only": True}

    monkeypatch.setattr(remote, "local_source_provenance", lambda: source)
    monkeypatch.setattr(remote.run_remote_qlora_preflight, "remote", fake_remote)

    remote.main.info.raw_f(
        config=str(FULL_CONFIG_PATH),
        run_id="he-v3-local-dispatch-test",
    )

    assert captured == [
        FULL_CONFIG_PATH.read_text(encoding="utf-8"),
        "he-v3-local-dispatch-test",
        source,
    ]
    assert json.loads(capsys.readouterr().out) == {
        "status": "succeeded",
        "diagnostic_only": True,
    }


def test_local_entrypoint_rejects_bad_run_id_before_git_or_remote_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected() -> SourceProvenance:
        raise AssertionError("Git must not be read for an invalid run id")

    monkeypatch.setattr(remote, "local_source_provenance", unexpected)

    with pytest.raises(UserInputError, match="invalid QLoRA preflight run id"):
        remote.main.info.raw_f(config=str(FULL_CONFIG_PATH), run_id="../escape")
