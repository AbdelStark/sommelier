from __future__ import annotations

import importlib.metadata
import os
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import remote_pipeline
import sommelier.pipeline as pipeline_module
from remote_pipeline import (
    MODAL_MAX_TIMEOUT_SECONDS,
    PIPELINE_MAX_RETRIES,
    _apply_pipeline_hf_policy,
    _package_versions,
    _remote_execution_boundary,
    _required_pipeline_timeout_seconds,
    _validate_remote_launch_boundary,
)
from sommelier.config import load_config
from sommelier.errors import UserInputError
from sommelier.remote.images import PIPELINE_RUNTIME_VERSIONS

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
PAIRED_REVISION = "d" * 40
V1_ADAPTER_ID = "abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora"
V1_ADAPTER_REVISION = "45a6e2fa3e29f8393ddf1e9bda51a9461b41ee0e"


def _full_hebrew_config_yaml() -> str:
    return (
        (EXAMPLES_DIR / "config.v3-he-full.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "dataset_revision: main",
            f"dataset_revision: {PAIRED_REVISION}",
        )
    )


def _full_hebrew_request(config_yaml: str) -> dict[str, Any]:
    return {
        "config_yaml": config_yaml,
        "mode": "full",
        "max_rows": 60_000,
        "run_id": "he-v3-contract-test",
        "adapter_id": None,
        "adapter_revision": None,
        "translation_run_id": None,
        "code_revision": "a" * 40,
        "source_tree_clean": True,
    }


def _install_expensive_boundary_guards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[str]:
    reached: list[str] = []

    def unexpected_export(*_args: object, **_kwargs: object) -> int:
        reached.append("export")
        raise AssertionError("dataset export must not run after a failed preflight")

    def unexpected_pipeline(*_args: object, **_kwargs: object) -> str:
        reached.append("pipeline")
        raise AssertionError("pipeline/model stages must not run after a failed preflight")

    monkeypatch.setattr(remote_pipeline, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(remote_pipeline, "GPU", "L40S")
    monkeypatch.setattr(remote_pipeline, "TIMEOUT_SECONDS", MODAL_MAX_TIMEOUT_SECONDS)
    monkeypatch.setattr(remote_pipeline, "_export_raw_rows", unexpected_export)
    monkeypatch.setattr(pipeline_module, "run_pipeline", unexpected_pipeline)
    return reached


def test_sequential_planning_estimate_sums_both_evaluation_arms() -> None:
    assert (
        _required_pipeline_timeout_seconds(
            data_timeout_seconds=900,
            train_timeout_seconds=3600,
            eval_timeout_seconds=1800,
            trains_adapter=True,
        )
        == 8100
    )


def test_external_adapter_estimate_omits_training_but_not_evaluations() -> None:
    assert (
        _required_pipeline_timeout_seconds(
            data_timeout_seconds=900,
            train_timeout_seconds=3600,
            eval_timeout_seconds=1800,
            trains_adapter=False,
        )
        == 4500
    )


def test_modal_timeout_cap_is_one_day() -> None:
    assert MODAL_MAX_TIMEOUT_SECONDS == 86_400


def test_pipeline_disables_automatic_whole_run_retries() -> None:
    assert PIPELINE_MAX_RETRIES == 0


def test_remote_metadata_distinguishes_outer_timeout_from_stage_estimates() -> None:
    boundary = _remote_execution_boundary(
        function_timeout_seconds=10_800,
        gpu_allocation_label="A10G",
        stage_planning_estimate_seconds=8_100,
    )

    assert boundary["function_timeout_seconds"] == 10_800
    assert boundary["configured_stage_planning_estimate_seconds"] == 8_100
    assert boundary["outer_timeout_planning_headroom_seconds"] == 2_700
    assert boundary["per_stage_watchdogs_enforced"] is False
    assert boundary["hf_hub_download_policy"] == {
        "disable_xet": True,
        "download_timeout_seconds": 600,
        "boundary": (
            "Forced by the pipeline image and again at remote function entry "
            "before Hugging Face dataset or model access."
        ),
    }
    assert "outer function timeout only" in boundary["boundary"]
    assert "planning estimates" in boundary["boundary"]


def test_runtime_identity_captures_complete_inference_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def fake_version(package: str) -> str:
        observed.append(package)
        return f"{package}-version"

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    versions = _package_versions()

    assert observed == [
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "datasets",
        "huggingface_hub",
    ]
    assert versions == {
        "python": platform.python_version(),
        **{package: f"{package}-version" for package in observed},
    }


def test_pipeline_forces_hugging_face_download_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "1")

    _apply_pipeline_hf_policy()

    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "600"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mode": "typo"}, "unsupported pipeline mode"),
        ({"gpu_allocation_label": "L40S"}, "does not match the Modal allocation"),
        ({"function_timeout_seconds": 1}, "below the configured sequential stage"),
        ({"adapter_revision": "a" * 40}, "requires --adapter-id"),
    ],
)
def test_remote_launch_boundary_rejects_locally_knowable_errors(
    overrides: dict[str, object],
    message: str,
) -> None:
    config = load_config(EXAMPLES_DIR / "config.v3-he-smoke.yaml")
    request: dict[str, object] = {
        "mode": "smoke",
        "adapter_id": None,
        "adapter_revision": None,
        "code_revision": "unknown",
        "source_tree_clean": None,
        "gpu_allocation_label": "A10G",
        "function_timeout_seconds": 10_800,
    }
    request.update(overrides)

    with pytest.raises(UserInputError, match=message):
        _validate_remote_launch_boundary(config, **request)  # type: ignore[arg-type]


def test_remote_launch_boundary_rejects_dirty_full_source() -> None:
    config = load_config(EXAMPLES_DIR / "config.full.yaml")

    with pytest.raises(UserInputError, match="clean, immutable local Git revision"):
        _validate_remote_launch_boundary(
            config,
            mode="full",
            adapter_id=None,
            adapter_revision=None,
            code_revision="unknown",
            source_tree_clean=False,
            gpu_allocation_label="L40S",
            function_timeout_seconds=MODAL_MAX_TIMEOUT_SECONDS,
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("  seed: 42", "  seed: 7", "preregistered seed 42"),
        (
            "name: sommelier-v3-he-full",
            "name: sommelier-v3-he-relocated",
            "project/output contract",
        ),
        ("artifact_root: artifacts", "artifact_root: relocated", "project/output contract"),
        (
            "base_model_id: nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
            "base_model_id: substitute/model",
            "preregistered base or tokenizer",
        ),
        (
            "dataset_id: Salesforce/xlam-function-calling-60k",
            "dataset_id: substitute/root",
            "preregistered English root corpus",
        ),
        (
            "dataset_id: abdelstark/sommelier-xlam-single-call-splits-he",
            "dataset_id: substitute/paired",
            "audited Hebrew corpus",
        ),
        (
            f"dataset_revision: {PAIRED_REVISION}",
            "dataset_revision: main",
            "immutable commit",
        ),
        ("n_train: 15000", "n_train: 14999", "preregistered cohort contract"),
        (
            "Select the correct tool and return only",
            "Choose any tool and return only",
            "preregistered formatting contract",
        ),
        ("lora_rank: 16", "lora_rank: 8", "preregistered QLoRA contract"),
        ("max_new_tokens: 512", "max_new_tokens: 256", "evaluation hardware contract"),
        ("gpu: L40S", "gpu: A10G", "evaluation hardware contract"),
        ("  enabled: true", "  enabled: false", "remote planning contract"),
        ("data_timeout_seconds: 1800", "data_timeout_seconds: 1801", "remote planning contract"),
        ("retain_raw_generations: true", "retain_raw_generations: false", "report evidence"),
        ("redact_fields: []", "redact_fields: [metrics]", "report evidence"),
        ("  enabled: false", "  enabled: true", "tracking contract"),
        ("project: sommelier", "project: substitute", "tracking contract"),
    ],
)
def test_full_hebrew_config_substitution_fails_before_export_or_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old: str,
    new: str,
    message: str,
) -> None:
    config_yaml = _full_hebrew_config_yaml()
    assert old in config_yaml
    reached = _install_expensive_boundary_guards(monkeypatch, tmp_path)

    with pytest.raises(UserInputError, match=message):
        remote_pipeline.run_remote_pipeline.get_raw_f()(
            **_full_hebrew_request(config_yaml.replace(old, new))
        )

    assert reached == []


@pytest.mark.parametrize(
    ("adapter_id", "adapter_revision"),
    [
        ("substitute/adapter", V1_ADAPTER_REVISION),
        (V1_ADAPTER_ID, "e" * 40),
        (V1_ADAPTER_ID, None),
        (None, V1_ADAPTER_REVISION),
        ("", ""),
    ],
)
def test_full_hebrew_adapter_substitution_fails_before_export_or_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_id: str | None,
    adapter_revision: str | None,
) -> None:
    reached = _install_expensive_boundary_guards(monkeypatch, tmp_path)
    request = _full_hebrew_request(_full_hebrew_config_yaml())
    request.update(adapter_id=adapter_id, adapter_revision=adapter_revision)

    with pytest.raises(UserInputError, match="exact preregistered v1 baseline adapter"):
        remote_pipeline.run_remote_pipeline.get_raw_f()(**request)

    assert reached == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_rows", 59_999, "preregistered --max-rows 60000"),
        ("translation_run_id", "diagnostic-translation", "cannot use --translation-run-id"),
    ],
)
def test_full_hebrew_selection_substitution_fails_before_export_or_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    reached = _install_expensive_boundary_guards(monkeypatch, tmp_path)
    request = _full_hebrew_request(_full_hebrew_config_yaml())
    request[field] = value

    with pytest.raises(UserInputError, match=message):
        remote_pipeline.run_remote_pipeline.get_raw_f()(**request)

    assert reached == []


@pytest.mark.parametrize(
    ("adapter_id", "adapter_revision"),
    [
        (None, None),
        (V1_ADAPTER_ID, V1_ADAPTER_REVISION),
    ],
)
def test_exact_v3_training_and_v1_baseline_arms_reach_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_id: str | None,
    adapter_revision: str | None,
) -> None:
    class ExportReached(RuntimeError):
        pass

    def export_marker(*_args: object, **_kwargs: object) -> int:
        raise ExportReached

    monkeypatch.setattr(remote_pipeline, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(remote_pipeline, "GPU", "L40S")
    monkeypatch.setattr(remote_pipeline, "TIMEOUT_SECONDS", MODAL_MAX_TIMEOUT_SECONDS)
    monkeypatch.setattr(
        remote_pipeline,
        "_package_versions",
        lambda: dict(PIPELINE_RUNTIME_VERSIONS),
    )
    monkeypatch.setattr(remote_pipeline, "_export_raw_rows", export_marker)
    request = _full_hebrew_request(_full_hebrew_config_yaml())
    request.update(adapter_id=adapter_id, adapter_revision=adapter_revision)

    with pytest.raises(ExportReached):
        remote_pipeline.run_remote_pipeline.get_raw_f()(**request)


def test_full_hebrew_runtime_drift_fails_before_export_or_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached = _install_expensive_boundary_guards(monkeypatch, tmp_path)
    drifted = dict(PIPELINE_RUNTIME_VERSIONS)
    drifted["transformers"] = "5.13.2"
    monkeypatch.setattr(remote_pipeline, "_package_versions", lambda: drifted)

    with pytest.raises(UserInputError, match="runtime does not match") as error:
        remote_pipeline.run_remote_pipeline.get_raw_f()(
            **_full_hebrew_request(_full_hebrew_config_yaml())
        )

    assert "transformers: expected 5.13.1, observed 5.13.2" in str(error.value)
    assert reached == []


def test_smoke_keeps_adapter_and_config_diagnostics_flexible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExportReached(RuntimeError):
        pass

    def export_marker(*_args: object, **_kwargs: object) -> int:
        raise ExportReached

    monkeypatch.setattr(remote_pipeline, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(remote_pipeline, "GPU", "L40S")
    monkeypatch.setattr(remote_pipeline, "TIMEOUT_SECONDS", MODAL_MAX_TIMEOUT_SECONDS)
    monkeypatch.setattr(remote_pipeline, "_export_raw_rows", export_marker)
    request = _full_hebrew_request(_full_hebrew_config_yaml().replace("  seed: 42", "  seed: 7"))
    request.update(
        mode="smoke",
        max_rows=17,
        adapter_id="diagnostic/adapter",
        adapter_revision="main",
        translation_run_id="diagnostic-translation",
        code_revision="unknown",
        source_tree_clean=None,
    )

    with pytest.raises(ExportReached):
        remote_pipeline.run_remote_pipeline.get_raw_f()(**request)


def test_non_hebrew_full_pipeline_keeps_existing_adapter_flexibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExportReached(RuntimeError):
        pass

    def export_marker(*_args: object, **_kwargs: object) -> int:
        raise ExportReached

    monkeypatch.setattr(remote_pipeline, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(remote_pipeline, "GPU", "L40S")
    monkeypatch.setattr(remote_pipeline, "TIMEOUT_SECONDS", MODAL_MAX_TIMEOUT_SECONDS)
    monkeypatch.setattr(remote_pipeline, "_export_raw_rows", export_marker)

    with pytest.raises(ExportReached):
        remote_pipeline.run_remote_pipeline.get_raw_f()(
            config_yaml=(EXAMPLES_DIR / "config.full.yaml").read_text(encoding="utf-8"),
            mode="full",
            max_rows=123,
            run_id="english-full-contract-test",
            adapter_id="another/project-adapter",
            adapter_revision="main",
            translation_run_id=None,
            code_revision="a" * 40,
            source_tree_clean=True,
        )


def test_independent_hebrew_full_pipeline_is_not_forced_into_v3_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExportReached(RuntimeError):
        pass

    def export_marker(*_args: object, **_kwargs: object) -> int:
        raise ExportReached

    config_yaml = (
        _full_hebrew_config_yaml()
        .replace(
            "name: sommelier-v3-he-full",
            "name: independent-hebrew-project",
        )
        .replace(
            "dataset_id: abdelstark/sommelier-xlam-single-call-splits-he",
            "dataset_id: independent/hebrew-pairs",
        )
    )
    monkeypatch.setattr(remote_pipeline, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(remote_pipeline, "GPU", "L40S")
    monkeypatch.setattr(remote_pipeline, "TIMEOUT_SECONDS", MODAL_MAX_TIMEOUT_SECONDS)
    monkeypatch.setattr(remote_pipeline, "_export_raw_rows", export_marker)

    with pytest.raises(ExportReached):
        remote_pipeline.run_remote_pipeline.get_raw_f()(**_full_hebrew_request(config_yaml))


def test_local_entrypoint_rejects_bad_v1_identity_before_remote_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_full_hebrew_config_yaml(), encoding="utf-8")
    dispatched: list[object] = []

    def unexpected_remote(*args: object) -> dict[str, object]:
        dispatched.extend(args)
        raise AssertionError("a failed local preflight must not rent a remote GPU")

    monkeypatch.setattr(remote_pipeline.run_remote_pipeline, "remote", unexpected_remote)

    with pytest.raises(UserInputError, match="exact preregistered v1 baseline adapter"):
        remote_pipeline.main.info.raw_f(
            config=str(config_path),
            mode="full",
            max_rows=60_000,
            run_id="he-v3-v1-baseline",
            adapter_id="substitute/adapter",
            adapter_revision=V1_ADAPTER_REVISION,
            translation_run_id="",
        )

    assert dispatched == []


def test_local_entrypoint_passes_allocation_identity_into_remote_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched: list[object] = []

    def capture_remote(*args: object) -> dict[str, object]:
        dispatched.extend(args)
        return {"run_id": "captured"}

    monkeypatch.setattr(remote_pipeline, "GPU", "A10G")
    monkeypatch.setattr(remote_pipeline, "TIMEOUT_SECONDS", 10_800)
    monkeypatch.setattr(remote_pipeline.run_remote_pipeline, "remote", capture_remote)

    remote_pipeline.main.info.raw_f(
        config=str(EXAMPLES_DIR / "config.v3-he-smoke.yaml"),
        mode="smoke",
        max_rows=2_500,
        run_id="allocation-capture",
        adapter_id="",
        adapter_revision="",
        translation_run_id="diagnostic-translation",
    )

    assert dispatched[-2:] == ["A10G", 10_800]


def test_remote_body_uses_explicit_allocation_identity_not_container_globals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundaryCaptured(RuntimeError):
        pass

    captured: dict[str, object] = {}

    def capture_pipeline(*_args: object, **kwargs: object) -> str:
        boundary = kwargs["remote_execution"]
        assert isinstance(boundary, dict)
        captured.update(boundary)
        raise BoundaryCaptured

    # Simulate Modal re-importing the module without the driver's environment.
    # The explicit dispatch arguments must remain authoritative inside the body.
    monkeypatch.setattr(remote_pipeline, "GPU", "L40S")
    monkeypatch.setattr(remote_pipeline, "TIMEOUT_SECONDS", MODAL_MAX_TIMEOUT_SECONDS)
    monkeypatch.setattr(remote_pipeline, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(remote_pipeline, "_export_raw_rows", lambda *_args: 1)
    monkeypatch.setattr(remote_pipeline, "_stage_paired_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_module, "run_pipeline", capture_pipeline)
    monkeypatch.setattr(remote_pipeline, "artifacts_volume", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(remote_pipeline, "hf_cache_volume", SimpleNamespace(commit=lambda: None))

    with pytest.raises(BoundaryCaptured):
        remote_pipeline.run_remote_pipeline.get_raw_f()(
            config_yaml=(EXAMPLES_DIR / "config.v3-he-smoke.yaml").read_text(encoding="utf-8"),
            mode="smoke",
            max_rows=2_500,
            run_id="explicit-boundary",
            translation_run_id="diagnostic-translation",
            code_revision="unknown",
            source_tree_clean=None,
            allocation_gpu="A10G",
            function_timeout_seconds=10_800,
        )

    assert captured["gpu_allocation_label"] == "A10G"
    assert captured["function_timeout_seconds"] == 10_800
    assert captured["configured_stage_planning_estimate_seconds"] == 8_100
    assert captured["outer_timeout_planning_headroom_seconds"] == 2_700
