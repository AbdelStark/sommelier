from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import remote_translate
import sommelier.data.export as export_module
import sommelier.data.load as load_module
import sommelier.data.openai_evidence as openai_evidence_module
import sommelier.data.openai_translate as openai_translate_module
import sommelier.data.split as split_module
import sommelier.data.translate as translate_module
import sommelier.hebrew_v3_preregistration as preregistration_module
from sommelier.data.translate import HEBREW_V3_TRANSLATION_LIST_PRICE_LIMIT_USD
from sommelier.errors import UserInputError
from sommelier.remote.images import OPENAI_TRANSLATION_RUNTIME_VERSIONS
from sommelier.reviewer import (
    canonical_reviewer_requirement,
    reviewer_preregistration_payload,
    reviewer_preregistration_sha256,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
MODEL_SNAPSHOT = "gpt-5.5-2026-04-23"
REVIEWER_PUBLIC_KEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f"
)
REVIEWER_REQUIREMENT = canonical_reviewer_requirement(
    "fixture-reviewer",
    REVIEWER_PUBLIC_KEY,
)


def _full_hebrew_config_yaml() -> str:
    config_yaml = (EXAMPLES_DIR / "config.v3-he-full.yaml").read_text(encoding="utf-8")
    return (
        config_yaml
        + "\nsemantic_review:\n"
        + "  reviewer:\n"
        + f"    reviewer_id: {REVIEWER_REQUIREMENT.reviewer_id}\n"
        + f"    ssh_public_key: {REVIEWER_REQUIREMENT.ssh_public_key}\n"
        + (f"    public_key_fingerprint: {REVIEWER_REQUIREMENT.public_key_fingerprint}\n")
    )


def _allow_test_config_at_fake_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preregistration_module,
        "require_committed_config_bytes",
        lambda config_path, **_kwargs: Path(config_path).read_bytes(),
    )


def _openai_args(config_yaml: str, *, mode: str = "smoke") -> dict[str, Any]:
    return {
        "config_yaml": config_yaml,
        "run_id": "openai-smoke",
        "mode": mode,
        "max_rows": 2500,
        "model_id": MODEL_SNAPSHOT,
        "model_revision": MODEL_SNAPSHOT,
        "max_new_tokens": 256,
        "translator_interface": "instruction_chat",
        "max_model_len": 0,
        "trust_remote_code": False,
        "output_decoder": "standard",
        "limit": 1 if mode == "smoke" else 0,
        "target_language": "he",
        "code_revision": "a" * 40,
        "source_tree_clean": True,
        "allocation_gpu": None,
        "function_timeout_seconds": 3600,
        "openai_service_tier": "default",
        "openai_max_workers": 1,
        "openai_list_price_limit_usd": (
            HEBREW_V3_TRANSLATION_LIST_PRICE_LIMIT_USD if mode == "full" else "1000.00"
        ),
    }


def test_openai_modal_function_is_cpu_only_and_has_only_artifact_volume_and_named_secrets() -> None:
    spec = remote_translate.run_remote_openai_translation.spec

    assert spec.gpus is None
    assert set(spec.volumes) == {"/artifacts"}
    assert "/hf-cache" not in spec.volumes
    assert "/vllm-cache" not in spec.volumes
    assert len(spec.secrets) == 2
    secret_description = " ".join(repr(secret) for secret in spec.secrets)
    assert remote_translate.OPENAI_SECRET_NAME in secret_description
    assert remote_translate.HF_READ_SECRET_NAME in secret_description
    assert "from_dotenv" not in secret_description


def test_openai_runtime_versions_only_query_pinned_cpu_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def fake_version(package: str) -> str:
        observed.append(package)
        return f"{package}-version"

    monkeypatch.setattr(platform, "python_version", lambda: "3.13.3")
    monkeypatch.setattr(remote_translate, "version", fake_version)

    versions = remote_translate._package_versions(OPENAI_TRANSLATION_RUNTIME_VERSIONS)

    assert observed == ["openai", "datasets"]
    assert versions == {
        "python": "3.13.3",
        "openai": "openai-version",
        "datasets": "datasets-version",
    }


def test_openai_backend_factory_plumbs_snapshot_journal_tier_and_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_factory(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        openai_translate_module,
        "load_openai_responses_translation_model",
        fake_factory,
    )
    journal = tmp_path / "openai_responses_provider.jsonl"
    translator = SimpleNamespace(model_id=MODEL_SNAPSHOT, max_new_tokens=384)

    model = remote_translate._load_translation_model_for_backend(
        translator,
        runtime_backend="openai_responses",
        provider_journal_path=journal,
        openai_service_tier="flex",
        openai_max_workers=7,
        openai_list_price_limit_usd="50.00",
    )

    assert model is sentinel
    assert captured == {
        "model_snapshot": MODEL_SNAPSHOT,
        "max_output_tokens": 384,
        "provider_journal_path": journal,
        "expected_sdk_version": "2.45.0",
        "max_retries": 0,
        "timeout_seconds": openai_translate_module.OPENAI_RESPONSES_TIMEOUT_SECONDS,
        "service_tier": "flex",
        "max_workers": 7,
        "openai_list_price_limit_usd": "50.00",
    }


def _provider_aggregate(*, usage_complete: bool = True) -> dict[str, object]:
    return {
        "schema_version": openai_translate_module.OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA,
        "journal_schema_version": openai_translate_module.OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
        "journal_sha256": "a" * 64,
        "requested_model": MODEL_SNAPSHOT,
        "returned_models": [MODEL_SNAPSHOT],
        "requested_service_tier": "flex",
        "requested_timeout_seconds": openai_translate_module.OPENAI_RESPONSES_TIMEOUT_SECONDS,
        "returned_service_tiers": ["flex"],
        "safety_identifier": openai_translate_module.OPENAI_RESPONSES_SAFETY_IDENTIFIER,
        "client_injected": False,
        "sdk_max_retries": openai_translate_module.OPENAI_RESPONSES_SDK_MAX_RETRIES,
        "resource_unavailable_retry_policy": (
            openai_translate_module.openai_flex_resource_unavailable_retry_policy()
        ),
        "max_canonical_request_body_utf8_bytes": 1024,
        "max_response_input_tokens": 600,
        "unique_requests": 2,
        "unique_source_attempts": 2,
        "usage_complete": usage_complete,
        "counts": {
            "records": 2 if usage_complete else 3,
            "responses": 2,
            "replayable_responses": 2,
            "replays": 0,
            "durable_journal_replays": 0,
            "batch_coalesced_replays": 0,
            "request_errors": 0 if usage_complete else 1,
            "resource_unavailable_events": 0,
            "resolved_resource_unavailable_events": 0,
            "pending_resource_unavailable_events": 0,
            "unresolved_resource_unavailable_events": 0,
            "provider_error_responses": 0,
            "error_records": 0 if usage_complete else 1,
            "model_mismatch_responses": 0,
            "service_tier_mismatch_responses": 0,
            "refusal_responses": 0,
            "incomplete_responses": 0,
            "responses_missing_usage": 0,
        },
        "usage": {
            "input_tokens": 1000,
            "cached_input_tokens": 200,
            "output_tokens": 100,
            "reasoning_output_tokens": 25,
            "total_tokens": 1100,
        },
    }


def test_provider_evidence_prices_cached_and_reasoning_tokens_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "openai_responses_provider.jsonl"
    monkeypatch.setattr(
        openai_evidence_module,
        "aggregate_openai_responses_provider_journal",
        lambda _path: _provider_aggregate(),
    )

    evidence = openai_evidence_module.build_openai_provider_evidence(
        journal, MODEL_SNAPSHOT, "flex"
    )

    estimate = evidence["list_price_estimate"]
    assert isinstance(estimate, dict)
    assert estimate["available"] is True
    assert estimate["calculated_usd"] == "0.003550000"
    assert estimate["billing_evidence"] is False
    pricing = estimate["pricing"]
    assert isinstance(pricing, dict)
    assert pricing["service_tier_multiplier"] == "0.5"
    assert evidence["journal"] == {
        "filename": "openai_responses_provider.jsonl",
        "schema_version": openai_translate_module.OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
        "summary_schema_version": (
            openai_translate_module.OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA
        ),
        "sha256": "a" * 64,
        "publication_boundary": (
            "The raw journal remains in the durable producer artifacts; this content-free "
            "aggregate is published in the translation summary."
        ),
    }


def test_provider_evidence_withholds_cost_when_usage_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        openai_evidence_module,
        "aggregate_openai_responses_provider_journal",
        lambda _path: _provider_aggregate(usage_complete=False),
    )

    evidence = openai_evidence_module.build_openai_provider_evidence(
        tmp_path / "openai_responses_provider.jsonl", MODEL_SNAPSHOT, "flex"
    )

    estimate = evidence["list_price_estimate"]
    assert isinstance(estimate, dict)
    assert estimate["available"] is False
    assert "calculated_usd" not in estimate


def test_provider_evidence_rejects_returned_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = _provider_aggregate()
    aggregate["returned_service_tiers"] = ["default"]
    monkeypatch.setattr(
        openai_evidence_module,
        "aggregate_openai_responses_provider_journal",
        lambda _path: aggregate,
    )

    with pytest.raises(UserInputError, match="service tier"):
        openai_evidence_module.build_openai_provider_evidence(
            tmp_path / "openai_responses_provider.jsonl", MODEL_SNAPSHOT, "flex"
        )


@pytest.mark.parametrize(
    ("model_id", "model_revision"),
    [
        ("gpt-5.5", "gpt-5.5"),
        (MODEL_SNAPSHOT, "gpt-5.5-2026-05-01"),
        ("gpt-5.5-2026-02-30", "gpt-5.5-2026-02-30"),
        ("google/madlad400-3b-mt", "a" * 40),
    ],
)
def test_openai_transport_rejects_moving_mismatched_or_invalid_snapshots(
    model_id: str,
    model_revision: str,
) -> None:
    with pytest.raises(UserInputError, match="snapshot"):
        remote_translate._validate_openai_model_snapshot(model_id, model_revision)


def test_openai_transport_accepts_matching_dated_snapshot() -> None:
    remote_translate._validate_openai_model_snapshot(MODEL_SNAPSHOT, MODEL_SNAPSHOT)


def test_local_entrypoint_requires_explicit_instruction_chat_for_openai_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (EXAMPLES_DIR / "config.v3-he-smoke.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    reached: list[str] = []

    def fake_remote(*_args: object) -> str:
        reached.append("openai")
        return "unexpected"

    monkeypatch.setattr(
        translate_module,
        "translator_interface_for_model",
        lambda _model, interface: interface,
    )
    monkeypatch.setattr(
        remote_translate.run_remote_openai_translation,
        "remote",
        fake_remote,
    )

    with pytest.raises(UserInputError, match="explicit instruction_chat"):
        remote_translate.main.info.raw_f(
            config=str(config_path),
            mode="smoke",
            model_id=MODEL_SNAPSHOT,
            model_revision=MODEL_SNAPSHOT,
            translator_interface="auto",
            runtime_backend="openai_responses",
        )

    with pytest.raises(UserInputError, match="explicit instruction_chat"):
        remote_translate.main.info.raw_f(
            config=str(config_path),
            mode="smoke",
            model_id=MODEL_SNAPSHOT,
            model_revision=MODEL_SNAPSHOT,
            translator_interface="madlad_seq2seq",
            runtime_backend="openai_responses",
        )

    assert reached == []


@pytest.mark.parametrize("service_tier", ["auto", "batch", ""])
def test_local_entrypoint_rejects_unregistered_openai_service_tier_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service_tier: str,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: test\n", encoding="utf-8")
    monkeypatch.setattr(
        translate_module,
        "translator_interface_for_model",
        lambda _model, interface: interface,
    )

    with pytest.raises(UserInputError, match="service tier"):
        remote_translate.main.info.raw_f(
            config=str(config_path),
            mode="smoke",
            model_id=MODEL_SNAPSHOT,
            model_revision=MODEL_SNAPSHOT,
            translator_interface="instruction_chat",
            runtime_backend="openai_responses",
            openai_service_tier=service_tier,
        )


def test_local_entrypoint_dispatches_openai_without_gpu_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (EXAMPLES_DIR / "config.v3-he-smoke.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    captured: list[object] = []

    def fake_remote(*args: object) -> str:
        captured.extend(args)
        return "openai-ok"

    monkeypatch.setattr(
        translate_module,
        "translator_interface_for_model",
        lambda _model, interface: interface,
    )
    monkeypatch.setattr(
        remote_translate.run_remote_openai_translation,
        "remote",
        fake_remote,
    )
    monkeypatch.setattr(
        remote_translate,
        "_local_source_identity",
        lambda: ("a" * 40, True),
    )

    remote_translate.main.info.raw_f(
        config=str(config_path),
        run_id="openai-dispatch",
        mode="smoke",
        max_rows=2500,
        model_id=MODEL_SNAPSHOT,
        model_revision=MODEL_SNAPSHOT,
        max_new_tokens=256,
        translator_interface="instruction_chat",
        max_model_len=0,
        trust_remote_code=False,
        output_decoder="standard",
        limit=3,
        target_language="he",
        runtime_backend="openai_responses",
        openai_service_tier="flex",
        openai_max_workers=4,
        openai_list_price_limit_usd="50.00",
    )

    assert captured[-5:] == [None, remote_translate.TIMEOUT_SECONDS, "flex", 4, "50.00"]
    assert capsys.readouterr().out.strip() == "openai-ok"


def test_local_entrypoint_dispatches_only_the_exact_full_hebrew_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_full_hebrew_config_yaml(), encoding="utf-8")
    captured: list[object] = []

    def fake_remote(*args: object) -> str:
        captured.extend(args)
        return "full-openai-ok"

    monkeypatch.setattr(
        remote_translate.run_remote_openai_translation,
        "remote",
        fake_remote,
    )
    monkeypatch.setattr(
        remote_translate,
        "_local_source_identity",
        lambda: ("a" * 40, True),
    )
    _allow_test_config_at_fake_revision(monkeypatch)

    remote_translate.main.info.raw_f(
        config=str(config_path),
        run_id="he-v3-full-dispatch",
        mode="full",
        max_rows=translate_module.HEBREW_V3_TRANSLATION_MAX_ROWS,
        model_id=translate_module.HEBREW_V3_FORWARD_TRANSLATOR_MODEL_ID,
        model_revision=translate_module.HEBREW_V3_FORWARD_TRANSLATOR_MODEL_REVISION,
        max_new_tokens=translate_module.HEBREW_V3_FORWARD_TRANSLATOR_MAX_NEW_TOKENS,
        translator_interface=translate_module.HEBREW_V3_FORWARD_TRANSLATOR_INTERFACE,
        max_model_len=translate_module.HEBREW_V3_FORWARD_TRANSLATOR_MAX_MODEL_LEN,
        trust_remote_code=translate_module.HEBREW_V3_FORWARD_TRANSLATOR_TRUST_REMOTE_CODE,
        output_decoder=translate_module.HEBREW_V3_FORWARD_TRANSLATOR_OUTPUT_DECODER,
        limit=0,
        target_language="he",
        runtime_backend="openai_responses",
        openai_service_tier=translate_module.HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
        openai_max_workers=translate_module.HEBREW_V3_TRANSLATION_PROVIDER_MAX_WORKERS,
        openai_list_price_limit_usd=HEBREW_V3_TRANSLATION_LIST_PRICE_LIMIT_USD,
    )

    assert captured[-5:] == [
        None,
        remote_translate.TIMEOUT_SECONDS,
        "flex",
        8,
        "50.00",
    ]
    assert capsys.readouterr().out.strip() == "full-openai-ok"


def test_openai_smoke_body_uses_journal_factory_and_never_touches_model_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _openai_args((EXAMPLES_DIR / "config.v3-he-smoke.yaml").read_text(encoding="utf-8"))
    captured_factory: dict[str, object] = {}
    captured_stats: dict[str, object] = {}
    cache_commits = {"hf": 0, "vllm": 0}

    def fake_export(_source: object, rows_path: Path, **_kwargs: object) -> int:
        rows_path.write_text("{}\n", encoding="utf-8")
        return 1

    def fake_factory(
        translator: object,
        *,
        runtime_backend: str,
        provider_journal_path: Path | None,
        openai_service_tier: str,
        openai_max_workers: int,
        openai_list_price_limit_usd: str,
    ) -> object:
        captured_factory.update(
            translator=translator,
            runtime_backend=runtime_backend,
            provider_journal_path=provider_journal_path,
            openai_service_tier=openai_service_tier,
            openai_max_workers=openai_max_workers,
            openai_list_price_limit_usd=openai_list_price_limit_usd,
        )
        return object()

    def fake_outputs(
        _out_dir: Path,
        _translated: object,
        stats: dict[str, object],
        **_kwargs: object,
    ) -> tuple[Path, Path]:
        captured_stats.update(stats)
        return tmp_path / "rows.he.jsonl", tmp_path / "translation_summary.json"

    def touch_cache(name: str) -> None:
        cache_commits[name] += 1

    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-must-not-be-logged")
    monkeypatch.setenv("HF_TOKEN", "test-hf-token-must-not-be-logged")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    monkeypatch.setattr(remote_translate, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(
        remote_translate,
        "_package_versions",
        lambda expected: dict(expected),
    )
    monkeypatch.setattr(remote_translate, "artifacts_volume", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(
        remote_translate,
        "hf_cache_volume",
        SimpleNamespace(commit=lambda: touch_cache("hf")),
    )
    monkeypatch.setattr(
        remote_translate,
        "vllm_cache_volume",
        SimpleNamespace(commit=lambda: touch_cache("vllm")),
    )
    monkeypatch.setattr(
        translate_module,
        "translator_interface_for_model",
        lambda _model, interface: interface,
    )
    monkeypatch.setattr(export_module, "export_raw_rows", fake_export)
    monkeypatch.setattr(load_module, "load_raw_rows", lambda _path: [{"source_id": "root-1"}])
    monkeypatch.setattr(split_module, "prepare_split_result", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(split_module, "all_examples", lambda _result: [{"example_id": "root-1"}])
    monkeypatch.setattr(remote_translate, "_load_translation_model_for_backend", fake_factory)
    monkeypatch.setattr(
        remote_translate,
        "build_openai_provider_evidence",
        lambda *_args, **_kwargs: {
            "schema_version": openai_evidence_module.OPENAI_PROVIDER_EVIDENCE_SCHEMA,
            "sentinel": True,
        },
    )
    captured_translate_kwargs: dict[str, object] = {}

    def fake_translate_rows(
        *_args: object, **kwargs: object
    ) -> tuple[list[object], dict[str, int]]:
        captured_translate_kwargs.update(kwargs)
        return [], {"input_rows": 1, "translated_rows": 0}

    monkeypatch.setattr(translate_module, "translate_rows", fake_translate_rows)
    monkeypatch.setattr(translate_module, "write_translation_outputs", fake_outputs)

    remote_translate.run_remote_openai_translation.get_raw_f()(**args)

    expected_journal = tmp_path / "translation" / "openai-smoke" / "openai_responses_provider.jsonl"
    assert captured_factory["runtime_backend"] == "openai_responses"
    assert getattr(captured_factory["translator"], "interface") == "instruction_chat"
    assert captured_factory["provider_journal_path"] == expected_journal
    assert captured_factory["openai_service_tier"] == "default"
    assert captured_factory["openai_max_workers"] == 1
    assert captured_factory["openai_list_price_limit_usd"] == "1000.00"
    runtime = captured_stats["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["backend"] == "openai_responses"
    assert runtime["gpu"] is None
    assert runtime["provider_service_tier"] == "default"
    assert runtime["provider_timeout_seconds"] == 900.0
    assert runtime["provider_max_workers"] == 1
    assert runtime["provider_journal_filename"] == expected_journal.name
    assert "provider_journal_path" not in runtime
    assert captured_stats["provider_evidence"] == {
        "schema_version": openai_evidence_module.OPENAI_PROVIDER_EVIDENCE_SCHEMA,
        "sentinel": True,
    }
    assert captured_stats["environment"] == dict(OPENAI_TRANSLATION_RUNTIME_VERSIONS)
    assert captured_stats["max_attempts"] == 3
    assert captured_translate_kwargs["max_attempts"] == 3
    assert captured_translate_kwargs["chunk_size"] == 32
    assert callable(captured_translate_kwargs["durable_checkpoint"])
    assert "download_policy" not in captured_stats
    assert cache_commits == {"hf": 0, "vllm": 0}
    assert not any(
        key in os.environ
        for key in (
            "HF_HOME",
            "HF_HUB_DISABLE_XET",
            "HF_HUB_DOWNLOAD_TIMEOUT",
            "VLLM_CACHE_ROOT",
        )
    )
    captured_output = capsys.readouterr().out
    assert "test-secret-must-not-be-logged" not in captured_output
    assert "test-hf-token-must-not-be-logged" not in captured_output


def test_full_hebrew_openai_backend_passes_exact_preregistration_before_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _openai_args(
        _full_hebrew_config_yaml(),
        mode="full",
    )
    args["max_rows"] = 60_000
    args["max_new_tokens"] = 512
    args["openai_service_tier"] = "flex"
    args["openai_max_workers"] = 8
    reached: list[str] = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("HF_TOKEN", "test-only")
    monkeypatch.setattr(remote_translate, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(
        translate_module,
        "translator_interface_for_model",
        lambda _model, interface: interface,
    )

    class ExportReached(RuntimeError):
        pass

    def mark_export(*_args: object, **_kwargs: object) -> int:
        reached.append("export")
        raise ExportReached

    def mark_factory(*_args: object, **_kwargs: object) -> object:
        reached.append("factory")
        return object()

    monkeypatch.setattr(export_module, "export_raw_rows", mark_export)
    monkeypatch.setattr(
        remote_translate,
        "_package_versions",
        lambda expected: dict(expected),
    )
    monkeypatch.setattr(
        remote_translate,
        "_load_translation_model_for_backend",
        mark_factory,
    )

    with pytest.raises(ExportReached):
        remote_translate.run_remote_openai_translation.get_raw_f()(**args)

    assert reached == ["export"]
    identity_path = (
        tmp_path
        / "translation"
        / str(args["run_id"])
        / translate_module.TRANSLATION_RUN_IDENTITY_FILENAME
    )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assert identity["reviewer_preregistration"] == reviewer_preregistration_payload(
        REVIEWER_REQUIREMENT
    )
    assert identity["reviewer_preregistration_sha256"] == reviewer_preregistration_sha256(
        REVIEWER_REQUIREMENT
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("openai_service_tier", "default", "provider_service_tier"),
        ("openai_max_workers", 7, "provider_max_workers"),
        ("openai_list_price_limit_usd", "1000.00", "list_price_limit_usd"),
    ],
)
def test_full_hebrew_provider_selection_rejects_before_export_or_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    args = _openai_args(
        _full_hebrew_config_yaml(),
        mode="full",
    )
    args.update(
        max_rows=60_000,
        max_new_tokens=512,
        openai_service_tier="flex",
        openai_max_workers=8,
    )
    args[field] = value
    reached: list[str] = []

    def unexpected_export(*_args: object, **_kwargs: object) -> int:
        reached.append("export")
        raise AssertionError("provider drift must fail before dataset access")

    def unexpected_factory(*_args: object, **_kwargs: object) -> object:
        reached.append("factory")
        raise AssertionError("provider drift must fail before provider construction")

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("HF_TOKEN", "test-only")
    monkeypatch.setattr(remote_translate, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(export_module, "export_raw_rows", unexpected_export)
    monkeypatch.setattr(
        remote_translate,
        "_load_translation_model_for_backend",
        unexpected_factory,
    )

    with pytest.raises(UserInputError, match=message):
        remote_translate.run_remote_openai_translation.get_raw_f()(**args)

    assert reached == []
