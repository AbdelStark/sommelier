from __future__ import annotations

import platform as platform_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import remote_translate
import sommelier.data.export as export_module
import sommelier.data.load as load_module
import sommelier.data.split as split_module
import sommelier.data.translate as translate_module
from sommelier.data.translate import (
    HEBREW_V3_FORWARD_TRANSLATOR_INTERFACE,
    HEBREW_V3_FORWARD_TRANSLATOR_MAX_MODEL_LEN,
    HEBREW_V3_FORWARD_TRANSLATOR_MAX_NEW_TOKENS,
    HEBREW_V3_FORWARD_TRANSLATOR_MODEL_ID,
    HEBREW_V3_FORWARD_TRANSLATOR_MODEL_REVISION,
    HEBREW_V3_FORWARD_TRANSLATOR_OUTPUT_DECODER,
    HEBREW_V3_FORWARD_TRANSLATOR_TRUST_REMOTE_CODE,
    HEBREW_V3_TRANSLATION_MAX_ROWS,
    HEBREW_V3_TRANSLATION_PROVIDER_MAX_WORKERS,
    HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
)
from sommelier.errors import UserInputError
from sommelier.remote.images import (
    OPENAI_TRANSLATION_RUNTIME_VERSIONS,
    SEQ2SEQ_TRANSLATION_RUNTIME_VERSIONS,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(autouse=True)
def _provider_test_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("HF_TOKEN", "test-only")


def test_translation_runtime_captures_tokenizer_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def fake_version(package: str) -> str:
        observed.append(package)
        return f"{package}-version"

    monkeypatch.setattr(platform_module, "python_version", lambda: "3.13.0")
    monkeypatch.setattr(remote_translate, "version", fake_version)

    versions = remote_translate._package_versions()

    assert observed == [
        "vllm",
        "huggingface_hub",
        "torch",
        "transformers",
        "tokenizers",
        "datasets",
        "accelerate",
        "sentencepiece",
    ]
    assert versions == {
        "python": "3.13.0",
        **{package: f"{package}-version" for package in observed},
    }

    observed.clear()
    versions = remote_translate._package_versions(SEQ2SEQ_TRANSLATION_RUNTIME_VERSIONS)

    assert observed == [
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
        "datasets",
        "safetensors",
    ]
    assert versions == {
        "python": "3.13.0",
        **{package: f"{package}-version" for package in observed},
    }


def _full_hebrew_args(config_yaml: str) -> dict[str, Any]:
    return {
        "config_yaml": config_yaml,
        "run_id": "hebrew-v3-preflight",
        "mode": "full",
        "max_rows": HEBREW_V3_TRANSLATION_MAX_ROWS,
        "model_id": HEBREW_V3_FORWARD_TRANSLATOR_MODEL_ID,
        "model_revision": HEBREW_V3_FORWARD_TRANSLATOR_MODEL_REVISION,
        "max_new_tokens": HEBREW_V3_FORWARD_TRANSLATOR_MAX_NEW_TOKENS,
        "translator_interface": HEBREW_V3_FORWARD_TRANSLATOR_INTERFACE,
        "max_model_len": HEBREW_V3_FORWARD_TRANSLATOR_MAX_MODEL_LEN,
        "trust_remote_code": HEBREW_V3_FORWARD_TRANSLATOR_TRUST_REMOTE_CODE,
        "output_decoder": HEBREW_V3_FORWARD_TRANSLATOR_OUTPUT_DECODER,
        "limit": 0,
        "target_language": "he",
        "code_revision": "a" * 40,
        "source_tree_clean": True,
        "allocation_gpu": None,
        "function_timeout_seconds": 3600,
    }


def _run_backend_for_args(args: dict[str, Any]) -> str:
    if args.get("mode") == "full" and args.get("target_language") == "he":
        return cast(
            str,
            remote_translate.run_remote_openai_translation.get_raw_f()(
                **args,
                openai_service_tier=HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
                openai_max_workers=HEBREW_V3_TRANSLATION_PROVIDER_MAX_WORKERS,
                openai_list_price_limit_usd="1000.00",
            ),
        )
    resolved = translate_module.translator_interface_for_model(
        str(args["model_id"]),
        str(args["translator_interface"]),
    )
    function = (
        remote_translate.run_remote_seq2seq_translation
        if resolved == "madlad_seq2seq"
        else remote_translate.run_remote_translation
    )
    return cast(str, function.get_raw_f()(**args))


def test_local_entrypoint_dispatches_exact_remote_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: test\n", encoding="utf-8")
    captured: list[object] = []

    def fake_remote(*args: object) -> str:
        captured.extend(args)
        return "remote-ok"

    monkeypatch.setattr(remote_translate.run_remote_translation, "remote", fake_remote)
    monkeypatch.setattr(
        remote_translate,
        "_local_source_identity",
        lambda: ("a" * 40, True),
    )
    monkeypatch.setattr(remote_translate, "GPU", "L40S:1")
    monkeypatch.setattr(remote_translate, "TIMEOUT_SECONDS", 3600)

    remote_translate.main.info.raw_f(
        config=str(config_path),
        run_id="dispatch-run",
        mode="smoke",
        max_rows=2500,
        model_id="example/translator",
        model_revision="b" * 40,
        max_new_tokens=321,
        translator_interface="instruction_chat",
        max_model_len=4096,
        trust_remote_code=True,
        output_decoder="bytelevel_unicode",
        limit=3,
        target_language="he",
    )

    assert captured == [
        "schema_version: test\n",
        "dispatch-run",
        "smoke",
        2500,
        "example/translator",
        "b" * 40,
        321,
        "instruction_chat",
        4096,
        True,
        "bytelevel_unicode",
        3,
        "he",
        "a" * 40,
        True,
        "L40S:1",
        3600,
    ]
    assert capsys.readouterr().out.strip() == "remote-ok"


def test_local_entrypoint_resolves_madlad_to_seq2seq_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: test\n", encoding="utf-8")
    captured: list[object] = []

    def fake_seq2seq_remote(*args: object) -> str:
        captured.extend(args)
        return "seq2seq-ok"

    def unexpected_vllm_remote(*_args: object) -> str:
        raise AssertionError("MADLAD must not dispatch to the vLLM image")

    monkeypatch.setattr(
        remote_translate.run_remote_seq2seq_translation,
        "remote",
        fake_seq2seq_remote,
    )
    monkeypatch.setattr(
        remote_translate.run_remote_translation,
        "remote",
        unexpected_vllm_remote,
    )
    monkeypatch.setattr(
        remote_translate,
        "_local_source_identity",
        lambda: ("a" * 40, True),
    )

    remote_translate.main.info.raw_f(
        config=str(config_path),
        run_id="madlad-dispatch",
        mode="smoke",
        max_rows=2500,
        model_id="google/madlad400-3b-mt",
        model_revision="b" * 40,
        max_new_tokens=512,
        translator_interface="auto",
        max_model_len=2048,
        trust_remote_code=False,
        output_decoder="standard",
        limit=3,
        target_language="he",
    )

    assert captured[7] == "madlad_seq2seq"
    assert capsys.readouterr().out.strip() == "seq2seq-ok"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "substitute/translator", "snapshot"),
        ("model_revision", "c" * 40, "snapshot"),
        ("max_new_tokens", 511, "forward translator decoding"),
        ("translator_interface", "translategemma", "instruction_chat interface"),
        ("max_model_len", 8192, "forward translator max_model_len"),
        ("trust_remote_code", True, "forward translator trust_remote_code"),
        ("output_decoder", "bytelevel_unicode", "forward translator output_decoder"),
        ("max_rows", 59_999, "translation selection max_rows"),
        ("limit", 1, "translation selection limit"),
        ("seed", 43, "translation selection seed"),
    ],
)
def test_full_hebrew_remote_preflight_rejects_before_export_or_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    config_yaml = (EXAMPLES_DIR / "config.v3-he-full.yaml").read_text(encoding="utf-8")
    args = _full_hebrew_args(config_yaml)
    if field == "seed":
        args["config_yaml"] = config_yaml.replace("  seed: 42", f"  seed: {value}")
    else:
        args[field] = value
    reached: list[str] = []

    def unexpected_export(*_args: object, **_kwargs: object) -> int:
        reached.append("export")
        raise AssertionError("dataset export must not run after a failed preflight")

    def unexpected_model_load(*_args: object, **_kwargs: object) -> object:
        reached.append("model_load")
        raise AssertionError("model loading must not run after a failed preflight")

    monkeypatch.setattr(remote_translate, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(export_module, "export_raw_rows", unexpected_export)
    monkeypatch.setattr(translate_module, "load_translation_model", unexpected_model_load)

    with pytest.raises(UserInputError, match=message):
        _run_backend_for_args(args)

    assert reached == []


def test_backend_interface_mismatch_rejects_before_artifact_or_data_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _full_hebrew_args((EXAMPLES_DIR / "config.v3-he-smoke.yaml").read_text(encoding="utf-8"))
    args.update(
        mode="smoke",
        max_rows=2500,
        model_id="google/madlad400-3b-mt",
        model_revision="main",
        max_new_tokens=64,
        translator_interface="madlad_seq2seq",
        max_model_len=2048,
        trust_remote_code=False,
        output_decoder="standard",
        limit=1,
        code_revision="unknown",
        source_tree_clean=None,
    )
    reached: list[str] = []

    def unexpected_export(*_args: object, **_kwargs: object) -> int:
        reached.append("export")
        raise AssertionError("backend mismatch must fail before dataset access")

    def unexpected_model_load(*_args: object, **_kwargs: object) -> object:
        reached.append("model_load")
        raise AssertionError("backend mismatch must fail before model access")

    monkeypatch.setattr(remote_translate, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(export_module, "export_raw_rows", unexpected_export)
    monkeypatch.setattr(translate_module, "load_translation_model", unexpected_model_load)

    with pytest.raises(UserInputError, match="requires runtime backend 'transformers_seq2seq'"):
        remote_translate.run_remote_translation.get_raw_f()(**args)

    assert reached == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("translator_interface", "model_id", "backend", "expected_cache_commits"),
    [
        (
            "instruction_chat",
            "diagnostic/chat-translator",
            "vllm",
            {"hf": 1, "vllm": 1},
        ),
        (
            "madlad_seq2seq",
            "google/madlad400-3b-mt",
            "seq2seq",
            {"hf": 1, "vllm": 0},
        ),
    ],
)
def test_model_load_failure_commits_resumable_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    translator_interface: str,
    model_id: str,
    backend: str,
    expected_cache_commits: dict[str, int],
) -> None:
    args = _full_hebrew_args((EXAMPLES_DIR / "config.v3-he-smoke.yaml").read_text(encoding="utf-8"))
    args.update(
        mode="smoke",
        max_rows=2500,
        model_id=model_id,
        model_revision="main",
        max_new_tokens=64,
        translator_interface=translator_interface,
        max_model_len=4096 if backend == "vllm" else 2048,
        trust_remote_code=False,
        output_decoder="standard",
        limit=1,
        code_revision="unknown",
        source_tree_clean=None,
    )
    cache_commits = {"hf": 0, "vllm": 0}

    class ModelLoadFailed(RuntimeError):
        pass

    def fake_export(_source: object, rows_path: Path, **_kwargs: object) -> int:
        rows_path.write_text("{}\n", encoding="utf-8")
        return 1

    def commit_cache(name: str) -> None:
        cache_commits[name] += 1

    def fail_model_load(_translator: object) -> object:
        raise ModelLoadFailed("interrupted checkpoint download")

    monkeypatch.setattr(remote_translate, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(remote_translate, "_package_versions", lambda expected: dict(expected))
    monkeypatch.setattr(remote_translate, "artifacts_volume", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(
        remote_translate,
        "hf_cache_volume",
        SimpleNamespace(commit=lambda: commit_cache("hf")),
    )
    monkeypatch.setattr(
        remote_translate,
        "vllm_cache_volume",
        SimpleNamespace(commit=lambda: commit_cache("vllm")),
    )
    monkeypatch.setattr(export_module, "export_raw_rows", fake_export)
    monkeypatch.setattr(load_module, "load_raw_rows", lambda _path: [{"source_id": "root-1"}])
    monkeypatch.setattr(split_module, "prepare_split_result", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(split_module, "all_examples", lambda _result: [{"example_id": "root-1"}])
    monkeypatch.setattr(translate_module, "load_translation_model", fail_model_load)

    remote_function = (
        remote_translate.run_remote_translation
        if backend == "vllm"
        else remote_translate.run_remote_seq2seq_translation
    )
    with pytest.raises(ModelLoadFailed, match="interrupted checkpoint download"):
        remote_function.get_raw_f()(**args)

    assert cache_commits == expected_cache_commits


def test_full_hebrew_runtime_drift_fails_before_export_or_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_yaml = (EXAMPLES_DIR / "config.v3-he-full.yaml").read_text(encoding="utf-8")
    args = _full_hebrew_args(config_yaml)
    reached: list[str] = []
    drifted = dict(OPENAI_TRANSLATION_RUNTIME_VERSIONS)
    drifted["openai"] = "2.44.0"

    def unexpected_export(*_args: object, **_kwargs: object) -> int:
        reached.append("export")
        raise AssertionError("runtime drift must fail before dataset network access")

    def unexpected_model_load(*_args: object, **_kwargs: object) -> object:
        reached.append("model_load")
        raise AssertionError("runtime drift must fail before model access")

    monkeypatch.setattr(remote_translate, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(remote_translate, "_package_versions", lambda _expected: drifted)
    monkeypatch.setattr(export_module, "export_raw_rows", unexpected_export)
    monkeypatch.setattr(translate_module, "load_translation_model", unexpected_model_load)

    with pytest.raises(UserInputError, match="runtime does not match") as error:
        _run_backend_for_args(args)

    assert "openai: expected 2.45.0, observed 2.44.0" in str(error.value)
    assert reached == []


@pytest.mark.parametrize("scope", ["smoke", "non_hebrew"])
def test_hebrew_v3_preflight_does_not_block_diagnostics_or_other_languages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    config_yaml = (EXAMPLES_DIR / "config.v3-he-full.yaml").read_text(encoding="utf-8")
    args = _full_hebrew_args(config_yaml)
    args.update(
        {
            "model_id": "diagnostic/translator",
            "model_revision": "c" * 40,
            "max_new_tokens": 17,
            "translator_interface": "translategemma",
            "max_model_len": 1234,
            "trust_remote_code": False,
            "output_decoder": "standard",
            "max_rows": 17,
        }
    )
    if scope == "smoke":
        args["mode"] = "smoke"
        args["model_revision"] = "main"
        args["limit"] = 1
    else:
        args["config_yaml"] = config_yaml.replace("  - language: he", "  - language: fr").replace(
            "    - he", "    - fr"
        )
        args["target_language"] = "fr"

    class ExportReached(RuntimeError):
        pass

    def export_marker(*_args: object, **_kwargs: object) -> int:
        raise ExportReached

    monkeypatch.setattr(remote_translate, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(export_module, "export_raw_rows", export_marker)

    with pytest.raises(ExportReached):
        _run_backend_for_args(args)


def test_remote_body_uses_explicit_allocation_identity_not_container_globals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_yaml = (EXAMPLES_DIR / "config.v3-he-smoke.yaml").read_text(encoding="utf-8")
    args = _full_hebrew_args(config_yaml)
    args.update(
        mode="smoke",
        max_rows=2500,
        model_id="diagnostic/translator",
        model_revision="main",
        max_new_tokens=17,
        translator_interface="instruction_chat",
        max_model_len=4096,
        trust_remote_code=False,
        output_decoder="standard",
        limit=1,
        code_revision="unknown",
        source_tree_clean=None,
        allocation_gpu="L40S:1",
        function_timeout_seconds=3600,
    )
    observed_versions = {"python": "diagnostic-probe"}
    captured_runtime: dict[str, object] = {}
    captured_environment: dict[str, object] = {}
    captured_translate_kwargs: dict[str, object] = {}
    cache_commits = {"hf": 0, "vllm": 0}

    def fake_export(_source: object, rows_path: Path, **_kwargs: object) -> int:
        rows_path.write_text("{}\n", encoding="utf-8")
        return 1

    def fake_outputs(
        _out_dir: Path,
        _translated: object,
        stats: dict[str, object],
        **_kwargs: object,
    ) -> tuple[Path, Path]:
        runtime = stats["runtime"]
        environment = stats["environment"]
        assert isinstance(runtime, dict)
        assert isinstance(environment, dict)
        captured_runtime.update(runtime)
        captured_environment.update(environment)
        return tmp_path / "rows.he.jsonl", tmp_path / "translation_summary.json"

    def fake_translate_rows(
        *_args: object,
        **kwargs: object,
    ) -> tuple[list[object], dict[str, int]]:
        captured_translate_kwargs.update(kwargs)
        return [], {"input_rows": 1, "translated_rows": 0}

    def commit_cache(name: str) -> None:
        cache_commits[name] += 1

    # Simulate Modal importing the module inside a container without the
    # driver's launch environment. Explicit dispatch arguments remain the
    # evidence identity even though these globals now disagree.
    monkeypatch.setattr(remote_translate, "GPU", "container-global-gpu")
    monkeypatch.setattr(remote_translate, "TIMEOUT_SECONDS", 999)
    monkeypatch.setattr(remote_translate, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(remote_translate, "_package_versions", lambda _expected: observed_versions)
    monkeypatch.setattr(remote_translate, "artifacts_volume", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(
        remote_translate,
        "hf_cache_volume",
        SimpleNamespace(commit=lambda: commit_cache("hf")),
    )
    monkeypatch.setattr(
        remote_translate,
        "vllm_cache_volume",
        SimpleNamespace(commit=lambda: commit_cache("vllm")),
    )
    monkeypatch.setattr(export_module, "export_raw_rows", fake_export)
    monkeypatch.setattr(load_module, "load_raw_rows", lambda _path: [{"source_id": "root-1"}])
    monkeypatch.setattr(split_module, "prepare_split_result", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        split_module,
        "all_examples",
        lambda _result: [{"example_id": "root-1"}],
    )
    monkeypatch.setattr(translate_module, "load_translation_model", lambda _info: object())
    monkeypatch.setattr(translate_module, "translate_rows", fake_translate_rows)
    monkeypatch.setattr(translate_module, "write_translation_outputs", fake_outputs)

    remote_translate.run_remote_translation.get_raw_f()(**args)

    assert captured_runtime["gpu"] == "L40S:1"
    assert captured_runtime["backend"] == "vllm_chat"
    assert captured_runtime["gpu_allocation_label"] == "L40S:1"
    assert captured_runtime["function_timeout_seconds"] == 3600
    assert captured_environment == observed_versions
    assert captured_translate_kwargs["max_attempts"] == 3
    assert cache_commits == {"hf": 2, "vllm": 2}


def test_seq2seq_body_records_exact_runtime_and_never_commits_vllm_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _full_hebrew_args((EXAMPLES_DIR / "config.v3-he-smoke.yaml").read_text(encoding="utf-8"))
    args.update(
        mode="smoke",
        max_rows=2500,
        model_id="google/madlad400-3b-mt",
        model_revision="main",
        max_new_tokens=64,
        translator_interface="madlad_seq2seq",
        max_model_len=2048,
        trust_remote_code=False,
        output_decoder="standard",
        limit=1,
        code_revision="unknown",
        source_tree_clean=None,
        allocation_gpu="L40S:1",
        function_timeout_seconds=3600,
    )
    observed_versions = dict(SEQ2SEQ_TRANSLATION_RUNTIME_VERSIONS)
    captured_stats: dict[str, object] = {}
    captured_translate_kwargs: dict[str, object] = {}
    cache_commits = {"hf": 0, "vllm": 0}

    def fake_export(_source: object, rows_path: Path, **_kwargs: object) -> int:
        rows_path.write_text("{}\n", encoding="utf-8")
        return 1

    def fake_outputs(
        _out_dir: Path,
        _translated: object,
        stats: dict[str, object],
        **_kwargs: object,
    ) -> tuple[Path, Path]:
        captured_stats.update(stats)
        return tmp_path / "rows.he.jsonl", tmp_path / "translation_summary.json"

    def fake_translate_rows(
        *_args: object,
        **kwargs: object,
    ) -> tuple[list[object], dict[str, int]]:
        captured_translate_kwargs.update(kwargs)
        return [], {"input_rows": 1, "translated_rows": 0}

    def commit_cache(name: str) -> None:
        cache_commits[name] += 1

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
        SimpleNamespace(commit=lambda: commit_cache("hf")),
    )
    monkeypatch.setattr(
        remote_translate,
        "vllm_cache_volume",
        SimpleNamespace(commit=lambda: commit_cache("vllm")),
    )
    monkeypatch.setattr(export_module, "export_raw_rows", fake_export)
    monkeypatch.setattr(load_module, "load_raw_rows", lambda _path: [{"source_id": "root-1"}])
    monkeypatch.setattr(split_module, "prepare_split_result", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        split_module,
        "all_examples",
        lambda _result: [{"example_id": "root-1"}],
    )
    monkeypatch.setattr(translate_module, "load_translation_model", lambda _info: object())
    monkeypatch.setattr(translate_module, "translate_rows", fake_translate_rows)
    monkeypatch.setattr(translate_module, "write_translation_outputs", fake_outputs)

    remote_translate.run_remote_seq2seq_translation.get_raw_f()(**args)

    runtime = captured_stats["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["backend"] == "transformers_seq2seq"
    assert captured_stats["environment"] == observed_versions
    assert captured_stats["max_attempts"] == 1
    assert captured_translate_kwargs["max_attempts"] == 1
    assert cache_commits == {"hf": 2, "vllm": 0}
