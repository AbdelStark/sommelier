from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Self, cast

import pytest
import yaml

import remote_semantic_review as remote_semantic_review_module
import sommelier.remote.images as image_module
from remote_semantic_review import run_remote_semantic_review
from sommelier.data.semantic_review import (
    BACK_TRANSLATOR_HF_ENV,
    EXPECTED_PRODUCER_PACKAGE_VERSIONS,
)
from sommelier.errors import UserInputError
from sommelier.remote.images import (
    SEMANTIC_REVIEW_HF_ENV,
    SEMANTIC_REVIEW_PACKAGES,
    SEMANTIC_REVIEW_PYTHON_VERSION,
)
from sommelier.reviewer import validated_reviewer_requirement
from tests.hebrew_v3_translation_evidence import (
    self_rehash_translation_contract_drift,
    write_phase_a_translation_evidence,
)

REVIEWER_ID = "fixture-reviewer"
REVIEWER_PUBLIC_KEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f"
)
REVIEWER_REQUIREMENT = validated_reviewer_requirement(REVIEWER_ID, REVIEWER_PUBLIC_KEY)


class _FakeImage:
    def pip_install(self, *packages: str) -> Self:
        assert packages == SEMANTIC_REVIEW_PACKAGES
        return self

    def env(self, values: dict[str, str]) -> Self:
        assert values == dict(SEMANTIC_REVIEW_HF_ENV)
        return self


def test_remote_dispatch_passes_local_source_identity_explicitly() -> None:
    signature = inspect.signature(run_remote_semantic_review.get_raw_f())
    assert list(signature.parameters) == [
        "translation_run_id",
        "code_revision",
        "source_tree_clean",
        "allocated_gpu",
        "allocated_timeout_seconds",
    ]
    assert list(inspect.signature(remote_semantic_review_module.main.info.raw_f).parameters) == [
        "translation_run_id"
    ]


def _write_phase_a_config(path: Path) -> bytes:
    payload = yaml.safe_load(Path("examples/config.v3-he-full.yaml").read_text(encoding="utf-8"))
    payload["semantic_review"] = {
        "reviewer": {
            "reviewer_id": REVIEWER_REQUIREMENT.reviewer_id,
            "ssh_public_key": REVIEWER_REQUIREMENT.ssh_public_key,
            "public_key_fingerprint": REVIEWER_REQUIREMENT.public_key_fingerprint,
        }
    }
    encoded = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    path.write_bytes(encoded)
    return encoded


def _write_phase_a_run_evidence(run_dir: Path, config_bytes: bytes) -> dict[str, object]:
    assert config_bytes == (run_dir / "config.yaml").read_bytes()
    summary_path, _identity_path = write_phase_a_translation_evidence(
        run_dir,
        config_path=run_dir / "config.yaml",
        run_id="he-v3-translate-full",
        source_boundary="Synthetic remote semantic-review producer identity.",
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("config_sha256", "does not bind the exact Phase-A config"),
        ("reviewer_payload", "reviewer preregistration does not match"),
        ("reviewer_sha256", "reviewer preregistration digest is invalid"),
        ("run_identity_sha256", "does not bind its pre-provider run identity"),
    ],
)
def test_remote_dispatch_rejects_unbound_phase_a_evidence_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    translation_root = tmp_path / "translation"
    run_dir = translation_root / "he-v3-translate-full"
    run_dir.mkdir(parents=True)
    config_bytes = _write_phase_a_config(run_dir / "config.yaml")
    (run_dir / "rows.en.jsonl").write_text("\n", encoding="utf-8")
    (run_dir / "rows.he.jsonl").write_text("\n", encoding="utf-8")
    summary = _write_phase_a_run_evidence(run_dir, config_bytes)
    if tamper == "config_sha256":
        selection = summary["selection"]
        assert isinstance(selection, dict)
        selection["config_sha256"] = "0" * 64
    elif tamper == "reviewer_payload":
        reviewer = summary["reviewer_preregistration"]
        assert isinstance(reviewer, dict)
        reviewer["reviewer_id"] = "post-hoc-reviewer"
    elif tamper == "reviewer_sha256":
        summary["reviewer_preregistration_sha256"] = "0" * 64
    else:
        summary["translation_run_identity_sha256"] = "0" * 64
    (run_dir / "translation_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    real_path = Path

    class StubVolume:
        def commit(self) -> None:
            return None

    def remapped_path(value: str) -> Path:
        if value == "/artifacts/translation":
            return translation_root
        return real_path(value)

    def unexpected_model_load() -> object:
        raise AssertionError("unbound Phase-A evidence must fail before model loading")

    monkeypatch.setattr(remote_semantic_review_module, "Path", remapped_path)
    monkeypatch.setattr(remote_semantic_review_module, "artifacts_volume", StubVolume())
    monkeypatch.setattr(
        "sommelier.data.semantic_review.load_transformers_backtranslator",
        unexpected_model_load,
    )

    with pytest.raises(UserInputError, match=message):
        run_remote_semantic_review.get_raw_f()(
            "he-v3-translate-full",
            "a" * 40,
            True,
            "A10G",
            14_400,
        )

    assert not (run_dir / "translation_semantic_review_template.json").exists()


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("translator", "request_sha256", "4" * 64, "forward translator request_sha256"),
        ("runtime", "translation_chunk_size", 31, "runtime translation_chunk_size"),
    ],
)
def test_remote_dispatch_rejects_self_rehashed_translation_contract_drift_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    translation_root = tmp_path / "translation"
    run_dir = translation_root / "he-v3-translate-full"
    run_dir.mkdir(parents=True)
    config_bytes = _write_phase_a_config(run_dir / "config.yaml")
    (run_dir / "rows.en.jsonl").write_text("\n", encoding="utf-8")
    (run_dir / "rows.he.jsonl").write_text("\n", encoding="utf-8")
    _write_phase_a_run_evidence(run_dir, config_bytes)
    self_rehash_translation_contract_drift(
        run_dir / "translation_summary.json",
        run_dir / "translation_run_identity.json",
        section=section,
        field=field,
        value=value,
    )
    real_path = Path

    class StubVolume:
        def commit(self) -> None:
            return None

    def remapped_path(value: str) -> Path:
        if value == "/artifacts/translation":
            return translation_root
        return real_path(value)

    def unexpected_model_load() -> object:
        raise AssertionError("translation contract drift must fail before model loading")

    monkeypatch.setattr(remote_semantic_review_module, "Path", remapped_path)
    monkeypatch.setattr(remote_semantic_review_module, "artifacts_volume", StubVolume())
    monkeypatch.setattr(
        "sommelier.data.semantic_review.load_transformers_backtranslator",
        unexpected_model_load,
    )

    with pytest.raises(UserInputError, match=message):
        run_remote_semantic_review.get_raw_f()(
            "he-v3-translate-full",
            "a" * 40,
            True,
            "A10G",
            14_400,
        )

    assert not (run_dir / "translation_semantic_review_template.json").exists()


def test_remote_dispatch_derives_template_reviewer_from_bound_phase_a_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_root = tmp_path / "translation"
    run_dir = translation_root / "he-v3-translate-full"
    run_dir.mkdir(parents=True)
    config_bytes = _write_phase_a_config(run_dir / "config.yaml")
    (run_dir / "rows.en.jsonl").write_text("\n", encoding="utf-8")
    (run_dir / "rows.he.jsonl").write_text("\n", encoding="utf-8")
    summary = _write_phase_a_run_evidence(run_dir, config_bytes)
    (run_dir / "translation_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    real_path = Path
    received: dict[str, object] = {}

    class StubVolume:
        def commit(self) -> None:
            return None

    def remapped_path(value: str) -> Path:
        if value == "/artifacts/translation":
            return translation_root
        return real_path(value)

    def fake_create_template(**kwargs: object) -> Path:
        received.update(kwargs)
        output_path = kwargs["output_path"]
        assert isinstance(output_path, Path)
        output_path.write_text("locked template\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr(remote_semantic_review_module, "Path", remapped_path)
    monkeypatch.setattr(remote_semantic_review_module, "artifacts_volume", StubVolume())
    monkeypatch.setattr(remote_semantic_review_module, "hf_cache_volume", StubVolume())
    monkeypatch.setattr("sommelier.data.load.load_raw_rows", lambda _path: [])
    monkeypatch.setattr(
        "sommelier.data.semantic_review.root_split_assignments",
        lambda _config, _rows: {},
    )
    monkeypatch.setattr(
        "sommelier.data.semantic_review.capture_producer_provenance",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "sommelier.data.semantic_review.validate_producer_provenance",
        lambda _producer, *, translation_summary: None,
    )
    monkeypatch.setattr(
        "sommelier.data.semantic_review.load_transformers_backtranslator",
        lambda: object(),
    )
    monkeypatch.setattr(
        "sommelier.data.semantic_review.create_semantic_review_template",
        fake_create_template,
    )

    result = run_remote_semantic_review.get_raw_f()(
        "he-v3-translate-full",
        "a" * 40,
        True,
        "A10G",
        14_400,
    )

    assert result == str(run_dir / "translation_semantic_review_template.json")
    assert received["reviewer_requirement"] == REVIEWER_REQUIREMENT


def test_local_entrypoint_rejects_unsafe_run_id_before_identity_or_remote_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched: list[object] = []

    def unexpected_identity() -> tuple[str, bool]:
        raise AssertionError("invalid run ids must fail before source identity lookup")

    def unexpected_remote(*args: object) -> str:
        dispatched.extend(args)
        raise AssertionError("invalid run ids must not rent a remote GPU")

    monkeypatch.setattr(
        remote_semantic_review_module,
        "_local_source_identity",
        unexpected_identity,
    )
    monkeypatch.setattr(run_remote_semantic_review, "remote", unexpected_remote)

    with pytest.raises(UserInputError, match="invalid semantic-review translation run id"):
        remote_semantic_review_module.main.info.raw_f(
            translation_run_id="../escape",
        )

    assert dispatched == []


@pytest.mark.parametrize(
    ("revision", "clean"),
    [("main", True), ("a" * 40, False), ("unknown", None)],
)
def test_local_entrypoint_rejects_unpublishable_source_before_remote_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    revision: str,
    clean: bool | None,
) -> None:
    dispatched: list[object] = []

    def unexpected_remote(*args: object) -> str:
        dispatched.extend(args)
        raise AssertionError("an unpublishable source must not rent a remote GPU")

    monkeypatch.setattr(
        remote_semantic_review_module,
        "_local_source_identity",
        lambda: (revision, clean),
    )
    monkeypatch.setattr(run_remote_semantic_review, "remote", unexpected_remote)

    with pytest.raises(UserInputError, match="clean immutable source revision"):
        remote_semantic_review_module.main.info.raw_f(
            translation_run_id="he-v3-translate-full",
        )

    assert dispatched == []


def test_semantic_review_image_matches_evidence_runtime_contract() -> None:
    assert SEMANTIC_REVIEW_PYTHON_VERSION == EXPECTED_PRODUCER_PACKAGE_VERSIONS["python"]
    assert SEMANTIC_REVIEW_PACKAGES == tuple(
        f"{name}=={package_version}"
        for name, package_version in EXPECTED_PRODUCER_PACKAGE_VERSIONS.items()
        if name != "python"
    )
    assert dict(SEMANTIC_REVIEW_HF_ENV) == BACK_TRANSLATOR_HF_ENV


def test_semantic_review_image_uses_exact_observed_python_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_versions: list[str] = []
    fake_image = _FakeImage()

    def fake_python_base(python_version: str) -> _FakeImage:
        observed_versions.append(python_version)
        return fake_image

    monkeypatch.setattr(image_module, "_python_base", fake_python_base)
    monkeypatch.setattr(image_module, "_with_source", lambda image: image)

    assert image_module.semantic_review_image() is fake_image
    assert observed_versions == [SEMANTIC_REVIEW_PYTHON_VERSION]


@pytest.mark.parametrize(
    ("revision", "clean"),
    [("main", True), ("a" * 40, False), ("unknown", None)],
)
def test_remote_dispatch_rejects_mutable_or_dirty_source(
    revision: str,
    clean: bool | None,
) -> None:
    with pytest.raises(UserInputError, match="clean immutable source revision"):
        run_remote_semantic_review.get_raw_f()(
            "missing-run",
            revision,
            clean,
            "A10G",
            14_400,
        )


@pytest.mark.parametrize(
    ("gpu", "timeout"),
    [("", 14_400), ("A10G", 0), ("A10G", -1), ("A10G", True)],
)
def test_remote_dispatch_rejects_invalid_allocation(
    gpu: str,
    timeout: int,
) -> None:
    with pytest.raises(UserInputError, match="explicit remote allocation"):
        run_remote_semantic_review.get_raw_f()(
            "missing-run",
            "a" * 40,
            True,
            gpu,
            timeout,
        )


@pytest.mark.parametrize(
    "translation_run_id",
    ["", "../escape", "/absolute", "name/child", r"name\child", "x" * 129],
)
def test_remote_dispatch_rejects_unsafe_run_id_before_filesystem_access(
    translation_run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_path(_value: str) -> Path:
        raise AssertionError("invalid run ids must fail before filesystem access")

    monkeypatch.setattr(remote_semantic_review_module, "Path", unexpected_path)

    with pytest.raises(UserInputError, match="invalid semantic-review translation run id"):
        run_remote_semantic_review.get_raw_f()(
            translation_run_id,
            "a" * 40,
            True,
            "A10G",
            14_400,
        )


def test_remote_dispatch_rejects_symlinked_translation_run_before_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_root = tmp_path / "translation"
    translation_root.mkdir()
    target = tmp_path / "aliased-target"
    target.mkdir()
    (translation_root / "he-v3-translate-full").symlink_to(target, target_is_directory=True)
    real_path = Path

    def remapped_path(value: str) -> Path:
        if value == "/artifacts/translation":
            return translation_root
        return real_path(value)

    def unexpected_config_load(_path: Path) -> None:
        raise AssertionError("a symlinked run must fail before reservation or config loading")

    monkeypatch.setattr(remote_semantic_review_module, "Path", remapped_path)
    monkeypatch.setattr("sommelier.config.load_config", unexpected_config_load)

    with pytest.raises(UserInputError, match="not a regular directory"):
        run_remote_semantic_review.get_raw_f()(
            "he-v3-translate-full",
            "a" * 40,
            True,
            "A10G",
            14_400,
        )

    assert list(target.iterdir()) == []


def test_remote_dispatch_refuses_to_overwrite_locked_template_before_loading_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_root = tmp_path / "translation"
    run_dir = translation_root / "he-v3-translate-full"
    run_dir.mkdir(parents=True)
    (run_dir / "translation_semantic_review_template.json").write_text(
        "locked\n",
        encoding="utf-8",
    )
    real_path = Path

    def remapped_path(value: str) -> Path:
        if value == "/artifacts/translation":
            return translation_root
        return real_path(value)

    def unexpected_config_load(_path: Path) -> None:
        raise AssertionError("locked-template refusal must precede config/data/model loading")

    monkeypatch.setattr(remote_semantic_review_module, "Path", remapped_path)
    monkeypatch.setattr("sommelier.config.load_config", unexpected_config_load)

    with pytest.raises(UserInputError, match="locked semantic-review template already exists"):
        run_remote_semantic_review.get_raw_f()(
            "he-v3-translate-full",
            "a" * 40,
            True,
            "A10G",
            14_400,
        )


def test_remote_dispatch_exclusively_reserves_template_and_releases_empty_failed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_root = tmp_path / "translation"
    run_dir = translation_root / "he-v3-translate-full"
    run_dir.mkdir(parents=True)
    output_path = run_dir / "translation_semantic_review_template.json"
    real_path = Path
    config_entered = Event()
    release_config = Event()
    config_calls: list[Path] = []
    model_calls: list[str] = []

    class ConfigStopped(RuntimeError):
        pass

    class ConcurrentConfigReached(RuntimeError):
        pass

    class StubVolume:
        def __init__(self) -> None:
            self.commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = StubVolume()

    def remapped_path(value: str) -> Path:
        if value == "/artifacts/translation":
            return translation_root
        return real_path(value)

    def blocking_config_load(path: Path) -> None:
        config_calls.append(path)
        if len(config_calls) > 1:
            raise ConcurrentConfigReached
        config_entered.set()
        if not release_config.wait(timeout=5):
            raise AssertionError("test did not release the first config load")
        raise ConfigStopped

    def unexpected_model_load() -> object:
        model_calls.append("model")
        raise AssertionError("failed reservation path must not load the model")

    monkeypatch.setattr(remote_semantic_review_module, "Path", remapped_path)
    monkeypatch.setattr(remote_semantic_review_module, "artifacts_volume", volume)
    monkeypatch.setattr("sommelier.config.load_config", blocking_config_load)
    monkeypatch.setattr(
        "sommelier.data.semantic_review.load_transformers_backtranslator",
        unexpected_model_load,
    )

    def invoke() -> str:
        return cast(
            str,
            run_remote_semantic_review.get_raw_f()(
                "he-v3-translate-full",
                "a" * 40,
                True,
                "A10G",
                14_400,
            ),
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(invoke)
        assert config_entered.wait(timeout=5)
        try:
            with pytest.raises(
                UserInputError,
                match="locked semantic-review template already exists",
            ):
                invoke()
        finally:
            release_config.set()
        with pytest.raises(ConfigStopped):
            first.result(timeout=5)

    assert not output_path.exists()
    assert volume.commits == 2
    assert len(config_calls) == 1
    assert model_calls == []

    with pytest.raises(ConcurrentConfigReached):
        invoke()
    assert len(config_calls) == 2
    assert not output_path.exists()
    assert volume.commits == 4


@pytest.mark.parametrize("replace_inode", [False, True])
def test_remote_dispatch_preserves_nonempty_or_replaced_failed_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_inode: bool,
) -> None:
    translation_root = tmp_path / "translation"
    run_dir = translation_root / "he-v3-translate-full"
    run_dir.mkdir(parents=True)
    output_path = run_dir / "translation_semantic_review_template.json"
    real_path = Path
    config_calls = 0

    class ConfigStopped(RuntimeError):
        pass

    class StubVolume:
        def __init__(self) -> None:
            self.commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = StubVolume()

    def remapped_path(value: str) -> Path:
        if value == "/artifacts/translation":
            return translation_root
        return real_path(value)

    def mutate_reservation_then_fail(_path: Path) -> None:
        nonlocal config_calls
        config_calls += 1
        if replace_inode:
            replacement = output_path.with_suffix(".replacement")
            replacement.write_bytes(b"")
            assert replacement.stat().st_ino != output_path.stat().st_ino
            replacement.replace(output_path)
        else:
            output_path.write_text("partial template\n", encoding="utf-8")
        raise ConfigStopped

    monkeypatch.setattr(remote_semantic_review_module, "Path", remapped_path)
    monkeypatch.setattr(remote_semantic_review_module, "artifacts_volume", volume)
    monkeypatch.setattr("sommelier.config.load_config", mutate_reservation_then_fail)

    def invoke() -> str:
        return cast(
            str,
            run_remote_semantic_review.get_raw_f()(
                "he-v3-translate-full",
                "a" * 40,
                True,
                "A10G",
                14_400,
            ),
        )

    with pytest.raises(ConfigStopped):
        invoke()

    assert output_path.is_file()
    assert output_path.read_bytes() == (b"" if replace_inode else b"partial template\n")
    assert volume.commits == 1

    with pytest.raises(
        UserInputError,
        match="locked semantic-review template already exists",
    ):
        invoke()
    assert config_calls == 1


def test_remote_dispatch_hard_crash_leaves_empty_reservation_for_explicit_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translation_root = tmp_path / "translation"
    run_dir = translation_root / "he-v3-translate-full"
    run_dir.mkdir(parents=True)
    output_path = run_dir / "translation_semantic_review_template.json"
    real_path = Path

    class HardCrash(BaseException):
        pass

    class StubVolume:
        def __init__(self) -> None:
            self.commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = StubVolume()

    def remapped_path(value: str) -> Path:
        if value == "/artifacts/translation":
            return translation_root
        return real_path(value)

    def crash(_path: Path) -> None:
        raise HardCrash

    monkeypatch.setattr(remote_semantic_review_module, "Path", remapped_path)
    monkeypatch.setattr(remote_semantic_review_module, "artifacts_volume", volume)
    monkeypatch.setattr("sommelier.config.load_config", crash)

    with pytest.raises(HardCrash):
        run_remote_semantic_review.get_raw_f()(
            "he-v3-translate-full",
            "a" * 40,
            True,
            "A10G",
            14_400,
        )

    assert output_path.is_file()
    assert output_path.read_bytes() == b""
    assert volume.commits == 1

    with pytest.raises(
        UserInputError,
        match="locked semantic-review template already exists",
    ):
        run_remote_semantic_review.get_raw_f()(
            "he-v3-translate-full",
            "a" * 40,
            True,
            "A10G",
            14_400,
        )
