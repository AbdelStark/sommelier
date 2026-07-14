from __future__ import annotations

import inspect
from typing import Self

import pytest

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
