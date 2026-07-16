from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from sommelier.data.semantic_review import (
    BACK_TRANSLATOR_BATCH_SIZE,
    BACK_TRANSLATOR_HF_ENV,
    BACK_TRANSLATOR_MODEL_ID,
    BACK_TRANSLATOR_MODEL_REVISION,
    load_transformers_backtranslator,
)
from sommelier.errors import UserInputError


def _install_fake_marian_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_lengths: list[int] | None = None,
    output_count: int | None = None,
) -> SimpleNamespace:
    state = SimpleNamespace(
        tokenizer_loads=[],
        model_loads=[],
        tokenizer_calls=[],
        decode_calls=[],
        generate_calls=[],
        eval_calls=0,
        source_lengths=source_lengths,
        output_count=output_count,
    )

    class FakeTensor:
        def __init__(self, batch_size: int, lengths: list[int]) -> None:
            self.batch_size = batch_size
            self.lengths = lengths
            self.devices: list[str] = []

        def sum(self, *, dim: int) -> SimpleNamespace:
            assert dim == 1
            return SimpleNamespace(tolist=lambda: self.lengths)

        def to(self, device: str) -> FakeTensor:
            self.devices.append(device)
            return self

    class StubTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> StubTokenizer:
            state.tokenizer_loads.append((model_id, kwargs))
            return cls()

        def __call__(
            self,
            texts: list[str],
            **kwargs: object,
        ) -> dict[str, FakeTensor]:
            state.tokenizer_calls.append((list(texts), kwargs))
            lengths = state.source_lengths or [len(text.split()) + 1 for text in texts]
            assert len(lengths) == len(texts)
            return {
                "input_ids": FakeTensor(len(texts), lengths),
                "attention_mask": FakeTensor(len(texts), lengths),
            }

        def batch_decode(self, generated: list[str], **kwargs: object) -> list[str]:
            state.decode_calls.append((list(generated), kwargs))
            return generated

    class StubModel:
        device = "cuda:0"

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> StubModel:
            state.model_loads.append((model_id, kwargs))
            return cls()

        def eval(self) -> None:
            state.eval_calls += 1

        def generate(self, **kwargs: object) -> list[str]:
            state.generate_calls.append(kwargs)
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, FakeTensor)
            count = state.output_count
            if count is None:
                count = input_ids.batch_size
            return [f"English backtranslation {index}" for index in range(count)]

    class InferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> None:
            return None

    fake_torch = ModuleType("torch")
    setattr(fake_torch, "float16", "stub-float16")
    setattr(fake_torch, "inference_mode", InferenceMode)
    fake_transformers = ModuleType("transformers")
    setattr(fake_transformers, "AutoModelForSeq2SeqLM", StubModel)
    setattr(fake_transformers, "AutoTokenizer", StubTokenizer)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return state


def test_marian_backtranslator_is_pinned_bounded_and_greedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_marian_runtime(monkeypatch)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    backtranslator = load_transformers_backtranslator()
    texts = ["מצא טיסות לברלין", "בטל את ההזמנה"]

    outputs = backtranslator.translate_batch(texts)

    assert outputs == ["English backtranslation 0", "English backtranslation 1"]
    assert state.tokenizer_loads == [
        (
            BACK_TRANSLATOR_MODEL_ID,
            {
                "revision": BACK_TRANSLATOR_MODEL_REVISION,
                "use_fast": False,
                "trust_remote_code": False,
            },
        )
    ]
    assert state.model_loads == [
        (
            BACK_TRANSLATOR_MODEL_ID,
            {
                "revision": BACK_TRANSLATOR_MODEL_REVISION,
                "device_map": "auto",
                "dtype": "stub-float16",
                "trust_remote_code": False,
            },
        )
    ]
    assert state.eval_calls == 1
    assert state.tokenizer_calls == [
        (
            texts,
            {
                "add_special_tokens": True,
                "padding": "longest",
                "truncation": False,
                "return_tensors": "pt",
            },
        )
    ]
    assert len(state.generate_calls) == 1
    generate = state.generate_calls[0]
    assert generate["do_sample"] is False
    assert generate["num_beams"] == 1
    assert generate["max_new_tokens"] == 512
    assert state.decode_calls[0][1] == {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
    assert {name: os.environ[name] for name in BACK_TRANSLATOR_HF_ENV} == (BACK_TRANSLATOR_HF_ENV)


def test_marian_backtranslator_rejects_source_over_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_marian_runtime(monkeypatch, source_lengths=[513])
    backtranslator = load_transformers_backtranslator()

    with pytest.raises(UserInputError, match="513 tokens, above the preregistered 512-token"):
        backtranslator.translate_batch(["טקסט ארוך"])

    assert state.generate_calls == []


def test_marian_backtranslator_rejects_oversized_batch_before_tokenization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_marian_runtime(monkeypatch)
    backtranslator = load_transformers_backtranslator()

    with pytest.raises(UserInputError, match="9 rows, above the preregistered 8-row"):
        backtranslator.translate_batch(["בדיקה"] * (BACK_TRANSLATOR_BATCH_SIZE + 1))

    assert state.tokenizer_calls == []


def test_marian_backtranslator_rejects_output_cardinality_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_marian_runtime(monkeypatch, output_count=1)
    backtranslator = load_transformers_backtranslator()

    with pytest.raises(UserInputError, match="wrong number of outputs"):
        backtranslator.translate_batch(["ראשון", "שני"])
