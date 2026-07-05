from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from sommelier.errors import (
    ExternalDependencyError,
    InvariantViolation,
    SchemaValidationError,
)
from sommelier.formatting.templates import prompt_sha256, render_formatted_example

SYSTEM_PROMPT = (
    "You are a tool-calling model. Select the correct tool and return only "
    "the JSON tool call. Do not include explanations."
)


class StubTokenizer:
    """Tiny deterministic chat template used instead of a downloaded tokenizer."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        rendered = "".join(
            f"<|{message['role']}|>{message['content']}<|end|>" for message in conversation
        )
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered


class PrefixBreakingTokenizer(StubTokenizer):
    """Renders prompts that are not a prefix of the full text."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        if add_generation_prompt:
            return "PROMPT-ONLY-RENDERING"
        return "FULL-RENDERING"


def make_example() -> dict[str, object]:
    return {
        "schema_version": "sommelier.prepared_example.v2",
        "example_id": "train-1",
        "source_id": "fixture:train-1",
        "language": "en",
        "source_example_id": None,
        "query": "What is the weather in Paris today?",
        "tools": [
            {
                "name": "lookup_weather",
                "description": "Look up weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "gold_calls": [{"name": "lookup_weather", "arguments": {"city": "Paris"}}],
        "split": "train",
        "query_sha256": "placeholder",
        "source_revision": "fixture",
    }


def render() -> dict[str, object]:
    return render_formatted_example(
        make_example(),
        tokenizer=StubTokenizer(),
        tokenizer_id="stub/tokenizer",
        tokenizer_revision="rev-1",
        system_prompt=SYSTEM_PROMPT,
        template_policy="tokenizer_chat_template",
    )


def test_formatted_example_contains_required_fields() -> None:
    record = render()
    for field in (
        "schema_version",
        "example_id",
        "split",
        "messages",
        "prompt_text",
        "target_text",
        "full_text",
        "prompt_sha256",
        "tokenizer_id",
        "tokenizer_revision",
        "template_policy",
    ):
        assert field in record, field
    assert record["schema_version"] == "sommelier.formatted_example.v1"
    assert record["tokenizer_id"] == "stub/tokenizer"
    assert record["tokenizer_revision"] == "rev-1"


def test_prompt_text_ends_with_generation_prompt() -> None:
    record = render()
    prompt_text = str(record["prompt_text"])
    assert prompt_text.endswith("<|assistant|>")
    assert "<|system|>" in prompt_text
    assert "<|user|>What is the weather in Paris today?<|end|>" in prompt_text
    assert '"name":"lookup_weather"' in prompt_text


def test_full_text_starts_with_prompt_and_contains_target() -> None:
    record = render()
    prompt_text = str(record["prompt_text"])
    full_text = str(record["full_text"])
    target_text = str(record["target_text"])
    assert full_text.startswith(prompt_text)
    assert target_text in full_text
    assert target_text == '[{"arguments":{"city":"Paris"},"name":"lookup_weather"}]'


def test_prompt_digest_is_sha256_of_prompt_text() -> None:
    record = render()
    prompt_text = str(record["prompt_text"])
    expected = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    assert record["prompt_sha256"] == expected
    assert prompt_sha256(prompt_text) == expected


def test_prompt_digest_is_stable_golden_value() -> None:
    first = render()
    second = render()
    assert first["prompt_sha256"] == second["prompt_sha256"]
    assert first["prompt_sha256"] == (
        "90d842d208ffbc385538d0ab12363e496935dfa22a232c83cf892093f563b4ab"
    )


def test_prefix_breaking_template_is_rejected() -> None:
    with pytest.raises(InvariantViolation):
        render_formatted_example(
            make_example(),
            tokenizer=PrefixBreakingTokenizer(),
            tokenizer_id="stub/tokenizer",
            tokenizer_revision="rev-1",
            system_prompt=SYSTEM_PROMPT,
            template_policy="tokenizer_chat_template",
        )


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is not None,
    reason="transformers installed; missing-dependency path not reachable",
)
def test_load_tokenizer_requires_transformers() -> None:
    from pathlib import Path

    from sommelier.config import load_config
    from sommelier.formatting.templates import load_tokenizer

    config = load_config(Path("examples/config.smoke.yaml"))
    with pytest.raises(ExternalDependencyError):
        load_tokenizer(config)


def test_format_stage_rejects_stale_prepared_schema(tmp_path: Path) -> None:
    import json

    from sommelier.config import load_config
    from sommelier.formatting.chat import build_formatted_splits_fixture
    from sommelier.run_context import ensure_run_context

    config = load_config(Path("examples/config.smoke.yaml"))
    context = ensure_run_context(
        config,
        config_path=Path("examples/config.smoke.yaml"),
        run_id="stale-schema",
        project_root=tmp_path,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    stale = dict(make_example())
    stale["schema_version"] = "sommelier.prepared_example.v1"
    for split in ("train", "validation", "test"):
        (data_dir / f"{split}.jsonl").write_text(
            json.dumps(stale) + "\n", encoding="utf-8"
        )
    with pytest.raises(SchemaValidationError, match="sommelier.prepared_example.v2"):
        build_formatted_splits_fixture(
            config,
            data_dir=data_dir,
            out_dir=tmp_path / "formatted",
            context=context,
            command=["test"],
        )
