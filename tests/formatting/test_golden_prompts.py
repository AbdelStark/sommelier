from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from sommelier.config import load_config
from sommelier.formatting.templates import render_formatted_example

FIXTURES_DIR = Path("tests/fixtures/formatting")
PREPARED_PATH = FIXTURES_DIR / "prepared_examples.jsonl"
GOLDEN_PATH = FIXTURES_DIR / "golden_formatted_examples.jsonl"
SMOKE_CONFIG = Path("examples/config.smoke.yaml")

REGENERATE_ENV = "SOMMELIER_REGENERATE_GOLDEN"


class StubTokenizer:
    """Deterministic local chat template; no model download in tests."""

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


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_all() -> list[dict[str, object]]:
    config = load_config(SMOKE_CONFIG)
    tokenizer = StubTokenizer()
    return [
        render_formatted_example(
            example,
            tokenizer=tokenizer,
            tokenizer_id="fixture/stub-tokenizer",
            tokenizer_revision="fixture-rev-1",
            system_prompt=config.formatting.system_prompt,
            template_policy=config.formatting.template_policy,
        )
        for example in load_jsonl(PREPARED_PATH)
    ]


def test_golden_rendered_messages_and_digests_are_stable() -> None:
    rendered = render_all()

    if os.environ.get(REGENERATE_ENV) == "1":
        GOLDEN_PATH.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in rendered),
            encoding="utf-8",
        )

    golden = load_jsonl(GOLDEN_PATH)
    assert len(golden) == len(rendered) == 3

    for got, expected in zip(rendered, golden, strict=True):
        got_normalized = json.loads(json.dumps(got, sort_keys=True))
        assert got_normalized == expected, (
            f"template drift for example {expected.get('example_id')}; "
            f"if intentional, regenerate with {REGENERATE_ENV}=1 and review the diff"
        )


def test_golden_fixture_covers_all_splits() -> None:
    golden = load_jsonl(GOLDEN_PATH)
    assert [record["split"] for record in golden] == ["train", "validation", "test"]


def test_golden_digests_match_prompt_text() -> None:
    for record in load_jsonl(GOLDEN_PATH):
        prompt_text = str(record["prompt_text"])
        expected = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        assert record["prompt_sha256"] == expected


def test_rendering_is_deterministic_across_calls() -> None:
    first = render_all()
    second = render_all()
    assert first == second
