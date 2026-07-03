from __future__ import annotations

import hashlib
from typing import Protocol

from sommelier.config import SommelierConfig
from sommelier.errors import ExternalDependencyError, InvariantViolation
from sommelier.formatting.chat import ChatMessage, build_messages

FORMATTED_EXAMPLE_SCHEMA = "sommelier.formatted_example.v1"


class ChatTemplateRenderer(Protocol):
    """The tokenizer surface the formatter depends on.

    Any object with transformers' ``apply_chat_template(...)`` shape works,
    including tiny local stubs in tests, so rendering never requires the
    optional training stack at import time.
    """

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


def prompt_sha256(prompt_text: str) -> str:
    """Digest that proves prompt identity across training and evaluation."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def _as_plain_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": message["role"], "content": message["content"]} for message in messages]


def render_formatted_example(
    example: dict[str, object],
    *,
    tokenizer: ChatTemplateRenderer,
    tokenizer_id: str,
    tokenizer_revision: str,
    system_prompt: str,
    template_policy: str,
) -> dict[str, object]:
    """Renders one prepared example through the tokenizer chat template.

    Produces a ``sommelier.formatted_example.v1`` record:

    - ``prompt_text``: system and user messages rendered with the generation
      prompt appended.
    - ``target_text``: the canonical JSON gold calls (assistant content).
    - ``full_text``: all three messages rendered through the template.
    - ``prompt_sha256``: SHA-256 of ``prompt_text``.

    Fails with InvariantViolation when the template does not render
    ``full_text`` as ``prompt_text`` followed by the target, because label
    masking needs a provable prompt boundary.
    """
    example_id = str(example.get("example_id", "<unknown>"))
    messages = build_messages(
        query=str(example["query"]),
        tools=list(_require_list(example, "tools", example_id)),
        gold_calls=example["gold_calls"],
        system_prompt=system_prompt,
        context=f"example {example_id}",
    )
    plain = _as_plain_messages(messages)
    prompt_text = tokenizer.apply_chat_template(
        plain[:2],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        plain,
        tokenize=False,
        add_generation_prompt=False,
    )
    target_text = messages[2]["content"]

    if not full_text.startswith(prompt_text):
        raise InvariantViolation(
            f"example {example_id}: chat template does not preserve the prompt prefix",
            hint="Label masking needs full_text to start with prompt_text; "
            "check the tokenizer chat template and template_policy.",
        )
    if target_text not in full_text[len(prompt_text) :]:
        raise InvariantViolation(
            f"example {example_id}: rendered full_text does not contain the target",
            hint="The assistant target must appear after the prompt in full_text.",
        )

    return {
        "schema_version": FORMATTED_EXAMPLE_SCHEMA,
        "example_id": example["example_id"],
        "split": example["split"],
        "messages": plain,
        "prompt_text": prompt_text,
        "target_text": target_text,
        "full_text": full_text,
        "prompt_sha256": prompt_sha256(prompt_text),
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "template_policy": template_policy,
    }


def _require_list(example: dict[str, object], key: str, example_id: str) -> list[object]:
    value = example.get(key)
    if not isinstance(value, list):
        raise InvariantViolation(
            f"example {example_id}: {key} must be a list",
            hint="Regenerate the prepared split with the current pipeline version.",
        )
    return value


def load_tokenizer(config: SommelierConfig) -> ChatTemplateRenderer:
    """Loads the configured tokenizer via transformers.

    transformers is an optional extra; importing it happens here, inside the
    stage function, never at package import time. trust_remote_code follows
    the config security policy and defaults to False.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ExternalDependencyError(
            "tokenizer rendering requires the transformers package",
            hint="Install the training extra, for example: uv sync --extra train",
        ) from error

    tokenizer: ChatTemplateRenderer = AutoTokenizer.from_pretrained(
        config.model.base_model_id,
        revision=config.model.tokenizer_revision,
        trust_remote_code=config.model.allow_remote_code,
    )
    return tokenizer
