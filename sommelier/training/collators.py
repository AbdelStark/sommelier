from __future__ import annotations

from typing import Final, Protocol, TypedDict

from sommelier.errors import SchemaValidationError

IGNORE_INDEX: Final = -100


class TokenEncoder(Protocol):
    """The tokenizer surface the collator depends on.

    Matches transformers tokenizers (``encode`` without special tokens plus
    a ``pad_token_id``), so tests can use tiny local stubs and the training
    stack stays optional.
    """

    pad_token_id: int

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


class CollatedBatch(TypedDict):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    labels: list[list[int]]


def build_completion_labels(
    input_ids: list[int],
    prompt_token_count: int,
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    """Builds completion-only labels: prompt masked, target preserved.

    Every prompt token receives ``ignore_index``; assistant target tokens
    keep their IDs (RFC-0004). A boundary that leaves no target tokens, or
    that is out of range, fails instead of silently training on the full
    sequence.
    """
    if prompt_token_count < 1:
        raise SchemaValidationError(
            f"prompt token count must be positive, got {prompt_token_count}",
            hint="The rendered prompt must contain at least one token.",
        )
    if prompt_token_count >= len(input_ids):
        raise SchemaValidationError(
            f"prompt token count {prompt_token_count} leaves no target tokens "
            f"in a sequence of {len(input_ids)}",
            hint="Check max_sequence_length and the formatted full_text; the "
            "assistant target must survive tokenization and truncation.",
        )
    return [ignore_index] * prompt_token_count + list(input_ids[prompt_token_count:])


def find_prompt_token_count(
    tokenizer: TokenEncoder,
    *,
    prompt_text: str,
    full_text: str,
    context: str = "example",
) -> int:
    """Proves the prompt boundary at the token level.

    ``full_text`` must start with ``prompt_text``, and the full sequence's
    token prefix must equal the prompt's own tokenization. When a tokenizer
    merges tokens across the boundary the proof fails and training must
    stop (RFC-0003: no fallback to full-sequence loss).
    """
    if not full_text.startswith(prompt_text):
        raise SchemaValidationError(
            f"{context}: full_text does not start with prompt_text",
            hint="Rebuild the formatted split; label masking requires the "
            "prompt-prefix property.",
        )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if not prompt_ids:
        raise SchemaValidationError(
            f"{context}: prompt tokenized to zero tokens",
            hint="The rendered prompt must contain at least one token.",
        )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise SchemaValidationError(
            f"{context}: tokenizer merges tokens across the prompt boundary",
            hint="The prompt boundary cannot be proven for this tokenizer "
            "and template; training refuses to fall back to full-sequence loss.",
        )
    return len(prompt_ids)


class CompletionOnlyCollator:
    """Collates formatted examples into completion-only training batches.

    Sequences are tokenized from ``full_text`` with the boundary proven per
    example, truncated to ``max_sequence_length`` (failing when truncation
    would remove every target token), and right-padded with the tokenizer
    pad token; padded positions get attention 0 and label ``ignore_index``.
    """

    def __init__(
        self,
        tokenizer: TokenEncoder,
        *,
        max_sequence_length: int,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if max_sequence_length < 2:
            raise SchemaValidationError(
                f"max_sequence_length must allow prompt and target tokens, "
                f"got {max_sequence_length}",
                hint="Set train.max_sequence_length to a larger value.",
            )
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.ignore_index = ignore_index

    def encode_example(self, example: dict[str, object]) -> tuple[list[int], list[int]]:
        example_id = str(example.get("example_id", "<unknown>"))
        prompt_text = str(example["prompt_text"])
        full_text = str(example["full_text"])
        prompt_token_count = find_prompt_token_count(
            self.tokenizer,
            prompt_text=prompt_text,
            full_text=full_text,
            context=f"example {example_id}",
        )
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        if len(input_ids) > self.max_sequence_length:
            if prompt_token_count >= self.max_sequence_length:
                raise SchemaValidationError(
                    f"example {example_id}: truncation to {self.max_sequence_length} "
                    "tokens would remove every target token",
                    hint="Raise train.max_sequence_length or shorten the prompt; "
                    "silent truncation of the target is not allowed.",
                )
            input_ids = input_ids[: self.max_sequence_length]
        labels = build_completion_labels(input_ids, prompt_token_count, self.ignore_index)
        return input_ids, labels

    def __call__(self, examples: list[dict[str, object]]) -> CollatedBatch:
        if not examples:
            raise SchemaValidationError(
                "cannot collate an empty batch",
                hint="Provide at least one formatted example per batch.",
            )
        encoded = [self.encode_example(example) for example in examples]
        batch_length = max(len(input_ids) for input_ids, _ in encoded)

        input_batch: list[list[int]] = []
        mask_batch: list[list[int]] = []
        label_batch: list[list[int]] = []
        for input_ids, labels in encoded:
            padding = batch_length - len(input_ids)
            input_batch.append(input_ids + [self.tokenizer.pad_token_id] * padding)
            mask_batch.append([1] * len(input_ids) + [0] * padding)
            label_batch.append(labels + [self.ignore_index] * padding)

        return CollatedBatch(
            input_ids=input_batch,
            attention_mask=mask_batch,
            labels=label_batch,
        )
