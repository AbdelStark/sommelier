from __future__ import annotations

import pytest

from sommelier.errors import SchemaValidationError
from sommelier.training.collators import (
    IGNORE_INDEX,
    CompletionOnlyCollator,
    build_completion_labels,
    find_prompt_token_count,
)


class CharTokenizer:
    """One token per character; deterministic and boundary-faithful."""

    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]


class BoundaryMergingTokenizer(CharTokenizer):
    """Simulates a tokenizer that merges tokens across the prompt boundary."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(char) for char in text]
        if len(ids) >= 2:
            ids[0] = ids[0] * 1000 + ids[1]
            del ids[1]
        return ids


def example(prompt: str, target: str, example_id: str = "e1") -> dict[str, object]:
    return {
        "example_id": example_id,
        "prompt_text": prompt,
        "full_text": prompt + target,
    }


def test_labels_mask_prompt_and_preserve_target() -> None:
    input_ids = [10, 11, 12, 13, 14]
    labels = build_completion_labels(input_ids, prompt_token_count=3)
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 13, 14]


def test_labels_reject_boundary_without_target_tokens() -> None:
    with pytest.raises(SchemaValidationError):
        build_completion_labels([1, 2, 3], prompt_token_count=3)


@pytest.mark.parametrize("count", [0, -1])
def test_labels_reject_non_positive_prompt_counts(count: int) -> None:
    with pytest.raises(SchemaValidationError):
        build_completion_labels([1, 2, 3], prompt_token_count=count)


def test_prompt_boundary_proven_at_token_level() -> None:
    count = find_prompt_token_count(
        CharTokenizer(),
        prompt_text="PROMPT|",
        full_text="PROMPT|TARGET",
    )
    assert count == len("PROMPT|")


def test_boundary_merging_tokenizer_is_rejected() -> None:
    # The merged first token differs between prompt-only and full encodings
    # only when the prefix property breaks; force a boundary merge by using
    # a one-character prompt whose token fuses with the target's first char.
    with pytest.raises(SchemaValidationError):
        find_prompt_token_count(
            BoundaryMergingTokenizer(),
            prompt_text="P",
            full_text="PT",
        )


def test_full_text_must_start_with_prompt_text() -> None:
    with pytest.raises(SchemaValidationError):
        find_prompt_token_count(
            CharTokenizer(),
            prompt_text="PROMPT",
            full_text="DIFFERENT",
        )


def test_collator_masks_pads_and_preserves_targets() -> None:
    collator = CompletionOnlyCollator(CharTokenizer(), max_sequence_length=64)
    batch = collator([example("ab|", "XY"), example("q|", "Z", example_id="e2")])

    assert batch["input_ids"][0] == [ord(c) for c in "ab|XY"]
    assert batch["labels"][0] == [IGNORE_INDEX] * 3 + [ord("X"), ord("Y")]
    assert batch["attention_mask"][0] == [1, 1, 1, 1, 1]

    assert batch["input_ids"][1] == [ord(c) for c in "q|Z"] + [0, 0]
    assert batch["labels"][1] == [IGNORE_INDEX] * 2 + [ord("Z"), IGNORE_INDEX, IGNORE_INDEX]
    assert batch["attention_mask"][1] == [1, 1, 1, 0, 0]


def test_collator_truncates_but_keeps_some_target() -> None:
    collator = CompletionOnlyCollator(CharTokenizer(), max_sequence_length=5)
    batch = collator([example("abc", "XYZ")])
    assert batch["input_ids"][0] == [ord(c) for c in "abcXY"]
    assert batch["labels"][0] == [IGNORE_INDEX] * 3 + [ord("X"), ord("Y")]


def test_collator_rejects_truncation_that_removes_target() -> None:
    collator = CompletionOnlyCollator(CharTokenizer(), max_sequence_length=3)
    with pytest.raises(SchemaValidationError):
        collator([example("abc", "XYZ")])


def test_collator_rejects_empty_batch() -> None:
    collator = CompletionOnlyCollator(CharTokenizer(), max_sequence_length=8)
    with pytest.raises(SchemaValidationError):
        collator([])


def test_collator_rejects_tiny_max_sequence_length() -> None:
    with pytest.raises(SchemaValidationError):
        CompletionOnlyCollator(CharTokenizer(), max_sequence_length=1)


def test_no_unmasked_prompt_token_ever_survives() -> None:
    collator = CompletionOnlyCollator(CharTokenizer(), max_sequence_length=64)
    prompt = "system and user text|"
    target = '{"a":1}'
    batch = collator([example(prompt, target)])
    labels = batch["labels"][0]
    assert all(label == IGNORE_INDEX for label in labels[: len(prompt)])
    assert labels[len(prompt) :] == [ord(char) for char in target]
