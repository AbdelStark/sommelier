from __future__ import annotations

import random
from dataclasses import dataclass

from sommelier.data.types import DropReason, PreparedExample, RawToolCallRow, SplitName
from sommelier.data.validate import validate_raw_row
from sommelier.errors import UserInputError

ALL_DROP_REASONS: tuple[DropReason, ...] = (
    "missing_query",
    "missing_tools",
    "missing_answers",
    "query_too_short",
    "query_too_long",
    "invalid_tools_json",
    "invalid_answers_json",
    "invalid_tool_shape",
    "invalid_answer_shape",
    "duplicate_query",
)


@dataclass(frozen=True)
class SplitResult:
    train: list[PreparedExample]
    validation: list[PreparedExample]
    test: list[PreparedExample]
    drop_counts: dict[DropReason, int]
    valid_rows: int
    deduplicated_rows: int


def empty_drop_counts() -> dict[DropReason, int]:
    return {reason: 0 for reason in ALL_DROP_REASONS}


def validate_and_deduplicate_rows(
    raw_rows: list[RawToolCallRow],
    *,
    min_query_chars: int,
    max_query_chars: int,
) -> tuple[list[PreparedExample], dict[DropReason, int], int]:
    drop_counts = empty_drop_counts()
    validated: list[PreparedExample] = []

    for row in raw_rows:
        result = validate_raw_row(
            row,
            min_query_chars=min_query_chars,
            max_query_chars=max_query_chars,
        )
        if isinstance(result, str):
            drop_counts[result] += 1
            continue
        validated.append(result)

    seen_queries: set[str] = set()
    deduplicated: list[PreparedExample] = []
    for example in validated:
        query_hash = example["query_sha256"]
        if query_hash in seen_queries:
            drop_counts["duplicate_query"] += 1
            continue
        seen_queries.add(query_hash)
        deduplicated.append(example)

    return deduplicated, drop_counts, len(validated)


def split_examples(
    examples: list[PreparedExample],
    *,
    n_train: int,
    n_validation: int,
    n_test: int,
    seed: int,
) -> tuple[list[PreparedExample], list[PreparedExample], list[PreparedExample]]:
    required = n_train + n_validation + n_test
    if len(examples) < required:
        raise UserInputError(
            f"insufficient valid rows: need {required}, got {len(examples)}",
            hint="Lower split counts or provide more valid deduplicated rows.",
        )

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    train = [_with_split(example, "train") for example in shuffled[:n_train]]
    validation = [
        _with_split(example, "validation")
        for example in shuffled[n_train : n_train + n_validation]
    ]
    test = [
        _with_split(example, "test")
        for example in shuffled[n_train + n_validation : required]
    ]
    return train, validation, test


def prepare_split_result(
    raw_rows: list[RawToolCallRow],
    *,
    min_query_chars: int,
    max_query_chars: int,
    n_train: int,
    n_validation: int,
    n_test: int,
    seed: int,
) -> SplitResult:
    deduplicated, drop_counts, valid_rows = validate_and_deduplicate_rows(
        raw_rows,
        min_query_chars=min_query_chars,
        max_query_chars=max_query_chars,
    )
    train, validation, test = split_examples(
        deduplicated,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
        seed=seed,
    )
    return SplitResult(
        train=train,
        validation=validation,
        test=test,
        drop_counts=drop_counts,
        valid_rows=valid_rows,
        deduplicated_rows=len(deduplicated),
    )


def assert_split_disjointness(result: SplitResult) -> None:
    train_hashes = {example["query_sha256"] for example in result.train}
    validation_hashes = {example["query_sha256"] for example in result.validation}
    test_hashes = {example["query_sha256"] for example in result.test}
    if train_hashes & validation_hashes:
        raise UserInputError("train and validation splits overlap on query_sha256")
    if train_hashes & test_hashes:
        raise UserInputError("train and test splits overlap on query_sha256")
    if validation_hashes & test_hashes:
        raise UserInputError("validation and test splits overlap on query_sha256")


def _with_split(example: PreparedExample, split: SplitName) -> PreparedExample:
    return PreparedExample(
        schema_version=example["schema_version"],
        example_id=example["example_id"],
        source_id=example["source_id"],
        query=example["query"],
        tools=example["tools"],
        gold_calls=example["gold_calls"],
        split=split,
        query_sha256=example["query_sha256"],
        source_revision=example["source_revision"],
    )
