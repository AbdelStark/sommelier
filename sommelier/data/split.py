from __future__ import annotations

import random
from dataclasses import dataclass
from typing import cast, get_args

from sommelier.data.types import DropReason, PreparedExample, RawToolCallRow, SplitName
from sommelier.data.validate import validate_raw_row
from sommelier.errors import UserInputError

# Derived from the DropReason literal so new reasons can never desync the
# counter initialization (a hand-maintained copy once missed a reason and
# crashed data preparation at runtime).
ALL_DROP_REASONS: tuple[DropReason, ...] = get_args(DropReason)


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
    language: str,
) -> tuple[list[PreparedExample], dict[DropReason, int], int]:
    drop_counts = empty_drop_counts()
    validated: list[PreparedExample] = []

    for row in raw_rows:
        result = validate_raw_row(
            row,
            min_query_chars=min_query_chars,
            max_query_chars=max_query_chars,
            language=language,
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
    language: str = "en",
) -> SplitResult:
    """Validates, deduplicates, and splits the root source's rows.

    Split assignment happens only here: paired sources inherit it through
    :func:`pair_split_result` and never shuffle on their own.
    """
    deduplicated, drop_counts, valid_rows = validate_and_deduplicate_rows(
        raw_rows,
        min_query_chars=min_query_chars,
        max_query_chars=max_query_chars,
        language=language,
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


def all_examples(result: SplitResult) -> list[PreparedExample]:
    return [*result.train, *result.validation, *result.test]


def examples_for_split(result: SplitResult, split: SplitName) -> list[PreparedExample]:
    per_split: dict[SplitName, list[PreparedExample]] = {
        "train": result.train,
        "validation": result.validation,
        "test": result.test,
    }
    return per_split[split]


def pair_split_result(
    root_result: SplitResult,
    root_rows: list[RawToolCallRow],
    paired_rows: list[RawToolCallRow],
    *,
    min_query_chars: int,
    max_query_chars: int,
    language: str,
) -> SplitResult:
    """Builds a paired language's splits by inheritance from the root result.

    Each paired row must name a root example through ``source_example_id``
    and carry byte-identical ``tools`` and ``answers``: the gold contract is
    enforced here as a pipeline invariant, not trusted from the dataset
    producer. Rows whose root example was dropped or not selected, duplicate
    pairings, and mutated payloads are dropped with their own counted
    reasons. Output splits follow the root split's example order, so the
    paired dataset is deterministic given the root result.
    """
    # Validation keeps each example next to the exact raw row it was built
    # from: the byte-identity checks below must compare that row, not a
    # lookalike found by source_id (a duplicated id could otherwise smuggle
    # a mutated payload past the check).
    drop_counts = empty_drop_counts()
    validated: list[tuple[PreparedExample, RawToolCallRow]] = []
    for row in paired_rows:
        result = validate_raw_row(
            row,
            min_query_chars=min_query_chars,
            max_query_chars=max_query_chars,
            language=language,
        )
        if isinstance(result, str):
            drop_counts[result] += 1
            continue
        validated.append((result, row))
    valid_rows = len(validated)

    seen_queries: set[str] = set()
    deduplicated: list[tuple[PreparedExample, RawToolCallRow]] = []
    for example, row in validated:
        if example["query_sha256"] in seen_queries:
            drop_counts["duplicate_query"] += 1
            continue
        seen_queries.add(example["query_sha256"])
        deduplicated.append((example, row))

    root_raw_by_id = {row["source_id"]: row for row in root_rows}
    root_examples = all_examples(root_result)
    root_split_by_id: dict[str, SplitName] = {
        example["example_id"]: example["split"] for example in root_examples
    }
    root_split_by_digest: dict[str, SplitName] = {
        example["query_sha256"]: example["split"] for example in root_examples
    }

    paired_by_root_id: dict[str, PreparedExample] = {}
    for example, row in deduplicated:
        source_example_id = example["source_example_id"]
        if source_example_id is None or source_example_id not in root_split_by_id:
            drop_counts["missing_source_example"] += 1
            continue
        if source_example_id in paired_by_root_id:
            drop_counts["duplicate_source_example"] += 1
            continue
        # A root example's id is its raw row's source_id, so the raw row is
        # guaranteed present; a KeyError here means the caller passed
        # inconsistent inputs and deserves the crash.
        root_raw = root_raw_by_id[source_example_id]
        if row["tools"] != root_raw["tools"]:
            drop_counts["pair_tools_mismatch"] += 1
            continue
        if row["answers"] != root_raw["answers"]:
            drop_counts["pair_answers_mismatch"] += 1
            continue
        inherited_split = root_split_by_id[source_example_id]
        colliding_split = root_split_by_digest.get(example["query_sha256"])
        if colliding_split is not None and colliding_split != inherited_split:
            # The paired query text coincidentally equals a root query that
            # sits in another split (e.g. a translation that came out
            # identical to a different English row). Keeping it would put
            # the same content on both sides of a split boundary.
            drop_counts["cross_split_duplicate"] += 1
            continue
        paired_by_root_id[source_example_id] = example

    def inherit(root_split: list[PreparedExample], split: SplitName) -> list[PreparedExample]:
        inherited: list[PreparedExample] = []
        for root_example in root_split:
            paired = paired_by_root_id.get(root_example["example_id"])
            if paired is not None:
                inherited.append(_with_split(paired, split))
        return inherited

    return SplitResult(
        train=inherit(root_result.train, "train"),
        validation=inherit(root_result.validation, "validation"),
        test=inherit(root_result.test, "test"),
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


def assert_multilingual_disjointness(
    results: dict[str, SplitResult],
    *,
    root_language: str,
) -> None:
    """Split safety across languages: globally unique example ids, every
    paired example in the same split as the root example it names, and no
    query digest on both sides of any split boundary (which subsumes
    per-language split disjointness)."""
    seen_ids: set[str] = set()
    for result in results.values():
        for example in all_examples(result):
            example_id = example["example_id"]
            if example_id in seen_ids:
                raise UserInputError(
                    f"example_id {example_id!r} appears more than once across the "
                    "prepared splits",
                    hint="Example ids must be unique across languages; a paired "
                    "source must namespace its source_id instead of reusing the "
                    "root row's.",
                )
            seen_ids.add(example_id)

    root_split_by_id = {
        example["example_id"]: example["split"]
        for example in all_examples(results[root_language])
    }
    for language, result in results.items():
        if language == root_language:
            continue
        for example in all_examples(result):
            source_example_id = example["source_example_id"]
            if source_example_id is None:
                raise UserInputError(
                    f"paired example {example['example_id']!r} has no source_example_id"
                )
            if root_split_by_id.get(source_example_id) != example["split"]:
                raise UserInputError(
                    f"paired example {example['example_id']!r} is in split "
                    f"{example['split']!r} but its root example is not"
                )

    # pair_split_result screens collisions against the root only; with more
    # than one paired language, two paired rows can still collide with each
    # other, and that surfaces here as a hard error rather than a drop.
    split_by_digest: dict[str, str] = {}
    for result in results.values():
        for example in all_examples(result):
            digest = example["query_sha256"]
            known_split = split_by_digest.setdefault(digest, example["split"])
            if known_split != example["split"]:
                raise UserInputError(
                    f"query digest {digest} appears in splits "
                    f"{known_split!r} and {example['split']!r}"
                )


def _with_split(example: PreparedExample, split: SplitName) -> PreparedExample:
    # A dict spread instead of a field-by-field copy so a future schema
    # field cannot be silently dropped here.
    return cast(PreparedExample, {**example, "split": split})
