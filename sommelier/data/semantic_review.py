"""Preregistered semantic review gate for published paired translations.

The lexical translation audit protects exact gold-bearing spans, but it cannot
detect a fluent translation that changes an action (``draw`` cards versus
illustrate), polarity, or entity relation.  This module adds a deliberately
separate release gate:

* select exactly 200 accepted pairs before any judgments are entered;
* backtranslate them with a pinned model that differs from the forward model;
* lock sample membership and every machine-produced review input by digest;
* accept reviewer edits only in the rubric/decision fields; and
* fail the whole publication when any sampled row has a critical error.

The reviewer is explicitly non-native.  Backtranslation-assisted review is a
bounded regression check, not a claim of native-speaker validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final, Literal, Protocol, cast

from sommelier.artifacts import sha256_file
from sommelier.config import SommelierConfig
from sommelier.data.load import load_raw_rows
from sommelier.data.split import all_examples, prepare_split_result
from sommelier.data.translate import PUBLICATION_CANONICAL_FIELDS, protected_spans
from sommelier.data.types import RawToolCallRow, SplitName, ToolCall
from sommelier.data.validate import parse_gold_calls
from sommelier.errors import ExternalDependencyError, UserInputError

SEMANTIC_REVIEW_SCHEMA: Final = "sommelier.translation_semantic_review.v1"
SEMANTIC_REVIEW_TEMPLATE_SCHEMA: Final = "sommelier.translation_semantic_review_template.v1"
SEMANTIC_SELECTION_SCHEMA: Final = "sommelier.semantic_review_selection.v1"
BACKTRANSLATION_REQUEST_SCHEMA: Final = "sommelier.marian_backtranslation_request.v1"
BACKTRANSLATION_BACKEND_SCHEMA: Final = "sommelier.transformers_marian_backtranslator.v1"
SEMANTIC_REVIEW_FILENAME: Final = "translation_semantic_review.json"
SEMANTIC_REVIEW_TEMPLATE_FILENAME: Final = "translation_semantic_review_template.json"
SEMANTIC_REVIEW_SAMPLE_SIZE: Final = 200
HIGH_RISK_SAMPLE_QUOTA: Final = 40

BACK_TRANSLATOR_MODEL_ID: Final = "Helsinki-NLP/opus-mt-tc-big-he-en"
BACK_TRANSLATOR_MODEL_REVISION: Final = "134c5a850dcaa763eec85bd1f4eb25112fecedbb"
BACK_TRANSLATOR_LICENSE: Final = "cc-by-4.0"
BACK_TRANSLATOR_ATTRIBUTION: Final = "Helsinki-NLP, OPUS-MT project"
BACK_TRANSLATOR_MODEL_CARD_URL: Final = "https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-he-en"
BACK_TRANSLATOR_MAX_NEW_TOKENS: Final = 512
BACK_TRANSLATOR_MAX_SOURCE_TOKENS: Final = 512
BACK_TRANSLATOR_BATCH_SIZE: Final = 8
BACK_TRANSLATOR_DTYPE: Final = "float16"
BACK_TRANSLATOR_DEVICE_MAP: Final = "auto"
BACK_TRANSLATOR_HF_ENV: Final = {
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_DOWNLOAD_TIMEOUT": "600",
}

EXPECTED_PRODUCER_PACKAGE_VERSIONS: Final = {
    "python": "3.13.3",
    "torch": "2.11.0",
    "transformers": "5.13.1",
    "tokenizers": "0.22.2",
    "accelerate": "1.14.0",
    "huggingface_hub": "1.22.0",
    "sentencepiece": "0.2.2",
    "sacremoses": "0.1.1",
}

# Hugging Face dataset export rewrites ``source_id`` and ``source_revision``.
# Review identity therefore uses the same consumed fields as the translation
# publication contract; all rows are covered, not only the sampled records.
FULL_PAIRED_ROW_FIELDS: Final = PUBLICATION_CANONICAL_FIELDS
RUBRIC_FIELDS: Final = (
    "action_tool_intent",
    "omissions_additions",
    "polarity",
    "quantities",
    "entity_relations",
)
REQUIRED_RUBRIC_FIELDS: Final = frozenset({"action_tool_intent", "omissions_additions"})
RUBRIC_VALUES: Final = frozenset({"pass", "fail", "not_applicable"})

NON_NATIVE_REVIEWER_BOUNDARY: Final = (
    "A non-native Hebrew reviewer compares the English source, Hebrew translation, "
    "and independently generated English backtranslation. This gate can detect "
    "material semantic regressions but is not native-speaker linguistic validation."
)

# These verbs are preregistered because a literal translation can plausibly
# preserve surface form while changing the API action.  The list is fixed in
# the schema implementation; it is never adapted after looking at judgments.
AMBIGUOUS_HIGH_RISK_ACTION_VERBS: Final = (
    "book",
    "cancel",
    "charge",
    "close",
    "delete",
    "deposit",
    "draw",
    "issue",
    "open",
    "order",
    "pay",
    "post",
    "refund",
    "remove",
    "reserve",
    "return",
    "schedule",
    "send",
    "set",
    "ship",
    "transfer",
    "withdraw",
)
_HIGH_RISK_PATTERN: Final = re.compile(
    r"\b(?:" + "|".join(map(re.escape, AMBIGUOUS_HIGH_RISK_ACTION_VERBS)) + r")\b",
    flags=re.IGNORECASE,
)

RubricValue = Literal["pass", "fail", "not_applicable"]


class BackTranslationModel(Protocol):
    """Small interface used by both Transformers and deterministic tests."""

    def translate_batch(self, texts: list[str]) -> list[str]: ...


@dataclass(frozen=True)
class BackTranslatorInfo:
    """Immutable identity and greedy decoding contract for backtranslation."""

    model_id: str = BACK_TRANSLATOR_MODEL_ID
    model_revision: str = BACK_TRANSLATOR_MODEL_REVISION
    max_new_tokens: int = BACK_TRANSLATOR_MAX_NEW_TOKENS
    max_source_tokens: int = BACK_TRANSLATOR_MAX_SOURCE_TOKENS
    batch_size: int = BACK_TRANSLATOR_BATCH_SIZE


@dataclass(frozen=True)
class SemanticReviewProducerProvenance:
    """Clean-code and exact-runtime identity for machine review inputs."""

    code_revision: str
    working_tree_clean: bool
    execution_boundary: Literal["modal_gpu", "local"]
    provider: str
    hardware: str
    allocation_timeout_seconds: int | None
    package_versions: Mapping[str, str]


@dataclass(frozen=True)
class ReviewCandidate:
    paired_row: RawToolCallRow
    source_row: RawToolCallRow
    root_split: SplitName
    source_query_length_decile: int
    protected_span_count: int
    tool_action_family: str
    high_risk_action_verbs: tuple[str, ...]

    @property
    def sample_id(self) -> str:
        return self.source_example_id

    @property
    def source_example_id(self) -> str:
        value = self.paired_row.get("source_example_id")
        if value is None:  # guarded while candidates are constructed
            raise AssertionError("paired review candidate lacks source_example_id")
        return value


def root_split_assignments(
    config: SommelierConfig,
    root_rows: Sequence[RawToolCallRow],
) -> dict[str, SplitName]:
    """Recreate the root split assignment used by translation and training."""
    result = prepare_split_result(
        list(root_rows),
        min_query_chars=config.data.min_query_chars,
        max_query_chars=config.data.max_query_chars,
        n_train=config.data.n_train,
        n_validation=config.data.n_validation,
        n_test=config.data.n_test,
        seed=config.project.seed,
        language=config.root_dataset.language,
    )
    return {example["example_id"]: example["split"] for example in all_examples(result)}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _full_row_payload(row: RawToolCallRow) -> dict[str, object]:
    missing = [field for field in FULL_PAIRED_ROW_FIELDS if field not in row]
    if missing:
        raise UserInputError(
            f"semantic-review paired row is missing {', '.join(missing)}",
            hint="Review only canonical accepted paired rows with source_example_id.",
        )
    row_payload = cast(dict[str, object], dict(row))
    return {field: row_payload[field] for field in FULL_PAIRED_ROW_FIELDS}


def full_paired_rows_canonical_identity(rows_path: Path) -> tuple[int, str]:
    """Digest every canonical raw-row field for the entire accepted corpus."""
    rows = load_raw_rows(rows_path, require_source_example_id=True)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_bytes(_full_row_payload(row)))
        digest.update(b"\n")
    return len(rows), digest.hexdigest()


def _full_row_sha256(row: RawToolCallRow) -> str:
    return _sha256_json(_full_row_payload(row))


def _source_row_sha256(row: RawToolCallRow) -> str:
    return _sha256_json(dict(row))


def _load_translation_summary(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise UserInputError(
            "semantic review requires a valid translation summary",
            hint="Use translation_summary.json from the exact accepted paired rows.",
        ) from error
    if not isinstance(payload, dict):
        raise UserInputError("translation summary must be a JSON object")
    return cast(dict[str, object], payload)


def _forward_translator(summary: Mapping[str, object]) -> dict[str, object]:
    translator = summary.get("translator")
    if not isinstance(translator, dict):
        raise UserInputError(
            "translation summary is missing forward-translator provenance",
            hint="Regenerate the translation summary before semantic review.",
        )
    model_id = translator.get("model_id")
    model_revision = translator.get("model_revision")
    if not isinstance(model_id, str) or not model_id:
        raise UserInputError("translation summary has no forward translator model id")
    if not isinstance(model_revision, str) or not model_revision:
        raise UserInputError("translation summary has no forward translator revision")
    return {"model_id": model_id, "model_revision": model_revision}


def capture_producer_provenance(
    *,
    code_revision: str,
    working_tree_clean: bool | None,
    execution_boundary: Literal["modal_gpu", "local"],
    provider: str,
    hardware: str,
    allocation_timeout_seconds: int | None = None,
) -> SemanticReviewProducerProvenance:
    """Capture the fixed runtime versions used to mint machine evidence."""
    packages = {"python": platform.python_version()}
    for package in (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
        "sacremoses",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "absent"
    return SemanticReviewProducerProvenance(
        code_revision=code_revision,
        working_tree_clean=working_tree_clean is True,
        execution_boundary=execution_boundary,
        provider=provider,
        hardware=hardware,
        allocation_timeout_seconds=allocation_timeout_seconds,
        package_versions=packages,
    )


def _producer_payload(
    provenance: SemanticReviewProducerProvenance,
) -> dict[str, object]:
    return {
        "code_revision": provenance.code_revision,
        "working_tree_clean": provenance.working_tree_clean,
        "runtime": {
            "execution_boundary": provenance.execution_boundary,
            "provider": provenance.provider,
            "hardware": provenance.hardware,
            "allocation_timeout_seconds": provenance.allocation_timeout_seconds,
        },
        "package_versions": dict(sorted(provenance.package_versions.items())),
    }


def validate_producer_provenance(
    provenance: SemanticReviewProducerProvenance,
    *,
    translation_summary: Mapping[str, object],
) -> None:
    """Reject dirty, mutable, or substituted semantic-review runtimes."""
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", provenance.code_revision) is None:
        raise UserInputError(
            "semantic-review producer code revision is mutable or invalid",
            hint="Run from an exact committed Git SHA.",
        )
    if not provenance.working_tree_clean:
        raise UserInputError(
            "semantic-review producer worktree is dirty",
            hint="Commit all gate and runtime changes before producing evidence.",
        )
    source_code = translation_summary.get("source_code")
    translation_revision = source_code.get("git_commit") if isinstance(source_code, dict) else None
    if provenance.code_revision != translation_revision:
        raise UserInputError(
            "semantic-review producer revision differs from the full translation revision",
            hint="Produce both artifacts from the same clean release commit.",
        )
    if provenance.package_versions != EXPECTED_PRODUCER_PACKAGE_VERSIONS:
        raise UserInputError(
            "semantic-review producer package versions differ from the pinned runtime",
            hint=(
                "Use exactly: "
                + ", ".join(
                    f"{name}=={package_version}"
                    for name, package_version in EXPECTED_PRODUCER_PACKAGE_VERSIONS.items()
                )
            ),
        )
    if provenance.execution_boundary == "modal_gpu":
        if (
            provenance.provider != "modal"
            or not provenance.hardware.strip()
            or not isinstance(provenance.allocation_timeout_seconds, int)
            or isinstance(provenance.allocation_timeout_seconds, bool)
            or provenance.allocation_timeout_seconds <= 0
        ):
            raise UserInputError(
                "remote semantic-review provenance lacks its Modal allocation identity"
            )
    elif provenance.execution_boundary == "local":
        if (
            provenance.provider != "local"
            or not provenance.hardware.strip()
            or provenance.allocation_timeout_seconds is not None
        ):
            raise UserInputError("local semantic-review provenance lacks its host/runtime identity")
    else:  # pragma: no cover - closed by the Literal in typed callers
        raise UserInputError("unsupported semantic-review execution boundary")


def _producer_from_payload(payload: Mapping[str, object]) -> SemanticReviewProducerProvenance:
    producer = payload.get("producer")
    if not isinstance(producer, dict):
        raise UserInputError("semantic-review template is missing producer provenance")
    runtime = producer.get("runtime")
    packages = producer.get("package_versions")
    code_revision = producer.get("code_revision")
    working_tree_clean = producer.get("working_tree_clean")
    if (
        not isinstance(runtime, dict)
        or not isinstance(packages, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in packages.items()
        )
        or not isinstance(code_revision, str)
        or not isinstance(working_tree_clean, bool)
    ):
        raise UserInputError("semantic-review producer provenance has invalid fields")
    boundary = runtime.get("execution_boundary")
    provider = runtime.get("provider")
    hardware = runtime.get("hardware")
    allocation_timeout_seconds = runtime.get("allocation_timeout_seconds")
    if boundary not in {"modal_gpu", "local"}:
        raise UserInputError("semantic-review producer execution boundary is invalid")
    if (
        not isinstance(provider, str)
        or not isinstance(hardware, str)
        or (allocation_timeout_seconds is not None and type(allocation_timeout_seconds) is not int)
    ):
        raise UserInputError("semantic-review producer runtime identity is invalid")
    return SemanticReviewProducerProvenance(
        code_revision=code_revision,
        working_tree_clean=working_tree_clean,
        execution_boundary=cast(Literal["modal_gpu", "local"], boundary),
        provider=provider,
        hardware=hardware,
        allocation_timeout_seconds=allocation_timeout_seconds,
        package_versions=cast(dict[str, str], packages),
    )


def _backtranslation_backend() -> dict[str, object]:
    return {
        "schema_version": BACKTRANSLATION_BACKEND_SCHEMA,
        "framework": "transformers",
        "model_loader": "AutoModelForSeq2SeqLM",
        "tokenizer_loader": "AutoTokenizer",
        "model_type": "marian",
        "dtype": BACK_TRANSLATOR_DTYPE,
        "device_map": BACK_TRANSLATOR_DEVICE_MAP,
        "trust_remote_code": False,
        "hugging_face_environment": dict(sorted(BACK_TRANSLATOR_HF_ENV.items())),
    }


def _backtranslation_tokenization(info: BackTranslatorInfo) -> dict[str, object]:
    return {
        "add_special_tokens": True,
        "padding": "longest",
        "truncation": False,
        "max_source_tokens": info.max_source_tokens,
        "use_fast": False,
        "punctuation_normalizer": "sacremoses.MosesPunctNormalizer",
    }


def _backtranslation_decoding(info: BackTranslatorInfo) -> dict[str, object]:
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": info.max_new_tokens,
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }


def validate_back_translator_info(
    info: BackTranslatorInfo,
    *,
    forward_model_id: str,
) -> None:
    """Reject mutable, substituted, or same-model backtranslation evidence."""
    if info.model_id != BACK_TRANSLATOR_MODEL_ID:
        raise UserInputError(
            "semantic review must use the preregistered back-translator "
            f"{BACK_TRANSLATOR_MODEL_ID}",
            hint="Do not choose a model after inspecting translation outputs.",
        )
    if info.model_revision != BACK_TRANSLATOR_MODEL_REVISION:
        raise UserInputError(
            "semantic review back-translator revision is mutable or not preregistered",
            hint=f"Use the exact commit {BACK_TRANSLATOR_MODEL_REVISION}.",
        )
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", info.model_revision) is None:
        raise UserInputError(
            "semantic review back-translator revision is not immutable",
            hint="Use an exact Hugging Face commit SHA, never main or a tag.",
        )
    if info.model_id.casefold() == forward_model_id.casefold():
        raise UserInputError(
            "semantic review back-translator must differ from the forward translator",
            hint="Use the preregistered independent Helsinki-NLP OPUS-MT checkpoint.",
        )
    if (
        info.max_new_tokens != BACK_TRANSLATOR_MAX_NEW_TOKENS
        or info.max_source_tokens != BACK_TRANSLATOR_MAX_SOURCE_TOKENS
        or info.batch_size != BACK_TRANSLATOR_BATCH_SIZE
    ):
        raise UserInputError(
            "semantic review decoding differs from the preregistered contract",
            hint="Use the default BackTranslatorInfo without tuning on review outputs.",
        )


def _high_risk_verbs(query: str) -> tuple[str, ...]:
    return tuple(
        sorted({match.group(0).casefold() for match in _HIGH_RISK_PATTERN.finditer(query)})
    )


def _single_gold_call(row: RawToolCallRow) -> ToolCall:
    parsed = parse_gold_calls(row["answers"])
    if isinstance(parsed, str):
        raise UserInputError(
            f"semantic-review row {row['source_id']!r} violates the single-call contract: {parsed}",
            hint="Review only the accepted single-call survivor set.",
        )
    return parsed[0]


def _review_candidates(
    root_rows: Sequence[RawToolCallRow],
    paired_rows: Sequence[RawToolCallRow],
    root_split_by_id: Mapping[str, SplitName],
) -> list[ReviewCandidate]:
    root_by_id = {row["source_id"]: row for row in root_rows}
    if len(root_by_id) != len(root_rows):
        raise UserInputError("semantic review root rows contain duplicate source_id values")

    joined: list[tuple[RawToolCallRow, RawToolCallRow, SplitName]] = []
    paired_ids: set[str] = set()
    source_example_ids: set[str] = set()
    for paired in paired_rows:
        paired_id = paired["source_id"]
        source_example_id = paired.get("source_example_id")
        if paired_id in paired_ids:
            raise UserInputError("semantic review paired rows contain duplicate source_id values")
        paired_ids.add(paired_id)
        if source_example_id is None or source_example_id not in root_by_id:
            raise UserInputError(
                f"semantic-review paired row {paired_id!r} has no matching root row"
            )
        if source_example_id in source_example_ids:
            raise UserInputError(
                "semantic review paired rows contain duplicate source_example_id values"
            )
        source_example_ids.add(source_example_id)
        split = root_split_by_id.get(source_example_id)
        if split is None:
            raise UserInputError(
                f"semantic-review paired row {paired_id!r} lacks a root split assignment",
                hint="Build the map from the same full config and exported root rows.",
            )
        source = root_by_id[source_example_id]
        if paired["tools"] != source["tools"] or paired["answers"] != source["answers"]:
            raise UserInputError(
                f"semantic-review pair {paired_id!r} changed tools or gold answers",
                hint="Use the exact accepted paired survivor set.",
            )
        joined.append((paired, source, split))

    if len(joined) < SEMANTIC_REVIEW_SAMPLE_SIZE:
        raise UserInputError(
            f"semantic review needs at least {SEMANTIC_REVIEW_SAMPLE_SIZE} accepted pairs, "
            f"got {len(joined)}",
            hint="A full publication cannot use a smaller review sample.",
        )

    ordered_by_length = sorted(
        joined,
        key=lambda item: (len(item[1]["query"]), item[1]["source_id"]),
    )
    length_decile_by_id = {
        paired["source_id"]: min(10, (index * 10) // len(ordered_by_length) + 1)
        for index, (paired, _source, _split) in enumerate(ordered_by_length)
    }

    candidates: list[ReviewCandidate] = []
    for paired, source, split in joined:
        gold = _single_gold_call(source)
        spans = protected_spans(source["query"], [gold])
        candidates.append(
            ReviewCandidate(
                paired_row=paired,
                source_row=source,
                root_split=split,
                source_query_length_decile=length_decile_by_id[paired["source_id"]],
                protected_span_count=len(spans),
                tool_action_family=gold["name"],
                high_risk_action_verbs=_high_risk_verbs(source["query"]),
            )
        )
    return candidates


def _candidate_dimension(candidate: ReviewCandidate, dimension: str) -> str:
    values = {
        "root_split": candidate.root_split,
        "source_query_length_decile": str(candidate.source_query_length_decile),
        "protected_span_count": str(candidate.protected_span_count),
        "tool_action_family": candidate.tool_action_family,
    }
    return values[dimension]


def _stable_order(seed: int, sample_id: str, *, namespace: str) -> str:
    value = f"{SEMANTIC_SELECTION_SCHEMA}\0{namespace}\0{seed}\0{sample_id}"
    return hashlib.sha256(value.encode()).hexdigest()


def _balanced_select(
    candidates: Sequence[ReviewCandidate],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> list[ReviewCandidate]:
    """Greedily balance preregistered marginal strata with stable tie breaks."""
    if count > len(candidates):
        raise AssertionError("balanced selection cannot exceed its pool")
    if count == len(candidates):
        return list(candidates)
    dimensions = (
        "root_split",
        "source_query_length_decile",
        "protected_span_count",
        "tool_action_family",
    )
    populations: dict[str, Counter[str]] = {
        dimension: Counter(_candidate_dimension(candidate, dimension) for candidate in candidates)
        for dimension in dimensions
    }
    targets: dict[str, dict[str, float]] = {
        dimension: {
            value: count * population / len(candidates)
            for value, population in populations[dimension].items()
        }
        for dimension in dimensions
    }
    selected_counts: dict[str, Counter[str]] = {dimension: Counter() for dimension in dimensions}
    remaining = list(candidates)
    selected: list[ReviewCandidate] = []
    for _ in range(count):

        def score(candidate: ReviewCandidate) -> tuple[float, str]:
            deficit_score = 0.0
            for dimension in dimensions:
                value = _candidate_dimension(candidate, dimension)
                target = targets[dimension][value]
                deficit = max(target - selected_counts[dimension][value], 0.0)
                deficit_score += deficit / max(target, 1.0)
            # max() is used below.  Invert the hexadecimal tie-break so the
            # lexicographically smallest stable hash wins.
            tie = _stable_order(seed, candidate.sample_id, namespace=namespace)
            inverted = "".join(f"{15 - int(char, 16):x}" for char in tie)
            return deficit_score, inverted

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
        for dimension in dimensions:
            selected_counts[dimension][_candidate_dimension(chosen, dimension)] += 1
    return selected


def _risk_quotas(high_risk_count: int, ordinary_count: int) -> tuple[int, int]:
    high_quota = min(HIGH_RISK_SAMPLE_QUOTA, high_risk_count)
    ordinary_quota = SEMANTIC_REVIEW_SAMPLE_SIZE - high_quota
    if ordinary_count < ordinary_quota:
        high_quota += ordinary_quota - ordinary_count
        ordinary_quota = ordinary_count
    return high_quota, ordinary_quota


def select_semantic_review_sample(
    candidates: Sequence[ReviewCandidate],
    *,
    seed: int,
) -> list[ReviewCandidate]:
    """Select the fixed 200-row, action-risk-oversampled review cohort."""
    if len(candidates) < SEMANTIC_REVIEW_SAMPLE_SIZE:
        raise UserInputError(
            f"semantic review requires {SEMANTIC_REVIEW_SAMPLE_SIZE} accepted pairs"
        )
    high_risk = [candidate for candidate in candidates if candidate.high_risk_action_verbs]
    ordinary = [candidate for candidate in candidates if not candidate.high_risk_action_verbs]
    high_quota, ordinary_quota = _risk_quotas(len(high_risk), len(ordinary))
    high_selected = _balanced_select(
        high_risk,
        high_quota,
        seed=seed,
        namespace="high-risk",
    )
    ordinary_selected = _balanced_select(
        ordinary,
        ordinary_quota,
        seed=seed,
        namespace="ordinary",
    )
    selected = [*high_selected, *ordinary_selected]
    return sorted(
        selected,
        key=lambda candidate: _stable_order(
            seed,
            candidate.sample_id,
            namespace="ordered-sample",
        ),
    )


def _strata_counts(candidates: Sequence[ReviewCandidate]) -> dict[str, dict[str, int]]:
    dimensions = (
        "root_split",
        "source_query_length_decile",
        "protected_span_count",
        "tool_action_family",
    )
    result = {
        dimension: dict(
            sorted(
                Counter(
                    _candidate_dimension(candidate, dimension) for candidate in candidates
                ).items()
            )
        )
        for dimension in dimensions
    }
    result["ambiguous_high_risk_action_verb"] = {
        "false": sum(not candidate.high_risk_action_verbs for candidate in candidates),
        "true": sum(bool(candidate.high_risk_action_verbs) for candidate in candidates),
    }
    return result


def _backtranslation_request_sha256(text: str, info: BackTranslatorInfo) -> str:
    return _sha256_json(
        {
            "schema_version": BACKTRANSLATION_REQUEST_SCHEMA,
            "model_id": info.model_id,
            "model_revision": info.model_revision,
            "backend": _backtranslation_backend(),
            "tokenization": _backtranslation_tokenization(info),
            "decoding": _backtranslation_decoding(info),
            "source_language": "he",
            "target_language": "en",
            "input": text,
        }
    )


def _locked_record_payload(record: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "sample_id",
        "source_example_id",
        "paired_row_sha256",
        "source_row_sha256",
        "source_query",
        "hebrew_query",
        "backtranslation_request_sha256",
        "english_backtranslation",
        "english_backtranslation_sha256",
        "strata",
    )
    return {field: record.get(field) for field in fields}


def _ordered_ids_sha256(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def create_semantic_review_template(
    *,
    root_rows_path: Path,
    paired_rows_path: Path,
    translation_summary_path: Path,
    root_split_by_id: Mapping[str, SplitName],
    output_path: Path,
    backtranslator: BackTranslationModel,
    seed: int,
    producer_provenance: SemanticReviewProducerProvenance,
    back_translator_info: BackTranslatorInfo = BackTranslatorInfo(),
) -> Path:
    """Create and lock the 200-row review input before human judgments."""
    summary = _load_translation_summary(translation_summary_path)
    if summary.get("language") != "he":
        raise UserInputError("the Hebrew semantic gate requires a Hebrew translation summary")
    selection = summary.get("selection")
    if not isinstance(selection, dict) or selection.get("mode") != "full":
        raise UserInputError(
            "semantic review release evidence must come from a full translation run",
            hint="Smoke translations may omit this publication gate.",
        )
    forward = _forward_translator(summary)
    validate_producer_provenance(
        producer_provenance,
        translation_summary=summary,
    )
    validate_back_translator_info(
        back_translator_info,
        forward_model_id=cast(str, forward["model_id"]),
    )
    root_rows = load_raw_rows(root_rows_path)
    paired_rows = load_raw_rows(paired_rows_path, require_source_example_id=True)
    candidates = _review_candidates(root_rows, paired_rows, root_split_by_id)
    selected = select_semantic_review_sample(candidates, seed=seed)
    hebrew_queries = [candidate.paired_row["query"] for candidate in selected]
    outputs: list[str] = []
    for offset in range(0, len(hebrew_queries), back_translator_info.batch_size):
        batch = hebrew_queries[offset : offset + back_translator_info.batch_size]
        translated = backtranslator.translate_batch(batch)
        if len(translated) != len(batch):
            raise UserInputError(
                "back-translator returned the wrong number of outputs",
                hint="Do not publish partial or positionally misaligned backtranslations.",
            )
        outputs.extend(output.strip() for output in translated)
    if len(outputs) != SEMANTIC_REVIEW_SAMPLE_SIZE or any(not output for output in outputs):
        raise UserInputError(
            "semantic review requires one nonempty backtranslation for every sampled row"
        )

    records: list[dict[str, object]] = []
    for candidate, backtranslation in zip(selected, outputs, strict=True):
        record: dict[str, object] = {
            "sample_id": candidate.sample_id,
            "source_example_id": candidate.source_example_id,
            "paired_row_sha256": _full_row_sha256(candidate.paired_row),
            "source_row_sha256": _source_row_sha256(candidate.source_row),
            "source_query": candidate.source_row["query"],
            "hebrew_query": candidate.paired_row["query"],
            "backtranslation_request_sha256": _backtranslation_request_sha256(
                candidate.paired_row["query"], back_translator_info
            ),
            "english_backtranslation": backtranslation,
            "english_backtranslation_sha256": hashlib.sha256(
                backtranslation.encode("utf-8")
            ).hexdigest(),
            "strata": {
                "root_split": candidate.root_split,
                "source_query_length_decile": candidate.source_query_length_decile,
                "protected_span_count": candidate.protected_span_count,
                "tool_action_family": candidate.tool_action_family,
                "ambiguous_high_risk_action_verb": bool(candidate.high_risk_action_verbs),
                "matched_high_risk_action_verbs": list(candidate.high_risk_action_verbs),
            },
            "review": {
                "rubric": {field: None for field in RUBRIC_FIELDS},
                "critical_error": None,
                "passes_review": None,
                "notes": "",
            },
        }
        record["locked_review_input_sha256"] = _sha256_json(_locked_record_payload(record))
        records.append(record)

    sample_ids = [candidate.sample_id for candidate in selected]
    full_rows, full_rows_sha256 = full_paired_rows_canonical_identity(paired_rows_path)
    high_risk_population = sum(bool(candidate.high_risk_action_verbs) for candidate in candidates)
    high_risk_quota, _ordinary_quota = _risk_quotas(
        high_risk_population,
        len(candidates) - high_risk_population,
    )
    locked_sample_sha256 = _sha256_json([_locked_record_payload(record) for record in records])
    payload: dict[str, object] = {
        "schema_version": SEMANTIC_REVIEW_TEMPLATE_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "language": "he",
        "translation_summary_sha256": sha256_file(translation_summary_path),
        "root_rows_sha256": sha256_file(root_rows_path),
        "paired_rows": {
            "rows": full_rows,
            "canonical_fields": list(FULL_PAIRED_ROW_FIELDS),
            "canonical_sha256": full_rows_sha256,
        },
        "forward_translator": forward,
        "back_translator": {
            "model_id": back_translator_info.model_id,
            "model_revision": back_translator_info.model_revision,
            "license": BACK_TRANSLATOR_LICENSE,
            "attribution": BACK_TRANSLATOR_ATTRIBUTION,
            "model_card": BACK_TRANSLATOR_MODEL_CARD_URL,
            "source_language": "he",
            "target_language": "en",
            "request_schema": BACKTRANSLATION_REQUEST_SCHEMA,
            "backend": _backtranslation_backend(),
            "tokenization": _backtranslation_tokenization(back_translator_info),
            "decoding": _backtranslation_decoding(back_translator_info),
            "batch_size": back_translator_info.batch_size,
        },
        "selection": {
            "schema_version": SEMANTIC_SELECTION_SCHEMA,
            "selected_before_judgments": True,
            "sample_size": SEMANTIC_REVIEW_SAMPLE_SIZE,
            "seed": seed,
            "algorithm": "greedy_marginal_balance_with_fixed_high_risk_quota",
            "stratification_dimensions": [
                "root_split",
                "source_query_length_decile",
                "protected_span_count",
                "tool_action_family",
                "ambiguous_high_risk_action_verb",
            ],
            "high_risk_action_verbs": list(AMBIGUOUS_HIGH_RISK_ACTION_VERBS),
            "high_risk_quota": high_risk_quota,
            "population_strata": _strata_counts(candidates),
            "sample_strata": _strata_counts(selected),
            "ordered_sample_ids": sample_ids,
            "ordered_sample_ids_sha256": _ordered_ids_sha256(sample_ids),
            "locked_sample_sha256": locked_sample_sha256,
        },
        "reviewer": {
            "reviewer_id": "unassigned",
            "native_hebrew_reviewer": False,
            "boundary": NON_NATIVE_REVIEWER_BOUNDARY,
        },
        "records": records,
        "gate": {
            "status": "pending",
            "complete_decisions": 0,
            "critical_errors": None,
            "review_decisions_sha256": None,
            "failure_scope": "whole_publication",
            "row_cherry_picking_allowed": False,
        },
        "producer": _producer_payload(producer_provenance),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _load_review(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise UserInputError(
            "translation semantic review is missing or invalid",
            hint=f"Publish a finalized {SEMANTIC_REVIEW_FILENAME}.",
        ) from error
    if not isinstance(payload, dict):
        raise UserInputError("translation semantic review must be a JSON object")
    return cast(dict[str, object], payload)


def _review_records(payload: Mapping[str, object]) -> list[dict[str, object]]:
    records = payload.get("records")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise UserInputError("semantic review records must be a list of objects")
    return cast(list[dict[str, object]], records)


def _machine_locked_artifact_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Return every machine-produced field, excluding human-owned decisions."""
    excluded = {
        "schema_version",
        "finalized_at",
        "machine_template",
        "reviewer",
        "records",
        "gate",
    }
    top_level = {key: value for key, value in payload.items() if key not in excluded}
    reviewer = payload.get("reviewer")
    reviewer_boundary = (
        {
            "native_hebrew_reviewer": reviewer.get("native_hebrew_reviewer"),
            "boundary": reviewer.get("boundary"),
        }
        if isinstance(reviewer, dict)
        else None
    )
    records = [
        {
            **_locked_record_payload(record),
            "locked_review_input_sha256": record.get("locked_review_input_sha256"),
        }
        for record in _review_records(payload)
    ]
    return {
        "top_level": top_level,
        "reviewer_boundary": reviewer_boundary,
        "records": records,
    }


def _validate_review_decisions(
    records: Sequence[Mapping[str, object]],
) -> tuple[int, int, str]:
    decisions: list[dict[str, object]] = []
    critical_errors = 0
    for index, record in enumerate(records, start=1):
        review = record.get("review")
        if not isinstance(review, dict):
            raise UserInputError(f"semantic review record {index} has no reviewer decision")
        rubric = review.get("rubric")
        if not isinstance(rubric, dict) or set(rubric) != set(RUBRIC_FIELDS):
            raise UserInputError(f"semantic review record {index} has incomplete rubric fields")
        values: dict[str, str] = {}
        for field in RUBRIC_FIELDS:
            value = rubric.get(field)
            if not isinstance(value, str) or value not in RUBRIC_VALUES:
                raise UserInputError(
                    f"semantic review record {index} rubric {field!r} is incomplete"
                )
            if field in REQUIRED_RUBRIC_FIELDS and value == "not_applicable":
                raise UserInputError(
                    f"semantic review record {index} rubric {field!r} cannot be not_applicable"
                )
            values[field] = value
        critical = review.get("critical_error")
        passes = review.get("passes_review")
        notes = review.get("notes")
        if not isinstance(critical, bool) or not isinstance(passes, bool):
            raise UserInputError(
                f"semantic review record {index} lacks a complete critical/pass decision"
            )
        if not isinstance(notes, str):
            raise UserInputError(f"semantic review record {index} notes must be text")
        any_failure = any(value == "fail" for value in values.values())
        if critical != any_failure or passes != (not critical):
            raise UserInputError(
                f"semantic review record {index} decision contradicts its rubric",
                hint="Any rubric failure is critical and fails the row.",
            )
        if critical and not notes.strip():
            raise UserInputError(
                f"semantic review record {index} critical error requires reviewer notes"
            )
        critical_errors += int(critical)
        decisions.append(
            {
                "sample_id": record.get("sample_id"),
                "rubric": values,
                "critical_error": critical,
                "passes_review": passes,
                "notes": notes,
            }
        )
    return len(decisions), critical_errors, _sha256_json(decisions)


def _validate_pristine_template(payload: Mapping[str, object]) -> None:
    reviewer = payload.get("reviewer")
    if reviewer != {
        "reviewer_id": "unassigned",
        "native_hebrew_reviewer": False,
        "boundary": NON_NATIVE_REVIEWER_BOUNDARY,
    }:
        raise UserInputError("machine semantic-review template has been assigned a reviewer")
    expected_review = {
        "rubric": {field: None for field in RUBRIC_FIELDS},
        "critical_error": None,
        "passes_review": None,
        "notes": "",
    }
    if any(record.get("review") != expected_review for record in _review_records(payload)):
        raise UserInputError(
            "machine semantic-review template contains human decisions",
            hint="Keep the producer template untouched and review a separate copy.",
        )
    if payload.get("gate") != {
        "status": "pending",
        "complete_decisions": 0,
        "critical_errors": None,
        "review_decisions_sha256": None,
        "failure_scope": "whole_publication",
        "row_cherry_picking_allowed": False,
    }:
        raise UserInputError("machine semantic-review template gate is not pristine")


def validate_semantic_review(
    review_path: Path,
    *,
    root_rows_path: Path,
    paired_rows_path: Path,
    translation_summary_path: Path,
    root_split_by_id: Mapping[str, SplitName],
    expected_seed: int,
    require_passed: bool = True,
    template_path: Path | None = None,
    require_pristine_template: bool = True,
) -> dict[str, object]:
    """Validate sample provenance, locked inputs, and optionally the release gate."""
    payload = _load_review(review_path)
    schema_version = payload.get("schema_version")
    if (
        schema_version
        not in {
            SEMANTIC_REVIEW_SCHEMA,
            SEMANTIC_REVIEW_TEMPLATE_SCHEMA,
        }
        or payload.get("language") != "he"
    ):
        raise UserInputError("translation semantic review has the wrong schema or language")
    if require_passed and schema_version != SEMANTIC_REVIEW_SCHEMA:
        raise UserInputError(
            "publication requires a finalized semantic review, not its machine template"
        )
    if schema_version == SEMANTIC_REVIEW_SCHEMA:
        if template_path is None:
            raise UserInputError(
                "final semantic review validation requires the immutable machine template"
            )
        template_payload = _load_review(template_path)
        if template_payload.get("schema_version") != SEMANTIC_REVIEW_TEMPLATE_SCHEMA:
            raise UserInputError("semantic review machine template has the wrong schema")
        _validate_pristine_template(template_payload)
        machine_template = payload.get("machine_template")
        if machine_template != {
            "filename": SEMANTIC_REVIEW_TEMPLATE_FILENAME,
            "schema_version": SEMANTIC_REVIEW_TEMPLATE_SCHEMA,
            "sha256": sha256_file(template_path),
        }:
            raise UserInputError(
                "final semantic review does not bind the immutable machine template"
            )
        if _machine_locked_artifact_payload(payload) != _machine_locked_artifact_payload(
            template_payload
        ):
            raise UserInputError(
                "final semantic review changed machine-locked sample or backtranslation fields"
            )
    if payload.get("translation_summary_sha256") != sha256_file(translation_summary_path):
        raise UserInputError("semantic review does not bind the current translation summary")
    if payload.get("root_rows_sha256") != sha256_file(root_rows_path):
        raise UserInputError("semantic review source rows were changed after sampling")

    full_rows, full_sha256 = full_paired_rows_canonical_identity(paired_rows_path)
    paired_identity = payload.get("paired_rows")
    if not isinstance(paired_identity, dict) or paired_identity != {
        "rows": full_rows,
        "canonical_fields": list(FULL_PAIRED_ROW_FIELDS),
        "canonical_sha256": full_sha256,
    }:
        raise UserInputError(
            "semantic review does not bind the full canonical paired-row digest",
            hint="Never drop a failed reviewed row and resample from a modified dataset.",
        )

    summary = _load_translation_summary(translation_summary_path)
    forward = _forward_translator(summary)
    producer = _producer_from_payload(payload)
    validate_producer_provenance(producer, translation_summary=summary)
    if payload.get("producer") != _producer_payload(producer):
        raise UserInputError("semantic-review producer provenance is not canonical")
    if payload.get("forward_translator") != forward:
        raise UserInputError("semantic review forward-translator identity was changed")
    info = BackTranslatorInfo()
    validate_back_translator_info(info, forward_model_id=cast(str, forward["model_id"]))
    expected_back_translator = {
        "model_id": info.model_id,
        "model_revision": info.model_revision,
        "license": BACK_TRANSLATOR_LICENSE,
        "attribution": BACK_TRANSLATOR_ATTRIBUTION,
        "model_card": BACK_TRANSLATOR_MODEL_CARD_URL,
        "source_language": "he",
        "target_language": "en",
        "request_schema": BACKTRANSLATION_REQUEST_SCHEMA,
        "backend": _backtranslation_backend(),
        "tokenization": _backtranslation_tokenization(info),
        "decoding": _backtranslation_decoding(info),
        "batch_size": info.batch_size,
    }
    if payload.get("back_translator") != expected_back_translator:
        raise UserInputError(
            "semantic review back-translator identity or decoding was changed",
            hint=(
                "Use the different pinned Helsinki-NLP OPUS-MT model and "
                "preregistered greedy decoding."
            ),
        )

    root_rows = load_raw_rows(root_rows_path)
    paired_rows = load_raw_rows(paired_rows_path, require_source_example_id=True)
    candidates = _review_candidates(root_rows, paired_rows, root_split_by_id)
    expected_sample = select_semantic_review_sample(candidates, seed=expected_seed)
    expected_ids = [candidate.sample_id for candidate in expected_sample]
    high_risk_population = sum(bool(candidate.high_risk_action_verbs) for candidate in candidates)
    expected_high_risk_quota, _ordinary_quota = _risk_quotas(
        high_risk_population,
        len(candidates) - high_risk_population,
    )
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise UserInputError("semantic review is missing its preregistered selection")
    if (
        selection.get("schema_version") != SEMANTIC_SELECTION_SCHEMA
        or selection.get("selected_before_judgments") is not True
        or selection.get("sample_size") != SEMANTIC_REVIEW_SAMPLE_SIZE
        or selection.get("seed") != expected_seed
        or selection.get("algorithm") != "greedy_marginal_balance_with_fixed_high_risk_quota"
        or selection.get("stratification_dimensions")
        != [
            "root_split",
            "source_query_length_decile",
            "protected_span_count",
            "tool_action_family",
            "ambiguous_high_risk_action_verb",
        ]
        or selection.get("high_risk_action_verbs") != list(AMBIGUOUS_HIGH_RISK_ACTION_VERBS)
        or selection.get("high_risk_quota") != expected_high_risk_quota
        or selection.get("population_strata") != _strata_counts(candidates)
        or selection.get("sample_strata") != _strata_counts(expected_sample)
        or selection.get("ordered_sample_ids") != expected_ids
        or selection.get("ordered_sample_ids_sha256") != _ordered_ids_sha256(expected_ids)
    ):
        raise UserInputError(
            "semantic review sample differs from the deterministic preregistration",
            hint="Do not replace, reorder, or resample rows after inspecting judgments.",
        )

    records = _review_records(payload)
    if len(records) != SEMANTIC_REVIEW_SAMPLE_SIZE:
        raise UserInputError(
            f"semantic review must contain exactly {SEMANTIC_REVIEW_SAMPLE_SIZE} records"
        )
    by_id = {candidate.sample_id: candidate for candidate in expected_sample}
    if [record.get("sample_id") for record in records] != expected_ids:
        raise UserInputError("semantic review record order does not match its sample IDs")
    for index, record in enumerate(records, start=1):
        candidate = by_id[expected_ids[index - 1]]
        expected_locked: dict[str, object] = {
            "sample_id": candidate.sample_id,
            "source_example_id": candidate.source_example_id,
            "paired_row_sha256": _full_row_sha256(candidate.paired_row),
            "source_row_sha256": _source_row_sha256(candidate.source_row),
            "source_query": candidate.source_row["query"],
            "hebrew_query": candidate.paired_row["query"],
            "backtranslation_request_sha256": _backtranslation_request_sha256(
                candidate.paired_row["query"], info
            ),
            "english_backtranslation": record.get("english_backtranslation"),
            "english_backtranslation_sha256": record.get("english_backtranslation_sha256"),
            "strata": {
                "root_split": candidate.root_split,
                "source_query_length_decile": candidate.source_query_length_decile,
                "protected_span_count": candidate.protected_span_count,
                "tool_action_family": candidate.tool_action_family,
                "ambiguous_high_risk_action_verb": bool(candidate.high_risk_action_verbs),
                "matched_high_risk_action_verbs": list(candidate.high_risk_action_verbs),
            },
        }
        backtranslation = record.get("english_backtranslation")
        if not isinstance(backtranslation, str) or not backtranslation.strip():
            raise UserInputError(f"semantic review record {index} has no backtranslation")
        expected_locked["english_backtranslation_sha256"] = hashlib.sha256(
            backtranslation.encode("utf-8")
        ).hexdigest()
        if _locked_record_payload(record) != expected_locked:
            raise UserInputError(f"semantic review record {index} has tampered locked inputs")
        if record.get("locked_review_input_sha256") != _sha256_json(expected_locked):
            raise UserInputError(f"semantic review record {index} input digest is invalid")
    locked_sample_sha256 = _sha256_json([_locked_record_payload(record) for record in records])
    if selection.get("locked_sample_sha256") != locked_sample_sha256:
        raise UserInputError("semantic review locked sample digest is invalid")

    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, dict):
        raise UserInputError("semantic review has no reviewer boundary")
    if (
        reviewer.get("native_hebrew_reviewer") is not False
        or reviewer.get("boundary") != NON_NATIVE_REVIEWER_BOUNDARY
    ):
        raise UserInputError(
            "semantic review must preserve the explicit non-native-reviewer boundary"
        )
    gate = payload.get("gate")
    if not isinstance(gate, dict):
        raise UserInputError("semantic review has no publication gate")
    if (
        gate.get("failure_scope") != "whole_publication"
        or gate.get("row_cherry_picking_allowed") is not False
    ):
        raise UserInputError("semantic review gate permits forbidden row cherry-picking")
    if schema_version == SEMANTIC_REVIEW_TEMPLATE_SCHEMA and require_pristine_template:
        _validate_pristine_template(payload)

    if require_passed or gate.get("status") == "passed":
        complete, critical_errors, decisions_sha256 = _validate_review_decisions(records)
        reviewer_id = reviewer.get("reviewer_id")
        if not isinstance(reviewer_id, str) or reviewer_id in {"", "unassigned"}:
            raise UserInputError("final semantic review requires a reviewer id")
        if (
            gate.get("status") != "passed"
            or gate.get("complete_decisions") != SEMANTIC_REVIEW_SAMPLE_SIZE
            or gate.get("complete_decisions") != complete
            or gate.get("critical_errors") != 0
            or critical_errors != 0
            or gate.get("review_decisions_sha256") != decisions_sha256
        ):
            raise UserInputError(
                "semantic review publication gate did not pass 200 complete decisions "
                "with zero critical errors",
                hint="Any sampled critical error fails the entire dataset publication.",
            )
    return payload


def finalize_semantic_review(
    input_path: Path,
    output_path: Path,
    *,
    template_path: Path,
    reviewer_id: str,
    root_rows_path: Path,
    paired_rows_path: Path,
    translation_summary_path: Path,
    root_split_by_id: Mapping[str, SplitName],
    expected_seed: int,
) -> Path:
    """Finalize reviewer-entered decisions without changing the locked sample."""
    _require_distinct_review_artifacts(template_path, input_path, output_path)
    if not reviewer_id.strip() or reviewer_id == "unassigned":
        raise UserInputError("semantic review finalization requires a stable reviewer id")
    template_payload = validate_semantic_review(
        template_path,
        root_rows_path=root_rows_path,
        paired_rows_path=paired_rows_path,
        translation_summary_path=translation_summary_path,
        root_split_by_id=root_split_by_id,
        expected_seed=expected_seed,
        require_passed=False,
    )
    reviewed_payload = validate_semantic_review(
        input_path,
        root_rows_path=root_rows_path,
        paired_rows_path=paired_rows_path,
        translation_summary_path=translation_summary_path,
        root_split_by_id=root_split_by_id,
        expected_seed=expected_seed,
        require_passed=False,
        require_pristine_template=False,
    )
    if reviewed_payload.get("schema_version") != SEMANTIC_REVIEW_TEMPLATE_SCHEMA:
        raise UserInputError("reviewer input must be a copy of the immutable machine template")
    if _machine_locked_artifact_payload(reviewed_payload) != _machine_locked_artifact_payload(
        template_payload
    ):
        raise UserInputError(
            "reviewer input changed machine-locked sample or backtranslation fields"
        )
    reviewed_records = _review_records(reviewed_payload)
    complete, critical_errors, decisions_sha256 = _validate_review_decisions(reviewed_records)
    if complete != SEMANTIC_REVIEW_SAMPLE_SIZE:
        raise UserInputError(
            f"semantic review requires exactly {SEMANTIC_REVIEW_SAMPLE_SIZE} complete decisions"
        )
    if critical_errors:
        raise UserInputError(
            f"semantic review found {critical_errors} critical error(s); "
            "the entire publication fails",
            hint="Fix and regenerate the translation dataset; never drop failed rows and resample.",
        )
    payload = cast(
        dict[str, object],
        json.loads(json.dumps(template_payload, ensure_ascii=False)),
    )
    payload["schema_version"] = SEMANTIC_REVIEW_SCHEMA
    payload["machine_template"] = {
        "filename": SEMANTIC_REVIEW_TEMPLATE_FILENAME,
        "schema_version": SEMANTIC_REVIEW_TEMPLATE_SCHEMA,
        "sha256": sha256_file(template_path),
    }
    records = _review_records(payload)
    for record, reviewed_record in zip(records, reviewed_records, strict=True):
        record["review"] = reviewed_record["review"]
    reviewer = cast(dict[str, object], payload["reviewer"])
    reviewer["reviewer_id"] = reviewer_id.strip()
    payload["finalized_at"] = datetime.now(UTC).isoformat()
    payload["gate"] = {
        "status": "passed",
        "complete_decisions": complete,
        "critical_errors": 0,
        "review_decisions_sha256": decisions_sha256,
        "failure_scope": "whole_publication",
        "row_cherry_picking_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    validate_semantic_review(
        output_path,
        root_rows_path=root_rows_path,
        paired_rows_path=paired_rows_path,
        translation_summary_path=translation_summary_path,
        root_split_by_id=root_split_by_id,
        expected_seed=expected_seed,
        require_passed=True,
        template_path=template_path,
    )
    return output_path


def _require_distinct_review_artifacts(
    template_path: Path,
    reviewed_path: Path,
    output_path: Path,
) -> None:
    paths = (template_path, reviewed_path, output_path)
    if len({path.resolve() for path in paths}) != len(paths):
        raise UserInputError(
            "semantic-review template, reviewed copy, and final output must be distinct files"
        )
    for index, left in enumerate(paths):
        if not left.exists():
            continue
        for right in paths[index + 1 :]:
            if right.exists() and left.samefile(right):
                raise UserInputError(
                    "semantic-review template, reviewed copy, and final output must not "
                    "alias the same file"
                )


def load_transformers_backtranslator(
    info: BackTranslatorInfo = BackTranslatorInfo(),
) -> BackTranslationModel:
    """Load the pinned Hebrew-to-English Marian backtranslator.

    The source is tokenized without truncation. Inputs above the model card's
    explicit 512-token tokenizer budget fail before generation, and callers
    cannot bypass the preregistered eight-row GPU batch bound.
    """
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional runtime boundary
        raise ExternalDependencyError(
            "semantic backtranslation requires torch, transformers, tokenizers, "
            "accelerate, huggingface-hub, sentencepiece, and sacremoses",
            hint="Use remote_semantic_review.py or install the model runtime dependencies.",
        ) from error

    validate_back_translator_info(info, forward_model_id="__loader_preflight__")
    for name, value in BACK_TRANSLATOR_HF_ENV.items():
        os.environ[name] = value

    tokenizer = AutoTokenizer.from_pretrained(
        info.model_id,
        revision=info.model_revision,
        use_fast=False,
        trust_remote_code=False,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        info.model_id,
        revision=info.model_revision,
        device_map=BACK_TRANSLATOR_DEVICE_MAP,
        dtype=torch.float16,
        trust_remote_code=False,
    )
    model.eval()

    class _TransformersBacktranslator:
        def translate_batch(self, texts: list[str]) -> list[str]:
            if len(texts) > info.batch_size:
                raise UserInputError(
                    f"semantic backtranslation batch has {len(texts)} rows, above "
                    f"the preregistered {info.batch_size}-row limit",
                    hint="Use the bounded batch loop in create_semantic_review_template.",
                )
            if not texts:
                return []
            encoded = tokenizer(
                texts,
                add_special_tokens=True,
                padding="longest",
                truncation=False,
                return_tensors="pt",
            )
            source_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for index, raw_length in enumerate(source_lengths):
                source_tokens = int(raw_length)
                if source_tokens > info.max_source_tokens:
                    raise UserInputError(
                        f"semantic backtranslation input {index} has {source_tokens} "
                        f"tokens, above the preregistered "
                        f"{info.max_source_tokens}-token limit",
                        hint="Fail the release instead of silently truncating review inputs.",
                    )
            encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=info.max_new_tokens,
                )
            decoded = cast(
                list[str],
                tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            )
            if len(decoded) != len(texts):
                raise UserInputError(
                    "semantic back-translator returned the wrong number of outputs",
                    hint="Do not publish partial or positionally misaligned backtranslations.",
                )
            return decoded

    return _TransformersBacktranslator()
