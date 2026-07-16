"""Paired tokenizer-cost analysis for multilingual formatted datasets.

The analysis is deliberately separate from training. It tokenizes the exact
stored strings that training and evaluation consume, joins every translated
example to its root through ``source_example_id``, and persists both
per-example evidence and aggregate statistics. No model or GPU is required.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

from sommelier.artifacts import ArtifactRef, make_artifact_ref, write_artifact_atomic
from sommelier.config import SommelierConfig
from sommelier.errors import InvariantViolation, SchemaValidationError, UserInputError
from sommelier.formatting.chat import FORMATTED_EXAMPLE_SCHEMA
from sommelier.run_context import RunContext, read_jsonl_records, record_stage_success
from sommelier.security import validate_no_secrets

TOKENIZER_TAX_RECORD_SCHEMA: Final = "sommelier.tokenizer_tax_record.v1"
TOKENIZER_TAX_REPORT_SCHEMA: Final = "sommelier.tokenizer_tax_report.v1"
TOKENIZER_TAX_RECORDS_FILENAME: Final = "tokenizer_tax_records.jsonl"
TOKENIZER_TAX_REPORT_FILENAME: Final = "tokenizer_tax_report.json"

SPLITS: Final = ("train", "validation", "test")
COUNT_NAMES: Final = (
    "query_chars",
    "query_utf8_bytes",
    "query_words",
    "query_tokens",
    "prompt_tokens",
    "target_tokens",
    "full_tokens",
)


class TokenEncoder(Protocol):
    """Minimal tokenizer surface used by the analysis and its CPU tests."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


@dataclass(frozen=True)
class _AnalyzedExample:
    example_id: str
    root_example_id: str
    language: str
    split: str
    source_example_id: str | None
    target_text: str
    counts: dict[str, int]


def _require_string(record: dict[str, object], field: str, *, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(
            f"{context}: {field} must be a non-empty string",
            hint="Rebuild the formatted splits with the current pipeline version.",
        )
    return value


def _user_query(record: dict[str, object], *, context: str) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise SchemaValidationError(
            f"{context}: messages must be a list",
            hint="Formatted records must retain their system, user, and assistant messages.",
        )
    queries: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise SchemaValidationError(
                f"{context}: every message must be an object",
                hint="Rebuild the formatted splits with the current pipeline version.",
            )
        if message.get("role") == "user":
            content = message.get("content")
            if not isinstance(content, str) or not content:
                raise SchemaValidationError(
                    f"{context}: user message content must be a non-empty string",
                    hint="Formatted records must retain the exact source query.",
                )
            queries.append(content)
    if len(queries) != 1:
        raise SchemaValidationError(
            f"{context}: expected exactly one user message, got {len(queries)}",
            hint="Tokenizer-tax pairing requires one query per formatted example.",
        )
    return queries[0]


def _token_count(tokenizer: TokenEncoder, text: str, *, context: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(token_ids, list) or any(
        not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in token_ids
    ):
        raise SchemaValidationError(
            f"{context}: tokenizer.encode must return list[int]",
            hint="Inject a tokenizer compatible with the TokenEncoder protocol.",
        )
    if not token_ids:
        raise SchemaValidationError(
            f"{context}: non-empty text encoded to zero tokens",
            hint="Use the same functioning tokenizer that formats and trains the model.",
        )
    return len(token_ids)


def _analyze_record(
    record: dict[str, object],
    *,
    split: str,
    tokenizer: TokenEncoder,
) -> _AnalyzedExample:
    context = f"formatted {split} example {record.get('example_id', '<unknown>')}"
    if record.get("schema_version") != FORMATTED_EXAMPLE_SCHEMA:
        raise SchemaValidationError(
            f"{context}: expected {FORMATTED_EXAMPLE_SCHEMA}",
            hint="Rebuild the formatted splits with the current pipeline version.",
        )
    recorded_split = _require_string(record, "split", context=context)
    if recorded_split != split:
        raise SchemaValidationError(
            f"{context}: record split {recorded_split!r} does not match file {split!r}",
            hint="Keep each formatted record in its declared split file.",
        )
    example_id = _require_string(record, "example_id", context=context)
    language = _require_string(record, "language", context=context)
    prompt_text = _require_string(record, "prompt_text", context=context)
    target_text = _require_string(record, "target_text", context=context)
    full_text = _require_string(record, "full_text", context=context)
    query = _user_query(record, context=context)

    source_value = record.get("source_example_id")
    if source_value is not None and (not isinstance(source_value, str) or not source_value):
        raise SchemaValidationError(
            f"{context}: source_example_id must be null or a non-empty string",
            hint="Paired rows must name the exact root example they derive from.",
        )
    source_example_id = source_value
    root_example_id = source_example_id or example_id
    counts = {
        "query_chars": len(query),
        "query_utf8_bytes": len(query.encode("utf-8")),
        "query_words": len(query.split()),
        "query_tokens": _token_count(tokenizer, query, context=f"{context} query"),
        "prompt_tokens": _token_count(tokenizer, prompt_text, context=f"{context} prompt_text"),
        "target_tokens": _token_count(tokenizer, target_text, context=f"{context} target_text"),
        "full_tokens": _token_count(tokenizer, full_text, context=f"{context} full_text"),
    }
    return _AnalyzedExample(
        example_id=example_id,
        root_example_id=root_example_id,
        language=language,
        split=split,
        source_example_id=source_example_id,
        target_text=target_text,
        counts=counts,
    )


def _nearest_rank(values: list[int] | list[float], percentile: float) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _count_summary(
    examples: list[_AnalyzedExample], *, max_sequence_length: int
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name in COUNT_NAMES:
        values = [example.counts[name] for example in examples]
        metrics[name] = {
            "total": sum(values),
            "mean": (sum(values) / len(values)) if values else None,
            "p50": _nearest_rank(values, 0.50),
            "p95": _nearest_rank(values, 0.95),
            "p99": _nearest_rank(values, 0.99),
            "max": max(values) if values else None,
        }
    query_tokens = sum(example.counts["query_tokens"] for example in examples)
    query_chars = sum(example.counts["query_chars"] for example in examples)
    query_bytes = sum(example.counts["query_utf8_bytes"] for example in examples)
    query_words = sum(example.counts["query_words"] for example in examples)
    return {
        "examples": len(examples),
        "over_budget": sum(
            example.counts["full_tokens"] > max_sequence_length for example in examples
        ),
        "counts": metrics,
        "rates": {
            "query_tokens_per_character": (
                _finite_ratio(query_tokens, query_chars, context="query tokens/characters")
                if examples
                else None
            ),
            "query_tokens_per_utf8_byte": (
                _finite_ratio(query_tokens, query_bytes, context="query tokens/bytes")
                if examples
                else None
            ),
            "query_tokens_per_whitespace_word": (
                _finite_ratio(query_tokens, query_words, context="query tokens/words")
                if examples
                else None
            ),
        },
    }


def _finite_ratio(numerator: int | float, denominator: int | float, *, context: str) -> float:
    if denominator <= 0:
        raise InvariantViolation(
            f"{context}: ratio denominator must be positive, got {denominator}",
            hint="Tokenizer-tax ratios require non-empty matched root evidence.",
        )
    value = numerator / denominator
    if not math.isfinite(value):
        raise InvariantViolation(
            f"{context}: ratio is not finite",
            hint="Do not persist NaN or infinity in tokenizer-tax artifacts.",
        )
    return value


def _paired_summary(
    paired: list[_AnalyzedExample],
    *,
    roots: dict[str, _AnalyzedExample],
    available_roots: list[_AnalyzedExample],
    context: str,
) -> dict[str, object]:
    coverage = {
        "paired": len(paired),
        "roots": len(available_roots),
        "ratio": _finite_ratio(len(paired), len(available_roots), context=f"{context} coverage"),
    }
    metrics: dict[str, object] = {}
    for name in COUNT_NAMES:
        if not paired:
            metrics[name] = {
                "paired_total": 0,
                "matched_root_total": 0,
                "ratio": None,
                "per_pair_p50": None,
                "per_pair_p95": None,
                "per_pair_max": None,
            }
            continue
        paired_total = sum(example.counts[name] for example in paired)
        root_total = sum(roots[example.root_example_id].counts[name] for example in paired)
        ratios = [
            _finite_ratio(
                example.counts[name],
                roots[example.root_example_id].counts[name],
                context=f"{context} {example.example_id} {name}",
            )
            for example in paired
        ]
        metrics[name] = {
            "paired_total": paired_total,
            "matched_root_total": root_total,
            "ratio": _finite_ratio(paired_total, root_total, context=f"{context} aggregate {name}"),
            "per_pair_p50": _nearest_rank(ratios, 0.50),
            "per_pair_p95": _nearest_rank(ratios, 0.95),
            "per_pair_max": max(ratios),
        }
    return {"coverage": coverage, "metrics": metrics}


def _load_tokenizer(config: SommelierConfig) -> TokenEncoder:
    from sommelier.formatting.templates import load_tokenizer

    return cast(TokenEncoder, load_tokenizer(config))


def analyze_tokenizer_tax(
    config: SommelierConfig,
    *,
    formatted_dir: Path,
    out_dir: Path,
    context: RunContext,
    command: list[str],
    tokenizer: TokenEncoder | None = None,
) -> list[ArtifactRef]:
    """Writes paired tokenization evidence and records a tokenization stage.

    Root examples are the configured root language's rows with a null
    ``source_example_id``. Every other configured language must use a
    non-null ``source_example_id`` that names one root in the same split.
    Missing translations are represented by coverage below one; dangling or
    duplicate pair identities fail closed.
    """
    active_tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(config)
    language_order = [source.language for source in config.datasets]
    configured_languages = set(language_order)
    root_language = config.root_dataset.language
    max_sequence_length = config.train.max_sequence_length

    input_refs: list[ArtifactRef] = []
    analyzed: list[_AnalyzedExample] = []
    seen_example_ids: set[str] = set()
    for split in SPLITS:
        split_path = formatted_dir / f"{split}.jsonl"
        records = read_jsonl_records(split_path)
        input_refs.append(
            make_artifact_ref(
                split_path,
                artifact_root=context.artifact_root,
                kind="formatted_split",
                schema_version=FORMATTED_EXAMPLE_SCHEMA,
            )
        )
        for record in records:
            example = _analyze_record(record, split=split, tokenizer=active_tokenizer)
            if example.example_id in seen_example_ids:
                raise SchemaValidationError(
                    f"duplicate formatted example_id {example.example_id!r}",
                    hint="Example identities must be globally unique across all splits.",
                )
            seen_example_ids.add(example.example_id)
            if example.language not in configured_languages:
                raise SchemaValidationError(
                    f"example {example.example_id!r} uses unconfigured language "
                    f"{example.language!r}",
                    hint="Analyze only records produced by the resolved config.",
                )
            analyzed.append(example)

    roots: dict[str, _AnalyzedExample] = {}
    for example in analyzed:
        if example.language == root_language:
            if example.source_example_id is not None:
                raise SchemaValidationError(
                    f"root-language example {example.example_id!r} names a source_example_id",
                    hint="Root rows must have source_example_id null.",
                )
            roots[example.example_id] = example
        elif example.source_example_id is None:
            raise SchemaValidationError(
                f"paired example {example.example_id!r} has no source_example_id",
                hint="Every non-root language row must name its exact root example.",
            )
    if not roots:
        raise UserInputError(
            f"no {root_language!r} root examples found in {formatted_dir}",
            hint="Run formatting on complete prepared splits before token analysis.",
        )
    for split in SPLITS:
        if not any(root.split == split for root in roots.values()):
            raise UserInputError(
                f"root language {root_language!r} has no examples in split {split!r}",
                hint="Tokenizer-tax evidence requires complete train, validation, and test roots.",
            )

    pairs_by_language: dict[str, list[_AnalyzedExample]] = {
        language: [] for language in language_order if language != root_language
    }
    seen_pair_keys: set[tuple[str, str]] = set()
    for example in analyzed:
        if example.language == root_language:
            continue
        root = roots.get(example.root_example_id)
        if root is None:
            raise SchemaValidationError(
                f"paired example {example.example_id!r} references missing root "
                f"{example.root_example_id!r}",
                hint="Analyze paired rows together with the exact root formatted splits.",
            )
        if root.split != example.split:
            raise SchemaValidationError(
                f"paired example {example.example_id!r} is in {example.split!r}, "
                f"but root {root.example_id!r} is in {root.split!r}",
                hint="Paired examples must inherit their root's split.",
            )
        pair_key = (example.language, example.root_example_id)
        if pair_key in seen_pair_keys:
            raise SchemaValidationError(
                f"duplicate {example.language!r} pair for root {example.root_example_id!r}",
                hint="Keep at most one paired example per language and root identity.",
            )
        seen_pair_keys.add(pair_key)
        if example.target_text != root.target_text:
            raise SchemaValidationError(
                f"paired example {example.example_id!r} target differs from root "
                f"{root.example_id!r}",
                hint="Paired language rows must retain byte-identical gold targets.",
            )
        pairs_by_language[example.language].append(example)

    for language, pairs in pairs_by_language.items():
        if not pairs:
            raise UserInputError(
                f"configured paired language {language!r} has no formatted examples",
                hint="Produce at least one valid pair before claiming tokenizer-tax results.",
            )

    split_rank = {split: index for index, split in enumerate(SPLITS)}
    language_rank = {language: index for index, language in enumerate(language_order)}
    analyzed.sort(
        key=lambda example: (
            split_rank[example.split],
            example.root_example_id,
            language_rank[example.language],
            example.example_id,
        )
    )

    per_example_records: list[dict[str, object]] = []
    for example in analyzed:
        root = roots[example.root_example_id]
        ratios = None
        if example.language != root_language:
            ratios = {
                name: _finite_ratio(
                    example.counts[name],
                    root.counts[name],
                    context=f"example {example.example_id} {name}",
                )
                for name in COUNT_NAMES
            }
        per_example_records.append(
            {
                "schema_version": TOKENIZER_TAX_RECORD_SCHEMA,
                "example_id": example.example_id,
                "root_example_id": example.root_example_id,
                "source_example_id": example.source_example_id,
                "language": example.language,
                "split": example.split,
                "counts": dict(example.counts),
                "over_budget": example.counts["full_tokens"] > max_sequence_length,
                "ratios_to_root": ratios,
            }
        )

    languages: dict[str, object] = {}
    for language in language_order:
        language_examples = [example for example in analyzed if example.language == language]
        languages[language] = {
            "all": _count_summary(language_examples, max_sequence_length=max_sequence_length),
            "splits": {
                split: _count_summary(
                    [example for example in language_examples if example.split == split],
                    max_sequence_length=max_sequence_length,
                )
                for split in SPLITS
            },
        }

    pairing: dict[str, object] = {}
    root_list = list(roots.values())
    for language, pairs in pairs_by_language.items():
        pairing[language] = {
            "all": _paired_summary(
                pairs,
                roots=roots,
                available_roots=root_list,
                context=f"language {language}",
            ),
            "splits": {
                split: _paired_summary(
                    [example for example in pairs if example.split == split],
                    roots=roots,
                    available_roots=[root for root in root_list if root.split == split],
                    context=f"language {language} split {split}",
                )
                for split in SPLITS
            },
        }

    records_text = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in per_example_records
    )
    validate_no_secrets(per_example_records, context="tokenizer-tax records")
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / TOKENIZER_TAX_RECORDS_FILENAME

    def write_records(temp_path: Path) -> None:
        temp_path.write_text(records_text, encoding="utf-8")

    records_ref = write_artifact_atomic(
        records_path,
        write_records,
        artifact_root=context.artifact_root,
        kind="tokenizer_tax_records",
        schema_version=TOKENIZER_TAX_RECORD_SCHEMA,
    )

    training_examples = [
        example
        for example in analyzed
        if example.split == "train" and example.language in config.train.languages
    ]
    tokens_per_epoch = sum(example.counts["full_tokens"] for example in training_examples)
    training_workload: dict[str, object] = {
        "languages": list(config.train.languages),
        "examples_per_epoch": len(training_examples),
        "non_padding_full_tokens_per_epoch": tokens_per_epoch,
        "epochs": config.train.epochs,
        "projected_non_padding_full_tokens": tokens_per_epoch * config.train.epochs,
        "boundary": (
            "Excludes dynamic padding and is a deterministic lower bound on tokens "
            "processed by training."
        ),
    }
    if root_language == "en" and "he" in config.train.languages:
        english_examples = [
            example for example in analyzed if example.split == "train" and example.language == "en"
        ]
        hebrew_examples = [
            example for example in analyzed if example.split == "train" and example.language == "he"
        ]
        english_tokens = sum(example.counts["full_tokens"] for example in english_examples)
        hebrew_tokens = sum(example.counts["full_tokens"] for example in hebrew_examples)
        english_projected_tokens = english_tokens * config.train.epochs
        hebrew_projected_tokens = hebrew_tokens * config.train.epochs
        training_workload.update(
            {
                "english_only_counterfactual": {
                    "language": "en",
                    "examples_per_epoch": len(english_examples),
                    "non_padding_full_tokens_per_epoch": english_tokens,
                    "epochs": config.train.epochs,
                    "projected_non_padding_full_tokens": english_projected_tokens,
                },
                "hebrew_increment": {
                    "language": "he",
                    "examples_per_epoch": len(hebrew_examples),
                    "examples_per_epoch_ratio_to_english_only": _finite_ratio(
                        len(hebrew_examples),
                        len(english_examples),
                        context="Hebrew/English training examples",
                    ),
                    "non_padding_full_tokens_per_epoch": hebrew_tokens,
                    "non_padding_full_tokens_per_epoch_ratio_to_english_only": _finite_ratio(
                        hebrew_tokens,
                        english_tokens,
                        context="Hebrew/English non-padding training tokens",
                    ),
                    "epochs": config.train.epochs,
                    "projected_non_padding_full_tokens": hebrew_projected_tokens,
                    "projected_non_padding_full_tokens_ratio_to_english_only": _finite_ratio(
                        hebrew_projected_tokens,
                        english_projected_tokens,
                        context="projected Hebrew/English non-padding training tokens",
                    ),
                },
                "combined_vs_english_only": {
                    "examples_per_epoch_multiplier": _finite_ratio(
                        len(training_examples),
                        len(english_examples),
                        context="combined/English training examples",
                    ),
                    "non_padding_full_tokens_per_epoch_multiplier": _finite_ratio(
                        tokens_per_epoch,
                        english_tokens,
                        context="combined/English non-padding training tokens",
                    ),
                    "projected_non_padding_full_tokens_multiplier": _finite_ratio(
                        tokens_per_epoch * config.train.epochs,
                        english_projected_tokens,
                        context="projected combined/English non-padding training tokens",
                    ),
                },
            }
        )

    report: dict[str, object] = {
        "schema_version": TOKENIZER_TAX_REPORT_SCHEMA,
        "run_id": context.run_id,
        "config_sha256": context.config_sha256,
        "tokenizer": {
            "id": config.model.base_model_id,
            "revision": config.model.tokenizer_revision,
        },
        "max_sequence_length": max_sequence_length,
        "inputs": {
            split: {
                "path": input_ref["path"],
                "sha256": input_ref["sha256"],
                "bytes": input_ref["bytes"],
            }
            for split, input_ref in zip(SPLITS, input_refs, strict=True)
        },
        "records": {
            "path": records_ref["path"],
            "sha256": hashlib.sha256(records_text.encode("utf-8")).hexdigest(),
            "count": len(per_example_records),
        },
        "count_names": list(COUNT_NAMES),
        "root_language": root_language,
        "languages": languages,
        "pairing": pairing,
        "training_workload": training_workload,
    }
    validate_no_secrets(report, context="tokenizer-tax report")
    report_path = out_dir / TOKENIZER_TAX_REPORT_FILENAME

    def write_report(temp_path: Path) -> None:
        temp_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    report_ref = write_artifact_atomic(
        report_path,
        write_report,
        artifact_root=context.artifact_root,
        kind="tokenizer_tax_report",
        schema_version=TOKENIZER_TAX_REPORT_SCHEMA,
    )

    outputs = [records_ref, report_ref]
    record_stage_success(
        context,
        stage="tokenization",
        command=command,
        seed=config.project.seed,
        inputs=input_refs,
        outputs=outputs,
        details={
            "tokenizer_id": config.model.base_model_id,
            "tokenizer_revision": config.model.tokenizer_revision,
            "max_sequence_length": max_sequence_length,
            "root_language": root_language,
            "paired_languages": list(pairs_by_language),
        },
    )
    return outputs
