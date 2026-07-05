"""Constrained query translation for paired dataset sources.

This is a dataset production tool, not a pipeline stage: it turns the
root source's raw rows into a paired language's raw rows by translating
only the query text. Tool schemas and gold answers are copied byte for
byte, and every translation is audited against the protected spans the
gold answer depends on, so the pairing contract that ``data prepare``
enforces is already satisfied by construction at production time.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Protocol

from sommelier.data.types import JsonObject, RawToolCallRow, ToolCall
from sommelier.data.validate import validate_raw_row
from sommelier.errors import ExternalDependencyError, UserInputError
from sommelier.run_context import read_jsonl_records, write_jsonl_records

TRANSLATION_SUMMARY_SCHEMA: Final = "sommelier.translation_summary.v1"
ROWS_FILENAME: Final = "rows.fr.jsonl"
SUMMARY_FILENAME: Final = "translation_summary.json"
PROGRESS_FILENAME: Final = "translation_progress.jsonl"

# One translation plus two feedback retries.
MAX_ATTEMPTS: Final = 3

TranslationDropReason = Literal[
    "invalid_row",
    "duplicate_source_id",
    "missing_protected_span",
    "empty_output",
    "untranslated_output",
    "output_too_long",
]
DROP_REASONS: Final[tuple[TranslationDropReason, ...]] = (
    "invalid_row",
    "duplicate_source_id",
    "missing_protected_span",
    "empty_output",
    "untranslated_output",
    "output_too_long",
)

# Rows translated per model call: small enough that an interrupted run
# loses at most one chunk of work, large enough for vLLM batching.
DEFAULT_CHUNK_SIZE: Final = 512

PROMPT_TEMPLATE: Final = """Translate the user request below from English to French.

Rules:
- Translate into natural, fluent French.
- Reproduce every protected span exactly as written, byte for byte, including casing.
- Keep numbers exactly as written: same digits, decimal point rather than decimal comma.
- Do not add explanations, quotes, labels, or anything besides the translated request.
{feedback}{spans}
User request:
{query}"""


class TranslationModel(Protocol):
    """Batched greedy translation; implementations load models lazily."""

    def translate_batch(self, prompts: list[str]) -> list[str]: ...


@dataclass(frozen=True)
class TranslatorInfo:
    model_id: str
    model_revision: str
    max_new_tokens: int


def prompt_template_sha256() -> str:
    """Identity of the translation prompt, recorded in the summary."""
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def _walk_values(value: object) -> list[object]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _walk_values(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _walk_values(item)]
    return [value]


def span_present(text: str, span: str) -> bool:
    """Boundary-aware span matching.

    Short or numeric spans match only at alphanumeric boundaries, so a
    gold value of ``2`` is neither claimed by ``2026`` in the source nor
    satisfied by ``2026`` in the translation. Longer textual spans use
    plain substring matching.
    """
    if len(span) >= 3 and not span.replace(".", "").replace("-", "").isdigit():
        return span in text
    pattern = rf"(?<![0-9A-Za-z]){re.escape(span)}(?![0-9A-Za-z])"
    return re.search(pattern, text) is not None


def protected_spans(query: str, gold_calls: list[ToolCall] | list[JsonObject]) -> list[str]:
    """Gold argument values that appear verbatim in the English query.

    Only what is present in the source can be required in the target:
    an argument value the model derived (a default, a code) is part of
    the gold answer either way, because answers are copied byte for byte.
    """
    spans: set[str] = set()
    for call in gold_calls:
        for leaf in _walk_values(call.get("arguments", {})):
            if isinstance(leaf, bool):
                continue
            if isinstance(leaf, str):
                candidate = leaf
            elif isinstance(leaf, int | float):
                candidate = json.dumps(leaf)
            else:
                continue
            if candidate and span_present(query, candidate):
                spans.add(candidate)
    return sorted(spans, key=lambda span: (-len(span), span))


def build_translation_prompt(
    query: str,
    spans: list[str],
    *,
    feedback: str | None = None,
) -> str:
    spans_block = ""
    if spans:
        lines = "\n".join(f"- {span}" for span in spans)
        spans_block = f"\nProtected spans:\n{lines}\n"
    feedback_block = ""
    if feedback:
        feedback_block = f"- The previous attempt was rejected: {feedback}\n"
    return PROMPT_TEMPLATE.format(feedback=feedback_block, spans=spans_block, query=query)


def strip_scaffolding(text: str) -> str:
    """Removes wrappers a chat model tends to add around the translation.

    Wrapping quotes are stripped only when they are the sole quotes of
    their kind, so a translation that legitimately begins and ends with
    two different quoted phrases is left intact.
    """
    out = text.strip()
    if out.startswith("```") and out.endswith("```"):
        inner = out[3:-3]
        first_newline = inner.find("\n")
        if first_newline != -1 and " " not in inner[:first_newline]:
            inner = inner[first_newline + 1 :]
        out = inner.strip()
    for opening, closing in (('"', '"'), ("«", "»"), ("“", "”")):
        if (
            len(out) >= 2
            and out.startswith(opening)
            and out.endswith(closing)
            and opening not in out[1:-1]
            and closing not in out[1:-1]
        ):
            out = out[1:-1].strip()
            break
    return out


def normalize_numeric_spans(output: str, spans: list[str]) -> str:
    """Restores protected decimal spans written with a French comma.

    The prompt pins the number format, but a French model still renders
    0.5 as 0,5 often enough to matter. When a protected span is a decimal
    number that is missing from the output while its comma variant is
    present, the variant is rewritten back; nothing else is touched.
    """
    for span in spans:
        if "." not in span or not span.replace(".", "").isdigit():
            continue
        if span_present(output, span):
            continue
        comma_variant = span.replace(".", ",")
        pattern = rf"(?<![0-9A-Za-z]){re.escape(comma_variant)}(?![0-9A-Za-z])"
        output = re.sub(pattern, span, output)
    return output


def _fully_protected(query: str, spans: list[str]) -> bool:
    remainder = query
    for span in spans:
        remainder = remainder.replace(span, "")
    return sum(1 for char in remainder if char.isalpha()) < 4


def audit_translation(
    source_query: str,
    output: str,
    spans: list[str],
    *,
    max_query_chars: int = 2000,
) -> str | None:
    """Returns a rejection description, or None when the output passes.

    The length rule mirrors the prepare stage's ``max_query_chars`` so a
    translation that could never survive preparation is rejected here,
    where it can still be retried, instead of silently dropping later.
    """
    if not output.strip():
        return "the output was empty (or hit the generation token budget)"
    missing = [span for span in spans if not span_present(output, span)]
    if missing:
        shown = ", ".join(repr(span) for span in missing[:3])
        return f"missing protected span(s): {shown}"
    if len(output.strip()) > max_query_chars:
        return f"the output is longer than {max_query_chars} characters"
    if output.strip() == source_query.strip() and not _fully_protected(source_query, spans):
        return "the output is identical to the English source"
    return None


def _categorize(reason: str) -> TranslationDropReason:
    if reason.startswith("invalid row"):
        return "invalid_row"
    if reason.startswith("duplicate source_id"):
        return "duplicate_source_id"
    if reason.startswith("missing protected span"):
        return "missing_protected_span"
    if "empty" in reason:
        return "empty_output"
    if "longer than" in reason:
        return "output_too_long"
    return "untranslated_output"


@dataclass
class _PendingRow:
    row: RawToolCallRow
    spans: list[str]
    feedback: str | None = None


def _paired_row(row: RawToolCallRow, translated_query: str) -> RawToolCallRow:
    paired = RawToolCallRow(
        schema_version="sommelier.raw_tool_call_row.v1",
        source_id=f"{row['source_id']}:fr",
        query=translated_query,
        tools=row["tools"],
        answers=row["answers"],
        source_revision=row["source_revision"],
    )
    paired["source_example_id"] = row["source_id"]
    return paired


def _query_digest(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _read_progress(progress_path: Path) -> dict[str, dict[str, object]]:
    """Loads checkpointed outcomes, tolerating a truncated final line.

    A hard kill can leave a partial JSON line at the tail of the append-only
    file; skipping unparseable lines just re-translates those rows, which is
    exactly what a resume is for.
    """
    done: dict[str, dict[str, object]] = {}
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "source_id" in record:
            done[str(record["source_id"])] = record
    return done


def translate_rows(
    rows: list[RawToolCallRow],
    model: TranslationModel,
    *,
    progress_path: Path | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    max_query_chars: int = 2000,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[list[RawToolCallRow], dict[str, object]]:
    """Translates every row's query, enforcing the protected span audit.

    Failures are retried up to ``max_attempts - 1`` times with the audit
    rejection appended to the prompt, then dropped with a counted reason.
    Rows go to the model in chunks and every resolved row is checkpointed
    when its chunk completes, so an interrupted run loses at most one
    chunk. A checkpoint is only reused when the source query's digest
    still matches, so changing the input re-translates instead of grafting
    a stale translation onto a different query. Results keep the input
    row order.
    """
    drop_counts: dict[TranslationDropReason, int] = {reason: 0 for reason in DROP_REASONS}
    resolved: dict[str, RawToolCallRow | None] = {}
    retried_ids: set[str] = set()

    done: dict[str, dict[str, object]] = {}
    if progress_path is not None and progress_path.exists():
        done = _read_progress(progress_path)

    def checkpoint(source_id: str, query: str, payload: dict[str, object]) -> None:
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as handle:
            record: dict[str, object] = {
                "source_id": source_id,
                "source_query_sha256": _query_digest(query),
                **payload,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    pending: list[_PendingRow] = []
    for row in rows:
        source_id = row["source_id"]
        if source_id in resolved:
            drop_counts["duplicate_source_id"] += 1
            continue
        previous = done.get(source_id)
        if previous is not None and previous.get("source_query_sha256") == _query_digest(
            row["query"]
        ):
            translated_query = previous.get("query")
            if translated_query is not None:
                resolved[source_id] = _paired_row(row, str(translated_query))
            else:
                reason = str(previous.get("dropped"))
                drop_counts[_categorize(reason)] += 1
                resolved[source_id] = None
            continue
        # Shape check only: length policy belongs to the prepare stage,
        # which already selected these rows under the config bounds.
        validated = validate_raw_row(
            row, min_query_chars=1, max_query_chars=1_000_000, language="en"
        )
        if isinstance(validated, str):
            drop_counts["invalid_row"] += 1
            resolved[source_id] = None
            checkpoint(source_id, row["query"], {"dropped": f"invalid row: {validated}"})
            continue
        resolved[source_id] = None
        pending.append(
            _PendingRow(row=row, spans=protected_spans(row["query"], validated["gold_calls"]))
        )

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        if attempt > 1:
            retried_ids.update(item.row["source_id"] for item in pending)
        still_pending: list[_PendingRow] = []
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start : start + chunk_size]
            prompts = [
                build_translation_prompt(item.row["query"], item.spans, feedback=item.feedback)
                for item in chunk
            ]
            outputs = model.translate_batch(prompts)
            if len(outputs) != len(prompts):
                raise UserInputError(
                    f"translator returned {len(outputs)} outputs for {len(prompts)} prompts",
                    hint="The translation model must return one output per prompt.",
                )
            for item, raw_output in zip(chunk, outputs, strict=True):
                output = normalize_numeric_spans(
                    strip_scaffolding(raw_output), item.spans
                )
                rejection = audit_translation(
                    item.row["query"], output, item.spans, max_query_chars=max_query_chars
                )
                source_id = item.row["source_id"]
                if rejection is None:
                    resolved[source_id] = _paired_row(item.row, output)
                    checkpoint(source_id, item.row["query"], {"query": output})
                    continue
                if attempt == max_attempts:
                    drop_counts[_categorize(rejection)] += 1
                    checkpoint(source_id, item.row["query"], {"dropped": rejection})
                    continue
                still_pending.append(
                    _PendingRow(row=item.row, spans=item.spans, feedback=rejection)
                )
        pending = still_pending

    emitted: set[str] = set()
    translated: list[RawToolCallRow] = []
    for row in rows:
        source_id = row["source_id"]
        if source_id in emitted:
            continue
        emitted.add(source_id)
        paired = resolved.get(source_id)
        if paired is not None:
            translated.append(paired)
    stats: dict[str, object] = {
        "input_rows": len(rows),
        "translated_rows": len(translated),
        "dropped": dict(drop_counts),
        "retried_rows": len(retried_ids),
    }
    return translated, stats


def select_example_ids(prepared_dir: Path) -> set[str]:
    """Example ids of every prepared split row, for selection filtering."""
    selected: set[str] = set()
    for split in ("train", "validation", "test"):
        for record in read_jsonl_records(prepared_dir / f"{split}.jsonl"):
            selected.add(str(record["example_id"]))
    if not selected:
        raise UserInputError(
            f"no prepared examples found under {prepared_dir}",
            hint="Point --select-from at a data directory with prepared splits.",
        )
    return selected


def write_translation_outputs(
    out_dir: Path,
    translated: list[RawToolCallRow],
    stats: dict[str, object],
    *,
    translator: TranslatorInfo,
    input_description: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / ROWS_FILENAME
    write_jsonl_records(rows_path, [dict(row) for row in translated])
    summary = {
        "schema_version": TRANSLATION_SUMMARY_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "language": "fr",
        "translator": {
            "model_id": translator.model_id,
            "model_revision": translator.model_revision,
            "decoding": {
                "temperature": 0.0,
                "do_sample": False,
                "max_new_tokens": translator.max_new_tokens,
            },
            "prompt_sha256": prompt_template_sha256(),
        },
        "input": input_description,
        **stats,
    }
    summary_path = out_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return rows_path, summary_path


def load_vllm_translator(info: TranslatorInfo) -> TranslationModel:
    """Batched greedy decoding through vLLM's offline chat interface.

    vllm is a remote-image dependency; importing it happens here, inside
    the tool entrypoint, never at package import time.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise ExternalDependencyError(
            "translation requires the vllm package",
            hint="Run the tool remotely (remote_translate.py) or install vllm.",
        ) from error

    # Queries are capped at 2000 characters and outputs at the token budget,
    # so a short context suffices; the model's native 128k default needs
    # more KV cache than an L40S has left after bf16 12B weights.
    llm = LLM(
        model=info.model_id,
        revision=info.model_revision,
        dtype="bfloat16",
        max_model_len=8192,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=info.max_new_tokens)

    class _VllmTranslator:
        def translate_batch(self, prompts: list[str]) -> list[str]:
            conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
            outputs = llm.chat(conversations, sampling, use_tqdm=True)
            texts: list[str] = []
            for output in outputs:
                completion = output.outputs[0]
                if completion.finish_reason != "stop":
                    # A truncated translation can pass the span audit with
                    # a garbled tail; an empty output is rejected instead.
                    texts.append("")
                else:
                    texts.append(completion.text)
            return texts

    return _VllmTranslator()
