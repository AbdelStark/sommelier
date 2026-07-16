#!/usr/bin/env python
"""Local Hebrew translation of the xLAM single-call corpus with Tencent Hy-MT2.

This is a dataset-production tool, not a pipeline stage. It reproduces the exact
root-cohort selection a full pipeline run would make, then translates only each
natural-language ``query`` into Hebrew with the locally served
``tencent/Hy-MT2-1.8B`` model (Q8_0 GGUF via ollama), keeping tool schemas and
gold answers byte-identical. It reuses Sommelier's audited protected-span and
translation-audit primitives so the emitted ``rows.he.jsonl`` satisfies the same
production-time contract the prepare stage re-validates independently.

The Hy-MT2 model is a pure translator: it will translate gold-argument values
(for example "Tel Aviv"). To keep those byte-identical we mask each protected
span with a short ASCII sentinel before translation and restore it afterward,
retrying with alternate sentinels and finally an unmasked pass. Every accepted
row still passes ``audit_translation`` (protected spans present, Hebrew script,
no unsafe bidi/control characters), so masking only raises yield and never
weakens the gate.

Determinism: greedy decoding (temperature 0) with a fixed per-attempt seed.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from sommelier.config import SommelierConfig, load_config
from sommelier.data.export import export_raw_rows
from sommelier.data.load import load_raw_rows
from sommelier.data.split import all_examples, prepare_split_result
from sommelier.data.translate import (
    DROP_REASONS,
    TranslationDropReason,
    _categorize,
    _paired_row,
    audit_translation,
    protected_spans,
    resolve_translation_target,
    rows_filename,
)
from sommelier.data.types import RawToolCallRow, SplitName
from sommelier.data.validate import validate_raw_row

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "hf.co/tencent/Hy-MT2-1.8B-GGUF:Q8_0"
# The GGUF's embedded ollama chat template is corrupt, so we build the Hunyuan
# turn format by hand and generate in raw mode.
HY_STOP = ["<｜hy_place▁holder▁no▁2｜>", "<｜hy_begin▁of▁sentence｜>", "<｜hy_User｜>"]
HY_INSTRUCTION = (
    "Translate the following text into Hebrew. Note that you should only output "
    "the translated result without any additional explanation"
)
# Retry ladder: short alphanumeric sentinels survive the translator far better
# than descriptive ones; the final entry disables masking entirely.
SentinelTemplate = Callable[[int], str]
SENTINEL_TEMPLATES: tuple[SentinelTemplate | None, ...] = (
    lambda i: f"ZZQ{i}ZZ",
    lambda i: f"QX{i:02d}QX",
    lambda i: f"WYW{i}WYW",
    None,  # unmasked pass
)


class AcceptedTranslationRecord(TypedDict):
    source_id: str
    status: Literal["ok"]
    attempt: int
    row: RawToolCallRow


class DroppedTranslationRecord(TypedDict):
    source_id: str
    status: Literal["drop"]
    reason: TranslationDropReason
    error: NotRequired[str]


TranslationRecord = AcceptedTranslationRecord | DroppedTranslationRecord


def _generate(model: str, source_text: str, seed: int, timeout: float) -> str:
    prompt = (
        f"<｜hy_begin▁of▁sentence｜><｜hy_User｜>{HY_INSTRUCTION}\n\n"
        f"{source_text}<｜hy_Assistant｜>"
    )
    body = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_predict": 640,
            "stop": HY_STOP,
        },
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return str(payload.get("response", "")).strip()


def _attempt(
    model: str,
    query: str,
    spans: list[str],
    template: SentinelTemplate | None,
    seed: int,
    timeout: float,
) -> tuple[str, str | None]:
    replacements: list[tuple[str, str]] = []
    masked = query
    if template is not None:
        for index, span in enumerate(spans):
            placeholder = template(index)
            masked = masked.replace(span, placeholder)
            replacements.append((placeholder, span))
    output = _generate(model, masked, seed, timeout)
    for placeholder, span in replacements:
        output = output.replace(placeholder, span)
    output = unicodedata.normalize("NFC", output)
    reason = audit_translation(query, output, spans, max_query_chars=2000, target_language="he")
    return output, reason


def translate_row(row: RawToolCallRow, *, model: str, timeout: float) -> TranslationRecord:
    """Translate one root row, returning a progress record."""
    source_id = row["source_id"]
    try:
        validated = validate_raw_row(
            row, min_query_chars=1, max_query_chars=1_000_000, language="en"
        )
        if isinstance(validated, str):
            return {"source_id": source_id, "status": "drop", "reason": "invalid_row"}
        query = row["query"]
        spans = protected_spans(query, validated["gold_calls"])
        last_reason: str | None = None
        for attempt_index, template in enumerate(SENTINEL_TEMPLATES):
            try:
                output, reason = _attempt(
                    model, query, spans, template, seed=42 + attempt_index, timeout=timeout
                )
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_reason = f"transport error: {error}"
                continue
            if reason is None:
                paired = _paired_row(row, output, target_language="he")
                return {
                    "source_id": source_id,
                    "status": "ok",
                    "attempt": attempt_index + 1,
                    "row": paired,
                }
            last_reason = reason
        return {
            "source_id": source_id,
            "status": "drop",
            "reason": _categorize(last_reason) if last_reason else "untranslated_output",
        }
    except Exception as error:  # noqa: BLE001 - one bad row must never kill the run
        return {
            "source_id": source_id,
            "status": "drop",
            "reason": "invalid_row",
            "error": repr(error),
        }


def _load_progress(path: Path) -> dict[str, TranslationRecord]:
    done: dict[str, TranslationRecord] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = cast(TranslationRecord, json.loads(line))
        done[record["source_id"]] = record
    return done


def select_root_rows(
    config: SommelierConfig, *, limit: int
) -> tuple[list[RawToolCallRow], dict[str, SplitName]]:
    """Export and select the exact root cohort, returning rows + split map."""
    source = config.root_dataset
    export_path = (
        Path(config.project.artifact_root)
        / "hy-translate-cache"
        / (f"rows.{source.language}.jsonl")
    )
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_path.exists():
        print(f"[export] exporting {source.dataset_id} @ {source.dataset_revision} ...", flush=True)
        count = export_raw_rows(source, export_path, seed=config.project.seed, max_rows=0)
        print(f"[export] wrote {count} raw rows to {export_path}", flush=True)
    rows = load_raw_rows(export_path)
    result = prepare_split_result(
        rows,
        min_query_chars=config.data.min_query_chars,
        max_query_chars=config.data.max_query_chars,
        n_train=config.data.n_train,
        n_validation=config.data.n_validation,
        n_test=config.data.n_test,
        seed=config.project.seed,
        language=source.language,
    )
    split_by_id = {ex["example_id"]: ex["split"] for ex in all_examples(result)}
    selected_ids = set(split_by_id)
    selected = [row for row in rows if row["source_id"] in selected_ids]
    if limit:
        selected = selected[:limit]
    return selected, split_by_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--limit", type=int, default=0, help="translate only the first N selected rows"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    target = resolve_translation_target("he")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    selected, split_by_id = select_root_rows(config, limit=args.limit)
    print(f"[select] {len(selected)} root rows selected", flush=True)

    done = _load_progress(progress_path)
    remaining = [row for row in selected if row["source_id"] not in done]
    print(f"[resume] {len(done)} already processed, {len(remaining)} remaining", flush=True)

    lock = threading.Lock()
    progress_file = progress_path.open("a", encoding="utf-8")
    counter = {"n": 0}
    start = time.monotonic()

    def worker(row: RawToolCallRow) -> TranslationRecord:
        record = translate_row(row, model=args.model, timeout=args.timeout)
        with lock:
            progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress_file.flush()
            done[str(record["source_id"])] = record
            counter["n"] += 1
            n = counter["n"]
            if n % 200 == 0 or n == len(remaining):
                rate = n / max(time.monotonic() - start, 1e-6)
                ok = sum(1 for r in done.values() if r["status"] == "ok")
                print(
                    f"[progress] {n}/{len(remaining)} this run "
                    f"| accepted total {ok}/{len(done)} | {rate:.2f} rows/s",
                    flush=True,
                )
        return record

    if remaining:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(worker, remaining))
    progress_file.close()

    # Assemble accepted rows in deterministic selected order.
    accepted: list[RawToolCallRow] = []
    drop_counts: dict[TranslationDropReason, int] = {reason: 0 for reason in DROP_REASONS}
    attempts_hist: dict[int, int] = {}
    split_counts: dict[SplitName, int] = {"train": 0, "validation": 0, "test": 0}
    for row in selected:
        record = done.get(row["source_id"])
        if record is None:
            continue
        if record["status"] == "ok":
            accepted.append(record["row"])
            attempt = record["attempt"]
            attempts_hist[attempt] = attempts_hist.get(attempt, 0) + 1
            split = split_by_id.get(row["source_id"])
            if split in split_counts:
                split_counts[split] += 1
        else:
            reason = record["reason"]
            drop_counts[reason] += 1

    rows_path = out_dir / rows_filename(target.code)
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "schema_version": "sommelier.local_hy_translation_summary.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "language": target.code,
        "language_name": target.name,
        "translator": {
            "backend": "ollama_raw_generate",
            "model": args.model,
            "source_model": "tencent/Hy-MT2-1.8B",
            "quantization": "Q8_0_GGUF",
            "decoding": "greedy_temperature_0",
            "instruction": HY_INSTRUCTION,
            "human_semantic_review": False,
        },
        "script_policy": {
            "required_script": target.required_script,
            "min_fraction": target.min_script_fraction,
            "unsafe_bidi_controls": "reject",
            "unicode_normalization": "NFC",
        },
        "root_dataset": {
            "dataset_id": config.root_dataset.dataset_id,
            "dataset_revision": config.root_dataset.dataset_revision,
        },
        "selected_rows": len(selected),
        "translated_rows": len(accepted),
        "dropped": drop_counts,
        "accepted_by_attempt": {str(k): v for k, v in sorted(attempts_hist.items())},
        "accepted_by_split": split_counts,
        "yield": round(len(accepted) / len(selected), 4) if selected else 0.0,
    }
    summary_path = out_dir / "translation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"[done] accepted {len(accepted)}/{len(selected)} "
        f"({summary['yield']:.1%}) | splits {split_counts} | drops "
        f"{ {k: v for k, v in drop_counts.items() if v} }",
        flush=True,
    )
    print(f"[done] rows -> {rows_path}", flush=True)
    print(f"[done] summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
