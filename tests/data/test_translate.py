from __future__ import annotations

import json
from pathlib import Path

import pytest

from sommelier.data.translate import (
    TranslatorInfo,
    audit_translation,
    build_translation_prompt,
    protected_spans,
    strip_scaffolding,
    translate_rows,
    write_translation_outputs,
)
from sommelier.data.types import RawToolCallRow
from sommelier.errors import UserInputError

TOOLS = '[{"name":"search_flights","description":"d","parameters":{}}]'


def _row(index: int, query: str, answers: str) -> RawToolCallRow:
    return RawToolCallRow(
        schema_version="sommelier.raw_tool_call_row.v1",
        source_id=f"en-{index}",
        query=query,
        tools=TOOLS,
        answers=answers,
        source_revision="rev-1",
    )


class FakeTranslator:
    """Returns queued outputs per prompt, recording every prompt seen."""

    def __init__(self, outputs_by_query: dict[str, list[str]]) -> None:
        self.outputs_by_query = outputs_by_query
        self.prompts: list[str] = []

    def translate_batch(self, prompts: list[str]) -> list[str]:
        self.prompts.extend(prompts)
        outputs = []
        for prompt in prompts:
            query = prompt.rsplit("User request:\n", 1)[1]
            outputs.append(self.outputs_by_query[query].pop(0))
        return outputs


def test_protected_spans_are_values_present_in_the_query() -> None:
    query = "Find flights from Berlin to Rome on 2026-08-01 for 2 adults"
    gold_calls = [
        {
            "name": "search_flights",
            "arguments": {
                "origin": "Berlin",
                "destination": "Rome",
                "date": "2026-08-01",
                "adults": 2,
                "cabin": "economy",
                "nested": {"note": ["Rome"]},
                "direct_only": True,
            },
        }
    ]
    spans = protected_spans(query, gold_calls)
    assert "Berlin" in spans and "Rome" in spans and "2026-08-01" in spans
    assert "2" in spans
    # Not in the query, or a boolean: never protected.
    assert "economy" not in spans and "True" not in spans and "true" not in spans


def test_audit_rejects_missing_span_empty_and_untranslated() -> None:
    query = "Find flights from Berlin to Rome"
    spans = ["Berlin", "Rome"]
    assert audit_translation(query, "Trouver des vols de Berlin a Rome", spans) is None
    missing = audit_translation(query, "Trouver des vols vers Rome", spans)
    assert missing is not None and "Berlin" in missing
    empty = audit_translation(query, "   ", spans)
    assert empty is not None and empty.startswith("the output was empty")
    untranslated = audit_translation(query, query, spans)
    assert untranslated is not None and "identical" in untranslated


def test_audit_accepts_identical_output_when_fully_protected() -> None:
    query = "XRP-USD 42.5 2026-08-01"
    spans = ["XRP-USD", "42.5", "2026-08-01"]
    assert audit_translation(query, query, spans) is None


def test_strip_scaffolding_removes_fences_and_quotes() -> None:
    assert strip_scaffolding("```\nBonjour Paris\n```") == "Bonjour Paris"
    assert strip_scaffolding('"Bonjour Paris"') == "Bonjour Paris"
    assert strip_scaffolding("« Bonjour Paris »") == "Bonjour Paris"
    assert strip_scaffolding("  Bonjour Paris  ") == "Bonjour Paris"


def test_translate_rows_emits_paired_rows_with_identical_payloads() -> None:
    answers = '[{"name":"search_flights","arguments":{"origin":"Berlin"}}]'
    rows = [_row(1, "Find flights from Berlin now please", answers)]
    fake = FakeTranslator(
        {"Find flights from Berlin now please": ["Trouver des vols depuis Berlin maintenant"]}
    )
    translated, stats = translate_rows(rows, fake)
    assert stats["translated_rows"] == 1
    row = translated[0]
    assert row["source_id"] == "en-1:fr"
    assert row["source_example_id"] == "en-1"
    assert row["query"] == "Trouver des vols depuis Berlin maintenant"
    assert row["tools"] == TOOLS
    assert row["answers"] == answers
    assert row["source_revision"] == "rev-1"


def test_translate_rows_retries_with_feedback_then_succeeds() -> None:
    answers = '[{"name":"search_flights","arguments":{"origin":"Berlin"}}]'
    rows = [_row(1, "Find flights from Berlin now please", answers)]
    fake = FakeTranslator(
        {
            "Find flights from Berlin now please": [
                "Trouver des vols depuis Berlim",
                "Trouver des vols depuis Berlin",
            ]
        }
    )
    translated, stats = translate_rows(rows, fake)
    assert stats["translated_rows"] == 1
    assert stats["retried_rows"] == 1
    assert "was rejected" in fake.prompts[1]
    assert "Berlin" in fake.prompts[1]


def test_translate_rows_drops_after_exhausted_retries() -> None:
    answers = '[{"name":"search_flights","arguments":{"origin":"Berlin"}}]'
    rows = [_row(1, "Find flights from Berlin now please", answers)]
    fake = FakeTranslator(
        {"Find flights from Berlin now please": ["mauvais", "mauvais", "mauvais"]}
    )
    translated, stats = translate_rows(rows, fake)
    assert translated == []
    dropped = stats["dropped"]
    assert isinstance(dropped, dict)
    assert dropped["missing_protected_span"] == 1


def test_translate_rows_counts_invalid_rows_without_translating() -> None:
    rows = [_row(1, "Query without valid answers", "not-json")]
    fake = FakeTranslator({})
    translated, stats = translate_rows(rows, fake)
    assert translated == []
    dropped = stats["dropped"]
    assert isinstance(dropped, dict)
    assert dropped["invalid_row"] == 1
    assert fake.prompts == []


def test_translate_rows_resumes_from_progress(tmp_path: Path) -> None:
    answers = '[{"name":"search_flights","arguments":{"origin":"Berlin"}}]'
    rows = [
        _row(1, "Find flights from Berlin now please", answers),
        _row(2, "Book flights from Berlin tomorrow please", answers),
    ]
    progress = tmp_path / "progress.jsonl"
    first = FakeTranslator(
        {
            "Find flights from Berlin now please": ["Trouver des vols depuis Berlin"],
            "Book flights from Berlin tomorrow please": ["Reserver des vols depuis Berlin"],
        }
    )
    translate_rows(rows, first, progress_path=progress)

    # A resumed run must not call the model again for resolved rows.
    second = FakeTranslator({})
    translated, stats = translate_rows(rows, second, progress_path=progress)
    assert second.prompts == []
    assert stats["translated_rows"] == 2
    assert [row["source_id"] for row in translated] == ["en-1:fr", "en-2:fr"]


def test_translate_rows_rejects_output_count_mismatch() -> None:
    answers = '[{"name":"search_flights","arguments":{"origin":"Berlin"}}]'
    rows = [_row(1, "Find flights from Berlin now please", answers)]

    class BrokenTranslator:
        def translate_batch(self, prompts: list[str]) -> list[str]:
            return []

    with pytest.raises(UserInputError, match="returned 0 outputs"):
        translate_rows(rows, BrokenTranslator())


def test_short_numeric_spans_match_only_at_boundaries() -> None:
    # "2" inside "2026" neither claims protection from the source nor
    # satisfies the audit in the output.
    gold_calls = [{"name": "search", "arguments": {"page": 2}}]
    assert protected_spans("top 20 results for 2026 trips", gold_calls) == []
    spans = protected_spans("show page 2 of results", gold_calls)
    assert spans == ["2"]
    rejection = audit_translation(
        "show page 2 of results", "afficher les resultats de 2026", spans
    )
    assert rejection is not None and "'2'" in rejection
    assert audit_translation("show page 2 of results", "afficher la page 2", spans) is None


def test_strip_scaffolding_keeps_embedded_quote_pairs() -> None:
    text = '"cheap hotel" contre "youth hostel"'
    assert strip_scaffolding(text) == text


def test_audit_rejects_output_longer_than_query_budget() -> None:
    rejection = audit_translation("court", "x" * 50, [], max_query_chars=40)
    assert rejection is not None and "longer than 40" in rejection


def test_stale_progress_entries_are_retranslated(tmp_path: Path) -> None:
    answers = '[{"name":"search_flights","arguments":{"origin":"Berlin"}}]'
    progress = tmp_path / "progress.jsonl"
    old_row = _row(1, "Old query about Berlin flights", answers)
    first = FakeTranslator({"Old query about Berlin flights": ["Ancienne traduction Berlin"]})
    translate_rows([old_row], first, progress_path=progress)

    # Same source_id, different query: the checkpoint must not be reused.
    new_row = _row(1, "New query about Berlin hotels", answers)
    second = FakeTranslator({"New query about Berlin hotels": ["Nouvelle traduction Berlin"]})
    translated, _ = translate_rows([new_row], second, progress_path=progress)
    assert len(second.prompts) == 1
    assert translated[0]["query"] == "Nouvelle traduction Berlin"


def test_duplicate_fresh_source_ids_translate_once(tmp_path: Path) -> None:
    answers = '[{"name":"search_flights","arguments":{"origin":"Berlin"}}]'
    row = _row(1, "Find flights from Berlin now please", answers)
    fake = FakeTranslator(
        {"Find flights from Berlin now please": ["Trouver des vols depuis Berlin"]}
    )
    translated, stats = translate_rows([row, dict(row)], fake)  # type: ignore[list-item]
    assert len(fake.prompts) == 1
    assert [item["source_id"] for item in translated] == ["en-1:fr"]
    dropped = stats["dropped"]
    assert isinstance(dropped, dict)
    assert dropped["duplicate_source_id"] == 1


def test_protected_spans_sort_longest_first() -> None:
    query = "Fly from Rome to Barcelona"
    gold_calls = [
        {"name": "search_flights", "arguments": {"origin": "Rome", "destination": "Barcelona"}}
    ]
    assert protected_spans(query, gold_calls) == ["Barcelona", "Rome"]


def test_prompt_contains_spans_and_query() -> None:
    prompt = build_translation_prompt("Fly to Rome", ["Rome"])
    assert "Protected spans:\n- Rome" in prompt
    assert prompt.endswith("User request:\nFly to Rome")


def test_write_translation_outputs_records_provenance(tmp_path: Path) -> None:
    answers = '[{"name":"search_flights","arguments":{"origin":"Berlin"}}]'
    rows = [_row(1, "Find flights from Berlin now please", answers)]
    fake = FakeTranslator(
        {"Find flights from Berlin now please": ["Trouver des vols depuis Berlin"]}
    )
    translated, stats = translate_rows(rows, fake)
    rows_path, summary_path = write_translation_outputs(
        tmp_path,
        translated,
        stats,
        translator=TranslatorInfo(
            model_id="stub/translator", model_revision="rev-9", max_new_tokens=128
        ),
        input_description="unit-test rows",
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "sommelier.translation_summary.v1"
    assert summary["translator"]["model_id"] == "stub/translator"
    assert summary["translator"]["model_revision"] == "rev-9"
    assert summary["translator"]["decoding"]["temperature"] == 0.0
    assert len(summary["translator"]["prompt_sha256"]) == 64
    assert summary["translated_rows"] == 1
    record = json.loads(rows_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["source_example_id"] == "en-1"
