"""Hermetic tests for the local Hy-MT2 Hebrew translation driver.

The network call to ollama is monkeypatched; these exercise the masking,
restoration, retry-ladder, audit, and paired-row construction using the real
Sommelier translation primitives.
"""

from __future__ import annotations

import json
import re

import pytest

import local_hy_translate as driver
from sommelier.data.types import RawToolCallRow


def _row() -> RawToolCallRow:
    return {
        "schema_version": "sommelier.raw_tool_call_row.v1",
        "source_id": "Salesforce/xlam-function-calling-60k:7",
        "query": "Get the status of order ORD123 please",
        "tools": json.dumps(
            [
                {
                    "name": "get_order",
                    "description": "Get an order by id.",
                    "parameters": {
                        "type": "object",
                        "properties": {"is_id": {"type": "string"}},
                    },
                }
            ]
        ),
        "answers": json.dumps([{"name": "get_order", "arguments": {"is_id": "ORD123"}}]),
        "source_revision": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
    }


def test_accepts_and_pairs_when_sentinel_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fake translator keeps whatever sentinel the driver injected, so the
    # protected span is restored byte-identically and the audit passes.
    def fake_generate(model: str, source_text: str, seed: int, timeout: float) -> str:
        # Echo whatever sentinel/protected token the driver injected so the
        # span is restored intact.
        preserved = " ".join(re.findall(r"[A-Z0-9]{3,}", source_text))
        return f"קבל את הסטטוס של ההזמנה {preserved} בבקשה".strip()

    monkeypatch.setattr(driver, "_generate", fake_generate)
    record = driver.translate_row(_row(), model="fake", timeout=1.0)

    assert record["status"] == "ok"
    paired = record["row"]
    assert isinstance(paired, dict)
    assert paired["source_id"] == "Salesforce/xlam-function-calling-60k:7:he"
    assert paired["source_example_id"] == "Salesforce/xlam-function-calling-60k:7"
    # tools and answers are copied byte-identically from the root row.
    assert paired["tools"] == _row()["tools"]
    assert paired["answers"] == _row()["answers"]
    # the protected span survives, translated Hebrew is present.
    assert "ORD123" in paired["query"]
    assert "קבל" in paired["query"]


def test_drops_when_protected_span_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fake translator never echoes the sentinel and drops the raw span, so
    # every ladder attempt fails the protected-span audit.
    def fake_generate(model: str, source_text: str, seed: int, timeout: float) -> str:
        return "קבל את הסטטוס של ההזמנה בבקשה"

    monkeypatch.setattr(driver, "_generate", fake_generate)
    record = driver.translate_row(_row(), model="fake", timeout=1.0)

    assert record["status"] == "drop"
    assert record["reason"] == "missing_protected_span"


def test_transport_error_falls_through_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(model: str, source_text: str, seed: int, timeout: float) -> str:
        raise TimeoutError("ollama unavailable")

    monkeypatch.setattr(driver, "_generate", boom)
    record = driver.translate_row(_row(), model="fake", timeout=1.0)

    assert record["status"] == "drop"
