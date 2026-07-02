from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

from sommelier.data.types import ToolCall
from sommelier.evaluation.metrics import (
    METRIC_NAMES,
    ScoredRecord,
    compute_metrics,
)
from sommelier.evaluation.parse import parse_tool_call

FIXTURES_DIR = Path("tests/fixtures/evaluation")
RAW_PATH = FIXTURES_DIR / "raw_generations.jsonl"
GOLDEN_SCORED_PATH = FIXTURES_DIR / "golden_scored_records.jsonl"
GOLDEN_METRICS_PATH = FIXTURES_DIR / "golden_metrics.json"

REGENERATE_ENV = "SOMMELIER_REGENERATE_GOLDEN"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score_all() -> list[ScoredRecord]:
    records: list[ScoredRecord] = []
    for fixture in load_jsonl(RAW_PATH):
        parsed_call, parse_status = parse_tool_call(str(fixture["raw_text"]))
        records.append(
            ScoredRecord(
                example_id=str(fixture["fixture_id"]),
                parse_status=parse_status,
                parsed_call=parsed_call,
                gold_call=cast(ToolCall, fixture["gold_call"]),
            )
        )
    return records


def test_scored_records_match_golden_snapshot() -> None:
    scored = score_all()

    if os.environ.get(REGENERATE_ENV) == "1":
        GOLDEN_SCORED_PATH.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in scored),
            encoding="utf-8",
        )

    golden = load_jsonl(GOLDEN_SCORED_PATH)
    assert len(golden) == len(scored)
    for got, expected in zip(scored, golden, strict=True):
        normalized = json.loads(json.dumps(got, sort_keys=True))
        assert normalized == expected, (
            f"parser/scoring drift for fixture {expected.get('example_id')}; "
            f"if intentional, regenerate with {REGENERATE_ENV}=1 and review the diff"
        )


def test_metrics_match_golden_snapshot() -> None:
    metrics = compute_metrics(score_all())

    if os.environ.get(REGENERATE_ENV) == "1":
        GOLDEN_METRICS_PATH.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    golden = json.loads(GOLDEN_METRICS_PATH.read_text(encoding="utf-8"))
    normalized = json.loads(json.dumps(metrics, sort_keys=True))
    assert normalized == golden


def test_fixtures_cover_every_parse_status() -> None:
    statuses = {record["parse_status"] for record in score_all()}
    assert statuses == {"ok", "no_json", "invalid_json", "invalid_shape"}


def test_every_metric_denominator_counts_all_or_pooled_pairs() -> None:
    scored = score_all()
    metrics = compute_metrics(scored)
    total = len(scored)

    for name in METRIC_NAMES:
        if name == "argument_f1":
            continue
        assert metrics[name]["denominator"] == total, name

    gold_pairs = 0
    predicted_pairs = 0
    from sommelier.evaluation.metrics import flatten_arguments

    for record in scored:
        gold_pairs += len(flatten_arguments(record["gold_call"]["arguments"]))
        if record["parse_status"] == "ok" and record["parsed_call"] is not None:
            predicted_pairs += len(flatten_arguments(record["parsed_call"]["arguments"]))
    assert metrics["argument_f1"]["denominator"] == gold_pairs + predicted_pairs


def test_failure_fixtures_have_null_parsed_call() -> None:
    for record in score_all():
        if record["parse_status"] != "ok":
            assert record["parsed_call"] is None, record["example_id"]
        else:
            assert record["parsed_call"] is not None, record["example_id"]
