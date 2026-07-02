from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from sommelier.errors import InvariantViolation
from sommelier.logs import (
    LOG_EVENT_SCHEMA,
    StageLogger,
    read_log_events,
    stage_logger_for_run_dir,
)


def make_logger(tmp_path: Path, console: io.StringIO | None = None) -> StageLogger:
    return StageLogger(
        run_id="run-123",
        stage="data",
        log_dir=tmp_path / "logs",
        console=console or io.StringIO(),
    )


def test_log_event_shape_and_schema(tmp_path: Path) -> None:
    logger = make_logger(tmp_path)
    event = logger.info("rows_loaded", "loaded raw rows", rows=42, gpu=False, note=None)

    assert event["schema_version"] == LOG_EVENT_SCHEMA
    assert event["level"] == "info"
    assert event["run_id"] == "run-123"
    assert event["stage"] == "data"
    assert event["event"] == "rows_loaded"
    assert event["fields"] == {"rows": 42, "gpu": False, "note": None}
    assert event["timestamp"].endswith("+00:00")

    lines = (tmp_path / "logs" / "data.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == event


def test_log_events_append_across_calls(tmp_path: Path) -> None:
    logger = make_logger(tmp_path)
    logger.info("start", "stage started")
    logger.warning("slow", "stage is slow", elapsed=1.5)
    logger.error("fail", "stage failed")

    events = read_log_events(logger.log_path)
    assert [event["level"] for event in events] == ["info", "warning", "error"]
    assert [event["event"] for event in events] == ["start", "slow", "fail"]


def test_console_mirror_is_concise(tmp_path: Path) -> None:
    console = io.StringIO()
    logger = make_logger(tmp_path, console=console)
    logger.info("rows_loaded", "loaded 42 rows")

    assert console.getvalue() == "[info] data:rows_loaded loaded 42 rows\n"


def test_messages_and_fields_are_redacted(tmp_path: Path) -> None:
    logger = make_logger(tmp_path)
    token = "hf_" + "a" * 30
    event = logger.info("auth", f"using token {token}", token_hint=f"value {token} here")

    assert token not in event["message"]
    assert "[redacted]" in event["message"]
    field_value = event["fields"]["token_hint"]
    assert isinstance(field_value, str)
    assert token not in field_value

    raw = logger.log_path.read_text(encoding="utf-8")
    assert token not in raw


def test_home_directory_paths_are_redacted(tmp_path: Path) -> None:
    logger = make_logger(tmp_path)
    home = Path.home().as_posix()
    event = logger.info("paths", f"reading {home}/dataset.jsonl")

    assert home not in event["message"]
    assert "~/dataset.jsonl" in event["message"]


def test_non_finite_field_is_rejected(tmp_path: Path) -> None:
    logger = make_logger(tmp_path)
    with pytest.raises(InvariantViolation):
        logger.info("bad", "non-finite metric", loss=float("nan"))


def test_stage_logger_for_run_dir_places_logs_under_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-123"
    logger = stage_logger_for_run_dir(
        run_dir,
        run_id="run-123",
        stage="format",
        console=io.StringIO(),
    )
    logger.info("start", "formatting")

    assert (run_dir / "logs" / "format.jsonl").exists()


def test_read_log_events_rejects_foreign_records(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"schema_version": "other.v1"}) + "\n", encoding="utf-8")
    with pytest.raises(InvariantViolation):
        read_log_events(path)
