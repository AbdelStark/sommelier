from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, TextIO, TypedDict, cast

from sommelier.errors import InvariantViolation
from sommelier.security import redact_text

LOG_EVENT_SCHEMA: Final = "sommelier.log_event.v1"

LogLevel = Literal["debug", "info", "warning", "error"]
FieldValue = str | int | float | bool | None


class LogEvent(TypedDict):
    schema_version: Literal["sommelier.log_event.v1"]
    timestamp: str
    level: LogLevel
    run_id: str
    stage: str
    event: str
    message: str
    fields: dict[str, FieldValue]


def _sanitize_field(name: str, value: FieldValue) -> FieldValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise InvariantViolation(
            f"log field {name!r} is not a finite number",
            hint="Log non-finite values explicitly as strings or omit them.",
        )
    if isinstance(value, str):
        return redact_text(value)
    return value


class StageLogger:
    """Writes schema-versioned JSONL log events for one pipeline stage.

    Events land in ``<log_dir>/<stage>.jsonl`` and are mirrored as concise
    human-readable lines on the console stream. The JSONL file is the source
    of truth; the console rendering is best-effort.
    """

    def __init__(
        self,
        *,
        run_id: str,
        stage: str,
        log_dir: Path,
        console: TextIO | None = None,
    ) -> None:
        self.run_id = run_id
        self.stage = stage
        self.log_path = log_dir / f"{stage}.jsonl"
        self._console = console if console is not None else sys.stderr

    def log(
        self,
        level: LogLevel,
        event: str,
        message: str,
        **fields: FieldValue,
    ) -> LogEvent:
        record: LogEvent = {
            "schema_version": LOG_EVENT_SCHEMA,
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "run_id": self.run_id,
            "stage": self.stage,
            "event": event,
            "message": redact_text(message),
            "fields": {name: _sanitize_field(name, value) for name, value in fields.items()},
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._console.write(self.render_console(record) + "\n")
        return record

    def debug(self, event: str, message: str, **fields: FieldValue) -> LogEvent:
        return self.log("debug", event, message, **fields)

    def info(self, event: str, message: str, **fields: FieldValue) -> LogEvent:
        return self.log("info", event, message, **fields)

    def warning(self, event: str, message: str, **fields: FieldValue) -> LogEvent:
        return self.log("warning", event, message, **fields)

    def error(self, event: str, message: str, **fields: FieldValue) -> LogEvent:
        return self.log("error", event, message, **fields)

    @staticmethod
    def render_console(record: LogEvent) -> str:
        return f"[{record['level']}] {record['stage']}:{record['event']} {record['message']}"


def stage_logger_for_run_dir(
    run_dir: Path,
    *,
    run_id: str,
    stage: str,
    console: TextIO | None = None,
) -> StageLogger:
    return StageLogger(
        run_id=run_id,
        stage=stage,
        log_dir=run_dir / "logs",
        console=console,
    )


def read_log_events(path: Path) -> list[LogEvent]:
    events: list[LogEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict) or payload.get("schema_version") != LOG_EVENT_SCHEMA:
                raise InvariantViolation(
                    f"{path}:{line_number} is not a {LOG_EVENT_SCHEMA} record",
                    hint="Log files must contain only schema-versioned log events.",
                )
            events.append(cast(LogEvent, payload))
    return events
