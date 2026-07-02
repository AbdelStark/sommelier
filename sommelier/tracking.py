from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from sommelier.config import SommelierConfig, TrackingConfig
from sommelier.errors import ExternalDependencyError
from sommelier.manifests import record_tracking_in_run_manifest
from sommelier.run_context import RunContext

NUMERIC_TYPES = (int, float)


class TrackerRun(Protocol):
    """One external tracker run; implementations wrap providers like wandb."""

    @property
    def url(self) -> str: ...

    def log(self, payload: dict[str, float | int], *, step: int | None = None) -> None: ...

    def finish(self) -> None: ...


TrackerFactory = Callable[[TrackingConfig, str], TrackerRun]


def _wandb_factory(tracking: TrackingConfig, run_id: str) -> TrackerRun:
    """Starts a wandb run; wandb is the optional tracking extra.

    Imported lazily inside the factory so tracking stays optional at both
    install and import time.
    """
    try:
        import wandb
    except ImportError as error:
        raise ExternalDependencyError(
            "experiment tracking is enabled but wandb is not installed",
            hint="Install the tracking extra or set tracking.enabled to false.",
        ) from error

    run = wandb.init(
        project=tracking.project,
        id=run_id,
        resume="allow",
        reinit=True,
    )

    class _WandbRun:
        @property
        def url(self) -> str:
            return str(run.url)

        def log(self, payload: dict[str, float | int], *, step: int | None = None) -> None:
            run.log(payload, step=step)

        def finish(self) -> None:
            run.finish()

    return _WandbRun()


# Module-level default so tests can substitute a fake provider without
# threading a factory through every stage function.
default_tracker_factory: TrackerFactory = _wandb_factory


def _numeric_fields(record: dict[str, object], *, prefix: str) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    for key, value in record.items():
        if isinstance(value, bool) or not isinstance(value, NUMERIC_TYPES):
            continue
        payload[f"{prefix}/{key}"] = value
    return payload


def track_stage_metrics(
    config: SommelierConfig,
    context: RunContext,
    *,
    stage: str,
    records: list[dict[str, object]],
    factory: TrackerFactory | None = None,
) -> str | None:
    """Logs stage metrics to the external tracker when tracking is enabled.

    Disabled tracking is a strict no-op: no provider import, no manifest
    change, and local artifacts are already complete because this runs
    after every local write. When enabled, the tracker run URL is recorded
    in the run manifest (docs/spec/05-observability.md).
    """
    if not config.tracking.enabled:
        return None

    active_factory = factory if factory is not None else default_tracker_factory
    run = active_factory(config.tracking, context.run_id)
    try:
        for index, record in enumerate(records):
            payload = _numeric_fields(record, prefix=stage)
            if not payload:
                continue
            step = record.get("step")
            run.log(payload, step=step if isinstance(step, int) else index)
    finally:
        run.finish()

    record_tracking_in_run_manifest(
        run_dir=context.run_dir,
        tracking={
            "provider": config.tracking.provider,
            "project": config.tracking.project,
            "run_url": run.url,
        },
    )
    return run.url
