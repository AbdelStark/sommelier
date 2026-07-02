from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from sommelier.config import SommelierConfig, TrackingConfig, load_config
from sommelier.data.prepare import prepare_dataset_fixture
from sommelier.errors import ExternalDependencyError
from sommelier.evaluation.generate import DecodingConfig, run_generation
from sommelier.evaluation.report import write_evaluation_report
from sommelier.formatting.chat import build_formatted_splits_fixture
from sommelier.run_context import RunContext, ensure_run_context
from sommelier.tracking import track_stage_metrics

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float | int], int | None]] = []
        self.finished = False

    @property
    def url(self) -> str:
        return "https://tracker.example/runs/fake-1"

    def log(self, payload: dict[str, float | int], *, step: int | None = None) -> None:
        self.logged.append((payload, step))

    def finish(self) -> None:
        self.finished = True


class FakeFactory:
    def __init__(self) -> None:
        self.run: FakeRun | None = None
        self.calls: list[tuple[TrackingConfig, str]] = []

    def __call__(self, tracking: TrackingConfig, run_id: str) -> FakeRun:
        self.calls.append((tracking, run_id))
        self.run = FakeRun()
        return self.run


def setup_run(
    tmp_path: Path, *, tracking_enabled: bool
) -> tuple[SommelierConfig, RunContext, Path]:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 2
    raw["data"]["n_validation"] = 1
    raw["data"]["n_test"] = 2
    if tracking_enabled:
        raw["tracking"] = {"enabled": True, "provider": "wandb", "project": "sommelier-test"}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id="tracking-test",
        project_root=tmp_path,
    )
    data_dir = context.run_dir / "data"
    formatted_dir = context.run_dir / "formatted"
    prepare_dataset_fixture(config, out_dir=data_dir, context=context, command=["test"])
    build_formatted_splits_fixture(
        config,
        data_dir=data_dir,
        out_dir=formatted_dir,
        context=context,
        command=["test"],
    )
    return config, context, formatted_dir


class EchoGenerator:
    def generate(self, prompt_text: str, *, decoding: DecodingConfig) -> str:
        return '{"arguments":{"city":"Paris"},"name":"lookup_weather"}'


def evaluate(config: SommelierConfig, context: RunContext, formatted_dir: Path) -> Path:
    eval_dir = context.run_dir / "eval" / "base"
    run_generation(
        config,
        formatted_dir=formatted_dir,
        out_dir=eval_dir,
        model_kind="base",
        context=context,
        command=["test"],
        generator=EchoGenerator(),
    )
    write_evaluation_report(
        config,
        formatted_dir=formatted_dir,
        eval_dir=eval_dir,
        model_kind="base",
        context=context,
        command=["test"],
    )
    return eval_dir


def test_disabled_tracking_is_a_strict_noop(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path, tracking_enabled=False)
    factory = FakeFactory()

    url = track_stage_metrics(
        config,
        context,
        stage="train",
        records=[{"step": 1, "train_loss": 1.0}],
        factory=factory,
    )

    assert url is None
    assert factory.calls == []
    manifest = json.loads((context.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "tracking" not in manifest


def test_disabled_tracking_keeps_local_reports_complete(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path, tracking_enabled=False)
    eval_dir = evaluate(config, context, formatted_dir)

    assert (eval_dir / "generations.jsonl").exists()
    assert (eval_dir / "evaluation_report.json").exists()
    manifest = json.loads((context.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "tracking" not in manifest


def test_enabled_tracking_logs_and_records_url(tmp_path: Path) -> None:
    config, context, _ = setup_run(tmp_path, tracking_enabled=True)
    factory = FakeFactory()

    url = track_stage_metrics(
        config,
        context,
        stage="train",
        records=[
            {"step": 1, "train_loss": 2.0, "note": "text ignored", "flag": True},
            {"step": 2, "train_loss": 1.5, "eval_loss": None},
        ],
        factory=factory,
    )

    assert url == "https://tracker.example/runs/fake-1"
    assert factory.calls[0][0].project == "sommelier-test"
    assert factory.calls[0][1] == "tracking-test"
    run = factory.run
    assert run is not None
    assert run.finished
    assert run.logged == [
        ({"train/step": 1, "train/train_loss": 2.0}, 1),
        ({"train/step": 2, "train/train_loss": 1.5}, 2),
    ]

    manifest = json.loads((context.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tracking"] == {
        "provider": "wandb",
        "project": "sommelier-test",
        "run_url": "https://tracker.example/runs/fake-1",
    }


def test_tracking_survives_later_stage_updates(tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path, tracking_enabled=True)
    factory = FakeFactory()
    track_stage_metrics(
        config,
        context,
        stage="train",
        records=[{"step": 1, "train_loss": 1.0}],
        factory=factory,
    )

    import sommelier.tracking as tracking_module

    original = tracking_module.default_tracker_factory
    tracking_module.default_tracker_factory = FakeFactory()
    try:
        evaluate(config, context, formatted_dir)
    finally:
        tracking_module.default_tracker_factory = original

    manifest = json.loads((context.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tracking"]["run_url"] == "https://tracker.example/runs/fake-1"
    assert "eval" in manifest["stages"]


@pytest.mark.skipif(
    importlib.util.find_spec("wandb") is not None,
    reason="wandb installed; missing-dependency path not reachable",
)
def test_enabled_tracking_without_wandb_fails_actionably(tmp_path: Path) -> None:
    config, context, _ = setup_run(tmp_path, tracking_enabled=True)
    with pytest.raises(ExternalDependencyError):
        track_stage_metrics(
            config,
            context,
            stage="train",
            records=[{"step": 1, "train_loss": 1.0}],
        )
