from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sommelier.config import SommelierConfig, load_config
from sommelier.data.prepare import prepare_dataset_fixture
from sommelier.errors import ResourceError, SchemaValidationError
from sommelier.formatting.chat import build_formatted_splits_fixture
from sommelier.run_context import RunContext, ensure_run_context
from sommelier.training.metrics import TrainingResult
from sommelier.training.qlora import map_resource_failure, train_adapter

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


class OutOfMemoryError(RuntimeError):
    """Named like torch.cuda.OutOfMemoryError for detection tests."""


class FailingTrainer:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0

    def train(
        self,
        train_examples: list[dict[str, object]],
        validation_examples: list[dict[str, object]],
        adapter_dir: Path,
    ) -> TrainingResult:
        self.calls += 1
        raise self.error


def setup_run(tmp_path: Path) -> tuple[SommelierConfig, RunContext, Path]:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 2
    raw["data"]["n_validation"] = 1
    raw["data"]["n_test"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)
    context = ensure_run_context(
        config,
        config_path=config_path,
        run_id="error-test",
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


def run_with(trainer: FailingTrainer, tmp_path: Path) -> None:
    config, context, formatted_dir = setup_run(tmp_path)
    train_adapter(
        config,
        formatted_dir,
        context.run_dir / "train" / "adapter",
        context=context,
        command=["test"],
        trainer=trainer,
    )


@pytest.mark.parametrize(
    "error",
    [
        OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB"),
        RuntimeError("CUDA error: out of memory"),
    ],
    ids=["torch_oom_class", "runtime_message"],
)
def test_oom_maps_to_resource_error_with_hints(tmp_path: Path, error: BaseException) -> None:
    trainer = FailingTrainer(error)
    with pytest.raises(ResourceError) as excinfo:
        run_with(trainer, tmp_path)

    assert excinfo.value.exit_code == 4
    hint = excinfo.value.hint or ""
    assert "train.per_device_batch_size=" in hint
    assert "train.max_sequence_length=" in hint
    assert "remote.gpu=" in hint
    assert "does not change these values automatically" in hint
    assert excinfo.value.__cause__ is error
    assert trainer.calls == 1  # no silent retry


@pytest.mark.parametrize(
    "error",
    [TimeoutError("job deadline reached"), RuntimeError("operation timed out")],
    ids=["timeout_class", "timeout_message"],
)
def test_timeout_maps_to_resource_error(tmp_path: Path, error: BaseException) -> None:
    trainer = FailingTrainer(error)
    with pytest.raises(ResourceError) as excinfo:
        run_with(trainer, tmp_path)

    assert excinfo.value.exit_code == 4
    hint = excinfo.value.hint or ""
    assert "remote.train_timeout_seconds=" in hint
    assert "planning estimate, not an enforced training watchdog" in hint
    assert trainer.calls == 1


def test_unrelated_errors_propagate_unchanged(tmp_path: Path) -> None:
    error = ValueError("bad tensor shape")
    with pytest.raises(ValueError, match="bad tensor shape"):
        run_with(FailingTrainer(error), tmp_path)


def test_sommelier_errors_pass_through_unmapped(tmp_path: Path) -> None:
    error = SchemaValidationError("boundary cannot be proven")
    with pytest.raises(SchemaValidationError):
        run_with(FailingTrainer(error), tmp_path)


def test_map_resource_failure_returns_none_for_unknown(tmp_path: Path) -> None:
    config, _, _ = setup_run(tmp_path)
    assert map_resource_failure(ValueError("x"), config) is None
