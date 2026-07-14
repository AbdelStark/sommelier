from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import yaml

import sommelier.data.translate as translate_module
from sommelier.config import SommelierConfig, load_config
from sommelier.errors import UserInputError
from sommelier.pipeline import (
    PipelinePaths,
    PipelineStages,
    apply_smoke_overrides,
    pipeline_run_id,
    run_pipeline,
)
from sommelier.run_context import RunContext, ensure_run_context

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
FIXTURE_INPUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


class StageRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.paths: PipelinePaths | None = None
        self.config: SommelierConfig | None = None
        self.context: RunContext | None = None

    def stage(self, name: str):  # type: ignore[no-untyped-def]
        def _stage(
            paths: PipelinePaths,
            config: SommelierConfig,
            context: RunContext,
            command: list[str],
        ) -> None:
            self.calls.append(name)
            self.paths = paths
            self.config = config
            self.context = context
            assert command[:3] == ["sommelier", "pipeline", "run"]

        return _stage

    def stages(self) -> PipelineStages:
        return PipelineStages(
            prepare=self.stage("data"),
            format=self.stage("format"),
            tokenization=self.stage("tokenization"),
            eval_base=self.stage("eval-base"),
            train=self.stage("train"),
            eval_adapter=self.stage("eval-adapter"),
            compare=self.stage("compare"),
        )


def write_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load((EXAMPLES_DIR / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["data"]["n_train"] = 5000
    raw["data"]["n_validation"] = 500
    raw["data"]["n_test"] = 500
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


def input_file(tmp_path: Path) -> Path:
    source = Path("tests/fixtures/preparation_rows.jsonl").resolve()
    target = tmp_path / "rows.jsonl"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_pipeline_chains_all_stages_in_order(tmp_path: Path) -> None:
    recorder = StageRecorder()
    run_id = run_pipeline(
        write_config(tmp_path),
        mode="full",
        input_path=input_file(tmp_path),
        run_id="full-1",
        project_root=tmp_path,
        stages=recorder.stages(),
    )

    assert run_id == "full-1"
    assert recorder.calls == [
        "data",
        "format",
        "tokenization",
        "eval-base",
        "train",
        "eval-adapter",
        "compare",
    ]
    manifest = json.loads(
        (tmp_path / "artifacts" / "runs" / "full-1" / "manifest.json").read_text()
    )
    assert manifest["status"] == "succeeded"


def test_pipeline_marks_root_manifest_failed_when_a_stage_raises(tmp_path: Path) -> None:
    recorder = StageRecorder()
    stages = recorder.stages()

    def fail_format(
        paths: PipelinePaths,
        config: SommelierConfig,
        context: RunContext,
        command: list[str],
    ) -> None:
        raise RuntimeError("fixture stage failure")

    stages.format = fail_format
    with pytest.raises(RuntimeError, match="fixture stage failure"):
        run_pipeline(
            write_config(tmp_path),
            mode="full",
            input_path=input_file(tmp_path),
            run_id="failed-1",
            project_root=tmp_path,
            stages=stages,
        )

    manifest = json.loads(
        (tmp_path / "artifacts" / "runs" / "failed-1" / "manifest.json").read_text()
    )
    assert manifest["status"] == "failed"


def test_pipeline_paths_follow_artifact_layout(tmp_path: Path) -> None:
    recorder = StageRecorder()
    run_pipeline(
        write_config(tmp_path),
        mode="full",
        input_path=input_file(tmp_path),
        run_id="layout-1",
        project_root=tmp_path,
        stages=recorder.stages(),
    )

    paths = recorder.paths
    assert paths is not None
    run_dir = tmp_path / "artifacts" / "runs" / "layout-1"
    assert paths.data_dir == run_dir / "data"
    assert paths.formatted_dir == run_dir / "formatted"
    assert paths.tokenization_dir == run_dir / "analysis" / "tokenization"
    assert paths.train_dir == run_dir / "train" / "adapter"
    assert paths.eval_base_dir == run_dir / "eval" / "base"
    assert paths.eval_adapter_dir == run_dir / "eval" / "adapter"
    assert paths.report_dir == run_dir / "report"
    assert (run_dir / "config.resolved.yaml").exists()


def test_smoke_mode_bounds_splits_and_prefixes_run_id(tmp_path: Path) -> None:
    recorder = StageRecorder()
    run_id = run_pipeline(
        write_config(tmp_path),
        mode="smoke",
        input_path=input_file(tmp_path),
        run_id="s1",
        project_root=tmp_path,
        stages=recorder.stages(),
    )

    assert run_id == "smoke-s1"
    assert recorder.config is not None
    assert recorder.config.data.n_train == 100
    assert recorder.config.data.n_validation == 20
    assert recorder.config.data.n_test == 20

    resolved = yaml.safe_load(
        (tmp_path / "artifacts" / "runs" / "smoke-s1" / "config.resolved.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert resolved["data"]["n_train"] == 100


def test_smoke_does_not_shrink_already_small_splits(tmp_path: Path) -> None:
    config = load_config(EXAMPLES_DIR / "config.smoke.yaml")
    config.data.n_train = 8
    config.data.n_validation = 4
    config.data.n_test = 4
    bounded = apply_smoke_overrides(config)
    assert bounded.data.n_train == 8
    assert bounded.data.n_validation == 4
    assert bounded.data.n_test == 4


def test_full_and_smoke_runs_use_separate_directories(tmp_path: Path) -> None:
    recorder = StageRecorder()
    config_path = write_config(tmp_path)
    rows = input_file(tmp_path)

    smoke_id = run_pipeline(
        config_path,
        mode="smoke",
        input_path=rows,
        run_id="r1",
        project_root=tmp_path,
        stages=recorder.stages(),
    )
    full_id = run_pipeline(
        config_path,
        mode="full",
        input_path=rows,
        run_id="r1",
        project_root=tmp_path,
        stages=recorder.stages(),
    )

    assert smoke_id == "smoke-r1"
    assert full_id == "r1"
    assert (tmp_path / "artifacts" / "runs" / "smoke-r1").exists()
    assert (tmp_path / "artifacts" / "runs" / "r1").exists()


def test_full_pipeline_rejects_existing_run_before_rewriting_evidence(
    tmp_path: Path,
) -> None:
    recorder = StageRecorder()
    run_dir = tmp_path / "artifacts" / "runs" / "already-used"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "existing-evidence.json"
    sentinel.write_text('{"attempt": 1}\n', encoding="utf-8")

    with pytest.raises(UserInputError, match="full pipeline run directory already exists"):
        run_pipeline(
            write_config(tmp_path),
            mode="full",
            input_path=input_file(tmp_path),
            run_id="already-used",
            project_root=tmp_path,
            stages=recorder.stages(),
        )

    assert recorder.calls == []
    assert sentinel.read_text(encoding="utf-8") == '{"attempt": 1}\n'
    assert sorted(path.name for path in run_dir.iterdir()) == ["existing-evidence.json"]


def test_fresh_run_reservation_admits_only_one_concurrent_attempt(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    config = load_config(config_path)
    ready = Barrier(2)

    def reserve() -> str | UserInputError:
        ready.wait()
        try:
            return ensure_run_context(
                config,
                config_path=config_path,
                run_id="contended-full-run",
                project_root=tmp_path,
                reject_existing_run=True,
            ).run_id
        except UserInputError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: reserve(), range(2)))

    assert outcomes.count("contended-full-run") == 1
    errors = [outcome for outcome in outcomes if isinstance(outcome, UserInputError)]
    assert len(errors) == 1
    assert "full pipeline run directory already exists" in str(errors[0])


def test_smoke_pipeline_can_rerun_in_its_existing_diagnostic_directory(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path)
    rows = input_file(tmp_path)

    first = run_pipeline(
        config_path,
        mode="smoke",
        input_path=rows,
        run_id="repeatable",
        project_root=tmp_path,
        stages=StageRecorder().stages(),
    )
    second_recorder = StageRecorder()
    second = run_pipeline(
        config_path,
        mode="smoke",
        input_path=rows,
        run_id="repeatable",
        project_root=tmp_path,
        stages=second_recorder.stages(),
    )

    assert first == second == "smoke-repeatable"
    assert second_recorder.calls == [
        "data",
        "format",
        "tokenization",
        "eval-base",
        "train",
        "eval-adapter",
        "compare",
    ]


def test_generated_run_ids_are_prefixed_for_smoke() -> None:
    assert pipeline_run_id("smoke").startswith("smoke-")
    assert not pipeline_run_id("full").startswith("smoke-")
    assert pipeline_run_id("smoke", "smoke-existing") == "smoke-existing"


@pytest.mark.parametrize(
    ("mode", "run_id"),
    [
        ("full", ".."),
        ("smoke", ".."),
        ("full", "nested/run"),
        ("smoke", "nested/run"),
        ("full", r"nested\run"),
        ("smoke", r"nested\run"),
        ("full", "/absolute/run"),
        ("smoke", r"C:\absolute\run"),
        ("smoke", "escape/../victim"),
    ],
)
def test_pipeline_rejects_unsafe_explicit_run_id_before_artifact_mutation(
    tmp_path: Path,
    mode: str,
    run_id: str,
) -> None:
    recorder = StageRecorder()

    with pytest.raises(UserInputError, match="invalid pipeline run id"):
        run_pipeline(
            write_config(tmp_path),
            mode=mode,  # type: ignore[arg-type]
            input_path=input_file(tmp_path),
            run_id=run_id,
            project_root=tmp_path,
            stages=recorder.stages(),
        )

    assert recorder.calls == []
    assert not (tmp_path / "artifacts").exists()


def test_full_pipeline_reserves_a_generated_run_id(tmp_path: Path) -> None:
    run_id = run_pipeline(
        write_config(tmp_path),
        mode="full",
        input_path=input_file(tmp_path),
        project_root=tmp_path,
        stages=StageRecorder().stages(),
    )

    assert not run_id.startswith("smoke-")
    assert (tmp_path / "artifacts" / "runs" / run_id / "manifest.json").exists()


def test_missing_input_fails_before_any_stage(tmp_path: Path) -> None:
    recorder = StageRecorder()
    with pytest.raises(UserInputError):
        run_pipeline(
            write_config(tmp_path),
            mode="full",
            input_path=tmp_path / "missing.jsonl",
            project_root=tmp_path,
            stages=recorder.stages(),
        )
    assert recorder.calls == []


def test_full_paired_trust_gate_runs_before_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = StageRecorder()

    def reject(_config: SommelierConfig, _root_rows_path: Path) -> dict[str, dict[str, Path]]:
        raise UserInputError("full paired-input trust gate rejected fixture")

    monkeypatch.setattr(
        translate_module,
        "validate_full_paired_input_contract",
        reject,
    )
    with pytest.raises(UserInputError, match="trust gate rejected"):
        run_pipeline(
            write_config(tmp_path),
            mode="full",
            input_path=input_file(tmp_path),
            run_id="rejected-before-output",
            project_root=tmp_path,
            stages=recorder.stages(),
        )
    assert recorder.calls == []
    assert not (tmp_path / "artifacts" / "runs" / "rejected-before-output").exists()


def test_smoke_pipeline_is_exempt_from_full_publication_trust_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = StageRecorder()

    def reject(_config: SommelierConfig, _root_rows_path: Path) -> dict[str, dict[str, Path]]:
        raise AssertionError("smoke must not enter the full publication trust gate")

    monkeypatch.setattr(
        translate_module,
        "validate_full_paired_input_contract",
        reject,
    )
    run_id = run_pipeline(
        write_config(tmp_path),
        mode="smoke",
        input_path=input_file(tmp_path),
        run_id="trust-exempt",
        project_root=tmp_path,
        stages=recorder.stages(),
    )
    assert run_id == "smoke-trust-exempt"
    assert recorder.calls


def test_invalid_mode_fails_before_any_stage(tmp_path: Path) -> None:
    recorder = StageRecorder()
    with pytest.raises(UserInputError, match="unsupported pipeline mode"):
        run_pipeline(
            write_config(tmp_path),
            mode="typo",  # type: ignore[arg-type]
            input_path=input_file(tmp_path),
            project_root=tmp_path,
            stages=recorder.stages(),
        )
    assert recorder.calls == []
