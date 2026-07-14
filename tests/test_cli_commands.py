from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from sommelier.cli import build_parser, main

SPEC_COMMANDS: list[list[str]] = [
    ["config", "validate", "--config", "config.yaml"],
    ["data", "prepare", "--config", "config.yaml", "--out", "artifacts/data"],
    ["data", "validate-fixtures"],
    [
        "data",
        "semantic-review-create",
        "--config",
        "config.yaml",
        "--root-input",
        "rows.en.jsonl",
        "--paired-input",
        "rows.he.jsonl",
        "--translation-summary",
        "translation_summary.json",
        "--out",
        "translation_semantic_review_template.json",
    ],
    [
        "data",
        "semantic-review-finalize",
        "--config",
        "config.yaml",
        "--root-input",
        "rows.en.jsonl",
        "--paired-input",
        "rows.he.jsonl",
        "--translation-summary",
        "translation_summary.json",
        "--template",
        "translation_semantic_review_template.json",
        "--reviewed",
        "reviewed.json",
        "--out",
        "translation_semantic_review.json",
        "--reviewer-id",
        "reviewer-1",
    ],
    [
        "format",
        "build",
        "--config",
        "config.yaml",
        "--data",
        "artifacts/data",
        "--out",
        "artifacts/formatted",
    ],
    [
        "eval",
        "run",
        "--config",
        "config.yaml",
        "--model",
        "base",
        "--data",
        "artifacts/formatted",
        "--out",
        "artifacts/eval/base",
    ],
    [
        "eval",
        "run",
        "--config",
        "config.yaml",
        "--model",
        "adapter",
        "--adapter",
        "artifacts/train/adapter",
        "--data",
        "artifacts/formatted",
        "--out",
        "artifacts/eval/adapter",
    ],
    [
        "train",
        "run",
        "--config",
        "config.yaml",
        "--data",
        "artifacts/formatted",
        "--out",
        "artifacts/train/adapter",
    ],
    [
        "report",
        "compare",
        "--base",
        "artifacts/eval/base",
        "--adapter",
        "artifacts/eval/adapter",
        "--out",
        "artifacts/report",
    ],
    [
        "report",
        "experiment",
        "--base",
        "artifacts/runs/base/eval/base",
        "--v1-en",
        "artifacts/runs/v1/eval/adapter",
        "--v3-en-he",
        "artifacts/runs/v3/eval/adapter",
        "--english-non-inferiority-margin",
        "0.01",
        "--seed",
        "42",
        "--resamples",
        "2000",
        "--out",
        "artifacts/experiment",
    ],
    ["pipeline", "run", "--config", "config.yaml", "--mode", "smoke"],
    ["pipeline", "run", "--config", "config.yaml", "--mode", "full"],
    ["release", "preflight", "--config", "config.yaml"],
    [
        "release",
        "publish-dataset",
        "--config",
        "config.yaml",
        "--bundle",
        "dataset-bundle",
        "--root-input",
        "rows.en.jsonl",
        "--repo-id",
        "owner/hebrew-dataset",
        "--commit-message",
        "Publish Hebrew v3 dataset",
    ],
    [
        "release",
        "publish-adapter",
        "--bundle",
        "adapter-bundle",
        "--repo-id",
        "owner/hebrew-adapter",
        "--commit-message",
        "Publish Hebrew v3 adapter",
    ],
    ["serve", "adapter", "--config", "config.yaml", "--adapter", "artifacts/train/adapter"],
]


@pytest.mark.parametrize("argv", SPEC_COMMANDS, ids=lambda argv: " ".join(argv[:2]))
def test_spec_command_shapes_parse(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert isinstance(args, argparse.Namespace)
    assert callable(args.handler)


def test_top_level_help_lists_all_command_groups(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for group in (
        "config",
        "data",
        "format",
        "eval",
        "train",
        "report",
        "pipeline",
        "release",
        "serve",
    ):
        assert group in help_text


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "--help"],
        ["train", "--help"],
        ["report", "--help"],
        ["pipeline", "--help"],
        ["release", "--help"],
        ["serve", "--help"],
    ],
    ids=lambda argv: argv[0],
)
def test_subcommand_help_exits_zero(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(argv)
    assert excinfo.value.code == 0


def test_serve_adapter_requires_existing_adapter_dir(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "serve",
            "adapter",
            "--config",
            "examples/config.smoke.yaml",
            "--adapter",
            "does/not/exist",
        ]
    )
    assert exit_code == 2
    assert "adapter directory not found" in capsys.readouterr().err


def test_eval_adapter_requires_adapter_path(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "eval",
            "run",
            "--config",
            "config.yaml",
            "--model",
            "adapter",
            "--data",
            "d",
            "--out",
            "o",
        ]
    )
    assert exit_code == 2
    assert "--adapter is required" in capsys.readouterr().err


def test_eval_base_rejects_adapter_path(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "eval",
            "run",
            "--config",
            "config.yaml",
            "--model",
            "base",
            "--adapter",
            "a",
            "--data",
            "d",
            "--out",
            "o",
        ]
    )
    assert exit_code == 2
    assert "--adapter is only valid" in capsys.readouterr().err


def test_config_validate_returns_zero() -> None:
    assert main(["config", "validate", "--config", "examples/config.smoke.yaml"]) == 0


def test_release_preflight_accepts_explicit_downloaded_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, object] = {}

    def fake_preflight(
        config: object,
        *,
        project_root: Path,
        artifact_root: Path,
    ) -> dict[str, object]:
        received.update(
            {
                "config": config,
                "project_root": project_root,
                "artifact_root": artifact_root,
            }
        )
        return {}

    monkeypatch.setattr("sommelier.release.run_release_preflight", fake_preflight)
    artifact_root = tmp_path / "downloaded-artifacts"

    exit_code = main(
        [
            "release",
            "preflight",
            "--config",
            "examples/config.smoke.yaml",
            "--artifact-root",
            str(artifact_root),
        ]
    )

    assert exit_code == 0
    assert received["project_root"] == Path.cwd()
    assert received["artifact_root"] == artifact_root.resolve()
    assert received["config"] is not None
    assert str(artifact_root.resolve()) in capsys.readouterr().out


def test_release_preflight_does_not_resolve_explicit_artifact_root_symlink(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    report = outside / "release_preflight.json"
    report.write_text("do not replace\n", encoding="utf-8")
    alias = tmp_path / "explicit-artifacts"
    alias.symlink_to(outside, target_is_directory=True)

    exit_code = main(
        [
            "release",
            "preflight",
            "--config",
            str(Path("examples/config.smoke.yaml").resolve()),
            "--artifact-root",
            str(alias),
        ]
    )

    assert exit_code == 5
    assert "No report was written" in capsys.readouterr().err
    assert report.read_text(encoding="utf-8") == "do not replace\n"


def test_release_publish_dataset_is_validate_only_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, object] = {}

    def fake_publish(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return {"status": "validated", "executed": False}

    monkeypatch.setattr("sommelier.publication.publish_hebrew_dataset_bundle", fake_publish)
    config = tmp_path / "config.yaml"
    bundle = tmp_path / "bundle"
    root = tmp_path / "rows.en.jsonl"

    exit_code = main(
        [
            "release",
            "publish-dataset",
            "--config",
            str(config),
            "--bundle",
            str(bundle),
            "--root-input",
            str(root),
            "--repo-id",
            "owner/hebrew-dataset",
            "--commit-message",
            "Publish Hebrew v3 dataset",
        ]
    )

    assert exit_code == 0
    assert received == {
        "config_path": config.resolve(),
        "bundle_dir": bundle.resolve(),
        "root_rows_path": root.resolve(),
        "repo_id": "owner/hebrew-dataset",
        "commit_message": "Publish Hebrew v3 dataset",
        "execute": False,
        "create_repo": False,
        "confirmed_repo_id": None,
        "receipt_path": None,
    }
    assert '"status": "validated"' in capsys.readouterr().out


def test_release_publish_adapter_forwards_explicit_mutation_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, object] = {}

    def fake_publish(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return {"status": "verified", "executed": True}

    monkeypatch.setattr("sommelier.publication.publish_hebrew_adapter_bundle", fake_publish)
    bundle = tmp_path / "bundle"
    receipt = tmp_path / "receipts" / "adapter.json"

    exit_code = main(
        [
            "release",
            "publish-adapter",
            "--bundle",
            str(bundle),
            "--repo-id",
            "owner/hebrew-adapter",
            "--commit-message",
            "Publish Hebrew v3 adapter",
            "--execute",
            "--create-repo",
            "--confirm-repo-id",
            "owner/hebrew-adapter",
            "--receipt",
            str(receipt),
        ]
    )

    assert exit_code == 0
    assert received == {
        "bundle_dir": bundle.resolve(),
        "repo_id": "owner/hebrew-adapter",
        "commit_message": "Publish Hebrew v3 adapter",
        "execute": True,
        "create_repo": True,
        "confirmed_repo_id": "owner/hebrew-adapter",
        "receipt_path": receipt.resolve(),
    }
    assert '"status": "verified"' in capsys.readouterr().out


def test_report_experiment_forwards_explicit_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    received: dict[str, object] = {}

    def fake_write_experiment_report(
        base_eval_dir: Path,
        v1_en_eval_dir: Path,
        v3_en_he_eval_dir: Path,
        out_dir: Path,
        *,
        english_non_inferiority_margin: float,
        seed: int,
        resamples: int,
    ) -> dict[str, object]:
        received.update(
            {
                "base": base_eval_dir,
                "v1_en": v1_en_eval_dir,
                "v3_en_he": v3_en_he_eval_dir,
                "out": out_dir,
                "margin": english_non_inferiority_margin,
                "seed": seed,
                "resamples": resamples,
            }
        )
        return {}

    monkeypatch.setattr(
        "sommelier.evaluation.experiment.write_experiment_report",
        fake_write_experiment_report,
    )
    base = tmp_path / "base"
    v1 = tmp_path / "v1"
    v3 = tmp_path / "v3"
    out = tmp_path / "out"

    exit_code = main(
        [
            "report",
            "experiment",
            "--base",
            str(base),
            "--v1-en",
            str(v1),
            "--v3-en-he",
            str(v3),
            "--english-non-inferiority-margin",
            "0.01",
            "--seed",
            "42",
            "--resamples",
            "2000",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    assert received == {
        "base": base.resolve(),
        "v1_en": v1.resolve(),
        "v3_en_he": v3.resolve(),
        "out": out.resolve(),
        "margin": 0.01,
        "seed": 42,
        "resamples": 2000,
    }
    assert "report experiment ok" in capsys.readouterr().out


def test_unexpected_errors_map_to_exit_five(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(args: argparse.Namespace) -> int:
        raise ValueError("boom")

    monkeypatch.setattr("sommelier.cli.cmd_config_validate", broken)
    parser_args = ["config", "validate", "--config", "examples/config.smoke.yaml"]
    # build_parser binds the handler at parser construction time, so rebuild
    # through main() which reconstructs the parser after the patch.
    exit_code = main(parser_args)
    assert exit_code == 5
    assert "SOM000: unexpected error" in capsys.readouterr().err


def test_missing_config_maps_to_exit_two(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    assert main(["config", "validate", "--config", str(missing)]) == 2
