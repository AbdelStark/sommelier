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
    ["pipeline", "run", "--config", "config.yaml", "--mode", "smoke"],
    ["pipeline", "run", "--config", "config.yaml", "--mode", "full"],
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
    for group in ("config", "data", "format", "eval", "train", "report", "pipeline", "serve"):
        assert group in help_text


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "--help"],
        ["train", "--help"],
        ["report", "--help"],
        ["pipeline", "--help"],
        ["serve", "--help"],
    ],
    ids=lambda argv: argv[0],
)
def test_subcommand_help_exits_zero(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(argv)
    assert excinfo.value.code == 0


@pytest.mark.parametrize(
    ("argv", "issue"),
    [
        (SPEC_COMMANDS[10], 40),
    ],
    ids=("serve",),
)
def test_pending_commands_fail_explicitly(
    argv: list[str], issue: int, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(argv)
    captured = capsys.readouterr()
    assert exit_code == 5
    assert "not implemented yet" in captured.err
    assert f"#{issue}" in captured.err


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
