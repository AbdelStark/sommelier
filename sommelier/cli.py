from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from sommelier.config import load_config, write_resolved_config
from sommelier.data.prepare import prepare_dataset_fixture, validate_fixture_files
from sommelier.errors import SommelierError, format_cli_error
from sommelier.formatting.chat import build_formatted_splits_fixture
from sommelier.run_context import ensure_run_context, infer_run_id_from_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sommelier")
    parser.add_argument("--debug", action="store_true", help="Show stack traces for errors.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="Configuration commands.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)

    validate_parser = config_subparsers.add_parser("validate", help="Validate a config file.")
    validate_parser.add_argument("--config", required=True, type=Path, help="Path to config YAML.")
    validate_parser.add_argument(
        "--write-resolved",
        type=Path,
        help="Optional directory to write config.resolved.yaml.",
    )

    data_parser = subparsers.add_parser("data", help="Dataset preparation commands.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    prepare_parser = data_subparsers.add_parser(
        "prepare",
        help="Prepare dataset splits from fixture rows.",
    )
    prepare_parser.add_argument("--config", required=True, type=Path)
    prepare_parser.add_argument("--out", required=True, type=Path)
    prepare_parser.add_argument("--run-id", type=str, default=None)

    validate_fixtures_parser = data_subparsers.add_parser(
        "validate-fixtures",
        help="Validate synthetic fixture JSONL files.",
    )
    validate_fixtures_parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path("tests/fixtures"),
    )

    format_parser = subparsers.add_parser("format", help="Formatting commands.")
    format_subparsers = format_parser.add_subparsers(dest="format_command", required=True)

    build_parser_cmd = format_subparsers.add_parser(
        "build",
        help="Build formatted splits from prepared data.",
    )
    build_parser_cmd.add_argument("--config", required=True, type=Path)
    build_parser_cmd.add_argument("--data", required=True, type=Path)
    build_parser_cmd.add_argument("--out", required=True, type=Path)
    build_parser_cmd.add_argument("--run-id", type=str, default=None)

    return parser


def cmd_config_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.write_resolved is not None:
        write_resolved_config(config, args.write_resolved)
    print(f"config ok: {args.config}")
    return 0


def cmd_data_prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    context = ensure_run_context(
        config,
        config_path=args.config,
        run_id=args.run_id,
        project_root=Path.cwd(),
    )
    command = ["sommelier", "data", "prepare", "--config", str(args.config), "--out", str(args.out)]
    if args.run_id is not None:
        command.extend(["--run-id", args.run_id])
    prepare_dataset_fixture(
        config,
        out_dir=args.out.resolve(),
        context=context,
        command=command,
    )
    print(f"data prepare ok: run_id={context.run_id} out={args.out}")
    return 0


def cmd_data_validate_fixtures(args: argparse.Namespace) -> int:
    validate_fixture_files(args.fixtures_dir.resolve())
    print(f"fixtures ok: {args.fixtures_dir}")
    return 0


def cmd_format_build(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_id = args.run_id or infer_run_id_from_path(args.data.resolve())
    context = ensure_run_context(
        config,
        config_path=args.config,
        run_id=run_id,
        project_root=Path.cwd(),
    )
    command = [
        "sommelier",
        "format",
        "build",
        "--config",
        str(args.config),
        "--data",
        str(args.data),
        "--out",
        str(args.out),
    ]
    if run_id is not None:
        command.extend(["--run-id", run_id])
    build_formatted_splits_fixture(
        config,
        data_dir=args.data.resolve(),
        out_dir=args.out.resolve(),
        context=context,
        command=command,
    )
    print(f"format build ok: run_id={context.run_id} out={args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "config" and args.config_command == "validate":
            return cmd_config_validate(args)
        if args.command == "data" and args.data_command == "prepare":
            return cmd_data_prepare(args)
        if args.command == "data" and args.data_command == "validate-fixtures":
            return cmd_data_validate_fixtures(args)
        if args.command == "format" and args.format_command == "build":
            return cmd_format_build(args)
        raise SommelierError(f"unsupported command: {args.command}")
    except SommelierError as error:
        print(format_cli_error(error), file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
