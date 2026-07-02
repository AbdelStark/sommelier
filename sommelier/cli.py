from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from sommelier.config import load_config, write_resolved_config
from sommelier.data.prepare import (
    prepare_dataset_fixture,
    prepare_dataset_from_file,
    validate_fixture_files,
)
from sommelier.errors import SommelierError, UserInputError, format_cli_error
from sommelier.formatting.chat import (
    build_formatted_splits,
    build_formatted_splits_fixture,
)
from sommelier.run_context import ensure_run_context, infer_run_id_from_path


def _not_implemented(command: str, issue: int) -> SommelierError:
    return SommelierError(
        f"command not implemented yet: {command}",
        hint=f"Implementation is tracked in issue #{issue}.",
    )


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
    validate_parser.set_defaults(handler=cmd_config_validate)

    data_parser = subparsers.add_parser("data", help="Dataset preparation commands.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    prepare_parser = data_subparsers.add_parser(
        "prepare",
        help="Prepare dataset splits from raw JSONL rows.",
    )
    prepare_parser.add_argument("--config", required=True, type=Path)
    prepare_parser.add_argument("--out", required=True, type=Path)
    prepare_parser.add_argument("--run-id", type=str, default=None)
    prepare_parser.add_argument(
        "--input",
        type=Path,
        help="Raw JSONL input with sommelier.raw_tool_call_row.v1 records.",
    )
    prepare_parser.add_argument(
        "--fixture",
        action="store_true",
        help="Use synthetic fixture rows instead of real validation and splitting.",
    )
    prepare_parser.add_argument(
        "--gpu",
        action="store_true",
        help="Apply GPU dataframe coarse filtering before Python validation.",
    )
    prepare_parser.set_defaults(handler=cmd_data_prepare)

    validate_fixtures_parser = data_subparsers.add_parser(
        "validate-fixtures",
        help="Validate synthetic fixture JSONL files.",
    )
    validate_fixtures_parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path("tests/fixtures"),
    )
    validate_fixtures_parser.set_defaults(handler=cmd_data_validate_fixtures)

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
    build_parser_cmd.add_argument(
        "--fixture",
        action="store_true",
        help="Build without a tokenizer using the fixture template policy.",
    )
    build_parser_cmd.set_defaults(handler=cmd_format_build)

    eval_parser = subparsers.add_parser("eval", help="Evaluation commands.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_run_parser = eval_subparsers.add_parser(
        "run",
        help="Evaluate the base model or an adapter on the test split.",
    )
    eval_run_parser.add_argument("--config", required=True, type=Path)
    eval_run_parser.add_argument("--model", required=True, choices=["base", "adapter"])
    eval_run_parser.add_argument("--data", required=True, type=Path)
    eval_run_parser.add_argument("--out", required=True, type=Path)
    eval_run_parser.add_argument(
        "--adapter",
        type=Path,
        help="Adapter directory; required with --model adapter.",
    )
    eval_run_parser.add_argument("--run-id", type=str, default=None)
    eval_run_parser.set_defaults(handler=cmd_eval_run)

    train_parser = subparsers.add_parser("train", help="Training commands.")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)

    train_run_parser = train_subparsers.add_parser(
        "run",
        help="Train a parameter-efficient adapter on the formatted train split.",
    )
    train_run_parser.add_argument("--config", required=True, type=Path)
    train_run_parser.add_argument("--data", required=True, type=Path)
    train_run_parser.add_argument("--out", required=True, type=Path)
    train_run_parser.add_argument("--run-id", type=str, default=None)
    train_run_parser.set_defaults(handler=cmd_train_run)

    report_parser = subparsers.add_parser("report", help="Reporting commands.")
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)

    report_compare_parser = report_subparsers.add_parser(
        "compare",
        help="Compare base and adapter evaluation reports.",
    )
    report_compare_parser.add_argument("--base", required=True, type=Path)
    report_compare_parser.add_argument("--adapter", required=True, type=Path)
    report_compare_parser.add_argument("--out", required=True, type=Path)
    report_compare_parser.set_defaults(handler=cmd_report_compare)

    pipeline_parser = subparsers.add_parser("pipeline", help="Pipeline commands.")
    pipeline_subparsers = pipeline_parser.add_subparsers(dest="pipeline_command", required=True)

    pipeline_run_parser = pipeline_subparsers.add_parser(
        "run",
        help="Run the staged pipeline end to end.",
    )
    pipeline_run_parser.add_argument("--config", required=True, type=Path)
    pipeline_run_parser.add_argument("--mode", required=True, choices=["smoke", "full"])
    pipeline_run_parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/fixtures/preparation_rows.jsonl"),
        help="Raw JSONL input with sommelier.raw_tool_call_row.v1 records.",
    )
    pipeline_run_parser.add_argument("--run-id", type=str, default=None)
    pipeline_run_parser.set_defaults(handler=cmd_pipeline_run)

    serve_parser = subparsers.add_parser("serve", help="Optional serving commands.")
    serve_subparsers = serve_parser.add_subparsers(dest="serve_command", required=True)

    serve_adapter_parser = serve_subparsers.add_parser(
        "adapter",
        help="Serve a trained adapter behind an OpenAI-compatible endpoint.",
    )
    serve_adapter_parser.add_argument("--config", required=True, type=Path)
    serve_adapter_parser.add_argument("--adapter", required=True, type=Path)
    serve_adapter_parser.set_defaults(handler=cmd_serve_adapter)

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
    if args.fixture:
        command.append("--fixture")
        prepare_dataset_fixture(
            config,
            out_dir=args.out.resolve(),
            context=context,
            command=command,
        )
    else:
        input_path = args.input or Path("tests/fixtures/preparation_rows.jsonl")
        command.extend(["--input", str(input_path)])
        if args.gpu:
            command.append("--gpu")
        prepare_dataset_from_file(
            config,
            input_path=input_path.resolve(),
            out_dir=args.out.resolve(),
            context=context,
            command=command,
            use_gpu=args.gpu,
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
    if args.fixture:
        command.append("--fixture")
        build_formatted_splits_fixture(
            config,
            data_dir=args.data.resolve(),
            out_dir=args.out.resolve(),
            context=context,
            command=command,
        )
    else:
        build_formatted_splits(
            config,
            data_dir=args.data.resolve(),
            out_dir=args.out.resolve(),
            context=context,
            command=command,
        )
    print(f"format build ok: run_id={context.run_id} out={args.out}")
    return 0


def cmd_eval_run(args: argparse.Namespace) -> int:
    if args.model == "adapter" and args.adapter is None:
        raise UserInputError(
            "--adapter is required when --model adapter",
            hint="Pass the trained adapter directory, e.g. artifacts/train/adapter.",
        )
    if args.model == "base" and args.adapter is not None:
        raise UserInputError(
            "--adapter is only valid with --model adapter",
            hint="Drop --adapter or evaluate with --model adapter.",
        )

    from sommelier.evaluation.generate import run_generation
    from sommelier.evaluation.report import write_evaluation_report

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
        "eval",
        "run",
        "--config",
        str(args.config),
        "--model",
        args.model,
        "--data",
        str(args.data),
        "--out",
        str(args.out),
    ]
    if args.adapter is not None:
        command.extend(["--adapter", str(args.adapter)])
    if run_id is not None:
        command.extend(["--run-id", run_id])
    run_generation(
        config,
        formatted_dir=args.data.resolve(),
        out_dir=args.out.resolve(),
        model_kind=args.model,
        context=context,
        command=command,
        adapter_dir=args.adapter.resolve() if args.adapter is not None else None,
    )
    write_evaluation_report(
        config,
        formatted_dir=args.data.resolve(),
        eval_dir=args.out.resolve(),
        model_kind=args.model,
        context=context,
        command=command,
    )
    print(f"eval run ok: run_id={context.run_id} out={args.out}")
    return 0


def cmd_train_run(args: argparse.Namespace) -> int:
    from sommelier.training.qlora import train_adapter

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
        "train",
        "run",
        "--config",
        str(args.config),
        "--data",
        str(args.data),
        "--out",
        str(args.out),
    ]
    if run_id is not None:
        command.extend(["--run-id", run_id])
    train_adapter(
        config,
        args.data.resolve(),
        args.out.resolve(),
        context=context,
        command=command,
    )
    print(f"train run ok: run_id={context.run_id} out={args.out}")
    return 0


def cmd_report_compare(args: argparse.Namespace) -> int:
    from sommelier.evaluation.report import compare_evaluations

    command = [
        "sommelier",
        "report",
        "compare",
        "--base",
        str(args.base),
        "--adapter",
        str(args.adapter),
        "--out",
        str(args.out),
    ]
    compare_evaluations(
        args.base.resolve(),
        args.adapter.resolve(),
        args.out.resolve(),
        command=command,
    )
    print(f"report compare ok: out={args.out}")
    return 0


def cmd_pipeline_run(args: argparse.Namespace) -> int:
    from sommelier.pipeline import run_pipeline

    run_id = run_pipeline(
        args.config,
        mode=args.mode,
        input_path=args.input,
        run_id=args.run_id,
        project_root=Path.cwd(),
    )
    print(f"pipeline run ok: mode={args.mode} run_id={run_id}")
    return 0


def cmd_serve_adapter(args: argparse.Namespace) -> int:
    raise _not_implemented("serve adapter", 40)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        handler = args.handler
        result: int = handler(args)
        return result
    except SommelierError as error:
        print(format_cli_error(error), file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return error.exit_code
    except Exception as error:  # noqa: BLE001 - CLI boundary maps unexpected errors to exit 5
        print(f"sommelier: SOM000: unexpected error: {error}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
