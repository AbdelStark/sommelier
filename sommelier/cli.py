from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from sommelier.config import load_config, resolve_config_artifact_root, write_resolved_config
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
        "--paired-input",
        action="append",
        default=[],
        metavar="LANG=PATH",
        help="Raw JSONL input for a paired dataset source, e.g. fr=rows.fr.jsonl. "
        "Repeatable. Defaults to <input stem>.<lang>.jsonl next to --input.",
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

    translate_parser = data_subparsers.add_parser(
        "translate",
        help="Translate root-source queries into a paired language dataset.",
    )
    translate_parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Raw JSONL of the root source's sommelier.raw_tool_call_row.v1 records.",
    )
    translate_parser.add_argument("--out", required=True, type=Path)
    translate_parser.add_argument(
        "--target-language",
        choices=["fr", "he"],
        default="fr",
        help="ISO 639-1 target language (default: fr).",
    )
    translate_parser.add_argument(
        "--model-id",
        required=True,
        help=(
            "Hugging Face id for the local translator path; the interface selects "
            "vLLM chat or Transformers seq2seq."
        ),
    )
    translate_parser.add_argument("--model-revision", default="main")
    translate_parser.add_argument("--max-new-tokens", type=int, default=1024)
    translate_parser.add_argument(
        "--output-decoder",
        choices=["standard", "bytelevel_unicode"],
        default="standard",
        help=(
            "Explicit model-output decoder; use bytelevel_unicode only for a declared "
            "ByteLevel tokenizer defect."
        ),
    )
    translate_parser.add_argument(
        "--max-query-chars",
        type=int,
        default=2000,
        help="Reject translations longer than this, mirroring data.max_query_chars.",
    )
    translate_parser.add_argument(
        "--select-from",
        type=Path,
        help="Prepared data directory; only rows selected into its splits translate.",
    )
    translate_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Translate only the first N selected rows (smoke runs).",
    )
    translate_parser.set_defaults(handler=cmd_data_translate)

    semantic_create_parser = data_subparsers.add_parser(
        "semantic-review-create",
        help="Create the immutable 200-row Hebrew semantic-review template.",
    )
    semantic_create_parser.add_argument("--config", required=True, type=Path)
    semantic_create_parser.add_argument("--root-input", required=True, type=Path)
    semantic_create_parser.add_argument("--paired-input", required=True, type=Path)
    semantic_create_parser.add_argument("--translation-summary", required=True, type=Path)
    semantic_create_parser.add_argument("--out", required=True, type=Path)
    semantic_create_parser.set_defaults(handler=cmd_data_semantic_review_create)

    semantic_finalize_parser = data_subparsers.add_parser(
        "semantic-review-finalize",
        help="Finalize reviewer decisions and bind them into the publication manifest.",
    )
    semantic_finalize_parser.add_argument("--config", required=True, type=Path)
    semantic_finalize_parser.add_argument("--root-input", required=True, type=Path)
    semantic_finalize_parser.add_argument("--paired-input", required=True, type=Path)
    semantic_finalize_parser.add_argument("--translation-summary", required=True, type=Path)
    semantic_finalize_parser.add_argument("--template", required=True, type=Path)
    semantic_finalize_parser.add_argument("--reviewed", required=True, type=Path)
    semantic_finalize_parser.add_argument("--out", required=True, type=Path)
    semantic_finalize_parser.add_argument("--reviewer-id", required=True)
    semantic_finalize_parser.add_argument(
        "--publication-manifest",
        type=Path,
        help="Output manifest path (default: translation_publication.json beside --out).",
    )
    semantic_finalize_parser.set_defaults(handler=cmd_data_semantic_review_finalize)

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

    analyze_parser = subparsers.add_parser("analyze", help="Analysis commands.")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_command", required=True)
    tokenizer_parser = analyze_subparsers.add_parser(
        "tokenization",
        help="Measure per-language and paired tokenizer cost on formatted splits.",
    )
    tokenizer_parser.add_argument("--config", required=True, type=Path)
    tokenizer_parser.add_argument("--data", required=True, type=Path)
    tokenizer_parser.add_argument("--out", required=True, type=Path)
    tokenizer_parser.add_argument("--run-id", type=str, default=None)
    tokenizer_parser.set_defaults(handler=cmd_analyze_tokenization)

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
        type=str,
        help="Adapter directory or Hugging Face repo id; required with --model adapter.",
    )
    eval_run_parser.add_argument(
        "--adapter-revision",
        type=str,
        default=None,
        help="Revision when --adapter is a Hugging Face repo id.",
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

    report_experiment_parser = report_subparsers.add_parser(
        "experiment",
        help="Gate a base/v1-English/v3-English-Hebrew experiment.",
    )
    report_experiment_parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="Base-model evaluation directory.",
    )
    report_experiment_parser.add_argument(
        "--v1-en",
        required=True,
        type=Path,
        help="v1 English-adapter evaluation directory.",
    )
    report_experiment_parser.add_argument(
        "--v3-en-he",
        required=True,
        type=Path,
        help="v3 English-Hebrew adapter evaluation directory.",
    )
    report_experiment_parser.add_argument(
        "--english-non-inferiority-margin",
        required=True,
        type=float,
        help="Predeclared tolerated absolute English full-call regression.",
    )
    report_experiment_parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Deterministic paired-bootstrap seed.",
    )
    report_experiment_parser.add_argument(
        "--resamples",
        required=True,
        type=int,
        help="Number of paired-bootstrap resamples.",
    )
    report_experiment_parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory for experiment_report.json.",
    )
    report_experiment_parser.set_defaults(handler=cmd_report_experiment)

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
    pipeline_run_parser.add_argument(
        "--adapter-id",
        type=str,
        default=None,
        help="Evaluate this published adapter (local dir or Hugging Face repo id) "
        "instead of training one; the train stage is skipped.",
    )
    pipeline_run_parser.add_argument(
        "--adapter-revision",
        type=str,
        default=None,
        help="Revision when --adapter-id is a Hugging Face repo id.",
    )
    pipeline_run_parser.set_defaults(handler=cmd_pipeline_run)

    release_parser = subparsers.add_parser("release", help="Release gate commands.")
    release_subparsers = release_parser.add_subparsers(dest="release_command", required=True)

    preflight_parser = release_subparsers.add_parser(
        "preflight",
        help="Run the license and artifact release preflight.",
    )
    preflight_parser.add_argument("--config", required=True, type=Path)
    preflight_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Explicit artifact tree to scan; defaults to project.artifact_root "
            "relative to the config file."
        ),
    )
    preflight_parser.set_defaults(handler=cmd_release_preflight)

    publish_dataset_parser = release_subparsers.add_parser(
        "publish-dataset",
        help="Validate or explicitly publish the audited Hebrew dataset bundle.",
    )
    publish_dataset_parser.add_argument("--config", required=True, type=Path)
    publish_dataset_parser.add_argument("--bundle", required=True, type=Path)
    publish_dataset_parser.add_argument("--root-input", required=True, type=Path)
    _add_publication_arguments(publish_dataset_parser)
    publish_dataset_parser.set_defaults(handler=cmd_release_publish_dataset)

    publish_adapter_parser = release_subparsers.add_parser(
        "publish-adapter",
        help="Validate or explicitly publish the evidence-bound Hebrew v3 adapter bundle.",
    )
    publish_adapter_parser.add_argument("--bundle", required=True, type=Path)
    _add_publication_arguments(publish_adapter_parser)
    publish_adapter_parser.set_defaults(handler=cmd_release_publish_adapter)

    serve_parser = subparsers.add_parser("serve", help="Optional serving commands.")
    serve_subparsers = serve_parser.add_subparsers(dest="serve_command", required=True)

    serve_adapter_parser = serve_subparsers.add_parser(
        "adapter",
        help="Serve a trained adapter behind an OpenAI-compatible endpoint.",
    )
    serve_adapter_parser.add_argument("--config", required=True, type=Path)
    serve_adapter_parser.add_argument("--adapter", required=True, type=Path)
    serve_adapter_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_adapter_parser.add_argument("--port", type=int, default=8000)
    serve_adapter_parser.set_defaults(handler=cmd_serve_adapter)

    return parser


def _add_publication_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-id", required=True, help="Exact Hugging Face namespace/name.")
    parser.add_argument(
        "--commit-message",
        required=True,
        help="One-line commit message recorded in the publication receipt.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the Hub mutation; without this flag the command only validates.",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create a new public repository with exist_ok=false before the first commit.",
    )
    parser.add_argument(
        "--confirm-repo-id",
        default=None,
        help="Must exactly repeat --repo-id when --execute is used.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="New local JSON receipt path; required with --execute and never overwritten.",
    )


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
        paired_input_paths: dict[str, Path] = {}
        for entry in args.paired_input:
            language, separator, raw_path = entry.partition("=")
            if not separator or not language or not raw_path:
                raise UserInputError(
                    f"invalid --paired-input value: {entry!r}",
                    hint="Use the form LANG=PATH, e.g. fr=rows.fr.jsonl.",
                )
            command.extend(["--paired-input", entry])
            paired_input_paths[language] = Path(raw_path).resolve()
        prepare_dataset_from_file(
            config,
            input_path=input_path.resolve(),
            out_dir=args.out.resolve(),
            context=context,
            command=command,
            use_gpu=args.gpu,
            paired_input_paths=paired_input_paths,
        )
    print(f"data prepare ok: run_id={context.run_id} out={args.out}")
    return 0


def cmd_data_translate(args: argparse.Namespace) -> int:
    from sommelier.artifacts import sha256_file
    from sommelier.data.load import load_raw_rows
    from sommelier.data.translate import (
        TranslatorInfo,
        load_vllm_translator,
        progress_filename,
        select_example_ids,
        translate_rows,
        write_translation_outputs,
    )
    from sommelier.manifests import get_git_commit

    rows = load_raw_rows(args.input.resolve())
    input_description = f"{args.input} ({len(rows)} rows)"
    if args.select_from is not None:
        selected_ids = select_example_ids(args.select_from.resolve())
        rows = [row for row in rows if row["source_id"] in selected_ids]
        input_description += f", selected {len(rows)}"
    if args.limit:
        rows = rows[: args.limit]

    translator = TranslatorInfo(
        model_id=args.model_id,
        model_revision=args.model_revision,
        max_new_tokens=args.max_new_tokens,
        output_decoder=args.output_decoder,
        implementation_revision=get_git_commit(),
    )
    model = load_vllm_translator(translator)
    out_dir = args.out.resolve()
    translated, stats = translate_rows(
        rows,
        model,
        progress_path=out_dir / progress_filename(args.target_language),
        max_query_chars=args.max_query_chars,
        target_language=args.target_language,
        translator=translator,
    )
    rows_path, summary_path = write_translation_outputs(
        out_dir,
        translated,
        stats,
        translator=translator,
        input_description=input_description,
        target_language=args.target_language,
        input_sha256=sha256_file(args.input.resolve()),
    )
    print(f"data translate ok: rows={rows_path} summary={summary_path}")
    return 0


def cmd_data_semantic_review_create(args: argparse.Namespace) -> int:
    import json
    import platform

    import torch

    from sommelier.data.load import load_raw_rows
    from sommelier.data.semantic_review import (
        capture_producer_provenance,
        create_semantic_review_template,
        load_transformers_backtranslator,
        root_split_assignments,
        validate_producer_provenance,
    )
    from sommelier.manifests import get_git_commit, get_git_worktree_clean

    config = load_config(args.config.resolve())
    root_path = args.root_input.resolve()
    root_rows = load_raw_rows(root_path)
    hardware = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else f"{platform.system()}-{platform.machine()}-cpu"
    )
    producer_provenance = capture_producer_provenance(
        code_revision=get_git_commit(),
        working_tree_clean=get_git_worktree_clean(),
        execution_boundary="local",
        provider="local",
        hardware=hardware,
    )
    summary_path = args.translation_summary.resolve()
    validate_producer_provenance(
        producer_provenance,
        translation_summary=json.loads(summary_path.read_text(encoding="utf-8")),
    )
    output = create_semantic_review_template(
        root_rows_path=root_path,
        paired_rows_path=args.paired_input.resolve(),
        translation_summary_path=summary_path,
        root_split_by_id=root_split_assignments(config, root_rows),
        output_path=args.out.resolve(),
        backtranslator=load_transformers_backtranslator(),
        seed=config.project.seed,
        producer_provenance=producer_provenance,
    )
    print(f"semantic review template ok: {output}")
    return 0


def cmd_data_semantic_review_finalize(args: argparse.Namespace) -> int:
    from sommelier.data.load import load_raw_rows
    from sommelier.data.semantic_review import (
        SEMANTIC_REVIEW_FILENAME,
        finalize_semantic_review,
        root_split_assignments,
    )
    from sommelier.data.translate import (
        PUBLICATION_MANIFEST_FILENAME,
        write_translation_publication_manifest,
    )

    config = load_config(args.config.resolve())
    root_path = args.root_input.resolve()
    paired_path = args.paired_input.resolve()
    summary_path = args.translation_summary.resolve()
    template_path = args.template.resolve()
    root_rows = load_raw_rows(root_path)
    output_path = args.out.resolve()
    if output_path.name != SEMANTIC_REVIEW_FILENAME:
        raise UserInputError(f"final semantic review must be named {SEMANTIC_REVIEW_FILENAME!r}")
    final = finalize_semantic_review(
        args.reviewed.resolve(),
        output_path,
        template_path=template_path,
        reviewer_id=args.reviewer_id,
        root_rows_path=root_path,
        paired_rows_path=paired_path,
        translation_summary_path=summary_path,
        root_split_by_id=root_split_assignments(config, root_rows),
        expected_seed=config.project.seed,
    )
    publication_path = (
        args.publication_manifest.resolve()
        if args.publication_manifest is not None
        else output_path.parent / PUBLICATION_MANIFEST_FILENAME
    )
    write_translation_publication_manifest(
        publication_path,
        translated_rows_path=paired_path,
        summary_path=summary_path,
        target_language="he",
        semantic_review_path=final,
        semantic_review_template_path=template_path,
    )
    print(f"semantic review final ok: review={final} publication={publication_path}")
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


def cmd_analyze_tokenization(args: argparse.Namespace) -> int:
    from sommelier.analysis.tokenization import analyze_tokenizer_tax

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
        "analyze",
        "tokenization",
        "--config",
        str(args.config),
        "--data",
        str(args.data),
        "--out",
        str(args.out),
    ]
    if run_id is not None:
        command.extend(["--run-id", run_id])
    analyze_tokenizer_tax(
        config,
        formatted_dir=args.data.resolve(),
        out_dir=args.out.resolve(),
        context=context,
        command=command,
    )
    print(f"tokenization analysis ok: run_id={context.run_id} out={args.out}")
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

    from sommelier.evaluation.generate import AdapterRef, run_generation
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
    if args.adapter_revision is not None:
        command.extend(["--adapter-revision", args.adapter_revision])
    if run_id is not None:
        command.extend(["--run-id", run_id])
    adapter = None
    if args.adapter is not None:
        adapter_path = Path(args.adapter)
        adapter = AdapterRef(
            source=str(adapter_path.resolve()) if adapter_path.exists() else args.adapter,
            revision=args.adapter_revision,
        )
    run_generation(
        config,
        formatted_dir=args.data.resolve(),
        out_dir=args.out.resolve(),
        model_kind=args.model,
        context=context,
        command=command,
        adapter=adapter,
    )
    write_evaluation_report(
        config,
        formatted_dir=args.data.resolve(),
        eval_dir=args.out.resolve(),
        model_kind=args.model,
        context=context,
        command=command,
        adapter=adapter,
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


def cmd_report_experiment(args: argparse.Namespace) -> int:
    from sommelier.evaluation.experiment import write_experiment_report

    write_experiment_report(
        args.base.resolve(),
        args.v1_en.resolve(),
        args.v3_en_he.resolve(),
        args.out.resolve(),
        english_non_inferiority_margin=args.english_non_inferiority_margin,
        seed=args.seed,
        resamples=args.resamples,
    )
    print(f"report experiment ok: out={args.out}")
    return 0


def cmd_pipeline_run(args: argparse.Namespace) -> int:
    from sommelier.pipeline import run_pipeline

    run_id = run_pipeline(
        args.config,
        mode=args.mode,
        input_path=args.input,
        run_id=args.run_id,
        project_root=Path.cwd(),
        adapter_id=args.adapter_id,
        adapter_revision=args.adapter_revision,
    )
    print(f"pipeline run ok: mode={args.mode} run_id={run_id}")
    return 0


def cmd_release_preflight(args: argparse.Namespace) -> int:
    from sommelier.release import PREFLIGHT_FILENAME, run_release_preflight

    config = load_config(args.config)
    config_dir = args.config.resolve().parent
    artifact_root = (
        args.artifact_root.absolute()
        if args.artifact_root is not None
        else resolve_config_artifact_root(config, config_dir=config_dir)
    )
    run_release_preflight(
        config,
        project_root=Path.cwd(),
        artifact_root=artifact_root,
    )
    print(f"release preflight ok: {artifact_root / PREFLIGHT_FILENAME}")
    return 0


def cmd_release_publish_dataset(args: argparse.Namespace) -> int:
    from sommelier.publication import publish_hebrew_dataset_bundle

    result = publish_hebrew_dataset_bundle(
        config_path=args.config.resolve(),
        bundle_dir=args.bundle.resolve(),
        root_rows_path=args.root_input.resolve(),
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        execute=args.execute,
        create_repo=args.create_repo,
        confirmed_repo_id=args.confirm_repo_id,
        receipt_path=args.receipt.resolve() if args.receipt is not None else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_release_publish_adapter(args: argparse.Namespace) -> int:
    from sommelier.publication import publish_hebrew_adapter_bundle

    result = publish_hebrew_adapter_bundle(
        bundle_dir=args.bundle.resolve(),
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        execute=args.execute,
        create_repo=args.create_repo,
        confirmed_repo_id=args.confirm_repo_id,
        receipt_path=args.receipt.resolve() if args.receipt is not None else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_serve_adapter(args: argparse.Namespace) -> int:
    from sommelier.serving.openai_compat import serve_adapter

    config = load_config(args.config)
    serve_adapter(
        config,
        args.adapter.resolve(),
        host=args.host,
        port=args.port,
    )
    return 0


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
