"""Modal entrypoint that runs the sommelier pipeline end to end on a GPU.

Usage:

    uv run modal run remote_pipeline.py --config examples/config.smoke.yaml \
        --mode smoke --max-rows 2500 [--run-id smoke-1]

The remote function writes the config into the artifacts volume, exports
raw rows from the configured Hugging Face dataset, and chains the shared
pipeline stages (data, format, base eval, train, adapter eval, compare)
inside the training image. Artifacts persist on the `sommelier-artifacts`
volume; the Hugging Face cache persists on `sommelier-hf-cache`.

GPU and timeout are read from SOMMELIER_GPU / SOMMELIER_TIMEOUT_SECONDS at
launch time (defaults: A10G, 4 hours). The HF token comes from the local
.env file via a dotenv-backed Modal secret and is never written to
artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from sommelier.remote.images import train_image

APP_NAME = "sommelier-pipeline"

GPU = os.environ.get("SOMMELIER_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("SOMMELIER_TIMEOUT_SECONDS", str(4 * 60 * 60)))

app = modal.App(APP_NAME)

artifacts_volume = modal.Volume.from_name("sommelier-artifacts", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sommelier-hf-cache", create_if_missing=True)


def _export_raw_rows(config_path: Path, rows_path: Path, max_rows: int) -> int:
    """Exports the configured HF dataset as raw_tool_call_row.v1 JSONL."""
    from sommelier.config import load_config
    from sommelier.data.export import export_raw_rows

    config = load_config(config_path)
    return export_raw_rows(
        config.root_dataset,
        rows_path,
        seed=config.project.seed,
        max_rows=max_rows,
    )


def _cleanup_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _audit_sequence_lengths(paths, config) -> None:  # type: ignore[no-untyped-def]
    """Fails fast when any rendered example cannot fit the training budget.

    The completion-only collator refuses (by design) to truncate away every
    target token; catching that here, right after formatting, costs seconds
    instead of failing training after evaluation already spent GPU time.
    """
    import json

    from sommelier.errors import UserInputError
    from sommelier.formatting.templates import load_tokenizer

    tokenizer = load_tokenizer(config)
    max_len = config.train.max_sequence_length
    lengths_by_language: dict[str, list[int]] = {}
    violations_by_language: dict[str, list[str]] = {}
    for split in ("train", "validation"):
        split_path = paths.formatted_dir / f"{split}.jsonl"
        for line in split_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            language = str(record.get("language", "unknown"))
            full_tokens = len(tokenizer.encode(record["full_text"], add_special_tokens=False))
            lengths_by_language.setdefault(language, []).append(full_tokens)
            if full_tokens > max_len:
                # Anything over budget either loses target tokens to
                # truncation (silently corrupting labels) or, when the
                # prompt alone exceeds it, fails the collator outright.
                violations_by_language.setdefault(language, []).append(
                    f"{record['example_id']}: full sequence {full_tokens} tokens "
                    f"> max_sequence_length {max_len}"
                )
    # French (or any added language) typically tokenizes longer than
    # English on this tokenizer, so the budget is verified per language
    # rather than trusted from the English numbers.
    for language in sorted(lengths_by_language):
        lengths = sorted(lengths_by_language[language])
        print(
            f"[pipeline] sequence-length audit [{language}]: "
            f"n={len(lengths)} p50={lengths[len(lengths) // 2]} "
            f"p95={lengths[int(len(lengths) * 0.95)]} max={lengths[-1]} "
            f"budget={max_len}",
            flush=True,
        )
    if violations_by_language:
        offending = sorted(violations_by_language)
        first_language = offending[0]
        total = sum(len(violations) for violations in violations_by_language.values())
        raise UserInputError(
            f"{total} example(s) cannot fit train.max_sequence_length "
            f"{max_len} (languages: {', '.join(offending)}); "
            f"first: {violations_by_language[first_language][0]}",
            hint="Raise train.max_sequence_length above the longest rendered "
            "sequence.",
        )


def _wrapped_stages() -> object:
    """Default stages wrapped with timing, GPU cleanup, and volume commits.

    Chaining stages that each load an 8B model in one process needs the
    previous model's CUDA memory released before the next load; committing
    after every stage makes partial progress visible on the volume. The
    format stage additionally audits rendered sequence lengths so budget
    violations surface before any model loads.
    """
    import time
    from dataclasses import fields

    from sommelier.pipeline import PipelineStages

    base = PipelineStages()

    original_format = base.format

    def format_with_audit(paths, config, context, command):  # type: ignore[no-untyped-def]
        result = original_format(paths, config, context, command)
        _audit_sequence_lengths(paths, config)
        return result

    base.format = format_with_audit

    def wrap(stage_name: str, fn):  # type: ignore[no-untyped-def]
        def inner(paths, config, context, command):  # type: ignore[no-untyped-def]
            print(f"[pipeline] stage {stage_name} starting", flush=True)
            started = time.monotonic()
            outcome = "failed"
            try:
                result = fn(paths, config, context, command)
                outcome = "finished"
                return result
            finally:
                _cleanup_gpu()
                artifacts_volume.commit()
                elapsed = time.monotonic() - started
                print(
                    f"[pipeline] stage {stage_name} {outcome} in {elapsed:.1f}s",
                    flush=True,
                )

        return inner

    kwargs = {
        field.name: wrap(field.name, getattr(base, field.name))
        for field in fields(PipelineStages)
    }
    return PipelineStages(**kwargs)


def _stage_paired_rows(
    config_path: Path,
    rows_path: Path,
    translation_run_id: str | None,
) -> None:
    """Copies translated paired rows next to the exported root rows.

    Prepare finds paired sources through the <input stem>.<lang>.jsonl
    convention; the translation entrypoint writes to the artifacts volume
    under translation/<run_id>/. A config with paired sources requires
    --translation-run-id naming a completed translation run.
    """
    import shutil

    from sommelier.config import load_config
    from sommelier.data.prepare import paired_input_path
    from sommelier.errors import UserInputError

    config = load_config(config_path)
    paired = [source for source in config.datasets if source.source_id_column is not None]
    if not paired:
        return
    if translation_run_id is None:
        raise UserInputError(
            "the config declares paired dataset sources but no translation "
            "run was named",
            hint="Pass --translation-run-id <id> of a completed "
            "remote_translate.py run.",
        )
    for source in paired:
        translated = (
            Path("/artifacts/translation")
            / translation_run_id
            / f"rows.{source.language}.jsonl"
        )
        if not translated.exists():
            raise UserInputError(
                f"translated rows for language {source.language!r} not found: "
                f"{translated}",
                hint="Run remote_translate.py with the same config and mode "
                "first.",
            )
        target = paired_input_path(rows_path, source.language)
        shutil.copy(translated, target)
        print(f"[pipeline] staged paired rows: {target}", flush=True)


def _package_versions() -> dict[str, str]:
    from importlib.metadata import version

    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "trl", "datasets"):
        try:
            versions[package] = version(package)
        except Exception:
            versions[package] = "absent"
    return versions


@app.function(
    image=train_image(),
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_dotenv(Path(__file__).parent)],
    volumes={"/artifacts": artifacts_volume, "/hf-cache": hf_cache_volume},
)
def run_remote_pipeline(
    config_yaml: str,
    mode: str,
    max_rows: int,
    run_id: str | None = None,
    adapter_id: str | None = None,
    adapter_revision: str | None = None,
    translation_run_id: str | None = None,
) -> dict[str, object]:
    os.environ.setdefault("HF_HOME", "/hf-cache")
    # Long-sequence batches fragment the allocator; expandable segments
    # avoid OOM from reserved-but-unallocated blocks (set before torch
    # initializes CUDA inside the stages).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from sommelier.pipeline import run_pipeline

    work = Path("/artifacts")
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / f"config-{mode}.yaml"
    config_path.write_text(config_yaml, encoding="utf-8")

    print(f"[pipeline] exporting raw rows (max_rows={max_rows})", flush=True)
    rows_path = Path("/tmp/raw_rows.jsonl")
    exported = _export_raw_rows(config_path, rows_path, max_rows)
    print(f"[pipeline] exported {exported} raw rows", flush=True)
    hf_cache_volume.commit()

    _stage_paired_rows(config_path, rows_path, translation_run_id)

    try:
        resolved_run_id = run_pipeline(
            config_path,
            mode=mode,  # type: ignore[arg-type]
            input_path=rows_path,
            run_id=run_id,
            project_root=work,
            stages=_wrapped_stages(),  # type: ignore[arg-type]
            adapter_id=adapter_id,
            adapter_revision=adapter_revision,
        )
    finally:
        artifacts_volume.commit()
        hf_cache_volume.commit()

    run_dir = work / "artifacts" / "runs" / resolved_run_id
    comparison = json.loads(
        (run_dir / "report" / "comparison_report.json").read_text(encoding="utf-8")
    )
    runtime = json.loads((run_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    from sommelier.config import load_config

    return {
        "run_id": resolved_run_id,
        # The config value is authoritative; the module-level GPU default
        # is not visible inside the container.
        "gpu": load_config(config_path).remote.gpu,
        "raw_rows": exported,
        "versions": _package_versions(),
        "metrics": {
            "base": comparison["base"]["metrics"],
            "adapter": comparison["adapter"]["metrics"],
            "deltas": comparison["deltas"],
        },
        "stage_seconds": {
            name: value["elapsed_seconds"] for name, value in runtime["stages"].items()
        },
        "report_path": f"runs/{resolved_run_id}/report/comparison_report.md",
    }


@app.local_entrypoint()
def main(
    config: str = "examples/config.smoke.yaml",
    mode: str = "smoke",
    max_rows: int = 2500,
    run_id: str = "",
    adapter_id: str = "",
    adapter_revision: str = "",
    translation_run_id: str = "",
) -> None:
    config_yaml = Path(config).read_text(encoding="utf-8")
    result = run_remote_pipeline.remote(
        config_yaml,
        mode,
        max_rows,
        run_id or None,
        adapter_id or None,
        adapter_revision or None,
        translation_run_id or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
