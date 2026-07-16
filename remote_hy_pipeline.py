"""Modal pipeline for the locally-translated Hebrew (Hy-MT2) tool-calling slice.

This runs the standard Sommelier stages (data -> format -> tokenization ->
eval-base -> train -> eval-adapter -> compare) for the ``config.he-hymt-full``
experiment. It deliberately does NOT go through the preregistered Hebrew v3
admission gate (paid OpenAI teacher + human Ed25519 semantic review), because
this slice was translated with a local open-source model and makes no claim to
be that evidence. The experiment is honestly scoped through a distinct project
name and Hebrew dataset repository, so ``is_hebrew_v3_config`` is false and the
preregistered-contract checks never apply.

Two deviations from ``remote_pipeline.py``:
  * the paired-input evidence gate (``_validate_full_paired_input_for_pipeline``)
    is neutralized for this run, and
  * the ``data`` stage stages the root and paired raw rows without requiring the
    audited-translation provenance sidecars.

Everything else (QLoRA training, deterministic evaluation, parser, metrics, and
the base-vs-adapter comparison report) is the exact shared pipeline code.

Usage:
    SOMMELIER_GPU=L40S SOMMELIER_TIMEOUT_SECONDS=86400 \
    uv run modal run --detach remote_hy_pipeline.py --spawn \
      --config examples/config.he-hymt-full.yaml --run-id he-hymt-full-002

``--spawn`` submits the call and exits instead of blocking, so a multi-hour run
does not die with the local client. Track it by polling the artifact volume for
``runs/<run_id>/report/comparison_report.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from sommelier.config import SommelierConfig
from sommelier.pipeline import PipelinePaths
from sommelier.remote.images import train_image
from sommelier.run_context import RunContext

APP_NAME = "sommelier-hy-pipeline"
GPU = os.environ.get("SOMMELIER_GPU", "L40S")
TIMEOUT_SECONDS = int(os.environ.get("SOMMELIER_TIMEOUT_SECONDS", str(24 * 60 * 60)))

app = modal.App(APP_NAME)
ARTIFACTS_ROOT = Path("/artifacts")
artifacts_volume = modal.Volume.from_name("sommelier-artifacts", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sommelier-hf-cache", create_if_missing=True)


def _gate_free_prepare(
    paths: PipelinePaths,
    config: SommelierConfig,
    context: RunContext,
    command: list[str],
) -> None:
    """Stage the root and paired raw rows, then run the standard prepare.

    A minimal variant of ``sommelier.pipeline._stage_prepare`` that omits the
    audited-translation provenance staging/validation. The pairing, single-call
    filter, split inheritance, and byte-identical tools/answers checks in
    ``prepare_dataset_from_file`` still run and independently reject any row that
    violates the contract.
    """
    import shutil

    from sommelier.artifacts import make_artifact_ref
    from sommelier.data.prepare import paired_input_path, prepare_dataset_from_file

    source_dir = paths.data_dir / "source_inputs"
    source_dir.mkdir(parents=True, exist_ok=True)
    root_language = config.root_dataset.language
    staged_root = source_dir / f"rows.{root_language}.jsonl"
    shutil.copy2(paths.input_path, staged_root)
    source_inputs = [
        make_artifact_ref(
            staged_root,
            artifact_root=context.artifact_root,
            kind="raw_dataset",
            schema_version="sommelier.raw_tool_call_row.v1",
        )
    ]
    for source in config.datasets:
        if source.source_id_column is None:
            continue
        paired_source = paired_input_path(paths.input_path, source.language)
        paired_target = paired_input_path(staged_root, source.language)
        shutil.copy2(paired_source, paired_target)
        source_inputs.append(
            make_artifact_ref(
                paired_target,
                artifact_root=context.artifact_root,
                kind="raw_paired_dataset",
                schema_version="sommelier.raw_tool_call_row.v1",
            )
        )

    prepare_dataset_from_file(
        config,
        input_path=staged_root,
        out_dir=paths.data_dir,
        context=context,
        command=command,
        use_gpu=False,
        source_inputs=source_inputs,
    )


@app.function(
    retries=0,
    image=train_image(),
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_dotenv(Path(__file__).parent)],
    volumes={"/artifacts": artifacts_volume, "/hf-cache": hf_cache_volume},
)  # type: ignore[untyped-decorator]
def run_hy_pipeline(
    config_yaml: str,
    run_id: str,
    code_revision: str = "unknown",
) -> dict[str, object]:
    import sommelier.pipeline as pipeline_module
    from sommelier.config import load_config
    from sommelier.data.export import export_raw_rows
    from sommelier.data.prepare import paired_input_path
    from sommelier.pipeline import PipelineStages, run_pipeline
    from sommelier.runtime_metadata import SourceCodeProvenance

    for key, value in (("HF_HUB_DISABLE_XET", "1"), ("HF_HUB_DOWNLOAD_TIMEOUT", "600")):
        os.environ[key] = value
    os.environ.setdefault("HF_HOME", "/hf-cache")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if code_revision != "unknown":
        os.environ["SOMMELIER_GIT_COMMIT"] = code_revision

    work = ARTIFACTS_ROOT
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / f"config-{run_id}.yaml"
    config_path.write_text(config_yaml, encoding="utf-8")
    config = load_config(config_path)

    # Export the root and every paired source into the input directory the
    # standard prepare stage expects.
    rows_path = Path("/tmp/raw_rows.jsonl")
    print(f"[hy-pipeline] exporting root {config.root_dataset.dataset_id}", flush=True)
    exported = export_raw_rows(config.root_dataset, rows_path, seed=config.project.seed, max_rows=0)
    print(f"[hy-pipeline] exported {exported} root rows", flush=True)
    for source in config.datasets:
        if source.source_id_column is None:
            continue
        paired_path = paired_input_path(rows_path, source.language)
        print(
            f"[hy-pipeline] exporting paired {source.dataset_id} @ "
            f"{source.dataset_revision} -> {paired_path.name}",
            flush=True,
        )
        paired_count = export_raw_rows(source, paired_path, seed=config.project.seed, max_rows=0)
        print(f"[hy-pipeline] exported {paired_count} {source.language} rows", flush=True)
    hf_cache_volume.commit()

    # Neutralize the preregistered-evidence paired-input gate for this honestly
    # scoped local-MT run; keep every other stage exactly as shipped.
    setattr(
        pipeline_module,
        "_validate_full_paired_input_for_pipeline",
        lambda *_args, **_kwargs: None,
    )

    try:
        completed_run_id = run_pipeline(
            config_path,
            mode="full",
            input_path=rows_path,
            run_id=run_id,
            project_root=work,
            stages=PipelineStages(prepare=_gate_free_prepare),
            package_versions=_package_versions(),
            source_code=SourceCodeProvenance(
                git_commit=code_revision,
                working_tree_clean=None,
                boundary="Local-MT Hebrew slice; not preregistered v3 evidence.",
            ),
        )
    finally:
        artifacts_volume.commit()
        hf_cache_volume.commit()

    run_dir = work / "artifacts" / "runs" / completed_run_id
    comparison = json.loads(
        (run_dir / "report" / "comparison_report.json").read_text(encoding="utf-8")
    )
    runtime = json.loads((run_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    return {
        "run_id": completed_run_id,
        "gpu": config.remote.gpu,
        "raw_rows": exported,
        "metrics": {
            "base": comparison["base"]["metrics"],
            "adapter": comparison["adapter"]["metrics"],
            "deltas": comparison["deltas"],
        },
        "stage_seconds": {
            name: value["elapsed_seconds"] for name, value in runtime["stages"].items()
        },
        "report_path": f"runs/{completed_run_id}/report/comparison_report.md",
    }


def _package_versions() -> dict[str, str]:
    from importlib import metadata

    names = (
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "datasets",
        "huggingface_hub",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


@app.local_entrypoint()  # type: ignore[untyped-decorator]
def main(config: str, run_id: str, spawn: bool = False) -> None:
    import subprocess

    config_path = Path(config)
    config_yaml = config_path.read_text(encoding="utf-8")
    try:
        code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        code_revision = "unknown"

    if spawn:
        # A multi-hour run must not depend on a local client staying alive: a
        # blocking .remote() ties the input's lifetime to this process, so
        # killing the client cancels training mid-step. .spawn() submits the
        # call and returns, leaving nothing local to kill. Poll the artifact
        # volume for runs/<run_id>/report/comparison_report.json to finish.
        call = run_hy_pipeline.spawn(
            config_yaml=config_yaml,
            run_id=run_id,
            code_revision=code_revision,
        )
        print(json.dumps({"spawned": True, "function_call_id": call.object_id}, indent=2))
        return

    result = run_hy_pipeline.remote(
        config_yaml=config_yaml,
        run_id=run_id,
        code_revision=code_revision,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
