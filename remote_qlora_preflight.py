"""Paid, diagnostic-only Modal preflight for the Hebrew-v3 QLoRA shape.

This entrypoint never reads experiment datasets and never calls a model
provider. It downloads only the pinned base/tokenizer checkpoint, generates a
tiny synthetic English+Hebrew formatted set, then exercises one real optimizer
step and one evaluation forward at the full 4096-token, batch-4 shape.

Future command (do not treat its output as release evidence):

    uv run modal run --detach remote_qlora_preflight.py \
      --config examples/config.v3-he-full.yaml \
      --run-id he-v3-l40s-shape-001
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Final

import modal

from sommelier.remote.images import PIPELINE_HF_ENV, train_image
from sommelier.training.qlora_preflight import (
    GPU_ALLOCATION,
    SourceProvenance,
    run_qlora_shape_preflight,
    validate_preflight_config,
    validate_run_id,
)

APP_NAME: Final = "sommelier-qlora-shape-preflight"
PREFLIGHT_TIMEOUT_SECONDS: Final = 4 * 60 * 60
PREFLIGHT_MAX_RETRIES: Final = 0
HF_READ_SECRET_NAME: Final = "huggingface-read-token"
HF_TOKEN_ENV: Final = "HF_TOKEN"
ARTIFACTS_ROOT: Final = Path("/artifacts/diagnostics/qlora-shape-preflight")

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name("sommelier-artifacts", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sommelier-hf-cache", create_if_missing=True)


def _git_output(arguments: list[str]) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        from sommelier.errors import UserInputError

        raise UserInputError(
            "could not measure local Git provenance for the QLoRA preflight",
            hint="Launch from a readable Git checkout with git available on PATH.",
        ) from error


def local_source_provenance() -> SourceProvenance:
    """Measures the exact launcher commit and a content-free dirty-state digest."""
    commit = _git_output(["rev-parse", "HEAD"]).decode("ascii", errors="replace").strip()
    status = _git_output(["status", "--porcelain=v1", "-z", "--untracked-files=normal"])
    return SourceProvenance(
        git_commit=commit,
        working_tree_clean=not bool(status),
        git_status_sha256=hashlib.sha256(status).hexdigest(),
        boundary=(
            "Measured by the local Modal launcher immediately before dispatch. "
            "The status digest records the path/status set without publishing filenames "
            "or diff contents; a dirty checkout is not immutable."
        ),
    )


@app.function(  # type: ignore[untyped-decorator]
    retries=PREFLIGHT_MAX_RETRIES,
    image=train_image(),
    gpu=GPU_ALLOCATION,
    timeout=PREFLIGHT_TIMEOUT_SECONDS,
    secrets=[
        modal.Secret.from_name(
            HF_READ_SECRET_NAME,
            required_keys=[HF_TOKEN_ENV],
        )
    ],
    volumes={
        "/artifacts": artifacts_volume,
        "/hf-cache": hf_cache_volume,
    },
)
def run_remote_qlora_preflight(
    config_yaml: str,
    run_id: str,
    source: SourceProvenance,
) -> dict[str, object]:
    os.environ.update(dict(PIPELINE_HF_ENV))
    os.environ["HF_HOME"] = "/hf-cache"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from sommelier.config import load_config

    run_dir = ARTIFACTS_ROOT / validate_run_id(run_id)
    staging_config = Path("/tmp/qlora-preflight-config.yaml")
    staging_config.write_text(config_yaml, encoding="utf-8")
    config = load_config(staging_config)
    validate_preflight_config(config)
    try:
        result = run_qlora_shape_preflight(
            config,
            config_yaml=config_yaml,
            output_dir=run_dir,
            run_id=run_id,
            source=source,
        )
    finally:
        artifacts_volume.commit()
        hf_cache_volume.commit()
    return {
        **result,
        "report_path": (f"diagnostics/qlora-shape-preflight/{run_id}/preflight_report.json"),
    }


@app.local_entrypoint()  # type: ignore[untyped-decorator]
def main(
    config: str = "examples/config.v3-he-full.yaml",
    run_id: str = "",
) -> None:
    from sommelier.config import load_config

    validate_run_id(run_id)
    config_path = Path(config)
    resolved = load_config(config_path)
    validate_preflight_config(resolved)
    config_yaml = config_path.read_text(encoding="utf-8")
    source = local_source_provenance()
    result = run_remote_qlora_preflight.remote(config_yaml, run_id, source)
    print(json.dumps(result, indent=2, sort_keys=True))
