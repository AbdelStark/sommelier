"""Modal producer for the preregistered Hebrew semantic-review template.

Usage:

    SOMMELIER_GPU=A10G uv run modal run remote_semantic_review.py \
        --translation-run-id he-v3-translate-full

The named full translation run must already contain ``config.yaml``, the
exported English root rows, accepted Hebrew paired rows, and
``translation_summary.json``.  This job deterministically selects 200 rows,
backtranslates them with the pinned independent Helsinki-NLP Hebrew-to-English
OPUS-MT checkpoint, and writes the immutable machine template beside the
translation artifacts.
Reviewer decisions are entered in a copy and finalized locally; the published
manifest binds both the untouched template and the final review artifact.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import modal

from sommelier.remote.images import semantic_review_image

APP_NAME = "sommelier-semantic-review"
GPU = os.environ.get("SOMMELIER_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("SOMMELIER_TIMEOUT_SECONDS", str(4 * 60 * 60)))

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name("sommelier-artifacts", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sommelier-hf-cache", create_if_missing=True)


def _local_source_identity() -> tuple[str, bool | None]:
    from sommelier.manifests import get_git_commit, get_git_worktree_clean

    return get_git_commit(), get_git_worktree_clean()


@app.function(  # type: ignore[untyped-decorator]
    image=semantic_review_image(),
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    volumes={"/artifacts": artifacts_volume, "/hf-cache": hf_cache_volume},
)
def run_remote_semantic_review(
    translation_run_id: str,
    code_revision: str,
    source_tree_clean: bool | None,
    allocated_gpu: str,
    allocated_timeout_seconds: int,
) -> str:
    os.environ.setdefault("HF_HOME", "/hf-cache")

    from sommelier.config import load_config
    from sommelier.data.load import load_raw_rows
    from sommelier.data.semantic_review import (
        SEMANTIC_REVIEW_TEMPLATE_FILENAME,
        capture_producer_provenance,
        create_semantic_review_template,
        load_transformers_backtranslator,
        root_split_assignments,
        validate_producer_provenance,
    )
    from sommelier.data.translate import SUMMARY_FILENAME, rows_filename
    from sommelier.errors import UserInputError

    if (
        re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", code_revision) is None
        or source_tree_clean is not True
    ):
        raise UserInputError(
            "semantic-review release evidence requires a clean immutable source revision",
            hint="Commit the gate implementation before launching this full evidence job.",
        )
    if (
        not allocated_gpu.strip()
        or type(allocated_timeout_seconds) is not int
        or allocated_timeout_seconds <= 0
    ):
        raise UserInputError(
            "semantic-review release evidence requires an explicit remote allocation",
            hint="Dispatch the decorated GPU and timeout values as function arguments.",
        )
    work = Path("/artifacts/translation") / translation_run_id
    config_path = work / "config.yaml"
    config = load_config(config_path)
    root_rows_path = work / f"rows.{config.root_dataset.language}.jsonl"
    paired_rows_path = work / rows_filename("he")
    summary_path = work / SUMMARY_FILENAME
    for path in (root_rows_path, paired_rows_path, summary_path):
        if not path.exists():
            raise UserInputError(
                f"semantic-review source artifact not found: {path}",
                hint="Name a completed full Hebrew translation run.",
            )

    root_rows = load_raw_rows(root_rows_path)
    split_by_id = root_split_assignments(config, root_rows)
    producer_provenance = capture_producer_provenance(
        code_revision=code_revision,
        working_tree_clean=source_tree_clean,
        execution_boundary="modal_gpu",
        provider="modal",
        hardware=allocated_gpu,
        allocation_timeout_seconds=allocated_timeout_seconds,
    )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_producer_provenance(
        producer_provenance,
        translation_summary=summary_payload,
    )
    backtranslator = load_transformers_backtranslator()
    output_path = work / SEMANTIC_REVIEW_TEMPLATE_FILENAME
    create_semantic_review_template(
        root_rows_path=root_rows_path,
        paired_rows_path=paired_rows_path,
        translation_summary_path=summary_path,
        root_split_by_id=split_by_id,
        output_path=output_path,
        backtranslator=backtranslator,
        seed=config.project.seed,
        producer_provenance=producer_provenance,
    )
    artifacts_volume.commit()
    hf_cache_volume.commit()
    return str(output_path)


@app.local_entrypoint()  # type: ignore[untyped-decorator]
def main(translation_run_id: str) -> None:
    code_revision, source_tree_clean = _local_source_identity()
    print(
        run_remote_semantic_review.remote(
            translation_run_id,
            code_revision,
            source_tree_clean,
            GPU,
            TIMEOUT_SECONDS,
        )
    )
