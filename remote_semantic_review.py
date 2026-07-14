"""Modal producer for the preregistered Hebrew semantic-review template.

Usage:

    SOMMELIER_GPU=A10G uv run modal run remote_semantic_review.py \
        --translation-run-id he-v3-translate-full

The named full translation run must already contain ``config.yaml``, the
exported English root rows, accepted Hebrew paired rows, and
``translation_summary.json``.  This job deterministically selects 200 rows,
backtranslates them with the pinned independent Helsinki-NLP Hebrew-to-English
OPUS-MT checkpoint, and writes the immutable machine template beside the
translation artifacts.  A full run is one-shot: the producer exclusively
creates and volume-commits an empty, invalid reservation at the final template
path before loading the config, data, or backtranslation model.  A caught
failure safely removes and commits only that exact still-empty inode, allowing
the semantic job—not the 17k-row translation—to retry.  A nonempty or replaced
marker remains fail-closed; a hard crash leaves the empty marker for explicit
operator recovery.  This is the mounted-filesystem boundary, not a claim of
provider-wide locking across separately launched Modal containers; operators
must not launch the same ID concurrently.  The pure local template builder
remains available for disposable fixtures.
Reviewer decisions are entered in a copy and finalized locally; the published
manifest binds both the untouched template and the final review artifact.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import modal

from sommelier.remote.images import semantic_review_image

APP_NAME = "sommelier-semantic-review"
GPU = os.environ.get("SOMMELIER_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("SOMMELIER_TIMEOUT_SECONDS", str(4 * 60 * 60)))
TRANSLATION_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name("sommelier-artifacts", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sommelier-hf-cache", create_if_missing=True)


def _local_source_identity() -> tuple[str, bool | None]:
    from sommelier.manifests import get_git_commit, get_git_worktree_clean

    return get_git_commit(), get_git_worktree_clean()


def _validate_translation_run_id(translation_run_id: str) -> str:
    """Validate one safe artifact-path component before filesystem access."""
    if TRANSLATION_RUN_ID_PATTERN.fullmatch(translation_run_id) is None:
        from sommelier.errors import UserInputError

        raise UserInputError(
            f"invalid semantic-review translation run id: {translation_run_id!r}",
            hint=(
                "Use 1-128 ASCII letters, digits, dots, underscores, or hyphens; "
                "the first character must be alphanumeric."
            ),
        )
    return translation_run_id


def _validate_semantic_launch_boundary(
    *,
    code_revision: str,
    source_tree_clean: bool | None,
    allocated_gpu: str,
    allocated_timeout_seconds: int,
) -> None:
    """Reject locally knowable evidence and allocation drift before dispatch."""
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


ReservationIdentity = tuple[int, int]


def _reserve_semantic_review_template(output_path: Path) -> ReservationIdentity:
    """Create a durable empty one-shot marker without replacing any inode."""
    from sommelier.errors import UserInputError

    try:
        with output_path.open("x", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
            observed = os.fstat(handle.fileno())
    except FileExistsError as error:
        raise UserInputError(
            f"locked semantic-review template already exists: {output_path}",
            hint=(
                "Full semantic-review evidence is one-shot. Keep the existing template "
                "immutable; use a new full translation run id for a new evidence attempt."
            ),
        ) from error
    except OSError as error:
        raise UserInputError(
            f"semantic-review template path could not be reserved: {output_path}",
            hint="Name a completed full translation run with a writable artifact directory.",
        ) from error
    return observed.st_dev, observed.st_ino


def _release_empty_reservation(
    output_path: Path,
    reservation: ReservationIdentity,
) -> bool:
    """Remove only the unchanged empty inode created by this invocation."""
    try:
        observed = output_path.lstat()
    except OSError:
        return False
    if (
        not stat.S_ISREG(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != reservation
        or observed.st_size != 0
    ):
        return False
    try:
        output_path.unlink()
    except OSError:
        return False
    return True


def _validate_finalized_reservation(
    output_path: Path,
    reservation: ReservationIdentity,
) -> None:
    """Require the producer to fill, not replace, its reserved template inode."""
    from sommelier.errors import UserInputError

    try:
        observed = output_path.lstat()
    except OSError as error:
        raise UserInputError("semantic-review producer lost its reserved template") from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != reservation
        or observed.st_size <= 0
    ):
        raise UserInputError(
            "semantic-review producer did not finalize its exact reserved template",
            hint="Preserve any changed marker for inspection and use a new translation run id.",
        )


@contextmanager
def _semantic_review_template_reservation(
    output_path: Path,
) -> Iterator[ReservationIdentity]:
    """Persist one reservation and safely release only an untouched empty failure."""
    reservation = _reserve_semantic_review_template(output_path)
    try:
        artifacts_volume.commit()
        yield reservation
    except Exception:
        if _release_empty_reservation(output_path, reservation):
            artifacts_volume.commit()
        raise


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
    translation_run_id = _validate_translation_run_id(translation_run_id)
    _validate_semantic_launch_boundary(
        code_revision=code_revision,
        source_tree_clean=source_tree_clean,
        allocated_gpu=allocated_gpu,
        allocated_timeout_seconds=allocated_timeout_seconds,
    )
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

    work = Path("/artifacts/translation") / translation_run_id
    if work.is_symlink() or not work.is_dir():
        raise UserInputError(
            f"semantic-review translation run is not a regular directory: {work}",
            hint="Name the completed full translation directory, never a symlink or path alias.",
        )
    output_path = work / SEMANTIC_REVIEW_TEMPLATE_FILENAME
    with _semantic_review_template_reservation(output_path) as reservation:
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
        _validate_finalized_reservation(output_path, reservation)
        artifacts_volume.commit()
        hf_cache_volume.commit()
    return str(output_path)


@app.local_entrypoint()  # type: ignore[untyped-decorator]
def main(translation_run_id: str) -> None:
    translation_run_id = _validate_translation_run_id(translation_run_id)
    code_revision, source_tree_clean = _local_source_identity()
    _validate_semantic_launch_boundary(
        code_revision=code_revision,
        source_tree_clean=source_tree_clean,
        allocated_gpu=GPU,
        allocated_timeout_seconds=TIMEOUT_SECONDS,
    )
    print(
        run_remote_semantic_review.remote(
            translation_run_id,
            code_revision,
            source_tree_clean,
            GPU,
            TIMEOUT_SECONDS,
        )
    )
