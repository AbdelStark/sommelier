from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from sommelier.artifacts import sha256_file
from sommelier.config import SommelierConfig
from sommelier.errors import UserInputError
from sommelier.evaluation.data_provenance import is_hebrew_v3_config

if TYPE_CHECKING:
    from collections.abc import Callable

    from sommelier.run_context import RunContext

__all__ = (
    "FullPairedInputValidationCapability",
    "consume_full_paired_input_validation",
    "requires_full_paired_input_validation",
)

_SMOKE_MAX_TRAIN = 100
_SMOKE_MAX_VALIDATION = 20
_SMOKE_MAX_TEST = 20
_TRAINING_SPLITS = ("train", "validation")


class FullPairedInputValidationCapability:
    """Opaque, process-local proof that the complete paired-input gate passed."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> FullPairedInputValidationCapability:
        raise TypeError("full paired-input validation capabilities are issued internally")


class _FullPairedInputValidationReceipt:
    """Opaque proof retained by the pipeline until formatted data exists."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> _FullPairedInputValidationReceipt:
        raise TypeError("full paired-input validation receipts are issued internally")


@dataclass(frozen=True)
class _ValidatedConfigBinding:
    canonical_config: bytes


@dataclass(frozen=True)
class _FormattedDataIdentity:
    directory: Path
    split_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _TrainingCapabilityBinding:
    canonical_config: bytes
    context: RunContext
    run_id: str
    run_dir: Path
    config_sha256: str
    formatted_data: _FormattedDataIdentity


def requires_full_paired_input_validation(config: SommelierConfig) -> bool:
    """Return whether this is a full-sized Hebrew v3 training configuration."""
    return is_hebrew_v3_config(config) and (
        config.data.n_train > _SMOKE_MAX_TRAIN
        or config.data.n_validation > _SMOKE_MAX_VALIDATION
        or config.data.n_test > _SMOKE_MAX_TEST
    )


def _canonical_config_snapshot(config: SommelierConfig) -> bytes:
    """Return a stable, immutable representation of the effective config."""
    return json.dumps(
        config.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _formatted_data_identity(formatted_dir: Path) -> _FormattedDataIdentity:
    directory = formatted_dir.resolve()
    try:
        split_sha256 = tuple(
            (split, sha256_file(directory / f"{split}.jsonl")) for split in _TRAINING_SPLITS
        )
    except OSError as error:
        raise UserInputError(
            f"formatted training data is unavailable under {directory}",
            hint="Complete the format stage before entering the full training boundary.",
        ) from error
    return _FormattedDataIdentity(directory=directory, split_sha256=split_sha256)


def _invalid_capability_error() -> UserInputError:
    return UserInputError(
        "Hebrew v3 full training received no valid paired-input capability",
        hint=(
            "Restart with sommelier pipeline run --mode full; capabilities are "
            "process-local, config-, run-, and data-bound, and single-use."
        ),
    )


def _build_authorization_boundary() -> tuple[
    Callable[[SommelierConfig, Path], _FullPairedInputValidationReceipt],
    Callable[
        [
            _FullPairedInputValidationReceipt | None,
            SommelierConfig,
            RunContext,
            Path,
            Path,
        ],
        FullPairedInputValidationCapability | None,
    ],
    Callable[
        [
            SommelierConfig,
            FullPairedInputValidationCapability | None,
            RunContext,
            Path,
        ],
        None,
    ],
]:
    # These registries intentionally live only in the closures returned below.
    # Keeping them out of module globals removes the writable injection surface
    # while retaining weak, process-local, single-use capability state.
    validated: WeakKeyDictionary[_FullPairedInputValidationReceipt, _ValidatedConfigBinding] = (
        WeakKeyDictionary()
    )
    issued: WeakKeyDictionary[FullPairedInputValidationCapability, _TrainingCapabilityBinding] = (
        WeakKeyDictionary()
    )

    def validate_for_pipeline(
        config: SommelierConfig,
        root_rows_path: Path,
    ) -> _FullPairedInputValidationReceipt:
        from sommelier.data.translate import validate_full_paired_input_contract

        validate_full_paired_input_contract(config, root_rows_path)
        receipt = object.__new__(_FullPairedInputValidationReceipt)
        validated[receipt] = _ValidatedConfigBinding(
            canonical_config=_canonical_config_snapshot(config)
        )
        return receipt

    def issue_for_training(
        receipt: _FullPairedInputValidationReceipt | None,
        config: SommelierConfig,
        context: RunContext,
        formatted_dir: Path,
        staged_root_rows_path: Path,
    ) -> FullPairedInputValidationCapability | None:
        if not requires_full_paired_input_validation(config):
            return None
        if not isinstance(receipt, _FullPairedInputValidationReceipt):
            raise _invalid_capability_error()
        validated_binding = validated.pop(receipt, None)
        canonical_config = _canonical_config_snapshot(config)
        if (
            validated_binding is None
            or validated_binding.canonical_config != canonical_config
            or formatted_dir.resolve() != (context.run_dir / "formatted").resolve()
            or staged_root_rows_path.resolve()
            != (
                context.run_dir
                / "data"
                / "source_inputs"
                / f"rows.{config.root_dataset.language}.jsonl"
            ).resolve()
        ):
            raise _invalid_capability_error()

        # The early validation prevents a full run directory from being
        # created for an invalid publication. Repeating the exact gate over
        # the materialized source bundle closes the mutation window between
        # that preflight and ``_stage_prepare`` copying external files.
        from sommelier.data.translate import validate_full_paired_input_contract

        validate_full_paired_input_contract(config, staged_root_rows_path)

        capability = object.__new__(FullPairedInputValidationCapability)
        issued[capability] = _TrainingCapabilityBinding(
            canonical_config=canonical_config,
            context=context,
            run_id=context.run_id,
            run_dir=context.run_dir.resolve(),
            config_sha256=context.config_sha256,
            formatted_data=_formatted_data_identity(formatted_dir),
        )
        return capability

    def consume_for_training(
        config: SommelierConfig,
        capability: FullPairedInputValidationCapability | None,
        context: RunContext,
        formatted_dir: Path,
    ) -> None:
        if not requires_full_paired_input_validation(config):
            return
        if not isinstance(capability, FullPairedInputValidationCapability):
            raise UserInputError(
                "Hebrew v3 full training requires the validated full pipeline",
                hint=(
                    "Use sommelier pipeline run --mode full so the complete paired-input, "
                    "publication, and semantic-review contract is checked before training."
                ),
            )

        binding = issued.pop(capability, None)
        if binding is None:
            raise _invalid_capability_error()
        if (
            binding.canonical_config != _canonical_config_snapshot(config)
            or binding.context is not context
            or binding.run_id != context.run_id
            or binding.run_dir != context.run_dir.resolve()
            or binding.config_sha256 != context.config_sha256
            or binding.formatted_data.directory != formatted_dir.resolve()
        ):
            raise _invalid_capability_error()
        try:
            formatted_data = _formatted_data_identity(formatted_dir)
        except UserInputError as error:
            raise _invalid_capability_error() from error
        if binding.formatted_data != formatted_data:
            raise _invalid_capability_error()

    return validate_for_pipeline, issue_for_training, consume_for_training


(
    _validate_full_paired_input_for_pipeline,
    _issue_full_paired_input_for_training,
    _consume_full_paired_input_validation,
) = _build_authorization_boundary()
del _build_authorization_boundary


def consume_full_paired_input_validation(
    config: SommelierConfig,
    capability: FullPairedInputValidationCapability | None,
    *,
    context: RunContext,
    formatted_dir: Path,
) -> None:
    """Consume a pipeline-issued capability bound to this config, run, and data."""
    _consume_full_paired_input_validation(config, capability, context, formatted_dir)
