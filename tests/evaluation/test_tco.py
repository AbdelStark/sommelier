from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest

from sommelier.errors import EvaluationError
from sommelier.evaluation.generate import (
    inference_timed_call_contract,
    inference_warmup_contract,
)
from sommelier.evaluation.tco import (
    SOVEREIGN_TCO_EVIDENCE_SCHEMA,
    ArtifactEvidence,
    InferenceArmInput,
    LocalAdapterInput,
    TCOIdentity,
    TokenizerTaxInput,
    TrainingInput,
    build_sovereign_tco_evidence,
)


def _artifact(
    path: str,
    kind: str,
    schema: str,
    marker: str,
    *,
    size: int = 10,
) -> ArtifactEvidence:
    digit = f"{sum(marker.encode()) % 16:x}"
    return ArtifactEvidence(
        path=path,
        kind=kind,
        schema_version=schema,
        sha256=digit * 64,
        bytes=size,
    )


def _ref(artifact: ArtifactEvidence) -> dict[str, object]:
    return artifact.payload()


def _identity() -> TCOIdentity:
    return TCOIdentity(
        run_id="v3-run",
        config_sha256="a" * 64,
        tokenizer_id="example/model",
        tokenizer_revision="b" * 40,
        test_split_sha256="c" * 64,
        train_languages=("en", "he"),
        train_epochs=2,
        configured_gpu_label="L40S",
    )


def _tokenizer_input() -> TokenizerTaxInput:
    formatted = {
        split: _artifact(
            f"runs/v3-run/formatted/{split}.jsonl",
            "formatted_split",
            "sommelier.formatted_example.v2",
            split,
        )
        for split in ("train", "validation", "test")
    }
    formatted["test"] = ArtifactEvidence(
        path=formatted["test"].path,
        kind=formatted["test"].kind,
        schema_version=formatted["test"].schema_version,
        sha256="c" * 64,
        bytes=formatted["test"].bytes,
    )
    records_artifact = _artifact(
        "runs/v3-run/analysis/tokenization/tokenizer_tax_records.jsonl",
        "tokenizer_tax_records",
        "sommelier.tokenizer_tax_record.v1",
        "records",
    )
    report_artifact = _artifact(
        "runs/v3-run/analysis/tokenization/tokenizer_tax_report.json",
        "tokenizer_tax_report",
        "sommelier.tokenizer_tax_report.v1",
        "report",
    )
    manifest_artifact = _artifact(
        "runs/v3-run/tokenization_manifest.json",
        "manifest",
        "sommelier.manifest.v1",
        "tokenization-manifest",
    )

    records: list[dict[str, Any]] = []
    counts = {
        "en": {"query_tokens": 2, "prompt_tokens": 10, "full_tokens": 15},
        "he": {"query_tokens": 4, "prompt_tokens": 12, "full_tokens": 18},
    }
    for split in ("train", "validation", "test"):
        root_id = f"{split}-root"
        records.extend(
            [
                {
                    "schema_version": "sommelier.tokenizer_tax_record.v1",
                    "example_id": root_id,
                    "root_example_id": root_id,
                    "source_example_id": None,
                    "language": "en",
                    "split": split,
                    "counts": counts["en"],
                },
                {
                    "schema_version": "sommelier.tokenizer_tax_record.v1",
                    "example_id": f"{root_id}:he",
                    "root_example_id": root_id,
                    "source_example_id": root_id,
                    "language": "he",
                    "split": split,
                    "counts": counts["he"],
                },
            ]
        )

    def paired_scope(multiplier: int) -> dict[str, Any]:
        return {
            "coverage": {
                "paired": multiplier,
                "roots": multiplier,
                "ratio": 1.0,
            },
            "metrics": {
                "query_tokens": {
                    "paired_total": 4 * multiplier,
                    "matched_root_total": 2 * multiplier,
                    "ratio": 2.0,
                },
                "prompt_tokens": {
                    "paired_total": 12 * multiplier,
                    "matched_root_total": 10 * multiplier,
                    "ratio": 1.2,
                },
                "full_tokens": {
                    "paired_total": 18 * multiplier,
                    "matched_root_total": 15 * multiplier,
                    "ratio": 1.2,
                },
            },
        }

    report: dict[str, Any] = {
        "schema_version": "sommelier.tokenizer_tax_report.v1",
        "run_id": "v3-run",
        "config_sha256": "a" * 64,
        "tokenizer": {"id": "example/model", "revision": "b" * 40},
        "inputs": {
            split: {
                "path": artifact.path,
                "sha256": artifact.sha256,
                "bytes": artifact.bytes,
            }
            for split, artifact in formatted.items()
        },
        "records": {
            "path": records_artifact.path,
            "sha256": records_artifact.sha256,
            "count": len(records),
        },
        "root_language": "en",
        "pairing": {
            "he": {
                "all": paired_scope(3),
                "splits": {split: paired_scope(1) for split in ("train", "validation", "test")},
            }
        },
        "training_workload": {
            "languages": ["en", "he"],
            "examples_per_epoch": 2,
            "non_padding_full_tokens_per_epoch": 33,
            "epochs": 2,
            "projected_non_padding_full_tokens": 66,
            "boundary": (
                "Excludes dynamic padding and is a deterministic lower bound on "
                "tokens processed by training."
            ),
        },
    }
    manifest = {
        "schema_version": "sommelier.manifest.v1",
        "stage": "tokenization",
        "status": "succeeded",
        "run_id": "v3-run",
        "config_sha256": "a" * 64,
        "git_commit": "e" * 40,
        "inputs": [_ref(formatted[split]) for split in ("train", "validation", "test")],
        "outputs": [_ref(records_artifact), _ref(report_artifact)],
    }
    return TokenizerTaxInput(
        report=report,
        records=records,
        manifest=manifest,
        report_artifact=report_artifact,
        records_artifact=records_artifact,
        manifest_artifact=manifest_artifact,
        formatted_inputs=formatted,
    )


def _training_input(tokenizer: TokenizerTaxInput) -> TrainingInput:
    metrics_artifact = _artifact(
        "runs/v3-run/train/training_metrics.jsonl",
        "training_metrics",
        "sommelier.training_metric.v1",
        "metrics",
    )
    manifest_artifact = _artifact(
        "runs/v3-run/train_manifest.json",
        "manifest",
        "sommelier.manifest.v1",
        "train-manifest",
    )
    runtime_artifact = _artifact(
        "runs/v3-run/runtime_metadata.json",
        "runtime_metadata",
        "sommelier.runtime_metadata.v1",
        "runtime",
    )
    adapter_files = (
        _artifact(
            "runs/v3-run/train/adapter/adapter_model.safetensors",
            "adapter_weights",
            "",
            "adapter-model",
            size=1_000,
        ),
        _artifact(
            "runs/v3-run/train/adapter/adapter_config.json",
            "adapter_weights",
            "",
            "adapter-config",
            size=200,
        ),
    )
    metrics = [
        {
            "schema_version": "sommelier.training_metric.v1",
            "step": 1,
            "tokens_seen": 100,
            "peak_gpu_memory_mb": None,
        },
        {
            "schema_version": "sommelier.training_metric.v1",
            "step": 2,
            "tokens_seen": 200,
            "peak_gpu_memory_mb": 4096,
        },
    ]
    manifest = {
        "schema_version": "sommelier.manifest.v1",
        "stage": "train",
        "status": "succeeded",
        "run_id": "v3-run",
        "config_sha256": "a" * 64,
        "git_commit": "e" * 40,
        "inputs": [
            _ref(tokenizer.formatted_inputs["train"]),
            _ref(tokenizer.formatted_inputs["validation"]),
        ],
        "outputs": [*(_ref(item) for item in adapter_files), _ref(metrics_artifact)],
        "details": {"train_languages": ["en", "he"]},
    }
    return TrainingInput(
        runtime_metadata={
            "schema_version": "sommelier.runtime_metadata.v1",
            "run_id": "v3-run",
            "config_sha256": "a" * 64,
            "stages": {"train": {"elapsed_seconds": 3600.0}},
            "hardware": {"gpu": "L40S", "source": "config"},
            "peak_gpu_memory_mb": 4096,
            "observed_cost_usd": None,
            "cost_source": "unavailable",
            "source_code": {
                "git_commit": "e" * 40,
                "working_tree_clean": True,
                "boundary": "Measured before remote dispatch.",
            },
            "remote_execution": {
                "provider": "modal",
                "function_timeout_seconds": 21_600,
                "gpu_allocation_label": "L40S",
                "boundary": "Outer function timeout; billing is separate.",
            },
        },
        runtime_artifact=runtime_artifact,
        metrics=metrics,
        metrics_artifact=metrics_artifact,
        manifest=manifest,
        manifest_artifact=manifest_artifact,
        adapter_identity={
            "source": "/artifact/run/train/adapter",
            "kind": "local_directory",
            "revision": None,
            "tree_sha256": "d" * 64,
            "artifact_path": "runs/v3-run/train/adapter",
            "revision_is_immutable": True,
        },
        local_adapter=LocalAdapterInput(
            tree_sha256="d" * 64,
            files=adapter_files,
        ),
    )


def _inference_arm() -> InferenceArmInput:
    telemetry_artifact = _artifact(
        "runs/v3-run/eval/adapter/inference_telemetry.json",
        "inference_telemetry",
        "sommelier.inference_telemetry.v2",
        "telemetry",
    )
    report_artifact = _artifact(
        "runs/v3-run/eval/adapter/evaluation_report.json",
        "evaluation_report",
        "sommelier.evaluation_report.v3",
        "eval-report",
    )
    generations = {
        language: _artifact(
            f"runs/v3-run/eval/adapter/generations.{language}.jsonl",
            "generations",
            "sommelier.generation.v2",
            f"generation-{language}",
        )
        for language in ("en", "he")
    }
    measurement = {
        "scope": "generator.generate_end_to_end_call_wall_time",
        "aggregation": "sum_of_per_example_call_intervals",
        "clock": "monotonic_seconds",
        "model_load_included": False,
        "parsing_and_artifact_io_included": False,
    }
    timed_call_contract = inference_timed_call_contract()
    warmup = inference_warmup_contract()
    sequential = {
        "boundary": "single_run_generation_invocation_after_model_load",
        "concurrency": 1,
        "single_model_instance": True,
        "slice_order": ["en", "he"],
        "example_order": "formatted_test_order_within_slice",
    }
    hardware = {"gpu_label": "L40S", "gpu_count": 1, "source": "config.remote.gpu"}
    slices: dict[str, dict[str, Any]] = {
        "en": {
            "examples": 2,
            "elapsed_seconds": 4.0,
            "seconds_per_example": 2.0,
            "generation_artifact": _ref(generations["en"]),
        },
        "he": {
            "examples": 2,
            "elapsed_seconds": 6.0,
            "seconds_per_example": 3.0,
            "generation_artifact": _ref(generations["he"]),
        },
    }
    telemetry: dict[str, Any] = {
        "schema_version": "sommelier.inference_telemetry.v2",
        "run_id": "v3-run",
        "model_kind": "adapter",
        "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 64},
        "measurement": measurement,
        "timed_call_contract": timed_call_contract,
        "warmup": warmup,
        "sequential_run": sequential,
        "hardware": hardware,
        "slices": slices,
        "total": {
            "examples": 4,
            "elapsed_seconds": 10.0,
            "seconds_per_example": 2.5,
        },
    }

    def ratio(elapsed: float, successes: int) -> dict[str, Any]:
        return {
            "available": successes > 0,
            "value": round(elapsed / successes, 6) if successes else None,
            "reason": None if successes else "zero_full_call_exact_successes",
            "unit": "gpu_seconds_per_full_call_exact_success",
            "full_call_exact_successes": successes,
            "basis": "generation_elapsed_seconds_x_configured_gpu_count",
        }

    efficiency_slices = {
        "en": {**slices["en"], "gpu_seconds_per_full_call_exact_success": ratio(4.0, 2)},
        "he": {**slices["he"], "gpu_seconds_per_full_call_exact_success": ratio(6.0, 1)},
    }
    efficiency = {
        "available": True,
        "telemetry_artifact": _ref(telemetry_artifact),
        "measurement": measurement,
        "timed_call_contract": timed_call_contract,
        "warmup": warmup,
        "sequential_run": sequential,
        "hardware": hardware,
        "slices": efficiency_slices,
        "overall": {
            **telemetry["total"],
            "gpu_seconds_per_full_call_exact_success": ratio(10.0, 3),
        },
    }
    evaluation_manifest_artifact = _artifact(
        "runs/v3-run/eval-adapter_manifest.json",
        "manifest",
        "sommelier.manifest.v1",
        "eval-manifest",
    )
    run_manifest_artifact = _artifact(
        "runs/v3-run/manifest.json",
        "manifest",
        "sommelier.manifest.v1",
        "run-manifest",
    )
    config_artifact = ArtifactEvidence(
        path="runs/v3-run/config.resolved.yaml",
        kind="config",
        schema_version="sommelier.config.v2",
        sha256="a" * 64,
        bytes=10,
    )
    runtime_artifact = _artifact(
        "runs/v3-run/runtime_metadata.json",
        "runtime_metadata",
        "sommelier.runtime_metadata.v1",
        "inference-runtime",
    )
    evaluation_manifest = {
        "schema_version": "sommelier.manifest.v1",
        "stage": "eval-adapter",
        "status": "succeeded",
        "run_id": "v3-run",
        "config_sha256": "a" * 64,
        "git_commit": "e" * 40,
        "inputs": [],
        "outputs": [
            _ref(generations["en"]),
            _ref(generations["he"]),
            _ref(telemetry_artifact),
            _ref(report_artifact),
        ],
    }
    run_manifest = {
        "schema_version": "sommelier.manifest.v1",
        "run_id": "v3-run",
        "stages": {"eval-adapter": evaluation_manifest_artifact.path},
        "config": _ref(config_artifact),
        "status": "succeeded",
    }
    return InferenceArmInput(
        name="v3_en_he",
        run_id="v3-run",
        model_kind="adapter",
        config_sha256="a" * 64,
        efficiency=efficiency,
        telemetry=telemetry,
        telemetry_artifact=telemetry_artifact,
        evaluation_report_artifact=report_artifact,
        generation_artifacts=generations,
        actual_examples={"en": 2, "he": 2},
        exact_successes={"en": 2, "he": 1, "overall": 3},
        decoding={"temperature": 0.0, "do_sample": False, "max_new_tokens": 64},
        evaluation_manifest=evaluation_manifest,
        evaluation_manifest_artifact=evaluation_manifest_artifact,
        run_manifest=run_manifest,
        run_manifest_artifact=run_manifest_artifact,
        resolved_config_artifact=config_artifact,
        configured_gpu_label="L40S",
        runtime_metadata={
            "schema_version": "sommelier.runtime_metadata.v1",
            "run_id": "v3-run",
            "config_sha256": "a" * 64,
            "hardware": {"gpu": "L40S", "source": "config"},
            "packages": {
                "python": "3.13.0",
                "torch": "2.11.0",
                "transformers": "5.13.1",
                "tokenizers": "0.22.2",
                "accelerate": "1.14.0",
                "peft": "0.18.0",
                "bitsandbytes": "0.49.1",
                "datasets": "5.0.0",
                "huggingface_hub": "1.23.0",
            },
            "source_code": {
                "git_commit": "e" * 40,
                "working_tree_clean": True,
            },
        },
        runtime_artifact=runtime_artifact,
    )


def _complete_inputs() -> tuple[TCOIdentity, TokenizerTaxInput, TrainingInput, InferenceArmInput]:
    tokenizer = _tokenizer_input()
    return _identity(), tokenizer, _training_input(tokenizer), _inference_arm()


def test_builds_bounded_observed_and_projected_tco_evidence() -> None:
    identity, tokenizer, training, inference = _complete_inputs()

    evidence = build_sovereign_tco_evidence(
        identity,
        tokenizer_tax=tokenizer,
        training=training,
        inference_arms=[inference],
    )

    assert evidence["schema_version"] == SOVEREIGN_TCO_EVIDENCE_SCHEMA
    paired = evidence["paired_tokenization"]
    assert paired["paired_scopes"]["all"]["coverage"] == {
        "paired": 3,
        "roots": 3,
        "ratio": 1.0,
    }
    assert (
        paired["paired_scopes"]["all"]["token_ratios"]["query_tokens"]["hebrew_to_english_ratio"]
        == 2.0
    )
    workload = paired["projected_training_workload"]
    assert workload["projected_non_padding_full_tokens"] == 66
    assert workload["evidence_kind"] == "deterministic_projection"

    qlora = evidence["qlora_training"]
    assert qlora["train_stage_runtime"]["elapsed_seconds"] == 3600.0
    assert qlora["train_stage_runtime"]["configured_gpu_hours"] == 1.0
    assert qlora["source_code"]["git_commit"] == "e" * 40
    assert qlora["source_code"]["working_tree_clean"] is True
    assert qlora["remote_execution"]["function_timeout_seconds"] == 21_600
    assert qlora["peak_gpu_memory"]["value"] == 4096
    assert qlora["tokens_seen"]["value"] == 200
    assert qlora["end_to_end_token_throughput"]["value"] == 200 / 3600
    assert qlora["adapter_storage"] == {
        "available": True,
        "tree_sha256": "d" * 64,
        "packaged_adapter": {
            "bytes": 1200,
            "files": 2,
            "boundary": (
                "All regular files under the evaluated local adapter directory, "
                "including configs and tokenizer assets."
            ),
        },
        "tensor_weights_only": {
            "bytes": 1000,
            "files": 1,
            "boundary": (
                "Files named adapter_model.safetensors, adapter_model.bin, "
                "or their numbered shards."
            ),
        },
        "evidence_kind": "observed_artifact_storage",
    }
    assert qlora["currency_cost"] == {
        "available": False,
        "value": None,
        "reason": "provider_billing_evidence_not_supplied",
    }
    assert qlora["full_finetune_savings"]["available"] is False

    inference_evidence = evidence["inference_efficiency"]["arms"]["v3_en_he"]
    assert (
        inference_evidence["slices"]["en"]["configured_gpu_seconds_per_full_call_exact_success"][
            "value"
        ]
        == 2.0
    )
    assert (
        inference_evidence["slices"]["he"]["configured_gpu_seconds_per_full_call_exact_success"][
            "value"
        ]
        == 6.0
    )
    assert inference_evidence["overall"]["configured_gpu_seconds_per_full_call_exact_success"][
        "value"
    ] == round(10 / 3, 6)
    assert evidence["explicitly_unavailable"]["currency_cost"]["available"] is False
    assert evidence["explicitly_unavailable"]["full_finetune_savings"]["available"] is False


def test_missing_optional_artifacts_stays_explicitly_unavailable() -> None:
    identity = _identity()
    report_artifact = _artifact(
        "runs/base/eval/base/evaluation_report.json",
        "evaluation_report",
        "sommelier.evaluation_report.v3",
        "missing-eval",
    )
    generations = {
        language: _artifact(
            f"runs/base/eval/base/generations.{language}.jsonl",
            "generations",
            "sommelier.generation.v2",
            f"missing-{language}",
        )
        for language in ("en", "he")
    }
    eval_manifest_artifact = _artifact(
        "runs/base/eval-base_manifest.json",
        "manifest",
        "sommelier.manifest.v1",
        "missing-eval-manifest",
    )
    run_manifest_artifact = _artifact(
        "runs/base/manifest.json",
        "manifest",
        "sommelier.manifest.v1",
        "missing-run-manifest",
    )
    config_artifact = ArtifactEvidence(
        path="runs/base/config.resolved.yaml",
        kind="config",
        schema_version="sommelier.config.v2",
        sha256="f" * 64,
        bytes=10,
    )
    runtime_artifact = _artifact(
        "runs/base/runtime_metadata.json",
        "runtime_metadata",
        "sommelier.runtime_metadata.v1",
        "missing-runtime",
    )
    evidence = build_sovereign_tco_evidence(
        identity,
        tokenizer_tax=None,
        training=None,
        inference_arms=[
            InferenceArmInput(
                name="base",
                run_id="base",
                model_kind="base",
                config_sha256="f" * 64,
                efficiency=None,
                telemetry=None,
                telemetry_artifact=None,
                evaluation_report_artifact=report_artifact,
                generation_artifacts=generations,
                actual_examples={"en": 1, "he": 1},
                exact_successes={},
                decoding={},
                evaluation_manifest={
                    "schema_version": "sommelier.manifest.v1",
                    "stage": "eval-base",
                    "status": "succeeded",
                    "run_id": "base",
                    "config_sha256": "f" * 64,
                    "git_commit": "e" * 40,
                    "outputs": [
                        _ref(generations["en"]),
                        _ref(generations["he"]),
                        _ref(report_artifact),
                    ],
                },
                evaluation_manifest_artifact=eval_manifest_artifact,
                run_manifest={
                    "schema_version": "sommelier.manifest.v1",
                    "run_id": "base",
                    "stages": {"eval-base": eval_manifest_artifact.path},
                    "config": _ref(config_artifact),
                    "status": "succeeded",
                },
                run_manifest_artifact=run_manifest_artifact,
                resolved_config_artifact=config_artifact,
                configured_gpu_label="L40S",
                runtime_metadata={
                    "schema_version": "sommelier.runtime_metadata.v1",
                    "run_id": "base",
                    "config_sha256": "f" * 64,
                    "hardware": {"gpu": "L40S", "source": "config"},
                    "source_code": {
                        "git_commit": "e" * 40,
                        "working_tree_clean": True,
                    },
                },
                runtime_artifact=runtime_artifact,
            )
        ],
    )

    assert evidence["paired_tokenization"] == {
        "available": False,
        "reason": "tokenizer_tax_evidence_missing",
    }
    assert evidence["qlora_training"]["available"] is False
    assert evidence["inference_efficiency"]["arms"]["base"] == {
        "available": False,
        "reason": "evaluation_report_has_no_inference_efficiency",
    }


def test_tokenizer_report_hash_linkage_fails_closed() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    tampered_report = copy.deepcopy(dict(tokenizer.report))
    tampered_report["records"]["sha256"] = "0" * 64
    tampered = TokenizerTaxInput(
        report=tampered_report,
        records=tokenizer.records,
        manifest=tokenizer.manifest,
        report_artifact=tokenizer.report_artifact,
        records_artifact=tokenizer.records_artifact,
        manifest_artifact=tokenizer.manifest_artifact,
        formatted_inputs=tokenizer.formatted_inputs,
    )

    with pytest.raises(EvaluationError, match="records sha256"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tampered,
            training=training,
            inference_arms=[inference],
        )


def test_training_peak_memory_disagreement_fails_closed() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    runtime = copy.deepcopy(dict(training.runtime_metadata or {}))
    runtime["peak_gpu_memory_mb"] = 8192
    tampered = TrainingInput(
        runtime_metadata=runtime,
        runtime_artifact=training.runtime_artifact,
        metrics=training.metrics,
        metrics_artifact=training.metrics_artifact,
        manifest=training.manifest,
        manifest_artifact=training.manifest_artifact,
        adapter_identity=training.adapter_identity,
        local_adapter=training.local_adapter,
    )

    with pytest.raises(EvaluationError, match="disagree on peak GPU memory"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tokenizer,
            training=tampered,
            inference_arms=[inference],
        )


def test_runtime_identity_mismatch_fails_closed() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    runtime = copy.deepcopy(dict(training.runtime_metadata or {}))
    runtime["run_id"] = "another-run"
    tampered = TrainingInput(
        runtime_metadata=runtime,
        runtime_artifact=training.runtime_artifact,
        metrics=training.metrics,
        metrics_artifact=training.metrics_artifact,
        manifest=training.manifest,
        manifest_artifact=training.manifest_artifact,
        adapter_identity=training.adapter_identity,
        local_adapter=training.local_adapter,
    )

    with pytest.raises(EvaluationError, match="runtime metadata run_id"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tokenizer,
            training=tampered,
            inference_arms=[inference],
        )


def test_hand_entered_runtime_cost_cannot_become_billing_evidence() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    runtime = copy.deepcopy(dict(training.runtime_metadata or {}))
    runtime["observed_cost_usd"] = 12.34
    runtime["cost_source"] = "modal dashboard"

    with pytest.raises(EvaluationError, match="provider billing artifact"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tokenizer,
            training=replace(training, runtime_metadata=runtime),
            inference_arms=[inference],
        )


def test_training_source_snapshot_requires_clean_worktree() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    runtime = copy.deepcopy(dict(training.runtime_metadata or {}))
    runtime["source_code"]["working_tree_clean"] = False
    evidence = build_sovereign_tco_evidence(
        identity,
        tokenizer_tax=tokenizer,
        training=replace(training, runtime_metadata=runtime),
        inference_arms=[inference],
    )

    assert evidence["qlora_training"]["source_code"] == {
        "available": False,
        "git_commit": "e" * 40,
        "working_tree_clean": False,
        "boundary": "Measured before remote dispatch.",
        "reason": "working_tree_not_recorded_clean",
    }


def test_train_manifest_commit_must_match_runtime_source() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    manifest = copy.deepcopy(dict(training.manifest or {}))
    manifest["git_commit"] = "f" * 40

    with pytest.raises(EvaluationError, match="train manifest git_commit"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tokenizer,
            training=replace(training, manifest=manifest),
            inference_arms=[inference],
        )


def test_missing_trainer_token_count_stays_explicitly_unavailable() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    metrics = [{**dict(metric), "tokens_seen": 0} for metric in training.metrics or ()]
    zero_tokens = TrainingInput(
        runtime_metadata=training.runtime_metadata,
        runtime_artifact=training.runtime_artifact,
        metrics=metrics,
        metrics_artifact=training.metrics_artifact,
        manifest=training.manifest,
        manifest_artifact=training.manifest_artifact,
        adapter_identity=training.adapter_identity,
        local_adapter=training.local_adapter,
    )

    evidence = build_sovereign_tco_evidence(
        identity,
        tokenizer_tax=tokenizer,
        training=zero_tokens,
        inference_arms=[inference],
    )

    assert evidence["qlora_training"]["tokens_seen"] == {
        "available": False,
        "value": None,
        "reason": "trainer_did_not_report_nonzero_tokens_seen",
    }
    assert evidence["qlora_training"]["end_to_end_token_throughput"]["available"] is False


def test_inference_efficiency_ratio_tampering_fails_closed() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    efficiency = copy.deepcopy(dict(inference.efficiency or {}))
    efficiency["slices"]["he"]["gpu_seconds_per_full_call_exact_success"]["value"] = 0.01
    tampered = InferenceArmInput(
        name=inference.name,
        run_id=inference.run_id,
        model_kind=inference.model_kind,
        config_sha256=inference.config_sha256,
        efficiency=efficiency,
        telemetry=inference.telemetry,
        telemetry_artifact=inference.telemetry_artifact,
        evaluation_report_artifact=inference.evaluation_report_artifact,
        generation_artifacts=inference.generation_artifacts,
        actual_examples=inference.actual_examples,
        exact_successes=inference.exact_successes,
        decoding=inference.decoding,
        evaluation_manifest=inference.evaluation_manifest,
        evaluation_manifest_artifact=inference.evaluation_manifest_artifact,
        run_manifest=inference.run_manifest,
        run_manifest_artifact=inference.run_manifest_artifact,
        resolved_config_artifact=inference.resolved_config_artifact,
        configured_gpu_label=inference.configured_gpu_label,
        runtime_metadata=inference.runtime_metadata,
        runtime_artifact=inference.runtime_artifact,
    )

    with pytest.raises(EvaluationError, match="GPU-seconds ratio"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tokenizer,
            training=training,
            inference_arms=[tampered],
        )


def test_inference_efficiency_missing_timed_call_contract_fails_closed() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    telemetry = copy.deepcopy(dict(inference.telemetry or {}))
    del telemetry["timed_call_contract"]

    with pytest.raises(EvaluationError, match="telemetry timed-call contract"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tokenizer,
            training=training,
            inference_arms=[replace(inference, telemetry=telemetry)],
        )


def test_inference_efficiency_tampered_warmup_contract_fails_closed() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    telemetry = copy.deepcopy(dict(inference.telemetry or {}))
    telemetry["warmup"]["timed"] = True

    with pytest.raises(EvaluationError, match="inference warmup contract"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tokenizer,
            training=training,
            inference_arms=[replace(inference, telemetry=telemetry)],
        )


def test_inference_total_timing_cannot_be_tampered_in_both_layers() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    telemetry = copy.deepcopy(dict(inference.telemetry or {}))
    efficiency = copy.deepcopy(dict(inference.efficiency or {}))
    telemetry["total"]["seconds_per_example"] = 99.0
    efficiency["overall"]["seconds_per_example"] = 99.0
    tampered = InferenceArmInput(
        name=inference.name,
        run_id=inference.run_id,
        model_kind=inference.model_kind,
        config_sha256=inference.config_sha256,
        efficiency=efficiency,
        telemetry=telemetry,
        telemetry_artifact=inference.telemetry_artifact,
        evaluation_report_artifact=inference.evaluation_report_artifact,
        generation_artifacts=inference.generation_artifacts,
        actual_examples=inference.actual_examples,
        exact_successes=inference.exact_successes,
        decoding=inference.decoding,
        evaluation_manifest=inference.evaluation_manifest,
        evaluation_manifest_artifact=inference.evaluation_manifest_artifact,
        run_manifest=inference.run_manifest,
        run_manifest_artifact=inference.run_manifest_artifact,
        resolved_config_artifact=inference.resolved_config_artifact,
        configured_gpu_label=inference.configured_gpu_label,
        runtime_metadata=inference.runtime_metadata,
        runtime_artifact=inference.runtime_artifact,
    )

    with pytest.raises(EvaluationError, match="total seconds/example"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tokenizer,
            training=training,
            inference_arms=[tampered],
        )


def test_inference_telemetry_count_must_match_observed_generations() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    telemetry = copy.deepcopy(dict(inference.telemetry or {}))
    efficiency = copy.deepcopy(dict(inference.efficiency or {}))
    telemetry["slices"]["en"]["examples"] = 3
    telemetry["slices"]["en"]["seconds_per_example"] = round(4 / 3, 6)
    efficiency["slices"]["en"]["examples"] = 3
    efficiency["slices"]["en"]["seconds_per_example"] = round(4 / 3, 6)
    telemetry["total"]["examples"] = 5
    telemetry["total"]["seconds_per_example"] = 2.0
    efficiency["overall"]["examples"] = 5
    efficiency["overall"]["seconds_per_example"] = 2.0

    with pytest.raises(EvaluationError, match="telemetry examples do not match"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tokenizer,
            training=training,
            inference_arms=[replace(inference, telemetry=telemetry, efficiency=efficiency)],
        )


def test_inference_requires_succeeded_root_manifest() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    root = copy.deepcopy(dict(inference.run_manifest))
    root["status"] = "failed"

    with pytest.raises(EvaluationError, match="root run manifest is not succeeded"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tokenizer,
            training=training,
            inference_arms=[replace(inference, run_manifest=root)],
        )


def test_inference_eval_commit_must_match_clean_runtime_source() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    manifest = copy.deepcopy(dict(inference.evaluation_manifest))
    manifest["git_commit"] = "f" * 40

    with pytest.raises(EvaluationError, match="git_commit does not match runtime"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tokenizer,
            training=training,
            inference_arms=[replace(inference, evaluation_manifest=manifest)],
        )


def test_tokenization_manifest_cannot_duplicate_one_output() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    manifest = copy.deepcopy(dict(tokenizer.manifest))
    manifest["outputs"] = [manifest["outputs"][0], manifest["outputs"][0]]
    tampered = TokenizerTaxInput(
        report=tokenizer.report,
        records=tokenizer.records,
        manifest=manifest,
        report_artifact=tokenizer.report_artifact,
        records_artifact=tokenizer.records_artifact,
        manifest_artifact=tokenizer.manifest_artifact,
        formatted_inputs=tokenizer.formatted_inputs,
    )

    with pytest.raises(EvaluationError, match="output set"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tampered,
            training=training,
            inference_arms=[inference],
        )


@pytest.mark.parametrize("git_commit", ["unknown", "f" * 40])
def test_tokenization_manifest_must_match_v3_runtime_source(git_commit: str) -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    manifest = copy.deepcopy(dict(tokenizer.manifest))
    manifest["git_commit"] = git_commit

    with pytest.raises(EvaluationError, match="tokenization manifest git_commit"):
        build_sovereign_tco_evidence(
            replace(identity, source_code_revision="e" * 40),
            tokenizer_tax=replace(tokenizer, manifest=manifest),
            training=training,
            inference_arms=[inference],
        )


def test_training_runtime_must_match_declared_v3_source() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    runtime = copy.deepcopy(dict(training.runtime_metadata or {}))
    runtime["source_code"]["git_commit"] = "f" * 40

    with pytest.raises(EvaluationError, match="training runtime git_commit"):
        build_sovereign_tco_evidence(
            replace(identity, source_code_revision="e" * 40),
            tokenizer_tax=tokenizer,
            training=replace(training, runtime_metadata=runtime),
            inference_arms=[inference],
        )


def test_training_runtime_cannot_omit_declared_v3_source() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    runtime = copy.deepcopy(dict(training.runtime_metadata or {}))
    runtime.pop("source_code")

    with pytest.raises(EvaluationError, match="no source-code provenance"):
        build_sovereign_tco_evidence(
            replace(identity, source_code_revision="e" * 40),
            tokenizer_tax=tokenizer,
            training=replace(training, runtime_metadata=runtime),
            inference_arms=[inference],
        )


def test_three_arm_release_requires_source_code_identity() -> None:
    with pytest.raises(EvaluationError, match="requires the v3 source-code revision"):
        build_sovereign_tco_evidence(
            _identity(),
            tokenizer_tax=None,
            training=None,
            inference_arms=[],
            require_three_arm_matrix=True,
        )


def test_tokenizer_artifact_must_use_expected_run_relative_path() -> None:
    identity, tokenizer, training, inference = _complete_inputs()
    tampered = TokenizerTaxInput(
        report=tokenizer.report,
        records=tokenizer.records,
        manifest=tokenizer.manifest,
        report_artifact=ArtifactEvidence(
            path="runs/v3-run/elsewhere/tokenizer_tax_report.json",
            kind=tokenizer.report_artifact.kind,
            schema_version=tokenizer.report_artifact.schema_version,
            sha256=tokenizer.report_artifact.sha256,
            bytes=tokenizer.report_artifact.bytes,
        ),
        records_artifact=tokenizer.records_artifact,
        manifest_artifact=tokenizer.manifest_artifact,
        formatted_inputs=tokenizer.formatted_inputs,
    )

    with pytest.raises(EvaluationError, match="expected run-relative path"):
        build_sovereign_tco_evidence(
            identity,
            tokenizer_tax=tampered,
            training=training,
            inference_arms=[inference],
        )
