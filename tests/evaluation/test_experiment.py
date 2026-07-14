from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import sommelier.evaluation.data_provenance as data_provenance
import sommelier.evaluation.experiment as experiment_module
from sommelier.config import SommelierConfig, write_resolved_config
from sommelier.errors import EvaluationError, UserInputError
from sommelier.evaluation.data_provenance import _same_cohort_identity_set
from sommelier.evaluation.experiment import (
    _load_arm,
    _training_tco_input,
    _validate_preregistered_adapter_arms,
    write_experiment_report,
)
from sommelier.evaluation.generate import (
    adapter_tree_sha256,
    inference_timed_call_contract,
    inference_warmup_contract,
)

SCHEMA = "sommelier.evaluation_report.v3"
GENERATION_SCHEMA = "sommelier.generation.v2"
FORMATTED_SCHEMA = "sommelier.formatted_example.v2"
MODEL_IDENTITY = {
    "base_model_id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "base_model_revision": "54641c1611fcff44fa4865626462445e0a153fc7",
    "tokenizer_id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "tokenizer_revision": "54641c1611fcff44fa4865626462445e0a153fc7",
}
DECODING = {"temperature": 0.0, "do_sample": False, "max_new_tokens": 512}
GOLD_CALL = {"name": "lookup", "arguments": {"value": "x"}}
_REAL_VALIDATE_OBSERVED_COHORTS = data_provenance._validate_observed_cohorts


@pytest.fixture(autouse=True)
def _stub_semantic_publication_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep comparator tests focused; the real semantic gate has its own suite."""

    def validate(_config: SommelierConfig, root_rows_path: Path) -> dict[str, dict[str, Path]]:
        return {
            "he": {
                "paired_rows": root_rows_path.with_name("rows.en.he.jsonl"),
                "translation_summary": root_rows_path.with_name("translation_summary.he.json"),
                "translation_publication": root_rows_path.with_name(
                    "translation_publication.he.json"
                ),
                "semantic_review_template": root_rows_path.with_name(
                    "translation_semantic_review_template.he.json"
                ),
                "semantic_review": root_rows_path.with_name("translation_semantic_review.he.json"),
            }
        }

    monkeypatch.setattr(data_provenance, "validate_full_paired_input_contract", validate)
    monkeypatch.setattr(experiment_module, "get_git_commit", lambda: "d" * 40)
    monkeypatch.setattr(experiment_module, "get_git_worktree_clean", lambda: True)

    def observed(*, run_dir: Path) -> dict[str, dict[str, int]]:
        assert run_dir.name == "v3-run"
        return {
            "train": {"en": 15_000, "he": 15_000, "total": 30_000},
            "validation": {"en": 1_000, "he": 1_000, "total": 2_000},
            "test": {"en": 1_000, "he": 1_000, "total": 2_000},
        }

    monkeypatch.setattr(
        data_provenance,
        "_validate_observed_cohorts",
        observed,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_tokenizer_cohort_join_is_order_independent_after_uniqueness_checks() -> None:
    formatted = [
        ("root-1", "en", "train", None),
        ("root-1:he", "he", "train", "root-1"),
    ]
    assert _same_cohort_identity_set(formatted, list(reversed(formatted)))


def _prompt_digest(language: str, index: int) -> str:
    return _sha256(f"{language}-prompt-{index}".encode())


def _prompt_set_digest(digests: list[str]) -> str:
    return _sha256("\n".join(digests).encode())


def _pair_set_digest(entries: list[dict[str, str]]) -> str:
    payload = "\n".join(
        json.dumps(entry, separators=(",", ":"), sort_keys=True) for entry in entries
    )
    return _sha256(payload.encode())


def _metric(numerator: int, denominator: int) -> dict[str, float | int]:
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def _metrics(correct: int, examples: int) -> dict[str, dict[str, float | int]]:
    metrics = {
        name: _metric(correct, examples)
        for name in (
            "valid_json_rate",
            "function_name_accuracy",
            "argument_exact_match",
            "full_call_exact_match",
        )
    }
    metrics["argument_f1"] = _metric(2 * correct, examples + correct)
    return metrics


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_test_config(run_dir: Path, *, run_id: str, marker: str) -> str:
    del run_id, marker
    payload = {
        "schema_version": "sommelier.config.v2",
        "project": {
            "name": "sommelier-v3-he-full",
            "artifact_root": "artifacts",
            "seed": 42,
        },
        "model": {
            "base_model_id": MODEL_IDENTITY["base_model_id"],
            "base_model_revision": MODEL_IDENTITY["base_model_revision"],
            "tokenizer_revision": MODEL_IDENTITY["tokenizer_revision"],
            "allow_remote_code": False,
        },
        "datasets": [
            {
                "language": "en",
                "dataset_id": "Salesforce/xlam-function-calling-60k",
                "dataset_revision": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
                "query_column": "query",
                "tools_column": "tools",
                "answers_column": "answers",
            },
            {
                "language": "he",
                "dataset_id": "abdelstark/sommelier-xlam-single-call-splits-he",
                "dataset_revision": "e" * 40,
                "query_column": "query",
                "tools_column": "tools",
                "answers_column": "answers",
                "source_id_column": "source_example_id",
            },
        ],
        "data": {
            "n_train": 15000,
            "n_validation": 1000,
            "n_test": 1000,
            "min_query_chars": 10,
            "max_query_chars": 2000,
            "dedupe_key": "normalized_query",
        },
        "formatting": {
            "system_prompt": (
                "You are a tool-calling model. Select the correct tool and return only "
                "the JSON tool call. Do not include explanations."
            ),
            "template_policy": "tokenizer_chat_template",
            "target_format": "json_tool_call",
        },
        "train": {
            "epochs": 2,
            "per_device_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.0002,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "max_sequence_length": 4096,
            "quantization": "nf4-4bit",
            "compute_dtype": "bfloat16",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "languages": ["en", "he"],
        },
        "eval": {
            "split": "test",
            "slices": ["en", "he"],
            **DECODING,
            "parser_version": "sommelier.parser.v1",
        },
        "remote": {
            "enabled": True,
            "gpu": "L40S",
            "data_timeout_seconds": 1800,
            "train_timeout_seconds": 43200,
            "eval_timeout_seconds": 18000,
        },
        "report": {"retain_raw_generations": True, "redact_fields": []},
        "tracking": {
            "enabled": False,
            "provider": "wandb",
            "project": "sommelier",
        },
    }
    config = SommelierConfig.model_validate(payload)
    _, digest = write_resolved_config(config, run_dir)
    return digest


def _write_eval_run(
    root: Path,
    *,
    run_id: str,
    model_kind: str,
    config_sha256: str,
    en_correct: bool,
    he_correct: bool,
) -> Path:
    run_dir = root / "artifacts" / "runs" / run_id
    config_sha256 = _write_test_config(
        run_dir,
        run_id=run_id,
        marker=config_sha256[:1],
    )
    (run_dir / "runtime_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "sommelier.runtime_metadata.v1",
                "run_id": run_id,
                "config_sha256": config_sha256,
                "stages": {},
                "hardware": {"gpu": "L40S", "source": "config"},
                "packages": {
                    "python": "3.13.0",
                    "torch": "2.11.0",
                    "transformers": "5.13.1",
                    "tokenizers": "0.22.2",
                    "accelerate": "1.14.0",
                    "peft": "0.18.0",
                    "bitsandbytes": "0.49.1",
                    "huggingface_hub": "1.23.0",
                    "datasets": "5.0.0",
                },
                "source_code": {
                    "git_commit": "d" * 40,
                    "working_tree_clean": True,
                    "boundary": "test dispatch boundary",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    eval_dir = run_dir / "eval" / model_kind
    formatted: list[dict[str, object]] = []
    per_language: dict[str, list[dict[str, object]]] = {"en": [], "he": []}
    pair_entries: list[dict[str, str]] = []
    for index in range(4):
        root_id = f"example-{index}"
        en_prompt = _prompt_digest("en", index)
        he_prompt = _prompt_digest("he", index)
        en: dict[str, object] = {
            "schema_version": FORMATTED_SCHEMA,
            "example_id": root_id,
            "split": "test",
            "language": "en",
            "source_example_id": None,
            "prompt_text": f"en prompt {index}",
            "target_text": json.dumps([GOLD_CALL], separators=(",", ":"), sort_keys=True),
            "prompt_sha256": en_prompt,
        }
        he: dict[str, object] = {
            "schema_version": FORMATTED_SCHEMA,
            "example_id": f"{root_id}:he",
            "split": "test",
            "language": "he",
            "source_example_id": root_id,
            "prompt_text": f"he prompt {index}",
            "target_text": json.dumps([GOLD_CALL], separators=(",", ":"), sort_keys=True),
            "prompt_sha256": he_prompt,
        }
        formatted.extend((en, he))
        per_language["en"].append(en)
        per_language["he"].append(he)
        pair_entries.append(
            {
                "reference_example_id": root_id,
                "target_example_id": f"{root_id}:he",
                "reference_prompt_sha256": en_prompt,
                "target_prompt_sha256": he_prompt,
            }
        )
    formatted_path = run_dir / "formatted" / "test.jsonl"
    _write_jsonl(formatted_path, formatted)

    correctness = {"en": en_correct, "he": he_correct}
    slices: dict[str, Any] = {}
    for language in ("en", "he"):
        generations: list[dict[str, object]] = []
        for example in per_language[language]:
            correct = correctness[language]
            generations.append(
                {
                    "schema_version": GENERATION_SCHEMA,
                    "example_id": example["example_id"],
                    "model_kind": model_kind,
                    "language": language,
                    "prompt_sha256": example["prompt_sha256"],
                    "raw_text": json.dumps(GOLD_CALL) if correct else "no call",
                    "parsed_call": GOLD_CALL if correct else None,
                    "parse_status": "ok" if correct else "no_json",
                    "decoding": DECODING,
                }
            )
        generations_path = eval_dir / f"generations.{language}.jsonl"
        _write_jsonl(generations_path, generations)
        slices[language] = {
            "metrics": _metrics(4 if correctness[language] else 0, 4),
            "examples": 4,
            "prompt_set_sha256": _prompt_set_digest(
                [str(example["prompt_sha256"]) for example in per_language[language]]
            ),
            "generation_artifact": (
                f"runs/{run_id}/eval/{model_kind}/generations.{language}.jsonl"
            ),
        }

    report = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "model_kind": model_kind,
        "model_identity": MODEL_IDENTITY,
        "config_sha256": config_sha256,
        "split": "test",
        "slices": slices,
        "paired_slices": {
            "he": {
                "reference_language": "en",
                "target_language": "he",
                "pairs": 4,
                "pair_set_sha256": _pair_set_digest(pair_entries),
            }
        },
        "metrics": _metrics((4 if en_correct else 0) + (4 if he_correct else 0), 8),
        "adapter_source": (
            None
            if model_kind == "base"
            else {
                "source": f"example/{run_id}",
                "revision": "c" * 40,
                "kind": "huggingface_repo",
                "tree_sha256": None,
                "artifact_path": None,
                "revision_is_immutable": True,
            }
        ),
        "parser_version": "sommelier.parser.v1",
        "test_split_sha256": _sha256(formatted_path.read_bytes()),
        "decoding": DECODING,
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_eval_manifests(eval_dir)
    return eval_dir


def _rewrite_report(eval_dir: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    path = eval_dir / "evaluation_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    mutate(report)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _write_eval_manifests(eval_dir: Path) -> None:
    report_path = eval_dir / "evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact_root = eval_dir.parents[3]
    run_dir = eval_dir.parents[1]
    model_kind = str(report["model_kind"])
    eval_stage = f"eval-{model_kind}"

    def artifact_ref(path: Path, kind: str, schema: str) -> dict[str, object]:
        return {
            "path": path.relative_to(artifact_root).as_posix(),
            "kind": kind,
            "schema_version": schema,
            "sha256": _sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
        }

    measured_outputs = [
        artifact_ref(
            eval_dir / f"generations.{language}.jsonl",
            "generations",
            GENERATION_SCHEMA,
        )
        for language in ("en", "he")
    ]
    telemetry_path = eval_dir / "inference_telemetry.json"
    if telemetry_path.exists():
        measured_outputs.append(
            artifact_ref(
                telemetry_path,
                "inference_telemetry",
                "sommelier.inference_telemetry.v2",
            )
        )
    outputs = [
        *measured_outputs,
        artifact_ref(report_path, "evaluation_report", SCHEMA),
    ]
    inputs = [
        artifact_ref(
            run_dir / "formatted" / "test.jsonl",
            "formatted_split",
            FORMATTED_SCHEMA,
        ),
        *measured_outputs,
    ]
    manifest = {
        "schema_version": "sommelier.manifest.v1",
        "stage": eval_stage,
        "status": "succeeded",
        "run_id": report["run_id"],
        "config_sha256": report["config_sha256"],
        "git_commit": "d" * 40,
        "inputs": inputs,
        "outputs": outputs,
    }
    manifest_path = run_dir / f"{eval_stage}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    root_path = run_dir / "manifest.json"
    root = (
        json.loads(root_path.read_text(encoding="utf-8"))
        if root_path.exists()
        else {
            "schema_version": "sommelier.manifest.v1",
            "run_id": report["run_id"],
            "stages": {},
            "config": artifact_ref(
                run_dir / "config.resolved.yaml",
                "config",
                "sommelier.config.v2",
            ),
            "status": "succeeded",
        }
    )
    root["stages"][eval_stage] = manifest_path.relative_to(artifact_root).as_posix()
    root_path.write_text(json.dumps(root, indent=2, sort_keys=True), encoding="utf-8")


def _add_inference_efficiency(eval_dir: Path) -> None:
    report_path = eval_dir / "evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact_root = eval_dir.parents[3]

    def artifact_ref(path: Path, kind: str, schema: str) -> dict[str, object]:
        return {
            "path": path.relative_to(artifact_root).as_posix(),
            "kind": kind,
            "schema_version": schema,
            "sha256": _sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
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
    elapsed = {"en": 4.0, "he": 8.0}
    telemetry_slices: dict[str, Any] = {}
    efficiency_slices: dict[str, Any] = {}
    for language in ("en", "he"):
        generation_path = eval_dir / f"generations.{language}.jsonl"
        generation_ref = artifact_ref(generation_path, "generations", "sommelier.generation.v2")
        raw = {
            "examples": 4,
            "elapsed_seconds": elapsed[language],
            "seconds_per_example": elapsed[language] / 4,
            "generation_artifact": generation_ref,
        }
        successes = report["slices"][language]["metrics"]["full_call_exact_match"]["numerator"]
        ratio = {
            "available": successes > 0,
            "value": round(elapsed[language] / successes, 6) if successes else None,
            "reason": None if successes else "zero_full_call_exact_successes",
            "unit": "gpu_seconds_per_full_call_exact_success",
            "full_call_exact_successes": successes,
            "basis": "generation_elapsed_seconds_x_configured_gpu_count",
        }
        telemetry_slices[language] = raw
        efficiency_slices[language] = {
            **raw,
            "gpu_seconds_per_full_call_exact_success": ratio,
        }

    total_successes = report["metrics"]["full_call_exact_match"]["numerator"]
    total = {"examples": 8, "elapsed_seconds": 12.0, "seconds_per_example": 1.5}
    overall_ratio = {
        "available": total_successes > 0,
        "value": round(12.0 / total_successes, 6) if total_successes else None,
        "reason": None if total_successes else "zero_full_call_exact_successes",
        "unit": "gpu_seconds_per_full_call_exact_success",
        "full_call_exact_successes": total_successes,
        "basis": "generation_elapsed_seconds_x_configured_gpu_count",
    }
    telemetry = {
        "schema_version": "sommelier.inference_telemetry.v2",
        "run_id": report["run_id"],
        "model_kind": report["model_kind"],
        "decoding": report["decoding"],
        "measurement": measurement,
        "timed_call_contract": timed_call_contract,
        "warmup": warmup,
        "sequential_run": sequential,
        "hardware": hardware,
        "slices": telemetry_slices,
        "total": total,
    }
    telemetry_path = eval_dir / "inference_telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry, indent=2, sort_keys=True), encoding="utf-8")
    report["inference_efficiency"] = {
        "available": True,
        "telemetry_artifact": artifact_ref(
            telemetry_path,
            "inference_telemetry",
            "sommelier.inference_telemetry.v2",
        ),
        "measurement": measurement,
        "timed_call_contract": timed_call_contract,
        "warmup": warmup,
        "sequential_run": sequential,
        "hardware": hardware,
        "slices": efficiency_slices,
        "overall": {
            **total,
            "gpu_seconds_per_full_call_exact_success": overall_ratio,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    _write_eval_manifests(eval_dir)


def _add_v3_data_contract(v3_dir: Path) -> None:
    run_dir = v3_dir.parents[1]
    artifact_root = v3_dir.parents[3]
    report = json.loads((v3_dir / "evaluation_report.json").read_text(encoding="utf-8"))

    def artifact_ref(path: Path, kind: str, schema: str) -> dict[str, object]:
        return {
            "path": path.relative_to(artifact_root).as_posix(),
            "kind": kind,
            "schema_version": schema,
            "sha256": _sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
        }

    source_dir = run_dir / "data" / "source_inputs"
    source_dir.mkdir(parents=True)
    source_files = (
        ("rows.en.jsonl", "raw_dataset", "sommelier.raw_tool_call_row.v1"),
        ("rows.en.he.jsonl", "raw_paired_dataset", "sommelier.raw_tool_call_row.v1"),
        (
            "translation_summary.he.json",
            "translation_summary",
            "sommelier.translation_summary.v2",
        ),
        (
            "translation_publication.he.json",
            "translation_publication_manifest",
            "sommelier.translation_publication_manifest.v1",
        ),
        (
            "translation_semantic_review_template.he.json",
            "translation_semantic_review_template",
            "sommelier.translation_semantic_review_template.v1",
        ),
        (
            "translation_semantic_review.he.json",
            "translation_semantic_review",
            "sommelier.translation_semantic_review.v1",
        ),
    )
    source_refs: list[dict[str, object]] = []
    for filename, kind, schema in source_files:
        path = source_dir / filename
        path.write_text("{}\n", encoding="utf-8")
        source_refs.append(artifact_ref(path, kind, schema))

    formatted_test_records = [
        json.loads(line)
        for line in (run_dir / "formatted" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    data_outputs: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        path = run_dir / "data" / f"{split}.jsonl"
        prepared_records = [
            {
                "schema_version": "sommelier.prepared_example.v2",
                "example_id": record["example_id"],
                "language": record["language"],
                "split": split,
                "source_example_id": record["source_example_id"],
            }
            for record in formatted_test_records
        ]
        _write_jsonl(path, prepared_records)
        data_outputs.append(artifact_ref(path, "dataset_split", "sommelier.prepared_example.v2"))
    drop_path = run_dir / "data" / "drop_summary.json"
    drop_path.write_text("{}\n", encoding="utf-8")
    data_outputs.append(artifact_ref(drop_path, "drop_summary", "sommelier.drop_summary.v2"))

    root = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    data_manifest = {
        "schema_version": "sommelier.manifest.v1",
        "stage": "data",
        "status": "succeeded",
        "run_id": "v3-run",
        "config_sha256": report["config_sha256"],
        "git_commit": "d" * 40,
        "inputs": [root["config"], *source_refs],
        "outputs": data_outputs,
    }
    data_manifest_path = run_dir / "data_manifest.json"
    data_manifest_path.write_text(
        json.dumps(data_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    formatted_refs: list[dict[str, object]] = []
    for split in ("train", "validation"):
        path = run_dir / "formatted" / f"{split}.jsonl"
        _write_jsonl(
            path,
            [{**record, "split": split} for record in formatted_test_records],
        )
    for split in ("train", "validation", "test"):
        formatted_refs.append(
            artifact_ref(
                run_dir / "formatted" / f"{split}.jsonl",
                "formatted_split",
                FORMATTED_SCHEMA,
            )
        )
    format_manifest = {
        "schema_version": "sommelier.manifest.v1",
        "stage": "format",
        "status": "succeeded",
        "run_id": "v3-run",
        "config_sha256": report["config_sha256"],
        "git_commit": "d" * 40,
        "inputs": data_outputs[:3],
        "outputs": formatted_refs,
    }
    format_manifest_path = run_dir / "format_manifest.json"
    format_manifest_path.write_text(
        json.dumps(format_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    token_dir = run_dir / "analysis" / "tokenization"
    token_dir.mkdir(parents=True)
    counts = {
        "en": {"query_tokens": 2, "prompt_tokens": 10, "full_tokens": 15},
        "he": {"query_tokens": 4, "prompt_tokens": 12, "full_tokens": 18},
    }
    records: list[dict[str, object]] = []
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
    records_path = token_dir / "tokenizer_tax_records.jsonl"
    _write_jsonl(records_path, records)
    records_ref = artifact_ref(
        records_path,
        "tokenizer_tax_records",
        "sommelier.tokenizer_tax_record.v1",
    )

    def paired_scope(multiplier: int) -> dict[str, object]:
        return {
            "coverage": {"paired": multiplier, "roots": multiplier, "ratio": 1.0},
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

    formatted_by_split = dict(zip(("train", "validation", "test"), formatted_refs))
    tokenizer_report = {
        "schema_version": "sommelier.tokenizer_tax_report.v1",
        "run_id": "v3-run",
        "config_sha256": report["config_sha256"],
        "tokenizer": {
            "id": MODEL_IDENTITY["tokenizer_id"],
            "revision": MODEL_IDENTITY["tokenizer_revision"],
        },
        "inputs": {
            split: {
                "path": ref["path"],
                "sha256": ref["sha256"],
                "bytes": ref["bytes"],
            }
            for split, ref in formatted_by_split.items()
        },
        "records": {
            "path": records_ref["path"],
            "sha256": records_ref["sha256"],
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
            "boundary": "Excludes dynamic padding.",
        },
    }
    tokenizer_report_path = token_dir / "tokenizer_tax_report.json"
    tokenizer_report_path.write_text(
        json.dumps(tokenizer_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    tokenizer_report_ref = artifact_ref(
        tokenizer_report_path,
        "tokenizer_tax_report",
        "sommelier.tokenizer_tax_report.v1",
    )
    tokenization_manifest = {
        "schema_version": "sommelier.manifest.v1",
        "stage": "tokenization",
        "status": "succeeded",
        "run_id": "v3-run",
        "config_sha256": report["config_sha256"],
        "git_commit": "d" * 40,
        "inputs": formatted_refs,
        "outputs": [records_ref, tokenizer_report_ref],
    }
    tokenization_manifest_path = run_dir / "tokenization_manifest.json"
    tokenization_manifest_path.write_text(
        json.dumps(tokenization_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    root["stages"].update(
        {
            "data": data_manifest_path.relative_to(artifact_root).as_posix(),
            "format": format_manifest_path.relative_to(artifact_root).as_posix(),
            "tokenization": tokenization_manifest_path.relative_to(artifact_root).as_posix(),
        }
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(root, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_three_arm_runs(
    tmp_path: Path,
    *,
    v1_en_correct: bool = True,
    v1_he_correct: bool = False,
    v3_en_correct: bool = True,
    v3_he_correct: bool = True,
) -> tuple[Path, Path, Path]:
    base_dir = _write_eval_run(
        tmp_path,
        run_id="base-run",
        model_kind="base",
        config_sha256="1" * 64,
        en_correct=False,
        he_correct=False,
    )
    v1_dir = _write_eval_run(
        tmp_path,
        run_id="v1-run",
        model_kind="adapter",
        config_sha256="2" * 64,
        en_correct=v1_en_correct,
        he_correct=v1_he_correct,
    )
    v3_dir = _write_eval_run(
        tmp_path,
        run_id="v3-run",
        model_kind="adapter",
        config_sha256="3" * 64,
        en_correct=v3_en_correct,
        he_correct=v3_he_correct,
    )
    for eval_dir in (base_dir, v1_dir, v3_dir):
        _add_inference_efficiency(eval_dir)
    _add_v3_data_contract(v3_dir)
    _add_preregistered_adapter_contract(v1_dir, v3_dir)
    return base_dir, v1_dir, v3_dir


def _add_preregistered_adapter_contract(v1_dir: Path, v3_dir: Path) -> None:
    _rewrite_report(
        v1_dir,
        lambda report: report.update(
            {
                "adapter_source": {
                    "source": ("abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora"),
                    "revision": "45a6e2fa3e29f8393ddf1e9bda51a9461b41ee0e",
                    "kind": "huggingface_repo",
                    "tree_sha256": None,
                    "artifact_path": None,
                    "revision_is_immutable": True,
                }
            }
        ),
    )
    _write_eval_manifests(v1_dir)

    run_dir = v3_dir.parents[1]
    artifact_root = v3_dir.parents[3]
    adapter_dir = run_dir / "train" / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA"}, sort_keys=True),
        encoding="utf-8",
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"v3-adapter-weights")
    tree_sha256 = adapter_tree_sha256(adapter_dir)
    _rewrite_report(
        v3_dir,
        lambda report: report.update(
            {
                "adapter_source": {
                    "source": str(adapter_dir.resolve()),
                    "revision": None,
                    "kind": "local_directory",
                    "tree_sha256": tree_sha256,
                    "artifact_path": "runs/v3-run/train/adapter",
                    "revision_is_immutable": True,
                }
            }
        ),
    )

    metrics_path = run_dir / "train" / "training_metrics.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "schema_version": "sommelier.training_metric.v1",
                "step": 1,
                "epoch": 1.0,
                "train_loss": 0.5,
                "eval_loss": 0.6,
                "learning_rate": 0.0002,
                "tokens_seen": 128,
                "peak_gpu_memory_mb": 4096,
            }
        ],
    )

    def artifact_ref(path: Path, kind: str, schema: str) -> dict[str, object]:
        return {
            "path": path.relative_to(artifact_root).as_posix(),
            "kind": kind,
            "schema_version": schema,
            "sha256": _sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
        }

    report = json.loads((v3_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    train_manifest = {
        "schema_version": "sommelier.manifest.v1",
        "stage": "train",
        "status": "succeeded",
        "run_id": "v3-run",
        "config_sha256": report["config_sha256"],
        "git_commit": "d" * 40,
        "inputs": [
            artifact_ref(
                run_dir / "formatted" / split,
                "formatted_split",
                FORMATTED_SCHEMA,
            )
            for split in ("train.jsonl", "validation.jsonl")
        ],
        "outputs": [
            *(artifact_ref(path, "adapter_weights", "") for path in sorted(adapter_dir.iterdir())),
            artifact_ref(
                metrics_path,
                "training_metrics",
                "sommelier.training_metric.v1",
            ),
        ],
        "details": {"train_languages": ["en", "he"]},
    }
    train_manifest_path = run_dir / "train_manifest.json"
    train_manifest_path.write_text(
        json.dumps(train_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    runtime_path = run_dir / "runtime_metadata.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["stages"]["train"] = {"elapsed_seconds": 60.0}
    runtime.update(
        {
            "peak_gpu_memory_mb": 4096,
            "observed_cost_usd": None,
            "cost_source": "unavailable",
        }
    )
    runtime_path.write_text(
        json.dumps(runtime, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    root_path = run_dir / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["stages"]["train"] = train_manifest_path.relative_to(artifact_root).as_posix()
    root_path.write_text(json.dumps(root, indent=2, sort_keys=True), encoding="utf-8")
    _write_eval_manifests(v3_dir)


def _validate_fixture_adapter_contract(v1_dir: Path, v3_dir: Path) -> None:
    _validate_preregistered_adapter_arms(
        _load_arm("v1_en", v1_dir, expected_kind="adapter"),
        _load_arm("v3_en_he", v3_dir, expected_kind="adapter"),
    )


def test_preregistered_adapter_contract_accepts_exact_v1_and_local_v3(
    tmp_path: Path,
) -> None:
    _, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)

    _validate_fixture_adapter_contract(v1_dir, v3_dir)


def test_preregistered_adapter_contract_rejects_wrong_immutable_v1(
    tmp_path: Path,
) -> None:
    _, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    _rewrite_report(
        v1_dir,
        lambda report: report["adapter_source"].update({"revision": "f" * 40}),
    )

    with pytest.raises(EvaluationError, match="committed Hebrew v3 baseline"):
        _validate_fixture_adapter_contract(v1_dir, v3_dir)


@pytest.mark.parametrize("substitution", ["external", "other_run"])
def test_preregistered_adapter_contract_rejects_arbitrary_v3_adapter(
    tmp_path: Path,
    substitution: str,
) -> None:
    _, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    if substitution == "external":
        replacement = {
            "source": "example/external-v3",
            "revision": "f" * 40,
            "kind": "huggingface_repo",
            "tree_sha256": None,
            "artifact_path": None,
            "revision_is_immutable": True,
        }
    else:
        replacement = {
            "source": str(
                (tmp_path / "artifacts" / "runs" / "other-run" / "train" / "adapter").resolve()
            ),
            "revision": None,
            "kind": "local_directory",
            "tree_sha256": "f" * 64,
            "artifact_path": "runs/other-run/train/adapter",
            "revision_is_immutable": True,
        }
    _rewrite_report(
        v3_dir,
        lambda report: report.update({"adapter_source": replacement}),
    )

    with pytest.raises(EvaluationError, match="canonical local adapter"):
        _validate_fixture_adapter_contract(v1_dir, v3_dir)


@pytest.mark.parametrize("substitution", ["same", "swapped"])
def test_preregistered_adapter_contract_rejects_same_or_swapped_adapters(
    tmp_path: Path,
    substitution: str,
) -> None:
    _, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    v1_path = v1_dir / "evaluation_report.json"
    v3_path = v3_dir / "evaluation_report.json"
    v1_report = json.loads(v1_path.read_text(encoding="utf-8"))
    v3_report = json.loads(v3_path.read_text(encoding="utf-8"))
    if substitution == "same":
        v3_report["adapter_source"] = v1_report["adapter_source"]
    else:
        v1_report["adapter_source"], v3_report["adapter_source"] = (
            v3_report["adapter_source"],
            v1_report["adapter_source"],
        )
        v1_path.write_text(
            json.dumps(v1_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    v3_path.write_text(
        json.dumps(v3_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="distinct|committed Hebrew v3 baseline"):
        _validate_fixture_adapter_contract(v1_dir, v3_dir)


@pytest.mark.parametrize(
    "failure",
    ["missing_manifest", "failed_manifest", "missing_metrics", "missing_runtime"],
)
def test_preregistered_adapter_contract_rejects_unavailable_training_evidence(
    tmp_path: Path,
    failure: str,
) -> None:
    _, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    run_dir = v3_dir.parents[1]
    if failure == "missing_manifest":
        (run_dir / "train_manifest.json").unlink()
        expected = "unavailable or incomplete"
    elif failure == "failed_manifest":
        manifest_path = run_dir / "train_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "failed"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        expected = "succeeded matching train manifest"
    elif failure == "missing_metrics":
        (run_dir / "train" / "training_metrics.jsonl").unlink()
        expected = "unavailable or incomplete"
    else:
        (run_dir / "runtime_metadata.json").unlink()
        expected = "unavailable or incomplete"

    with pytest.raises(EvaluationError, match=expected):
        _validate_fixture_adapter_contract(v1_dir, v3_dir)


def test_preregistered_adapter_contract_rejects_adapter_tree_drift(
    tmp_path: Path,
) -> None:
    _, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    adapter_path = v3_dir.parents[1] / "train" / "adapter" / "adapter_model.safetensors"
    adapter_path.write_bytes(b"substituted-after-evaluation")

    with pytest.raises(EvaluationError, match="tree sha256"):
        _validate_fixture_adapter_contract(v1_dir, v3_dir)


@pytest.mark.parametrize("artifact_kind", ["adapter_weights", "training_metrics"])
def test_preregistered_adapter_contract_rejects_unbound_training_artifact(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    _, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    manifest_path = v3_dir.parents[1] / "train_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = next(value for value in manifest["outputs"] if value["kind"] == artifact_kind)
    output["sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    expected = "adapter output" if artifact_kind == "adapter_weights" else "training metrics"
    with pytest.raises(EvaluationError, match=expected):
        _validate_fixture_adapter_contract(v1_dir, v3_dir)


def test_experiment_report_gates_three_arms_from_recomputed_evidence(
    tmp_path: Path,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)

    report = write_experiment_report(
        base_dir,
        v1_dir,
        v3_dir,
        tmp_path / "experiment",
        english_non_inferiority_margin=0.01,
        seed=42,
        resamples=2000,
    )

    assert report["schema_version"] == "sommelier.experiment_report.v1"
    assert set(report["arms"]) == {"base", "v1_en", "v3_en_he"}
    assert report["arms"]["v1_en"]["config_sha256"] == report["arms"]["v3_en_he"]["config_sha256"]
    for arm in report["arms"].values():
        for artifact in arm["artifacts"].values():
            assert artifact["path"].startswith("runs/")
            assert str(tmp_path) not in artifact["path"]
    comparison = report["comparisons"]["v3_vs_v1"]
    assert comparison["en"]["deltas"]["full_call_exact_match"] == 0.0
    assert comparison["he"]["deltas"]["full_call_exact_match"] == 1.0
    assert comparison["he"]["ci95"]["resamples"] == 2000
    assert comparison["en"]["mcnemar"]["discordant_pairs"] == 0
    assert comparison["en"]["mcnemar"]["p_value"] == 1.0
    assert comparison["he"]["mcnemar"] == {
        "method": "sommelier.exact_mcnemar.v1",
        "metric": "full_call_exact_match",
        "alternative": "two-sided",
        "pairs": 4,
        "discordant_pairs": 4,
        "discordant_counts": {
            "reference_correct_candidate_incorrect": 0,
            "reference_incorrect_candidate_correct": 4,
        },
        "p_value": 0.125,
    }
    assert report["claims"]["hebrew_full_call_uplift"]["passed"] is True
    assert report["claims"]["english_full_call_non_inferiority"]["passed"] is True
    assert len(report["approved_claims"]) == 2
    assert report["preregistration"] == {
        "schema_version": "sommelier.hebrew_v3_preregistration.v1",
        "status": "committed_in_source_before_full_results",
        "english_non_inferiority_margin": 0.01,
        "bootstrap": {
            "seed": 42,
            "resamples": 2000,
            "confidence_level": 0.95,
            "method": "sommelier.paired_bootstrap.v1",
        },
        "primary_claim_rules": {
            "hebrew_full_call_uplift": "95% lower bound > 0",
            "english_full_call_non_inferiority": "95% lower bound >= -0.01",
        },
        "finalizer_source_code": {
            "git_commit": "d" * 40,
            "working_tree_clean": True,
            "boundary": "Observed before loading experiment outcome artifacts.",
        },
    }
    tco = report["sovereign_tco_evidence"]
    assert tco["schema_version"] == "sommelier.sovereign_tco_evidence.v1"
    assert tco["paired_tokenization"]["available"] is True
    assert (
        tco["paired_tokenization"]["paired_scopes"]["all"]["token_ratios"]["query_tokens"][
            "hebrew_to_english_ratio"
        ]
        == 2.0
    )
    assert tco["qlora_training"]["currency_cost"]["available"] is False
    assert tco["qlora_training"]["full_finetune_savings"]["available"] is False
    assert report["data_provenance"]["contract"]["semantic_review"] == {
        "sample_size": 200,
        "required_critical_errors": 0,
        "status": "validated",
    }
    assert (
        tco["sources"]["data_provenance"]["he_semantic_review"]["sha256"]
        == (report["data_provenance"]["sources"]["he_semantic_review"]["sha256"])
    )
    assert (
        json.loads((tmp_path / "experiment" / "experiment_report.json").read_text(encoding="utf-8"))
        == report
    )


def test_experiment_report_propagates_semantic_publication_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)

    def reject(_config: SommelierConfig, _root: Path) -> dict[str, dict[str, Path]]:
        raise UserInputError("semantic publication gate did not pass")

    monkeypatch.setattr(data_provenance, "validate_full_paired_input_contract", reject)
    output = tmp_path / "experiment"
    with pytest.raises(UserInputError, match="semantic publication gate"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            output,
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )
    assert not (output / "experiment_report.json").exists()


def test_experiment_report_rejects_format_output_disconnected_from_manifest(
    tmp_path: Path,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    train_path = v3_dir.parents[1] / "formatted" / "train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(EvaluationError, match="format outputs"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_rejects_truncated_full_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    monkeypatch.setattr(
        data_provenance,
        "_validate_observed_cohorts",
        _REAL_VALIDATE_OBSERVED_COHORTS,
    )

    with pytest.raises(EvaluationError, match="expected 15000"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_integrates_three_arm_inference_tco_evidence(
    tmp_path: Path,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    for eval_dir in (base_dir, v1_dir, v3_dir):
        _add_inference_efficiency(eval_dir)

    report = write_experiment_report(
        base_dir,
        v1_dir,
        v3_dir,
        tmp_path / "experiment",
        english_non_inferiority_margin=0.01,
        seed=42,
        resamples=2000,
    )

    inference = report["sovereign_tco_evidence"]["inference_efficiency"]
    assert set(inference["arms"]) == {"base", "v1_en", "v3_en_he"}
    assert inference["cross_arm_comparability"] == {
        "available": True,
        "configured_gpu": {"label": "L40S", "count": 1},
        "observed_packages": {
            "python": "3.13.0",
            "torch": "2.11.0",
            "transformers": "5.13.1",
            "tokenizers": "0.22.2",
            "accelerate": "1.14.0",
            "peft": "0.18.0",
            "bitsandbytes": "0.49.1",
            "huggingface_hub": "1.23.0",
            "datasets": "5.0.0",
        },
        "boundary": "Identical sequential end-to-end generator-call measurement contract.",
    }
    assert (
        inference["arms"]["v3_en_he"]["slices"]["he"][
            "configured_gpu_seconds_per_full_call_exact_success"
        ]["value"]
        == 2.0
    )
    assert (
        inference["arms"]["base"]["slices"]["he"][
            "configured_gpu_seconds_per_full_call_exact_success"
        ]["reason"]
        == "zero_full_call_exact_successes"
    )


def test_experiment_report_rejects_cross_arm_runtime_package_drift(
    tmp_path: Path,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    runtime_path = v1_dir.parents[1] / "runtime_metadata.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["packages"]["transformers"] = "5.13.2"
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(EvaluationError, match="comparable inference telemetry"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


@pytest.mark.parametrize("package", ["accelerate", "tokenizers"])
def test_experiment_report_requires_complete_inference_runtime_identity(
    tmp_path: Path,
    package: str,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    for eval_dir in (base_dir, v1_dir, v3_dir):
        runtime_path = eval_dir.parents[1] / "runtime_metadata.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime["packages"][package] = "absent"
        runtime_path.write_text(
            json.dumps(runtime, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    with pytest.raises(EvaluationError, match=f"missing required inference package {package}"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


@pytest.mark.parametrize("margin", [0.0, -0.01, float("inf"), float("nan")])
def test_experiment_report_requires_positive_finite_english_margin(
    tmp_path: Path, margin: float
) -> None:
    with pytest.raises(UserInputError, match="positive finite"):
        write_experiment_report(
            tmp_path / "base",
            tmp_path / "v1",
            tmp_path / "v3",
            tmp_path / "out",
            english_non_inferiority_margin=margin,
            seed=1,
            resamples=10,
        )


@pytest.mark.parametrize(
    ("margin", "seed", "resamples"),
    [
        (0.02, 42, 2000),
        (0.01, 41, 2000),
        (0.01, 42, 1),
    ],
)
def test_experiment_report_rejects_posthoc_claim_gate_tuning(
    tmp_path: Path,
    margin: float,
    seed: int,
    resamples: int,
) -> None:
    with pytest.raises(UserInputError, match="committed Hebrew v3 preregistration"):
        write_experiment_report(
            tmp_path / "base",
            tmp_path / "v1",
            tmp_path / "v3",
            tmp_path / "out",
            english_non_inferiority_margin=margin,
            seed=seed,
            resamples=resamples,
        )


def test_experiment_report_rejects_dirty_finalizer_before_loading_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment_module, "get_git_worktree_clean", lambda: False)

    with pytest.raises(EvaluationError, match="finalizer is not running from a clean"):
        write_experiment_report(
            tmp_path / "base",
            tmp_path / "v1",
            tmp_path / "v3",
            tmp_path / "out",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_rejects_finalizer_revision_different_from_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    monkeypatch.setattr(experiment_module, "get_git_commit", lambda: "e" * 40)

    with pytest.raises(EvaluationError, match="finalizer revision does not match"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_rejects_mismatched_pinned_model_identity(
    tmp_path: Path,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    _rewrite_report(
        v3_dir,
        lambda report: report["model_identity"].update({"tokenizer_revision": "d" * 40}),
    )

    with pytest.raises(EvaluationError, match="model_identity.tokenizer_revision"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_rejects_mismatched_tokenizer_id(tmp_path: Path) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    _rewrite_report(
        v3_dir,
        lambda report: report["model_identity"].update({"tokenizer_id": "example/other-tokenizer"}),
    )

    with pytest.raises(EvaluationError, match="model_identity.tokenizer_id"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


@pytest.mark.parametrize("field", ["base_model_revision", "tokenizer_revision"])
def test_experiment_report_rejects_shared_mutable_model_revision(
    tmp_path: Path,
    field: str,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)

    def set_mutable(report: dict[str, Any]) -> None:
        report["model_identity"].update({field: "main"})

    for eval_dir in (base_dir, v1_dir, v3_dir):
        _rewrite_report(eval_dir, set_mutable)

    with pytest.raises(EvaluationError, match=f"mutable model_identity.{field}"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


@pytest.mark.parametrize("arm", ["v1", "v3"])
def test_experiment_report_rejects_mutable_adapter_revision(
    tmp_path: Path,
    arm: str,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    adapter_dir = v1_dir if arm == "v1" else v3_dir
    _rewrite_report(
        adapter_dir,
        lambda report: report["adapter_source"].update(
            {"revision": "main", "revision_is_immutable": False}
        ),
    )

    with pytest.raises(EvaluationError, match="immutable adapter identity"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_downloaded_run_resolves_local_adapter_by_canonical_artifact_path(
    tmp_path: Path,
) -> None:
    _, _, v3_dir = _write_three_arm_runs(tmp_path)
    adapter_dir = v3_dir.parents[1] / "train" / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"downloaded-weights")
    tree_sha256 = adapter_tree_sha256(adapter_dir)
    _rewrite_report(
        v3_dir,
        lambda report: report.update(
            {
                "adapter_source": {
                    "source": "/artifacts/artifacts/runs/v3-run/train/adapter",
                    "revision": None,
                    "kind": "local_directory",
                    "tree_sha256": tree_sha256,
                    "artifact_path": "runs/v3-run/train/adapter",
                    "revision_is_immutable": True,
                }
            }
        ),
    )

    arm = _load_arm("v3_en_he", v3_dir, expected_kind="adapter")
    training = _training_tco_input(arm)

    assert training.local_adapter is not None
    assert training.local_adapter.tree_sha256 == tree_sha256
    assert {artifact.path for artifact in training.local_adapter.files} == {
        "runs/v3-run/train/adapter/adapter_config.json",
        "runs/v3-run/train/adapter/adapter_model.safetensors",
    }


def test_experiment_report_rejects_paired_cohort_digest_not_backed_by_artifacts(
    tmp_path: Path,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    _rewrite_report(
        v1_dir,
        lambda report: report["paired_slices"]["he"].update({"pair_set_sha256": "0" * 64}),
    )

    with pytest.raises(EvaluationError, match="paired cohort digest"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_rejects_generation_example_order_drift(tmp_path: Path) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    generations_path = v3_dir / "generations.he.jsonl"
    records = [json.loads(line) for line in generations_path.read_text().splitlines()]
    _write_jsonl(generations_path, list(reversed(records)))

    with pytest.raises(EvaluationError, match="generation example IDs"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_rejects_generation_decoding_drift(tmp_path: Path) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    generations_path = v3_dir / "generations.he.jsonl"
    records = [json.loads(line) for line in generations_path.read_text().splitlines()]
    records[0]["decoding"] = {**DECODING, "max_new_tokens": 65}
    _write_jsonl(generations_path, records)

    with pytest.raises(EvaluationError, match="generation decoding"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_reparses_raw_generation_text(tmp_path: Path) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    generations_path = v1_dir / "generations.he.jsonl"
    records = [json.loads(line) for line in generations_path.read_text().splitlines()]
    assert records[0]["raw_text"] == "no call"
    records[0]["parsed_call"] = GOLD_CALL
    records[0]["parse_status"] = "ok"
    _write_jsonl(generations_path, records)

    with pytest.raises(EvaluationError, match="parse result does not match raw_text"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_rejects_unsucceeded_v3_eval_manifest(
    tmp_path: Path,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    for eval_dir in (base_dir, v1_dir, v3_dir):
        _add_inference_efficiency(eval_dir)
    manifest_path = v3_dir.parents[1] / "eval-adapter_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "failed"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(EvaluationError, match="not a succeeded eval-adapter stage"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_requires_succeeded_root_for_every_arm(
    tmp_path: Path,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    root_path = v1_dir.parents[1] / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["status"] = "failed"
    root_path.write_text(json.dumps(root, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(EvaluationError, match="v1_en root run manifest is not succeeded"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_requires_one_generation_source_revision(
    tmp_path: Path,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    v1_run_dir = v1_dir.parents[1]
    runtime_path = v1_run_dir / "runtime_metadata.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["source_code"]["git_commit"] = "f" * 40
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True), encoding="utf-8")
    eval_manifest_path = v1_run_dir / "eval-adapter_manifest.json"
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    eval_manifest["git_commit"] = "f" * 40
    eval_manifest_path.write_text(
        json.dumps(eval_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(EvaluationError, match="source-code revisions differ"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_rejects_metrics_not_backed_by_generations(
    tmp_path: Path,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    _rewrite_report(
        v1_dir,
        lambda report: report["slices"]["en"]["metrics"]["full_call_exact_match"].update(
            {"value": 0.5}
        ),
    )

    with pytest.raises(EvaluationError, match="metrics do not match"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_requires_base_then_two_adapter_arms(tmp_path: Path) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(tmp_path)
    _rewrite_report(v1_dir, lambda report: report.update({"model_kind": "base"}))

    with pytest.raises(EvaluationError, match="must contain adapter generations"):
        write_experiment_report(
            base_dir,
            v1_dir,
            v3_dir,
            tmp_path / "experiment",
            english_non_inferiority_margin=0.01,
            seed=42,
            resamples=2000,
        )


def test_experiment_report_withholds_claims_when_confidence_bounds_fail(
    tmp_path: Path,
) -> None:
    base_dir, v1_dir, v3_dir = _write_three_arm_runs(
        tmp_path,
        v1_he_correct=True,
        v3_en_correct=False,
    )

    report = write_experiment_report(
        base_dir,
        v1_dir,
        v3_dir,
        tmp_path / "experiment",
        english_non_inferiority_margin=0.01,
        seed=42,
        resamples=2000,
    )

    assert report["all_claims_passed"] is False
    assert report["approved_claims"] == []
    assert report["claims"]["hebrew_full_call_uplift"]["passed"] is False
    assert "statement" not in report["claims"]["hebrew_full_call_uplift"]
    assert report["claims"]["english_full_call_non_inferiority"]["passed"] is False
    assert "statement" not in report["claims"]["english_full_call_non_inferiority"]
