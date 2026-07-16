from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

import sommelier.publication as publication
import sommelier.release as release
from sommelier.artifacts import sha256_file
from sommelier.config import SommelierConfig, load_config
from sommelier.data.openai_evidence import OPENAI_PROVIDER_JOURNAL_FILENAME
from sommelier.data.prepare import paired_input_path
from sommelier.data.semantic_review import (
    SEMANTIC_REVIEW_FILENAME,
    SEMANTIC_REVIEW_TEMPLATE_FILENAME,
)
from sommelier.data.translate import (
    PUBLICATION_MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    TRANSLATION_CONFIG_FILENAME,
    TRANSLATION_RUN_IDENTITY_FILENAME,
    TRANSLATION_RUN_IDENTITY_SCHEMA,
    translation_provenance_sidecar_path,
    translation_selection_contract_sha256,
    validate_hebrew_v3_translation_run_identity,
)
from sommelier.errors import (
    ExternalDependencyError,
    SecurityPolicyError,
    UserInputError,
)
from sommelier.evaluation.data_provenance import HEBREW_V3_PAIRED_DATASET_ID
from sommelier.evaluation.experiment import EXPERIMENT_REPORT_SCHEMA
from sommelier.evaluation.generate import (
    GENERATION_TIMING_AGGREGATION,
    GENERATION_TIMING_SCOPE,
    SEQUENTIAL_RUN_BOUNDARY,
    inference_timed_call_contract,
    inference_warmup_contract,
)
from sommelier.evaluation.statistics import stable_bootstrap_seed
from sommelier.hebrew_v3_preregistration import (
    reviewer_anchor_payload,
    reviewer_anchor_sha256,
)
from sommelier.publication import (
    ADAPTER_LICENSE_FILE_SHA256,
    ADAPTER_UPLOAD_OPTIONAL_FILES,
    ADAPTER_UPLOAD_REQUIRED_FILES,
    DATASET_REQUIRED_FILES,
    PreparedPublication,
    prepare_hebrew_adapter_publication,
    prepare_hebrew_dataset_publication,
)
from sommelier.publication import (
    _publish_prepared_bundle as publish_prepared_bundle,
)
from sommelier.release import ACK_ENV_NAME, REQUIRED_DERIVED_NOTICE, run_release_preflight
from sommelier.reviewer import canonical_reviewer_requirement

_REVIEWER_PUBLIC_KEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f"
)
_REVIEWER_REQUIREMENT = canonical_reviewer_requirement(
    "fixture-reviewer",
    _REVIEWER_PUBLIC_KEY,
)


class FakeHubClient:
    def __init__(
        self,
        root: Path,
        *,
        repository_exists: bool = True,
        existing_files: set[str] | None = None,
    ) -> None:
        self.root = root
        self.repository_exists = repository_exists
        self.existing_files = (
            set(existing_files) if existing_files is not None else {".gitattributes"}
        )
        self.revision = "a" * 40
        self.head_revision: str | None = "b" * 40
        self.parent_commits: list[str | None] = []
        self.post_commit_files: set[str] | None = None
        self.corrupt_file: str | None = None
        self.symlink_download: str | None = None
        self.fail_commit = False
        self.events: list[str] = []
        self.committed: dict[str, bytes] = {}
        self.commit_messages: list[str] = []

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: publication.PublicationRepoType,
    ) -> None:
        self.events.append(f"create:{repo_type}:{repo_id}")
        if self.repository_exists:
            raise RuntimeError("repository already exists")
        self.repository_exists = True
        self.existing_files = {".gitattributes"} if self.head_revision is not None else set()

    def list_files(
        self,
        *,
        repo_id: str,
        repo_type: publication.PublicationRepoType,
        revision: str | None,
    ) -> Sequence[str]:
        del repo_id, repo_type
        self.events.append(f"list:{revision or 'head'}")
        if not self.repository_exists:
            raise RuntimeError("repository not found")
        if revision is None or not self.committed:
            return sorted(self.existing_files)
        if self.post_commit_files is not None:
            return sorted(self.post_commit_files)
        return sorted(set(self.committed) | {".gitattributes"})

    def resolve_revision(
        self,
        *,
        repo_id: str,
        repo_type: publication.PublicationRepoType,
    ) -> str | None:
        del repo_id, repo_type
        self.events.append("resolve")
        if not self.repository_exists:
            raise RuntimeError("repository not found")
        return self.head_revision

    def create_commit(
        self,
        *,
        repo_id: str,
        repo_type: publication.PublicationRepoType,
        files: Mapping[str, Path],
        commit_message: str,
        parent_commit: str | None,
    ) -> str:
        del repo_id, repo_type
        self.events.append("commit")
        if self.fail_commit:
            raise RuntimeError("commit failed")
        self.parent_commits.append(parent_commit)
        self.commit_messages.append(commit_message)
        self.committed = {name: path.read_bytes() for name, path in files.items()}
        return self.revision

    def inspect_commit(
        self,
        *,
        repo_id: str,
        repo_type: publication.PublicationRepoType,
        revision: str,
    ) -> publication._HubCommitMetadata:
        del repo_id, repo_type
        self.events.append(f"inspect:{revision}")
        if revision != self.revision or not self.parent_commits or not self.commit_messages:
            raise RuntimeError("commit not found")
        return publication._HubCommitMetadata(
            parent_commit=self.parent_commits[-1],
            title=self.commit_messages[-1],
        )

    def download_file(
        self,
        *,
        repo_id: str,
        repo_type: publication.PublicationRepoType,
        filename: str,
        revision: str,
    ) -> Path:
        del repo_id, repo_type, revision
        self.events.append(f"download:{filename}")
        destination = self.root / "downloads" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if filename == self.symlink_download:
            destination.symlink_to(self.root / "missing-target")
            return destination
        data = self.committed[filename]
        if filename == self.corrupt_file:
            data += b"corrupt"
        destination.write_bytes(data)
        return destination


def _write_files(root: Path, names: Sequence[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {name}\n", encoding="utf-8")
        files[name] = path
    return files


def _prepared_dataset(tmp_path: Path) -> PreparedPublication:
    return PreparedPublication(
        repo_type="dataset",
        files=_write_files(tmp_path / "dataset", sorted(DATASET_REQUIRED_FILES)),
    )


def _prepared_adapter(tmp_path: Path, *, optional: bool = False) -> PreparedPublication:
    names = set(ADAPTER_UPLOAD_REQUIRED_FILES)
    if optional:
        names.update(ADAPTER_UPLOAD_OPTIONAL_FILES)
    return PreparedPublication(
        repo_type="model",
        files=_write_files(tmp_path / "adapter", sorted(names)),
    )


def _hebrew_config(
    tmp_path: Path,
    *,
    dataset_revision: str,
    filename: str = "config.yaml",
) -> tuple[SommelierConfig, Path]:
    text = Path("examples/config.v3-he-full.yaml").read_text(encoding="utf-8")
    assert text.count("dataset_revision: main") == 1
    reviewer = _REVIEWER_REQUIREMENT
    anchored = text.replace(
        "dataset_revision: main",
        f"dataset_revision: {dataset_revision}",
    ) + (
        "\nsemantic_review:\n"
        "  reviewer:\n"
        f"    reviewer_id: {reviewer.reviewer_id}\n"
        f"    ssh_public_key: {reviewer.ssh_public_key}\n"
        f"    public_key_fingerprint: {reviewer.public_key_fingerprint}\n"
    )
    config_path = tmp_path / filename
    config_path.write_text(anchored, encoding="utf-8")
    return load_config(config_path), config_path


def _phase_a_hebrew_config(tmp_path: Path) -> tuple[SommelierConfig, Path]:
    return _hebrew_config(tmp_path, dataset_revision="main")


def _immutable_hebrew_config(tmp_path: Path) -> tuple[SommelierConfig, Path]:
    return _hebrew_config(tmp_path, dataset_revision="e" * 40)


def _dataset_bundle(
    root: Path,
    config_path: Path,
    *,
    source_revision: str = "f" * 40,
) -> Path:
    phase_a_config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    provider_sdk_version = "2.45.0"
    list_price_limit_usd = "50.00"
    reviewer_preregistration = reviewer_anchor_payload(phase_a_config)
    reviewer_preregistration_sha256 = reviewer_anchor_sha256(phase_a_config)
    source_code = {
        "git_commit": source_revision,
        "working_tree_clean": True,
        "boundary": "Fixture source identity captured before provider access.",
    }
    selection = {
        "config_sha256": config_sha256,
        "contract_sha256": translation_selection_contract_sha256(
            phase_a_config,
            mode="full",
            max_rows=60_000,
            limit=0,
        ),
        "mode": "full",
        "max_rows": 60_000,
        "limit": 0,
        "seed": phase_a_config.project.seed,
        "selected_rows": 17_000,
        "selected_source_ids_sha256": "2" * 64,
    }
    translator = {
        "model_id": "gpt-5.5-2026-04-23",
        "model_revision": "gpt-5.5-2026-04-23",
        "request_sha256": "3" * 64,
        "implementation_revision": source_code["git_commit"],
        "provider_request": {"sdk_version": provider_sdk_version},
    }
    runtime = {
        "backend": "openai_responses",
        "translation_chunk_size": 32,
        "gpu_allocation_label": None,
        "function_timeout_seconds": 14_400,
        "provider_service_tier": "flex",
        "provider_timeout_seconds": 900.0,
        "provider_max_workers": 8,
        "openai_list_price_ceiling": {"limit_usd": list_price_limit_usd},
    }
    run_identity = {
        "schema_version": TRANSLATION_RUN_IDENTITY_SCHEMA,
        "run_id": "fixture-hebrew-v3-full",
        "config_sha256": config_sha256,
        "selection": {
            field: selection[field]
            for field in ("contract_sha256", "mode", "max_rows", "limit", "seed")
        },
        "translator": {
            field: translator[field]
            for field in (
                "model_id",
                "model_revision",
                "request_sha256",
                "implementation_revision",
            )
        }
        | {"max_attempts": 3},
        "runtime": {
            "backend": runtime["backend"],
            "translation_chunk_size": runtime["translation_chunk_size"],
            "allocation_gpu": runtime["gpu_allocation_label"],
            "function_timeout_seconds": runtime["function_timeout_seconds"],
            "provider_service_tier": runtime["provider_service_tier"],
            "provider_sdk_version": provider_sdk_version,
            "provider_timeout_seconds": runtime["provider_timeout_seconds"],
            "provider_max_workers": runtime["provider_max_workers"],
            "openai_list_price_limit_usd": list_price_limit_usd,
        },
        "source_code": source_code,
        "reviewer_preregistration": reviewer_preregistration,
        "reviewer_preregistration_sha256": reviewer_preregistration_sha256,
    }
    root.mkdir()
    identity_path = root / TRANSLATION_RUN_IDENTITY_FILENAME
    identity_path.write_text(
        json.dumps(run_identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "max_attempts": 3,
        "selection": selection,
        "translator": translator,
        "runtime": runtime,
        "source_code": source_code,
        "reviewer_preregistration": reviewer_preregistration,
        "reviewer_preregistration_sha256": reviewer_preregistration_sha256,
        "translation_run_identity_sha256": sha256_file(identity_path),
    }
    for name in DATASET_REQUIRED_FILES:
        path = root / name
        if name == TRANSLATION_RUN_IDENTITY_FILENAME:
            continue
        if name == "README.md":
            path.write_text(
                "---\nlicense: cc-by-4.0\n---\n"
                "# Hebrew machine-translated data\n"
                "Derived from Salesforce/xlam-function-calling-60k.\n",
                encoding="utf-8",
            )
        elif name == TRANSLATION_CONFIG_FILENAME:
            shutil.copy2(config_path, path)
        elif name == SUMMARY_FILENAME:
            path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
        elif name.endswith(".jsonl"):
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text("{}\n", encoding="utf-8")
    return root


def _commit_config(config_path: Path) -> str:
    repository = config_path.parent
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", config_path.name], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "Phase A config"],
        cwd=repository,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _adapter_bundle(root: Path, config_path: Path, *, optional: bool = True) -> Path:
    root.mkdir()
    for name in publication.ADAPTER_REQUIRED_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in ADAPTER_LICENSE_FILE_SHA256:
            shutil.copy2(Path("licenses") / name, path)
        elif name == "config.resolved.yaml":
            shutil.copy2(config_path, path)
        elif name.endswith(".json"):
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text(f"fixture for {name}\n", encoding="utf-8")
    if optional:
        tokenizer = root / "adapter" / "tokenizer.json"
        tokenizer.write_text("{}\n", encoding="utf-8")
    return root


def _valid_adapter_card(
    config: SommelierConfig,
    *,
    experiment_sha256: str,
    tree_sha256: str,
    source_revision: str,
    dataset_revision: str,
    experiment_report: Mapping[str, object] | None = None,
) -> str:
    lines = [
        "---",
        "license: llama3.1",
        f"base_model: {config.model.base_model_id}",
        "---",
        "# Llama Hebrew tool-calling adapter",
        REQUIRED_DERIVED_NOTICE,
        "NVIDIA Open Model License",
        "Llama 3.1 Community License",
        config.model.base_model_id,
        config.model.base_model_revision,
        experiment_sha256,
        tree_sha256,
        source_revision,
        dataset_revision,
    ]
    if experiment_report is not None:
        lines.extend(
            (
                "",
                publication.render_hebrew_v3_claim_section(experiment_report),
                "",
                "## Limitations",
                "Fixture limitations.",
            )
        )
    return "\n".join(lines)


_HEBREW_UPLIFT_STATEMENT = (
    "The v3 en+he adapter improves Hebrew full-call exact match "
    "over the v1 English adapter on the gated cohort."
)
_ENGLISH_NON_INFERIORITY_STATEMENT = (
    "The v3 en+he adapter is non-inferior to the v1 English adapter "
    "on English full-call exact match at the declared margin."
)


def _metric_payload(numerator: int, *, denominator: int = 4) -> dict[str, object]:
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def _metrics_payload(numerator: int, *, denominator: int = 4) -> dict[str, dict[str, object]]:
    metrics = {
        name: _metric_payload(numerator, denominator=denominator)
        for name in (
            "valid_json_rate",
            "function_name_accuracy",
            "argument_exact_match",
            "full_call_exact_match",
        )
    }
    metrics["argument_f1"] = _metric_payload(
        numerator * 2,
        denominator=denominator * 2,
    )
    return metrics


def _paired_slice_payload(
    *,
    reference_numerator: int,
    target_numerator: int,
) -> dict[str, object]:
    reference_metrics = _metrics_payload(reference_numerator)
    target_metrics = _metrics_payload(target_numerator)
    gap = (target_numerator - reference_numerator) / 4
    return {
        "reference_language": "en",
        "target_language": "he",
        "pairs": 4,
        "coverage": {
            "paired": 4,
            "reference_slice_examples": 4,
            "target_slice_examples": 4,
            "reference_fraction": 1.0,
        },
        "pair_set_sha256": "a" * 64,
        "reference": {"metrics": reference_metrics},
        "target": {"metrics": target_metrics},
        "gaps": {name: gap for name in reference_metrics},
        "gap_ci95": {
            "method": "sommelier.paired_bootstrap.v1",
            "seed": stable_bootstrap_seed(42, "language-gap:en:he"),
            "confidence_level": 0.95,
            "resamples": 2000,
            "intervals": {name: {"lower": gap, "upper": gap} for name in reference_metrics},
        },
    }


def _arm_payload(
    *,
    run_id: str,
    model_kind: str,
    adapter_source: Mapping[str, object] | None,
    en_numerator: int,
    he_numerator: int,
    config_sha256: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "model_kind": model_kind,
        "config_sha256": config_sha256,
        "adapter_source": dict(adapter_source) if adapter_source is not None else None,
        "metrics": {
            "overall": _metrics_payload(en_numerator + he_numerator, denominator=8),
            "slices": {
                "en": _metrics_payload(en_numerator),
                "he": _metrics_payload(he_numerator),
            },
        },
        "paired_slices": {
            "he": _paired_slice_payload(
                reference_numerator=en_numerator,
                target_numerator=he_numerator,
            )
        },
        "artifacts": {
            "evaluation_report": {
                "path": f"runs/{run_id}/eval/{model_kind}/evaluation_report.json",
                "sha256": "3" * 64,
            },
            "formatted_test": {
                "path": f"runs/{run_id}/formatted/test.jsonl",
                "sha256": "4" * 64,
            },
            "generations.en": {
                "path": f"runs/{run_id}/eval/{model_kind}/generations.en.jsonl",
                "sha256": "5" * 64,
            },
            "generations.he": {
                "path": f"runs/{run_id}/eval/{model_kind}/generations.he.jsonl",
                "sha256": "6" * 64,
            },
        },
    }


def _comparison_payload(
    *,
    seed: int,
    delta: float,
    lower: float,
    upper: float,
) -> dict[str, object]:
    return {
        "deltas": {
            name: delta
            for name in (
                "valid_json_rate",
                "function_name_accuracy",
                "argument_exact_match",
                "argument_f1",
                "full_call_exact_match",
            )
        },
        "ci95": {
            "method": "sommelier.paired_bootstrap.v1",
            "seed": seed,
            "confidence_level": 0.95,
            "resamples": 2000,
            "intervals": {
                name: {"lower": lower, "upper": upper}
                for name in (
                    "valid_json_rate",
                    "function_name_accuracy",
                    "argument_exact_match",
                    "argument_f1",
                    "full_call_exact_match",
                )
            },
        },
        "mcnemar": {
            "method": "sommelier.exact_mcnemar.v1",
            "metric": "full_call_exact_match",
            "alternative": "two-sided",
            "pairs": 4,
            "discordant_pairs": 0,
            "discordant_counts": {
                "reference_correct_candidate_incorrect": 0,
                "reference_incorrect_candidate_correct": 0,
            },
            "p_value": 1.0,
        },
    }


def _tco_artifact_source(
    path: str,
    kind: str,
    schema_version: str,
    *,
    sha256: str = "1" * 64,
    bytes_count: int = 100,
) -> dict[str, object]:
    return {
        "path": path,
        "kind": kind,
        "schema_version": schema_version,
        "sha256": sha256,
        "bytes": bytes_count,
    }


def _tco_paired_scope(*, paired: int, roots: int) -> dict[str, object]:
    def tokens(hebrew_per_pair: int, english_per_pair: int) -> dict[str, object]:
        hebrew = paired * hebrew_per_pair
        english = paired * english_per_pair
        return {
            "paired_hebrew_tokens": hebrew,
            "matched_english_tokens": english,
            "hebrew_to_english_ratio": hebrew / english,
        }

    return {
        "coverage": {"paired": paired, "roots": roots, "ratio": paired / roots},
        "token_ratios": {
            "query_tokens": tokens(4, 2),
            "prompt_tokens": tokens(12, 10),
            "full_tokens": tokens(18, 15),
        },
    }


def _tco_efficiency_ratio(*, elapsed_seconds: float, successes: int) -> dict[str, object]:
    common = {
        "unit": "gpu_seconds_per_full_call_exact_success",
        "full_call_exact_successes": successes,
        "basis": "generation_elapsed_seconds_x_configured_gpu_count",
    }
    if successes == 0:
        return {
            **common,
            "available": False,
            "value": None,
            "reason": "zero_full_call_exact_successes",
        }
    return {
        **common,
        "available": True,
        "value": round(elapsed_seconds / successes, 6),
        "reason": None,
    }


def _tco_inference_arm(
    *,
    en_elapsed: float,
    he_elapsed: float,
    en_successes: int,
    he_successes: int,
) -> dict[str, object]:
    def slice_payload(elapsed: float, successes: int) -> dict[str, object]:
        return {
            "examples": 4,
            "generation_elapsed_seconds": elapsed,
            "seconds_per_example": elapsed / 4,
            "full_call_exact_successes": successes,
            "configured_gpu_seconds_per_full_call_exact_success": _tco_efficiency_ratio(
                elapsed_seconds=elapsed,
                successes=successes,
            ),
        }

    total_elapsed = en_elapsed + he_elapsed
    total_successes = en_successes + he_successes
    return {
        "available": True,
        "measurement": {
            "scope": GENERATION_TIMING_SCOPE,
            "aggregation": GENERATION_TIMING_AGGREGATION,
            "clock": "monotonic_seconds",
            "model_load_included": False,
            "parsing_and_artifact_io_included": False,
        },
        "timed_call_contract": inference_timed_call_contract(),
        "warmup": inference_warmup_contract(),
        "sequential_run": {
            "boundary": SEQUENTIAL_RUN_BOUNDARY,
            "concurrency": 1,
            "single_model_instance": True,
            "slice_order": ["en", "he"],
            "example_order": "formatted_test_order_within_slice",
        },
        "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 128},
        "configured_gpu": {"label": "L40S", "count": 1, "source": "config.remote.gpu"},
        "slices": {
            "en": slice_payload(en_elapsed, en_successes),
            "he": slice_payload(he_elapsed, he_successes),
        },
        "overall": {
            "examples": 8,
            "generation_elapsed_seconds": total_elapsed,
            "seconds_per_example": total_elapsed / 8,
            "full_call_exact_successes": total_successes,
            "configured_gpu_seconds_per_full_call_exact_success": _tco_efficiency_ratio(
                elapsed_seconds=total_elapsed,
                successes=total_successes,
            ),
        },
    }


def _tco_inference_sources(
    *,
    run_id: str,
    model_kind: str,
    config_sha256: str,
) -> dict[str, object]:
    prefix = f"runs/{run_id}"
    stage = f"eval-{model_kind}"
    return {
        "evaluation_report": _tco_artifact_source(
            f"{prefix}/eval/{model_kind}/evaluation_report.json",
            "evaluation_report",
            "sommelier.evaluation_report.v3",
            sha256="3" * 64,
        ),
        "evaluation_manifest": _tco_artifact_source(
            f"{prefix}/{stage}_manifest.json",
            "manifest",
            "sommelier.manifest.v1",
        ),
        "run_manifest": _tco_artifact_source(
            f"{prefix}/manifest.json",
            "manifest",
            "sommelier.manifest.v1",
        ),
        "resolved_config": _tco_artifact_source(
            f"{prefix}/config.resolved.yaml",
            "config",
            "sommelier.config.v2",
            sha256=config_sha256,
        ),
        "runtime_metadata": _tco_artifact_source(
            f"{prefix}/runtime_metadata.json",
            "runtime_metadata",
            "sommelier.runtime_metadata.v1",
        ),
        "inference_telemetry": _tco_artifact_source(
            f"{prefix}/eval/{model_kind}/inference_telemetry.json",
            "inference_telemetry",
            "sommelier.inference_telemetry.v2",
        ),
    }


def _valid_experiment_report(
    *,
    run_id: str = "run-1",
    source_revision: str = "f" * 40,
    config_sha256: str = "0" * 64,
    tree_sha256: str = "2" * 64,
) -> dict[str, object]:
    v1_source = {
        "source": "abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-lora",
        "revision": "45a6e2fa3e29f8393ddf1e9bda51a9461b41ee0e",
        "kind": "huggingface_repo",
        "tree_sha256": None,
        "artifact_path": None,
        "revision_is_immutable": True,
    }
    v3_source = {
        "source": f"runs/{run_id}/train/adapter",
        "revision": None,
        "kind": "local_directory",
        "tree_sha256": tree_sha256,
        "artifact_path": f"runs/{run_id}/train/adapter",
        "revision_is_immutable": True,
    }
    unavailable_currency = {
        "available": False,
        "value": None,
        "reason": "provider_billing_evidence_not_supplied",
    }
    unavailable_full_finetune = {
        "available": False,
        "value": None,
        "reason": "matched_full_finetune_evidence_not_supplied",
    }
    resolved_config_source = _tco_artifact_source(
        f"runs/{run_id}/config.resolved.yaml",
        "config",
        "sommelier.config.v2",
        sha256=config_sha256,
    )
    runtime_source = _tco_artifact_source(
        f"runs/{run_id}/runtime_metadata.json",
        "runtime_metadata",
        "sommelier.runtime_metadata.v1",
    )
    root_manifest_source = _tco_artifact_source(
        f"runs/{run_id}/manifest.json",
        "manifest",
        "sommelier.manifest.v1",
    )
    tokenization_manifest_source = _tco_artifact_source(
        f"runs/{run_id}/tokenization_manifest.json",
        "manifest",
        "sommelier.manifest.v1",
    )
    v3_telemetry_source = _tco_artifact_source(
        f"runs/{run_id}/eval/adapter/inference_telemetry.json",
        "inference_telemetry",
        "sommelier.inference_telemetry.v2",
    )
    provenance_sources = {
        "resolved_config": resolved_config_source,
        "runtime_metadata": runtime_source,
        "root_manifest": root_manifest_source,
        "data_manifest": _tco_artifact_source(
            f"runs/{run_id}/data_manifest.json",
            "manifest",
            "sommelier.manifest.v1",
        ),
        "format_manifest": _tco_artifact_source(
            f"runs/{run_id}/format_manifest.json",
            "manifest",
            "sommelier.manifest.v1",
        ),
        "tokenization_manifest": tokenization_manifest_source,
        "eval_adapter_manifest": _tco_artifact_source(
            f"runs/{run_id}/eval-adapter_manifest.json",
            "manifest",
            "sommelier.manifest.v1",
        ),
        "v3_inference_telemetry": v3_telemetry_source,
        "root_rows": _tco_artifact_source(
            f"runs/{run_id}/data/source_inputs/rows.en.jsonl",
            "raw_dataset",
            "sommelier.raw_tool_call_row.v1",
        ),
        "he_paired_rows": _tco_artifact_source(
            f"runs/{run_id}/data/source_inputs/rows.en.he.jsonl",
            "raw_paired_dataset",
            "sommelier.raw_tool_call_row.v1",
        ),
        "he_translation_summary": _tco_artifact_source(
            f"runs/{run_id}/data/source_inputs/translation_summary.he.json",
            "translation_summary",
            "sommelier.translation_summary.v2",
        ),
        "he_translation_publication": _tco_artifact_source(
            f"runs/{run_id}/data/source_inputs/translation_publication.he.json",
            "translation_publication_manifest",
            "sommelier.translation_publication_manifest.v1",
        ),
        "he_semantic_review_template": _tco_artifact_source(
            f"runs/{run_id}/data/source_inputs/translation_semantic_review_template.he.json",
            "translation_semantic_review_template",
            "sommelier.translation_semantic_review_template.v1",
        ),
        "he_semantic_review": _tco_artifact_source(
            f"runs/{run_id}/data/source_inputs/translation_semantic_review.he.json",
            "translation_semantic_review",
            "sommelier.translation_semantic_review.v1",
        ),
        "he_translation_config": _tco_artifact_source(
            f"runs/{run_id}/data/source_inputs/translation_config.he.yaml",
            "config",
            "sommelier.config.v2",
        ),
        "he_translation_run_identity": _tco_artifact_source(
            f"runs/{run_id}/data/source_inputs/translation_run_identity.he.json",
            "translation_run_identity",
            "sommelier.translation_run_identity.v1",
        ),
        **{
            f"prepared_{split}": _tco_artifact_source(
                f"runs/{run_id}/data/{split}.jsonl",
                "dataset_split",
                "sommelier.prepared_example.v2",
            )
            for split in ("train", "validation", "test")
        },
        "prepared_drop_summary": _tco_artifact_source(
            f"runs/{run_id}/data/drop_summary.json",
            "drop_summary",
            "sommelier.drop_summary.v2",
        ),
        **{
            f"formatted_{split}": _tco_artifact_source(
                f"runs/{run_id}/formatted/{split}.jsonl",
                "formatted_split",
                "sommelier.formatted_example.v2",
            )
            for split in ("train", "validation", "test")
        },
        "tokenization_tokenizer_tax_records": _tco_artifact_source(
            f"runs/{run_id}/analysis/tokenization/tokenizer_tax_records.jsonl",
            "tokenizer_tax_records",
            "sommelier.tokenizer_tax_record.v1",
        ),
        "tokenization_tokenizer_tax_report": _tco_artifact_source(
            f"runs/{run_id}/analysis/tokenization/tokenizer_tax_report.json",
            "tokenizer_tax_report",
            "sommelier.tokenizer_tax_report.v1",
        ),
    }
    paired_scopes = {
        "all": _tco_paired_scope(paired=15_800, roots=17_000),
        "train": _tco_paired_scope(paired=14_000, roots=15_000),
        "validation": _tco_paired_scope(paired=900, roots=1_000),
        "test": _tco_paired_scope(paired=900, roots=1_000),
    }
    adapter_files = [
        _tco_artifact_source(
            f"runs/{run_id}/train/adapter/adapter_model.safetensors",
            "adapter_weights",
            "",
            bytes_count=800,
        ),
        _tco_artifact_source(
            f"runs/{run_id}/train/adapter/adapter_config.json",
            "adapter_weights",
            "",
            bytes_count=200,
        ),
    ]
    tokenization_sources = {
        "tokenizer_tax_report": _tco_artifact_source(
            f"runs/{run_id}/analysis/tokenization/tokenizer_tax_report.json",
            "tokenizer_tax_report",
            "sommelier.tokenizer_tax_report.v1",
        ),
        "tokenizer_tax_records": _tco_artifact_source(
            f"runs/{run_id}/analysis/tokenization/tokenizer_tax_records.jsonl",
            "tokenizer_tax_records",
            "sommelier.tokenizer_tax_record.v1",
        ),
        "tokenization_manifest": tokenization_manifest_source,
        "formatted_inputs": {
            split: _tco_artifact_source(
                f"runs/{run_id}/formatted/{split}.jsonl",
                "formatted_split",
                "sommelier.formatted_example.v2",
            )
            for split in ("train", "validation", "test")
        },
    }
    training_sources = {
        "train_manifest": _tco_artifact_source(
            f"runs/{run_id}/train_manifest.json",
            "manifest",
            "sommelier.manifest.v1",
        ),
        "runtime_metadata": runtime_source,
        "training_metrics": _tco_artifact_source(
            f"runs/{run_id}/train/training_metrics.jsonl",
            "training_metrics",
            "sommelier.training_metric.v1",
        ),
        "adapter_files": adapter_files,
    }
    inference_sources = {
        "base": _tco_inference_sources(
            run_id=run_id,
            model_kind="base",
            config_sha256=config_sha256,
        ),
        "v1_en": _tco_inference_sources(
            run_id="v1-run",
            model_kind="adapter",
            config_sha256=config_sha256,
        ),
        "v3_en_he": _tco_inference_sources(
            run_id=run_id,
            model_kind="adapter",
            config_sha256=config_sha256,
        ),
    }
    return {
        "schema_version": EXPERIMENT_REPORT_SCHEMA,
        "created_at": "2026-07-14T12:00:00+00:00",
        "preregistration": {
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
                "git_commit": source_revision,
                "working_tree_clean": True,
                "boundary": "Observed before loading experiment outcome artifacts.",
            },
        },
        "shared_evaluation_identity": {
            "model_identity": {
                "base_model_id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                "base_model_revision": "54641c1611fcff44fa4865626462445e0a153fc7",
                "tokenizer_id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                "tokenizer_revision": "54641c1611fcff44fa4865626462445e0a153fc7",
            },
            "split": "test",
            "parser_version": "sommelier.parser.v1",
            "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 128},
            "test_split_sha256": "7" * 64,
            "slices": {
                language: {
                    "examples": 4,
                    "example_ids_sha256": "8" * 64,
                    "prompt_set_sha256": "9" * 64,
                }
                for language in ("en", "he")
            },
            "paired_cohorts": {
                "he": {
                    "pairs": 4,
                    "pair_set_sha256": "a" * 64,
                    "reference_row_indices_sha256": "b" * 64,
                }
            },
        },
        "data_provenance": {
            "schema_version": "sommelier.hebrew_v3_data_provenance.v1",
            "contract": {
                "seed": 42,
                "root_dataset": {
                    "dataset_id": "Salesforce/xlam-function-calling-60k",
                    "dataset_revision": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
                },
                "paired_dataset": {
                    "dataset_id": "abdelstark/sommelier-xlam-single-call-splits-he",
                    "dataset_revision": "e" * 40,
                },
                "requested_splits": {"train": 15_000, "validation": 1_000, "test": 1_000},
                "observed_cohorts": {
                    "train": {"en": 15_000, "he": 14_000, "total": 29_000},
                    "validation": {"en": 1_000, "he": 900, "total": 1_900},
                    "test": {"en": 1_000, "he": 900, "total": 1_900},
                },
                "semantic_review": {
                    "sample_size": 200,
                    "required_critical_errors": 0,
                    "status": "validated",
                },
                "source_code_revision": source_revision,
            },
            "sources": provenance_sources,
        },
        "bootstrap": {"seed": 42, "resamples": 2000, "confidence_level": 0.95},
        "arms": {
            "base": _arm_payload(
                run_id=run_id,
                model_kind="base",
                adapter_source=None,
                en_numerator=1,
                he_numerator=0,
                config_sha256=config_sha256,
            ),
            "v1_en": _arm_payload(
                run_id="v1-run",
                model_kind="adapter",
                adapter_source=v1_source,
                en_numerator=3,
                he_numerator=1,
                config_sha256=config_sha256,
            ),
            "v3_en_he": _arm_payload(
                run_id=run_id,
                model_kind="adapter",
                adapter_source=v3_source,
                en_numerator=3,
                he_numerator=3,
                config_sha256=config_sha256,
            ),
        },
        "comparisons": {
            "v3_vs_v1": {
                "en": _comparison_payload(seed=42, delta=0.0, lower=-0.01, upper=0.01),
                "he": _comparison_payload(seed=43, delta=0.5, lower=0.25, upper=0.75),
            }
        },
        "sovereign_tco_evidence": {
            "schema_version": "sommelier.sovereign_tco_evidence.v1",
            "subject": {
                "run_id": run_id,
                "config_sha256": config_sha256,
                "tokenizer": {
                    "id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                    "revision": "54641c1611fcff44fa4865626462445e0a153fc7",
                },
            },
            "evidence_policy": {
                "scope": "bounded_observed_or_deterministically_projected_quantities",
                "currency_estimation": "forbidden_without_observed_billing",
                "full_finetune_savings_estimation": (
                    "forbidden_without_matched_full_finetune_evidence"
                ),
            },
            "paired_tokenization": {
                "available": True,
                "evidence_kind": "deterministic_measurement_from_pinned_tokenizer",
                "reference_language": "en",
                "target_language": "he",
                "paired_scopes": paired_scopes,
                "projected_training_workload": {
                    "languages": ["en", "he"],
                    "examples_per_epoch": 29_000,
                    "non_padding_full_tokens_per_epoch": 1_000_000,
                    "epochs": 1,
                    "projected_non_padding_full_tokens": 1_000_000,
                    "english_only_counterfactual": {
                        "language": "en",
                        "examples_per_epoch": 15_000,
                        "non_padding_full_tokens_per_epoch": 400_000,
                        "epochs": 1,
                        "projected_non_padding_full_tokens": 400_000,
                    },
                    "hebrew_increment": {
                        "language": "he",
                        "examples_per_epoch": 14_000,
                        "examples_per_epoch_ratio_to_english_only": 14 / 15,
                        "non_padding_full_tokens_per_epoch": 600_000,
                        "non_padding_full_tokens_per_epoch_ratio_to_english_only": 1.5,
                        "epochs": 1,
                        "projected_non_padding_full_tokens": 600_000,
                        "projected_non_padding_full_tokens_ratio_to_english_only": 1.5,
                    },
                    "combined_vs_english_only": {
                        "examples_per_epoch_multiplier": 29 / 15,
                        "non_padding_full_tokens_per_epoch_multiplier": 2.5,
                        "projected_non_padding_full_tokens_multiplier": 2.5,
                    },
                    "evidence_kind": "deterministic_projection",
                    "boundary": "Excludes dynamic padding.",
                },
            },
            "qlora_training": {
                "available": True,
                "train_stage_runtime": {
                    "available": True,
                    "elapsed_seconds": 3_600.0,
                    "boundary": "Observed end-to-end train-stage wall clock.",
                    "configured_gpu": {
                        "label": "L40S",
                        "count": 1,
                        "source": "resolved config via runtime metadata",
                    },
                    "configured_gpu_hours": 1.0,
                    "configured_gpu_hours_kind": ("observed_wall_time_x_configured_gpu_count"),
                },
                "source_code": {
                    "available": True,
                    "git_commit": source_revision,
                    "working_tree_clean": True,
                    "boundary": "Measured before remote dispatch.",
                    "reason": None,
                },
                "remote_execution": {
                    "available": True,
                    "provider": "modal",
                    "function_timeout_seconds": 21_600,
                    "gpu_allocation_label": "L40S",
                    "boundary": "Outer function timeout; billing is separate.",
                },
                "tokens_seen": {
                    "available": True,
                    "value": 1_000_000,
                    "unit": "trainer_reported_input_tokens",
                    "source_label": "maximum_positive_transformers.num_input_tokens_seen",
                    "boundary": "Backend-reported input tokens.",
                },
                "end_to_end_token_throughput": {
                    "available": True,
                    "value": 1_000_000 / 3_600,
                    "unit": "trainer_reported_input_tokens_per_train_stage_second",
                    "boundary": "Uses the end-to-end train-stage wall clock boundary.",
                },
                "peak_gpu_memory": {
                    "available": True,
                    "value": 4_096,
                    "unit": "MiB",
                    "evidence_kind": "observed_peak_allocated_gpu_memory",
                },
                "adapter_storage": {
                    "available": True,
                    "tree_sha256": tree_sha256,
                    "packaged_adapter": {
                        "bytes": 1_000,
                        "files": 2,
                        "boundary": "All regular files in the adapter directory.",
                    },
                    "tensor_weights_only": {
                        "bytes": 800,
                        "files": 1,
                        "boundary": "Recognized adapter tensor files only.",
                    },
                    "evidence_kind": "observed_artifact_storage",
                },
                "currency_cost": unavailable_currency,
                "full_finetune_savings": unavailable_full_finetune,
            },
            "inference_efficiency": {
                "arms": {
                    "base": _tco_inference_arm(
                        en_elapsed=4.0,
                        he_elapsed=6.0,
                        en_successes=1,
                        he_successes=0,
                    ),
                    "v1_en": _tco_inference_arm(
                        en_elapsed=3.0,
                        he_elapsed=5.0,
                        en_successes=3,
                        he_successes=1,
                    ),
                    "v3_en_he": _tco_inference_arm(
                        en_elapsed=2.0,
                        he_elapsed=4.0,
                        en_successes=3,
                        he_successes=3,
                    ),
                },
                "cross_arm_comparability": {
                    "available": True,
                    "configured_gpu": {"label": "L40S", "count": 1},
                    "observed_packages": {
                        "python": "3.13.3",
                        "torch": "2.11.0",
                        "transformers": "5.13.1",
                        "tokenizers": "0.22.2",
                        "accelerate": "1.12.0",
                        "peft": "0.18.1",
                        "datasets": "4.7.0",
                        "huggingface_hub": "1.7.1",
                    },
                    "boundary": (
                        "Identical sequential end-to-end generator-call measurement contract."
                    ),
                },
            },
            "explicitly_unavailable": {
                "currency_cost": unavailable_currency,
                "full_finetune_savings": unavailable_full_finetune,
            },
            "sources": {
                "resolved_config": resolved_config_source,
                "tokenization": tokenization_sources,
                "training": training_sources,
                "inference": inference_sources,
                "run_manifest": root_manifest_source,
                "data_provenance": provenance_sources,
            },
        },
        "claims": {
            "hebrew_full_call_uplift": {
                "passed": True,
                "metric": "full_call_exact_match",
                "estimate": 0.5,
                "ci95": {"lower": 0.25, "upper": 0.75},
                "criterion": "95% paired-bootstrap lower bound > 0",
                "statement": _HEBREW_UPLIFT_STATEMENT,
            },
            "english_full_call_non_inferiority": {
                "passed": True,
                "metric": "full_call_exact_match",
                "estimate": 0.0,
                "ci95": {"lower": -0.01, "upper": 0.01},
                "margin": 0.01,
                "criterion": "95% paired-bootstrap lower bound >= -margin",
                "statement": _ENGLISH_NON_INFERIORITY_STATEMENT,
            },
        },
        "all_claims_passed": True,
        "approved_claims": [
            _HEBREW_UPLIFT_STATEMENT,
            _ENGLISH_NON_INFERIORITY_STATEMENT,
        ],
    }


def test_validate_only_is_default_and_performs_no_hub_calls(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)

    plan = publish_prepared_bundle(
        prepared,
        repo_id="owner/hebrew-data",
        commit_message="Publish audited Hebrew dataset",
        create_repo=True,
        client=client,
    )

    assert plan["status"] == "validated"
    assert plan["executed"] is False
    repository = cast("dict[str, object]", plan["repository"])
    assert repository["create_repo"] is True
    assert repository["commit_sha"] is None
    assert client.events == []


def test_validate_only_rejects_receipt_path(tmp_path: Path) -> None:
    with pytest.raises(UserInputError, match="only valid with execute"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Validate",
            receipt_path=tmp_path / "receipt.json",
        )


@pytest.mark.parametrize("repo_id", ["missing-namespace", "/owner/name", "owner/name/extra"])
def test_invalid_repo_ids_fail_before_hub_access(tmp_path: Path, repo_id: str) -> None:
    client = FakeHubClient(tmp_path)
    with pytest.raises(UserInputError, match="invalid Hugging Face repo ID"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id=repo_id,
            commit_message="Validate",
            client=client,
        )
    assert client.events == []


@pytest.mark.parametrize("message", ["", "   ", "line one\nline two", "line one\rline two"])
def test_invalid_commit_messages_fail_closed(tmp_path: Path, message: str) -> None:
    with pytest.raises(UserInputError, match="commit_message"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message=message,
        )


def test_execute_requires_exact_confirmation_and_receipt(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)
    with pytest.raises(UserInputError, match="does not exactly match"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/other",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )
    with pytest.raises(UserInputError, match="requires an explicit receipt"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            client=client,
        )
    assert client.events == []


def test_existing_repository_commit_is_round_trip_verified(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(
        tmp_path,
        existing_files={".gitattributes", "README.md"},
    )
    receipt_path = tmp_path / "receipts" / "dataset.json"

    result = publish_prepared_bundle(
        prepared,
        repo_id="owner/hebrew-data",
        commit_message="Publish audited Hebrew dataset",
        execute=True,
        confirmed_repo_id="owner/hebrew-data",
        receipt_path=receipt_path,
        client=client,
    )

    assert result["status"] == "verified"
    assert result["platform_files"] == [".gitattributes"]
    repository = cast("dict[str, object]", result["repository"])
    assert repository["commit_sha"] == "a" * 40
    assert repository["create_repo"] is False
    assert client.events[:4] == ["resolve", f"list:{'b' * 40}", "commit", f"list:{'a' * 40}"]
    assert client.parent_commits == ["b" * 40]
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == result
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    recorded = {
        cast("str", item["path"]): cast("str", item["sha256"])
        for item in cast("list[dict[str, object]]", result["files"])
    }
    assert recorded == prepared.sha256


def test_public_dataset_upload_uses_validated_private_snapshot_when_source_mutates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config_path = _phase_a_hebrew_config(tmp_path)
    source_revision = _commit_config(config_path)
    bundle = _dataset_bundle(
        tmp_path / "bundle",
        config_path,
        source_revision=source_revision,
    )
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text('{"source": "en"}\n', encoding="utf-8")
    original_readme = (bundle / "README.md").read_bytes()
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    receipt_path = tmp_path / "receipt.json"

    def fake_validate(
        config: SommelierConfig,
        staged_root: Path,
    ) -> dict[str, dict[str, Path]]:
        del config
        return {"he": {"rows": paired_input_path(staged_root, "he")}}

    monkeypatch.setattr(publication, "validate_full_paired_input_contract", fake_validate)

    class SourceMutatingClient(FakeHubClient):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.upload_paths: dict[str, Path] = {}
            self.upload_modes: dict[str, int] = {}
            self.upload_sha256: dict[str, str] = {}

        def resolve_revision(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
        ) -> str | None:
            revision = super().resolve_revision(repo_id=repo_id, repo_type=repo_type)
            assert json.loads(receipt_path.read_text(encoding="utf-8"))["status"] == "pending"
            (bundle / "README.md").write_text(secret, encoding="utf-8")
            return revision

        def create_commit(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
            files: Mapping[str, Path],
            commit_message: str,
            parent_commit: str | None,
        ) -> str:
            self.upload_paths = dict(files)
            self.upload_modes = {name: path.stat().st_mode & 0o777 for name, path in files.items()}
            self.upload_sha256 = {name: sha256_file(path) for name, path in files.items()}
            return super().create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                files=files,
                commit_message=commit_message,
                parent_commit=parent_commit,
            )

    client = SourceMutatingClient(tmp_path)
    receipt = publication.publish_hebrew_dataset_bundle(
        config_path=config_path,
        bundle_dir=bundle,
        root_rows_path=root_rows,
        repo_id=HEBREW_V3_PAIRED_DATASET_ID,
        commit_message="Publish immutable Hebrew dataset snapshot",
        execute=True,
        confirmed_repo_id=HEBREW_V3_PAIRED_DATASET_ID,
        receipt_path=receipt_path,
        client=client,
    )

    assert receipt["status"] == "verified"
    assert (bundle / "README.md").read_text(encoding="utf-8") == secret
    assert client.committed["README.md"] == original_readme
    assert all(secret.encode() not in payload for payload in client.committed.values())
    assert all(not path.is_relative_to(bundle) for path in client.upload_paths.values())
    assert set(client.upload_modes.values()) == {0o400}
    readme_evidence = next(
        item
        for item in cast("list[dict[str, object]]", receipt["files"])
        if item["path"] == "README.md"
    )
    assert readme_evidence == {
        "path": "README.md",
        "sha256": client.upload_sha256["README.md"],
        "bytes": len(original_readme),
    }


def test_absent_repository_is_created_only_with_explicit_flag(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path, repository_exists=False)

    result = publish_prepared_bundle(
        prepared,
        repo_id="owner/hebrew-data",
        commit_message="Create audited Hebrew dataset",
        execute=True,
        create_repo=True,
        confirmed_repo_id="owner/hebrew-data",
        receipt_path=tmp_path / "receipt.json",
        client=client,
    )

    assert result["status"] == "verified"
    assert client.events[:3] == [
        "create:dataset:owner/hebrew-data",
        "resolve",
        f"list:{'b' * 40}",
    ]


def test_first_commit_to_new_empty_repository_allows_no_parent(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path, repository_exists=False)
    client.head_revision = None

    result = publish_prepared_bundle(
        prepared,
        repo_id="owner/hebrew-data",
        commit_message="Create audited Hebrew dataset",
        execute=True,
        create_repo=True,
        confirmed_repo_id="owner/hebrew-data",
        receipt_path=tmp_path / "receipt.json",
        client=client,
    )

    assert result["status"] == "verified"
    assert client.events[:5] == [
        "create:dataset:owner/hebrew-data",
        "resolve",
        "list:head",
        "resolve",
        "commit",
    ]
    assert client.parent_commits == [None]


def test_preexisting_empty_repository_cannot_receive_parentless_commit(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path, repository_exists=True, existing_files=set())
    client.head_revision = None

    with pytest.raises(ExternalDependencyError, match="not created by this publication"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Refuse ambiguous initial commit",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )

    assert client.events == ["resolve", "list:head"]
    assert "commit" not in client.events


def test_absent_repository_without_create_flag_fails_before_commit(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path, repository_exists=False)
    with pytest.raises(ExternalDependencyError, match="could not inspect"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )
    assert client.events == ["resolve"]


def test_create_flag_refuses_to_adopt_existing_repository(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path, repository_exists=True)
    with pytest.raises(ExternalDependencyError, match="could not create new public"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            create_repo=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )
    assert client.events == ["create:dataset:owner/hebrew-data"]


def test_production_create_repo_is_public_and_never_exist_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeApi:
        def create_repo(self, **kwargs: object) -> None:
            calls.update(kwargs)

    class FakeCommitOperationAdd:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    def fake_download(**kwargs: object) -> str:
        del kwargs
        return str(tmp_path / "download")

    module = ModuleType("huggingface_hub")
    setattr(module, "HfApi", FakeApi)
    setattr(module, "CommitOperationAdd", FakeCommitOperationAdd)
    setattr(module, "hf_hub_download", fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    client = publication._HuggingFaceHubClient()
    client.create_repo(repo_id="owner/hebrew-data", repo_type="dataset")

    assert calls == {
        "repo_id": "owner/hebrew-data",
        "repo_type": "dataset",
        "private": False,
        "exist_ok": False,
    }


def test_production_commit_is_bound_to_observed_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeInfo:
        sha = "b" * 40
        oid = "a" * 40

    class FakeApi:
        def repo_info(self, **kwargs: object) -> FakeInfo:
            calls["repo_info"] = kwargs
            return FakeInfo()

        def create_commit(self, **kwargs: object) -> FakeInfo:
            calls["create_commit"] = kwargs
            return FakeInfo()

    class FakeCommitOperationAdd:
        def __init__(self, **kwargs: object) -> None:
            calls["operation"] = kwargs

    def fake_download(**kwargs: object) -> str:
        del kwargs
        return str(tmp_path / "download")

    module = ModuleType("huggingface_hub")
    setattr(module, "HfApi", FakeApi)
    setattr(module, "CommitOperationAdd", FakeCommitOperationAdd)
    setattr(module, "hf_hub_download", fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    source = tmp_path / "README.md"
    source.write_text("release\n", encoding="utf-8")

    client = publication._HuggingFaceHubClient()
    parent = client.resolve_revision(repo_id="owner/repo", repo_type="model")
    revision = client.create_commit(
        repo_id="owner/repo",
        repo_type="model",
        files={"README.md": source},
        commit_message="Publish",
        parent_commit=parent,
    )

    assert parent == "b" * 40
    assert revision == "a" * 40
    commit_call = cast("dict[str, object]", calls["create_commit"])
    assert commit_call["parent_commit"] == "b" * 40


def test_production_download_materializes_hub_cache_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = tmp_path / "cache" / "blobs" / "sha"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"verified bytes")
    cache_pointer = tmp_path / "cache" / "snapshots" / ("a" * 40) / "README.md"
    cache_pointer.parent.mkdir(parents=True)
    cache_pointer.symlink_to(blob)

    class FakeApi:
        pass

    class FakeCommitOperationAdd:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    def fake_download(**kwargs: object) -> str:
        del kwargs
        return str(cache_pointer)

    module = ModuleType("huggingface_hub")
    setattr(module, "HfApi", FakeApi)
    setattr(module, "CommitOperationAdd", FakeCommitOperationAdd)
    setattr(module, "hf_hub_download", fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    client = publication._HuggingFaceHubClient()
    materialized = client.download_file(
        repo_id="owner/repo",
        repo_type="model",
        filename="README.md",
        revision="a" * 40,
    )

    assert materialized.read_bytes() == b"verified bytes"
    assert materialized.is_file()
    assert not materialized.is_symlink()


def test_public_api_exposes_only_fully_validated_publication_boundaries() -> None:
    assert publication.__all__ == [
        "publish_hebrew_adapter_bundle",
        "publish_hebrew_dataset_bundle",
    ]
    assert not hasattr(publication, "publish_prepared_bundle")


def test_unexpected_existing_remote_file_blocks_upload(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path, existing_files={"unrelated.bin"})
    with pytest.raises(UserInputError, match="outside the publication allowlist"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )
    assert "commit" not in client.events


def test_noncanonical_remote_path_fails_closed(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path, existing_files={"../escape"})
    with pytest.raises(ExternalDependencyError, match="non-canonical"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )


def test_nonimmutable_commit_identity_is_journaled_but_not_verified(tmp_path: Path) -> None:
    client = FakeHubClient(tmp_path)
    client.revision = "main"
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ExternalDependencyError, match="non-immutable"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )
    journal = json.loads(receipt.read_text(encoding="utf-8"))
    assert journal["status"] == "commit_returned_unverified"
    assert journal["repository"]["commit_sha"] == "main"


@pytest.mark.parametrize("extra", [False, True])
def test_round_trip_file_tree_must_match_allowlist(tmp_path: Path, extra: bool) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)
    files = set(prepared.files)
    if extra:
        files.add("surprise.txt")
    else:
        files.remove(next(iter(files)))
    client.post_commit_files = files

    with pytest.raises(ExternalDependencyError, match="filename verification failed"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )


def test_round_trip_sha_mismatch_retains_unverified_commit_journal(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)
    client.corrupt_file = "README.md"
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ExternalDependencyError, match="SHA256 mismatch"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )
    journal = json.loads(receipt.read_text(encoding="utf-8"))
    assert journal["status"] == "commit_returned_unverified"
    assert journal["repository"]["commit_sha"] == "a" * 40


def test_round_trip_symlink_is_rejected(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)
    client.symlink_download = "README.md"
    with pytest.raises(ExternalDependencyError, match="regular file"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )


def test_existing_receipt_fails_before_remote_mutation_and_is_unchanged(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("original\n", encoding="utf-8")
    client = FakeHubClient(tmp_path)
    with pytest.raises(UserInputError, match="receipt already exists"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )
    assert receipt.read_text(encoding="utf-8") == "original\n"
    assert client.events == []


def test_broken_symlink_receipt_target_fails_before_remote_mutation(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.symlink_to(tmp_path / "missing")
    client = FakeHubClient(tmp_path)
    with pytest.raises(UserInputError, match="receipt already exists"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )
    assert receipt.is_symlink()
    assert client.events == []


def test_unwritable_receipt_destination_fails_before_remote_mutation(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied\n", encoding="utf-8")
    client = FakeHubClient(tmp_path)

    with pytest.raises(UserInputError, match="could not reserve publication receipt"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=blocked_parent / "receipt.json",
            client=client,
        )

    assert client.events == []


def test_commit_failure_leaves_durable_submission_journal(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    client = FakeHubClient(tmp_path)
    client.fail_commit = True

    with pytest.raises(ExternalDependencyError, match="commit failed"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )

    journal = json.loads(receipt.read_text(encoding="utf-8"))
    assert journal["status"] == "commit_submitting"
    assert journal["repository"]["commit_sha"] is None
    assert journal["repository"]["parent_commit"] == "b" * 40
    assert receipt.stat().st_mode & 0o777 == 0o600


def test_retry_adopts_commit_accepted_before_local_client_error(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    prepared = _prepared_dataset(tmp_path)

    class AcceptedThenRaisedClient(FakeHubClient):
        def create_commit(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
            files: Mapping[str, Path],
            commit_message: str,
            parent_commit: str | None,
        ) -> str:
            revision = super().create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                files=files,
                commit_message=commit_message,
                parent_commit=parent_commit,
            )
            self.head_revision = revision
            raise RuntimeError("response was lost after the Hub accepted the commit")

    client = AcceptedThenRaisedClient(tmp_path)
    with pytest.raises(ExternalDependencyError, match="commit failed"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )

    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "commit_submitting"
    recovered = publish_prepared_bundle(
        prepared,
        repo_id="owner/hebrew-data",
        commit_message="Publish",
        execute=True,
        confirmed_repo_id="owner/hebrew-data",
        receipt_path=receipt,
        client=client,
    )

    assert recovered["status"] == "verified"
    assert cast("dict[str, object]", recovered["repository"])["commit_sha"] == "a" * 40
    assert client.events.count("commit") == 1
    assert f"inspect:{'a' * 40}" in client.events
    assert json.loads(receipt.read_text(encoding="utf-8")) == recovered


def test_retry_rejects_nonexact_remote_commit_without_resubmitting(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    prepared = _prepared_dataset(tmp_path)

    class AcceptedThenRaisedClient(FakeHubClient):
        def create_commit(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
            files: Mapping[str, Path],
            commit_message: str,
            parent_commit: str | None,
        ) -> str:
            revision = super().create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                files=files,
                commit_message=commit_message,
                parent_commit=parent_commit,
            )
            self.head_revision = revision
            raise RuntimeError("response was lost after the Hub accepted the commit")

    client = AcceptedThenRaisedClient(tmp_path)
    with pytest.raises(ExternalDependencyError, match="commit failed"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )

    client.post_commit_files = set(prepared.files) | {"surprise.txt"}
    with pytest.raises(ExternalDependencyError, match="filename verification failed"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )

    assert client.events.count("commit") == 1
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "commit_submitting"


def test_receipt_content_continuity_rejects_same_inode_tampering_without_leak(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    reservation = publication._reserve_receipt(
        receipt,
        {"status": "pending"},
        source_roots=(),
    )
    secret = "hf_" + "z" * 30
    replacement = f'{{"replacement":"{secret}"}}\n'.encode()
    replacement += b" " * (reservation.capacity - len(replacement))
    try:
        receipt.write_bytes(replacement)

        with pytest.raises(
            ExternalDependencyError,
            match="durably update publication receipt",
        ) as captured:
            publication._write_receipt(
                reservation,
                {"status": "commit_submitting"},
            )

        assert secret not in str(captured.value)
        assert receipt.read_bytes() == replacement
    finally:
        publication._close_receipt_reservation(reservation)


def test_receipt_content_identity_advances_across_every_durable_stage(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    reservation = publication._reserve_receipt(
        receipt,
        {"status": "pending"},
        source_roots=(),
    )
    observed_hashes = [reservation.expected_sha256]
    try:
        for status in ("commit_submitting", "commit_returned_unverified", "verified"):
            reservation = publication._write_receipt(reservation, {"status": status})
            assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == status
            assert reservation.expected_sha256 == sha256_file(receipt)
            observed_hashes.append(reservation.expected_sha256)
    finally:
        publication._close_receipt_reservation(reservation)

    assert len(set(observed_hashes)) == len(observed_hashes)


def test_receipt_update_verifies_written_content_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    reservation = publication._reserve_receipt(
        receipt,
        {"status": "pending"},
        source_roots=(),
    )
    real_write_all = publication._write_all

    def corrupt_after_write(descriptor: int, data: bytes) -> None:
        real_write_all(descriptor, data)
        os.lseek(descriptor, 0, os.SEEK_SET)
        assert os.write(descriptor, b"x") == 1

    monkeypatch.setattr(publication, "_write_all", corrupt_after_write)
    try:
        with pytest.raises(
            ExternalDependencyError,
            match="durably update publication receipt",
        ):
            publication._write_receipt(
                reservation,
                {"status": "commit_submitting"},
            )
    finally:
        publication._close_receipt_reservation(reservation)


@pytest.mark.parametrize("outcome", ("success", "inspect-error", "commit-error"))
def test_receipt_handle_closes_on_every_publication_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    captured: dict[str, int] = {}
    real_reserve = publication._reserve_receipt
    real_close = os.close
    closed: list[int] = []

    def tracking_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    def capture_reservation(
        path: Path,
        payload: Mapping[str, object],
        *,
        source_roots: Sequence[Path],
    ) -> publication._ReceiptReservation:
        reservation = real_reserve(path, payload, source_roots=source_roots)
        captured["descriptor"] = reservation.descriptor
        captured["prior_closes"] = closed.count(reservation.descriptor)
        return reservation

    monkeypatch.setattr(os, "close", tracking_close)
    monkeypatch.setattr(publication, "_reserve_receipt", capture_reservation)

    class InspectFailureClient(FakeHubClient):
        def resolve_revision(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
        ) -> str | None:
            super().resolve_revision(repo_id=repo_id, repo_type=repo_type)
            raise RuntimeError("inspection failed")

    if outcome == "inspect-error":
        client: FakeHubClient = InspectFailureClient(tmp_path)
    else:
        client = FakeHubClient(tmp_path)
        client.fail_commit = outcome == "commit-error"

    if outcome == "success":
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=tmp_path / "receipt.json",
            client=client,
        )
    else:
        with pytest.raises(ExternalDependencyError):
            publish_prepared_bundle(
                _prepared_dataset(tmp_path),
                repo_id="owner/hebrew-data",
                commit_message="Publish",
                execute=True,
                confirmed_repo_id="owner/hebrew-data",
                receipt_path=tmp_path / "receipt.json",
                client=client,
            )

    descriptor = captured["descriptor"]
    assert closed.count(descriptor) == captured["prior_closes"] + 1
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_receipt_inode_replacement_is_detected_before_commit(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"

    class ReplacingClient(FakeHubClient):
        def resolve_revision(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
        ) -> str | None:
            revision = super().resolve_revision(repo_id=repo_id, repo_type=repo_type)
            receipt.unlink()
            receipt.write_text("replacement\n", encoding="utf-8")
            return revision

    client = ReplacingClient(tmp_path)
    with pytest.raises(ExternalDependencyError, match="durably update publication receipt"):
        publish_prepared_bundle(
            _prepared_dataset(tmp_path),
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )

    assert receipt.read_text(encoding="utf-8") == "replacement\n"
    assert "commit" not in client.events


@pytest.mark.skipif(os.name != "posix", reason="POSIX dirfd receipt hardening")
def test_receipt_parent_replacement_is_detected_before_commit(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    source_root = prepared.files["README.md"].parent
    receipt_parent = tmp_path / "receipts"
    receipt = receipt_parent / "attempt.json"
    moved_parent = tmp_path / "receipts-moved"

    class ReplacingParentClient(FakeHubClient):
        def resolve_revision(
            self,
            *,
            repo_id: str,
            repo_type: publication.PublicationRepoType,
        ) -> str | None:
            revision = super().resolve_revision(repo_id=repo_id, repo_type=repo_type)
            receipt_parent.rename(moved_parent)
            receipt_parent.symlink_to(source_root, target_is_directory=True)
            return revision

    client = ReplacingParentClient(tmp_path)
    with pytest.raises(ExternalDependencyError, match="update reserved publication receipt"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )

    assert (moved_parent / "attempt.json").is_file()
    assert not (source_root / "attempt.json").exists()
    assert "commit" not in client.events


def test_receipt_cannot_be_an_upload_source(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    with pytest.raises(UserInputError, match="outside the source bundle"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=prepared.files["README.md"],
            client=FakeHubClient(tmp_path),
        )


def test_receipt_cannot_be_nested_anywhere_in_source_bundle(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    client = FakeHubClient(tmp_path)
    source_root = prepared.files["README.md"].parent

    with pytest.raises(UserInputError, match="outside the source bundle"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=source_root / "receipts" / "attempt.json",
            client=client,
        )

    assert client.events == []


def test_receipt_cannot_use_case_alias_of_source_bundle(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path / "Bundle")
    source_root = prepared.files["README.md"].parent
    case_alias = tmp_path / "bundle" / "dataset"
    try:
        aliases_source = case_alias.exists() and os.path.samefile(case_alias, source_root)
    except OSError:
        aliases_source = False
    if not aliases_source:
        pytest.skip("filesystem is case-sensitive")
    client = FakeHubClient(tmp_path)

    with pytest.raises(UserInputError, match="outside the source bundle"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=case_alias / "receipts" / "attempt.json",
            client=client,
        )

    assert client.events == []


def test_receipt_parent_swap_into_source_is_rejected_before_hub_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_dataset(tmp_path)
    source_root = prepared.files["README.md"].parent
    outside = tmp_path / "outside-receipts"
    outside.mkdir()
    alias = tmp_path / "receipt-parent"
    alias.symlink_to(outside, target_is_directory=True)
    receipt = alias / "attempt.json"
    real_reserve = publication._reserve_receipt

    def swap_then_reserve(
        path: Path,
        payload: Mapping[str, object],
        *,
        source_roots: Sequence[Path],
    ) -> object:
        alias.unlink()
        alias.symlink_to(source_root, target_is_directory=True)
        return real_reserve(path, payload, source_roots=source_roots)

    monkeypatch.setattr(publication, "_reserve_receipt", swap_then_reserve)
    client = FakeHubClient(tmp_path)

    with pytest.raises(UserInputError, match="could not reserve publication receipt"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/hebrew-data",
            commit_message="Publish",
            execute=True,
            confirmed_repo_id="owner/hebrew-data",
            receipt_path=receipt,
            client=client,
        )

    assert not (source_root / "attempt.json").exists()
    assert client.events == []


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_constructed_dataset_preparation_cannot_bypass_upload_allowlist(
    tmp_path: Path,
    mutation: str,
) -> None:
    prepared = _prepared_dataset(tmp_path)
    files = dict(prepared.files)
    if mutation == "missing":
        files.pop("README.md")
    else:
        unexpected = tmp_path / "unexpected.txt"
        unexpected.write_text("no\n", encoding="utf-8")
        files["unexpected.txt"] = unexpected
    with pytest.raises(UserInputError, match="exact upload allowlist"):
        publish_prepared_bundle(
            PreparedPublication(repo_type="dataset", files=files),
            repo_id="owner/hebrew-data",
            commit_message="Validate",
        )


def test_constructed_preparation_rejects_symlink_source(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    target = prepared.files["README.md"]
    link = tmp_path / "README-link.md"
    link.symlink_to(target)
    files = dict(prepared.files)
    files["README.md"] = link
    with pytest.raises(UserInputError, match="not a regular file"):
        publish_prepared_bundle(
            PreparedPublication(repo_type="dataset", files=files),
            repo_id="owner/hebrew-data",
            commit_message="Validate",
        )


def test_model_upload_allowlist_accepts_only_explicit_peft_and_evidence_files(
    tmp_path: Path,
) -> None:
    plan = publish_prepared_bundle(
        _prepared_adapter(tmp_path, optional=True),
        repo_id="owner/Llama-hebrew-adapter",
        commit_message="Validate adapter",
    )
    paths = {cast("str", item["path"]) for item in cast("list[dict[str, object]]", plan["files"])}
    assert paths == ADAPTER_UPLOAD_REQUIRED_FILES | ADAPTER_UPLOAD_OPTIONAL_FILES


@pytest.mark.parametrize("repo_id", ["owner/hebrew-adapter", "owner/llama-hebrew-adapter"])
def test_llama_derived_model_repo_name_is_enforced(tmp_path: Path, repo_id: str) -> None:
    with pytest.raises(UserInputError, match="must begin with 'Llama'"):
        publish_prepared_bundle(
            _prepared_adapter(tmp_path),
            repo_id=repo_id,
            commit_message="Validate adapter",
        )


def test_dataset_preparation_validates_full_paired_contract_and_exact_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_a_config, config_path = _phase_a_hebrew_config(tmp_path)
    source_revision = _commit_config(config_path)
    bundle = _dataset_bundle(
        tmp_path / "bundle",
        config_path,
        source_revision=source_revision,
    )
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text('{"source": "en"}\n', encoding="utf-8")
    observed: dict[str, str] = {}

    def fake_validate(
        config: SommelierConfig,
        staged_root: Path,
    ) -> dict[str, dict[str, Path]]:
        assert config.dataset_for("he").dataset_revision == "0" * 40
        observed["root"] = staged_root.read_text(encoding="utf-8")
        observed["paired"] = paired_input_path(staged_root, "he").read_text(encoding="utf-8")
        for filename in (
            SUMMARY_FILENAME,
            PUBLICATION_MANIFEST_FILENAME,
            SEMANTIC_REVIEW_FILENAME,
            SEMANTIC_REVIEW_TEMPLATE_FILENAME,
            TRANSLATION_CONFIG_FILENAME,
            TRANSLATION_RUN_IDENTITY_FILENAME,
        ):
            assert translation_provenance_sidecar_path(staged_root, filename, "he").is_file()
        staged_config = translation_provenance_sidecar_path(
            staged_root,
            TRANSLATION_CONFIG_FILENAME,
            "he",
        )
        assert staged_config.read_bytes() == config_path.read_bytes()
        summary = json.loads(
            translation_provenance_sidecar_path(
                staged_root,
                SUMMARY_FILENAME,
                "he",
            ).read_text(encoding="utf-8")
        )
        assert summary["selection"]["config_sha256"] == sha256_file(config_path)
        assert summary["reviewer_preregistration"] == reviewer_anchor_payload(phase_a_config)
        assert summary["reviewer_preregistration_sha256"] == reviewer_anchor_sha256(phase_a_config)
        identity_path = translation_provenance_sidecar_path(
            staged_root,
            TRANSLATION_RUN_IDENTITY_FILENAME,
            "he",
        )
        identity = validate_hebrew_v3_translation_run_identity(
            identity_path,
            summary=summary,
            expected_config_sha256=sha256_file(config_path),
        )
        assert summary["translation_run_identity_sha256"] == sha256_file(identity_path)
        assert identity["config_sha256"] == sha256_file(config_path)
        assert identity["reviewer_preregistration"] == summary["reviewer_preregistration"]
        assert identity["source_code"] == summary["source_code"]
        return {"he": {"rows": paired_input_path(staged_root, "he")}}

    monkeypatch.setattr(publication, "validate_full_paired_input_contract", fake_validate)
    prepared = prepare_hebrew_dataset_publication(
        config_path=config_path,
        bundle_dir=bundle,
        root_rows_path=root_rows,
    )

    assert set(prepared.files) == DATASET_REQUIRED_FILES
    assert prepared.expected_repo_id == HEBREW_V3_PAIRED_DATASET_ID
    assert observed["root"] == '{"source": "en"}\n'
    assert observed["paired"] == "{}\n"


def test_dataset_preparation_rejects_config_not_committed_at_phase_a_source(
    tmp_path: Path,
) -> None:
    _, config_path = _phase_a_hebrew_config(tmp_path)
    source_revision = _commit_config(config_path)
    config_path.write_bytes(config_path.read_bytes() + b"# changed after Phase A\n")
    bundle = _dataset_bundle(
        tmp_path / "bundle",
        config_path,
        source_revision=source_revision,
    )
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text('{"source": "en"}\n', encoding="utf-8")

    with pytest.raises(UserInputError, match="config bytes differ from the immutable"):
        prepare_hebrew_dataset_publication(
            config_path=config_path,
            bundle_dir=bundle,
            root_rows_path=root_rows,
        )


def test_dataset_preparation_rejects_translation_config_byte_drift(tmp_path: Path) -> None:
    _, config_path = _phase_a_hebrew_config(tmp_path)
    bundle = _dataset_bundle(tmp_path / "bundle", config_path)
    published_config = bundle / TRANSLATION_CONFIG_FILENAME
    published_config.write_bytes(published_config.read_bytes() + b"# byte drift\n")
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text('{"source": "en"}\n', encoding="utf-8")

    with pytest.raises(UserInputError, match="not the exact Phase-A config"):
        prepare_hebrew_dataset_publication(
            config_path=config_path,
            bundle_dir=bundle,
            root_rows_path=root_rows,
        )


def test_prepared_repository_identity_cannot_be_redirected(tmp_path: Path) -> None:
    prepared = PreparedPublication(
        repo_type="dataset",
        files=_write_files(tmp_path / "dataset", sorted(DATASET_REQUIRED_FILES)),
        expected_repo_id=HEBREW_V3_PAIRED_DATASET_ID,
    )
    client = FakeHubClient(tmp_path)

    with pytest.raises(UserInputError, match="does not match the prepared artifact identity"):
        publish_prepared_bundle(
            prepared,
            repo_id="owner/lookalike-hebrew-dataset",
            commit_message="Validate",
            client=client,
        )

    assert client.events == []


def test_dataset_card_requires_cc_by_attribution_and_translation_disclosure(
    tmp_path: Path,
) -> None:
    card = tmp_path / "README.md"
    card.write_text("---\nlicense: mit\n---\nHebrew data\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="cc-by-4.0"):
        publication._validate_dataset_card(card)


def test_dataset_card_does_not_accept_license_prose_outside_frontmatter(tmp_path: Path) -> None:
    card = tmp_path / "README.md"
    card.write_text(
        "---\nlicense: mit\n---\n"
        "license: cc-by-4.0\nSalesforce/xlam-function-calling-60k machine-translated Hebrew\n",
        encoding="utf-8",
    )
    with pytest.raises(UserInputError, match="frontmatter license"):
        publication._validate_dataset_card(card)


def test_dataset_card_rejects_unresolved_verified_bundle_marker(tmp_path: Path) -> None:
    card = tmp_path / "README.md"
    card.write_text(
        "---\nlicense: cc-by-4.0\n---\n"
        "Salesforce/xlam-function-calling-60k machine-translated Hebrew\n"
        "REPLACE_FROM_VERIFIED_DATASET_BUNDLE\n",
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="unresolved release template markers"):
        publication._validate_dataset_card(card)


def test_dataset_card_rejects_duplicate_yaml_frontmatter_keys(tmp_path: Path) -> None:
    card = tmp_path / "README.md"
    card.write_text(
        "---\nlicense: mit\nlicense: cc-by-4.0\n---\n"
        "Salesforce/xlam-function-calling-60k machine-translated Hebrew\n",
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="invalid YAML frontmatter"):
        publication._validate_dataset_card(card)


def test_publication_json_objects_reject_duplicate_keys_without_exposing_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    path.write_text(
        f'{{"status":"{secret}","status":"safe"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="missing or invalid JSON") as captured:
        publication._load_json_object(path, context="run manifest")

    assert secret not in str(captured.value)


def test_dataset_bundle_rejects_unexpected_files_before_contract_validation(
    tmp_path: Path,
) -> None:
    _, config_path = _phase_a_hebrew_config(tmp_path)
    bundle = _dataset_bundle(tmp_path / "bundle", config_path)
    (bundle / "notes.txt").write_text("not curated\n", encoding="utf-8")
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text("{}\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="unexpected files: notes.txt"):
        prepare_hebrew_dataset_publication(
            config_path=config_path,
            bundle_dir=bundle,
            root_rows_path=root_rows,
        )


def test_dataset_bundle_rejects_secrets(tmp_path: Path) -> None:
    _, config_path = _phase_a_hebrew_config(tmp_path)
    bundle = _dataset_bundle(tmp_path / "bundle", config_path)
    (bundle / SUMMARY_FILENAME).write_text(
        f'{{"token": "hf_{"a" * 30}"}}\n',
        encoding="utf-8",
    )
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SecurityPolicyError, match="secret-like"):
        prepare_hebrew_dataset_publication(
            config_path=config_path,
            bundle_dir=bundle,
            root_rows_path=root_rows,
        )


def test_dataset_bundle_rejects_duplicate_json_keys_before_contract_validation(
    tmp_path: Path,
) -> None:
    _, config_path = _phase_a_hebrew_config(tmp_path)
    bundle = _dataset_bundle(tmp_path / "bundle", config_path)
    (bundle / SUMMARY_FILENAME).write_text(
        '{"status":"pending","status":"accepted"}\n',
        encoding="utf-8",
    )
    root_rows = tmp_path / "rows.en.jsonl"
    root_rows.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SecurityPolicyError, match="duplicate_key"):
        prepare_hebrew_dataset_publication(
            config_path=config_path,
            bundle_dir=bundle,
            root_rows_path=root_rows,
        )


def test_raw_provider_journal_is_never_publishable(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / OPENAI_PROVIDER_JOURNAL_FILENAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(SecurityPolicyError, match="raw OpenAI provider journal"):
        publication._assert_no_raw_provider_journal(bundle)


def test_adapter_preparation_maps_peft_root_and_namespaces_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path = _immutable_hebrew_config(tmp_path)
    bundle = _adapter_bundle(tmp_path / "bundle", config_path)

    def fake_manifests(
        *,
        bundle_dir: Path,
        config: SommelierConfig,
        adapter_files: Sequence[str],
    ) -> tuple[str, str, dict[str, Any]]:
        del bundle_dir, config
        assert "adapter/tokenizer.json" in adapter_files
        return (
            "run-1",
            "f" * 40,
            {
                "config_sha256": "0" * 64,
                "dependency_lock_sha256": "d" * 64,
            },
        )

    def fake_experiment(**kwargs: object) -> dict[str, Any]:
        del kwargs
        return {}

    monkeypatch.setattr(publication, "_validate_adapter_config", lambda *_: None)
    monkeypatch.setattr(publication, "_validate_safetensors", lambda *_: None)
    monkeypatch.setattr(publication, "_validate_adapter_manifests", fake_manifests)
    monkeypatch.setattr(publication, "_validate_experiment_identity", fake_experiment)
    monkeypatch.setattr(
        publication,
        "validate_evaluation_release_evidence",
        lambda **_: None,
    )
    monkeypatch.setattr(publication, "_validate_release_evidence", lambda *_, **__: None)
    monkeypatch.setattr(publication, "_validate_adapter_card", lambda *_, **__: None)

    prepared = prepare_hebrew_adapter_publication(bundle_dir=bundle)

    assert prepared.repo_type == "model"
    assert set(prepared.files) == ADAPTER_UPLOAD_REQUIRED_FILES | {"tokenizer.json"}
    assert prepared.files["adapter_model.safetensors"] == (
        bundle / "adapter" / "adapter_model.safetensors"
    )
    assert prepared.files["sommelier/evaluation_evidence/v3_en_he/correctness.he.jsonl"] == (
        bundle / "evaluation_evidence" / "v3_en_he" / "correctness.he.jsonl"
    )
    assert prepared.files["sommelier/config.resolved.yaml"] == bundle / "config.resolved.yaml"
    assert config.model.base_model_id in config_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_adapter_publication_allowlist_closes_evaluation_evidence_tree(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, config_path = _immutable_hebrew_config(tmp_path)
    bundle = _adapter_bundle(tmp_path / "bundle", config_path)
    if mutation == "missing":
        (bundle / "evaluation_evidence" / "v3_en_he" / "correctness.he.jsonl").unlink()
    else:
        (bundle / "evaluation_evidence" / "v3_en_he" / "notes.txt").write_text(
            "not allowlisted\n",
            encoding="utf-8",
        )

    with pytest.raises(UserInputError, match="exact allowlist"):
        publication._validate_exact_tree(
            bundle,
            required=publication.ADAPTER_REQUIRED_FILES,
            optional=publication.ADAPTER_OPTIONAL_FILES,
        )


def test_adapter_license_files_must_match_reviewed_project_copies(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for filename in ADAPTER_LICENSE_FILE_SHA256:
        shutil.copy2(Path("licenses") / filename, bundle / filename)
    publication._validate_adapter_license_files(bundle)

    (bundle / "LICENSE-LLAMA-3.1.txt").write_text("abbreviated terms\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="does not match the reviewed project copy"):
        publication._validate_adapter_license_files(bundle)


def test_experiment_identity_requires_clean_finalizer_source(tmp_path: Path) -> None:
    report = tmp_path / "experiment_report.json"
    source_revision = "f" * 40
    tree_sha256 = "2" * 64
    finalizer: dict[str, object] = {
        "git_commit": source_revision,
        "working_tree_clean": False,
    }
    payload = {
        "schema_version": EXPERIMENT_REPORT_SCHEMA,
        "arms": {
            "v3_en_he": {
                "run_id": "run-1",
                "config_sha256": "0" * 64,
                "adapter_source": {
                    "source": "runs/run-1/train/adapter",
                    "revision": None,
                    "kind": "local_directory",
                    "tree_sha256": tree_sha256,
                    "artifact_path": "runs/run-1/train/adapter",
                    "revision_is_immutable": True,
                },
            }
        },
        "preregistration": {"finalizer_source_code": finalizer},
        "data_provenance": {"contract": {"source_code_revision": source_revision}},
        "all_claims_passed": False,
        "approved_claims": [],
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match="finalizer source"):
        publication._validate_experiment_identity(
            path=report,
            run_id="run-1",
            source_revision=source_revision,
            config_sha256="0" * 64,
            tree_sha256=tree_sha256,
            dataset_revision="e" * 40,
        )

    finalizer["working_tree_clean"] = True
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UserInputError, match="top-level fields"):
        publication._validate_experiment_identity(
            path=report,
            run_id="run-1",
            source_revision=source_revision,
            config_sha256="0" * 64,
            tree_sha256=tree_sha256,
            dataset_revision="e" * 40,
        )


def test_experiment_identity_rejects_derived_claim_gate_tampering(tmp_path: Path) -> None:
    report_path = tmp_path / "experiment_report.json"
    payload = _valid_experiment_report()
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    publication._validate_experiment_identity(
        path=report_path,
        run_id="run-1",
        source_revision="f" * 40,
        config_sha256="0" * 64,
        tree_sha256="2" * 64,
        dataset_revision="e" * 40,
    )
    arms = cast("dict[str, object]", payload["arms"])
    v3 = cast("dict[str, object]", arms["v3_en_he"])
    source = cast("dict[str, object]", v3["adapter_source"])
    source["source"] = "/mnt/artifacts/runs/run-1/train/adapter"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    publication._validate_experiment_identity(
        path=report_path,
        run_id="run-1",
        source_revision="f" * 40,
        config_sha256="0" * 64,
        tree_sha256="2" * 64,
        dataset_revision="e" * 40,
    )

    payload["all_claims_passed"] = False
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UserInputError, match="all_claims_passed"):
        publication._validate_experiment_identity(
            path=report_path,
            run_id="run-1",
            source_revision="f" * 40,
            config_sha256="0" * 64,
            tree_sha256="2" * 64,
            dataset_revision="e" * 40,
        )


def test_experiment_identity_rejects_incomplete_tco_contract(tmp_path: Path) -> None:
    report_path = tmp_path / "experiment_report.json"
    payload = _valid_experiment_report()
    tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
    del tco["inference_efficiency"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match="TCO evidence has unexpected or missing fields"):
        publication._validate_experiment_identity(
            path=report_path,
            run_id="run-1",
            source_revision="f" * 40,
            config_sha256="0" * 64,
            tree_sha256="2" * 64,
            dataset_revision="e" * 40,
        )


@pytest.mark.parametrize(
    "source_name",
    (
        "he_translation_config",
        "he_translation_run_identity",
        "he_semantic_review",
        "he_semantic_review_template",
        "he_translation_summary",
        "he_translation_publication",
        "he_paired_rows",
        "root_rows",
    ),
)
def test_experiment_identity_requires_every_hebrew_source_binding(
    tmp_path: Path,
    source_name: str,
) -> None:
    report_path = tmp_path / "experiment_report.json"
    payload = _valid_experiment_report()
    provenance = cast("dict[str, object]", payload["data_provenance"])
    sources = cast("dict[str, object]", provenance["sources"])
    del sources[source_name]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match="data_provenance.sources has unexpected or missing"):
        publication._validate_experiment_identity(
            path=report_path,
            run_id="run-1",
            source_revision="f" * 40,
            config_sha256="0" * 64,
            tree_sha256="2" * 64,
            dataset_revision="e" * 40,
        )


def test_experiment_identity_rejects_hebrew_source_identity_tampering(tmp_path: Path) -> None:
    report_path = tmp_path / "experiment_report.json"
    payload = _valid_experiment_report()
    provenance = cast("dict[str, object]", payload["data_provenance"])
    sources = cast("dict[str, object]", provenance["sources"])
    semantic_review = cast("dict[str, object]", sources["he_semantic_review"])
    semantic_review["path"] = "runs/run-1/data/source_inputs/unreviewed.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match="he_semantic_review identity drifted"):
        publication._validate_experiment_identity(
            path=report_path,
            run_id="run-1",
            source_revision="f" * 40,
            config_sha256="0" * 64,
            tree_sha256="2" * 64,
            dataset_revision="e" * 40,
        )


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    (
        (("seed",), 42.0, "data_provenance seed"),
        (("requested_splits", "train"), 15_000.0, "requested train rows"),
        (("semantic_review", "sample_size"), 200.0, "semantic-review sample_size"),
        (
            ("semantic_review", "required_critical_errors"),
            0.0,
            "semantic-review required_critical_errors",
        ),
    ),
)
def test_experiment_identity_rejects_float_provenance_counts(
    tmp_path: Path,
    field_path: tuple[str, ...],
    value: float,
    message: str,
) -> None:
    payload = _valid_experiment_report()
    provenance = cast("dict[str, object]", payload["data_provenance"])
    current = cast("dict[str, object]", provenance["contract"])
    for field in field_path[:-1]:
        current = cast("dict[str, object]", current[field])
    current[field_path[-1]] = value
    report_path = tmp_path / "experiment_report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match=message):
        publication._validate_experiment_identity(
            path=report_path,
            run_id="run-1",
            source_revision="f" * 40,
            config_sha256="0" * 64,
            tree_sha256="2" * 64,
            dataset_revision="e" * 40,
        )


def test_experiment_identity_accepts_pooled_argument_f1_denominators(tmp_path: Path) -> None:
    payload = _valid_experiment_report()
    arms = cast("dict[str, object]", payload["arms"])
    for raw_arm in arms.values():
        arm = cast("dict[str, object]", raw_arm)
        paired_slices = cast("dict[str, object]", arm["paired_slices"])
        paired_hebrew = cast("dict[str, object]", paired_slices["he"])
        pairs = cast(int, paired_hebrew["pairs"])
        for cohort_name in ("reference", "target"):
            cohort = cast("dict[str, object]", paired_hebrew[cohort_name])
            metrics = cast("dict[str, object]", cohort["metrics"])
            argument_f1 = cast("dict[str, object]", metrics["argument_f1"])
            assert cast(int, argument_f1["denominator"]) > pairs

    report_path = tmp_path / "experiment_report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    publication._validate_experiment_identity(
        path=report_path,
        run_id="run-1",
        source_revision="f" * 40,
        config_sha256="0" * 64,
        tree_sha256="2" * 64,
        dataset_revision="e" * 40,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        (
            "missing_row_map_digest",
            "shared_evaluation_identity.paired_cohorts.he has unexpected or missing fields",
        ),
        (
            "invalid_row_map_digest",
            "shared Hebrew reference row indices must be a lowercase SHA-256 digest",
        ),
        ("coverage", "cohort identity or coverage is inconsistent"),
        ("pair_digest", "disagrees with shared evaluation identity"),
        (
            "reference_denominator",
            "reference.metrics per-example denominators do not match the pair count",
        ),
        (
            "target_denominator",
            "target.metrics per-example denominators do not match the pair count",
        ),
        (
            "reference_value",
            "reference.metrics.valid_json_rate value does not match numerator/denominator",
        ),
        (
            "target_value",
            "target.metrics.valid_json_rate value does not match numerator/denominator",
        ),
        ("gap", "gaps.valid_json_rate is not Hebrew minus matched English"),
        ("interval_identity", "gap_ci95 bootstrap contract drifted"),
    ),
)
def test_experiment_identity_rejects_matched_pair_tampering(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    payload = _valid_experiment_report()
    shared = cast("dict[str, object]", payload["shared_evaluation_identity"])
    cohorts = cast("dict[str, object]", shared["paired_cohorts"])
    shared_hebrew = cast("dict[str, object]", cohorts["he"])
    arms = cast("dict[str, object]", payload["arms"])
    base = cast("dict[str, object]", arms["base"])
    paired_slices = cast("dict[str, object]", base["paired_slices"])
    paired_hebrew = cast("dict[str, object]", paired_slices["he"])

    if case == "missing_row_map_digest":
        del shared_hebrew["reference_row_indices_sha256"]
    elif case == "invalid_row_map_digest":
        shared_hebrew["reference_row_indices_sha256"] = "not-a-sha256"
    elif case == "coverage":
        coverage = cast("dict[str, object]", paired_hebrew["coverage"])
        coverage["reference_fraction"] = 0.5
    elif case == "pair_digest":
        paired_hebrew["pair_set_sha256"] = "c" * 64
    elif case in {
        "reference_denominator",
        "target_denominator",
        "reference_value",
        "target_value",
    }:
        cohort_name = "reference" if case.startswith("reference") else "target"
        cohort = cast("dict[str, object]", paired_hebrew[cohort_name])
        metrics = cast("dict[str, object]", cohort["metrics"])
        metric = cast("dict[str, object]", metrics["valid_json_rate"])
        if case.endswith("denominator"):
            metric["denominator"] = 3
            numerator = cast(int, metric["numerator"])
            metric["value"] = numerator / 3
        else:
            metric["value"] = 0.5
    elif case == "gap":
        gaps = cast("dict[str, object]", paired_hebrew["gaps"])
        gaps["valid_json_rate"] = 0.0
    elif case == "interval_identity":
        interval = cast("dict[str, object]", paired_hebrew["gap_ci95"])
        interval["seed"] = cast(int, interval["seed"]) + 1
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)

    report_path = tmp_path / "experiment_report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UserInputError, match=message):
        publication._validate_experiment_identity(
            path=report_path,
            run_id="run-1",
            source_revision="f" * 40,
            config_sha256="0" * 64,
            tree_sha256="2" * 64,
            dataset_revision="e" * 40,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("preregistration", "preregistered bootstrap contract drifted"),
        ("shared_identity", "shared_evaluation_identity has unexpected or missing fields"),
        ("data_provenance", "data_provenance contract identity drifted"),
        ("claim_passed", "claims.hebrew_full_call_uplift disagrees with the derived gate"),
        ("comparison_bound", "claims.hebrew_full_call_uplift"),
        ("approved_claims", "approved_claims disagrees with the derived gates"),
        ("tco_policy", "TCO unavailable cost claims drifted"),
        ("tco_counterfactual", "TCO English-only counterfactual is inconsistent"),
        ("tco_multiplier", "TCO combined-vs-English multiplier is inconsistent"),
        ("tco_scope_negative", "TCO paired train full_tokens"),
        ("tco_scope_ratio", "TCO paired train coverage is inconsistent"),
        ("tco_training_negative_seconds", "TCO train elapsed_seconds"),
        ("tco_training_throughput", "TCO training throughput is inconsistent"),
        ("tco_adapter_negative_bytes", "TCO packaged adapter bytes"),
        ("tco_inference_scope", "TCO inference arm base measurement contract drifted"),
        ("tco_inference_negative_seconds", "TCO base en elapsed seconds"),
        ("tco_inference_ratio", "TCO base en GPU-seconds ratio is inconsistent"),
        ("tco_sources_empty", "TCO tokenization sources"),
    ),
)
def test_experiment_identity_rejects_internal_contract_tampering(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    payload = _valid_experiment_report()
    if case == "preregistration":
        preregistration = cast("dict[str, object]", payload["preregistration"])
        bootstrap = cast("dict[str, object]", preregistration["bootstrap"])
        bootstrap["resamples"] = 1999
    elif case == "shared_identity":
        shared = cast("dict[str, object]", payload["shared_evaluation_identity"])
        del shared["paired_cohorts"]
    elif case == "data_provenance":
        provenance = cast("dict[str, object]", payload["data_provenance"])
        contract = cast("dict[str, object]", provenance["contract"])
        paired = cast("dict[str, object]", contract["paired_dataset"])
        paired["dataset_revision"] = "main"
    elif case == "claim_passed":
        claims = cast("dict[str, object]", payload["claims"])
        claim = cast("dict[str, object]", claims["hebrew_full_call_uplift"])
        claim["passed"] = False
    elif case == "comparison_bound":
        comparisons = cast("dict[str, object]", payload["comparisons"])
        comparison = cast("dict[str, object]", comparisons["v3_vs_v1"])
        hebrew = cast("dict[str, object]", comparison["he"])
        ci95 = cast("dict[str, object]", hebrew["ci95"])
        intervals = cast("dict[str, object]", ci95["intervals"])
        full_call = cast("dict[str, object]", intervals["full_call_exact_match"])
        full_call["lower"] = -0.1
    elif case == "approved_claims":
        approved = cast("list[object]", payload["approved_claims"])
        approved.reverse()
    elif case == "tco_policy":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        unavailable = cast("dict[str, object]", tco["explicitly_unavailable"])
        currency = cast("dict[str, object]", unavailable["currency_cost"])
        currency["available"] = True
    elif case == "tco_counterfactual":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        paired = cast("dict[str, object]", tco["paired_tokenization"])
        workload = cast("dict[str, object]", paired["projected_training_workload"])
        english = cast("dict[str, object]", workload["english_only_counterfactual"])
        english["projected_non_padding_full_tokens"] = 399_999
    elif case == "tco_multiplier":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        paired = cast("dict[str, object]", tco["paired_tokenization"])
        workload = cast("dict[str, object]", paired["projected_training_workload"])
        multipliers = cast("dict[str, object]", workload["combined_vs_english_only"])
        multipliers["non_padding_full_tokens_per_epoch_multiplier"] = 2.4
    elif case == "tco_scope_negative":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        paired = cast("dict[str, object]", tco["paired_tokenization"])
        scopes = cast("dict[str, object]", paired["paired_scopes"])
        train = cast("dict[str, object]", scopes["train"])
        ratios = cast("dict[str, object]", train["token_ratios"])
        full = cast("dict[str, object]", ratios["full_tokens"])
        full["paired_hebrew_tokens"] = -1
    elif case == "tco_scope_ratio":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        paired = cast("dict[str, object]", tco["paired_tokenization"])
        scopes = cast("dict[str, object]", paired["paired_scopes"])
        train = cast("dict[str, object]", scopes["train"])
        coverage = cast("dict[str, object]", train["coverage"])
        coverage["ratio"] = 0.1
    elif case == "tco_training_negative_seconds":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        training = cast("dict[str, object]", tco["qlora_training"])
        runtime = cast("dict[str, object]", training["train_stage_runtime"])
        runtime["elapsed_seconds"] = -1.0
    elif case == "tco_training_throughput":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        training = cast("dict[str, object]", tco["qlora_training"])
        throughput = cast("dict[str, object]", training["end_to_end_token_throughput"])
        throughput["value"] = 999.0
    elif case == "tco_adapter_negative_bytes":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        training = cast("dict[str, object]", tco["qlora_training"])
        storage = cast("dict[str, object]", training["adapter_storage"])
        packaged = cast("dict[str, object]", storage["packaged_adapter"])
        packaged["bytes"] = -1
    elif case == "tco_inference_scope":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        inference = cast("dict[str, object]", tco["inference_efficiency"])
        arms = cast("dict[str, object]", inference["arms"])
        base = cast("dict[str, object]", arms["base"])
        measurement = cast("dict[str, object]", base["measurement"])
        measurement["scope"] = "model.generate_only"
    elif case == "tco_inference_negative_seconds":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        inference = cast("dict[str, object]", tco["inference_efficiency"])
        arms = cast("dict[str, object]", inference["arms"])
        base = cast("dict[str, object]", arms["base"])
        slices = cast("dict[str, object]", base["slices"])
        english = cast("dict[str, object]", slices["en"])
        english["generation_elapsed_seconds"] = -1.0
    elif case == "tco_inference_ratio":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        inference = cast("dict[str, object]", tco["inference_efficiency"])
        arms = cast("dict[str, object]", inference["arms"])
        base = cast("dict[str, object]", arms["base"])
        slices = cast("dict[str, object]", base["slices"])
        english = cast("dict[str, object]", slices["en"])
        ratio = cast(
            "dict[str, object]",
            english["configured_gpu_seconds_per_full_call_exact_success"],
        )
        ratio["value"] = 999.0
    elif case == "tco_sources_empty":
        tco = cast("dict[str, object]", payload["sovereign_tco_evidence"])
        sources = cast("dict[str, object]", tco["sources"])
        sources["tokenization"] = {}
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)

    report_path = tmp_path / "experiment_report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UserInputError, match=message):
        publication._validate_experiment_identity(
            path=report_path,
            run_id="run-1",
            source_revision="f" * 40,
            config_sha256="0" * 64,
            tree_sha256="2" * 64,
            dataset_revision="e" * 40,
        )


def test_mcnemar_rejects_oversized_cohort_before_combinatorics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_comb(*args: object) -> int:
        nonlocal calls
        calls += 1
        raise AssertionError(f"math.comb must not run for oversized evidence: {args!r}")

    monkeypatch.setattr(math, "comb", forbidden_comb)
    with pytest.raises(UserInputError, match="McNemar pairs exceeds the 1000-example"):
        publication._validate_mcnemar(
            {
                "method": "sommelier.exact_mcnemar.v1",
                "metric": "full_call_exact_match",
                "alternative": "two-sided",
                "pairs": 10**12,
                "discordant_pairs": 10**12,
                "discordant_counts": {
                    "reference_correct_candidate_incorrect": 5 * 10**11,
                    "reference_incorrect_candidate_correct": 5 * 10**11,
                },
                "p_value": 1.0,
            },
            language="he",
        )
    assert calls == 0


def test_adapter_card_requires_exact_claim_gated_result_section(tmp_path: Path) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    report = _valid_experiment_report()
    card = tmp_path / "README.md"
    card.write_text(
        _valid_adapter_card(
            config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
        ),
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="claim-gated result section"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )


def test_claim_section_renderer_is_deterministic() -> None:
    assert publication.render_hebrew_v3_claim_section(_valid_experiment_report()) == (
        "## Claim-gated result\n"
        "\n"
        "- **Hebrew full-call uplift versus immutable v1 — passed.** "
        "Estimate `0.5`; 95% paired-bootstrap CI `[0.25, 0.75]`. "
        f"{_HEBREW_UPLIFT_STATEMENT}\n"
        "- **English full-call non-inferiority at the 0.01 absolute margin — passed.** "
        "Estimate `0.0`; 95% paired-bootstrap CI `[-0.01, 0.01]`. "
        f"{_ENGLISH_NON_INFERIORITY_STATEMENT}"
    )


def test_adapter_card_rejects_duplicate_claim_gated_heading(tmp_path: Path) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    report = _valid_experiment_report()
    card = tmp_path / "README.md"
    card.write_text(
        _valid_adapter_card(
            config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )
        + "\n\n## Claim-gated result\n",
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="exactly one Claim-gated result heading"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )


@pytest.mark.parametrize(
    "prose",
    (
        "Hebrew full-call exact match improved by 50% over v1.",
        "Hebrew accuracy: 75%.",
        "English accuracy remains uncompromised.",
    ),
)
def test_adapter_card_rejects_result_prose_outside_rendered_section(
    tmp_path: Path,
    prose: str,
) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    report = _valid_experiment_report()
    card = tmp_path / "README.md"
    card.write_text(
        _valid_adapter_card(
            config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )
        + f"\n\n{prose}\n",
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="unapproved result or claim prose"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )


def test_adapter_card_withholds_failed_claim_and_rejects_reinsertion(tmp_path: Path) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    report = _valid_experiment_report()
    comparisons = cast("dict[str, object]", report["comparisons"])
    comparison = cast("dict[str, object]", comparisons["v3_vs_v1"])
    hebrew = cast("dict[str, object]", comparison["he"])
    ci95 = cast("dict[str, object]", hebrew["ci95"])
    intervals = cast("dict[str, object]", ci95["intervals"])
    full_call = cast("dict[str, object]", intervals["full_call_exact_match"])
    full_call["lower"] = -0.1
    claims = cast("dict[str, object]", report["claims"])
    claim = cast("dict[str, object]", claims["hebrew_full_call_uplift"])
    claim["passed"] = False
    claim_ci = cast("dict[str, object]", claim["ci95"])
    claim_ci["lower"] = -0.1
    del claim["statement"]
    report["all_claims_passed"] = False
    report["approved_claims"] = [_ENGLISH_NON_INFERIORITY_STATEMENT]

    card = tmp_path / "README.md"
    card.write_text(
        _valid_adapter_card(
            config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        ),
        encoding="utf-8",
    )
    publication._validate_adapter_card(
        card,
        config=config,
        experiment_sha256="1" * 64,
        tree_sha256="2" * 64,
        source_revision="f" * 40,
        dataset_revision=config.dataset_for("he").dataset_revision,
        experiment_report=report,
    )
    assert "Hebrew full-call uplift versus immutable v1 — withheld" in card.read_text(
        encoding="utf-8"
    )
    assert _HEBREW_UPLIFT_STATEMENT not in card.read_text(encoding="utf-8")

    card.write_text(
        card.read_text(encoding="utf-8") + f"\n\n{_HEBREW_UPLIFT_STATEMENT}\n",
        encoding="utf-8",
    )
    with pytest.raises(UserInputError, match="unapproved claim statement"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )


def test_adapter_card_rejects_uncompromised_accuracy_prose_when_english_gate_is_withheld(
    tmp_path: Path,
) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    report = _valid_experiment_report()
    comparisons = cast("dict[str, object]", report["comparisons"])
    comparison = cast("dict[str, object]", comparisons["v3_vs_v1"])
    english = cast("dict[str, object]", comparison["en"])
    ci95 = cast("dict[str, object]", english["ci95"])
    intervals = cast("dict[str, object]", ci95["intervals"])
    full_call = cast("dict[str, object]", intervals["full_call_exact_match"])
    full_call["lower"] = -0.02
    claims = cast("dict[str, object]", report["claims"])
    claim = cast("dict[str, object]", claims["english_full_call_non_inferiority"])
    claim["passed"] = False
    claim_ci = cast("dict[str, object]", claim["ci95"])
    claim_ci["lower"] = -0.02
    del claim["statement"]
    report["all_claims_passed"] = False
    report["approved_claims"] = [_HEBREW_UPLIFT_STATEMENT]

    card = tmp_path / "README.md"
    card.write_text(
        _valid_adapter_card(
            config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )
        + "\n\nEnglish accuracy remains uncompromised.\n",
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="unapproved result or claim prose"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )


def test_adapter_card_requires_llama_and_nvidia_obligations(tmp_path: Path) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    report = _valid_experiment_report()
    card = tmp_path / "README.md"
    experiment_sha256 = "1" * 64
    tree_sha256 = "2" * 64
    card.write_text(
        _valid_adapter_card(
            config,
            experiment_sha256=experiment_sha256,
            tree_sha256=tree_sha256,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        ),
        encoding="utf-8",
    )
    publication._validate_adapter_card(
        card,
        config=config,
        experiment_sha256=experiment_sha256,
        tree_sha256=tree_sha256,
        source_revision="f" * 40,
        dataset_revision=config.dataset_for("he").dataset_revision,
        experiment_report=report,
    )

    card.write_text(
        card.read_text(encoding="utf-8").replace("NVIDIA Open Model License", ""),
        encoding="utf-8",
    )
    with pytest.raises(UserInputError, match="NVIDIA Open Model License"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256=experiment_sha256,
            tree_sha256=tree_sha256,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )


def test_adapter_card_rejects_wrong_frontmatter_and_unresolved_markers(tmp_path: Path) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    report = _valid_experiment_report()
    card = tmp_path / "README.md"
    text = _valid_adapter_card(
        config,
        experiment_sha256="1" * 64,
        tree_sha256="2" * 64,
        source_revision="f" * 40,
        dataset_revision=config.dataset_for("he").dataset_revision,
        experiment_report=report,
    )
    card.write_text(text.replace("license: llama3.1", "license: mit"), encoding="utf-8")
    with pytest.raises(UserInputError, match="frontmatter license"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )

    card.write_text(text + "\nREPLACE_FROM_VERIFIED_BUNDLE\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="unresolved release template markers"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )


@pytest.mark.parametrize(
    "replacement",
    (
        "# Hebrew tool-calling adapter",
        "## Llama Hebrew tool-calling adapter",
        "# Hebrew adapter\n# Llama appears too late",
    ),
)
def test_adapter_card_first_markdown_h1_must_begin_with_llama(
    tmp_path: Path,
    replacement: str,
) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    report = _valid_experiment_report()
    card = tmp_path / "README.md"
    text = _valid_adapter_card(
        config,
        experiment_sha256="1" * 64,
        tree_sha256="2" * 64,
        source_revision="f" * 40,
        dataset_revision=config.dataset_for("he").dataset_revision,
        experiment_report=report,
    )
    card.write_text(
        text.replace("# Llama Hebrew tool-calling adapter", replacement),
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="first Markdown H1 must begin with 'Llama'"):
        publication._validate_adapter_card(
            card,
            config=config,
            experiment_sha256="1" * 64,
            tree_sha256="2" * 64,
            source_revision="f" * 40,
            dataset_revision=config.dataset_for("he").dataset_revision,
            experiment_report=report,
        )


def test_release_evidence_is_identity_bound_and_requires_notices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source_revision = "f" * 40
    dependency_lock_sha256 = sha256_file(Path("uv.lock"))
    source_identity = release.SourceCodeIdentity(
        discovery="git-project-root-v1",
        git_commit=source_revision,
        working_tree_clean=True,
        git_status_sha256=release._EMPTY_SHA256,
    )
    lock_identity = release.DependencyLockIdentity(
        path="uv.lock",
        sha256=dependency_lock_sha256,
        bytes=Path("uv.lock").stat().st_size,
    )
    monkeypatch.setattr(release, "_discover_project_source", lambda _: source_identity)
    monkeypatch.setattr(
        release,
        "_project_file_matches_revision",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        release,
        "_dependency_lock_identity",
        lambda *_args, **_kwargs: lock_identity,
    )

    (bundle / "release_preflight.json").write_text(
        json.dumps(
            {
                "schema_version": release.PREFLIGHT_SCHEMA,
                "status": "pass",
                "gates": [],
            }
        ),
        encoding="utf-8",
    )
    notices = "\n".join(
        (
            config.model.base_model_id,
            "NVIDIA Open Model License",
            "Llama 3.1 Community License",
            REQUIRED_DERIVED_NOTICE,
            config.root_dataset.dataset_id,
            "CC-BY-4.0",
        )
    )
    (bundle / "THIRD_PARTY.md").write_text(notices, encoding="utf-8")
    with pytest.raises(UserInputError, match="identity is invalid"):
        publication._validate_release_evidence(
            bundle,
            config,
            source_revision=source_revision,
            dependency_lock_sha256=dependency_lock_sha256,
        )

    run_release_preflight(
        config,
        project_root=Path.cwd(),
        artifact_root=bundle,
        environ={ACK_ENV_NAME: config.model.base_model_id},
    )
    publication._validate_release_evidence(
        bundle,
        config,
        source_revision=source_revision,
        dependency_lock_sha256=dependency_lock_sha256,
    )

    (bundle / "THIRD_PARTY.md").write_text(
        notices.replace("NVIDIA Open Model License", ""),
        encoding="utf-8",
    )
    run_release_preflight(
        config,
        project_root=Path.cwd(),
        artifact_root=bundle,
        environ={ACK_ENV_NAME: config.model.base_model_id},
    )
    with pytest.raises(UserInputError, match="NVIDIA Open Model License"):
        publication._validate_release_evidence(
            bundle,
            config,
            source_revision=source_revision,
            dependency_lock_sha256=dependency_lock_sha256,
        )


def _write_safetensors(path: Path, tensors: dict[str, object]) -> None:
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    tensor_metadata = [
        cast("dict[str, object]", metadata)
        for name, metadata in tensors.items()
        if name != "__metadata__"
    ]
    payload_size = max(
        cast("list[int]", metadata["data_offsets"])[1] for metadata in tensor_metadata
    )
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(payload_size))


def _write_raw_safetensors_header(path: Path, header: str, *, payload_bytes: int) -> None:
    encoded = header.encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(payload_bytes))


def test_safetensors_gate_accepts_only_complete_lora_pairs(tmp_path: Path) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    tensors: dict[str, object] = {
        "__metadata__": {"format": "pt"},
        f"{prefix}A.default.weight": {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [0, 2],
        },
        f"{prefix}B.default.weight": {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [2, 4],
        },
    }
    _write_safetensors(path, tensors)
    publication._validate_safetensors(path)

    tensors.pop(f"{prefix}B.default.weight")
    _write_safetensors(path, tensors)
    with pytest.raises(UserInputError, match="incomplete LoRA"):
        publication._validate_safetensors(path)


def test_safetensors_gate_rejects_base_model_tensor(tmp_path: Path) -> None:
    path = tmp_path / "adapter_model.safetensors"
    _write_safetensors(
        path,
        {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        },
    )
    with pytest.raises(UserInputError, match="non-LoRA tensor"):
        publication._validate_safetensors(path)


def test_safetensors_gate_scans_metadata_for_secrets_without_exposing_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    _write_safetensors(
        path,
        {
            "__metadata__": {"training_note": secret},
            f"{prefix}A.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            },
            f"{prefix}B.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [2, 4],
            },
        },
    )

    with pytest.raises(SecurityPolicyError, match="__metadata__ contains") as captured:
        publication._validate_safetensors(path)

    assert secret not in str(captured.value)


@pytest.mark.parametrize("duplicate_location", ("metadata", "tensor"))
def test_safetensors_gate_rejects_duplicate_json_keys_without_exposing_values(
    tmp_path: Path,
    duplicate_location: str,
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    metadata = (
        f'"__metadata__":{{"note":"{secret}"}},"__metadata__":{{"format":"pt"}},'
        if duplicate_location == "metadata"
        else '"__metadata__":{"format":"pt"},'
    )
    tensor_a = (
        f'"{prefix}A.default.weight":'
        '{"dtype":"F16","dtype":"F32","shape":[1],"data_offsets":[0,2]},'
        if duplicate_location == "tensor"
        else f'"{prefix}A.default.weight":{{"dtype":"F16","shape":[1],"data_offsets":[0,2]}},'
    )
    header = (
        "{" + metadata + tensor_a + f'"{prefix}B.default.weight":'
        '{"dtype":"F16","shape":[1],"data_offsets":[2,4]}}'
    )
    _write_raw_safetensors_header(path, header, payload_bytes=4)

    with pytest.raises(UserInputError, match="invalid JSON header") as captured:
        publication._validate_safetensors(path)

    assert secret not in str(captured.value)


@pytest.mark.parametrize("mutation", ("leading-space", "trailing-tab"))
def test_safetensors_gate_rejects_noncanonical_header_padding(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    header = json.dumps(
        {
            f"{prefix}A.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            },
            f"{prefix}B.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [2, 4],
            },
        },
        separators=(",", ":"),
    )
    malformed = f" {header}" if mutation == "leading-space" else f"{header}\t"
    _write_raw_safetensors_header(path, malformed, payload_bytes=4)

    with pytest.raises(UserInputError, match="header envelope"):
        publication._validate_safetensors(path)


@pytest.mark.parametrize(
    ("dtype", "shape", "offsets", "error"),
    (
        ("NOT_A_DTYPE", [1], ([0, 2], [2, 4]), "incomplete"),
        ("F16", [-1], ([0, 2], [2, 4]), "incomplete"),
        ("F16", [True], ([0, 2], [2, 4]), "incomplete"),
        ("F16", [2], ([0, 2], [2, 4]), "does not match its offsets"),
        ("F4", [1], ([0, 0], [0, 0]), "does not match its offsets"),
    ),
)
def test_safetensors_gate_rejects_invalid_dtype_shape_and_byte_span(
    tmp_path: Path,
    dtype: str,
    shape: list[object],
    offsets: tuple[list[int], list[int]],
    error: str,
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    _write_safetensors(
        path,
        {
            f"{prefix}A.default.weight": {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": offsets[0],
            },
            f"{prefix}B.default.weight": {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": offsets[1],
            },
        },
    )

    with pytest.raises(UserInputError, match=error):
        publication._validate_safetensors(path)


@pytest.mark.parametrize(
    ("shape", "offsets"),
    (
        ([0, 4], ([0, 0], [0, 0])),
        ([], ([0, 2], [2, 4])),
    ),
)
def test_safetensors_gate_preserves_empty_and_scalar_tensors(
    tmp_path: Path,
    shape: list[int],
    offsets: tuple[list[int], list[int]],
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    _write_safetensors(
        path,
        {
            f"{prefix}A.default.weight": {
                "dtype": "F16",
                "shape": shape,
                "data_offsets": offsets[0],
            },
            f"{prefix}B.default.weight": {
                "dtype": "F16",
                "shape": shape,
                "data_offsets": offsets[1],
            },
        },
    )

    publication._validate_safetensors(path)


def test_safetensors_gate_rejects_unrecognized_tensor_metadata_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adapter_model.safetensors"
    prefix = "base_model.model.layers.0.self_attn.q_proj.lora_"
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    _write_safetensors(
        path,
        {
            f"{prefix}A.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
                "note": secret,
            },
            f"{prefix}B.default.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [2, 4],
            },
        },
    )

    with pytest.raises(UserInputError, match="unexpected fields") as captured:
        publication._validate_safetensors(path)

    assert secret not in str(captured.value)


def test_adapter_config_must_match_bound_base_and_qlora_contract(tmp_path: Path) -> None:
    config, _ = _immutable_hebrew_config(tmp_path)
    path = tmp_path / "adapter_config.json"
    payload: dict[str, object] = {
        "base_model_name_or_path": config.model.base_model_id,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": config.train.lora_rank,
        "lora_alpha": config.train.lora_alpha,
        "lora_dropout": config.train.lora_dropout,
        "bias": "none",
        "target_modules": list(config.train.target_modules),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    publication._validate_adapter_config(path, config)

    payload["base_model_name_or_path"] = "other/model"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UserInputError, match="base_model_name_or_path"):
        publication._validate_adapter_config(path, config)


def test_prepared_hashes_are_deterministic_and_bound_to_bytes(tmp_path: Path) -> None:
    prepared = _prepared_dataset(tmp_path)
    assert prepared.sha256 == {
        name: sha256_file(path) for name, path in sorted(prepared.files.items())
    }
    readme = prepared.files["README.md"]
    before = prepared.sha256["README.md"]
    readme.write_text("changed\n", encoding="utf-8")
    assert prepared.sha256["README.md"] != before
