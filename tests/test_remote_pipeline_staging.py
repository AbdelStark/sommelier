from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import sommelier.data.export as export_module
from remote_pipeline import _stage_paired_rows
from sommelier.artifacts import sha256_file
from sommelier.config import DatasetSourceConfig, load_config
from sommelier.data.load import load_raw_rows
from sommelier.data.semantic_review import (
    EXPECTED_PRODUCER_PACKAGE_VERSIONS,
    SEMANTIC_REVIEW_FILENAME,
    SEMANTIC_REVIEW_TEMPLATE_FILENAME,
    SemanticReviewProducerProvenance,
    create_semantic_review_attestation,
    create_semantic_review_template,
    finalize_semantic_review,
    root_split_assignments,
)
from sommelier.data.translate import (
    PUBLICATION_CANONICAL_FIELDS,
    PUBLICATION_MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    TRANSLATION_CONFIG_FILENAME,
    TRANSLATION_RUN_IDENTITY_FILENAME,
    TRANSLATION_RUN_IDENTITY_SCHEMA,
    published_rows_canonical_identity,
    translation_selection_contract_sha256,
    write_translation_publication_manifest,
)
from sommelier.errors import UserInputError
from sommelier.hebrew_v3_preregistration import (
    reviewer_anchor_payload,
    reviewer_anchor_sha256,
)
from sommelier.reviewer import (
    ReviewerRequirement,
    canonical_reviewer_requirement,
    validated_reviewer_requirement,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
REVIEWER_PUBLIC_KEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4f"
)
REVIEWER_REQUIREMENT = canonical_reviewer_requirement(
    "fixture-reviewer",
    REVIEWER_PUBLIC_KEY,
)


def _full_config(
    tmp_path: Path,
    *,
    paired_revision: str,
    reviewer_requirement: ReviewerRequirement = REVIEWER_REQUIREMENT,
) -> Path:
    config_text = (EXAMPLES_DIR / "config.v3-he-full.yaml").read_text(encoding="utf-8")
    config_text = (
        config_text.replace("n_train: 15000", "n_train: 200")
        .replace("n_validation: 1000", "n_validation: 1")
        .replace("n_test: 1000", "n_test: 1")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        config_text.replace("dataset_revision: main", f"dataset_revision: {paired_revision}")
        + "\nsemantic_review:\n"
        + "  reviewer:\n"
        + f"    reviewer_id: {reviewer_requirement.reviewer_id}\n"
        + f"    ssh_public_key: {reviewer_requirement.ssh_public_key}\n"
        + (f"    public_key_fingerprint: {reviewer_requirement.public_key_fingerprint}\n"),
        encoding="utf-8",
    )
    return config_path


class _StubBacktranslator:
    def translate_batch(self, texts: list[str]) -> list[str]:
        return [f"Backtranslated {text}" for text in texts]


def test_full_pipeline_exports_pinned_paired_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "d" * 40
    private_key = tmp_path / "reviewer-test-key"
    subprocess.run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "",
            "-f",
            str(private_key),
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )
    reviewer_requirement = validated_reviewer_requirement(
        "fixture-reviewer",
        private_key.with_suffix(".pub").read_text(encoding="ascii"),
    )
    config_path = _full_config(
        tmp_path,
        paired_revision=revision,
        reviewer_requirement=reviewer_requirement,
    )
    config = load_config(config_path)
    translation_config_path = tmp_path / TRANSLATION_CONFIG_FILENAME
    translation_config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            f"dataset_revision: {revision}",
            "dataset_revision: main",
        ),
        encoding="utf-8",
    )
    translation_config = load_config(translation_config_path)
    root_rows_path = tmp_path / "rows.en.jsonl"
    root_records: list[dict[str, object]] = []
    paired_records: list[dict[str, object]] = []
    for index in range(202):
        source_id = f"en-{index:03d}"
        tool_name = f"search_items_{index % 5}"
        tools = json.dumps(
            [
                {
                    "name": tool_name,
                    "description": "Search items",
                    "parameters": {"type": "object"},
                }
            ]
        )
        answers = json.dumps([{"name": tool_name, "arguments": {"count": index}}])
        root_records.append(
            {
                "schema_version": "sommelier.raw_tool_call_row.v1",
                "source_id": source_id,
                "query": f"Find exactly {index} items for this account",
                "tools": tools,
                "answers": answers,
                "source_revision": config.root_dataset.dataset_revision,
            }
        )
        paired_records.append(
            {
                "schema_version": "sommelier.raw_tool_call_row.v1",
                # HF export intentionally rewrites this producer-side field.
                "source_id": f"published-he:{index}",
                "source_example_id": source_id,
                "query": f"מצא בדיוק {index} פריטים עבור החשבון הזה",
                "tools": tools,
                "answers": answers,
                "source_revision": revision,
            }
        )
    root_rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in root_records),
        encoding="utf-8",
    )
    calls: list[tuple[str, str, int, int]] = []
    exported_path: Path | None = None

    def fake_export(
        source: DatasetSourceConfig,
        out_path: Path,
        *,
        seed: int,
        max_rows: int,
    ) -> int:
        nonlocal exported_path
        calls.append((source.language, source.dataset_revision, seed, max_rows))
        exported_path = out_path
        out_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in paired_records),
            encoding="utf-8",
        )
        return len(paired_records)

    summary_path = tmp_path / SUMMARY_FILENAME
    publication_path = tmp_path / PUBLICATION_MANIFEST_FILENAME
    template_path = tmp_path / SEMANTIC_REVIEW_TEMPLATE_FILENAME
    review_path = tmp_path / SEMANTIC_REVIEW_FILENAME
    identity_path = tmp_path / TRANSLATION_RUN_IDENTITY_FILENAME

    ordered_ids = [str(row["source_id"]) for row in root_records]
    selection_contract_sha256 = translation_selection_contract_sha256(
        translation_config,
        mode="full",
        max_rows=60_000,
        limit=0,
    )
    reviewer_preregistration = reviewer_anchor_payload(translation_config)
    reviewer_preregistration_sha256 = reviewer_anchor_sha256(translation_config)
    run_identity = {
        "schema_version": TRANSLATION_RUN_IDENTITY_SCHEMA,
        "run_id": "hebrew-v3-fixture",
        "config_sha256": sha256_file(translation_config_path),
        "selection": {
            "contract_sha256": selection_contract_sha256,
            "mode": "full",
            "max_rows": 60_000,
            "limit": 0,
            "seed": config.project.seed,
        },
        "translator": {
            "model_id": "dicta-il/DictaLM-3.0-Nemotron-12B-Instruct",
            "model_revision": "b" * 40,
            "request_sha256": "c" * 64,
            "max_attempts": 3,
            "implementation_revision": "a" * 40,
        },
        "runtime": {
            "backend": "openai_responses",
            "translation_chunk_size": 32,
            "allocation_gpu": None,
            "function_timeout_seconds": 3_600,
            "provider_service_tier": "flex",
            "provider_sdk_version": "2.45.0",
            "provider_timeout_seconds": 900.0,
            "provider_max_workers": 8,
            "openai_list_price_limit_usd": "50.00",
        },
        "source_code": {
            "git_commit": "a" * 40,
            "working_tree_clean": True,
        },
        "reviewer_preregistration": reviewer_preregistration,
        "reviewer_preregistration_sha256": reviewer_preregistration_sha256,
    }
    identity_path.write_text(
        json.dumps(run_identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def ensure_semantic_review() -> None:
        assert exported_path is not None
        if review_path.exists():
            return
        root_rows = load_raw_rows(root_rows_path)
        split_by_id = root_split_assignments(config, root_rows)
        create_semantic_review_template(
            root_rows_path=root_rows_path,
            paired_rows_path=exported_path,
            translation_summary_path=summary_path,
            root_split_by_id=split_by_id,
            output_path=template_path,
            backtranslator=_StubBacktranslator(),
            seed=config.project.seed,
            producer_provenance=SemanticReviewProducerProvenance(
                code_revision="a" * 40,
                working_tree_clean=True,
                execution_boundary="modal_gpu",
                provider="modal",
                hardware="A10G",
                allocation_timeout_seconds=14_400,
                package_versions=dict(EXPECTED_PRODUCER_PACKAGE_VERSIONS),
            ),
            reviewer_requirement=reviewer_requirement,
        )
        reviewed_path = tmp_path / "reviewed.json"
        shutil.copy2(template_path, reviewed_path)
        reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
        for record in reviewed["records"]:
            record["review"] = {
                "rubric": {
                    "action_tool_intent": "pass",
                    "omissions_additions": "pass",
                    "polarity": "not_applicable",
                    "quantities": "pass",
                    "entity_relations": "pass",
                },
                "critical_error": False,
                "passes_review": True,
                "notes": "",
            }
        reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
        attestation_path = tmp_path / "review-attestation.json"
        create_semantic_review_attestation(
            reviewed_path,
            attestation_path,
            template_path=template_path,
            root_rows_path=root_rows_path,
            paired_rows_path=exported_path,
            translation_summary_path=summary_path,
            root_split_by_id=split_by_id,
            expected_seed=config.project.seed,
            backtranslator=_StubBacktranslator(),
        )
        subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(private_key),
                "-n",
                "sommelier-hebrew-v3-semantic-review",
                str(attestation_path),
            ],
            check=True,
            capture_output=True,
            timeout=10,
        )
        finalize_semantic_review(
            reviewed_path,
            review_path,
            template_path=template_path,
            attestation_path=attestation_path,
            signature_path=Path(f"{attestation_path}.sig"),
            root_rows_path=root_rows_path,
            paired_rows_path=exported_path,
            translation_summary_path=summary_path,
            root_split_by_id=split_by_id,
            expected_seed=config.project.seed,
            backtranslator=_StubBacktranslator(),
        )

    def fake_download(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str,
    ) -> str:
        assert (repo_id, repo_type, revision) == (
            "abdelstark/sommelier-xlam-single-call-splits-he",
            "dataset",
            "d" * 40,
        )
        assert exported_path is not None
        if filename == SUMMARY_FILENAME:
            rows, canonical_sha256 = published_rows_canonical_identity(exported_path)
            summary_path.write_text(
                json.dumps(
                    {
                        "schema_version": "sommelier.translation_summary.v2",
                        "language": "he",
                        "input": {"sha256": sha256_file(root_rows_path)},
                        "input_rows": len(root_records),
                        "translated_rows": rows,
                        "selection": {
                            "config_sha256": sha256_file(translation_config_path),
                            "contract_sha256": selection_contract_sha256,
                            "mode": "full",
                            "seed": config.project.seed,
                            "max_rows": 60_000,
                            "limit": 0,
                            "selected_rows": len(ordered_ids),
                            "selected_source_ids_sha256": hashlib.sha256(
                                "\n".join(ordered_ids).encode()
                            ).hexdigest(),
                        },
                        "publication_identity": {
                            "rows": rows,
                            "canonical_fields": list(PUBLICATION_CANONICAL_FIELDS),
                            "canonical_sha256": canonical_sha256,
                        },
                        "translation_run_identity_sha256": sha256_file(identity_path),
                        "max_attempts": 3,
                        "runtime": {
                            "backend": "openai_responses",
                            "translation_chunk_size": 32,
                            "gpu_allocation_label": None,
                            "function_timeout_seconds": 3_600,
                            "provider_service_tier": "flex",
                            "provider_timeout_seconds": 900.0,
                            "provider_max_workers": 8,
                            "openai_list_price_ceiling": {"limit_usd": "50.00"},
                        },
                        "source_code": {
                            "git_commit": "a" * 40,
                            "working_tree_clean": True,
                        },
                        "translator": {
                            "model_id": "dicta-il/DictaLM-3.0-Nemotron-12B-Instruct",
                            "model_revision": "b" * 40,
                            "implementation_revision": "a" * 40,
                            "request_sha256": "c" * 64,
                            "provider_request": {"sdk_version": "2.45.0"},
                        },
                        "reviewer_preregistration": reviewer_preregistration,
                        "reviewer_preregistration_sha256": (reviewer_preregistration_sha256),
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path)
        if filename == TRANSLATION_CONFIG_FILENAME:
            return str(translation_config_path)
        if filename == TRANSLATION_RUN_IDENTITY_FILENAME:
            return str(identity_path)
        if filename == PUBLICATION_MANIFEST_FILENAME:
            ensure_semantic_review()
            write_translation_publication_manifest(
                publication_path,
                translated_rows_path=exported_path,
                summary_path=summary_path,
                target_language="he",
                semantic_review_path=review_path,
                semantic_review_template_path=template_path,
            )
            return str(publication_path)
        if filename == SEMANTIC_REVIEW_FILENAME:
            ensure_semantic_review()
            return str(review_path)
        assert filename == SEMANTIC_REVIEW_TEMPLATE_FILENAME
        ensure_semantic_review()
        return str(template_path)

    monkeypatch.setattr(export_module, "export_raw_rows", fake_export)
    hub_module = ModuleType("huggingface_hub")
    hub_module.hf_hub_download = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)

    _stage_paired_rows(
        config_path,
        root_rows_path,
        None,
        mode="full",
        max_rows=60_000,
    )

    assert calls == [("he", revision, 42, 0)]
    paired_path = tmp_path / "rows.en.he.jsonl"
    paired = json.loads(paired_path.read_text(encoding="utf-8").splitlines()[0])
    assert paired["source_revision"] == revision
    assert (tmp_path / "translation_summary.he.json").exists()
    assert (tmp_path / "translation_publication.he.json").exists()
    assert (tmp_path / "translation_semantic_review.he.json").exists()
    assert (tmp_path / "translation_semantic_review_template.he.json").exists()
    assert (tmp_path / "translation_config.he.yaml").read_bytes() == (
        translation_config_path.read_bytes()
    )
    assert (tmp_path / "translation_run_identity.he.json").read_bytes() == (
        identity_path.read_bytes()
    )
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    assert publication["semantic_review"]["review"]["sha256"] == sha256_file(review_path)
    assert publication["semantic_review"]["machine_template"]["sha256"] == sha256_file(
        template_path
    )


def test_full_pipeline_rejects_translation_staging_override(tmp_path: Path) -> None:
    config_path = _full_config(tmp_path, paired_revision="d" * 40)
    root_rows_path = tmp_path / "rows.en.jsonl"
    root_rows_path.write_text("", encoding="utf-8")

    with pytest.raises(UserInputError, match="cannot stage rows"):
        _stage_paired_rows(
            config_path,
            root_rows_path,
            "diagnostic-translation",
            mode="full",
            max_rows=60_000,
        )


def test_full_pipeline_rejects_mutable_paired_dataset_revision(tmp_path: Path) -> None:
    config_path = _full_config(tmp_path, paired_revision="main")
    root_rows_path = tmp_path / "rows.en.jsonl"
    root_rows_path.write_text("", encoding="utf-8")

    with pytest.raises(UserInputError, match="immutable revision"):
        _stage_paired_rows(
            config_path,
            root_rows_path,
            None,
            mode="full",
            max_rows=60_000,
        )
