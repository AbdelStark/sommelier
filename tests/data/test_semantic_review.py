from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

import sommelier.data.openai_evidence as openai_evidence_module
from sommelier.artifacts import sha256_file
from sommelier.config import SommelierConfig, load_config, write_resolved_config
from sommelier.data.load import load_raw_rows
from sommelier.data.openai_evidence import OPENAI_PROVIDER_JOURNAL_FILENAME
from sommelier.data.openai_pricing import openai_list_price_ceiling_runtime_summary
from sommelier.data.openai_translate import (
    OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
    OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA,
    OPENAI_RESPONSES_SAFETY_IDENTIFIER,
    OPENAI_RESPONSES_SDK_MAX_RETRIES,
    OPENAI_RESPONSES_TIMEOUT_SECONDS,
    openai_flex_resource_unavailable_retry_policy,
)
from sommelier.data.semantic_review import (
    BACK_TRANSLATOR_ATTRIBUTION,
    BACK_TRANSLATOR_BATCH_SIZE,
    BACK_TRANSLATOR_DTYPE,
    BACK_TRANSLATOR_LICENSE,
    BACK_TRANSLATOR_MAX_SOURCE_TOKENS,
    BACK_TRANSLATOR_MODEL_ID,
    BACK_TRANSLATOR_MODEL_REVISION,
    BACKTRANSLATION_BACKEND_SCHEMA,
    BACKTRANSLATION_REQUEST_SCHEMA,
    EXPECTED_PRODUCER_PACKAGE_VERSIONS,
    NON_NATIVE_REVIEWER_BOUNDARY,
    REVIEWED_PUBLICATION_MANIFEST_FILENAME,
    SEMANTIC_REVIEW_FILENAME,
    SEMANTIC_REVIEW_SAMPLE_SIZE,
    SEMANTIC_REVIEW_TEMPLATE_FILENAME,
    BackTranslatorInfo,
    SemanticReviewProducerProvenance,
    create_semantic_review_attestation,
    create_semantic_review_template,
    finalize_semantic_review,
    root_split_assignments,
    validate_back_translator_info,
    validate_semantic_review,
)
from sommelier.data.translate import (
    DROP_REASONS,
    HEBREW_V3_FORWARD_TRANSLATOR_INTERFACE,
    HEBREW_V3_FORWARD_TRANSLATOR_MAX_MODEL_LEN,
    HEBREW_V3_FORWARD_TRANSLATOR_MAX_NEW_TOKENS,
    HEBREW_V3_FORWARD_TRANSLATOR_MODEL_ID,
    HEBREW_V3_FORWARD_TRANSLATOR_MODEL_REVISION,
    HEBREW_V3_FORWARD_TRANSLATOR_OUTPUT_DECODER,
    HEBREW_V3_FORWARD_TRANSLATOR_TRUST_REMOTE_CODE,
    HEBREW_V3_TRANSLATION_CHUNK_SIZE,
    HEBREW_V3_TRANSLATION_MAX_ATTEMPTS,
    HEBREW_V3_TRANSLATION_MAX_ROWS,
    HEBREW_V3_TRANSLATION_PROVIDER_MAX_WORKERS,
    HEBREW_V3_TRANSLATION_PROVIDER_SDK_VERSION,
    HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
    HEBREW_V3_TRANSLATION_RUNTIME_BACKEND,
    PUBLICATION_CANONICAL_FIELDS,
    PUBLICATION_MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    TRANSLATION_CONFIG_FILENAME,
    TRANSLATION_RUN_IDENTITY_FILENAME,
    TRANSLATION_RUN_IDENTITY_SCHEMA,
    TranslationStagingContract,
    TranslatorInfo,
    published_rows_canonical_identity,
    rows_filename,
    translation_provenance_sidecar_path,
    translation_selection_contract_sha256,
    translator_request_sha256,
    validate_full_paired_input_contract,
    validate_translation_publication,
    validate_translation_selection_provenance,
    write_translation_outputs,
    write_translation_publication_manifest,
)
from sommelier.data.types import RawToolCallRow, SplitName
from sommelier.errors import UserInputError
from sommelier.hebrew_v3_preregistration import (
    reviewer_anchor_payload,
    reviewer_anchor_sha256,
)
from sommelier.publication import prepare_hebrew_dataset_publication
from sommelier.remote.images import OPENAI_TRANSLATION_RUNTIME_VERSIONS
from sommelier.reviewer import ReviewerRequirement, validated_reviewer_requirement

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


class StubBacktranslator:
    def translate_batch(self, texts: list[str]) -> list[str]:
        return [f"English backtranslation {index}: {text}" for index, text in enumerate(texts)]


def _reviewer_requirement(tmp_path: Path) -> tuple[ReviewerRequirement, Path]:
    private_key = tmp_path / "reviewer-test-key"
    if not private_key.exists():
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
    requirement = validated_reviewer_requirement(
        "fixture-reviewer",
        private_key.with_suffix(".pub").read_text(encoding="ascii"),
    )
    return requirement, private_key


def _producer(
    **changes: object,
) -> SemanticReviewProducerProvenance:
    values: dict[str, object] = {
        "code_revision": "a" * 40,
        "working_tree_clean": True,
        "execution_boundary": "modal_gpu",
        "provider": "modal",
        "hardware": "A10G",
        "allocation_timeout_seconds": 14_400,
        "package_versions": dict(EXPECTED_PRODUCER_PACKAGE_VERSIONS),
    }
    values.update(changes)
    return SemanticReviewProducerProvenance(**values)  # type: ignore[arg-type]


def _write_rows(tmp_path: Path, count: int = 210) -> tuple[Path, Path, dict[str, SplitName]]:
    root_path = tmp_path / "rows.en.jsonl"
    paired_path = tmp_path / "rows.he.jsonl"
    split_by_id: dict[str, SplitName] = {}
    root_records: list[RawToolCallRow] = []
    paired_records: list[RawToolCallRow] = []
    splits: tuple[SplitName, ...] = ("train", "validation", "test")
    verbs = ("draw", "book", "find", "list", "calculate")
    for index in range(count):
        source_id = f"root-{index:04d}"
        verb = verbs[index % len(verbs)]
        tool_name = f"{verb}_items_{index % 7}"
        query = f"Please {verb} {index} items for account user-{index % 11}"
        tools = json.dumps(
            [
                {
                    "name": tool_name,
                    "description": "Perform the action",
                    "parameters": {"type": "object"},
                }
            ]
        )
        answers = json.dumps(
            [
                {
                    "name": tool_name,
                    "arguments": {"count": index, "account": f"user-{index % 11}"},
                }
            ]
        )
        root = RawToolCallRow(
            schema_version="sommelier.raw_tool_call_row.v1",
            source_id=source_id,
            query=query,
            tools=tools,
            answers=answers,
            source_revision="root-revision",
        )
        paired = RawToolCallRow(
            schema_version="sommelier.raw_tool_call_row.v1",
            source_id=f"{source_id}:he",
            query=f"בקשה בעברית מספר {index} עבור user-{index % 11}",
            tools=tools,
            answers=answers,
            source_revision="root-revision",
            source_example_id=source_id,
        )
        root_records.append(root)
        paired_records.append(paired)
        split_by_id[source_id] = splits[index % len(splits)]
    root_path.write_text(
        "".join(json.dumps(dict(row)) + "\n" for row in root_records),
        encoding="utf-8",
    )
    paired_path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in paired_records),
        encoding="utf-8",
    )
    return root_path, paired_path, split_by_id


def _write_summary(tmp_path: Path) -> Path:
    path = tmp_path / "translation_summary.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "sommelier.translation_summary.v2",
                "language": "he",
                "selection": {"mode": "full"},
                "source_code": {
                    "git_commit": "a" * 40,
                    "working_tree_clean": True,
                },
                "translator": {
                    "model_id": "dicta-il/DictaLM-3.0-Nemotron-12B-Instruct",
                    "model_revision": "d" * 40,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _create(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, SplitName]]:
    root, paired, splits = _write_rows(tmp_path)
    summary = _write_summary(tmp_path)
    review = tmp_path / "translation_semantic_review.json"
    reviewer_requirement, _private_key = _reviewer_requirement(tmp_path)
    create_semantic_review_template(
        root_rows_path=root,
        paired_rows_path=paired,
        translation_summary_path=summary,
        root_split_by_id=splits,
        output_path=review,
        backtranslator=StubBacktranslator(),
        seed=42,
        producer_provenance=_producer(),
        reviewer_requirement=reviewer_requirement,
    )
    return review, root, paired, splits


def _complete_pass_decisions(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for record in payload["records"]:
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
    path.write_text(json.dumps(payload), encoding="utf-8")


def _finalize_signed(
    reviewed: Path,
    output: Path,
    *,
    template: Path,
    root: Path,
    paired: Path,
    summary: Path,
    splits: dict[str, SplitName],
    seed: int = 42,
) -> Path:
    attestation, signature = _create_signed_attestation(
        reviewed,
        output,
        template=template,
        root=root,
        paired=paired,
        summary=summary,
        splits=splits,
        seed=seed,
    )
    return finalize_semantic_review(
        reviewed,
        output,
        template_path=template,
        attestation_path=attestation,
        signature_path=signature,
        root_rows_path=root,
        paired_rows_path=paired,
        translation_summary_path=summary,
        root_split_by_id=splits,
        expected_seed=seed,
        backtranslator=StubBacktranslator(),
    )


def _create_signed_attestation(
    reviewed: Path,
    output: Path,
    *,
    template: Path,
    root: Path,
    paired: Path,
    summary: Path,
    splits: dict[str, SplitName],
    seed: int = 42,
    private_key: Path | None = None,
) -> tuple[Path, Path]:
    _requirement, registered_private_key = _reviewer_requirement(template.parent)
    signing_key = private_key or registered_private_key
    attestation = output.with_name(f"{output.stem}.attestation.json")
    create_semantic_review_attestation(
        reviewed,
        attestation,
        template_path=template,
        root_rows_path=root,
        paired_rows_path=paired,
        translation_summary_path=summary,
        root_split_by_id=splits,
        expected_seed=seed,
        backtranslator=StubBacktranslator(),
    )
    return attestation, _sign_attestation(attestation, signing_key)


def _sign_attestation(attestation: Path, private_key: Path) -> Path:
    subprocess.run(
        [
            "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            str(private_key),
            "-n",
            "sommelier-hebrew-v3-semantic-review",
            str(attestation),
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )
    return Path(f"{attestation}.sig")


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _locked_record(record: dict[str, object]) -> dict[str, object]:
    fields = (
        "sample_id",
        "source_example_id",
        "paired_row_sha256",
        "source_row_sha256",
        "source_query",
        "hebrew_query",
        "backtranslation_request_sha256",
        "english_backtranslation",
        "english_backtranslation_sha256",
        "strata",
    )
    return {field: record[field] for field in fields}


def _clean_openai_provider_evidence(
    tmp_path: Path,
    *,
    request_count: int,
) -> dict[str, object]:
    aggregate = {
        "schema_version": OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA,
        "journal_schema_version": OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
        "journal_sha256": "e" * 64,
        "requested_model": HEBREW_V3_FORWARD_TRANSLATOR_MODEL_ID,
        "returned_models": [HEBREW_V3_FORWARD_TRANSLATOR_MODEL_ID],
        "requested_service_tier": HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
        "requested_timeout_seconds": OPENAI_RESPONSES_TIMEOUT_SECONDS,
        "returned_service_tiers": [HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER],
        "safety_identifier": OPENAI_RESPONSES_SAFETY_IDENTIFIER,
        "client_injected": False,
        "sdk_max_retries": OPENAI_RESPONSES_SDK_MAX_RETRIES,
        "resource_unavailable_retry_policy": (openai_flex_resource_unavailable_retry_policy()),
        "max_canonical_request_body_utf8_bytes": 1024,
        "max_response_input_tokens": 1_000,
        "unique_requests": request_count,
        "unique_source_attempts": request_count,
        "usage_complete": True,
        "counts": {
            "records": request_count,
            "responses": request_count,
            "replayable_responses": request_count,
            "replays": 0,
            "durable_journal_replays": 0,
            "batch_coalesced_replays": 0,
            "request_errors": 0,
            "resource_unavailable_events": 0,
            "resolved_resource_unavailable_events": 0,
            "pending_resource_unavailable_events": 0,
            "unresolved_resource_unavailable_events": 0,
            "provider_error_responses": 0,
            "error_records": 0,
            "model_mismatch_responses": 0,
            "service_tier_mismatch_responses": 0,
            "refusal_responses": 0,
            "incomplete_responses": 0,
            "responses_missing_usage": 0,
        },
        "usage": {
            "input_tokens": 1_000,
            "cached_input_tokens": 200,
            "output_tokens": 100,
            "reasoning_output_tokens": 25,
            "total_tokens": 1_100,
        },
    }
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            openai_evidence_module,
            "aggregate_openai_responses_provider_journal",
            lambda _path: aggregate,
        )
        return openai_evidence_module.build_openai_provider_evidence(
            tmp_path / OPENAI_PROVIDER_JOURNAL_FILENAME,
            HEBREW_V3_FORWARD_TRANSLATOR_MODEL_ID,
            HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
        )


_FULL_HEBREW_BUNDLE_FILENAMES = (
    "rows.en.jsonl",
    "rows.en.he.jsonl",
    "translation_config.he.yaml",
    "translation_publication.he.json",
    "translation_run_identity.he.json",
    "translation_semantic_review.he.json",
    "translation_semantic_review_template.he.json",
    "translation_summary.he.json",
)


def _load_full_hebrew_contract_bundle(
    root: Path,
) -> tuple[SommelierConfig, Path, dict[str, Path]]:
    phase_a_config = load_config(root / "translation_config.he.yaml")
    config = phase_a_config.model_copy(deep=True)
    config.datasets[1].dataset_revision = "d" * 40
    rows = root / "rows.en.jsonl"
    return (
        config,
        rows,
        {
            "paired_rows": root / "rows.en.he.jsonl",
            "translation_summary": root / "translation_summary.he.json",
            "translation_publication": root / "translation_publication.he.json",
            "semantic_review_template": root / "translation_semantic_review_template.he.json",
            "semantic_review": root / "translation_semantic_review.he.json",
            "translation_config": root / "translation_config.he.yaml",
            "translation_run_identity": root / "translation_run_identity.he.json",
        },
    )


def _write_self_consistent_translation_identity_tamper(
    paths: dict[str, Path],
    *,
    summary: dict[str, object],
    identity: dict[str, object],
) -> None:
    identity_path = paths["translation_run_identity"]
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["translation_run_identity_sha256"] = sha256_file(identity_path)
    paths["translation_summary"].write_text(json.dumps(summary), encoding="utf-8")


def _full_hebrew_contract_bundle(
    tmp_path: Path,
) -> tuple[SommelierConfig, Path, dict[str, Path]]:
    # The exact preregistered cohort is intentionally 17,000 rows and building
    # its signed semantic evidence is expensive. Cache one immutable base per
    # pytest worker, then copy regular files so mutation tests remain isolated.
    cache_root = tmp_path.parent / "_full_hebrew_contract_bundle_v4"
    cache_marker = cache_root / ".complete"
    if cache_marker.is_file():
        for filename in _FULL_HEBREW_BUNDLE_FILENAMES:
            shutil.copy2(cache_root / filename, tmp_path / filename)
        shutil.copytree(cache_root / ".git", tmp_path / ".git")
        return _load_full_hebrew_contract_bundle(tmp_path)

    reviewer_requirement, _private_key = _reviewer_requirement(tmp_path)
    phase_a_payload = load_config(EXAMPLES_DIR / "config.v3-he-full.yaml").model_dump(mode="json")
    phase_a_payload["semantic_review"] = {
        "reviewer": {
            "reviewer_id": reviewer_requirement.reviewer_id,
            "ssh_public_key": reviewer_requirement.ssh_public_key,
            "public_key_fingerprint": reviewer_requirement.public_key_fingerprint,
        }
    }
    phase_a_config = SommelierConfig.model_validate(phase_a_payload)
    resolved_phase_a, _phase_a_digest = write_resolved_config(
        phase_a_config,
        tmp_path / "phase-a-config",
    )
    translation_config = tmp_path / "translation_config.he.yaml"
    shutil.copy2(resolved_phase_a, translation_config)
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "add", translation_config.name], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "Phase A reviewer config"],
        cwd=tmp_path,
        check=True,
    )
    implementation_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Reload the exact bytes published with the translation, then construct the
    # consuming Phase-B config by changing only the now-published Hebrew commit.
    phase_a_config = load_config(translation_config)
    config = phase_a_config.model_copy(deep=True)
    config.datasets[1].dataset_revision = "d" * 40
    root, initial_paired, _fixture_splits = _write_rows(tmp_path, count=17_000)
    paired = tmp_path / "rows.en.he.jsonl"
    initial_paired.replace(paired)
    splits = root_split_assignments(config, load_raw_rows(root))
    ordered_ids = [f"root-{index:04d}" for index in range(17_000)]
    translator_info = TranslatorInfo(
        model_id=HEBREW_V3_FORWARD_TRANSLATOR_MODEL_ID,
        model_revision=HEBREW_V3_FORWARD_TRANSLATOR_MODEL_REVISION,
        max_new_tokens=HEBREW_V3_FORWARD_TRANSLATOR_MAX_NEW_TOKENS,
        interface=HEBREW_V3_FORWARD_TRANSLATOR_INTERFACE,
        max_model_len=HEBREW_V3_FORWARD_TRANSLATOR_MAX_MODEL_LEN,
        trust_remote_code=HEBREW_V3_FORWARD_TRANSLATOR_TRUST_REMOTE_CODE,
        output_decoder=HEBREW_V3_FORWARD_TRANSLATOR_OUTPUT_DECODER,
        implementation_revision=implementation_revision,
        runtime_backend=HEBREW_V3_TRANSLATION_RUNTIME_BACKEND,
        provider_service_tier=HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
        provider_sdk_version=HEBREW_V3_TRANSLATION_PROVIDER_SDK_VERSION,
        provider_timeout_seconds=OPENAI_RESPONSES_TIMEOUT_SECONDS,
    )
    provider_evidence = _clean_openai_provider_evidence(
        tmp_path,
        request_count=len(ordered_ids),
    )
    selection_contract = translation_selection_contract_sha256(
        phase_a_config,
        mode="full",
        max_rows=HEBREW_V3_TRANSLATION_MAX_ROWS,
        limit=0,
    )
    reviewer_preregistration = reviewer_anchor_payload(phase_a_config)
    reviewer_preregistration_digest = reviewer_anchor_sha256(phase_a_config)
    translation_run_identity = translation_provenance_sidecar_path(
        root,
        TRANSLATION_RUN_IDENTITY_FILENAME,
        "he",
    )
    translation_run_identity.write_text(
        json.dumps(
            {
                "schema_version": TRANSLATION_RUN_IDENTITY_SCHEMA,
                "run_id": "fixture-hebrew-v3-full",
                "config_sha256": sha256_file(translation_config),
                "selection": {
                    "contract_sha256": selection_contract,
                    "mode": "full",
                    "max_rows": HEBREW_V3_TRANSLATION_MAX_ROWS,
                    "limit": 0,
                    "seed": phase_a_config.project.seed,
                },
                "translator": {
                    "model_id": translator_info.model_id,
                    "model_revision": translator_info.model_revision,
                    "request_sha256": translator_request_sha256(translator_info, "he"),
                    "max_attempts": HEBREW_V3_TRANSLATION_MAX_ATTEMPTS,
                    "implementation_revision": implementation_revision,
                },
                "runtime": {
                    "backend": HEBREW_V3_TRANSLATION_RUNTIME_BACKEND,
                    "translation_chunk_size": HEBREW_V3_TRANSLATION_CHUNK_SIZE,
                    "allocation_gpu": None,
                    "function_timeout_seconds": 3_600,
                    "provider_service_tier": HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
                    "provider_sdk_version": HEBREW_V3_TRANSLATION_PROVIDER_SDK_VERSION,
                    "provider_timeout_seconds": OPENAI_RESPONSES_TIMEOUT_SECONDS,
                    "provider_max_workers": HEBREW_V3_TRANSLATION_PROVIDER_MAX_WORKERS,
                    "openai_list_price_limit_usd": "50.00",
                },
                "source_code": {
                    "git_commit": implementation_revision,
                    "working_tree_clean": True,
                    "boundary": "Synthetic fixture for the exact committed producer boundary.",
                },
                "reviewer_preregistration": reviewer_preregistration,
                "reviewer_preregistration_sha256": reviewer_preregistration_digest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated-openai-translation"
    generated_rows, generated_summary = write_translation_outputs(
        generated_dir,
        load_raw_rows(paired, require_source_example_id=True),
        {
            "input_rows": len(ordered_ids),
            "translated_rows": len(ordered_ids),
            "max_attempts": HEBREW_V3_TRANSLATION_MAX_ATTEMPTS,
            "translation_attempts": len(ordered_ids),
            "retried_rows": 0,
            "dropped": {reason: 0 for reason in DROP_REASONS},
            "translation_identity_sha256": hashlib.sha256(
                b"synthetic exact-provider translation identity"
            ).hexdigest(),
            "environment": dict(OPENAI_TRANSLATION_RUNTIME_VERSIONS),
            "runtime": {
                "gpu": None,
                "backend": HEBREW_V3_TRANSLATION_RUNTIME_BACKEND,
                "provider": "openai",
                "execution_provider": "modal",
                "gpu_allocation_label": None,
                "function_timeout_seconds": 3_600,
                "model_load_seconds": 0.0,
                "translation_seconds": 1.0,
                "provider_service_tier": HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
                "provider_timeout_seconds": OPENAI_RESPONSES_TIMEOUT_SECONDS,
                "provider_max_workers": HEBREW_V3_TRANSLATION_PROVIDER_MAX_WORKERS,
                "provider_journal_filename": OPENAI_PROVIDER_JOURNAL_FILENAME,
                "translation_chunk_size": HEBREW_V3_TRANSLATION_CHUNK_SIZE,
                "boundary": "Synthetic fixture for the exact remote producer boundary.",
                "openai_list_price_ceiling": openai_list_price_ceiling_runtime_summary(
                    Decimal("50.00"),
                    service_tier=HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
                ),
                "credential_source": "Synthetic fixture; no provider credentials used.",
            },
            "provider_evidence": provider_evidence,
            "selection": {
                "contract_sha256": selection_contract,
                "config_sha256": sha256_file(translation_config),
                "mode": "full",
                "seed": config.project.seed,
                "max_rows": HEBREW_V3_TRANSLATION_MAX_ROWS,
                "limit": 0,
                "selected_rows": len(ordered_ids),
                "selected_source_ids_sha256": hashlib.sha256(
                    "\n".join(ordered_ids).encode()
                ).hexdigest(),
            },
            "source_code": {
                "git_commit": implementation_revision,
                "working_tree_clean": True,
                "boundary": "Synthetic fixture for the exact committed producer boundary.",
            },
            "reviewer_preregistration": reviewer_preregistration,
            "reviewer_preregistration_sha256": reviewer_preregistration_digest,
            "translation_run_identity_sha256": sha256_file(translation_run_identity),
        },
        translator=translator_info,
        input_description="synthetic exact-provider semantic-review fixture",
        target_language="he",
        input_sha256=sha256_file(root),
    )
    shutil.copy2(generated_rows, paired)
    summary = tmp_path / "translation_summary.he.json"
    shutil.copy2(generated_summary, summary)
    template = tmp_path / "translation_semantic_review_template.he.json"
    create_semantic_review_template(
        root_rows_path=root,
        paired_rows_path=paired,
        translation_summary_path=summary,
        root_split_by_id=splits,
        output_path=template,
        backtranslator=StubBacktranslator(),
        seed=config.project.seed,
        producer_provenance=_producer(code_revision=implementation_revision),
        reviewer_requirement=reviewer_requirement,
    )
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    review = tmp_path / "translation_semantic_review.he.json"
    _finalize_signed(
        reviewed,
        review,
        template=template,
        root=root,
        paired=paired,
        summary=summary,
        splits=splits,
        seed=config.project.seed,
    )
    publication = tmp_path / "translation_publication.he.json"
    write_translation_publication_manifest(
        publication,
        translated_rows_path=paired,
        summary_path=summary,
        target_language="he",
        semantic_review_path=review,
        semantic_review_template_path=template,
    )
    cache_root.mkdir()
    for filename in _FULL_HEBREW_BUNDLE_FILENAMES:
        shutil.copy2(tmp_path / filename, cache_root / filename)
    shutil.copytree(tmp_path / ".git", cache_root / ".git")
    cache_marker.write_text("complete\n", encoding="ascii")
    return _load_full_hebrew_contract_bundle(tmp_path)


@pytest.mark.parametrize(
    "reviewer_id",
    ["", " reviewer", "reviewer ", "reviewer name", "../reviewer", "reviewer/one"],
)
def test_reviewer_requirement_rejects_unsafe_identity(
    tmp_path: Path,
    reviewer_id: str,
) -> None:
    _requirement, private_key = _reviewer_requirement(tmp_path)
    with pytest.raises(UserInputError, match="safe stable reviewer id"):
        validated_reviewer_requirement(
            reviewer_id,
            private_key.with_suffix(".pub").read_text(encoding="ascii"),
        )


@pytest.mark.parametrize(
    "public_key",
    [
        "",
        "ssh-rsa AAAA",
        "ssh-ed25519 not-base64",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
    ],
)
def test_reviewer_requirement_rejects_invalid_or_private_key(public_key: str) -> None:
    with pytest.raises(UserInputError, match="OpenSSH Ed25519|valid base64|wire format"):
        validated_reviewer_requirement("fixture-reviewer", public_key)


def test_template_rejects_duplicate_translation_summary_keys(tmp_path: Path) -> None:
    root, paired, splits = _write_rows(tmp_path)
    summary = _write_summary(tmp_path)
    text = summary.read_text(encoding="utf-8")
    summary.write_text(
        text.replace(
            '"schema_version":',
            '"schema_version": "duplicate", "schema_version":',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(UserInputError, match="valid translation summary"):
        create_semantic_review_template(
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=summary,
            root_split_by_id=splits,
            output_path=tmp_path / "review.json",
            backtranslator=StubBacktranslator(),
            seed=42,
            producer_provenance=_producer(),
            reviewer_requirement=_reviewer_requirement(tmp_path)[0],
        )


def test_template_preregisters_exact_stratified_sample_and_model(tmp_path: Path) -> None:
    review, root, paired, splits = _create(tmp_path)
    payload = json.loads(review.read_text(encoding="utf-8"))

    assert len(payload["records"]) == SEMANTIC_REVIEW_SAMPLE_SIZE
    assert len(set(payload["selection"]["ordered_sample_ids"])) == 200
    assert payload["selection"]["selected_before_judgments"] is True
    assert payload["selection"]["stratification_dimensions"] == [
        "root_split",
        "source_query_length_decile",
        "protected_span_count",
        "tool_action_family",
        "ambiguous_high_risk_action_verb",
    ]
    assert payload["selection"]["sample_strata"]["root_split"].keys() == {
        "train",
        "validation",
        "test",
    }
    assert payload["selection"]["sample_strata"]["ambiguous_high_risk_action_verb"]["true"] >= 40
    assert payload["back_translator"]["model_id"] == BACK_TRANSLATOR_MODEL_ID
    assert payload["back_translator"] == {
        "model_id": BACK_TRANSLATOR_MODEL_ID,
        "model_revision": BACK_TRANSLATOR_MODEL_REVISION,
        "license": "cc-by-4.0",
        "attribution": "Helsinki-NLP, OPUS-MT project",
        "model_card": "https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-he-en",
        "source_language": "he",
        "target_language": "en",
        "request_schema": "sommelier.marian_backtranslation_request.v1",
        "backend": {
            "schema_version": "sommelier.transformers_marian_backtranslator.v1",
            "framework": "transformers",
            "model_loader": "AutoModelForSeq2SeqLM",
            "tokenizer_loader": "AutoTokenizer",
            "model_type": "marian",
            "dtype": "float16",
            "device_map": "auto",
            "trust_remote_code": False,
            "hugging_face_environment": {
                "HF_HUB_DISABLE_XET": "1",
                "HF_HUB_DOWNLOAD_TIMEOUT": "600",
            },
        },
        "tokenization": {
            "add_special_tokens": True,
            "padding": "longest",
            "truncation": False,
            "max_source_tokens": 512,
            "use_fast": False,
            "punctuation_normalizer": "sacremoses.MosesPunctNormalizer",
        },
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 512,
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        },
        "batch_size": 8,
    }
    assert payload["producer"]["runtime"]["allocation_timeout_seconds"] == 14_400
    assert payload["reviewer"] == {
        "status": "pending",
        "attestation": None,
        "signature": None,
    }
    assert payload["reviewer_requirement"]["reviewer_id"] == "fixture-reviewer"
    assert payload["reviewer_requirement"]["native_hebrew_reviewer"] is False
    assert payload["reviewer_requirement"]["boundary"] == NON_NATIVE_REVIEWER_BOUNDARY
    assert payload["reviewer_requirement"]["public_key_fingerprint"].startswith("SHA256:")

    original = review.read_bytes()
    create_semantic_review_template(
        root_rows_path=root,
        paired_rows_path=paired,
        translation_summary_path=_write_summary(tmp_path),
        root_split_by_id=splits,
        output_path=review,
        backtranslator=StubBacktranslator(),
        seed=42,
        producer_provenance=_producer(),
        reviewer_requirement=_reviewer_requirement(tmp_path)[0],
    )
    assert review.read_bytes() == original

    # Even the local helper preserves an occupied path; release-valid evidence
    # never relies on write_text replacement semantics.
    second = tmp_path / "second.json"
    second.write_text("disposable fixture placeholder\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="missing or invalid"):
        create_semantic_review_template(
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            output_path=second,
            backtranslator=StubBacktranslator(),
            seed=42,
            producer_provenance=_producer(),
            reviewer_requirement=_reviewer_requirement(tmp_path)[0],
        )
    assert second.read_text(encoding="utf-8") == "disposable fixture placeholder\n"

    victim = tmp_path / "template-victim.json"
    victim.write_text("preserve\n", encoding="utf-8")
    symlink = tmp_path / "template-symlink.json"
    symlink.symlink_to(victim)
    with pytest.raises(UserInputError, match="unsafe path"):
        create_semantic_review_template(
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            output_path=symlink,
            backtranslator=StubBacktranslator(),
            seed=42,
            producer_provenance=_producer(),
            reviewer_requirement=_reviewer_requirement(tmp_path)[0],
        )
    assert victim.read_text(encoding="utf-8") == "preserve\n"


def test_template_rejects_less_than_200_accepted_rows(tmp_path: Path) -> None:
    root, paired, splits = _write_rows(tmp_path, count=199)
    with pytest.raises(UserInputError, match="at least 200"):
        create_semantic_review_template(
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            output_path=tmp_path / "review.json",
            backtranslator=StubBacktranslator(),
            seed=42,
            producer_provenance=_producer(),
            reviewer_requirement=_reviewer_requirement(tmp_path)[0],
        )


def test_template_rejects_mutable_or_substituted_backtranslator(tmp_path: Path) -> None:
    root, paired, splits = _write_rows(tmp_path)
    with pytest.raises(UserInputError, match="mutable or not preregistered"):
        create_semantic_review_template(
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            output_path=tmp_path / "review.json",
            backtranslator=StubBacktranslator(),
            seed=42,
            producer_provenance=_producer(),
            reviewer_requirement=_reviewer_requirement(tmp_path)[0],
            back_translator_info=BackTranslatorInfo(model_revision="main"),
        )


def test_backtranslator_contract_is_pinned_independent_marian() -> None:
    info = BackTranslatorInfo()

    assert info == BackTranslatorInfo(
        model_id="Helsinki-NLP/opus-mt-tc-big-he-en",
        model_revision="134c5a850dcaa763eec85bd1f4eb25112fecedbb",
        max_new_tokens=512,
        max_source_tokens=512,
        batch_size=8,
    )
    assert BACK_TRANSLATOR_MODEL_ID == info.model_id
    assert BACK_TRANSLATOR_MODEL_REVISION == info.model_revision
    assert BACK_TRANSLATOR_MAX_SOURCE_TOKENS == 512
    assert BACK_TRANSLATOR_BATCH_SIZE == 8
    assert BACK_TRANSLATOR_LICENSE == "cc-by-4.0"
    assert BACK_TRANSLATOR_ATTRIBUTION == "Helsinki-NLP, OPUS-MT project"
    assert BACK_TRANSLATOR_DTYPE == "float16"
    assert BACKTRANSLATION_REQUEST_SCHEMA.endswith(".v1")
    assert BACKTRANSLATION_BACKEND_SCHEMA.endswith(".v1")
    validate_back_translator_info(
        info,
        forward_model_id="google/madlad400-3b-mt",
    )
    with pytest.raises(UserInputError, match="must differ from the forward translator"):
        validate_back_translator_info(info, forward_model_id=info.model_id)


@pytest.mark.parametrize(
    ("producer", "message"),
    [
        (_producer(package_versions={}), "package versions"),
        (_producer(working_tree_clean=False), "worktree is dirty"),
        (_producer(code_revision="main"), "revision is mutable"),
        (
            _producer(
                package_versions={
                    **EXPECTED_PRODUCER_PACKAGE_VERSIONS,
                    "transformers": "0.0.0",
                }
            ),
            "package versions",
        ),
        (_producer(provider="other-cloud"), "Modal allocation identity"),
    ],
)
def test_template_rejects_unreproducible_producer_provenance(
    tmp_path: Path,
    producer: SemanticReviewProducerProvenance,
    message: str,
) -> None:
    root, paired, splits = _write_rows(tmp_path)
    with pytest.raises(UserInputError, match=message):
        create_semantic_review_template(
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            output_path=tmp_path / "review.json",
            backtranslator=StubBacktranslator(),
            seed=42,
            producer_provenance=producer,
            reviewer_requirement=_reviewer_requirement(tmp_path)[0],
        )


@pytest.mark.parametrize("tamper", ["reviewer", "decision", "gate"])
def test_machine_template_must_remain_pristine(tmp_path: Path, tamper: str) -> None:
    template, root, paired, splits = _create(tmp_path)
    payload = json.loads(template.read_text(encoding="utf-8"))
    if tamper == "reviewer":
        payload["reviewer"]["reviewer_id"] = "assigned-too-early"
    elif tamper == "decision":
        payload["records"][0]["review"]["rubric"]["action_tool_intent"] = "pass"
    else:
        payload["gate"]["complete_decisions"] = 1
    template.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match="machine semantic-review template"):
        validate_semantic_review(
            template,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            require_passed=False,
        )


def test_finalizer_rejects_same_path_and_hardlink_aliases(tmp_path: Path) -> None:
    template, root, paired, splits = _create(tmp_path)
    final = tmp_path / "final.json"
    with pytest.raises(UserInputError, match="distinct files"):
        finalize_semantic_review(
            template,
            final,
            template_path=template,
            attestation_path=tmp_path / "attestation.json",
            signature_path=tmp_path / "attestation.sig",
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
        )

    reviewed_hardlink = tmp_path / "reviewed-hardlink.json"
    reviewed_hardlink.hardlink_to(template)
    with pytest.raises(UserInputError, match="must not alias"):
        finalize_semantic_review(
            reviewed_hardlink,
            final,
            template_path=template,
            attestation_path=tmp_path / "attestation.json",
            signature_path=tmp_path / "attestation.sig",
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("sample", "deterministic preregistration"),
        ("backtranslation", "tampered locked inputs"),
    ],
)
def test_validation_rejects_tampered_sample_or_backtranslation(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    review, root, paired, splits = _create(tmp_path)
    payload = json.loads(review.read_text(encoding="utf-8"))
    if tamper == "sample":
        payload["selection"]["ordered_sample_ids"][0] = "replacement-row"
    else:
        payload["records"][0]["english_backtranslation"] = "tampered"
    review.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match=message):
        validate_semantic_review(
            review,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            require_passed=False,
        )


def test_validation_rejects_paired_rows_changed_after_sampling(tmp_path: Path) -> None:
    review, root, paired, splits = _create(tmp_path)
    lines = paired.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["query"] += " שינוי"
    lines[0] = json.dumps(row, ensure_ascii=False)
    paired.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(UserInputError, match="full canonical paired-row digest"):
        validate_semantic_review(
            review,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            require_passed=False,
        )


def test_hub_rewritten_source_identity_preserves_review_contract(tmp_path: Path) -> None:
    template, root, paired, splits = _create(tmp_path)
    rewritten: list[str] = []
    for index, line in enumerate(paired.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        row["source_id"] = f"published-dataset:{index}"
        row["source_revision"] = "e" * 40
        rewritten.append(json.dumps(row, ensure_ascii=False))
    paired.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    validate_semantic_review(
        template,
        root_rows_path=root,
        paired_rows_path=paired,
        translation_summary_path=_write_summary(tmp_path),
        root_split_by_id=splits,
        expected_seed=42,
        require_passed=False,
    )


def test_finalizer_rejects_rehashed_backtranslation_without_original_template(
    tmp_path: Path,
) -> None:
    template, root, paired, splits = _create(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    payload = json.loads(reviewed.read_text(encoding="utf-8"))
    record = payload["records"][0]
    record["english_backtranslation"] = "attacker replacement"
    record["english_backtranslation_sha256"] = hashlib.sha256(
        record["english_backtranslation"].encode("utf-8")
    ).hexdigest()
    record["locked_review_input_sha256"] = _sha256_json(_locked_record(record))
    payload["selection"]["locked_sample_sha256"] = _sha256_json(
        [_locked_record(item) for item in payload["records"]]
    )
    reviewed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match="machine-locked sample or backtranslation"):
        _finalize_signed(
            reviewed,
            tmp_path / "final.json",
            template=template,
            root=root,
            paired=paired,
            summary=_write_summary(tmp_path),
            splits=splits,
        )


def test_attestation_recomputes_and_rejects_self_rehashed_machine_template(
    tmp_path: Path,
) -> None:
    template, root, paired, splits = _create(tmp_path)
    payload = json.loads(template.read_text(encoding="utf-8"))
    record = payload["records"][0]
    record["english_backtranslation"] = "attacker-authored reassuring paraphrase"
    record["english_backtranslation_sha256"] = hashlib.sha256(
        record["english_backtranslation"].encode("utf-8")
    ).hexdigest()
    record["locked_review_input_sha256"] = _sha256_json(_locked_record(record))
    payload["selection"]["locked_sample_sha256"] = _sha256_json(
        [_locked_record(item) for item in payload["records"]]
    )
    template.write_text(json.dumps(payload), encoding="utf-8")
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)

    with pytest.raises(UserInputError, match="independent recomputation"):
        create_semantic_review_attestation(
            reviewed,
            tmp_path / "attestation.json",
            template_path=template,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            backtranslator=StubBacktranslator(),
        )


def test_finalize_and_validation_bind_all_reviewer_decisions(tmp_path: Path) -> None:
    template, root, paired, splits = _create(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    final = tmp_path / "final.json"
    _finalize_signed(
        reviewed,
        final,
        template=template,
        root=root,
        paired=paired,
        summary=_write_summary(tmp_path),
        splits=splits,
    )
    validate_semantic_review(
        final,
        root_rows_path=root,
        paired_rows_path=paired,
        translation_summary_path=_write_summary(tmp_path),
        root_split_by_id=splits,
        expected_seed=42,
        template_path=template,
    )

    payload = json.loads(final.read_text(encoding="utf-8"))
    payload["records"][0]["review"]["notes"] = "tampered after finalization"
    final.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UserInputError, match="publication gate did not pass"):
        validate_semantic_review(
            final,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            template_path=template,
        )


def test_final_schema_never_skips_gate_or_signature_validation(tmp_path: Path) -> None:
    template, root, paired, splits = _create(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    final = _finalize_signed(
        reviewed,
        tmp_path / "final.json",
        template=template,
        root=root,
        paired=paired,
        summary=_write_summary(tmp_path),
        splits=splits,
    )
    payload = json.loads(final.read_text(encoding="utf-8"))
    payload["gate"]["status"] = "pending"
    final.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match="publication gate did not pass"):
        validate_semantic_review(
            final,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            require_passed=False,
            template_path=template,
        )


def test_final_gate_rejects_boolean_critical_error_without_resigning(tmp_path: Path) -> None:
    template, root, paired, splits = _create(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    final = _finalize_signed(
        reviewed,
        tmp_path / "final.json",
        template=template,
        root=root,
        paired=paired,
        summary=_write_summary(tmp_path),
        splits=splits,
    )
    payload = json.loads(final.read_text(encoding="utf-8"))
    payload["gate"]["critical_errors"] = False
    final.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        UserInputError,
        match="gate critical_errors must be a non-negative JSON integer",
    ):
        validate_semantic_review(
            final,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            template_path=template,
        )


@pytest.mark.parametrize(
    "field_path",
    [
        ("critical_errors",),
        ("backtranslation_verification", "decoding", "num_beams"),
    ],
)
def test_finalizer_rejects_signed_attestation_boolean_count_fields(
    tmp_path: Path,
    field_path: tuple[str, ...],
) -> None:
    template, root, paired, splits = _create(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    final = tmp_path / "final.json"
    attestation, signature = _create_signed_attestation(
        reviewed,
        final,
        template=template,
        root=root,
        paired=paired,
        summary=_write_summary(tmp_path),
        splits=splits,
    )
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    current = payload
    for field in field_path[:-1]:
        current = current[field]
    current[field_path[-1]] = False if field_path == ("critical_errors",) else True
    attestation.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    signature.unlink()
    _requirement, private_key = _reviewer_requirement(tmp_path)
    signature = _sign_attestation(attestation, private_key)

    with pytest.raises(UserInputError, match="must be a non-negative JSON integer"):
        finalize_semantic_review(
            reviewed,
            final,
            template_path=template,
            attestation_path=attestation,
            signature_path=signature,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            backtranslator=StubBacktranslator(),
        )


def test_signed_review_retry_is_byte_idempotent(tmp_path: Path) -> None:
    template, root, paired, splits = _create(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    final = tmp_path / "final.json"
    attestation, signature = _create_signed_attestation(
        reviewed,
        final,
        template=template,
        root=root,
        paired=paired,
        summary=_write_summary(tmp_path),
        splits=splits,
    )
    attestation_bytes = attestation.read_bytes()

    create_semantic_review_attestation(
        reviewed,
        attestation,
        template_path=template,
        root_rows_path=root,
        paired_rows_path=paired,
        translation_summary_path=_write_summary(tmp_path),
        root_split_by_id=splits,
        expected_seed=42,
        backtranslator=StubBacktranslator(),
    )
    assert attestation.read_bytes() == attestation_bytes

    first = finalize_semantic_review(
        reviewed,
        final,
        template_path=template,
        attestation_path=attestation,
        signature_path=signature,
        root_rows_path=root,
        paired_rows_path=paired,
        translation_summary_path=_write_summary(tmp_path),
        root_split_by_id=splits,
        expected_seed=42,
        backtranslator=StubBacktranslator(),
    )
    first_bytes = first.read_bytes()
    second = finalize_semantic_review(
        reviewed,
        final,
        template_path=template,
        attestation_path=attestation,
        signature_path=signature,
        root_rows_path=root,
        paired_rows_path=paired,
        translation_summary_path=_write_summary(tmp_path),
        root_split_by_id=splits,
        expected_seed=42,
        backtranslator=StubBacktranslator(),
    )
    assert second.read_bytes() == first_bytes


def test_finalizer_rejects_signature_from_unregistered_key(tmp_path: Path) -> None:
    template, root, paired, splits = _create(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    wrong_key = tmp_path / "wrong-reviewer-key"
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
            str(wrong_key),
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )
    final = tmp_path / "final.json"
    attestation, signature = _create_signed_attestation(
        reviewed,
        final,
        template=template,
        root=root,
        paired=paired,
        summary=_write_summary(tmp_path),
        splits=splits,
        private_key=wrong_key,
    )

    with pytest.raises(UserInputError, match="signature is invalid"):
        finalize_semantic_review(
            reviewed,
            final,
            template_path=template,
            attestation_path=attestation,
            signature_path=signature,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            backtranslator=StubBacktranslator(),
        )


@pytest.mark.parametrize("tamper", ["attestation", "reviewed_decision"])
def test_finalizer_rejects_changes_after_human_signature(
    tmp_path: Path,
    tamper: str,
) -> None:
    template, root, paired, splits = _create(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    final = tmp_path / "final.json"
    attestation, signature = _create_signed_attestation(
        reviewed,
        final,
        template=template,
        root=root,
        paired=paired,
        summary=_write_summary(tmp_path),
        splits=splits,
    )
    if tamper == "attestation":
        payload = json.loads(attestation.read_text(encoding="utf-8"))
        payload["review_decisions_sha256"] = "0" * 64
        attestation.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        payload = json.loads(reviewed.read_text(encoding="utf-8"))
        payload["records"][0]["review"]["notes"] = "changed after signing"
        reviewed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match="does not bind"):
        finalize_semantic_review(
            reviewed,
            final,
            template_path=template,
            attestation_path=attestation,
            signature_path=signature,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            backtranslator=StubBacktranslator(),
        )


@pytest.mark.parametrize("occupied_kind", ["file", "symlink"])
def test_finalizer_never_replaces_occupied_output(
    tmp_path: Path,
    occupied_kind: str,
) -> None:
    template, root, paired, splits = _create(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    final = tmp_path / "final.json"
    attestation, signature = _create_signed_attestation(
        reviewed,
        final,
        template=template,
        root=root,
        paired=paired,
        summary=_write_summary(tmp_path),
        splits=splits,
    )
    victim = tmp_path / "victim.json"
    victim.write_text("preserve me\n", encoding="utf-8")
    if occupied_kind == "file":
        final.write_text("existing evidence\n", encoding="utf-8")
    else:
        final.symlink_to(victim)

    with pytest.raises(UserInputError, match="exists|unsafe"):
        finalize_semantic_review(
            reviewed,
            final,
            template_path=template,
            attestation_path=attestation,
            signature_path=signature,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            backtranslator=StubBacktranslator(),
        )
    assert victim.read_text(encoding="utf-8") == "preserve me\n"


@pytest.mark.parametrize("tamper", ["duplicate_key", "extra_review_field", "missing_record"])
def test_review_schema_rejects_ambiguous_or_incomplete_decisions(
    tmp_path: Path,
    tamper: str,
) -> None:
    template, root, paired, splits = _create(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    if tamper == "duplicate_key":
        text = reviewed.read_text(encoding="utf-8")
        reviewed.write_text(
            text.replace(
                '"schema_version":', '"schema_version": "duplicate",\n  "schema_version":', 1
            ),
            encoding="utf-8",
        )
        message = "missing or invalid"
    else:
        payload = json.loads(reviewed.read_text(encoding="utf-8"))
        if tamper == "extra_review_field":
            payload["records"][0]["review"]["automation"] = "bot"
            message = "review fields"
        else:
            payload["records"].pop()
            message = "exactly 200 records"
        reviewed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match=message):
        create_semantic_review_attestation(
            reviewed,
            tmp_path / "attestation.json",
            template_path=template,
            root_rows_path=root,
            paired_rows_path=paired,
            translation_summary_path=_write_summary(tmp_path),
            root_split_by_id=splits,
            expected_seed=42,
            backtranslator=StubBacktranslator(),
        )


def test_any_critical_error_fails_whole_publication(tmp_path: Path) -> None:
    template, root, paired, splits = _create(tmp_path)
    reviewed = tmp_path / "reviewed.json"
    shutil.copy2(template, reviewed)
    _complete_pass_decisions(reviewed)
    payload = json.loads(reviewed.read_text(encoding="utf-8"))
    payload["records"][0]["review"] = {
        "rubric": {
            "action_tool_intent": "fail",
            "omissions_additions": "pass",
            "polarity": "not_applicable",
            "quantities": "pass",
            "entity_relations": "pass",
        },
        "critical_error": True,
        "passes_review": False,
        "notes": "Action intent changed.",
    }
    reviewed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UserInputError, match="entire publication fails"):
        _finalize_signed(
            reviewed,
            tmp_path / "final.json",
            template=template,
            root=root,
            paired=paired,
            summary=_write_summary(tmp_path),
            splits=splits,
        )


def test_full_selection_provenance_binds_max_rows_ids_and_root_bytes(
    tmp_path: Path,
) -> None:
    root, _paired, _splits = _write_rows(tmp_path)
    ordered_ids = [f"root-{index:04d}" for index in range(210)]
    selected_ids_sha256 = hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest()
    summary = tmp_path / "selection-summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "sommelier.translation_summary.v2",
                "language": "he",
                "input": {"sha256": hashlib.sha256(root.read_bytes()).hexdigest()},
                "input_rows": 210,
                "selection": {
                    "contract_sha256": "c" * 64,
                    "mode": "full",
                    "seed": 42,
                    "max_rows": 60_000,
                    "limit": 0,
                    "selected_rows": 210,
                    "selected_source_ids_sha256": selected_ids_sha256,
                },
            }
        ),
        encoding="utf-8",
    )
    expected = TranslationStagingContract(
        selection_contract_sha256="c" * 64,
        mode="full",
        seed=42,
        max_rows=60_000,
        selected_rows=210,
        selected_source_ids_sha256=selected_ids_sha256,
    )
    validate_translation_selection_provenance(
        summary_path=summary,
        root_rows_path=root,
        target_language="he",
        expected=expected,
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["selection"]["max_rows"] = 2_500
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UserInputError, match="max_rows"):
        validate_translation_selection_provenance(
            summary_path=summary,
            root_rows_path=root,
            target_language="he",
            expected=expected,
        )


def test_french_full_publication_does_not_require_hebrew_semantic_gate(
    tmp_path: Path,
) -> None:
    _root, paired, _splits = _write_rows(tmp_path, count=1)
    row = json.loads(paired.read_text(encoding="utf-8"))
    row["query"] = "Trouvez un article pour ce compte"
    french_rows = tmp_path / "rows.fr.jsonl"
    french_rows.write_text(json.dumps(row) + "\n", encoding="utf-8")
    count, canonical_sha256 = published_rows_canonical_identity(french_rows)
    summary = tmp_path / "translation_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "sommelier.translation_summary.v2",
                "language": "fr",
                "translated_rows": count,
                "publication_identity": {
                    "rows": count,
                    "canonical_fields": list(PUBLICATION_CANONICAL_FIELDS),
                    "canonical_sha256": canonical_sha256,
                },
                "source_code": {
                    "git_commit": "a" * 40,
                    "working_tree_clean": True,
                },
                "translator": {
                    "model_id": "independent/forward-translator",
                    "model_revision": "b" * 40,
                    "implementation_revision": "a" * 40,
                },
            }
        ),
        encoding="utf-8",
    )
    publication = tmp_path / "translation_publication.json"
    write_translation_publication_manifest(
        publication,
        translated_rows_path=french_rows,
        summary_path=summary,
        target_language="fr",
    )
    validated = validate_translation_publication(
        translated_rows_path=french_rows,
        summary_path=summary,
        publication_manifest_path=publication,
        target_language="fr",
        require_full_provenance=True,
    )
    assert validated["semantic_review"] is None


def test_reviewed_publication_manifest_is_idempotent_and_never_replaced(
    tmp_path: Path,
) -> None:
    _root, paired, _splits = _write_rows(tmp_path, count=1)
    summary = _write_summary(tmp_path)
    review = tmp_path / "translation_semantic_review.json"
    template = tmp_path / "translation_semantic_review_template.json"
    review.write_text("review evidence\n", encoding="utf-8")
    template.write_text("template evidence\n", encoding="utf-8")
    manifest = tmp_path / "translation_publication.reviewed.json"

    write_translation_publication_manifest(
        manifest,
        translated_rows_path=paired,
        summary_path=summary,
        target_language="he",
        semantic_review_path=review,
        semantic_review_template_path=template,
        replace_existing=False,
    )
    expected = manifest.read_bytes()
    write_translation_publication_manifest(
        manifest,
        translated_rows_path=paired,
        summary_path=summary,
        target_language="he",
        semantic_review_path=review,
        semantic_review_template_path=template,
        replace_existing=False,
    )
    assert manifest.read_bytes() == expected

    manifest.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="already exists"):
        write_translation_publication_manifest(
            manifest,
            translated_rows_path=paired,
            summary_path=summary,
            target_language="he",
            semantic_review_path=review,
            semantic_review_template_path=template,
            replace_existing=False,
        )

    victim = tmp_path / "victim.json"
    victim.write_text("preserve\n", encoding="utf-8")
    symlink = tmp_path / "translation_publication.symlink.json"
    symlink.symlink_to(victim)
    with pytest.raises(UserInputError, match="already exists"):
        write_translation_publication_manifest(
            symlink,
            translated_rows_path=paired,
            summary_path=summary,
            target_language="he",
            semantic_review_path=review,
            semantic_review_template_path=template,
            replace_existing=False,
        )
    assert victim.read_text(encoding="utf-8") == "preserve\n"


def test_full_paired_input_contract_validates_complete_hebrew_bundle(
    tmp_path: Path,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    validated = validate_full_paired_input_contract(config, root)
    assert validated == {"he": paths}
    summary = json.loads(paths["translation_summary"].read_text(encoding="utf-8"))
    # selected_rows is the actual validated/deduplicated split cohort.  The
    # 60,000 preregistration value is only the upstream export cap.
    assert summary["selection"]["selected_rows"] == 17_000


def test_signed_review_requires_fresh_reviewed_manifest_for_dataset_publication(
    tmp_path: Path,
) -> None:
    _config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    stale_manifest = tmp_path / "translation_publication.initial.json"
    write_translation_publication_manifest(
        stale_manifest,
        translated_rows_path=paths["paired_rows"],
        summary_path=paths["translation_summary"],
        target_language="he",
    )
    reviewed_manifest = tmp_path / REVIEWED_PUBLICATION_MANIFEST_FILENAME
    write_translation_publication_manifest(
        reviewed_manifest,
        translated_rows_path=paths["paired_rows"],
        summary_path=paths["translation_summary"],
        target_language="he",
        semantic_review_path=paths["semantic_review"],
        semantic_review_template_path=paths["semantic_review_template"],
        replace_existing=False,
    )
    assert reviewed_manifest.read_bytes() == paths["translation_publication"].read_bytes()

    def fresh_bundle(destination: Path, manifest: Path) -> Path:
        destination.mkdir()
        (destination / "README.md").write_text(
            "---\nlicense: cc-by-4.0\n---\n"
            "# Hebrew machine-translated data\n\n"
            "Derived from Salesforce/xlam-function-calling-60k with signed human "
            "semantic review.\n",
            encoding="utf-8",
        )
        for source, filename in (
            (paths["paired_rows"], rows_filename("he")),
            (paths["translation_summary"], SUMMARY_FILENAME),
            (manifest, PUBLICATION_MANIFEST_FILENAME),
            (paths["semantic_review"], SEMANTIC_REVIEW_FILENAME),
            (paths["semantic_review_template"], SEMANTIC_REVIEW_TEMPLATE_FILENAME),
            (paths["translation_config"], TRANSLATION_CONFIG_FILENAME),
            (paths["translation_run_identity"], TRANSLATION_RUN_IDENTITY_FILENAME),
        ):
            shutil.copy2(source, destination / filename)
        return destination

    stale_bundle = fresh_bundle(tmp_path / "stale-bundle", stale_manifest)
    with pytest.raises(UserInputError, match="semantic-review gate"):
        prepare_hebrew_dataset_publication(
            config_path=paths["translation_config"],
            bundle_dir=stale_bundle,
            root_rows_path=root,
        )

    reviewed_bundle = fresh_bundle(tmp_path / "reviewed-bundle", reviewed_manifest)
    prepared = prepare_hebrew_dataset_publication(
        config_path=paths["translation_config"],
        bundle_dir=reviewed_bundle,
        root_rows_path=root,
    )
    assert set(prepared.files) == {
        "README.md",
        rows_filename("he"),
        SUMMARY_FILENAME,
        PUBLICATION_MANIFEST_FILENAME,
        SEMANTIC_REVIEW_FILENAME,
        SEMANTIC_REVIEW_TEMPLATE_FILENAME,
        TRANSLATION_CONFIG_FILENAME,
        TRANSLATION_RUN_IDENTITY_FILENAME,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("config", "does not bind the exact Phase-A config"),
        ("runtime", "runtime does not match the summary"),
        ("reviewer", "reviewer does not match the summary"),
    ],
)
def test_full_paired_input_contract_rejects_self_rehashed_run_identity_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    identity_path = paths["translation_run_identity"]
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if mutation == "config":
        identity["config_sha256"] = "0" * 64
    elif mutation == "runtime":
        identity["runtime"]["function_timeout_seconds"] += 1
    else:
        identity["reviewer_preregistration"]["reviewer_id"] = "substituted-reviewer"
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = paths["translation_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # Rehash the edited sidecar so this proves field-level comparison, not only
    # the outer summary digest link.
    summary["translation_run_identity_sha256"] = sha256_file(identity_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(UserInputError, match=message):
        validate_full_paired_input_contract(config, root)


@pytest.mark.parametrize("mutation", ["top_level", "publication_identity"])
def test_full_paired_input_contract_rejects_extended_hebrew_summary_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    summary_path = paths["translation_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation == "top_level":
        summary["posthoc_claim"] = True
    else:
        summary["publication_identity"]["posthoc_subset"] = "accepted-only"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(UserInputError, match="unknown or missing"):
        validate_full_paired_input_contract(config, root)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("model_id",), "substitute/immutable-translator", "exact dated model snapshot"),
        (("model_revision",), "c" * 40, "exact dated model snapshot"),
        (("decoding", "max_output_tokens"), 511, "forward translator decoding="),
        (("interface",), "translategemma", "forward translator interface="),
        (("output_decoder",), "bytelevel_unicode", "forward translator output_decoder="),
        (
            ("output_postprocessing_schema",),
            "sommelier.translation_output_postprocessing.substituted",
            "forward translator output_postprocessing_schema=",
        ),
        (("request_sha256",), "e" * 64, "forward translator request_sha256="),
    ],
)
def test_full_paired_input_contract_rejects_substituted_hebrew_forward_method(
    tmp_path: Path,
    field_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    summary = json.loads(paths["translation_summary"].read_text(encoding="utf-8"))
    current = summary["translator"]
    for field in field_path[:-1]:
        current = current[field]
    current[field_path[-1]] = value
    identity = json.loads(paths["translation_run_identity"].read_text(encoding="utf-8"))
    if field_path[-1] in {"model_id", "model_revision", "request_sha256"}:
        identity["translator"][field_path[-1]] = value
    _write_self_consistent_translation_identity_tamper(
        paths,
        summary=summary,
        identity=identity,
    )

    with pytest.raises(UserInputError, match=message):
        validate_full_paired_input_contract(config, root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("safetensors_load_strategy", "prefetch"),
        ("max_model_len", 8_192),
        ("trust_remote_code", True),
    ],
)
def test_full_paired_input_contract_rejects_local_only_translator_metadata(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    summary = json.loads(paths["translation_summary"].read_text(encoding="utf-8"))
    summary["translator"][field] = value
    paths["translation_summary"].write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(UserInputError, match="contains unregistered method fields"):
        validate_full_paired_input_contract(config, root)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("runtime", "backend"), "vllm_chat", "runtime backend"),
        (("runtime", "provider"), "other-provider", "runtime provider="),
        (("runtime", "execution_provider"), "local", "runtime execution_provider="),
        (("runtime", "provider_service_tier"), "default", "runtime provider_service_tier="),
        (("runtime", "provider_max_workers"), 7, "runtime provider_max_workers="),
        (
            ("runtime", "provider_journal_filename"),
            "renamed-provider-journal.jsonl",
            "runtime provider_journal_filename=",
        ),
        (("runtime", "translation_chunk_size"), 31, "runtime translation_chunk_size="),
        (("environment",), {"openai": "0.0.0"}, "translation environment"),
    ],
)
def test_full_paired_input_contract_rejects_translation_runtime_drift(
    tmp_path: Path,
    field_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    summary = json.loads(paths["translation_summary"].read_text(encoding="utf-8"))
    current = summary
    for field in field_path[:-1]:
        current = current[field]
    current[field_path[-1]] = value
    identity = json.loads(paths["translation_run_identity"].read_text(encoding="utf-8"))
    identity_runtime_fields = {
        "backend",
        "provider_service_tier",
        "provider_max_workers",
        "translation_chunk_size",
    }
    if field_path[-1] in identity_runtime_fields:
        identity["runtime"][field_path[-1]] = value
    _write_self_consistent_translation_identity_tamper(
        paths,
        summary=summary,
        identity=identity,
    )

    with pytest.raises(UserInputError, match=message):
        validate_full_paired_input_contract(config, root)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("decoding", "reasoning_effort"), "high", "forward translator decoding="),
        (
            ("provider_request", "sdk_version"),
            "0.0.0",
            "forward translator provider_sdk_version=",
        ),
        (("provider_request", "store"), True, "forward translator provider_request="),
        (
            ("context_budget", "provider_truncation"),
            "auto",
            "forward translator context_budget=",
        ),
    ],
)
def test_full_paired_input_contract_rejects_provider_request_drift(
    tmp_path: Path,
    field_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    summary = json.loads(paths["translation_summary"].read_text(encoding="utf-8"))
    current = summary["translator"]
    for field in field_path[:-1]:
        current = current[field]
    current[field_path[-1]] = value
    identity = json.loads(paths["translation_run_identity"].read_text(encoding="utf-8"))
    if field_path == ("provider_request", "sdk_version"):
        identity["runtime"]["provider_sdk_version"] = value
    _write_self_consistent_translation_identity_tamper(
        paths,
        summary=summary,
        identity=identity,
    )

    with pytest.raises(UserInputError, match=message):
        validate_full_paired_input_contract(config, root)


def test_full_paired_input_contract_requires_clean_provider_evidence(
    tmp_path: Path,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    summary = json.loads(paths["translation_summary"].read_text(encoding="utf-8"))
    evidence = summary["provider_evidence"]
    evidence["identity"]["returned_service_tiers"] = ["default"]
    paths["translation_summary"].write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(UserInputError, match="returned identity has drifted"):
        validate_full_paired_input_contract(config, root)


def test_full_paired_input_contract_rejects_provider_attribution_drift(
    tmp_path: Path,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    summary = json.loads(paths["translation_summary"].read_text(encoding="utf-8"))
    summary["provider_evidence"]["unique_source_attempts"] = 0
    paths["translation_summary"].write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(UserInputError, match="unique source-attempt count"):
        validate_full_paired_input_contract(config, root)


def test_full_paired_input_contract_rejects_missing_provider_evidence(
    tmp_path: Path,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    summary = json.loads(paths["translation_summary"].read_text(encoding="utf-8"))
    del summary["provider_evidence"]
    paths["translation_summary"].write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(UserInputError, match="unknown or missing top-level fields"):
        validate_full_paired_input_contract(config, root)


def test_full_paired_input_contract_rejects_self_consistent_posthoc_max_rows(
    tmp_path: Path,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    summary = json.loads(paths["translation_summary"].read_text(encoding="utf-8"))
    selection = summary["selection"]
    selection["max_rows"] = 59_999
    # Rehashing the altered selection contract must not turn a post-hoc cohort
    # into the preregistered Hebrew v3 cohort.
    selection["contract_sha256"] = translation_selection_contract_sha256(
        config,
        mode="full",
        max_rows=59_999,
        limit=0,
    )
    identity = json.loads(paths["translation_run_identity"].read_text(encoding="utf-8"))
    identity["selection"]["max_rows"] = selection["max_rows"]
    identity["selection"]["contract_sha256"] = selection["contract_sha256"]
    _write_self_consistent_translation_identity_tamper(
        paths,
        summary=summary,
        identity=identity,
    )

    with pytest.raises(UserInputError, match="translation selection contract_sha256"):
        validate_full_paired_input_contract(config, root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "smoke"),
        ("seed", 7),
        ("limit", 1),
    ],
)
def test_full_paired_input_contract_rejects_posthoc_hebrew_selection_controls(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    summary = json.loads(paths["translation_summary"].read_text(encoding="utf-8"))
    summary["selection"][field] = value
    identity = json.loads(paths["translation_run_identity"].read_text(encoding="utf-8"))
    identity["selection"][field] = value
    _write_self_consistent_translation_identity_tamper(
        paths,
        summary=summary,
        identity=identity,
    )

    with pytest.raises(UserInputError, match=rf"translation selection {field}="):
        validate_full_paired_input_contract(config, root)


def test_full_paired_input_contract_rejects_non_preregistered_hebrew_config_seed(
    tmp_path: Path,
) -> None:
    config, root, _paths = _full_hebrew_contract_bundle(tmp_path)
    config.project.seed = 7
    with pytest.raises(
        UserInputError,
        match="Phase-B config differs from Phase A by more than.*dataset revision",
    ):
        validate_full_paired_input_contract(config, root)


def test_full_paired_input_contract_rejects_missing_hebrew_template(
    tmp_path: Path,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    paths["semantic_review_template"].unlink()
    with pytest.raises(UserInputError, match="semantic_review_template is not a regular file"):
        validate_full_paired_input_contract(config, root)


def test_full_paired_input_contract_rejects_tampered_paired_rows(
    tmp_path: Path,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    rows = paths["paired_rows"].read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["query"] += " שינוי"
    rows[0] = json.dumps(first, ensure_ascii=False)
    paths["paired_rows"].write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(UserInputError, match="canonical identity"):
        validate_full_paired_input_contract(config, root)


def test_full_paired_input_contract_rejects_critical_hebrew_review(
    tmp_path: Path,
) -> None:
    config, root, paths = _full_hebrew_contract_bundle(tmp_path)
    review = json.loads(paths["semantic_review"].read_text(encoding="utf-8"))
    review["records"][0]["review"] = {
        "rubric": {
            "action_tool_intent": "fail",
            "omissions_additions": "pass",
            "polarity": "not_applicable",
            "quantities": "pass",
            "entity_relations": "pass",
        },
        "critical_error": True,
        "passes_review": False,
        "notes": "Critical action-intent regression.",
    }
    paths["semantic_review"].write_text(json.dumps(review), encoding="utf-8")
    write_translation_publication_manifest(
        paths["translation_publication"],
        translated_rows_path=paths["paired_rows"],
        summary_path=paths["translation_summary"],
        target_language="he",
        semantic_review_path=paths["semantic_review"],
        semantic_review_template_path=paths["semantic_review_template"],
    )
    with pytest.raises(UserInputError, match="publication gate did not pass"):
        validate_full_paired_input_contract(config, root)


def test_full_paired_input_contract_rejects_mutable_dataset_revision(
    tmp_path: Path,
) -> None:
    config, root, _paths = _full_hebrew_contract_bundle(tmp_path)
    config.datasets[1].dataset_revision = "main"
    with pytest.raises(UserInputError, match="immutable revision"):
        validate_full_paired_input_contract(config, root)
