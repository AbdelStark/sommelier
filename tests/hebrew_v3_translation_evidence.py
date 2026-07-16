"""Small, provider-free Hebrew v3 translation evidence for boundary tests."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from sommelier.artifacts import sha256_file
from sommelier.config import load_config
from sommelier.data.openai_evidence import (
    OPENAI_PROVIDER_JOURNAL_FILENAME,
    build_openai_provider_evidence,
)
from sommelier.data.openai_pricing import openai_list_price_ceiling_runtime_summary
from sommelier.data.openai_translate import (
    OPENAI_RESPONSES_PROVIDER_JOURNAL_SCHEMA,
    OPENAI_RESPONSES_PROVIDER_JOURNAL_SUMMARY_SCHEMA,
    OPENAI_RESPONSES_SAFETY_IDENTIFIER,
    OPENAI_RESPONSES_SDK_MAX_RETRIES,
    OPENAI_RESPONSES_TIMEOUT_SECONDS,
    openai_flex_resource_unavailable_retry_policy,
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
    TRANSLATION_RUN_IDENTITY_FILENAME,
    TRANSLATION_RUN_IDENTITY_SCHEMA,
    TranslatorInfo,
    translation_selection_contract_sha256,
    translator_request_sha256,
    write_translation_outputs,
)
from sommelier.hebrew_v3_preregistration import (
    reviewer_anchor_payload,
    reviewer_anchor_sha256,
)
from sommelier.remote.images import OPENAI_TRANSLATION_RUNTIME_VERSIONS


def _provider_evidence(request_count: int) -> dict[str, object]:
    aggregate: dict[str, object] = {
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
        "resource_unavailable_retry_policy": openai_flex_resource_unavailable_retry_policy(),
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
    with patch(
        "sommelier.data.openai_evidence.aggregate_openai_responses_provider_journal",
        return_value=aggregate,
    ):
        return build_openai_provider_evidence(
            Path(OPENAI_PROVIDER_JOURNAL_FILENAME),
            HEBREW_V3_FORWARD_TRANSLATOR_MODEL_ID,
            HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
        )


def write_phase_a_translation_evidence(
    root: Path,
    *,
    config_path: Path,
    run_id: str,
    source_boundary: str,
) -> tuple[Path, Path]:
    """Write a closed, self-consistent summary and pre-provider identity."""
    config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    selected_rows = config.data.n_train + config.data.n_validation + config.data.n_test
    implementation_revision = "a" * 40
    source_code = {
        "git_commit": implementation_revision,
        "working_tree_clean": True,
        "boundary": source_boundary,
    }
    translator = TranslatorInfo(
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
    selection = {
        "config_sha256": config_sha256,
        "contract_sha256": translation_selection_contract_sha256(
            config,
            mode="full",
            max_rows=HEBREW_V3_TRANSLATION_MAX_ROWS,
            limit=0,
        ),
        "mode": "full",
        "max_rows": HEBREW_V3_TRANSLATION_MAX_ROWS,
        "limit": 0,
        "seed": config.project.seed,
        "selected_rows": selected_rows,
        "selected_source_ids_sha256": "b" * 64,
    }
    reviewer_payload = reviewer_anchor_payload(config)
    reviewer_sha256 = reviewer_anchor_sha256(config)
    identity_path = root / TRANSLATION_RUN_IDENTITY_FILENAME
    identity_path.write_text(
        json.dumps(
            {
                "schema_version": TRANSLATION_RUN_IDENTITY_SCHEMA,
                "run_id": run_id,
                "config_sha256": config_sha256,
                "selection": {
                    field: selection[field]
                    for field in ("contract_sha256", "mode", "max_rows", "limit", "seed")
                },
                "translator": {
                    "model_id": translator.model_id,
                    "model_revision": translator.model_revision,
                    "request_sha256": translator_request_sha256(translator, "he"),
                    "max_attempts": HEBREW_V3_TRANSLATION_MAX_ATTEMPTS,
                    "implementation_revision": implementation_revision,
                },
                "runtime": {
                    "backend": HEBREW_V3_TRANSLATION_RUNTIME_BACKEND,
                    "translation_chunk_size": HEBREW_V3_TRANSLATION_CHUNK_SIZE,
                    "allocation_gpu": None,
                    "function_timeout_seconds": 14_400,
                    "provider_service_tier": HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
                    "provider_sdk_version": HEBREW_V3_TRANSLATION_PROVIDER_SDK_VERSION,
                    "provider_timeout_seconds": OPENAI_RESPONSES_TIMEOUT_SECONDS,
                    "provider_max_workers": HEBREW_V3_TRANSLATION_PROVIDER_MAX_WORKERS,
                    "openai_list_price_limit_usd": "50.00",
                },
                "source_code": source_code,
                "reviewer_preregistration": reviewer_payload,
                "reviewer_preregistration_sha256": reviewer_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = root / "translation_summary.json"
    _generated_rows, generated_summary = write_translation_outputs(
        root,
        [],
        {
            "input_rows": selected_rows,
            "translated_rows": selected_rows,
            "max_attempts": HEBREW_V3_TRANSLATION_MAX_ATTEMPTS,
            "translation_attempts": selected_rows,
            "retried_rows": 0,
            "dropped": {reason: 0 for reason in DROP_REASONS},
            "rows_sha256": "c" * 64,
            "publication_identity": {
                "rows": selected_rows,
                "canonical_fields": ["source_example_id", "query", "tools", "answers"],
                "canonical_sha256": "d" * 64,
            },
            "translation_identity_sha256": "f" * 64,
            "environment": dict(OPENAI_TRANSLATION_RUNTIME_VERSIONS),
            "runtime": {
                "gpu": None,
                "provider": "openai",
                "execution_provider": "modal",
                "backend": HEBREW_V3_TRANSLATION_RUNTIME_BACKEND,
                "gpu_allocation_label": None,
                "function_timeout_seconds": 14_400,
                "model_load_seconds": 0.0,
                "translation_seconds": 1.0,
                "translation_chunk_size": HEBREW_V3_TRANSLATION_CHUNK_SIZE,
                "boundary": source_boundary,
                "provider_service_tier": HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
                "provider_timeout_seconds": OPENAI_RESPONSES_TIMEOUT_SECONDS,
                "provider_max_workers": HEBREW_V3_TRANSLATION_PROVIDER_MAX_WORKERS,
                "provider_journal_filename": OPENAI_PROVIDER_JOURNAL_FILENAME,
                "openai_list_price_ceiling": openai_list_price_ceiling_runtime_summary(
                    Decimal("50.00"),
                    service_tier=HEBREW_V3_TRANSLATION_PROVIDER_SERVICE_TIER,
                ),
                "credential_source": "Synthetic boundary fixture; no credentials used.",
            },
            "provider_evidence": _provider_evidence(selected_rows),
            "selection": selection,
            "source_code": source_code,
            "reviewer_preregistration": reviewer_payload,
            "reviewer_preregistration_sha256": reviewer_sha256,
            "translation_run_identity_sha256": hashlib.sha256(
                identity_path.read_bytes()
            ).hexdigest(),
        },
        translator=translator,
        input_description="synthetic provider-free Hebrew v3 boundary fixture",
        target_language="he",
        input_sha256="1" * 64,
    )
    assert generated_summary == summary_path
    return summary_path, identity_path


def self_rehash_translation_contract_drift(
    summary_path: Path,
    identity_path: Path,
    *,
    section: str,
    field: str,
    value: object,
) -> None:
    """Apply matching summary/identity drift and refresh the identity binding."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assert isinstance(summary, dict)
    assert isinstance(identity, dict)
    summary_section = summary[section]
    identity_section = identity[section]
    assert isinstance(summary_section, dict)
    assert isinstance(identity_section, dict)
    summary_section[field] = value
    identity_section[field] = value
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["translation_run_identity_sha256"] = sha256_file(identity_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
