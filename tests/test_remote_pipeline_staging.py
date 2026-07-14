from __future__ import annotations

import hashlib
import json
import shutil
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
    create_semantic_review_template,
    finalize_semantic_review,
    root_split_assignments,
)
from sommelier.data.translate import (
    PUBLICATION_CANONICAL_FIELDS,
    PUBLICATION_MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    published_rows_canonical_identity,
    translation_selection_contract_sha256,
    write_translation_publication_manifest,
)
from sommelier.errors import UserInputError

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _full_config(tmp_path: Path, *, paired_revision: str) -> Path:
    config_text = (EXAMPLES_DIR / "config.v3-he-full.yaml").read_text(encoding="utf-8")
    config_text = (
        config_text.replace("n_train: 15000", "n_train: 200")
        .replace("n_validation: 1000", "n_validation: 1")
        .replace("n_test: 1000", "n_test: 1")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        config_text.replace("dataset_revision: main", f"dataset_revision: {paired_revision}"),
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
    config_path = _full_config(tmp_path, paired_revision=revision)
    config = load_config(config_path)
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
        finalize_semantic_review(
            reviewed_path,
            review_path,
            template_path=template_path,
            reviewer_id="fixture-reviewer",
            root_rows_path=root_rows_path,
            paired_rows_path=exported_path,
            translation_summary_path=summary_path,
            root_split_by_id=split_by_id,
            expected_seed=config.project.seed,
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
            ordered_ids = [str(row["source_id"]) for row in root_records]
            summary_path.write_text(
                json.dumps(
                    {
                        "schema_version": "sommelier.translation_summary.v2",
                        "language": "he",
                        "input": {"sha256": sha256_file(root_rows_path)},
                        "input_rows": len(root_records),
                        "translated_rows": rows,
                        "selection": {
                            "contract_sha256": translation_selection_contract_sha256(
                                config,
                                mode="full",
                                max_rows=60_000,
                                limit=0,
                            ),
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
                        "source_code": {
                            "git_commit": "a" * 40,
                            "working_tree_clean": True,
                        },
                        "translator": {
                            "model_id": "dicta-il/DictaLM-3.0-Nemotron-12B-Instruct",
                            "model_revision": "b" * 40,
                            "implementation_revision": "a" * 40,
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(summary_path)
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
