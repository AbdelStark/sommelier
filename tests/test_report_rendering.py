from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sommelier.evaluation.render import render_comparison_markdown

FIXTURES_DIR = Path("tests/fixtures/reports")
GOLDEN_MD_PATH = FIXTURES_DIR / "golden_comparison_report.md"

REGENERATE_ENV = "SOMMELIER_REGENERATE_GOLDEN"


def metric(value: float, numerator: int, denominator: int) -> dict[str, Any]:
    return {"value": value, "numerator": numerator, "denominator": denominator}


def fixture_comparison() -> dict[str, Any]:
    base_metrics = {
        "valid_json_rate": metric(0.5, 10, 20),
        "function_name_accuracy": metric(0.45, 9, 20),
        "argument_exact_match": metric(0.25, 5, 20),
        "argument_f1": metric(0.6, 48, 80),
        "full_call_exact_match": metric(0.2, 4, 20),
    }
    adapter_metrics = {
        "valid_json_rate": metric(0.95, 19, 20),
        "function_name_accuracy": metric(0.9, 18, 20),
        "argument_exact_match": metric(0.75, 15, 20),
        "argument_f1": metric(0.9, 72, 80),
        "full_call_exact_match": metric(0.7, 14, 20),
    }
    deltas = {
        name: adapter_metrics[name]["value"] - base_metrics[name]["value"]
        for name in base_metrics
    }
    return {
        "schema_version": "sommelier.comparison_report.v2",
        "created_at": "2026-07-02T12:00:00+00:00",
        "run_id": "smoke-fixture-1",
        "shared": {
            "config_sha256": "c" * 64,
            "split": "test",
            "test_split_sha256": "t" * 64,
            "parser_version": "sommelier.parser.v1",
            "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 512},
        },
        "slices": {
            "en": {
                "examples": 20,
                "prompt_set_sha256": "p" * 64,
                "base": {"metrics": base_metrics},
                "adapter": {"metrics": adapter_metrics},
                "deltas": dict(deltas),
                "generation_artifacts": {
                    "base": "runs/smoke-fixture-1/eval/base/generations.en.jsonl",
                    "adapter": "runs/smoke-fixture-1/eval/adapter/generations.en.jsonl",
                },
            },
            "fr": {
                "examples": 20,
                "prompt_set_sha256": "q" * 64,
                "base": {"metrics": base_metrics},
                "adapter": {"metrics": adapter_metrics},
                "deltas": dict(deltas),
                "generation_artifacts": {
                    "base": "runs/smoke-fixture-1/eval/base/generations.fr.jsonl",
                    "adapter": "runs/smoke-fixture-1/eval/adapter/generations.fr.jsonl",
                },
            },
        },
        "language_gaps": {
            "reference": "en",
            "base": {"fr": {name: 0.0 for name in base_metrics}},
            "adapter": {"fr": {name: 0.0 for name in adapter_metrics}},
        },
        "base": {
            "run_id": "smoke-fixture-1",
            "metrics": base_metrics,
            "adapter_source": None,
        },
        "adapter": {
            "run_id": "smoke-fixture-1",
            "metrics": adapter_metrics,
            "adapter_source": {
                "source": "abdelstark/example-adapter",
                "revision": "main",
                "kind": "huggingface_repo",
            },
        },
        "deltas": deltas,
        "runtime": {
            "available": True,
            "schema_version": "sommelier.runtime_metadata.v1",
            "stages": {
                "train": {"elapsed_seconds": 180.5},
                "eval-base": {"elapsed_seconds": 60.0},
            },
            "hardware": {"gpu": "A10G", "source": "config"},
            "peak_gpu_memory_mb": 15872,
            "observed_cost_usd": None,
            "cost_source": "unavailable",
        },
    }


def test_markdown_snapshot_matches_golden() -> None:
    rendered = render_comparison_markdown(fixture_comparison())

    if os.environ.get(REGENERATE_ENV) == "1":
        GOLDEN_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_MD_PATH.write_text(rendered, encoding="utf-8")

    golden = GOLDEN_MD_PATH.read_text(encoding="utf-8")
    assert rendered == golden, (
        f"report rendering drift; if intentional, regenerate with {REGENERATE_ENV}=1"
    )


def test_markdown_contains_required_sections() -> None:
    rendered = render_comparison_markdown(fixture_comparison())
    for heading in (
        "## Run Identity",
        "## Split Summary",
        "## Metrics, all slices",
        "## Metrics, slice `en`",
        "## Metrics, slice `fr`",
        "## Language Gaps",
        "## Runtime and Cost",
        "## Reproduction",
        "## Limitations",
    ):
        assert heading in rendered, heading
    assert "authoritative for" in rendered
    assert "Evidence class: smoke run" in rendered
    assert "sommelier report compare" in rendered
    assert "production readiness" in rendered
    assert "machine-translated" in rendered
    assert "abdelstark/example-adapter" in rendered


def test_markdown_renders_metric_deltas() -> None:
    rendered = render_comparison_markdown(fixture_comparison())
    assert "| valid_json_rate | 0.5000 (10/20) | 0.9500 (19/20) | +0.4500 |" in rendered


def test_markdown_marks_unavailable_runtime() -> None:
    comparison = fixture_comparison()
    comparison["runtime"] = {"available": False}
    rendered = render_comparison_markdown(comparison)
    assert "Runtime metadata is unavailable for this run." in rendered


def test_markdown_marks_unavailable_cost_explicitly() -> None:
    rendered = render_comparison_markdown(fixture_comparison())
    assert "Observed cost: unavailable (source: unavailable)" in rendered
    assert "Peak GPU memory: 15872 MiB" in rendered


def test_full_run_evidence_class() -> None:
    comparison = fixture_comparison()
    comparison["run_id"] = "20260702T120000Z-abcd1234"
    rendered = render_comparison_markdown(comparison)
    assert "Evidence class: full run" in rendered


def test_json_stays_authoritative() -> None:
    comparison = fixture_comparison()
    rendered = render_comparison_markdown(comparison)
    # Rendering must not mutate the source dict.
    assert comparison == fixture_comparison()
    assert json.dumps(comparison, sort_keys=True)
    assert rendered.startswith("# Sommelier Comparison Report")
