from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")

EXTERNAL_PREFIXES = ("http://", "https://", "#", "mailto:")


def markdown_files() -> list[Path]:
    files = [REPO_ROOT / "README.md", REPO_ROOT / "CHANGELOG.md"]
    files.extend((REPO_ROOT / "docs").rglob("*.md"))
    files.extend((REPO_ROOT / "licenses").rglob("*.md"))
    return [path for path in files if path.exists()]


def test_relative_documentation_links_resolve() -> None:
    broken: list[str] = []
    for doc in markdown_files():
        for target in LINK_PATTERN.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(EXTERNAL_PREFIXES):
                continue
            file_part = target.split("#", 1)[0]
            if not file_part:
                continue
            resolved = (doc.parent / file_part).resolve()
            if not resolved.exists():
                broken.append(f"{doc.relative_to(REPO_ROOT)}: {target}")
    assert broken == [], "broken documentation links:\n" + "\n".join(broken)


def test_readme_has_install_and_quickstart() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Install and Quickstart" in readme
    assert "uv sync --extra dev" in readme
    assert "docs/guides/reproduction.md" in readme


def test_reproduction_guide_covers_required_topics() -> None:
    guide = (REPO_ROOT / "docs" / "guides" / "reproduction.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. Install",
        "## 2. Local validation",
        "## 3. Remote prerequisites",
        "## 4. Smoke run",
        "## 5. Full run",
        "## 6. Reading the report",
        "## 7. Caveats",
    ):
        assert heading in guide, heading
    assert "SOMMELIER_ACK_BASE_MODEL_LICENSE" in guide
    assert "HF_TOKEN" in guide
    assert "costs money" in guide or "bill" in guide


def test_release_checklist_links_implemented_commands() -> None:
    checklist = (REPO_ROOT / "docs" / "release" / "v1.0-checklist.md").read_text(
        encoding="utf-8"
    )
    for command in (
        "uv run pytest",
        "sommelier pipeline run",
        "--mode smoke",
        "--mode full",
        "sommelier release preflight",
        "sommelier report compare",
    ):
        assert command in checklist, command
    for topic in ("secret scan", "Changelog", "license"):
        assert topic.lower() in checklist.lower(), topic


def test_changelog_states_policy_categories() -> None:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## Changelog policy" in changelog
    for category in ("Added", "Changed", "Fixed", "Deprecated", "Removed", "Security"):
        assert category in changelog, category
    assert "migration note" in changelog.lower()


def test_readme_labels_serving_as_illustrative() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "optional and illustrative" in readme
    assert "no production readiness" in readme.lower() or "not a\nproduction" in readme
    assert "autoscaling" in readme
    assert "multi-tenant" in readme
    assert "/v1/chat/completions" in readme
