"""DOC-07: README mówi, od kiedy i dlaczego Fast Loop stoi oraz jak go wznowić.

Pierwszym twardym faktem w README było samo „paused” — bez daty, powodu i warunków
wznowienia. Dopóki cron `fast_loop_12h.yml` jest zakomentowany, README ma sekcję
„## Status” z datą pauzy, powodem, odnośnikiem do checklisty wznowienia i do
strony workflowów w GitHub Actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parent.parent
README = ROOT / "README.md"
FAST_LOOP_WORKFLOW = ROOT / ".github" / "workflows" / "fast_loop_12h.yml"
STATUS_HEADING = "## Status"


def _fast_loop_schedule() -> list[Any]:
    spec = yaml.safe_load(FAST_LOOP_WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) czyta gołe `on:` jako klucz True.
    triggers = spec.get("on", spec.get(True)) or {}
    return list(triggers.get("schedule") or []) if isinstance(triggers, dict) else []


def _skip_unless_paused() -> None:
    if _fast_loop_schedule():
        pytest.skip("cron Fast Loopa aktywny — sekcja Status opisuje pauzę")


def _status_section(readme: str) -> str:
    _, _, rest = readme.partition(f"\n{STATUS_HEADING}\n")
    return rest.split("\n## ", 1)[0]


def test_paused_fast_loop_readme_has_a_status_section_with_resume_conditions() -> None:
    _skip_unless_paused()
    readme = README.read_text(encoding="utf-8")

    assert STATUS_HEADING in readme
    assert "Resume" in _status_section(readme)


def test_status_section_dates_the_pause_and_links_the_checklist_and_the_runs() -> None:
    _skip_unless_paused()
    section = _status_section(README.read_text(encoding="utf-8"))

    assert "2026-08-17" in section
    assert "02fc397" in section
    assert "docs/resume-checklist.md" in section
    assert "https://github.com/DominikSienkiewicz/StockAgent/actions" in section
