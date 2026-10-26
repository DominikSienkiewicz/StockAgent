"""Sekcja 🧠 Wnioski z poprzednich cykli (Self-Reflection).

`reflect_node` ocenia przeszłą predykcję PRZED bramką volatility; tu trafia jej
wniosek dla symboli zapisanych w tym cyklu. Wniosek pochodzi z LLM-a, więc
w HTML przechodzi przez `_html`.
"""

from __future__ import annotations

from src.application.report_formatting import (
    company_label as _company_label,
)
from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_formatting import (
    section_h2 as _section_h2,
)
from src.application.report_models import SymbolResult
from src.application.reporting.section import ReportSection, SectionInput


def _render_reflections_html(saved: list[SymbolResult]) -> str:
    """🧠 Wnioski z poprzednich cykli (Self-Reflection). Pusty string, gdy brak."""
    reflections = [r for r in saved if r.reflection_insight]
    if not reflections:
        return ""
    parts = [_section_h2("🧠 Wnioski z poprzednich cykli (Self-Reflection)")]
    for r in reflections:
        parts.append(f"""
              <div style="margin-bottom: 8px; padding: 10px 12px; background: #faf5ff;
                          border-left: 3px solid #9333ea; border-radius: 4px;
                          font-size: 12px;">
                <strong>{_html(_company_label(r.symbol))}:</strong>
                <span style="color: #581c87;">{_html(r.reflection_insight)}</span>
              </div>
            """)
    return "".join(parts)


def _render_reflections_text(saved: list[SymbolResult]) -> list[str]:
    """Sekcja 'Wnioski z poprzednich cykli' (plain). Pusta lista, gdy brak."""
    reflections = [r for r in saved if r.reflection_insight]
    if not reflections:
        return []
    lines = ["WNIOSKI Z POPRZEDNICH CYKLI (Self-Reflection)", "-" * 64]
    lines.extend(f"  {_company_label(r.symbol)}: {r.reflection_insight}" for r in reflections)
    lines.append("")
    return lines


def _section_html(data: SectionInput) -> str:
    return _render_reflections_html(data.saved)


def _section_text(data: SectionInput) -> list[str]:
    return _render_reflections_text(data.saved)


REFLECTIONS_SECTION = ReportSection(
    key="reflections", render_html=_section_html, render_text=_section_text
)
