"""Separator 🗄️ Zaplecze (MAIL-17) — granica strefy „Na dziś” i kontekstu.

Za nim idą ryzyko, skuteczność modelu i rady, nastroje, koszt cyklu, pominięte
symbole i błędy. Gmail przycina mail od dołu, więc przy przycięciu znika
Zaplecze, nie część akcyjna.
"""

from __future__ import annotations

from src.application.reporting.section import ReportSection, SectionInput

_BACKOFFICE_TITLE = "🗄️ Zaplecze"
_BACKOFFICE_NOTE = (
    "Kontekst i kontrola: ryzyko, skuteczność modelu i rady, nastroje, koszt cyklu, "
    "pominięte symbole i błędy."
)
_BACKOFFICE_SEPARATOR_HTML = (
    '<div style="margin: 32px 0 8px 0; padding-top: 12px; border-top: 2px solid #d1d5db;">'
    '<div style="font-size: 18px; font-weight: 700; color: #374151;">'
    f"{_BACKOFFICE_TITLE}</div>"
    f'<div style="font-size: 12px; color: #6b7280;">{_BACKOFFICE_NOTE}</div>'
    "</div>"
)


def _section_html(data: SectionInput) -> str:
    return _BACKOFFICE_SEPARATOR_HTML


def _section_text(data: SectionInput) -> list[str]:
    return ["=" * 64, "ZAPLECZE", _BACKOFFICE_NOTE, "=" * 64, ""]


BACKOFFICE_SECTION = ReportSection(
    key="backoffice", render_html=_section_html, render_text=_section_text
)
