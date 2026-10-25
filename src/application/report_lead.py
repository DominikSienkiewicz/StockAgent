"""Render sekcji „Lead 5 sekund" — pierwszy ekran maila, tuż PO quota bannerze.

Lead to najmocniejszy sygnał cyklu wyeksponowany na wejściu: czytelnik ma w 5
sekund wiedzieć, „czy coś się pali". Domena (`build_lead`) już ustaliła
priorytety (CRITICAL quota > portfel > rada podzielona > flip rady > silny
konsensus > werdykt) i zamroziła kolejność — ten moduł jest CZYSTO prezentacyjny:
przepisuje gotowe `LeadItem`-y do HTML/plain-text i NIE zmienia ich kolejności
ani doboru.

Sekcja samosupresująca: pusty lead → `render_*` zwraca "" (orkiestrator po
prostu jej nie renderuje).

Escapowanie: `icon`, `headline` i `detail` pochodzą pośrednio z newsów (przez
werdykty rady), więc w wariancie HTML przechodzą przez `html.escape`.
"""

from __future__ import annotations

import html as _html

from src.application.reporting.section import (
    ReportSection,
    SectionInput,
    no_text,
    text_block,
)
from src.domain.digest_lead import LeadItem, lead_preview

# Akcent leadu — ciemny pasek z lewej, żeby pierwszy ekran wizualnie „krzyczał".
# Neutralny grafit (nie czerwień/żółć), bo severity niesie już ikona pozycji.
_ACCENT_COLOR = "#111827"
_ACCENT_BG = "#f9fafb"


def render_lead_html(items: list[LeadItem]) -> str:
    """Renderuje wyróżniony blok leadu w HTML lub "" gdy brak pozycji."""
    if not items:
        return ""

    rows = [_render_item_html(item) for item in items]

    return (
        f"<div style='padding: 12px 16px; background: {_ACCENT_BG}; "
        f"border-left: 4px solid {_ACCENT_COLOR}; border-radius: 4px; "
        f"margin-bottom: 16px;'>"
        f"<div style='font-size: 13px; font-weight: 700; color: {_ACCENT_COLOR}; "
        f"text-transform: uppercase; letter-spacing: 0.03em; "
        f"margin-bottom: 8px;'>⚡ Najważniejsze w tym cyklu</div>"
        + "".join(rows)
        + "</div>"
    )


def _render_item_html(item: LeadItem) -> str:
    """Jedna pozycja leadu: ikona + wytłuszczony headline, pod nim opcjonalny detail."""
    icon = _html.escape(item.icon)
    headline = _html.escape(item.headline)
    detail_html = ""
    if item.detail:
        detail = _html.escape(item.detail)
        detail_html = (
            f"<div style='font-size: 12px; color: #6b7280; "
            f"margin: 2px 0 0 24px;'>{detail}</div>"
        )
    return (
        f"<div style='margin-bottom: 6px;'>"
        f"<div style='font-size: 15px; font-weight: 600; color: #111827;'>"
        f"{icon} {headline}</div>"
        f"{detail_html}"
        f"</div>"
    )


def render_lead_text(items: list[LeadItem]) -> str:
    """Renderuje blok leadu w plain-text lub "" gdy brak pozycji."""
    if not items:
        return ""

    lines = ["=== ⚡ NAJWAŻNIEJSZE W TYM CYKLU ==="]
    for item in items:
        lines.append(f"  {item.icon} {item.headline}")
        if item.detail:
            lines.append(f"    → {item.detail}")
    lines.append("")
    return "\n".join(lines)


# MAIL-13: wypełniacz za treścią preheadera — bez niego klient poczty dociąga do
# podglądu dalszy tekst maila („📊 StockAgent — raport cyklu Czas rozpoczęcia…”).
_PREHEADER_FILLER = "&zwnj;&nbsp;" * 100


def _render_preheader_html(lead_items: list[LeadItem], closed_count: int | None) -> str:
    """MAIL-13: ukryty div na samym początku HTML — klient poczty pokazuje go jako
    podgląd pod tematem. Treść leadu pochodzi z rady i newsów, więc jest escapowana.

    `closed_count` None (MAIL-25) = odczyt rozliczeń padł: podgląd nie mówi „0”."""
    closed = (
        "rozliczenia niedostępne"
        if closed_count is None
        else f"{closed_count} zamkniętych predykcji"
    )
    preview = lead_preview(lead_items) or f"Brak sygnałów ponad próg · {closed}"
    return (
        '<div style="display:none; max-height:0; overflow:hidden; mso-hide:all;">'
        f"{_html.escape(preview, quote=True)}{_PREHEADER_FILLER}</div>"
    )


def _preheader_html(data: SectionInput) -> str:
    unavailable = data.context.resolved_unavailable and not data.resolved
    return _render_preheader_html(data.lead_items, None if unavailable else len(data.resolved))


def _lead_html(data: SectionInput) -> str:
    return render_lead_html(data.lead_items)


def _lead_text(data: SectionInput) -> list[str]:
    return text_block(render_lead_text(data.lead_items), spaced=True)


PREHEADER_SECTION = ReportSection(key="preheader", render_html=_preheader_html, render_text=no_text)
# #1: lead cyklu — pierwszy ekran, tuż za bannerem kwot.
LEAD_SECTION = ReportSection(key="lead", render_html=_lead_html, render_text=_lead_text)
