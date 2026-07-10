"""Render sekcji „Lead 5 sekund" — pierwszy ekran maila, tuż PO quota bannerze.

Lead to najmocniejszy sygnał cyklu wyeksponowany na wejściu: czytelnik ma w 5
sekund wiedzieć, „czy coś się pali". Domena (`build_lead`) już ustaliła
priorytety (CRITICAL quota > rada podzielona > silny konsensus > werdykt) i
zamroziła kolejność — ten moduł jest CZYSTO prezentacyjny: przepisuje gotowe
`LeadItem`-y do HTML/plain-text i NIE zmienia ich kolejności ani doboru.

Sekcja samosupresująca: pusty lead → `render_*` zwraca "" (orkiestrator po
prostu jej nie renderuje).

Escapowanie: `icon`, `headline` i `detail` pochodzą pośrednio z newsów (przez
werdykty rady), więc w wariancie HTML przechodzą przez `html.escape`.
"""

from __future__ import annotations

import html as _html

from src.domain.digest_lead import LeadItem

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
