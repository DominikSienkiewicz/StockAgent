"""Nagłówek maila i pierwszy ekran: tytuł, sesja NYSE, trzy liczby, wiarygodność.

Trzy sekcje z rejestru ARCH-12:

- `masthead` — otwiera ramkę maila (zamyka ją stopka, `report_footer.py`),
  tytuł, czas startu cyklu i linia reżimu (ALR-05). W wersji tekstowej pusta:
  tekst stawia bannery PRZED tytułem, więc tytuł drukuje sekcja `session`.
- `session` — status sesji NYSE; w tekście cały blok tytułowy.
- `first_screen` — MAIL-16: zmiana portfela (tylko właściciel z pozycjami),
  sygnały ponad próg i rozliczenia; MAIL-17: pasek wiarygodności modelu.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_formatting import (
    pct as _pct,
)
from src.application.report_formatting import safe_http_url
from src.application.report_models import ResolvedPrediction, SymbolResult
from src.application.report_regime import render_regime_html, render_regime_text
from src.application.report_resolved import RESOLVED_WINDOW_24H, RESOLVED_WINDOW_THIS_CYCLE
from src.application.report_signals import market_status
from src.application.report_user_portfolio import (
    render_portfolio_change_box_html,
    render_portfolio_change_text_line,
)
from src.application.reporting.section import ReportSection, SectionInput, no_text

# ---------------------------------------------------------------------------
# Nagłówek i sesja
# ---------------------------------------------------------------------------

# Ramka maila: otwiera ją nagłówek, zamyka stopka.
_FRAME_STYLE = (
    "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; "
    "max-width: 720px; margin: 0 auto; color: #1f2937;"
)


# MAIL-24: link do pełnej wersji w prywatnym archiwum (podpisany URL, TTL 7 dni).
ARCHIVE_LINK_LABEL = "Otwórz w przeglądarce"
ARCHIVE_TEXT_LABEL = "Wersja w przeglądarce (link ważny 7 dni)"


def _archive_link_html(archive_url: str | None) -> str:
    """„ · Otwórz w przeglądarce” z podpisanym URL-em albo "" — tylko http(s)."""
    url = safe_http_url(archive_url)
    if url is None:
        return ""
    return f' · <a href="{_html(url)}" style="color: #2563eb;">{ARCHIVE_LINK_LABEL}</a>'


def _masthead_html(data: SectionInput) -> str:
    regime_html = render_regime_html(data.context.regime)
    archive_html = _archive_link_html(data.context.archive_url)
    return f"""
    <div style="{_FRAME_STYLE}">
      <h1 style="font-size: 22px; margin: 0 0 4px 0;">📊 StockAgent — raport cyklu</h1>
      <div style="font-size: 13px; color: #6b7280; margin-bottom: 24px;">
        Czas rozpoczęcia: {data.started_local}{archive_html}{regime_html}
      </div>
    """


def _session_html(data: SectionInput) -> str:
    session = market_status(data.started_at)
    return f"""
      <div style="padding: 10px 14px; background: #f9fafb; border-radius: 4px;
                  margin-bottom: 16px; font-size: 13px;">
        <strong>Sesja NYSE:</strong> {session["label"]}
        <span style="color: #6b7280;">· {session["detail"]}</span>
      </div>
    """


def _session_text(data: SectionInput) -> list[str]:
    """Blok tytułowy wersji tekstowej — za bannerami, przed pierwszym ekranem."""
    session = market_status(data.started_at)
    regime_text = render_regime_text(data.context.regime)
    return [
        "=" * 64,
        "STOCKAGENT — RAPORT CYKLU",
        "=" * 64,
        f"Czas rozpoczęcia:  {data.started_local}",
        f"Sesja NYSE:        {session['label']} — {session['detail']}",
        *([regime_text] if regime_text else []),
        *_archive_link_text(data.context.archive_url),
        "",
    ]


def _archive_link_text(archive_url: str | None) -> list[str]:
    url = safe_http_url(archive_url)
    return [] if url is None else [f"{ARCHIVE_TEXT_LABEL}: {url}"]


# ---------------------------------------------------------------------------
# MAIL-16: pierwszy ekran — liczby dla czytelnika, liczniki do stopki
# ---------------------------------------------------------------------------

# Okno rozliczeń w boxie pierwszego ekranu — krócej niż w nagłówku sekcji.
_SETTLEMENT_WINDOW_SHORT = {
    RESOLVED_WINDOW_24H: "ostatnie 24h",
    RESOLVED_WINDOW_THIS_CYCLE: "w tym cyklu",
}


def _signals_above_threshold(saved: list[SymbolResult]) -> int:
    """MAIL-16: zapisane predykcje z kierunkiem (≠ SIDEWAYS) — sygnały, które
    przeszły bramkę volatility i coś mówią. „Bez ruchu” nie jest sygnałem."""
    return sum(1 for r in saved if r.trend is not None and r.trend != "SIDEWAYS")


@dataclasses.dataclass(frozen=True)
class _SettlementTally:
    """MAIL-16: „X/Y trafionych” z rozliczeń okna — albo jawny brak / awaria
    odczytu (MAIL-25: awaria nie może udawać „brak rozliczeń”)."""

    value: str
    note: str
    text: str


def _settlement_tally(
    resolved: list[ResolvedPrediction], *, unavailable: bool, window: str
) -> _SettlementTally:
    short_window = _SETTLEMENT_WINDOW_SHORT.get(window, window)
    if resolved:
        hits = f"{sum(1 for p in resolved if p.is_correct)}/{len(resolved)}"
        return _SettlementTally(
            value=hits,
            note=f"trafionych · {short_window}",
            text=f"{hits} trafionych ({short_window})",
        )
    if unavailable:
        return _SettlementTally(
            value="—", note="rozliczenia niedostępne", text="niedostępne (błąd odczytu)"
        )
    return _SettlementTally(
        value="—", note=f"brak rozliczeń · {short_window}", text=f"brak ({short_window})"
    )


def _render_reader_box_html(
    label: str, value: str, note: str, *, accent: str, background: str
) -> str:
    """MAIL-23: box pierwszego ekranu jako komórka tabeli `width: 33%` — same style
    inline, bo pierwszy ekran musi wyglądać tak samo bez arkusza."""
    return (
        f'<td style="width: 33%; padding: 12px; background: {background}; '
        f'border-left: 3px solid {accent}; border-radius: 4px; vertical-align: top;">'
        '<div style="font-size: 11px; color: #4b5563; text-transform: uppercase;">'
        f"{_html(label)}</div>"
        '<div style="font-size: 24px; font-weight: 600; color: #1f2937;">'
        f"{_html(value)}</div>"
        f'<div style="font-size: 12px; color: #6b7280;">{_html(note)}</div>'
        "</td>"
    )


_CREDIBILITY_TITLE = "Wiarygodność modelu"


def _credibility_label(accuracy_stats: Mapping[str, Any] | None) -> str | None:
    """MAIL-17: jednolinijkowy pasek wiarygodności — trafność kierunkowa i n z tych
    samych statystyk co „Historia trafności” w Zapleczu. Bez liczników kierunkowych
    → None (pasek się chowa)."""
    if not accuracy_stats:
        return None
    n = accuracy_stats.get("directional_count")
    hits = accuracy_stats.get("directional_correct_count")
    if n is None or hits is None:
        return None
    days = accuracy_stats.get("days_window")
    window = f", ostatnie {days} dni" if days is not None else ""
    if n == 0:
        return f"trafność kierunkowa — (brak prognoz kierunkowych{window})"
    return f"trafność kierunkowa {_pct(hits / n, decimals=1)} (n={n}{window})"


def _render_credibility_bar_html(accuracy_stats: Mapping[str, Any] | None) -> str:
    label = _credibility_label(accuracy_stats)
    if label is None:
        return ""
    return (
        '<div style="margin: -8px 0 16px 0; font-size: 12px; color: #4b5563;">'
        f"📏 <strong>{_CREDIBILITY_TITLE}:</strong> {_html(label)}"
        " · szczegóły w Zapleczu (Track Record)</div>"
    )


def _settlements(data: SectionInput) -> _SettlementTally:
    return _settlement_tally(
        data.resolved, unavailable=data.context.resolved_unavailable, window=data.resolved_window
    )


def _first_screen_html(data: SectionInput) -> str:
    """Trzy liczby dla czytelnika: zmiana portfela (tylko właściciel z pozycjami,
    PORT-06; inaczej box się chowa), sygnały ponad próg i trafność rozliczeń.
    Liczniki operacyjne są w stopce. Pod nimi pasek wiarygodności (MAIL-17)."""
    change = data.portfolio_change
    portfolio_change_box = (
        render_portfolio_change_box_html(
            change.change, label=change.label, usd_pln=data.usd_pln, currency=change.currency
        )
        if change is not None
        else ""
    )
    settlements = _settlements(data)
    signals_box = _render_reader_box_html(
        "Sygnały ponad próg",
        str(_signals_above_threshold(data.saved)),
        "wzrost lub spadek",
        accent="#16a34a",
        background="#f0fdf4",
    )
    settlements_box = _render_reader_box_html(
        "Rozliczone",
        settlements.value,
        settlements.note,
        accent="#2563eb",
        background="#eff6ff",
    )
    # MAIL-23: tabela zamiast `display: flex` — Outlook flexa nie zna, a trzy
    # komórki po 33% mieszczą się na telefonie bez łamania słów po literze.
    return (
        "<table role='presentation' style='width: 100%; border-collapse: separate; "
        "border-spacing: 8px 0; margin: 0 -8px 16px -8px;'><tr>"
        f"{portfolio_change_box}{signals_box}{settlements_box}</tr></table>"
    ) + _render_credibility_bar_html(data.context.accuracy_stats)


def _first_screen_text(data: SectionInput) -> list[str]:
    lines: list[str] = []
    change = data.portfolio_change
    # Stary push (Telegram/Slack) dostaje tekst bez kwot portfela.
    if change is not None and data.context.user_portfolio_in_text:
        portfolio_change_line = render_portfolio_change_text_line(
            change.change, label=change.label, usd_pln=data.usd_pln, currency=change.currency
        )
        if portfolio_change_line:
            lines.append(portfolio_change_line)
    lines.append(f"Sygnały ponad próg: {_signals_above_threshold(data.saved)}")
    lines.append(f"Rozliczone: {_settlements(data).text}")
    # MAIL-17: pasek wiarygodności — te same liczby co Historia trafności w Zapleczu.
    credibility = _credibility_label(data.context.accuracy_stats)
    if credibility is not None:
        lines.append(f"{_CREDIBILITY_TITLE}: {credibility} — szczegóły w Zapleczu")
    lines.append("")
    return lines


MASTHEAD_SECTION = ReportSection(key="masthead", render_html=_masthead_html, render_text=no_text)
SESSION_SECTION = ReportSection(key="session", render_html=_session_html, render_text=_session_text)
FIRST_SCREEN_SECTION = ReportSection(
    key="first_screen", render_html=_first_screen_html, render_text=_first_screen_text
)
