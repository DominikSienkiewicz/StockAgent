"""MAIL-35: zwarty mail cichego dnia — kilka linijek zamiast pełnego raportu.

W dzień, w którym cykl nie ma nic ponad stan (`src/domain/digest_policy.py`), mail
z polityką `compact_when_quiet` składa się z nagłówka, bannerów, statusu sesji,
tego bloku i pełnej stopki (`COMPACT_REPORT_SECTIONS`). Blok mówi trzy rzeczy:
największy ruch z jego progiem, że żaden symbol nie przekroczył progu, i stan
portfela właściciela — albo, bez pozycji, średni ruch watchlisty (równe wagi, więc
„watchlista”, nie „portfel”). Sekcja istnieje tylko w zwartym mailu i czyta
wyłącznie wspólne wejście sekcji.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.application.report_formatting import (
    PALETTE,
    delta_color,
    direction_pct,
    plural_pl,
)
from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_formatting import (
    pct as _pct,
)
from src.application.report_models import SymbolResult
from src.application.report_user_portfolio import render_portfolio_change_text_line
from src.application.reporting.section import ReportSection, SectionInput

_TITLE = "🌙 Cichy dzień — nic ponad stan"
_TEXT_TITLE = "CICHY DZIEŃ — NIC PONAD STAN"
_FOOTNOTE = "Pełny raport wróci z pierwszym sygnałem, rozliczeniem albo szokiem."


@dataclass(frozen=True)
class _QuietDayLines:
    """Linie bloku w obu wersjach maila; `largest` niesie ruch do koloru w HTML."""

    largest: SymbolResult | None
    largest_line: str
    gate_line: str
    state_line: str


def _largest_move(results: list[SymbolResult]) -> SymbolResult | None:
    moved = [r for r in results if r.delta is not None]
    return max(moved, key=lambda r: abs(r.delta or Decimal(0)), default=None)


def _threshold_suffix(r: SymbolResult) -> str:
    if r.effective_threshold is None:
        return ""
    return f" · próg {_pct(r.effective_threshold, decimals=1)}"


def _largest_line(largest: SymbolResult | None) -> str:
    if largest is None:
        return "Największy ruch: brak notowań w tym cyklu"
    return f"Największy ruch: {largest.symbol} {_pct(largest.delta, signed=True)}" + (
        _threshold_suffix(largest)
    )


def _gate_line(results: list[SymbolResult]) -> str:
    count = len(results)
    symbols = plural_pl(count, "symbolu", "symboli", "symboli")
    return f"Żaden z {count} {symbols} nie przekroczył progu — cykl bez płatnej analizy."


def _state_line(data: SectionInput) -> str:
    """Portfel właściciela (PORT-06) albo średni ruch watchlisty z równych wag."""
    change = data.portfolio_change
    if change is not None and data.context.user_portfolio_in_text:
        return render_portfolio_change_text_line(
            change.change, label=change.label, usd_pln=data.usd_pln, currency=change.currency
        )
    deltas = [r.delta for r in data.results if r.delta is not None]
    if not deltas:
        return "Watchlista: brak notowań w tym cyklu"
    mean = sum(deltas, Decimal(0)) / len(deltas)
    return f"Watchlista (równe wagi): średnio {_pct(mean, signed=True)} od poprzedniego cyklu"


def _lines(data: SectionInput) -> _QuietDayLines:
    largest = _largest_move(data.results)
    return _QuietDayLines(
        largest=largest,
        largest_line=_largest_line(largest),
        gate_line=_gate_line(data.results),
        state_line=_state_line(data),
    )


def _largest_html(lines: _QuietDayLines) -> str:
    largest = lines.largest
    if largest is None:
        return _html(lines.largest_line)
    return (
        f"Największy ruch: <strong>{_html(largest.symbol)}</strong> "
        f"<span style='color: {delta_color(largest.delta)};'>{direction_pct(largest.delta)}</span>"
        f"{_html(_threshold_suffix(largest))}"
    )


def _section_html(data: SectionInput) -> str:
    lines = _lines(data)
    return (
        f"<div style='padding: 12px 16px; background: {PALETTE.surface}; "
        f"border-left: 3px solid {PALETTE.muted}; border-radius: 4px; "
        "margin-bottom: 16px; font-size: 13px;'>"
        f"<div style='font-size: 15px; font-weight: 600; margin-bottom: 6px;'>{_TITLE}</div>"
        f"<div>{_largest_html(lines)}</div>"
        f"<div>{_html(lines.gate_line)}</div>"
        f"<div>{_html(lines.state_line)}</div>"
        f"<div style='color: {PALETTE.muted}; font-size: 12px; margin-top: 6px;'>"
        f"{_html(_FOOTNOTE)}</div>"
        "</div>"
    )


def _section_text(data: SectionInput) -> list[str]:
    lines = _lines(data)
    return [
        _TEXT_TITLE,
        "-" * 64,
        f"  {lines.largest_line}",
        f"  {lines.gate_line}",
        f"  {lines.state_line}",
        f"  {_FOOTNOTE}",
        "",
    ]


# Tylko w `COMPACT_REPORT_SECTIONS` — pełny mail mówi to samo swoimi sekcjami.
QUIET_DAY_SECTION = ReportSection(
    key="quiet_day", render_html=_section_html, render_text=_section_text
)
