"""Sekcje cyklu poza analizą: nowy punkt odniesienia, budżet, pominięte, błędy.

- `rebaselined` (FIN-14, „Na dziś”) — referencja sprzed pauzy: cykl zapisał
  nową cenę bez analizy, prognozy wracają od następnego cyklu.
- `budget_exhausted` (FIN-12, Zaplecze) — ruch przeszedł bramkę, ale cykl
  wyczerpał sufit płatnych wywołań LLM.
- `ignored` (Zaplecze) — poniżej progu zmienności, bez dwóch grup powyżej.
- `errors` (Zaplecze) — symbole, które padły w cyklu.

Etykiety statusu (`rebaseline_label`, `BUDGET_CUT_*`) bierze też sekcja 🪙 Krypto.
"""

from __future__ import annotations

from datetime import datetime

from src.application.report_formatting import (
    company_label as _company_label,
)
from src.application.report_formatting import (
    delta_color as _delta_color,
)
from src.application.report_formatting import (
    direction_pct,
    format_local_date,
)
from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_formatting import (
    pct as _pct,
)
from src.application.report_formatting import (
    section_h2 as _section_h2,
)
from src.application.report_models import SymbolResult
from src.application.reporting.section import ReportSection, SectionInput
from src.domain.asset import REFERENCE_MAX_AGE_HOURS
from src.domain.cycle_maturity import SkipReason

# ---------------------------------------------------------------------------
# FIN-14: nowy punkt odniesienia
# ---------------------------------------------------------------------------


def is_rebaselined(r: SymbolResult) -> bool:
    """FIN-14: referencja sprzed pauzy — cykl zapisał nowy punkt odniesienia.
    Bez daty referencji nie ma czego napisać, więc symbol zostaje w „Pominięte”."""
    return r.skip_reason is SkipReason.STALE_REFERENCE and r.reference_at is not None


_REBASELINE_NOTE = (
    "Ostatnia zapisana cena była starsza niż {hours} h (pauza), więc ruch od niej "
    "nie jest ruchem cyklu. Ten cykl zapisał nową cenę bez analizy — prognozy "
    "wracają od następnego cyklu."
)


def _rebaseline_lines(ignored: list[SymbolResult], display_timezone: str) -> list[str]:
    """„AAPL, MSFT: referencja sprzed 19.09.2026 — nowy punkt odniesienia”,
    jedna linia na dzień referencji (rosnąco), symbole w kolejności cyklu."""
    by_day: dict[str, list[str]] = {}
    oldest: dict[str, datetime] = {}
    for r in ignored:
        if not is_rebaselined(r) or r.reference_at is None:
            continue
        day = format_local_date(r.reference_at, display_timezone)
        by_day.setdefault(day, []).append(r.symbol)
        oldest[day] = min(oldest.get(day, r.reference_at), r.reference_at)
    return [
        f"{', '.join(by_day[day])}: {rebaseline_label(day)}"
        for day in sorted(by_day, key=lambda d: oldest[d])
    ]


def rebaseline_label(day: str) -> str:
    return f"referencja sprzed {day} — nowy punkt odniesienia"


def _render_rebaselined_html(ignored: list[SymbolResult], display_timezone: str) -> str:
    """🔄 FIN-14: symbole z referencją sprzed pauzy. Pusty string, gdy brak."""
    lines = _rebaseline_lines(ignored, display_timezone)
    if not lines:
        return ""
    rows = "".join(
        f"<div style='font-size: 13px; margin: 2px 0;'>{_html(line)}</div>" for line in lines
    )
    return (
        _section_h2("🔄 Nowy punkt odniesienia")
        + "<div style='font-size: 12px; color: #6b7280; margin: 0 0 6px 0;'>"
        + _html(_REBASELINE_NOTE.format(hours=REFERENCE_MAX_AGE_HOURS))
        + f"</div>{rows}"
    )


def _render_rebaselined_text(ignored: list[SymbolResult], display_timezone: str) -> list[str]:
    """Sekcja 'Nowy punkt odniesienia' (plain). Pusta lista, gdy brak."""
    lines = _rebaseline_lines(ignored, display_timezone)
    if not lines:
        return []
    return [
        "NOWY PUNKT ODNIESIENIA",
        "-" * 64,
        *(f"  {line}" for line in lines),
        f"  {_REBASELINE_NOTE.format(hours=REFERENCE_MAX_AGE_HOURS)}",
        "",
    ]


# ---------------------------------------------------------------------------
# FIN-12: wyczerpany budżet cyklu
# ---------------------------------------------------------------------------


def is_budget_cut(r: SymbolResult) -> bool:
    """FIN-12: ruch przeszedł bramkę, ale cykl wyczerpał sufit płatnych wywołań LLM."""
    return r.skip_reason is SkipReason.BUDGET_EXHAUSTED


_BUDGET_CUT_TITLE = "Pominięte — wyczerpany budżet cyklu"
BUDGET_CUT_LABEL = "pominięte — wyczerpany budżet płatnych wywołań cyklu"
BUDGET_CUT_SHORT = "budżet cyklu wyczerpany"
_BUDGET_CUT_NOTE = (
    "Ruch przekroczył próg zmienności, ale cykl wyczerpał limit płatnych wywołań "
    "LLM, więc te symbole nie dostały analizy. Cena odniesienia zostaje bez zmian — "
    "następny cykl porówna się do niej."
)


def _render_budget_exhausted_html(ignored: list[SymbolResult]) -> str:
    """💸 FIN-12: symbole odcięte budżetem cyklu, z ruchem. Pusty string, gdy brak."""
    cut = [r for r in ignored if is_budget_cut(r)]
    if not cut:
        return ""
    rows = "".join(
        "<span style='display: inline-block; padding: 2px 8px; margin: 2px; "
        "background: #fef3c7; border-radius: 3px; font-size: 12px;'>"
        f"<strong>{_html(r.symbol)}</strong> "
        f"<span style='color: {_delta_color(r.delta)}'>{direction_pct(r.delta)}</span></span>"
        for r in cut
    )
    return (
        _section_h2(f"💸 {_BUDGET_CUT_TITLE}")
        + "<div style='font-size: 12px; color: #6b7280; margin: 0 0 6px 0;'>"
        + _html(_BUDGET_CUT_NOTE)
        + f"</div><div>{rows}</div>"
    )


def _render_budget_exhausted_text(ignored: list[SymbolResult]) -> list[str]:
    """Sekcja FIN-12 'Pominięte — wyczerpany budżet cyklu' (plain). Pusta lista, gdy brak."""
    cut = [r for r in ignored if is_budget_cut(r)]
    if not cut:
        return []
    cut_str = ", ".join(f"{r.symbol}({_pct(r.delta)})" for r in cut)
    return [_BUDGET_CUT_TITLE.upper(), "-" * 64, f"  {cut_str}", f"  {_BUDGET_CUT_NOTE}", ""]


# ---------------------------------------------------------------------------
# Pominięte poniżej progu i błędy
# ---------------------------------------------------------------------------


def _render_ignored_html(ignored: list[SymbolResult]) -> str:
    """⏸ Symbole pominięte poniżej progu zmienności. Pusty string, gdy brak.
    FIN-14: symbole z nowym punktem odniesienia mają własną sekcję; FIN-12: symbole
    odcięte budżetem cyklu też."""
    ignored = [r for r in ignored if not is_rebaselined(r) and not is_budget_cut(r)]
    if not ignored:
        return ""
    rows = "".join(
        "<span style='display: inline-block; padding: 2px 8px; margin: 2px; "
        "background: #f3f4f6; border-radius: 3px; font-size: 12px;'>"
        f"<strong>{_html(r.symbol)}</strong> "
        f"<span style='color: {_delta_color(r.delta)}'>{direction_pct(r.delta)}</span></span>"
        for r in ignored
    )
    return (
        _section_h2("⏸ Pominięte (poniżej progu zmienności)")
        + f"<div>{rows}</div>"
    )


def _render_ignored_text(ignored: list[SymbolResult]) -> list[str]:
    """Sekcja 'Pominięte' (plain). Pusta lista, gdy brak. FIN-14: bez symboli
    z nowym punktem odniesienia (mają własną sekcję); FIN-12: bez odciętych budżetem."""
    ignored = [r for r in ignored if not is_rebaselined(r) and not is_budget_cut(r)]
    if not ignored:
        return []
    ignored_str = ", ".join(f"{r.symbol}({_pct(r.delta)})" for r in ignored)
    return ["POMINIĘTE (poniżej progu zmienności)", "-" * 64, f"  {ignored_str}", ""]


_ERROR_ROW_STYLE = (
    "margin-bottom: 8px; padding: 10px; background: #fef2f2; "
    "border-left: 3px solid #dc2626; border-radius: 4px; font-size: 12px;"
)


def _render_errors_html(errors: list[SymbolResult]) -> str:
    """⚠️ Symbole, które padły w cyklu. Pusty string, gdy brak błędów."""
    if not errors:
        return ""
    parts = [
        "<h2 style='font-size: 16px; margin: 20px 0 8px 0; color: #991b1b;'>"
        "⚠️ Błędy</h2>"
    ]
    for r in errors:
        parts.append(f"""
              <div style="{_ERROR_ROW_STYLE}">
                <strong>{_html(_company_label(r.symbol))}</strong>: {_html(r.error_message)}
              </div>
            """)
    return "".join(parts)


def _render_errors_text(errors: list[SymbolResult]) -> list[str]:
    """Sekcja 'Błędy' (plain). Pusta lista, gdy brak błędów."""
    if not errors:
        return []
    lines = ["BŁĘDY", "-" * 64]
    lines.extend(f"  {_company_label(r.symbol)}: {r.error_message}" for r in errors)
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Sekcje rejestru (ARCH-12)
# ---------------------------------------------------------------------------


def _rebaselined_html(data: SectionInput) -> str:
    return _render_rebaselined_html(data.ignored, data.context.display_timezone)


def _rebaselined_text(data: SectionInput) -> list[str]:
    return _render_rebaselined_text(data.ignored, data.context.display_timezone)


def _budget_exhausted_html(data: SectionInput) -> str:
    return _render_budget_exhausted_html(data.ignored)


def _budget_exhausted_text(data: SectionInput) -> list[str]:
    return _render_budget_exhausted_text(data.ignored)


def _ignored_html(data: SectionInput) -> str:
    return _render_ignored_html(data.ignored)


def _ignored_text(data: SectionInput) -> list[str]:
    return _render_ignored_text(data.ignored)


def _errors_html(data: SectionInput) -> str:
    return _render_errors_html(data.errors)


def _errors_text(data: SectionInput) -> list[str]:
    return _render_errors_text(data.errors)


REBASELINED_SECTION = ReportSection(
    key="rebaselined", render_html=_rebaselined_html, render_text=_rebaselined_text
)
BUDGET_EXHAUSTED_SECTION = ReportSection(
    key="budget_exhausted", render_html=_budget_exhausted_html, render_text=_budget_exhausted_text
)
IGNORED_SECTION = ReportSection(key="ignored", render_html=_ignored_html, render_text=_ignored_text)
ERRORS_SECTION = ReportSection(key="errors", render_html=_errors_html, render_text=_errors_text)
