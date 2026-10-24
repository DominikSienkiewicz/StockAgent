"""Sekcja 📊 Zamknięte predykcje — rozliczenia z okna, strefa „Na dziś”.

Okno to ostatnie 24 h albo (WIA-19) predykcje rozliczone w tym cyklu. Udany
odczyt bez rozliczeń mówi „brak rozliczeń”, a awaria odczytu ostrzega o sobie
zamiast udawać pustkę (MAIL-25). Pod chybieniem idą diagnoza z reflect i linia
kwitu decyzyjnego (WIA-56).
"""

from __future__ import annotations

from collections.abc import Mapping

from src.application.decision_receipt import receipt_line_pl
from src.application.report_accuracy_history import (
    DIRECTIONAL_ACCURACY_LABEL,
    TOTAL_ACCURACY_LABEL,
)
from src.application.report_formatting import (
    PALETTE,
    direction_pct,
)
from src.application.report_formatting import (
    company_label_with_sector as _company_label_with_sector,
)
from src.application.report_formatting import (
    delta_color as _delta_color,
)
from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_formatting import (
    money as _money,
)
from src.application.report_formatting import (
    pct as _pct,
)
from src.application.report_formatting import (
    section_h2 as _section_h2,
)
from src.application.report_formatting import (
    trend_label as _trend_label,
)
from src.application.report_models import ResolvedPrediction
from src.application.reporting.section import ReportSection, SectionInput
from src.domain.equity_curve import directional_hit_rate
from src.domain.prediction import is_substantive_insight

# WIA-19: okno sekcji zamkniętych predykcji w jej nagłówku (HTML i tekst).
RESOLVED_WINDOW_24H = "ostatnie 24h"
RESOLVED_WINDOW_THIS_CYCLE = "rozliczone w tym cyklu"
# MAIL-25: linia zamiast pustego miejsca — zgodna z oknem w nagłówku sekcji.
_RESOLVED_EMPTY_NOTE = {
    RESOLVED_WINDOW_24H: "Brak rozliczeń predykcji z ostatnich 24h.",
    RESOLVED_WINDOW_THIS_CYCLE: "Brak rozliczeń w tym cyklu.",
}
_RESOLVED_UNAVAILABLE_NOTE = (
    "⚠️ Nie udało się odczytać rozliczeń — w tym mailu ich brakuje "
    "(szczegóły w logach przebiegu)."
)


def resolved_window(this_cycle: bool) -> str:
    """WIA-19: okno sekcji — rozliczone w tym cyklu albo ostatnie 24 h."""
    return RESOLVED_WINDOW_THIS_CYCLE if this_cycle else RESOLVED_WINDOW_24H


def _resolved_why(p: ResolvedPrediction) -> str:
    """Zwięzłe 'dlaczego' werdyktu: dla chybionych — diagnoza z reflect,
    dla trafionych — potwierdzenie tezy. Placeholdery reflect_node („Trafiona
    predykcja.”, „diagnoza pominięta/odłożona”) nie są diagnozą, więc ich nie
    pokazujemy dosłownie (WIA-10)."""
    if p.is_correct:
        return "Teza prognozy się potwierdziła."
    if p.insight is not None and is_substantive_insight(p.insight):
        return p.insight
    return "Kierunek się nie potwierdził."


def _resolved_receipt_line(p: ResolvedPrediction) -> str | None:
    """WIA-56: pod chybieniem jedna linia z kwitu — na czym predykcja stała.
    Trafiona nie potrzebuje audytu, a brak kwitu to brak linii."""
    if p.is_correct or p.receipt is None:
        return None
    return receipt_line_pl(p.receipt)


def _render_resolved_item_html(p: ResolvedPrediction, sector_map: Mapping[str, str]) -> str:
    mark = "✅" if p.is_correct else "❌"
    color = PALETTE.positive if p.is_correct else PALETTE.negative
    verdict = "Trafiona" if p.is_correct else "Błędna"
    reason_line = (
        f"<div style='font-size: 11px; color: #4b5563; margin-top: 3px;'>"
        f"<strong>Prognoza ({_html(_trend_label(p.predicted_trend))})</strong>, bo: "
        f"{_html(p.reasoning)}</div>"
        if p.reasoning
        else ""
    )
    move_line = ""
    if p.actual_change_pct is not None:
        move_txt = direction_pct(p.actual_change_pct)
        prices = (
            f"{_money(p.price_at_prediction)} → {_money(p.actual_price)} "
            if p.price_at_prediction is not None and p.actual_price is not None
            else ""
        )
        move_line = (
            f"<div style='font-size: 11px; color: #4b5563; margin-top: 2px;'>"
            f"Faktycznie: {prices}"
            f"<span style='color: {_delta_color(p.actual_change_pct)};'>"
            f"({move_txt})</span></div>"
        )
    why_line = (
        f"<div style='font-size: 11px; margin-top: 2px;'>"
        f"<strong style='color: {color};'>Dlaczego:</strong> "
        f"<span style='color: #4b5563;'>{_html(_resolved_why(p))}</span></div>"
    )
    receipt = _resolved_receipt_line(p)
    if receipt:
        why_line += (
            "<div style='font-size: 11px; color: #4b5563; margin-top: 2px;'>"
            f"🧾 <strong>Kwit:</strong> {_html(receipt)}</div>"
        )
    return f"""
      <div style="margin-bottom: 8px; padding: 8px 10px; background: #fafafa;
                  border-left: 3px solid {color}; border-radius: 4px;
                  font-size: 12px;">
        {mark} <strong>{_html(_company_label_with_sector(p.symbol, sector_map))}</strong>
        <span style="color: {color}; font-weight: 600;">· {verdict}</span>
        {reason_line}
        {move_line}
        {why_line}
      </div>
    """


def _resolved_note(data: SectionInput) -> str | None:
    """MAIL-25: linia sekcji zamkniętych predykcji, gdy nie ma czego wylistować.

    Awaria odczytu ma pierwszeństwo przed pustą listą — nie może udawać „brak
    rozliczeń”. None w `resolved_predictions` (wołający nie dociągnął rozliczeń)
    → brak linii, czyli brak sekcji, jak przed MAIL-25.
    """
    context = data.context
    if context.resolved_predictions:
        return None
    if context.resolved_unavailable:
        return _RESOLVED_UNAVAILABLE_NOTE
    if context.resolved_predictions is not None:
        return _RESOLVED_EMPTY_NOTE[data.resolved_window]
    return None


def _render_resolved_predictions_html(
    resolved_predictions: list[ResolvedPrediction],
    window: str = RESOLVED_WINDOW_24H,
    note: str | None = None,
    *,
    sector_map: Mapping[str, str],
) -> str:
    """📊 Zamknięte predykcje z okna `window`. Bez predykcji: nagłówek z linią
    `note` (MAIL-25) albo pusty string, gdy `note` None."""
    if not resolved_predictions:
        if note is None:
            return ""
        return _section_h2(f"📊 Zamknięte predykcje ({window})") + (
            "<div style='font-size: 12px; color: #6b7280; margin: 4px 0 16px 0;'>"
            f"{_html(note)}</div>"
        )
    correct = [p for p in resolved_predictions if p.is_correct]
    wrong = [p for p in resolved_predictions if not p.is_correct]
    parts = [_section_h2(f"📊 Zamknięte predykcje ({window})")]
    parts.extend(_render_resolved_item_html(p, sector_map) for p in resolved_predictions)
    parts.append(
        "<div style='font-size: 11px; color: #6b7280; margin: 4px 0 16px 0;'>"
        f"Suma: {len(correct)} trafionych / {len(wrong)} błędnych · "
        f"{_resolved_accuracy_pair(resolved_predictions)}"
        "</div>"
    )
    return "".join(parts)


def _resolved_accuracy_pair(resolved_predictions: list[ResolvedPrediction]) -> str:
    """Kierunkowa obok łącznej — te same etykiety co Historia trafności (WIA-29)."""
    directional = directional_hit_rate(
        (p.predicted_trend, p.is_correct) for p in resolved_predictions
    )
    directional_txt = (
        "— (brak prognoz kierunkowych)"
        if directional.rate is None
        else f"{_pct(directional.rate, decimals=0)} (n={directional.directional})"
    )
    total = len(resolved_predictions)
    correct = sum(1 for p in resolved_predictions if p.is_correct)
    return (
        f"{DIRECTIONAL_ACCURACY_LABEL}: {directional_txt} · "
        f"{TOTAL_ACCURACY_LABEL}: {_pct(correct / total, decimals=0)} (n={total})"
    )


def _render_resolved_predictions_text(
    resolved_predictions: list[ResolvedPrediction],
    window: str = RESOLVED_WINDOW_24H,
    note: str | None = None,
    *,
    sector_map: Mapping[str, str],
) -> list[str]:
    """Sekcja 'Zamknięte predykcje' (plain). Bez predykcji: nagłówek z linią
    `note` (MAIL-25) albo pusta lista, gdy `note` None."""
    if not resolved_predictions:
        if note is None:
            return []
        return [f"ZAMKNIĘTE PREDYKCJE ({window})", "-" * 64, f"  {note}", ""]
    correct = sum(1 for p in resolved_predictions if p.is_correct)
    lines = [f"ZAMKNIĘTE PREDYKCJE ({window})", "-" * 64]
    for p in resolved_predictions:
        mark = "✅" if p.is_correct else "❌"
        verdict = "Trafiona" if p.is_correct else "Błędna"
        lines.append(
            f"  {mark} {_company_label_with_sector(p.symbol, sector_map)}  "
            f"prognoza {_trend_label(p.predicted_trend)} — {verdict}"
        )
        if p.reasoning:
            lines.append(f"        bo: {p.reasoning}")
        if p.actual_change_pct is not None:
            prices = (
                f"{_money(p.price_at_prediction)} → {_money(p.actual_price)} "
                if p.price_at_prediction is not None and p.actual_price is not None
                else ""
            )
            lines.append(
                f"        faktycznie: {prices}({_pct(p.actual_change_pct, signed=True)})"
            )
        lines.append(f"        dlaczego: {_resolved_why(p)}")
        receipt = _resolved_receipt_line(p)
        if receipt:
            lines.append(f"        kwit: {receipt}")
    lines.append(
        f"  Suma: {correct}/{len(resolved_predictions)} · "
        f"{_resolved_accuracy_pair(resolved_predictions)}"
    )
    lines.append("")
    return lines


def _section_html(data: SectionInput) -> str:
    return _render_resolved_predictions_html(
        data.resolved, data.resolved_window, _resolved_note(data), sector_map=data.sector_map
    )


def _section_text(data: SectionInput) -> list[str]:
    return _render_resolved_predictions_text(
        data.resolved, data.resolved_window, _resolved_note(data), sector_map=data.sector_map
    )


RESOLVED_PREDICTIONS_SECTION = ReportSection(
    key="resolved_predictions", render_html=_section_html, render_text=_section_text
)
