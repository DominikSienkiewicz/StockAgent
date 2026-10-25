"""Sekcja 🔮 Wygenerowane predykcje — tabela, wykres prognozy i uzasadnienia.

MAIL-06: pełny blok 💡 Uzasadnień dostaje top N symboli wg siły sygnału i symbole
z leadu, reszta idzie skrótem (jedna linia rady, zawsze widoczni dysydenci) —
Gmail przycina maile powyżej ~102 KB. Wersja tekstowa ma każde uzasadnienie
w całości. ALR-15: horyzont prognozy z kalendarza sesji NYSE.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.application.report_charts import build_forecast_chart_html
from src.application.report_formatting import (
    COLD_START_FORECAST as _COLD_START_FORECAST,
)
from src.application.report_formatting import (
    company_label as _company_label,
)
from src.application.report_formatting import (
    company_label_with_sector as _company_label_with_sector,
)
from src.application.report_formatting import direction_pct
from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_formatting import (
    money as _money,
)
from src.application.report_formatting import (
    number as _number,
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
from src.application.report_models import SymbolResult, TradeSignal
from src.application.report_reasoning import (
    invalidation_label,
    model_signal,
    model_signal_reason_text,
    move_interval_label,
    render_council_line,
    render_reasoning_block_html,
    render_valuation_text,
)
from src.application.report_signals import forecast_horizon_label
from src.application.report_styles import SIGNAL_TONE, delta_tone, trend_tone
from src.application.reporting.section import ReportSection, SectionInput
from src.domain.digest_lead import LeadItem
from src.domain.polish_labels import MODEL_SIGNAL_TITLE
from src.domain.provenance import ProvenanceLevel

# MAIL-06: tyle symboli (wg siły sygnału) dostaje pełny blok uzasadnień z radą.
FULL_DETAIL_TOP_N = 10


def full_detail_symbols(
    saved: list[SymbolResult],
    ranked_signals: list[TradeSignal],
    lead_items: list[LeadItem],
) -> frozenset[str] | None:
    """MAIL-06: symbole z pełnym blokiem — top N wg siły sygnału (kolejność
    „🎯 Najsilniejsze sygnały”) i symbole z leadu. Symbole bez sygnału (brak
    pewności albo prognozy) idą za rankingiem w kolejności cyklu. None = cykl
    mieści się w top N, więc pełny blok dostaje każdy symbol."""
    if len(saved) <= FULL_DETAIL_TOP_N:
        return None
    ranked = [sig.symbol for sig in ranked_signals]
    ranked_set = set(ranked)
    ranked.extend(r.symbol for r in saved if r.symbol not in ranked_set)
    lead = {item.symbol for item in lead_items if item.symbol is not None}
    return frozenset(ranked[:FULL_DETAIL_TOP_N]) | lead


def _render_prediction_row_html(r: SymbolResult) -> str:
    """Pojedynczy wiersz tabeli predykcji (saved). MAIL-23: pięć kolumn — cena
    bieżąca, sentyment, newsy i sygnał modelu są w bloku uzasadnień."""
    confidence_text = (
        _pct(r.confidence_score, decimals=0)
        if r.confidence_score is not None
        else "—"
    )
    if r.ml_cold_start:
        forecast_text = _COLD_START_FORECAST
    elif r.target_price is not None:
        forecast_text = (
            f"{_money(r.target_price)} "
            f"<span class='{delta_tone(r.expected_change)}'>"
            f"({direction_pct(r.expected_change)})</span>"
        )
    else:
        forecast_text = "—"
    # MAIL-06 (c): układ komórki i ton kierunku w klasach arkusza.
    return f"""
              <tr>
                <td class="pc b6">{_html(_company_label(r.symbol))}</td>
                <td class="pc {delta_tone(r.delta)}">{direction_pct(r.delta)}</td>
                <td class="pc b6 {trend_tone(r.trend)}">{_html(_trend_label(r.trend))}</td>
                <td class="pc">{forecast_text}</td>
                <td class="pc">{confidence_text}</td>
              </tr>
            """


def _render_saved_section_html(
    saved: list[SymbolResult],
    full_detail: frozenset[str] | None = None,
    *,
    sector_map: Mapping[str, str],
    forecast_horizon: str = "",
    symbol_chips: Mapping[str, str] | None = None,
    one_voice: bool = False,
) -> str:
    """🔮 Sekcja wygenerowanych predykcji: tabela + wykres prognozy +
    uzasadnienia. Pusty string, gdy w tym cyklu nic nie zapisano.

    `full_detail` (MAIL-06): symbole z pełnym blokiem uzasadnień; reszta trafia
    do skrótu (jedna linia rady, zawsze widoczni dysydenci). None = każdy symbol.
    `forecast_horizon` (ALR-15): najbliższe zamknięcie NYSE z kalendarza sesji."""
    if not saved:
        return ""
    parts = [
        _section_h2("🔮 Wygenerowane predykcje"),
        _render_forecast_horizon_html(forecast_horizon),
        "<table class='pt'>",
        """
          <tr class="pr">
            <th scope="col" class="ph">Spółka</th>
            <th scope="col" class="ph">Zmiana (cykl)</th>
            <th scope="col" class="ph">Kierunek</th>
            <th scope="col" class="ph">Prognoza (nast. cykl)</th>
            <th scope="col" class="ph">Pewność</th>
          </tr>
        """,
    ]
    parts.extend(_render_prediction_row_html(r) for r in saved)
    parts.append("</table>")

    # #2: wykres prognozy jako INLINE HTML — koniec z <img> z zewnętrznego hosta.
    # Fragment jest już zescapowany przez renderer, więc wstawiamy go wprost.
    forecast_chart = build_forecast_chart_html(saved)
    if forecast_chart:
        parts.append(forecast_chart)

    # Uzasadnienia
    parts.append("<h3 style='font-size: 14px; margin: 16px 0 8px 0;'>💡 Uzasadnienia</h3>")

    def has_full_block(r: SymbolResult) -> bool:
        return full_detail is None or r.symbol in full_detail

    chips = symbol_chips or {}
    parts.extend(
        render_reasoning_block_html(r, sector_map, chips.get(r.symbol), one_voice=one_voice)
        for r in saved
        if has_full_block(r)
    )
    parts.append(
        _render_compact_predictions_html([r for r in saved if not has_full_block(r)])
    )
    return "".join(parts)


_FORECAST_HORIZON_PREFIX = "Horyzont prognozy: następny cykl · "


def _render_forecast_horizon_html(forecast_horizon: str) -> str:
    """Linia horyzontu nad tabelą predykcji (ALR-15). Pusty string bez etykiety."""
    if not forecast_horizon:
        return ""
    return (
        "<div style='font-size: 12px; color: #6b7280; margin: 0 0 6px 0;'>"
        f"{_FORECAST_HORIZON_PREFIX}{_html(forecast_horizon)}</div>"
    )


_COMPACT_PREDICTIONS_NOTE = (
    f"Pełne uzasadnienie: top {FULL_DETAIL_TOP_N} sygnałów i lead. Reszta w skrócie "
    "(limit rozmiaru Gmaila) — pełne w wersji tekstowej maila."
)


def _render_compact_predictions_html(rest: list[SymbolResult]) -> str:
    """MAIL-06 (d): pozostałe predykcje jako zwarte wiersze — symbol, sygnał modelu
    i jedna linia rady. Pusty string, gdy wszystko ma pełny blok."""
    if not rest:
        return ""
    rows = []
    for r in rest:
        signal = model_signal(r)
        council = (
            render_council_line(r.council_verdict) if r.council_verdict is not None else "—"
        )
        rows.append(
            f'<tr><td class="pk b6">{_html(_company_label(r.symbol))}</td>'
            f'<td class="pk b7 {SIGNAL_TONE.get(signal, "nt")}">'
            f"{signal}</td>"
            f'<td class="pk">{council}</td></tr>'
        )
    return (
        "<h3 style='font-size: 14px; margin: 16px 0 8px 0;'>📋 Pozostałe predykcje (skrót)</h3>"
        f"<div class='kn'>{_html(_COMPACT_PREDICTIONS_NOTE)}</div>"
        "<table class='kt'>"
        '<tr class="pr">'
        '<th scope="col" class="pk">Symbol</th>'
        f'<th scope="col" class="pk">{MODEL_SIGNAL_TITLE}</th>'
        '<th scope="col" class="pk">Rada</th></tr>'
        + "".join(rows)
        + "</table>"
    )


def _render_prediction_text(
    r: SymbolResult, sector_map: Mapping[str, str], chip: str | None = None
) -> list[str]:
    """Linie jednej predykcji (plain): nagłówek + sentyment + sygnał modelu +
    trafność na symbolu (WIA-35) + proweniencja + reasoning + analogi + newsy + wycena."""
    conf_part = (
        f" · pewność {_pct(r.confidence_score, decimals=0)}"
        if r.confidence_score is not None
        else ""
    )
    if r.ml_cold_start:
        forecast_part = f"prognoza {_COLD_START_FORECAST}"
    elif r.target_price is not None and r.expected_change is not None:
        forecast_part = (
            f"prognoza {_money(r.target_price)} ({_pct(r.expected_change, signed=True)})"
        )
    else:
        forecast_part = "prognoza —"
    sentiment_part = (
        f"sentyment {_number(r.sentiment_score, decimals=2, signed=True)}"
        if r.sentiment_score is not None
        else "sentyment —"
    )
    rec_text = model_signal_reason_text(r)
    lines = [
        f"  {_company_label_with_sector(r.symbol, sector_map):55s} {_money(r.current_price):>10s}  "
        f"Δ {_pct(r.delta, signed=True):>8s}  →  "
        f"{_trend_label(r.trend):11s} {forecast_part}{conf_part}",
        f"         {sentiment_part}",
        f"         {rec_text}",
    ]
    if r.move_interval is not None:
        lines.append(f"         {move_interval_label(r.move_interval)}")
    invalidation = invalidation_label(r)
    if invalidation is not None:
        lines.append(f"         🛑 {invalidation}")
    if chip:
        lines.append(f"         {chip}")
    # Q6: odznaki proweniencji (pomijamy FRESH — brak szumu).
    badge_labels = [
        b.label for b in r.provenance_badges
        if b.level is not ProvenanceLevel.FRESH
    ]
    if badge_labels:
        lines.append(f"         proweniencja: {', '.join(badge_labels)}")
    if r.reasoning:
        lines.append(f"        └ {r.reasoning}")
    # Q5: precedent receipts (analogi RAG).
    for prec in r.similar_precedents:
        if prec.is_trend_correct:
            outcome = "trafił"
        elif prec.is_trend_correct is False:
            outcome = "chybił"
        else:
            outcome = "wynik nieznany"
        lines.append(
            f"        🧭 analog [{_trend_label(prec.predicted_trend)}, {outcome}]: "
            f"{prec.summary[:72]}"
        )
    for n in r.top_news:
        lines.append(
            f"        📰 [{n.source or '?'}] {n.title[:78]}"
            f" (rel={_number(n.relevance, decimals=2)}, "
            f"sent={_number(n.sentiment, decimals=2, signed=True)})"
        )
    valuation_text = render_valuation_text(r.valuation)
    if valuation_text:
        lines.append(valuation_text)
    return lines


def _render_saved_section_text(
    saved: list[SymbolResult],
    sector_map: Mapping[str, str],
    forecast_horizon: str = "",
    symbol_chips: Mapping[str, str] | None = None,
) -> list[str]:
    """Sekcja 'Predykcje' (plain). Pusta lista, gdy nic nie zapisano."""
    if not saved:
        return []
    lines = ["PREDYKCJE", "-" * 64]
    if forecast_horizon:
        lines.append(f"  {_FORECAST_HORIZON_PREFIX}{forecast_horizon}")
    chips = symbol_chips or {}
    for r in saved:
        lines.extend(_render_prediction_text(r, sector_map, chips.get(r.symbol)))
    lines.append("")
    return lines


def _section_html(data: SectionInput) -> str:
    return _render_saved_section_html(
        data.saved,
        data.full_detail,
        sector_map=data.sector_map,
        forecast_horizon=forecast_horizon_label(data.started_at, data.context.display_timezone),
        symbol_chips=data.symbol_chips,
        one_voice=_council_is_one_voice(data),
    )


def _section_text(data: SectionInput) -> list[str]:
    return _render_saved_section_text(
        data.saved,
        data.sector_map,
        forecast_horizon_label(data.started_at, data.context.display_timezone),
        data.symbol_chips,
    )


def _council_is_one_voice(data: SectionInput) -> bool:
    """WIA-45: zmierzona redundancja rady (flaga) mówi, że to jeden efektywny głos."""
    redundancy = data.context.council_redundancy
    return redundancy is not None and redundancy.is_effectively_single_voter


PREDICTIONS_SECTION = ReportSection(
    key="predictions",
    render_html=_section_html,
    render_text=_section_text,
    context_fields=frozenset({"council_redundancy"}),
)
