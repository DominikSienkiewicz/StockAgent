"""Buduje raport HTML+text z wyników cyklu Fast Loop.

Czysta logika prezentacji — bez I/O, łatwo testowalne.
"""

from __future__ import annotations

import dataclasses
import functools
import html
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from src.application.alpha_signals import AlphaSignals
from src.application.report_alpha import render_alpha_html
from src.application.report_charts import (
    build_correlation_chart_html,
    build_delta_chart_html,
    build_forecast_chart_html,
)
from src.application.report_consensus_shift import (
    render_consensus_shift_html,
)
from src.application.report_council_history import (
    InvestorHistory,
    render_council_history_html,
)
from src.application.report_evidence import wilson_label
from src.application.report_finops import render_finops_html
from src.application.report_formatting import (
    DEFAULT_DISPLAY_TIMEZONE,
    INVESTMENT_DISCLAIMER,
    PALETTE,
    direction_pct,
    format_local,
    format_local_date,
)
from src.application.report_formatting import (
    company_label as _company_label,
)
from src.application.report_formatting import (
    company_label_with_sector as _company_label_with_sector,
)
from src.application.report_formatting import (
    delta_color as _delta_color,
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
    provenance_badges_html as _provenance_badges_html,
)
from src.application.report_formatting import (
    sector_label as _sector_label,
)
from src.application.report_formatting import (
    sentiment_label as _sentiment_label,
)
from src.application.report_formatting import (
    trend_color as _trend_color,
)
from src.application.report_formatting import (
    trend_label as _trend_label,
)
from src.application.report_lead import render_lead_html, render_lead_text
from src.application.report_lessons import LessonsReport
from src.application.report_model_scorecard import (
    ModelScorecardView,
    render_model_scorecard_html,
)
from src.application.report_models import (
    ResolvedPrediction,
    RiskSignal,
    SimilarPrecedent,
    SymbolResult,
    TopNewsItem,
    TradeSignal,
    ValuationSection,
)
from src.application.report_onboarding import (
    render_onboarding_html,
    render_onboarding_text,
)
from src.application.report_persona_leaderboard import (
    render_persona_leaderboard_html,
)
from src.application.report_portfolio import (
    render_correlation_html,
    render_correlation_text,
)
from src.application.report_quota_banner import (
    render_quota_banner_html,
    render_quota_banner_text,
)
from src.application.report_risk_watch import (
    render_risk_watch_html,
    render_risk_watch_text,
)
from src.application.report_signals import (
    build_portfolio_mood,
    build_trade_signals,
    detect_risk_signals,
    market_status,
    parse_resolved_predictions,
)
from src.application.report_styles import (
    COUNCIL_FILL,
    COUNCIL_ROW_TONE,
    COUNCIL_TONE,
    EDGE_TONE,
    EMAIL_STYLESHEET,
    SIGNAL_TONE,
    delta_tone,
    trend_tone,
)
from src.application.report_suggestions import (
    build_sector_suggestions,
    render_suggestions_html,
    render_suggestions_text,
)
from src.application.report_templates import _env, render_template
from src.application.report_track_record import (
    TrackRecordStats,
    render_track_record_html,
)
from src.application.report_user_portfolio import (
    portfolio_change_label,
    render_portfolio_change_box_html,
    render_portfolio_change_text_line,
    render_user_portfolio_html,
    render_user_portfolio_text,
)
from src.application.run_context import RunContext
from src.application.use_cases.monitor_macro_risk import MacroRiskReport
from src.application.use_cases.portfolio_risk import PortfolioRiskReport, PositionRisk
from src.domain.asset import REFERENCE_MAX_AGE_HOURS, PriceDelta
from src.domain.calibration_curve import CalibrationBucket
from src.domain.conformal import MoveInterval
from src.domain.consensus_shift import ConsensusShift
from src.domain.council import CouncilVerdict
from src.domain.cycle_maturity import CycleMaturity, SkipReason
from src.domain.digest_lead import LeadItem, LeadSignal, build_lead, lead_preview
from src.domain.equity_curve import EquityCurve, directional_hit_rate
from src.domain.feature_attribution import FeatureContribution
from src.domain.finops import CycleCost
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.persona_track_record import PersonaTrackRecord
from src.domain.polish_labels import (
    MODEL_SIGNAL_TITLE,
    council_recommendation_label,
    model_signal_label,
    trend_forecast_label,
)
from src.domain.portfolio import Portfolio, PortfolioChange
from src.domain.prediction import is_substantive_insight
from src.domain.provenance import ProvenanceLevel, build_provenance_badges
from src.domain.quota import QuotaAlert
from src.domain.value_objects import (
    FUNDAMENTALS_CACHE_TTL_HOURS,
    ValuationVerdict,
)

__all__ = [
    "ResolvedPrediction",
    "RiskSignal",
    "SimilarPrecedent",
    "SymbolResult",
    "TopNewsItem",
    "TradeSignal",
    "ValuationSection",
    "build_correlation_chart_html",
    "build_cycle_lead",
    "build_delta_chart_html",
    "build_forecast_chart_html",
    "build_html_report",
    "build_portfolio_mood",
    "build_trade_signals",
    "to_lead_signals",
    "detect_risk_signals",
    "market_status",
    "parse_resolved_predictions",
    "to_symbol_result",
]

_SAFE_URL_SCHEMES = {"http", "https"}

# MAIL-06 (c): tło bloku sygnału modelu jako klasa tonu (report_styles).
_SIGNAL_FILL = {"up": "gu", "dn": "gd", "nt": "gn"}

# Markery slotów — placeholdery wstawiane przy renderze i podmieniane treścią
# w build_html_report. Trzymane jako stałe (jedno źródło prawdy), żeby literówka
# w jednym z dwóch miejsc (szablon ↔ replace) nie zostawiła cicho pustego slotu.
_QUOTA_BANNER_SLOT = "<!-- QUOTA_BANNER_SLOT -->"
# #1: lead cyklu — PO quota bannerze (ten jest celowo pierwszy: gdy płatny
# klucz padł, to ważniejsze niż jakikolwiek ruch ceny).
_LEAD_SLOT = "<!-- LEAD_SLOT -->"
# #4: sekcja powitalna pierwszego cyklu ("Dzień 1") — zamiast ściany "Pominięte".
_ONBOARDING_SLOT = "<!-- ONBOARDING_SLOT -->"
_RISK_WATCH_SLOT = "<!-- RISK_WATCH_SLOT -->"
_PORTFOLIO_SLOT = "<!-- PORTFOLIO_SLOT -->"
# #15: sekcja 💼 realnego portfela użytkownika (wagi, P/L, klaster, PLN).
# Nazwa slotu rozłączna z _PORTFOLIO_SLOT (Portfolio Radar) — to inna sekcja.
_USER_PORTFOLIO_SLOT = "<!-- USER_PORTFOLIO_SLOT -->"
_COUNCIL_HISTORY_SLOT = "<!-- COUNCIL_HISTORY_SLOT -->"
_PERSONA_LEADERBOARD_SLOT = "<!-- PERSONA_LEADERBOARD_SLOT -->"
# #8: "🔄 Zmiany nastawienia" — flipy rady i skoki sentymentu vs poprzedni cykl.
_CONSENSUS_SHIFT_SLOT = "<!-- CONSENSUS_SHIFT_SLOT -->"
_TRACK_RECORD_SLOT = "<!-- TRACK_RECORD_SLOT -->"
# #12: karta kondycji modelu — tuż za Track Recordem (ta sama oś zaufania).
_MODEL_SCORECARD_SLOT = "<!-- MODEL_SCORECARD_SLOT -->"
# FinOps: „agent, który raportuje własny rachunek" — koszt cyklu.
_FINOPS_SLOT = "<!-- FINOPS_SLOT -->"
_ALPHA_SLOT = "<!-- ALPHA_SLOT -->"
_SUGGESTIONS_SLOT = "<!-- SUGGESTIONS_SLOT -->"
# MAIL-42: linia proweniencji w stopce (przebieg · commit · logi).
_RUN_FOOTER_SLOT = "<!-- RUN_FOOTER_SLOT -->"

_COUNCIL_REC_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}

_VERDICT_COLOR = {
    "UNDERVALUED": "#2e7d32",  # zielony
    "FAIR": "#616161",         # szary
    "OVERVALUED": "#c62828",   # czerwony
    "UNKNOWN": "#9e9e9e",      # jasnoszary
}


def _section_h2(title: str) -> str:
    """Nagłówek sekcji raportu (H2) — jedno źródło prawdy dla stylu."""
    return f"<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>{title}</h2>"


def _render_valuation(section: ValuationSection | None) -> str:
    """Renderuje sekcję wyceny fundamentalnej do HTML (Jinja2 template)."""
    if section is None:
        return ""

    def fmt(v: float | None) -> str:
        return _number(v, decimals=2) if isinstance(v, (int, float)) else "—"

    growth = (
        _pct(section.eps_growth_yoy, decimals=1)
        if section.eps_growth_yoy is not None
        else "—"
    )
    return render_template(
        "valuation_section.html.j2",
        {
            "section": section,
            "verdict_value": section.verdict.value,
            "verdict_color": _VERDICT_COLOR.get(section.verdict.value, "#616161"),
            "trailing_pe_fmt": fmt(section.trailing_pe),
            "forward_pe_fmt": fmt(section.forward_pe),
            "peg_ratio_fmt": fmt(section.peg_ratio),
            "growth_fmt": growth,
            "fetched_at_iso": section.fetched_at.isoformat(),
        },
    )


def _render_valuation_text(section: ValuationSection | None) -> str:
    """Renderuje sekcję wyceny fundamentalnej do plain text."""
    if section is None:
        return ""

    def fmt(v: float | None) -> str:
        return _number(v, decimals=2) if isinstance(v, (int, float)) else "—"

    growth = (
        _pct(section.eps_growth_yoy, decimals=1)
        if section.eps_growth_yoy is not None
        else "—"
    )

    lines = [
        "  Wycena fundamentalna:",
        f"    Trailing P/E: {fmt(section.trailing_pe)}  · Forward P/E: {fmt(section.forward_pe)}",
        f"    PEG ratio: {fmt(section.peg_ratio)}  · EPS growth YoY: {growth}",
        f"    Werdykt: {section.verdict.value}",
    ]
    return "\n".join(lines)


def _model_signal(r: SymbolResult) -> str:
    """MAIL-27: sygnał modelu (KUP/SPRZEDAJ/TRZYMAJ) — etykieta ze wspólnego
    słownika domeny, nie „rekomendacja”."""
    return model_signal_label(r.trend)


# G03: przy cold-starcie target == bieżąca cena, więc "+0.00%" udawałoby prognozę.
_COLD_START_FORECAST = "— (ML: cold-start)"
# WIA-13: przy cold-starcie siła sygnału to sama pewność — sekcja mówi to wprost.
_SIGNALS_COLD_START_NOTE = (
    "Model ML bez wag (cold-start): ranking wg pewności LLM, skalibrowanej historią, "
    "gdy jest próbka — bez prognozy ruchu."
)


def _signed_move(value: float) -> str:
    """Zwrot jako „+2,4%” / „-2,1%” (minus ASCII, jedno miejsce, MAIL-58)."""
    return _pct(value, signed=True, decimals=1)


def _move_interval_label(interval: MoveInterval) -> str:
    """ML-34: „80% przedział: -2,1% … +2,4% (n=57)” albo jawne „za mało
    historii (n/20)" — krótkie okno nie udaje liczby."""
    level = f"{_pct(interval.nominal_coverage, decimals=0)} przedział"
    if interval.low is None or interval.high is None:
        return (
            f"{level}: za mało historii "
            f"({interval.calibration_size}/{interval.required_size})"
        )
    return (
        f"{level}: {_signed_move(interval.low)} … {_signed_move(interval.high)} "
        f"(n={interval.calibration_size})"
    )


def _model_signal_reason_html(r: SymbolResult) -> str:
    """Zwraca krótki HTML-owy powód sygnału modelu (1–2 zdania)."""
    direction = _model_signal(r)
    tone = SIGNAL_TONE.get(direction, "nt")

    conf_part = (
        f"Pewność: <strong>{_pct(r.confidence_score, decimals=0)}</strong>."
        if r.confidence_score is not None
        else ""
    )
    change_part = ""
    if r.expected_change is not None and not r.ml_cold_start:
        change_part = (
            f" Prognozowana zmiana: "
            f"<strong class='{delta_tone(r.expected_change)}'>"
            f"{direction_pct(r.expected_change)}</strong>."
        )
    sentiment_part = ""
    if r.sentiment_score is not None:
        if r.sentiment_score > 0.2:
            sentiment_part = " Sentyment rynkowy pozytywny."
        elif r.sentiment_score < -0.2:
            sentiment_part = " Sentyment rynkowy negatywny."
        else:
            sentiment_part = " Sentyment neutralny."

    return (
        f'<div class="rs {_SIGNAL_FILL[tone]}">'
        f'<span class="b7 {tone}">➤ {MODEL_SIGNAL_TITLE}: {direction}</span> — '
        f"{conf_part}{change_part}{sentiment_part}"
        f"</div>"
    )


def _model_signal_reason_text(r: SymbolResult) -> str:
    direction = _model_signal(r)
    conf_part = f"pewność {_pct(r.confidence_score, decimals=0)}" if r.confidence_score is not None else ""
    change_part = (
        f"prognoza {_pct(r.expected_change, signed=True)}"
        if r.expected_change is not None and not r.ml_cold_start
        else ""
    )
    parts = [p for p in [conf_part, change_part] if p]
    detail = ", ".join(parts)
    return f"{MODEL_SIGNAL_TITLE}: {direction}" + (f" ({detail})" if detail else "")


def _html(value: object) -> str:
    return html.escape(str(value), quote=True)


def _is_crypto(r: SymbolResult) -> bool:
    return r.asset_class == "CRYPTO"


def _crypto_skip_label(r: SymbolResult, display_timezone: str, *, short: bool) -> str:
    """Status pominiętego krypto: „poniżej progu…” albo (FIN-14) nowy punkt odniesienia."""
    if _is_rebaselined(r) and r.reference_at is not None:
        return _rebaseline_label(format_local_date(r.reference_at, display_timezone))
    return "poniżej progu" if short else "poniżej progu zmienności"


def _render_crypto_section_html(
    results: list[SymbolResult], display_timezone: str = DEFAULT_DISPLAY_TIMEZONE
) -> str:
    """Dedykowana sekcja 🪙 Krypto — widoczna NIEZALEŻNIE od bramki volatility.

    Krypto ma osobny próg (5%) i 24/7 rynek, więc często ląduje jako 'ignored'
    i ginęło w chipach 'Pominięte'. Tu pokazujemy każdy ticker krypto z ceną,
    Δ i statusem (prognoza dla 'saved', 'poniżej progu' dla 'ignored')."""
    cryptos = [r for r in results if _is_crypto(r) and r.status != "error"]
    if not cryptos:
        return ""
    rows = []
    for r in cryptos:
        if r.status == "saved":
            trend_part = (
                f"<span style='color: {_trend_color(r.trend)}; font-weight: 600;'>"
                f"{_html(_trend_label(r.trend))}</span>"
            )
            forecast_part = (
                f" · prognoza <span style='color: {_delta_color(r.expected_change)};'>"
                f"{direction_pct(r.expected_change)}</span>"
                if r.expected_change is not None and not r.ml_cold_start
                else ""
            )
            conf_part = (
                f" · pewność {_pct(r.confidence_score, decimals=0)}"
                if r.confidence_score is not None
                else ""
            )
            status_html = f"{trend_part}{forecast_part}{conf_part}"
        else:
            status_html = (
                "<span style='color: #6b7280;'>"
                f"{_html(_crypto_skip_label(r, display_timezone, short=False))}</span>"
            )
        rows.append(f"""
          <div style="margin-bottom: 6px; padding: 8px 12px; background: #fdf4ff;
                      border-left: 3px solid #a855f7; border-radius: 4px; font-size: 13px;">
            <strong>{_html(_company_label(r.symbol))}</strong>
            <span style="color: #4b5563;">· {_money(r.current_price)}
              · Δ <span style="color: {_delta_color(r.delta)};">{direction_pct(r.delta)}</span>
              · {status_html}</span>
          </div>
        """)
    return (
        _section_h2("🪙 Krypto")
        + "".join(rows)
    )


def _render_crypto_section_text(
    results: list[SymbolResult], display_timezone: str = DEFAULT_DISPLAY_TIMEZONE
) -> str:
    cryptos = [r for r in results if _is_crypto(r) and r.status != "error"]
    if not cryptos:
        return ""
    lines = ["KRYPTO", "-" * 64]
    for r in cryptos:
        if r.status == "saved":
            status = (
                f"{_trend_label(r.trend)} "
                f"prognoza {_pct(r.expected_change, signed=True)}"
                if r.expected_change is not None and not r.ml_cold_start
                else _trend_label(r.trend)
            )
        else:
            status = _crypto_skip_label(r, display_timezone, short=True)
        lines.append(
            f"  {_company_label(r.symbol):40s} {_money(r.current_price):>12s}  "
            f"Δ {_pct(r.delta, signed=True):>8s}  {status}"
        )
    return "\n".join(lines)


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


def _render_council_section(verdict: CouncilVerdict) -> str:
    """Renderuje sekcję rady doradczej (Jinja2 template).

    Logika decyzyjna (split decision, strong consensus, vote distribution)
    pochodzi z metod domenowych `CouncilVerdict` — template tylko renderuje.
    """
    final = verdict.final_recommendation
    opinion_rows = [
        {
            "name": op.investor_name,
            "label": council_recommendation_label(op.recommendation),
            "emoji": _COUNCIL_REC_EMOJI.get(op.recommendation, "⬜"),
            "tone": COUNCIL_TONE.get(op.recommendation, "nt"),
            "row_tone": COUNCIL_ROW_TONE.get(op.recommendation, "gn"),
            "confidence_pct": int(op.confidence * 100),
            "confidence_label": op.confidence_label(),
            "factors": ", ".join(op.key_factors[:3]),
        }
        for op in verdict.investor_opinions
    ]
    return render_template(
        "council_section.html.j2",
        {
            "verdict": verdict,
            "label": council_recommendation_label(final),
            "fill": COUNCIL_FILL.get(final, "fn"),
            "row_tone": COUNCIL_ROW_TONE.get(final, "gn"),
            "strength_pct": int(verdict.consensus_strength * 100),
            "opinion_rows": opinion_rows,
            "vote_dist": verdict.vote_distribution(),
            "is_split": verdict.is_split_decision(),
            "strong_consensus": verdict.has_strong_consensus(),
            "dissenting_views": verdict.dissenting_views,
        },
    )


# MAIL-06 (b): rada spoza top N sygnałów i spoza leadu — jedna linia (werdykt,
# rozkład głosów, podsumowanie) i zawsze widoczne głosy niezgodne. Pełna tabela
# person to ~4 KB na symbol, a Gmail przycina mail powyżej ~102 KB. Summary i
# dysydenci pochodzą z LLM-a, więc linia idzie przez autoescapowany `_env`.
_COUNCIL_LINE_TEMPLATE = _env.from_string(
    "<span class='b7 {{ tone }}'>{{ label }}</span>"
    " · zgodność {{ strength_pct }}%"
    "{% if is_split %} · ⚠️ podzielona{% endif %} · "
    "🟢 {{ dist['BUY'] }} KUP · 🔴 {{ dist['SELL'] }} SPRZEDAJ · 🟡 {{ dist['HOLD'] }} TRZYMAJ"
    " — „{{ summary }}”"
    "{% if dissent %}<div class='dv'>"
    "⚠️ Głosy niezgodne: {{ dissent | join('; ') }}</div>{% endif %}"
)


def _render_council_line(verdict: CouncilVerdict) -> str:
    """MAIL-06 (b): zwinięta rada — jedna linia zamiast tabeli person (komórka
    „Rada” w skrócie pozostałych predykcji)."""
    final = verdict.final_recommendation
    return _COUNCIL_LINE_TEMPLATE.render(
        label=council_recommendation_label(final),
        tone=COUNCIL_TONE.get(final, "nt"),
        strength_pct=int(verdict.consensus_strength * 100),
        is_split=verdict.is_split_decision(),
        dist=verdict.vote_distribution(),
        summary=verdict.summary,
        dissent=verdict.dissenting_views,
    )


# ---------------------------------------------------------------------------
# ⚠️  BEZPIECZEŃSTWO — escapowanie niezaufanych pól w treści maila
# ---------------------------------------------------------------------------
# Większość HTML-a maila składana jest f-stringami (funkcje `_render_html`,
# `_render_crypto_section_html`, `_render_resolved_item_html` itd.). W tym trybie
# escapowanie NIE jest domyślne: KAŻDA wartość pochodząca z danych zewnętrznych
# (tytuły/źródła newsów z Alpha Vantage, `reasoning`/`insight` z LLM-a,
# `reflection_insight`, `error_message`, dowolny symbol/tekst spoza whitelisty)
# MUSI być ręcznie owinięta w `_html(...)` przed wstawieniem do f-stringa.
# Jedno zapomniane `_html()` = stored-HTML-injection sink.
#
# Regresje pilnujące tej reguły: `TestHtmlEscaping` oraz
# `TestHtmlEscapingRegression` w tests/application/test_report_builder.py —
# wstrzykują payload XSS w każde niezaufane pole i żądają zescapowanego outputu.
# Dodając nowe pole z danych zewnętrznych: owiń je w `_html()` ORAZ dopisz tam
# test regresyjny.
#
# Docelowo te sekcje migrują na autoescapowany Jinja env (jak council/valuation
# w templates/), gdzie escapowanie jest domyślne, nie opt-in. Pierwszą sekcją
# przeniesioną na ten model jest blok "Top newsy" (`_render_news_block_html`
# poniżej, renderowany przez `_env.from_string` z włączonym autoescape).
# Pozostałe sekcje wciąż używają ręcznego `_html()` — patrz końcowy raport.
# ---------------------------------------------------------------------------

# Inline Jinja template bloku "Top newsy" — renderowany przez WSPÓLNY,
# autoescapowany `_env` (select_autoescape(default_for_string=True)).
# Escapowanie `title`/`source` jest tu DOMYŚLNE; `href` to zwalidowany surowy
# URL (sam autoescape zadba o atrybut, bez podwójnego escapowania).
_NEWS_BLOCK_TEMPLATE = _env.from_string(
    "{%- if items -%}"
    "<div class='nw'>"
    "📰 <strong>Top newsy:</strong>"
    "<ul class='nl'>"
    "{%- for n in items -%}"
    "<li class='ni'>"
    "<strong>[{{ n.source_label }}]</strong> "
    "{% if n.href %}"
    "<a href='{{ n.href }}' target='_blank' class='na'>{{ n.title }}</a>"
    "{% else %}{{ n.title }}{% endif %}"
    " <span class='nm'>"
    "(relevance {{ n.relevance_fmt }}, sentyment {{ n.sentiment_fmt }})"
    "</span></li>"
    "{%- endfor -%}"
    "</ul></div>"
    "{%- endif -%}"
)


def _safe_news_url(url: str | None) -> str | None:
    """Zwraca ZWALIDOWANY surowy URL (http/https + netloc) albo None.

    Waliduje schemat (tylko http/https) i obecność hosta — odrzuca m.in.
    `javascript:`. NIE escapuje: autoescape Jinja zrobi to sam w atrybucie
    `href`, więc unikamy podwójnego escapowania w template'cie.
    """
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _SAFE_URL_SCHEMES or not parsed.netloc:
        return None
    return url


def _render_news_block_html(top_news: list[TopNewsItem]) -> str:
    """Renderuje blok 'Top newsy' przez autoescapowany Jinja env.

    Migracja przyrostowa f-string → Jinja (finding #35): escapowanie tytułów
    i źródeł newsów jest tu domyślne, nie opt-in. Output równoważny dawnemu
    f-stringowi (te same style inline, kolejność i format relevance/sentyment)."""
    items = [
        {
            "source_label": n.source or "unknown",
            "title": n.title,
            "href": _safe_news_url(n.url),
            "relevance_fmt": _number(n.relevance, decimals=2),
            "sentiment_fmt": _number(n.sentiment, decimals=2, signed=True),
        }
        for n in top_news
    ]
    return _NEWS_BLOCK_TEMPLATE.render(items=items)


# Q5: etykieta wyniku analogu (kierunkowo trafiony / chybiony / nieoceniony).
_PRECEDENT_OUTCOME = {
    True: ("✅ trafił", PALETTE.positive),
    False: ("❌ chybił", PALETTE.negative),
}


def _render_precedents_block_html(precedents: list[SimilarPrecedent]) -> str:
    """Renderuje blok 'Na podstawie analogów' (Q5) — audytowalny ślad RAG.

    Pokazuje historyczne sytuacje, które agent pobrał w tym cyklu, z ich
    rzeczywistym wynikiem kierunkowym. Pusta lista → pusty string (brak bloku,
    brak błędu). `summary` pochodzi z LLM-a (dane zewnętrzne) → escapujemy
    przez `_html`."""
    if not precedents:
        return ""
    items = []
    for p in precedents:
        outcome_txt, outcome_color = (
            _PRECEDENT_OUTCOME[p.is_trend_correct]
            if p.is_trend_correct is not None
            else ("• wynik nieznany", "#6b7280")
        )
        items.append(
            f"<li style='margin: 2px 0;'>"
            f"<span style='color: {_trend_color(p.predicted_trend)}; font-weight: 600;'>"
            f"{_html(_trend_label(p.predicted_trend))}</span> "
            f"<span style='color: {outcome_color};'>({outcome_txt})</span> — "
            f"{_html(p.summary)}</li>"
        )
    return (
        "<div style='font-size: 11px; color: #4b5563; margin-top: 8px;'>"
        "🧭 <strong>Na podstawie analogów:</strong>"
        "<ul style='margin: 4px 0 0 20px; padding: 0;'>"
        + "".join(items)
        + "</ul></div>"
    )


def _render_attribution_block_html(
    contribs: tuple[FeatureContribution, ...],
) -> str:
    """Renderuje blok 'Wkład cech' (Q3 SHAP-lite) — które cechy ML pchnęły
    predykcję w górę/dół i o ile. Pusta krotka → pusty string."""
    if not contribs:
        return ""
    chips = []
    for c in contribs:
        positive = c.contribution >= 0
        color = PALETTE.positive if positive else PALETTE.negative
        chips.append(
            f"<span style='display:inline-block; margin-right:10px;'>"
            f"{_html(c.feature)} "
            f"<strong style='color:{color};'>"
            f"{_number(c.contribution, decimals=4, signed=True)}</strong></span>"
        )
    return (
        "<div style='font-size: 11px; color: #475569; margin-top: 6px; "
        "padding: 4px 8px; background: #f8fafc; border-radius: 3px;'>"
        "🔍 <strong>Wkład cech:</strong> " + "".join(chips) + "</div>"
    )


def _sizing_track_record(
    accuracy_stats: Mapping[str, Any] | None,
) -> tuple[float | None, float | None]:
    """PORT-01: `(hit_rate, n_eff)` dla sizingu z `get_accuracy_stats`.

    Bez `n_eff` nie wiadomo, ile niezależnych zakładów stoi za hit-rate (40
    trafień jednego dnia spadku NASDAQ to jeden zakład), więc hit-rate idzie
    jako nieznany — jak przy cold-starcie.
    """
    if not accuracy_stats:
        return None, None
    n_eff = accuracy_stats.get("n_eff")
    if n_eff is None:
        return None, None
    return accuracy_stats.get("mean_accuracy"), float(n_eff)


def _cluster_exposure_per_symbol(
    portfolio: Portfolio | None,
    clusters: Mapping[str, str] | None,
    current_prices: Mapping[str, Decimal],
) -> dict[str, Decimal] | None:
    """#15 faza 2 — mapa symbol → udział JEGO klastra korelacji w kapitale.

    Trzy pozycje w jednym klastrze to jedna pozycja przebrana za trzy, więc
    sizing musi widzieć sumę klastra, nie tylko wagę symbolu.
    Brak portfela lub brak mapy klastrów → None (sizing jak przed fazą 2).
    """
    if portfolio is None or not clusters:
        return None
    exposures = portfolio.cluster_exposure(clusters, current_prices)
    by_cluster = {e.cluster: e.weight for e in exposures}
    return {
        symbol: by_cluster[cluster]
        for symbol, cluster in clusters.items()
        if cluster in by_cluster
    }


def to_lead_signals(
    results: list[SymbolResult],
    resolved: list[ResolvedPrediction],
    portfolio_weights: Mapping[str, Decimal] | None = None,
) -> list[LeadSignal]:
    """Mapuje DTO raportu na domenowe `LeadSignal` (#1).

    Granica architektury: `src/domain/digest_lead.py` bierze wyłącznie typy
    domenowe i prymitywy, więc `resolved_label` powstaje TU jako gotowy string.
    Symbole bez zmiany ceny nie wnoszą nic do rankingu i są pomijane.

    PORT-10: `portfolio_weights` (z `Portfolio.weights`) daje sygnałom wagę
    pozycji. Status "saved" znaczy, że ruch przeszedł bramkę volatility (graf
    poszedł do analizy) — tylko taki ruch pozycji może być leadem.
    """
    # MAIL-15: etykieta trafia do leadu i tematu maila — po polsku, ze wspólnego
    # słownika domeny („trafiona prognoza wzrostu”), nie „predykcja BULLISH”.
    labels = {
        r.symbol: (
            f"{'trafiona' if r.is_correct else 'chybiona'} "
            f"{trend_forecast_label(r.predicted_trend)}"
        )
        for r in resolved
    }
    signals: list[LeadSignal] = []
    for r in results:
        if r.delta is None:
            continue
        signals.append(
            LeadSignal(
                symbol=r.symbol,
                # `SymbolResult.delta` to ułamek, a lead formatuje PROCENT.
                price_delta_pct=PriceDelta(r.delta).percent,
                council_verdict=r.council_verdict,
                resolved_label=labels.get(r.symbol),
                position_weight=(
                    portfolio_weights.get(r.symbol) if portfolio_weights else None
                ),
                crossed_volatility_gate=r.status == "saved",
            )
        )
    return signals


def _report_prices(
    results: list[SymbolResult], context: ReportContext
) -> dict[str, Decimal]:
    """#15/PORT-07: ceny raportu — z wyników cyklu, uzupełnione cenami pozycji
    spoza watchlisty (`held_prices`). Cena z cyklu ma pierwszeństwo."""
    return {**(context.held_prices or {}), **_current_prices_of(results)}


def build_cycle_lead(
    results: list[SymbolResult], context: ReportContext
) -> list[LeadItem]:
    """#1/PORT-10: lead cyklu — jedno źródło dla treści maila i jego tematu.

    Gdy znamy realny portfel, sygnały niosą wagi pozycji, więc ruch i sygnał
    na Twojej pozycji wyprzedzają ruch spółki, której nie masz.
    """
    weights = (
        context.user_portfolio.weights(_report_prices(results, context))
        if context.user_portfolio is not None
        else None
    )
    signals = to_lead_signals(results, context.resolved_predictions or [], weights)
    return build_lead(signals, context.quota_alerts or [])


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
        f"{_html(preview)}{_PREHEADER_FILLER}</div>"
    )


# Sloty pomijane przy pustej treści: nagłówek sekcji rysuje się w szablonie, więc
# podmiana na "" zostawiłaby pustą ramkę. Nieusunięty znacznik jest komentarzem
# HTML, czyli w mailu niewidoczny — to bezpieczniejszy z dwóch wyników.
_SKIP_WHEN_EMPTY_SLOTS = frozenset({_QUOTA_BANNER_SLOT, _RISK_WATCH_SLOT})


# MAIL-06: ciąg białych znaków ASCII — wcięcia szablonów między tagami, w tekście
# i w wieloliniowych atrybutach `style`. Twarda spacja (U+00A0) z formatera kwot
# nie jest wcięciem i zostaje.
_WHITESPACE_RUN = re.compile(r"\s+", re.ASCII)


# Spacja między dwoma tagami struktury tabeli (`</td> <td>`, `<tr> <td>`) nie jest
# treścią komórki — parser HTML ją pomija.
_SPACE_BETWEEN_TABLE_TAGS = re.compile(
    r"(</?(?:table|thead|tbody|tr|td|th)\b[^>]*>) (?=</?(?:table|thead|tbody|tr|td|th)\b)"
)


def _compact_markup(markup: str) -> str:
    """Gmail przycina wiadomości powyżej ~102 KB, a wcięcia szablonów to kilka procent
    maila. Każdy ciąg białych znaków zamieniamy na jedną spację: mail nie ma `<pre>`
    ani `<textarea>`, więc HTML renderuje się identycznie (tekst i tak zwija białe
    znaki, CSS w `style` też), a sąsiednie elementy inline się nie sklejają. Spację
    między tagami struktury tabeli usuwamy w ogóle."""
    collapsed = _WHITESPACE_RUN.sub(" ", markup).strip()
    return _SPACE_BETWEEN_TABLE_TAGS.sub(r"\1", collapsed)


def _email_document(body: str) -> str:
    """MAIL-57: kompletny dokument — `lang="pl"` (czytnik ekranu czyta polską
    fonetyką zamiast angielskiej), kodowanie i arkusz klas (MAIL-06 c) w `<head>`.
    Preheader zostaje pierwszym elementem `<body>`, więc podgląd w skrzynce się
    nie zmienia."""
    return (
        '<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8">'
        f"{EMAIL_STYLESHEET}</head><body>{body}</body></html>"
    )


def _fill_html_slots(html: str, sections: Mapping[str, str]) -> str:
    """Wstawia treść w sloty HTML — klucz to znacznik slotu, wartość to gotowy
    HTML sekcji.

    Sloty spoza `_SKIP_WHEN_EMPTY_SLOTS` podmieniamy zawsze (też na ""), żeby
    znacznik nie został w treści maila."""
    for slot, content in sections.items():
        if not content and slot in _SKIP_WHEN_EMPTY_SLOTS:
            continue
        html = html.replace(slot, content, 1)
    return html


def _fill_text_slots(
    text: str,
    *,
    quota_text: str,
    macro_risk_text: str,
    lead_text: str,
    onboarding_text: str,
    suggestions_text: str,
    portfolio_text: str = "",
    user_portfolio_text: str = "",
) -> str:
    """Wstawia treść w sloty plain-text. Pusty slot usuwa cały wiersz znacznika —
    tekst nie ma komentarzy HTML, więc znacznik byłby w nim dosłowną linią.
    W HTML QUOTA/RISK_WATCH zostają przy pustej treści (`_SKIP_WHEN_EMPTY_SLOTS`)."""
    text = text.replace(
        f"{_QUOTA_BANNER_SLOT}\n",
        f"{quota_text}\n" if quota_text else "",
        1,
    ).replace(_QUOTA_BANNER_SLOT, quota_text, 1)
    text = text.replace(
        f"{_RISK_WATCH_SLOT}\n",
        f"{macro_risk_text}\n" if macro_risk_text else "",
        1,
    ).replace(_RISK_WATCH_SLOT, macro_risk_text, 1)
    text = text.replace(
        f"{_PORTFOLIO_SLOT}\n",
        f"{portfolio_text}\n" if portfolio_text else "",
        1,
    ).replace(_PORTFOLIO_SLOT, portfolio_text, 1)
    text = text.replace(
        f"{_USER_PORTFOLIO_SLOT}\n",
        f"{user_portfolio_text}\n\n" if user_portfolio_text else "",
        1,
    ).replace(_USER_PORTFOLIO_SLOT, user_portfolio_text, 1)
    text = text.replace(
        f"{_LEAD_SLOT}\n",
        f"{lead_text}\n\n" if lead_text else "",
        1,
    ).replace(_LEAD_SLOT, lead_text, 1)
    text = text.replace(
        f"{_ONBOARDING_SLOT}\n",
        f"{onboarding_text}\n\n" if onboarding_text else "",
        1,
    ).replace(_ONBOARDING_SLOT, onboarding_text, 1)
    return text.replace(
        f"{_SUGGESTIONS_SLOT}\n",
        f"{suggestions_text}\n\n" if suggestions_text else "",
        1,
    ).replace(_SUGGESTIONS_SLOT, suggestions_text, 1)


def _partition_by_status(
    results: list[SymbolResult],
) -> tuple[list[SymbolResult], list[SymbolResult], list[SymbolResult]]:
    """Rozbija wyniki cyklu na `(saved, ignored, errors)` — kolejność zgodna z
    sekcjami raportu."""
    return (
        [r for r in results if r.status == "saved"],
        [r for r in results if r.status == "ignored"],
        [r for r in results if r.status == "error"],
    )


def _current_prices_of(results: list[SymbolResult]) -> dict[str, Decimal]:
    """#15: ceny bieżące biorą się z wyników TEGO cyklu — renderer nie dostaje
    żadnego portu. Symbol bez ceny (błąd źródła) po prostu nie wchodzi."""
    return {r.symbol: r.current_price for r in results if r.current_price is not None}


def _previous_prices_of(results: list[SymbolResult]) -> dict[str, Decimal]:
    """PORT-06: cena referencji odtworzona z delty cyklu: prev = current / (1 + Δ).

    Działa dla każdego symbolu z ceną i deltą — także pominiętego bramką albo
    z referencją sprzed pauzy (FIN-14). Cold-start nie ma referencji (Δ = 0 to
    brak punktu odniesienia, nie „bez zmian”), więc nie wchodzi."""
    previous: dict[str, Decimal] = {}
    for r in results:
        if r.current_price is None or r.delta is None:
            continue
        if r.skip_reason is SkipReason.COLD_START or r.delta == -1:
            continue
        previous[r.symbol] = r.current_price / (1 + r.delta)
    return previous


@dataclasses.dataclass(frozen=True)
class _PortfolioChangeView:
    """PORT-06: zmiana portfela z domeny + podpis okna („od ostatniego maila”)."""

    change: PortfolioChange
    label: str


def _portfolio_change_view(
    results: list[SymbolResult],
    portfolio: Portfolio | None,
    current_prices: Mapping[str, Decimal],
    *,
    now: datetime,
    display_timezone: str,
) -> _PortfolioChangeView | None:
    """PORT-06: Δ portfela od referencji cyklu. Okno zaczyna się od NAJSTARSZEJ
    referencji zmierzonych pozycji — po pauzie podpis mówi „od <data>”.
    Brak portfela (np. obcy subskrybent, BEZ-01) → None, sekcje się chowają."""
    if portfolio is None:
        return None
    change = portfolio.change_since(_previous_prices_of(results), current_prices)
    if change is None:
        return None
    measured = {p.symbol for p in portfolio.positions} - set(change.unmeasured_symbols)
    references = [
        r.reference_at for r in results if r.symbol in measured and r.reference_at is not None
    ]
    since = min(references) if references else None
    return _PortfolioChangeView(
        change=change, label=portfolio_change_label(since, now, display_timezone)
    )


def _build_onboarding_blocks(
    cycle_maturity: CycleMaturity | None,
    *,
    instrument_count: int,
    portfolio_sectors: Mapping[str, int] | None,
    sector_map: Mapping[str, str],
) -> tuple[str, str]:
    """#4: powitanie pierwszego cyklu jako `(html, text)`. Renderer nigdy nie
    decyduje o tonie — dostaje gotową klasyfikację z domeny (STEADY_STATE → "").
    Kolejność sektorów: kolejność wstawienia mapy sektorów z configu."""
    maturity = cycle_maturity or CycleMaturity.STEADY_STATE
    sectors = portfolio_sectors or {}
    order = tuple(dict.fromkeys(sector_map.values()))
    return (
        render_onboarding_html(
            maturity, instrument_count=instrument_count, sectors=sectors, sector_order=order
        ),
        render_onboarding_text(
            maturity, instrument_count=instrument_count, sectors=sectors, sector_order=order
        ),
    )


def _build_user_portfolio_html(
    user_portfolio: Portfolio | None,
    current_prices: Mapping[str, Decimal],
    *,
    clusters: Mapping[str, str] | None,
    macro_risk_report: MacroRiskReport | None,
    now: datetime,
    change: _PortfolioChangeView | None = None,
) -> str:
    """#15: sekcja realnego portfela. Kurs USD/PLN pochodzi z `MacroRiskReport`,
    nie z portu. Brak portfela → "" (sekcja sama się chowa). PORT-06: linia
    zmiany od ostatniego maila, gdy da się ją policzyć."""
    if user_portfolio is None:
        return ""
    return render_user_portfolio_html(
        user_portfolio,
        current_prices,
        clusters=clusters,
        now=now,
        usd_pln=_usd_pln_of(macro_risk_report),
        **_change_kwargs(change),
    )


def _change_kwargs(change: _PortfolioChangeView | None) -> dict[str, Any]:
    """PORT-06: opcjonalne argumenty linii zmiany dla rendererów sekcji 💼."""
    if change is None:
        return {}
    return {"change": change.change, "change_label": change.label}


def _build_user_portfolio_text(
    user_portfolio: Portfolio | None,
    current_prices: Mapping[str, Decimal],
    *,
    clusters: Mapping[str, str] | None,
    macro_risk_report: MacroRiskReport | None,
    now: datetime,
    change: _PortfolioChangeView | None = None,
) -> str:
    """Wariant plain-text sekcji portfela (decyzja 2026-10-19) — te same dane co HTML."""
    if user_portfolio is None:
        return ""
    return render_user_portfolio_text(
        user_portfolio,
        current_prices,
        clusters=clusters,
        now=now,
        usd_pln=_usd_pln_of(macro_risk_report),
        **_change_kwargs(change),
    )


def _usd_pln_of(macro_risk_report: MacroRiskReport | None) -> Decimal | None:
    """Kurs USD/PLN z `MacroRiskReport.polish_macro` (wartość, nie port)."""
    if macro_risk_report is None or macro_risk_report.polish_macro is None:
        return None
    return macro_risk_report.polish_macro.usd_pln


def _build_model_scorecard_html(
    model_scorecard: ModelScorecardView | None,
    *,
    now: datetime,
    model_loaded: bool,
) -> str:
    """#12: karta kondycji modelu. Brak danych → "" (sekcja sama się chowa)."""
    if model_scorecard is None:
        return ""
    return render_model_scorecard_html(
        model_scorecard.summary,
        trained_at=model_scorecard.trained_at,
        now=now,
        model_loaded=model_loaded,
        candidate_rmse=model_scorecard.candidate_rmse,
        baseline_rmse=model_scorecard.baseline_rmse,
        directional_hit_rate=model_scorecard.directional_hit_rate,
        folds=model_scorecard.folds,
        metrics_symbol=model_scorecard.metrics_symbol,
        n_samples=model_scorecard.metrics_n_samples,
        evaluations=model_scorecard.evaluations,
        drift_baseline_rmse=model_scorecard.drift_baseline_rmse,
        llm_baseline_rmse=model_scorecard.llm_baseline_rmse,
    )


@dataclasses.dataclass(frozen=True)
class ReportContext:
    """Wszystko, co orkiestrator dociąga OBOK wyników cyklu: statystyki, sekcje
    ryzyka, historia rady, dane alfa, koszt.

    Każde pole jest opcjonalne, bo każda sekcja raportu chowa się sama, gdy jej
    dane nie dotarły — dociąganie kontekstu jest best-effort i nie może wywalić
    wysyłki. Zebranie ich w jeden obiekt zdejmuje z `build_html_report` 22
    argumenty, których przy wywołaniu i tak nie dało się rozróżnić inaczej niż
    po nazwie.

    Pola:
        accuracy_stats: wynik `RepositoryPort.get_accuracy_stats(days)`.
        resolved_predictions: zamknięte predykcje z ostatnich N godzin
            (do sekcji day-over-day).
        resolved_this_cycle: `resolved_predictions` to predykcje rozliczone
            w TYM cyklu (WIA-19, okno po `settled_at`), nie z ostatnich 24 h —
            nagłówek sekcji mówi wtedy, jakie okno pokazuje.
            Pusta lista (udany odczyt) → sekcja mówi „brak rozliczeń” (MAIL-25);
            None (nie dociągnięto) → brak sekcji.
        resolved_unavailable: odczyt zamkniętych predykcji padł (MAIL-25) —
            sekcja ostrzega o awarii zamiast udawać „brak rozliczeń”.
        macro_risk_report: wynik `MonitorMacroRiskUseCase.run(...)` — gdy
            podany, w raporcie pojawia się sekcja 🚨 Risk Watch.
        symbols_filter: watchlista subskrybenta; raport jest do niej tnięty.
    """

    accuracy_stats: dict[str, Any] | None = None
    resolved_predictions: list[ResolvedPrediction] | None = None
    resolved_this_cycle: bool = False
    resolved_unavailable: bool = False
    macro_risk_report: MacroRiskReport | None = None
    portfolio_risk_report: PortfolioRiskReport | None = None
    council_history: list[InvestorHistory] | None = None
    persona_track_record: list[PersonaTrackRecord] | None = None
    quota_alerts: list[QuotaAlert] | None = None
    symbols_filter: frozenset[str] | None = None
    equity_curve: EquityCurve | None = None
    calibration_buckets: list[CalibrationBucket] | None = None
    lessons: LessonsReport | None = None
    alpha_signals: dict[str, AlphaSignals] | None = None
    yield_curve: YieldCurveSnapshot | None = None
    alpha_prices: dict[str, Decimal] | None = None
    signal_calibration_buckets: list[CalibrationBucket] | None = None
    cycle_maturity: CycleMaturity | None = None
    portfolio_sectors: Mapping[str, int] | None = None
    model_scorecard: ModelScorecardView | None = None
    # G03: czy port ML ma wagi. False → karta kondycji mówi o cold-starcie
    # zamiast o "sprawdzonym modelu" (którego wtedy nie ma).
    ml_model_loaded: bool = True
    consensus_shifts: list[tuple[str, ConsensusShift]] | None = None
    user_portfolio: Portfolio | None = None
    # PORT-07: ceny pozycji spoza watchlisty (sam darmowy /quote, bez snapshotu).
    # Cena z cyklu ma pierwszeństwo, gdy symbol jest w obu.
    held_prices: Mapping[str, Decimal] | None = None
    # Decyzja 2026-10-19: sekcja portfela trafia też do wersji tekstowej maila.
    # False = tekst dla starego pushu (Telegram/Slack) — bez kwot P/L.
    user_portfolio_in_text: bool = True
    portfolio_clusters: Mapping[str, str] | None = None
    # PORT-05: symbol → etykieta sektora (sektory akcji z `sectors`, ETF, Krypto)
    # i sektor → peerzy sekcji 💡 Warte uwagi — oba z config.toml, jedno źródło.
    # None = brak etykiet sektorów i brak sekcji sugestii.
    sector_map: Mapping[str, str] | None = None
    sector_peers: Mapping[str, Sequence[str]] | None = None
    # PORT-04: ryzyko realnego portfela (VaR, obsunięcie, hedge) do sizingu.
    # Orkiestrator podaje je tylko za `portfolio_var_enabled`; bez
    # `user_portfolio` jest ignorowane (nigdy VaR na równych wagach).
    position_risk: PositionRisk | None = None
    cycle_cost: CycleCost | None = None
    # WIA-17/22/23/31/32/36: dowód trafności i Brier pod sekcją Track Record.
    track_record_stats: TrackRecordStats | None = None
    # MAIL-14: strefa (IANA) czasu startu cyklu w nagłówku HTML i tekstu.
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE
    # MAIL-42: przebieg, który wygenerował mail — stopka z numerem, krótkim SHA
    # i (tylko w GHA) linkiem do logów. None = stopka jak przed MAIL-42.
    run_context: RunContext | None = None


def _resolved_note(context: ReportContext, window: str) -> str | None:
    """MAIL-25: linia sekcji zamkniętych predykcji, gdy nie ma czego wylistować.

    Awaria odczytu ma pierwszeństwo przed pustą listą — nie może udawać „brak
    rozliczeń”. None w `resolved_predictions` (wołający nie dociągnął rozliczeń)
    → brak linii, czyli brak sekcji, jak przed MAIL-25.
    """
    if context.resolved_predictions:
        return None
    if context.resolved_unavailable:
        return _RESOLVED_UNAVAILABLE_NOTE
    if context.resolved_predictions is not None:
        return _RESOLVED_EMPTY_NOTE[window]
    return None


# MAIL-06: tyle symboli (wg siły sygnału) dostaje pełny blok uzasadnień z radą.
_FULL_DETAIL_TOP_N = 10


def _full_detail_symbols(
    saved: list[SymbolResult],
    ranked_signals: list[TradeSignal],
    lead_items: list[LeadItem],
) -> frozenset[str] | None:
    """MAIL-06: symbole z pełnym blokiem — top N wg siły sygnału (kolejność
    „🎯 Najsilniejsze sygnały”) i symbole z leadu. Symbole bez sygnału (brak
    pewności albo prognozy) idą za rankingiem w kolejności cyklu. None = cykl
    mieści się w top N, więc pełny blok dostaje każdy symbol."""
    if len(saved) <= _FULL_DETAIL_TOP_N:
        return None
    ranked = [sig.symbol for sig in ranked_signals]
    ranked_set = set(ranked)
    ranked.extend(r.symbol for r in saved if r.symbol not in ranked_set)
    lead = {item.symbol for item in lead_items if item.symbol is not None}
    return frozenset(ranked[:_FULL_DETAIL_TOP_N]) | lead


def build_html_report(
    results: list[SymbolResult],
    started_at: datetime,
    duration_seconds: float,
    context: ReportContext | None = None,
) -> tuple[str, str]:
    """Zwraca (html_body, plain_text) — oba reprezentacje raportu.

    `context` niesie opcjonalne dane sekcji (patrz `ReportContext`). Brak
    kontekstu = raport z samych wyników cyklu: sekcje bez danych się chowają.
    """
    context = context or ReportContext()
    # U2: personalizacja — gdy podany filtr symboli (subskrybent), tniemy wyniki
    # do jego watchlisty. Analiza i tak poszła RAZ nad unią; tu tylko re-slicing.
    if context.symbols_filter is not None:
        results = [r for r in results if r.symbol in context.symbols_filter]
    # Domyślne puste kolekcje liczymy RAZ — każde `or []` w ciele buildera to
    # kolejna gałąź do przeczytania, a te same dane trafiają do kilku sekcji.
    resolved = context.resolved_predictions or []
    alerts = context.quota_alerts or []
    sector_map = context.sector_map or {}
    saved, ignored, errors = _partition_by_status(results)
    mood = build_portfolio_mood(results)
    session = market_status(started_at)
    # Q7: hit-rate cyklu (z accuracy_stats) napędza pasmo wielkości pozycji.
    hit_rate, hit_sample = _sizing_track_record(context.accuracy_stats)
    # #9: gdy orkiestrator dostarczył kubełki kalibracji (flaga
    # `calibrated_confidence_enabled`), ranking sygnałów liczy się na pewności
    # skorygowanej historią. None → surowa pewność, jak przed #9.
    current_prices = _report_prices(results, context)
    # #15 faza 2: gdy znamy realny portfel, sizing widzi istniejącą ekspozycję
    # (symbolu i jego klastra) i może zaproponować wyłącznie WĘŻSZĄ bandę.
    portfolio_weights = (
        context.user_portfolio.weights(current_prices) if context.user_portfolio is not None else None
    )
    cluster_exposures = _cluster_exposure_per_symbol(
        context.user_portfolio, context.portfolio_clusters, current_prices
    )
    sizing_risk = (
        context.position_risk
        if context.user_portfolio is not None and context.position_risk is not None
        else PositionRisk()
    )
    rank_signals = functools.partial(
        build_trade_signals,
        results,
        hit_rate=hit_rate,
        hit_sample=hit_sample,
        calibration_buckets=context.signal_calibration_buckets,
        portfolio_weights=portfolio_weights,
        cluster_exposures=cluster_exposures,
        portfolio_var=sizing_risk.var,
        drawdown=sizing_risk.drawdown,
        hedge_effective=sizing_risk.hedge_effective,
    )
    trade_signals = rank_signals()
    risk_signals = detect_risk_signals(results)
    macro_risk_html = (
        render_risk_watch_html(context.macro_risk_report) if context.macro_risk_report else ""
    )
    # Q4: sekcja korelacji/koncentracji portfela (render_correlation_html zwraca
    # "" gdy brak klastrów → sekcja sama się chowa).
    portfolio_html = (
        render_correlation_html(context.portfolio_risk_report)
        if context.portfolio_risk_report
        else ""
    )
    portfolio_text = (
        render_correlation_text(context.portfolio_risk_report)
        if context.portfolio_risk_report
        else ""
    )
    # #1: lead cyklu. Ranking robi domena; TU jest jedyne miejsce, gdzie DTO
    # warstwy application (SymbolResult / ResolvedPrediction) są mapowane na
    # typy domenowe — `src/domain/digest_lead.py` ich nie importuje.
    lead_items = build_cycle_lead(results, context)
    # MAIL-06: pełna rada dla top N wg siły sygnału i symboli z leadu.
    full_detail = _full_detail_symbols(
        saved, rank_signals(top_n=len(results)), lead_items
    )
    lead_html = render_lead_html(lead_items)
    lead_text = render_lead_text(lead_items)
    onboarding_html, onboarding_text = _build_onboarding_blocks(
        context.cycle_maturity,
        instrument_count=len(results),
        portfolio_sectors=context.portfolio_sectors,
        sector_map=sector_map,
    )
    # U5: panel historii głosów rady per inwestor (render zwraca "" gdy brak).
    council_history_html = render_council_history_html(context.council_history or [])
    # #3: ranking wiarygodności person — ranking (z progiem min_votes) przychodzi
    # gotowy z domeny; render zwraca "" gdy żadna persona nie ma dość głosów.
    persona_leaderboard_html = render_persona_leaderboard_html(
        context.persona_track_record or []
    )
    # PORT-06: zmiana portfela od ostatniego maila — sekcja 💼 i pierwszy ekran.
    # Nie trafia do tematu maila (decyzja 2026-10-19).
    portfolio_change = _portfolio_change_view(
        results,
        context.user_portfolio,
        current_prices,
        now=started_at,
        display_timezone=context.display_timezone,
    )
    usd_pln = _usd_pln_of(context.macro_risk_report)
    portfolio_change_box = (
        render_portfolio_change_box_html(
            portfolio_change.change, label=portfolio_change.label, usd_pln=usd_pln
        )
        if portfolio_change is not None
        else ""
    )
    # Stary push (Telegram/Slack) dostaje tekst bez kwot portfela.
    portfolio_change_line = (
        render_portfolio_change_text_line(
            portfolio_change.change, label=portfolio_change.label, usd_pln=usd_pln
        )
        if portfolio_change is not None and context.user_portfolio_in_text
        else ""
    )
    user_portfolio_html = _build_user_portfolio_html(
        context.user_portfolio,
        current_prices,
        clusters=context.portfolio_clusters,
        macro_risk_report=context.macro_risk_report,
        now=started_at,
        change=portfolio_change,
    )
    user_portfolio_text = (
        _build_user_portfolio_text(
            context.user_portfolio,
            current_prices,
            clusters=context.portfolio_clusters,
            macro_risk_report=context.macro_risk_report,
            now=started_at,
            change=portfolio_change,
        )
        if context.user_portfolio_in_text
        else ""
    )
    # #8: zmiany nastawienia rady. Render odfiltrowuje STABLE i jawnie oznacza
    # stęchłe porównania — pusta lista → sekcja się chowa.
    consensus_shift_html = render_consensus_shift_html(context.consensus_shifts or [])
    # FinOps: koszt cyklu. Cykl darmowy (0 wywołań) renderuje się JAWNIE —
    # to najciekawsza informacja w tej sekcji, nie brak danych.
    finops_html = render_finops_html(context.cycle_cost)
    model_scorecard_html = _build_model_scorecard_html(
        context.model_scorecard,
        now=started_at,
        model_loaded=context.ml_model_loaded,
    )
    # T1/T2/T4: sekcja Track Record (krzywa kapitału, kalibracja, lessons).
    # render zwraca "" gdy wszystkie trzy podsekcje puste.
    track_record_html = render_track_record_html(
        context.equity_curve, context.calibration_buckets, context.lessons,
        context.track_record_stats,
    )
    # Dane: sekcja Alpha Signals (insider/analitycy/opcje/social/earnings per
    # symbol + krzywa rentowności FRED). render zwraca "" gdy brak danych.
    alpha_html = render_alpha_html(
        context.alpha_signals or {}, context.yield_curve, context.alpha_prices
    )
    macro_risk_text = (
        render_risk_watch_text(context.macro_risk_report) if context.macro_risk_report else ""
    )
    quota_html = render_quota_banner_html(alerts)
    quota_text = render_quota_banner_text(alerts)
    suggestions = build_sector_suggestions(results, sector_map, context.sector_peers or {})
    suggestions_html = render_suggestions_html(suggestions)
    suggestions_text = render_suggestions_text(suggestions)

    started_local = format_local(started_at, context.display_timezone)
    resolved_window = (
        _RESOLVED_WINDOW_THIS_CYCLE if context.resolved_this_cycle else _RESOLVED_WINDOW_24H
    )
    resolved_note = _resolved_note(context, resolved_window)
    html = _render_html(
        results, saved, ignored, errors, started_local, duration_seconds,
        mood, session, context.accuracy_stats, trade_signals, risk_signals,
        resolved, resolved_window=resolved_window, resolved_note=resolved_note,
        display_timezone=context.display_timezone,
        portfolio_change_box=portfolio_change_box,
        settlements_unavailable=context.resolved_unavailable,
        full_detail=full_detail,
        sector_map=sector_map,
    )
    # Kolejność jak w poprzedniej wersji — znaczniki są rozłączne, więc nie ma
    # znaczenia dla wyniku, ale trzyma diff czytelnym przy porównaniu z historią.
    html = _fill_html_slots(html, {
        _QUOTA_BANNER_SLOT: quota_html,
        _LEAD_SLOT: lead_html,
        _ONBOARDING_SLOT: onboarding_html,
        _RISK_WATCH_SLOT: macro_risk_html,
        _PORTFOLIO_SLOT: portfolio_html,
        _USER_PORTFOLIO_SLOT: user_portfolio_html,
        _COUNCIL_HISTORY_SLOT: council_history_html,
        _PERSONA_LEADERBOARD_SLOT: persona_leaderboard_html,
        _CONSENSUS_SHIFT_SLOT: consensus_shift_html,
        _TRACK_RECORD_SLOT: track_record_html,
        _MODEL_SCORECARD_SLOT: model_scorecard_html,
        _FINOPS_SLOT: finops_html,
        _ALPHA_SLOT: alpha_html,
        _SUGGESTIONS_SLOT: suggestions_html,
        _RUN_FOOTER_SLOT: _render_run_footer_html(context.run_context),
    })
    closed_count = None if context.resolved_unavailable and not resolved else len(resolved)
    html = _email_document(_compact_markup(_render_preheader_html(lead_items, closed_count) + html))
    text = _render_plain(
        results, saved, ignored, errors, started_local, duration_seconds,
        mood, session, context.accuracy_stats, trade_signals, risk_signals,
        resolved, resolved_window=resolved_window, resolved_note=resolved_note,
        display_timezone=context.display_timezone,
        portfolio_change_line=portfolio_change_line,
        settlements_unavailable=context.resolved_unavailable,
        sector_map=sector_map,
    )
    text = _fill_text_slots(
        text,
        quota_text=quota_text,
        macro_risk_text=macro_risk_text,
        portfolio_text=portfolio_text,
        user_portfolio_text=user_portfolio_text,
        lead_text=lead_text,
        onboarding_text=onboarding_text,
        suggestions_text=suggestions_text,
    )
    run_footer_text = _render_run_footer_text(context.run_context)
    if run_footer_text:
        text = f"{text}\n{run_footer_text}"
    return html, text


# ---------------------------------------------------------------------------
# Stopka: proweniencja przebiegu (MAIL-42)
# ---------------------------------------------------------------------------


def _run_footer_label(run_context: RunContext) -> str:
    """„przebieg #77 · commit abc1234” — logi repo są publiczne, więc stopka nie
    niesie nic poza numerem przebiegu, krótkim SHA i linkiem."""
    return f"przebieg #{run_context.run_id} · commit {run_context.short_sha}"


def _render_run_footer_html(run_context: RunContext | None) -> str:
    if run_context is None:
        return ""
    label = _html(_run_footer_label(run_context))
    url = _safe_news_url(run_context.run_url)
    logs = (
        f' · <a href="{_html(url)}" style="color: {PALETTE.muted};">logi</a>'
        if url is not None
        else ""
    )
    return f"<br/>{label}{logs}"


def _render_run_footer_text(run_context: RunContext | None) -> str:
    if run_context is None:
        return ""
    url = _safe_news_url(run_context.run_url)
    logs = f" · logi: {url}" if url is not None else ""
    return "\n".join(["-" * 64, f"{_run_footer_label(run_context)}{logs}"])


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _calibration_suffix(sig: TradeSignal) -> str:
    """„ po kalibracji (n kubełka: 12)” — n pomijamy, gdy sygnał go nie niesie."""
    if sig.calibration_bucket_count is None:
        return " po kalibracji"
    return f" po kalibracji (n kubełka: {sig.calibration_bucket_count})"


def _confidence_html(sig: TradeSignal) -> str:
    """WIA-12: „pewność 85% → 58% po kalibracji (n kubełka: 12)” albo sama surowa."""
    raw = f"pewność <strong>{_pct(sig.confidence, decimals=0)}</strong>"
    if sig.calibrated_confidence is None:
        return raw
    return (
        f"{raw} → <strong>{_pct(sig.calibrated_confidence, decimals=0)}</strong>"
        f"{_html(_calibration_suffix(sig))}"
    )


def _confidence_text(sig: TradeSignal) -> str:
    raw = _pct(sig.confidence, decimals=0)
    if sig.calibrated_confidence is None:
        return raw
    return f"{raw} → {_pct(sig.calibrated_confidence, decimals=0)}{_calibration_suffix(sig)}"


def _strength_label(rank: int, sig: TradeSignal) -> str:
    """WIA-12: pozycja w rankingu i składowe siły zamiast gołej liczby.

    Siła to |ruch| × pewność (skalibrowana, gdy jest historia), a przy
    cold-starcie ML sama pewność — etykieta mówi, z czego wynika kolejność.
    """
    calibrated = sig.calibrated_confidence is not None
    effective = sig.confidence if sig.calibrated_confidence is None else sig.calibrated_confidence
    confidence_pct = _pct(effective, decimals=0)
    if sig.ml_cold_start:
        basis = "pewności skalibrowanej" if calibrated else "pewności"
        return f"#{rank} · wg {basis} {confidence_pct}"
    basis = "pewność skalibrowana" if calibrated else "pewność"
    move_pct = _pct(abs(sig.expected_change), decimals=1)
    return f"#{rank} · ruch {move_pct} × {basis} {confidence_pct}"


def _render_trade_signals_html(trade_signals: list[TradeSignal]) -> str:
    """🎯 Najsilniejsze sygnały transakcyjne. Pusty string, gdy brak sygnałów."""
    if not trade_signals:
        return ""
    parts = [_section_h2("🎯 Najsilniejsze sygnały")]
    if any(sig.ml_cold_start for sig in trade_signals):
        parts.append(
            "<div style='font-size: 12px; color: #6b7280; margin-bottom: 8px;'>"
            f"{_html(_SIGNALS_COLD_START_NOTE)}</div>"
        )
    for rank, sig in enumerate(trade_signals, start=1):
        dir_color = {
            "KUP": PALETTE.positive,
            "SPRZEDAJ": PALETTE.negative,
            "OBSERWUJ": PALETTE.neutral,
        }[sig.direction]
        dir_bg = {
            "KUP": "#f0fdf4",
            "SPRZEDAJ": "#fef2f2",
            "OBSERWUJ": "#f9fafb",
        }[sig.direction]
        forecast_html = (
            f'<span style="color: #6b7280;">prognoza <strong>{_COLD_START_FORECAST}</strong></span>'
            if sig.ml_cold_start
            else (
                f'<span style="color: {_delta_color(sig.expected_change)};">'
                f"prognoza <strong>{direction_pct(sig.expected_change)}</strong></span>"
            )
        )
        # Q7: pasmo wielkości pozycji (Kelly-lite z konsensusu rady + hit-rate).
        size_html = (
            f'<span style="color: #6b7280; font-size: 12px;">'
            f"{_html(sig.size_band.label)}</span>"
            if sig.size_band
            else ""
        )
        parts.append(f"""
              <div style="margin-bottom: 8px; padding: 10px 14px; background: {dir_bg};
                          border-left: 4px solid {dir_color}; border-radius: 4px;
                          font-size: 13px; display: flex; gap: 12px; align-items: center;">
                <span style="font-weight: 700; color: {dir_color}; min-width: 90px;">
                  {sig.direction}
                </span>
                <span style="font-weight: 600; min-width: 60px;">{_html(_company_label(sig.symbol))}</span>
                <span style="color: #4b5563;">
                  {_confidence_html(sig)}
                </span>
                {forecast_html}
                {size_html}
                <span style="color: #6b7280; margin-left: auto; font-size: 12px;">
                  {_html(_strength_label(rank, sig))}
                </span>
              </div>
            """)
    return "".join(parts)


def _render_risk_signals_html(risk_signals: list[RiskSignal]) -> str:
    """🚨 Sygnały ostrzegawcze. Pusty string, gdy brak sygnałów."""
    if not risk_signals:
        return ""
    parts = [
        "<h2 style='font-size: 16px; margin: 20px 0 8px 0; color: #991b1b;'>"
        "🚨 Sygnały ostrzegawcze</h2>"
    ]
    type_label = {
        "DIVERGENCE": "Rozbieżność cena ↔ sentyment",
        "AV_LLM_CONFLICT": "Niezgodność AV ↔ LLM",
        "LOW_SIGNAL": "Słaby sygnał",
    }
    for rs in risk_signals:
        sev_color = {"high": "#dc2626", "medium": "#f59e0b", "low": "#6b7280"}[rs.severity]
        parts.append(f"""
              <div style="margin-bottom: 6px; padding: 8px 12px; background: #fffbeb;
                          border-left: 3px solid {sev_color}; border-radius: 4px;
                          font-size: 12px;">
                <strong>{_html(_company_label(rs.symbol))}</strong>
                <span style="color: {sev_color}; font-weight: 600;">
                  · {type_label.get(rs.type, _html(rs.type))}
                </span><br/>
                <span style="color: #4b5563;">{_html(rs.description)}</span>
              </div>
            """)
    return "".join(parts)


# WIA-19: okno sekcji zamkniętych predykcji w jej nagłówku (HTML i tekst).
_RESOLVED_WINDOW_24H = "ostatnie 24h"
_RESOLVED_WINDOW_THIS_CYCLE = "rozliczone w tym cyklu"
# MAIL-25: linia zamiast pustego miejsca — zgodna z oknem w nagłówku sekcji.
_RESOLVED_EMPTY_NOTE = {
    _RESOLVED_WINDOW_24H: "Brak rozliczeń predykcji z ostatnich 24h.",
    _RESOLVED_WINDOW_THIS_CYCLE: "Brak rozliczeń w tym cyklu.",
}
_RESOLVED_UNAVAILABLE_NOTE = (
    "⚠️ Nie udało się odczytać rozliczeń — w tym mailu ich brakuje "
    "(szczegóły w logach przebiegu)."
)


def _render_resolved_predictions_html(
    resolved_predictions: list[ResolvedPrediction],
    window: str = _RESOLVED_WINDOW_24H,
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
        f"{_DIRECTIONAL_ACCURACY_LABEL}: {directional_txt} · "
        f"{_TOTAL_ACCURACY_LABEL}: {_pct(correct / total, decimals=0)} (n={total})"
    )


# WIA-29 (c): dwie definicje trafności obok siebie, każda nazwana tym, co liczy.
_DIRECTIONAL_ACCURACY_LABEL = "Trafność kierunkowa (bez prognoz „bez ruchu”)"
_TOTAL_ACCURACY_LABEL = "Trafność łączna (z prognozami „bez ruchu”)"


def _directional_accuracy(accuracy_stats: dict[str, Any]) -> tuple[str, str] | None:
    """(wartość, dopisek) trafności kierunkowej; None, gdy statystyki jej nie niosą."""
    n = accuracy_stats.get("directional_count")
    hits = accuracy_stats.get("directional_correct_count")
    if n is None or hits is None:
        return None
    if n == 0:
        return "—", "brak prognoz kierunkowych"
    return _pct(hits / n, decimals=1), wilson_label(hits, n)


def _render_accuracy_history_html(accuracy_stats: dict[str, Any] | None) -> str:
    """🎯 Historia trafności. Pusty string, gdy brak danych statystyk."""
    if accuracy_stats and accuracy_stats.get("mean_accuracy") is not None:
        acc = accuracy_stats["mean_accuracy"]
        n = accuracy_stats["sample_count"]
        correct = accuracy_stats["correct_count"]
        days = accuracy_stats["days_window"]
        # WIA-32: przedział Wilsona obok procentu — goły % jest nieweryfikowalny.
        interval = wilson_label(correct, n)
        directional = _directional_accuracy(accuracy_stats)
        directional_html = (
            f"{_DIRECTIONAL_ACCURACY_LABEL}: <strong>{directional[0]}</strong>"
            f" ({directional[1]})<br/>"
            if directional is not None
            else ""
        )
        return f"""
      <div style="padding: 12px 14px; background: #fef3c7; border-left: 3px solid #f59e0b;
                  border-radius: 4px; margin-bottom: 20px; font-size: 13px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: #92400e;">
          🎯 Historia trafności (ostatnie {days} dni)
        </div>
        <div style="color: #78350f;">
          {directional_html}
          {_TOTAL_ACCURACY_LABEL}: <strong>{_pct(acc, decimals=1)}</strong> ({interval})
          · Predykcji ocenionych: {n}
          · Poprawnych: {correct}
        </div>
      </div>
        """
    if accuracy_stats is not None:
        return """
      <div style="padding: 10px 14px; background: #f9fafb; border-radius: 4px;
                  margin-bottom: 20px; font-size: 12px; color: #6b7280;">
        🎯 Historia trafności: brak ocenionych predykcji (potrzeba ≥1 zamkniętego cyklu).
      </div>
        """
    return ""


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


def _is_rebaselined(r: SymbolResult) -> bool:
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
        if not _is_rebaselined(r) or r.reference_at is None:
            continue
        day = format_local_date(r.reference_at, display_timezone)
        by_day.setdefault(day, []).append(r.symbol)
        oldest[day] = min(oldest.get(day, r.reference_at), r.reference_at)
    return [
        f"{', '.join(by_day[day])}: {_rebaseline_label(day)}"
        for day in sorted(by_day, key=lambda d: oldest[d])
    ]


def _rebaseline_label(day: str) -> str:
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


def _render_ignored_html(ignored: list[SymbolResult]) -> str:
    """⏸ Symbole pominięte poniżej progu zmienności. Pusty string, gdy brak.
    FIN-14: symbole z nowym punktem odniesienia mają własną sekcję."""
    ignored = [r for r in ignored if not _is_rebaselined(r)]
    if not ignored:
        return ""
    rows = "".join(
        f"<span style='display: inline-block; padding: 2px 8px; margin: 2px; background: #f3f4f6; border-radius: 3px; font-size: 12px;'>"
        f"<strong>{_html(r.symbol)}</strong> <span style='color: {_delta_color(r.delta)}'>{direction_pct(r.delta)}</span></span>"
        for r in ignored
    )
    return (
        _section_h2("⏸ Pominięte (poniżej progu zmienności)")
        + f"<div>{rows}</div>"
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
              <div style="margin-bottom: 8px; padding: 10px; background: #fef2f2; border-left: 3px solid #dc2626; border-radius: 4px; font-size: 12px;">
                <strong>{_html(_company_label(r.symbol))}</strong>: {_html(r.error_message)}
              </div>
            """)
    return "".join(parts)


def _render_prediction_row_html(r: SymbolResult) -> str:
    """Pojedynczy wiersz tabeli predykcji (saved)."""
    sentiment_text = (
        f"{_html(_sentiment_label(r.sentiment_label))} ({_number(r.sentiment_score, decimals=2, signed=True)})"
        if r.sentiment_score is not None
        else "—"
    )
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
    rec = _model_signal(r)
    # MAIL-06 (c): układ komórki i ton kierunku w klasach arkusza.
    return f"""
              <tr>
                <td class="pc b6">{_html(_company_label(r.symbol))}</td>
                <td class="pc">{_money(r.current_price)}</td>
                <td class="pc {delta_tone(r.delta)}">{direction_pct(r.delta)}</td>
                <td class="pc b6 {trend_tone(r.trend)}">{_html(_trend_label(r.trend))}</td>
                <td class="pc">{forecast_text}</td>
                <td class="pc">{confidence_text}</td>
                <td class="pc">{sentiment_text}</td>
                <td class="pc">{r.news_volume or 0}</td>
                <td class="pc b7 {SIGNAL_TONE.get(rec, "nt")}">{rec}</td>
              </tr>
            """


def _render_reasoning_block_html(r: SymbolResult, sector_map: Mapping[str, str]) -> str:
    """Blok 'Uzasadnienia' dla jednej predykcji. Pusty string, gdy nie ma
    czego pokazać (brak reasoning / rady / odznak / analogów / atrybucji)."""
    # Q5/Q6: blok renderujemy też, gdy są analogi (precedent receipts)
    # lub odznaki proweniencji — nawet bez reasoning/rady.
    badges_html = _provenance_badges_html(r.provenance_badges)
    precedents_block = _render_precedents_block_html(r.similar_precedents)
    attribution_block = _render_attribution_block_html(r.feature_attribution)
    if (
        not r.reasoning
        and r.council_verdict is None
        and not badges_html
        and not precedents_block
        and not attribution_block
        and r.move_interval is None
    ):
        return ""
    move_line = ""
    if r.current_price is not None and r.target_price is not None:
        conf_part = (
            f" · Pewność {_pct(r.confidence_score, decimals=0)}"
            if r.confidence_score is not None
            else ""
        )
        move_text = (
            _COLD_START_FORECAST
            if r.ml_cold_start
            else (
                f"{_money(r.current_price)} → {_money(r.target_price)} "
                f"<span class='{delta_tone(r.expected_change)}'>"
                f"({direction_pct(r.expected_change)})</span>"
            )
        )
        move_line = f'<div class="rm">Oczekiwany ruch: {move_text}{conf_part}</div>'
    if r.move_interval is not None:
        move_line += f'<div class="rm">{_move_interval_label(r.move_interval)}</div>'
    # Blok 'Top newsy' — zmigrowany na autoescapowany Jinja env
    # (finding #35). Escapowanie title/source jest tu domyślne.
    news_block = _render_news_block_html(r.top_news)
    rec_block = _model_signal_reason_html(r)
    council_block = (
        _render_council_section(r.council_verdict)
        if r.council_verdict is not None
        else ""
    )
    valuation_block = _render_valuation(r.valuation)
    _sector = _sector_label(r.symbol, sector_map)
    sector_tag = (
        f' <span style="color: #6b7280; font-weight: 500;">· {_html(_sector)}</span>'
        if _sector
        else ""
    )
    return f"""
              <div class="rb {EDGE_TONE[trend_tone(r.trend)]}">
                <div class="rh">{_html(_company_label(r.symbol))}{sector_tag} <span class="{trend_tone(r.trend)}">{_html(_trend_label(r.trend))}</span>{badges_html}</div>
                {move_line}
                {rec_block}
                {f'<div class="rt">{_html(r.reasoning)}</div>' if r.reasoning else ''}
                {attribution_block}
                {precedents_block}
                {news_block}
                {council_block}
                {valuation_block}
              </div>
            """


def _render_saved_section_html(
    saved: list[SymbolResult],
    full_detail: frozenset[str] | None = None,
    *,
    sector_map: Mapping[str, str],
) -> str:
    """🔮 Sekcja wygenerowanych predykcji: tabela + wykres prognozy +
    uzasadnienia. Pusty string, gdy w tym cyklu nic nie zapisano.

    `full_detail` (MAIL-06): symbole z pełnym blokiem uzasadnień; reszta trafia
    do skrótu (jedna linia rady, zawsze widoczni dysydenci). None = każdy symbol."""
    if not saved:
        return ""
    parts = [
        _section_h2("🔮 Wygenerowane predykcje"),
        "<table class='pt'>",
        f"""
          <tr class="pr">
            <th scope="col" class="ph">Symbol</th>
            <th scope="col" class="ph">Cena</th>
            <th scope="col" class="ph">Zmiana (cykl)</th>
            <th scope="col" class="ph">Trend</th>
            <th scope="col" class="ph">Prognoza (nast. cykl)</th>
            <th scope="col" class="ph">Pewność</th>
            <th scope="col" class="ph">Sentyment</th>
            <th scope="col" class="ph">Newsy</th>
            <th scope="col" class="ph">{MODEL_SIGNAL_TITLE}</th>
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

    parts.extend(
        _render_reasoning_block_html(r, sector_map) for r in saved if has_full_block(r)
    )
    parts.append(
        _render_compact_predictions_html([r for r in saved if not has_full_block(r)])
    )
    return "".join(parts)


_COMPACT_PREDICTIONS_NOTE = (
    f"Pełne uzasadnienie: top {_FULL_DETAIL_TOP_N} sygnałów i lead. Reszta w skrócie "
    "(limit rozmiaru Gmaila) — pełne w wersji tekstowej maila."
)


def _render_compact_predictions_html(rest: list[SymbolResult]) -> str:
    """MAIL-06 (d): pozostałe predykcje jako zwarte wiersze — symbol, sygnał modelu
    i jedna linia rady. Pusty string, gdy wszystko ma pełny blok."""
    if not rest:
        return ""
    rows = []
    for r in rest:
        signal = _model_signal(r)
        council = (
            _render_council_line(r.council_verdict) if r.council_verdict is not None else "—"
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


# ---------------------------------------------------------------------------
# MAIL-16: pierwszy ekran — liczby dla czytelnika, liczniki do stopki
# ---------------------------------------------------------------------------

# Okno rozliczeń w boxie pierwszego ekranu — krócej niż w nagłówku sekcji.
_SETTLEMENT_WINDOW_SHORT = {
    _RESOLVED_WINDOW_24H: "ostatnie 24h",
    _RESOLVED_WINDOW_THIS_CYCLE: "w tym cyklu",
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
    return (
        f'<div style="flex: 1; padding: 12px; background: {background}; '
        f'border-left: 3px solid {accent}; border-radius: 4px;">'
        '<div style="font-size: 11px; color: #4b5563; text-transform: uppercase;">'
        f"{_html(label)}</div>"
        '<div style="font-size: 24px; font-weight: 600; color: #1f2937;">'
        f"{_html(value)}</div>"
        f'<div style="font-size: 12px; color: #6b7280;">{_html(note)}</div>'
        "</div>"
    )


def _render_technical_footer_html(
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    duration_seconds: float,
) -> str:
    """MAIL-16: liczniki pracy pipeline'u w stopce — błąd nadal jest czerwony."""
    errors_color = PALETTE.negative if errors else PALETTE.muted
    return (
        '<div style="margin-bottom: 8px; color: #6b7280;">'
        f"Symboli: <strong>{len(results)}</strong>"
        f" · Predykcje: <strong>{len(saved)}</strong>"
        f" · Pominięte: <strong>{len(ignored)}</strong>"
        f' · <span style="color: {errors_color};">Błędy: <strong>{len(errors)}</strong></span>'
        f" · Czas trwania: {_number(duration_seconds, decimals=1)}s"
        "</div>"
    )


def _technical_footer_text(
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    duration_seconds: float,
) -> str:
    return (
        f"Symboli: {len(results)} · predykcji: {len(saved)} · "
        f"pominiętych: {len(ignored)} · błędów: {len(errors)} · "
        f"czas trwania: {_number(duration_seconds, decimals=1)}s"
    )


# ---------------------------------------------------------------------------
# MAIL-17: dwie strefy — „Na dziś” przed „Zapleczem”
# ---------------------------------------------------------------------------

_BACKOFFICE_TITLE = "🗄️ Zaplecze"
_BACKOFFICE_NOTE = (
    "Kontekst i kontrola: ryzyko, skuteczność modelu i rady, nastroje, koszt cyklu, "
    "pominięte symbole i błędy."
)
_BACKOFFICE_SEPARATOR_HTML = (
    '<div style="margin: 32px 0 8px 0; padding-top: 12px; border-top: 2px solid #d1d5db;">'
    f'<div style="font-size: 18px; font-weight: 700; color: #374151;">{_BACKOFFICE_TITLE}</div>'
    f'<div style="font-size: 12px; color: #6b7280;">{_BACKOFFICE_NOTE}</div>'
    "</div>"
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


def _render_mood_html(mood: dict[str, Any]) -> str:
    """📊 Nastroje portfela — sentyment cyklu w Zapleczu."""
    bullish_part = (
        f"<strong>{_html(_company_label(mood['most_bullish'].symbol))}</strong> "
        f"({_number(mood['most_bullish'].sentiment_score, decimals=2, signed=True)})"
        if mood["most_bullish"] else "—"
    )
    bearish_part = (
        f"<strong>{_html(_company_label(mood['most_bearish'].symbol))}</strong> "
        f"({_number(mood['most_bearish'].sentiment_score, decimals=2, signed=True)})"
        if mood["most_bearish"] else "—"
    )
    return f"""
      <div style="padding: 12px 14px; background: #eff6ff; border-left: 3px solid #2563eb;
                  border-radius: 4px; margin-bottom: 20px; font-size: 13px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: #1e3a8a;">
          📊 Nastroje portfela
        </div>
        <div style="color: #1e40af; line-height: 1.7;">
          Średni sentyment: <strong>{_number(mood['avg_sentiment'], decimals=2, signed=True)}</strong>
          ({mood['avg_sentiment_label']})<br/>
          Najbardziej pozytywny: {bullish_part} · Najbardziej negatywny: {bearish_part}<br/>
          Predykcje z wysoką pewnością (≥70%): {mood['high_confidence_count']} / {mood['saved_count']}
          · Łącznie newsów: {mood['total_news']}
        </div>
      </div>
    """


def _render_html(
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    started_local: str,
    duration_seconds: float,
    mood: dict[str, Any],
    session: dict[str, str],
    accuracy_stats: dict[str, Any] | None,
    trade_signals: list[TradeSignal],
    risk_signals: list[RiskSignal],
    resolved_predictions: list[ResolvedPrediction],
    *,
    resolved_window: str = _RESOLVED_WINDOW_24H,
    resolved_note: str | None = None,
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE,
    portfolio_change_box: str = "",
    settlements_unavailable: bool = False,
    full_detail: frozenset[str] | None = None,
    sector_map: Mapping[str, str],
) -> str:
    sections = []

    # Header
    sections.append(f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; max-width: 720px; margin: 0 auto; color: #1f2937;">
      <h1 style="font-size: 22px; margin: 0 0 4px 0;">📊 StockAgent — raport cyklu</h1>
      <div style="font-size: 13px; color: #6b7280; margin-bottom: 24px;">
        Czas rozpoczęcia: {started_local}
      </div>
    """)

    # Slot bannera kwot — wypełniany przez build_html_report, gdy są alerty.
    # Trafia NAJWYŻEJ (przed sesją), bo wyczerpanie limitu to top-priority info.
    sections.append(_QUOTA_BANNER_SLOT)
    # #1: lead cyklu — pierwszy ekran, tuż za bannerem kwot.
    sections.append(_LEAD_SLOT)
    sections.append(_ONBOARDING_SLOT)

    # Status sesji giełdowej
    sections.append(f"""
      <div style="padding: 10px 14px; background: #f9fafb; border-radius: 4px;
                  margin-bottom: 16px; font-size: 13px;">
        <strong>Sesja NYSE:</strong> {session["label"]}
        <span style="color: #6b7280;">· {session["detail"]}</span>
      </div>
    """)

    # MAIL-16: pierwszy ekran — trzy liczby dla czytelnika: zmiana portfela
    # (tylko właściciel z pozycjami, PORT-06; inaczej box się chowa), sygnały
    # ponad próg i trafność rozliczeń. Liczniki operacyjne są w stopce.
    settlements = _settlement_tally(
        resolved_predictions, unavailable=settlements_unavailable, window=resolved_window
    )
    signals_box = _render_reader_box_html(
        "Sygnały ponad próg",
        str(_signals_above_threshold(saved)),
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
    sections.append(
        '<div style="display: flex; gap: 8px; margin-bottom: 16px;">'
        f"{portfolio_change_box}{signals_box}{settlements_box}</div>"
    )

    # MAIL-17: strefa „Na dziś” — pasek wiarygodności zamiast Track Recordu na
    # górze, potem portfel, zmiany od wczoraj, sygnały, predykcje i krypto.
    sections.append(_render_credibility_bar_html(accuracy_stats))
    sections.append(_USER_PORTFOLIO_SLOT)
    # Zmiany od wczoraj: rozliczenia, zmiany nastawienia, nowy punkt odniesienia.
    sections.append(
        _render_resolved_predictions_html(
            resolved_predictions, resolved_window, resolved_note, sector_map=sector_map
        )
    )
    sections.append(_CONSENSUS_SHIFT_SLOT)
    sections.append(_render_rebaselined_html(ignored, display_timezone))
    sections.append(_render_trade_signals_html(trade_signals))
    sections.append(_render_risk_signals_html(risk_signals))
    sections.append(_render_saved_section_html(saved, full_detail, sector_map=sector_map))
    # 🪙 Krypto — osobna sekcja, widoczna też gdy cykl 'ignored' (próg 5%).
    sections.append(_render_crypto_section_html(results, display_timezone))

    # MAIL-17: „Zaplecze” — kontekst i kontrola za separatorem. Przy przycięciu
    # przez Gmaila znika ta część, nie akcyjna.
    sections.append(_BACKOFFICE_SEPARATOR_HTML)
    # Slot na sekcję Risk Watch — wypełniany przez build_html_report,
    # gdy wywołujący przekaże MacroRiskReport. Bez slotu sekcja jest pomijana.
    sections.append(_RISK_WATCH_SLOT)
    # Slot na radar korelacji portfela (Q4) — wypełniany przez build_html_report.
    sections.append(_PORTFOLIO_SLOT)
    # Slot na sekcję Alpha Signals (Dane — insider/analitycy/opcje/social/FRED).
    sections.append(_ALPHA_SLOT)
    # Slot na sekcję Track Record (T1 equity curve, T2 calibration, T4 lessons).
    sections.append(_TRACK_RECORD_SLOT)
    sections.append(_render_accuracy_history_html(accuracy_stats))
    sections.append(_MODEL_SCORECARD_SLOT)
    # Slot na ranking wiarygodności person (#3) — "kto z rady miał rację".
    sections.append(_PERSONA_LEADERBOARD_SLOT)
    # Slot na panel historii głosów rady (U5).
    sections.append(_COUNCIL_HISTORY_SLOT)
    sections.append(_render_mood_html(mood))
    # #2: wykres Δ cen jako inline HTML (żadne dane nie opuszczają maila).
    sections.append(build_delta_chart_html(results))
    # Self-Reflection — wnioski z poprzednich cykli
    sections.append(_render_reflections_html(saved))
    sections.append(_SUGGESTIONS_SLOT)
    # #2: dawny scatter → tabela kwadrantów, inline (≥3 punkty).
    correlation_chart = build_correlation_chart_html(results)
    if correlation_chart:
        sections.append(
            '<h3 style="font-size: 14px; margin: 20px 0 8px 0;">'
            "📈 Korelacja sentymentu z ruchem ceny</h3>"
        )
        sections.append(correlation_chart)
    sections.append(_FINOPS_SLOT)
    sections.append(_render_ignored_html(ignored))
    sections.append(_render_errors_html(errors))

    # MAIL-08: zastrzeżenie w stałej stopce, nie w sekcji warunkowej — sekcja,
    # która się chowa, zabrałaby je ze sobą.
    sections.append(f"""
      <div style="margin-top: 32px; padding-top: 16px; border-top: 1px solid #e5e7eb; font-size: 11px; color: {PALETTE.muted};">
        <div style="margin-bottom: 8px; color: #6b7280;">{_html(INVESTMENT_DISCLAIMER)}</div>
        {_render_technical_footer_html(results, saved, ignored, errors, duration_seconds)}
        StockAgent · Agentic RAG + Continual Learning · raport automatyczny
        {_RUN_FOOTER_SLOT}
      </div>
    </div>
    """)

    return "".join(sections)


# ---------------------------------------------------------------------------
# Plain text fallback
# ---------------------------------------------------------------------------


def _render_trade_signals_text(trade_signals: list[TradeSignal]) -> list[str]:
    """Sekcja 'Najsilniejsze sygnały' (plain). Pusta lista, gdy brak sygnałów."""
    if not trade_signals:
        return []
    lines = ["NAJSILNIEJSZE SYGNAŁY", "-" * 64]
    if any(sig.ml_cold_start for sig in trade_signals):
        lines.append(f"  {_SIGNALS_COLD_START_NOTE}")
    for rank, sig in enumerate(trade_signals, start=1):
        size_txt = f"  [{sig.size_band.label}]" if sig.size_band else ""
        forecast = (
            _COLD_START_FORECAST if sig.ml_cold_start else _pct(sig.expected_change, signed=True)
        )
        lines.append(
            f"  {sig.direction:9s} {_company_label(sig.symbol):40s} "
            f"pewność {_confidence_text(sig)}  "
            f"prognoza {forecast:>8s}  "
            f"{_strength_label(rank, sig)}{size_txt}"
        )
    lines.append("")
    return lines


def _render_risk_signals_text(risk_signals: list[RiskSignal]) -> list[str]:
    """Sekcja 'Sygnały ostrzegawcze' (plain). Pusta lista, gdy brak."""
    if not risk_signals:
        return []
    lines = ["SYGNAŁY OSTRZEGAWCZE", "-" * 64]
    type_label = {
        "DIVERGENCE": "Rozbieżność",
        "AV_LLM_CONFLICT": "Niezgodność AV↔LLM",
        "LOW_SIGNAL": "Słaby sygnał",
    }
    for rs in risk_signals:
        lines.append(
            f"  ⚠ {_company_label(rs.symbol)}: [{type_label.get(rs.type, rs.type)}] {rs.description}"
        )
    lines.append("")
    return lines


def _render_resolved_predictions_text(
    resolved_predictions: list[ResolvedPrediction],
    window: str = _RESOLVED_WINDOW_24H,
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
    lines.append(
        f"  Suma: {correct}/{len(resolved_predictions)} · "
        f"{_resolved_accuracy_pair(resolved_predictions)}"
    )
    lines.append("")
    return lines


def _render_accuracy_history_text(accuracy_stats: dict[str, Any] | None) -> list[str]:
    """Sekcja 'Historia trafności' (plain). Pusta lista, gdy brak danych."""
    if not (accuracy_stats and accuracy_stats.get("mean_accuracy") is not None):
        return []
    lines = [
        f"HISTORIA TRAFNOŚCI (ostatnie {accuracy_stats['days_window']} dni)",
        "-" * 64,
    ]
    directional = _directional_accuracy(accuracy_stats)
    if directional is not None:
        lines.append(f"  {_DIRECTIONAL_ACCURACY_LABEL}: {directional[0]} ({directional[1]})")
    lines.extend([
        f"  {_TOTAL_ACCURACY_LABEL}: {_pct(accuracy_stats['mean_accuracy'], decimals=1)}"
        f" ({wilson_label(accuracy_stats['correct_count'], accuracy_stats['sample_count'])})"
        f"  ·  Predykcji ocenionych: {accuracy_stats['sample_count']}"
        f"  ·  Poprawnych: {accuracy_stats['correct_count']}",
        "",
    ])
    return lines


def _render_mood_text(mood: dict[str, Any]) -> list[str]:
    """Sekcja 'Nastroje portfela' (plain) — w Zapleczu, jak w HTML."""
    lines = [
        "NASTROJE PORTFELA",
        "-" * 64,
        f"  Średni sentyment: {_number(mood['avg_sentiment'], decimals=2, signed=True)} ({mood['avg_sentiment_label']})",
    ]
    if mood["most_bullish"]:
        mb = mood["most_bullish"]
        lines.append(f"  Najbardziej pozytywny: {_company_label(mb.symbol)} ({_number(mb.sentiment_score, decimals=2, signed=True)})")
    if mood["most_bearish"]:
        mbe = mood["most_bearish"]
        lines.append(f"  Najbardziej negatywny: {_company_label(mbe.symbol)} ({_number(mbe.sentiment_score, decimals=2, signed=True)})")
    lines.append(
        f"  Wysoka pewność (≥70%): {mood['high_confidence_count']} / {mood['saved_count']}"
        f"  ·  Newsów: {mood['total_news']}"
    )
    lines.append("")
    return lines


def _render_reflections_text(saved: list[SymbolResult]) -> list[str]:
    """Sekcja 'Wnioski z poprzednich cykli' (plain). Pusta lista, gdy brak."""
    reflections = [r for r in saved if r.reflection_insight]
    if not reflections:
        return []
    lines = ["WNIOSKI Z POPRZEDNICH CYKLI (Self-Reflection)", "-" * 64]
    lines.extend(
        f"  {_company_label(r.symbol)}: {r.reflection_insight}" for r in reflections
    )
    lines.append("")
    return lines


def _render_prediction_text(r: SymbolResult, sector_map: Mapping[str, str]) -> list[str]:
    """Linie jednej predykcji (plain): nagłówek + sentyment + sygnał modelu +
    proweniencja + reasoning + analogi + newsy + wycena."""
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
    rec_text = _model_signal_reason_text(r)
    lines = [
        f"  {_company_label_with_sector(r.symbol, sector_map):55s} {_money(r.current_price):>10s}  "
        f"Δ {_pct(r.delta, signed=True):>8s}  →  "
        f"{_trend_label(r.trend):11s} {forecast_part}{conf_part}",
        f"         {sentiment_part}",
        f"         {rec_text}",
    ]
    if r.move_interval is not None:
        lines.append(f"         {_move_interval_label(r.move_interval)}")
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
            f" (rel={_number(n.relevance, decimals=2)}, sent={_number(n.sentiment, decimals=2, signed=True)})"
        )
    valuation_text = _render_valuation_text(r.valuation)
    if valuation_text:
        lines.append(valuation_text)
    return lines


def _render_saved_section_text(
    saved: list[SymbolResult], sector_map: Mapping[str, str]
) -> list[str]:
    """Sekcja 'Predykcje' (plain). Pusta lista, gdy nic nie zapisano."""
    if not saved:
        return []
    lines = ["PREDYKCJE", "-" * 64]
    for r in saved:
        lines.extend(_render_prediction_text(r, sector_map))
    lines.append("")
    return lines


def _render_ignored_text(ignored: list[SymbolResult]) -> list[str]:
    """Sekcja 'Pominięte' (plain). Pusta lista, gdy brak. FIN-14: bez symboli
    z nowym punktem odniesienia (mają własną sekcję)."""
    ignored = [r for r in ignored if not _is_rebaselined(r)]
    if not ignored:
        return []
    ignored_str = ", ".join(f"{r.symbol}({_pct(r.delta)})" for r in ignored)
    return ["POMINIĘTE (poniżej progu zmienności)", "-" * 64, f"  {ignored_str}", ""]


def _render_errors_text(errors: list[SymbolResult]) -> list[str]:
    """Sekcja 'Błędy' (plain). Pusta lista, gdy brak błędów."""
    if not errors:
        return []
    lines = ["BŁĘDY", "-" * 64]
    lines.extend(f"  {_company_label(r.symbol)}: {r.error_message}" for r in errors)
    lines.append("")
    return lines


def _render_plain(
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    started_local: str,
    duration_seconds: float,
    mood: dict[str, Any],
    session: dict[str, str],
    accuracy_stats: dict[str, Any] | None,
    trade_signals: list[TradeSignal],
    risk_signals: list[RiskSignal],
    resolved_predictions: list[ResolvedPrediction],
    *,
    resolved_window: str = _RESOLVED_WINDOW_24H,
    resolved_note: str | None = None,
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE,
    portfolio_change_line: str = "",
    settlements_unavailable: bool = False,
    sector_map: Mapping[str, str],
) -> str:
    lines: list[str] = []
    lines.append(_QUOTA_BANNER_SLOT)
    lines.append(_LEAD_SLOT)
    lines.append(_ONBOARDING_SLOT)
    lines.append("=" * 64)
    lines.append("STOCKAGENT — RAPORT CYKLU")
    lines.append("=" * 64)
    lines.append(f"Czas rozpoczęcia:  {started_local}")
    lines.append(f"Sesja NYSE:        {session['label']} — {session['detail']}")
    lines.append("")
    # MAIL-16: trzy liczby dla czytelnika (portfel tylko u właściciela, PORT-06);
    # liczniki operacyjne są w stopce technicznej.
    if portfolio_change_line:
        lines.append(portfolio_change_line)
    lines.append(f"Sygnały ponad próg: {_signals_above_threshold(saved)}")
    settlements = _settlement_tally(
        resolved_predictions, unavailable=settlements_unavailable, window=resolved_window
    )
    lines.append(f"Rozliczone: {settlements.text}")
    # MAIL-17: pasek wiarygodności — te same liczby co Historia trafności w Zapleczu.
    credibility = _credibility_label(accuracy_stats)
    if credibility is not None:
        lines.append(f"{_CREDIBILITY_TITLE}: {credibility} — szczegóły w Zapleczu")
    lines.append("")

    # MAIL-17: strefa „Na dziś”, kolejność jak w HTML.
    # 💼 Twój portfel — zaraz za pierwszym ekranem.
    lines.append(_USER_PORTFOLIO_SLOT)
    # Zmiany od wczoraj: rozliczenia i nowy punkt odniesienia.
    lines.extend(
        _render_resolved_predictions_text(
            resolved_predictions, resolved_window, resolved_note, sector_map=sector_map
        )
    )
    lines.extend(_render_rebaselined_text(ignored, display_timezone))
    lines.extend(_render_trade_signals_text(trade_signals))
    lines.extend(_render_risk_signals_text(risk_signals))
    lines.extend(_render_saved_section_text(saved, sector_map))
    # Krypto — osobna sekcja (widoczna też gdy 'ignored').
    crypto_text = _render_crypto_section_text(results, display_timezone)
    if crypto_text:
        lines.append(crypto_text)
        lines.append("")

    # MAIL-17: „Zaplecze” — kontekst i kontrola za separatorem.
    lines.extend(["=" * 64, "ZAPLECZE", _BACKOFFICE_NOTE, "=" * 64, ""])
    # Slot na sekcję Risk Watch w wariancie plain text — analogicznie do HTML.
    lines.append(_RISK_WATCH_SLOT)
    # PORT-22: radar portfela (klastry, VaR, stress, „za mało historii”).
    lines.append(_PORTFOLIO_SLOT)
    lines.extend(_render_accuracy_history_text(accuracy_stats))
    lines.extend(_render_mood_text(mood))
    # Wnioski z poprzednich cykli
    lines.extend(_render_reflections_text(saved))
    lines.append(_SUGGESTIONS_SLOT)
    lines.extend(_render_ignored_text(ignored))
    lines.extend(_render_errors_text(errors))

    # MAIL-08: zastrzeżenie zamyka treść; stopka techniczna (MAIL-16) i stopka
    # przebiegu (MAIL-42) idą za nim.
    lines.extend([
        "-" * 64,
        INVESTMENT_DISCLAIMER,
        _technical_footer_text(results, saved, ignored, errors, duration_seconds),
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: convert use case output → SymbolResult
# ---------------------------------------------------------------------------


def _skip_reason_of(status: str, raw: dict[str, Any]) -> SkipReason | None:
    """#4: cold-start jest widoczny w stanie końcowym jako previous_price == 0
    (graf nie miał punktu odniesienia). Symbol odcięty bramką volatility ma
    poprzednią cenę — to zupełnie inna historia niż pierwszy dzień deploymentu.
    Status inny niż "ignored" nie niesie powodu pominięcia."""
    if status != "ignored":
        return None
    previous_price = raw.get("previous_price")
    if previous_price is None or Decimal(str(previous_price)) == 0:
        return SkipReason.COLD_START
    # FIN-14: werdykt stęchłości wydała domena przy odczycie referencji.
    if raw.get("stale_reference") is True:
        return SkipReason.STALE_REFERENCE
    return SkipReason.BELOW_THRESHOLD


def _reference_at_of(raw: dict[str, Any]) -> datetime | None:
    """FIN-14: moment referencji ceny ze stanu grafu albo None (brak / zniekształcony)."""
    reference_at = raw.get("reference_at")
    return reference_at if isinstance(reference_at, datetime) else None


def _asset_class_of(raw: dict[str, Any]) -> str | None:
    """Klasa aktywa z domeny — graf trzyma Asset w stanie (main_agent klasyfikuje
    STOCK/ETF/CRYPTO przed wywołaniem). Pozwala raportowi rozpoznać krypto
    niezawodnie, niezależnie od statusu (saved/ignored)."""
    asset = raw.get("asset")
    if asset is None or not hasattr(asset, "asset_type"):
        return None
    value: str = asset.asset_type.value
    return value


def _valuation_section_of(raw: dict[str, Any]) -> ValuationSection | None:
    """Sekcja wyceny powstaje TYLKO przy rozstrzygniętym werdykcie i obecnych
    fundamentach — UNKNOWN albo brak danych to brak sekcji, nie sekcja pusta."""
    verdict = raw.get("valuation_verdict")
    fundamentals = raw.get("fundamentals")
    if (
        not isinstance(verdict, ValuationVerdict)
        or verdict is ValuationVerdict.UNKNOWN
        or fundamentals is None
    ):
        return None
    return ValuationSection(
        trailing_pe=fundamentals.trailing_pe,
        forward_pe=fundamentals.forward_pe,
        peg_ratio=fundamentals.peg_ratio,
        eps_growth_yoy=fundamentals.eps_growth_yoy,
        verdict=verdict,
        fetched_at=fundamentals.fetched_at,
    )


def to_symbol_result(
    symbol: str,
    raw: dict[str, Any] | None,
    error: str | None = None,
    *,
    now: datetime | None = None,
) -> SymbolResult:
    """Mapper z wyjścia AnalyzeMarketUseCase.run() na SymbolResult.

    `now` (ARCH-26) to zegar cyklu — orkiestrator podaje `started_at`, ten sam,
    który `build_html_report` drukuje w nagłówku — od niego liczy się wiek
    fundamentów w odznakach proweniencji. None = bieżący czas, jak przed ARCH-26.
    """
    if error is not None:
        return SymbolResult(symbol=symbol, status="error", error_message=error)
    if raw is None:
        return SymbolResult(symbol=symbol, status="error", error_message="No result")

    status = str(raw.get("status", "unknown"))
    sentiment = raw.get("sentiment") or {}
    llm = raw.get("llm_analysis") or {}

    skip_reason = _skip_reason_of(status, raw)
    asset_class = _asset_class_of(raw)
    valuation = _valuation_section_of(raw)
    fundamentals = raw.get("fundamentals")

    confidence = llm.get("confidence_score")
    av_agreement = llm.get("av_agreement")
    council_verdict = raw.get("council_verdict")

    # Q5: precedent receipts — kompaktowe analogi RAG z bieżącego cyklu.
    similar_precedents = _extract_precedents(raw.get("similar_precedents"))

    # Q6: odznaki proweniencji — z degraded_reason, wieku fundamentów (TTL) i
    # data_quality_flags. Cała ocena progu żyje w domenie (build_provenance_badges).
    fundamentals_fetched_at = (
        fundamentals.fetched_at if fundamentals is not None else None
    )
    provenance_badges = build_provenance_badges(
        degraded_reason=sentiment.get("degraded_reason"),
        fundamentals_fetched_at=fundamentals_fetched_at,
        data_quality_flags=raw.get("data_quality_flags") or [],
        now=now if now is not None else datetime.now(UTC),
        ttl_hours=FUNDAMENTALS_CACHE_TTL_HOURS,
    )

    return SymbolResult(
        symbol=symbol,
        status=status,
        skip_reason=skip_reason,
        reference_at=_reference_at_of(raw),
        delta=_as_decimal(raw.get("delta")),
        current_price=_as_decimal(raw.get("current_price")),
        trend=llm.get("trend_direction"),
        target_price=_as_decimal(raw.get("ml_target_price")),
        confidence_score=float(confidence) if confidence is not None else None,
        reasoning=llm.get("reasoning"),
        sentiment_score=sentiment.get("av_sentiment_score"),
        sentiment_label=sentiment.get("av_sentiment_label"),
        news_volume=sentiment.get("news_volume_24h"),
        av_llm_agreement=float(av_agreement) if av_agreement is not None else None,
        reflection_insight=_clean_reflection(raw.get("reflection_context")),
        top_news=_extract_top_news(raw.get("news") or []),
        council_verdict=council_verdict if isinstance(council_verdict, CouncilVerdict) else None,
        valuation=valuation,
        asset_class=asset_class,
        similar_precedents=similar_precedents,
        provenance_badges=provenance_badges,
        # Q3: atrybucja cech ML (SHAP-lite) — render-only.
        feature_attribution=tuple(raw.get("feature_attribution") or ()),
        ml_cold_start=bool(raw.get("ml_cold_start", False)),
        move_interval=_move_interval_of(raw),
    )


def _move_interval_of(raw: dict[str, Any]) -> MoveInterval | None:
    """ML-34: przedział z grafu albo None (brak / zniekształcony)."""
    interval = raw.get("move_interval")
    return interval if isinstance(interval, MoveInterval) else None


def _extract_precedents(raw_precedents: Any) -> list[SimilarPrecedent]:
    """Mapuje surowe rekordy analogów z grafu na DTO `SimilarPrecedent`.

    Wejście to lista dictów `{summary, predicted_trend, is_trend_correct}`
    z predict_node (`_build_similar_context`). Defensywnie pomijamy rekordy
    bez streszczenia — pusty/brak wejścia → pusta lista (raport bez bloku)."""
    if not isinstance(raw_precedents, list):
        return []
    out: list[SimilarPrecedent] = []
    for item in raw_precedents:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or "").strip()
        if not summary:
            continue
        out.append(
            SimilarPrecedent(
                summary=summary,
                predicted_trend=str(item.get("predicted_trend") or "?"),
                is_trend_correct=item.get("is_trend_correct"),
            )
        )
    return out


def _clean_reflection(raw_ctx: Any) -> str | None:
    """Filtruje cold-start placeholdery — pokazujemy reflection tylko gdy
    jest realny wniosek (LLM analyze_mistake lub potwierdzenie trafności)."""
    if not raw_ctx or not isinstance(raw_ctx, str):
        return None
    txt = raw_ctx.strip()
    if not txt:
        return None
    # Skip cold-start placeholder messages
    if "brak danych historycznych" in txt.lower():
        return None
    return txt


def _extract_top_news(news_items: list[dict[str, Any]], top_n: int = 2) -> list[TopNewsItem]:
    """Top N artykułów po relevance_score (z bonusem dla high-sentiment)."""
    valid = [
        item for item in news_items
        if item.get("relevance_score") is not None and item.get("title")
    ]
    sorted_news = sorted(valid, key=lambda x: float(x["relevance_score"] or 0), reverse=True)
    return [
        TopNewsItem(
            title=str(item["title"]),
            source=item.get("source"),
            url=item.get("url"),
            relevance=float(item.get("relevance_score") or 0),
            sentiment=float(item.get("ticker_sentiment_score") or 0),
        )
        for item in sorted_news[:top_n]
    ]


def _as_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None
