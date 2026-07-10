"""Buduje raport HTML+text z wyników cyklu Fast Loop.

Czysta logika prezentacji — bez I/O, łatwo testowalne.
"""

from __future__ import annotations

import html
import urllib.parse
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from src.application.alpha_signals import AlphaSignals
from src.application.report_alpha import render_alpha_html
from src.application.report_charts import (
    build_chart_url,
    build_correlation_chart_url,
    build_forecast_chart_url,
)
from src.application.report_consensus_shift import (
    render_consensus_shift_html,
)
from src.application.report_council_history import (
    InvestorHistory,
    render_council_history_html,
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
from src.application.report_portfolio import render_correlation_html
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
from src.application.report_suggestions import (
    build_sector_suggestions,
    render_suggestions_html,
    render_suggestions_text,
)
from src.application.report_templates import _env, render_template
from src.application.report_track_record import render_track_record_html
from src.application.use_cases.monitor_macro_risk import MacroRiskReport
from src.application.use_cases.portfolio_risk import PortfolioRiskReport
from src.domain.calibration_curve import CalibrationBucket
from src.domain.consensus_shift import ConsensusShift
from src.domain.council import CouncilVerdict
from src.domain.cycle_maturity import CycleMaturity, SkipReason
from src.domain.digest_lead import LeadSignal, build_lead
from src.domain.equity_curve import EquityCurve
from src.domain.feature_attribution import FeatureContribution
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.persona_track_record import PersonaTrackRecord
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
    "build_chart_url",
    "build_correlation_chart_url",
    "build_forecast_chart_url",
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

_DIRECTION_LABEL = {
    "BULLISH": "KUP",
    "BEARISH": "SPRZEDAJ",
    "SIDEWAYS": "WSTRZYMAJ",
}

_DIRECTION_COLOR = {
    "KUP": "#16a34a",
    "SPRZEDAJ": "#dc2626",
    "WSTRZYMAJ": "#737373",
}

_DIRECTION_BG = {
    "KUP": "#f0fdf4",
    "SPRZEDAJ": "#fef2f2",
    "WSTRZYMAJ": "#f9fafb",
}

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
_COUNCIL_HISTORY_SLOT = "<!-- COUNCIL_HISTORY_SLOT -->"
_PERSONA_LEADERBOARD_SLOT = "<!-- PERSONA_LEADERBOARD_SLOT -->"
# #8: "🔄 Zmiany nastawienia" — flipy rady i skoki sentymentu vs poprzedni cykl.
_CONSENSUS_SHIFT_SLOT = "<!-- CONSENSUS_SHIFT_SLOT -->"
_TRACK_RECORD_SLOT = "<!-- TRACK_RECORD_SLOT -->"
# #12: karta kondycji modelu — tuż za Track Recordem (ta sama oś zaufania).
_MODEL_SCORECARD_SLOT = "<!-- MODEL_SCORECARD_SLOT -->"
_ALPHA_SLOT = "<!-- ALPHA_SLOT -->"
_SUGGESTIONS_SLOT = "<!-- SUGGESTIONS_SLOT -->"

_COUNCIL_REC_LABEL = {"BUY": "KUP", "SELL": "SPRZEDAJ", "HOLD": "TRZYMAJ"}
_COUNCIL_REC_COLOR = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#ca8a04"}
_COUNCIL_REC_BG = {"BUY": "#f0fdf4", "SELL": "#fef2f2", "HOLD": "#fefce8"}
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
        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

    growth = (
        f"{section.eps_growth_yoy * 100:.1f}%"
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
        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

    growth = (
        f"{section.eps_growth_yoy * 100:.1f}%"
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


def _recommendation(r: SymbolResult) -> str:
    return _DIRECTION_LABEL.get(r.trend or "", "WSTRZYMAJ")


def _recommendation_reason_html(r: SymbolResult) -> str:
    """Zwraca krótki HTML-owy powód rekomendacji (1–2 zdania)."""
    direction = _recommendation(r)
    color = _DIRECTION_COLOR.get(direction, "#737373")
    bg = _DIRECTION_BG.get(direction, "#f9fafb")

    conf_part = (
        f"Pewność: <strong>{r.confidence_score * 100:.0f}%</strong>."
        if r.confidence_score is not None
        else ""
    )
    change_part = ""
    if r.expected_change is not None:
        change_part = (
            f" Prognozowana zmiana: "
            f"<strong style='color: {_delta_color(r.expected_change)};'>"
            f"{_pct(r.expected_change, signed=True)}</strong>."
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
        f"<div style='margin-top: 6px; padding: 6px 10px; background: {bg}; "
        f"border-radius: 3px; font-size: 12px;'>"
        f"<span style='font-weight: 700; color: {color};'>➤ {direction}</span> — "
        f"{conf_part}{change_part}{sentiment_part}"
        f"</div>"
    )


def _recommendation_reason_text(r: SymbolResult) -> str:
    direction = _recommendation(r)
    conf_part = f"pewność {r.confidence_score * 100:.0f}%" if r.confidence_score is not None else ""
    change_part = (
        f"prognoza {_pct(r.expected_change, signed=True)}"
        if r.expected_change is not None
        else ""
    )
    parts = [p for p in [conf_part, change_part] if p]
    detail = ", ".join(parts)
    return f"Rekomendacja: {direction}" + (f" ({detail})" if detail else "")


def _html(value: object) -> str:
    return html.escape(str(value), quote=True)


def _is_crypto(r: SymbolResult) -> bool:
    return r.asset_class == "CRYPTO"


def _render_crypto_section_html(results: list[SymbolResult]) -> str:
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
                f"{_pct(r.expected_change, signed=True)}</span>"
                if r.expected_change is not None
                else ""
            )
            conf_part = (
                f" · pewność {r.confidence_score * 100:.0f}%"
                if r.confidence_score is not None
                else ""
            )
            status_html = f"{trend_part}{forecast_part}{conf_part}"
        else:
            status_html = (
                "<span style='color: #6b7280;'>poniżej progu zmienności</span>"
            )
        rows.append(f"""
          <div style="margin-bottom: 6px; padding: 8px 12px; background: #fdf4ff;
                      border-left: 3px solid #a855f7; border-radius: 4px; font-size: 13px;">
            <strong>{_html(_company_label(r.symbol))}</strong>
            <span style="color: #4b5563;">· {_money(r.current_price)}
              · Δ <span style="color: {_delta_color(r.delta)};">{_pct(r.delta, signed=True)}</span>
              · {status_html}</span>
          </div>
        """)
    return (
        _section_h2("🪙 Krypto")
        + "".join(rows)
    )


def _render_crypto_section_text(results: list[SymbolResult]) -> str:
    cryptos = [r for r in results if _is_crypto(r) and r.status != "error"]
    if not cryptos:
        return ""
    lines = ["KRYPTO", "-" * 64]
    for r in cryptos:
        if r.status == "saved":
            status = (
                f"{_trend_label(r.trend)} "
                f"prognoza {_pct(r.expected_change, signed=True)}"
                if r.expected_change is not None
                else _trend_label(r.trend)
            )
        else:
            status = "poniżej progu"
        lines.append(
            f"  {_company_label(r.symbol):40s} {_money(r.current_price):>12s}  "
            f"Δ {_pct(r.delta, signed=True):>8s}  {status}"
        )
    return "\n".join(lines)


def _resolved_why(p: ResolvedPrediction) -> str:
    """Zwięzłe 'dlaczego' werdyktu: dla chybionych — diagnoza z reflect,
    dla trafionych — potwierdzenie tezy (insight 'Trafiona predykcja.' jest
    placeholderem, więc go nie pokazujemy dosłownie)."""
    if p.is_correct:
        return "Teza prognozy się potwierdziła."
    if p.insight and "trafiona" not in p.insight.lower():
        return p.insight
    return "Kierunek się nie potwierdził."


def _render_resolved_item_html(p: ResolvedPrediction) -> str:
    mark = "✅" if p.is_correct else "❌"
    color = "#16a34a" if p.is_correct else "#dc2626"
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
        move_txt = _pct(p.actual_change_pct, signed=True)
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
        {mark} <strong>{_html(_company_label_with_sector(p.symbol))}</strong>
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
            "label": _COUNCIL_REC_LABEL.get(op.recommendation, op.recommendation),
            "emoji": _COUNCIL_REC_EMOJI.get(op.recommendation, "⬜"),
            "color": _COUNCIL_REC_COLOR.get(op.recommendation, "#737373"),
            "bg": _COUNCIL_REC_BG.get(op.recommendation, "#f9fafb"),
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
            "label": _COUNCIL_REC_LABEL.get(final, final),
            "color": _COUNCIL_REC_COLOR.get(final, "#737373"),
            "bg": _COUNCIL_REC_BG.get(final, "#f9fafb"),
            "strength_pct": int(verdict.consensus_strength * 100),
            "opinion_rows": opinion_rows,
            "vote_dist": verdict.vote_distribution(),
            "is_split": verdict.is_split_decision(),
            "strong_consensus": verdict.has_strong_consensus(),
            "dissenting_views": verdict.dissenting_views,
        },
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
    "<div style='font-size: 11px; color: #4b5563; margin-top: 8px;'>"
    "📰 <strong>Top newsy:</strong>"
    "<ul style='margin: 4px 0 0 20px; padding: 0;'>"
    "{%- for n in items -%}"
    "<li style='margin: 2px 0;'>"
    "<strong>[{{ n.source_label }}]</strong> "
    "{% if n.href %}"
    "<a href='{{ n.href }}' target='_blank' "
    "style='color: #2563eb; text-decoration: none;'>{{ n.title }}</a>"
    "{% else %}{{ n.title }}{% endif %}"
    " <span style='color: #6b7280;'>"
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
            "relevance_fmt": f"{n.relevance:.2f}",
            "sentiment_fmt": f"{n.sentiment:+.2f}",
        }
        for n in top_news
    ]
    return _NEWS_BLOCK_TEMPLATE.render(items=items)


# Q5: etykieta wyniku analogu (kierunkowo trafiony / chybiony / nieoceniony).
_PRECEDENT_OUTCOME = {
    True: ("✅ trafił", "#16a34a"),
    False: ("❌ chybił", "#dc2626"),
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
        color = "#16a34a" if positive else "#dc2626"
        sign = "+" if positive else ""
        chips.append(
            f"<span style='display:inline-block; margin-right:10px;'>"
            f"{_html(c.feature)} "
            f"<strong style='color:{color};'>{sign}{c.contribution:.4f}</strong></span>"
        )
    return (
        "<div style='font-size: 11px; color: #475569; margin-top: 6px; "
        "padding: 4px 8px; background: #f8fafc; border-radius: 3px;'>"
        "🔍 <strong>Wkład cech:</strong> " + "".join(chips) + "</div>"
    )


def to_lead_signals(
    results: list[SymbolResult], resolved: list[ResolvedPrediction]
) -> list[LeadSignal]:
    """Mapuje DTO raportu na domenowe `LeadSignal` (#1).

    Granica architektury: `src/domain/digest_lead.py` bierze wyłącznie typy
    domenowe i prymitywy, więc `resolved_label` powstaje TU jako gotowy string.
    Symbole bez zmiany ceny nie wnoszą nic do rankingu i są pomijane.
    """
    labels = {
        r.symbol: (
            f"trafiona predykcja {r.predicted_trend}"
            if r.is_correct
            else f"chybiona predykcja {r.predicted_trend}"
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
                price_delta_pct=r.delta,
                council_verdict=r.council_verdict,
                resolved_label=labels.get(r.symbol),
            )
        )
    return signals


def _fill_html_slots(
    html: str,
    *,
    quota_html: str,
    macro_risk_html: str,
    portfolio_html: str,
    lead_html: str,
    onboarding_html: str,
    council_history_html: str,
    persona_leaderboard_html: str,
    consensus_shift_html: str,
    model_scorecard_html: str,
    track_record_html: str,
    alpha_html: str,
    suggestions_html: str,
) -> str:
    """Wstawia treść w sloty HTML. QUOTA i RISK_WATCH podmieniamy tylko gdy
    niepuste (pusty komentarz HTML jest nieszkodliwy); pozostałe sloty zawsze
    (też na ""), żeby znacznik nie został w treści."""
    if quota_html:
        html = html.replace(_QUOTA_BANNER_SLOT, quota_html, 1)
    html = html.replace(_LEAD_SLOT, lead_html, 1)
    html = html.replace(_ONBOARDING_SLOT, onboarding_html, 1)
    if macro_risk_html:
        html = html.replace(_RISK_WATCH_SLOT, macro_risk_html, 1)
    html = html.replace(_PORTFOLIO_SLOT, portfolio_html, 1)
    html = html.replace(_COUNCIL_HISTORY_SLOT, council_history_html, 1)
    html = html.replace(_PERSONA_LEADERBOARD_SLOT, persona_leaderboard_html, 1)
    html = html.replace(_CONSENSUS_SHIFT_SLOT, consensus_shift_html, 1)
    html = html.replace(_TRACK_RECORD_SLOT, track_record_html, 1)
    html = html.replace(_MODEL_SCORECARD_SLOT, model_scorecard_html, 1)
    html = html.replace(_ALPHA_SLOT, alpha_html, 1)
    return html.replace(_SUGGESTIONS_SLOT, suggestions_html, 1)


def _fill_text_slots(
    text: str,
    *,
    quota_text: str,
    macro_risk_text: str,
    lead_text: str,
    onboarding_text: str,
    suggestions_text: str,
) -> str:
    """Wstawia treść w sloty plain-text. QUOTA/RISK_WATCH tylko gdy niepuste;
    LEAD i SUGGESTIONS usuwają cały wiersz znacznika, gdy pusto."""
    if quota_text:
        text = text.replace(_QUOTA_BANNER_SLOT, quota_text, 1)
    if macro_risk_text:
        text = text.replace(_RISK_WATCH_SLOT, macro_risk_text, 1)
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


def build_html_report(
    results: list[SymbolResult],
    started_at: datetime,
    duration_seconds: float,
    accuracy_stats: dict[str, Any] | None = None,
    resolved_predictions: list[ResolvedPrediction] | None = None,
    macro_risk_report: MacroRiskReport | None = None,
    portfolio_risk_report: PortfolioRiskReport | None = None,
    council_history: list[InvestorHistory] | None = None,
    persona_track_record: list[PersonaTrackRecord] | None = None,
    quota_alerts: list[QuotaAlert] | None = None,
    symbols_filter: frozenset[str] | None = None,
    equity_curve: EquityCurve | None = None,
    calibration_buckets: list[CalibrationBucket] | None = None,
    lessons: LessonsReport | None = None,
    alpha_signals: dict[str, AlphaSignals] | None = None,
    yield_curve: YieldCurveSnapshot | None = None,
    alpha_prices: dict[str, Decimal] | None = None,
    signal_calibration_buckets: list[CalibrationBucket] | None = None,
    cycle_maturity: CycleMaturity | None = None,
    portfolio_sectors: Mapping[str, int] | None = None,
    model_scorecard: ModelScorecardView | None = None,
    consensus_shifts: list[tuple[str, ConsensusShift]] | None = None,
) -> tuple[str, str]:
    """Zwraca (html_body, plain_text) — oba reprezentacje raportu.

    Parametry opcjonalne:
        accuracy_stats: wynik `RepositoryPort.get_accuracy_stats(days)`.
        resolved_predictions: zamknięte predykcje z ostatnich N godzin
            (do sekcji day-over-day).
        macro_risk_report: wynik `MonitorMacroRiskUseCase.run(...)` — gdy
            podany, w raporcie pojawia się sekcja 🚨 Risk Watch.
    """
    # U2: personalizacja — gdy podany filtr symboli (subskrybent), tniemy wyniki
    # do jego watchlisty. Analiza i tak poszła RAZ nad unią; tu tylko re-slicing.
    if symbols_filter is not None:
        results = [r for r in results if r.symbol in symbols_filter]
    saved = [r for r in results if r.status == "saved"]
    ignored = [r for r in results if r.status == "ignored"]
    errors = [r for r in results if r.status == "error"]
    mood = build_portfolio_mood(results)
    session = market_status(started_at)
    # Q7: hit-rate cyklu (z accuracy_stats) napędza pasmo wielkości pozycji.
    hit_rate = accuracy_stats.get("mean_accuracy") if accuracy_stats else None
    # #9: gdy orkiestrator dostarczył kubełki kalibracji (flaga
    # `calibrated_confidence_enabled`), ranking sygnałów liczy się na pewności
    # skorygowanej historią. None → surowa pewność, jak przed #9.
    trade_signals = build_trade_signals(
        results, hit_rate=hit_rate, calibration_buckets=signal_calibration_buckets
    )
    risk_signals = detect_risk_signals(results)
    macro_risk_html = (
        render_risk_watch_html(macro_risk_report) if macro_risk_report else ""
    )
    # Q4: sekcja korelacji/koncentracji portfela (render_correlation_html zwraca
    # "" gdy brak klastrów → sekcja sama się chowa).
    portfolio_html = (
        render_correlation_html(portfolio_risk_report)
        if portfolio_risk_report
        else ""
    )
    # #1: lead cyklu. Ranking robi domena; TU jest jedyne miejsce, gdzie DTO
    # warstwy application (SymbolResult / ResolvedPrediction) są mapowane na
    # typy domenowe — `src/domain/digest_lead.py` ich nie importuje.
    lead_items = build_lead(
        to_lead_signals(results, resolved_predictions or []),
        quota_alerts or [],
    )
    lead_html = render_lead_html(lead_items)
    lead_text = render_lead_text(lead_items)
    # #4: powitanie pierwszego cyklu. Renderer nigdy nie decyduje o tonie —
    # dostaje gotową klasyfikację z domeny (STEADY_STATE → "").
    maturity = cycle_maturity or CycleMaturity.STEADY_STATE
    onboarding_html = render_onboarding_html(
        maturity, instrument_count=len(results), sectors=portfolio_sectors or {}
    )
    onboarding_text = render_onboarding_text(
        maturity, instrument_count=len(results), sectors=portfolio_sectors or {}
    )
    # U5: panel historii głosów rady per inwestor (render zwraca "" gdy brak).
    council_history_html = render_council_history_html(council_history or [])
    # #3: ranking wiarygodności person — ranking (z progiem min_votes) przychodzi
    # gotowy z domeny; render zwraca "" gdy żadna persona nie ma dość głosów.
    persona_leaderboard_html = render_persona_leaderboard_html(
        persona_track_record or []
    )
    # #8: zmiany nastawienia rady. Render odfiltrowuje STABLE i jawnie oznacza
    # stęchłe porównania — pusta lista → sekcja się chowa.
    consensus_shift_html = render_consensus_shift_html(consensus_shifts or [])
    # #12: karta kondycji modelu (render zwraca "" gdy brak danych).
    model_scorecard_html = (
        render_model_scorecard_html(
            model_scorecard.summary,
            trained_at=model_scorecard.trained_at,
            now=started_at,
            candidate_rmse=model_scorecard.candidate_rmse,
            baseline_rmse=model_scorecard.baseline_rmse,
            directional_hit_rate=model_scorecard.directional_hit_rate,
            folds=model_scorecard.folds,
        )
        if model_scorecard is not None
        else ""
    )
    # T1/T2/T4: sekcja Track Record (krzywa kapitału, kalibracja, lessons).
    # render zwraca "" gdy wszystkie trzy podsekcje puste.
    track_record_html = render_track_record_html(
        equity_curve, calibration_buckets, lessons
    )
    # Dane: sekcja Alpha Signals (insider/analitycy/opcje/social/earnings per
    # symbol + krzywa rentowności FRED). render zwraca "" gdy brak danych.
    alpha_html = render_alpha_html(
        alpha_signals or {}, yield_curve, alpha_prices
    )
    macro_risk_text = (
        render_risk_watch_text(macro_risk_report) if macro_risk_report else ""
    )
    quota_html = render_quota_banner_html(quota_alerts or [])
    quota_text = render_quota_banner_text(quota_alerts or [])
    suggestions = build_sector_suggestions(results)
    suggestions_html = render_suggestions_html(suggestions)
    suggestions_text = render_suggestions_text(suggestions)

    html = _render_html(
        results, saved, ignored, errors, started_at, duration_seconds,
        mood, session, accuracy_stats, trade_signals, risk_signals,
        resolved_predictions or [],
    )
    html = _fill_html_slots(
        html,
        quota_html=quota_html,
        macro_risk_html=macro_risk_html,
        portfolio_html=portfolio_html,
        lead_html=lead_html,
        onboarding_html=onboarding_html,
        council_history_html=council_history_html,
        persona_leaderboard_html=persona_leaderboard_html,
        consensus_shift_html=consensus_shift_html,
        model_scorecard_html=model_scorecard_html,
        track_record_html=track_record_html,
        alpha_html=alpha_html,
        suggestions_html=suggestions_html,
    )
    text = _render_plain(
        results, saved, ignored, errors, started_at, duration_seconds,
        mood, session, accuracy_stats, trade_signals, risk_signals,
        resolved_predictions or [],
    )
    text = _fill_text_slots(
        text,
        quota_text=quota_text,
        macro_risk_text=macro_risk_text,
        lead_text=lead_text,
        onboarding_text=onboarding_text,
        suggestions_text=suggestions_text,
    )
    return html, text


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _render_trade_signals_html(trade_signals: list[TradeSignal]) -> str:
    """🎯 Najsilniejsze sygnały transakcyjne. Pusty string, gdy brak sygnałów."""
    if not trade_signals:
        return ""
    parts = [_section_h2("🎯 Najsilniejsze sygnały")]
    for sig in trade_signals:
        dir_color = {
            "KUP": "#16a34a",
            "SPRZEDAJ": "#dc2626",
            "OBSERWUJ": "#737373",
        }[sig.direction]
        dir_bg = {
            "KUP": "#f0fdf4",
            "SPRZEDAJ": "#fef2f2",
            "OBSERWUJ": "#f9fafb",
        }[sig.direction]
        change_color = _delta_color(sig.expected_change)
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
                  pewność <strong>{sig.confidence * 100:.0f}%</strong>
                </span>
                <span style="color: {change_color};">
                  prognoza <strong>{_pct(sig.expected_change, signed=True)}</strong>
                </span>
                {size_html}
                <span style="color: #6b7280; margin-left: auto; font-size: 12px;">
                  siła sygnału: <strong>{sig.strength:.2f}</strong>
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


def _render_resolved_predictions_html(
    resolved_predictions: list[ResolvedPrediction],
) -> str:
    """📊 Zamknięte predykcje z ostatnich 24h. Pusty string, gdy brak."""
    if not resolved_predictions:
        return ""
    correct = [p for p in resolved_predictions if p.is_correct]
    wrong = [p for p in resolved_predictions if not p.is_correct]
    parts = [_section_h2("📊 Zamknięte predykcje (ostatnie 24h)")]
    parts.extend(_render_resolved_item_html(p) for p in resolved_predictions)
    parts.append(
        "<div style='font-size: 11px; color: #6b7280; margin: 4px 0 16px 0;'>"
        f"Suma: {len(correct)} trafionych / {len(wrong)} błędnych "
        f"({len(correct) / max(1, len(resolved_predictions)) * 100:.0f}% accuracy)"
        "</div>"
    )
    return "".join(parts)


def _render_accuracy_history_html(accuracy_stats: dict[str, Any] | None) -> str:
    """🎯 Historia trafności. Pusty string, gdy brak danych statystyk."""
    if accuracy_stats and accuracy_stats.get("mean_accuracy") is not None:
        acc = accuracy_stats["mean_accuracy"]
        n = accuracy_stats["sample_count"]
        correct = accuracy_stats["correct_count"]
        days = accuracy_stats["days_window"]
        return f"""
      <div style="padding: 12px 14px; background: #fef3c7; border-left: 3px solid #f59e0b;
                  border-radius: 4px; margin-bottom: 20px; font-size: 13px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: #92400e;">
          🎯 Historia trafności (ostatnie {days} dni)
        </div>
        <div style="color: #78350f;">
          Średnia trafność: <strong>{acc * 100:.1f}%</strong>
          · Predykcji ocenionych: {n}
          · Poprawnych kierunkowo: {correct}
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


def _render_ignored_html(ignored: list[SymbolResult]) -> str:
    """⏸ Symbole pominięte poniżej progu zmienności. Pusty string, gdy brak."""
    if not ignored:
        return ""
    rows = "".join(
        f"<span style='display: inline-block; padding: 2px 8px; margin: 2px; background: #f3f4f6; border-radius: 3px; font-size: 12px;'>"
        f"<strong>{_html(r.symbol)}</strong> <span style='color: {_delta_color(r.delta)}'>{_pct(r.delta)}</span></span>"
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
        f"{_html(_sentiment_label(r.sentiment_label))} ({r.sentiment_score:+.2f})"
        if r.sentiment_score is not None
        else "—"
    )
    confidence_text = (
        f"{r.confidence_score * 100:.0f}%"
        if r.confidence_score is not None
        else "—"
    )
    forecast_text = (
        f"{_money(r.target_price)} "
        f"<span style='color: {_delta_color(r.expected_change)};'>"
        f"({_pct(r.expected_change, signed=True)})</span>"
        if r.target_price is not None
        else "—"
    )
    rec = _recommendation(r)
    rec_color = _DIRECTION_COLOR.get(rec, "#737373")
    return f"""
              <tr>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6; font-weight: 600;">{_html(_company_label(r.symbol))}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{_money(r.current_price)}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6; color: {_delta_color(r.delta)};">{_pct(r.delta)}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6; color: {_trend_color(r.trend)}; font-weight: 600;">{_html(_trend_label(r.trend))}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{forecast_text}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{confidence_text}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{sentiment_text}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{r.news_volume or 0}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6; font-weight: 700; color: {rec_color};">{rec}</td>
              </tr>
            """


def _render_reasoning_block_html(r: SymbolResult) -> str:
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
    ):
        return ""
    move_line = ""
    if r.current_price is not None and r.target_price is not None:
        conf_part = (
            f" · Pewność {r.confidence_score * 100:.0f}%"
            if r.confidence_score is not None
            else ""
        )
        move_line = (
            f"<div style='font-size: 11px; color: #6b7280; margin-top: 2px;'>"
            f"Oczekiwany ruch: {_money(r.current_price)} → {_money(r.target_price)} "
            f"<span style='color: {_delta_color(r.expected_change)};'>"
            f"({_pct(r.expected_change, signed=True)})</span>"
            f"{conf_part}"
            f"</div>"
        )
    # Blok 'Top newsy' — zmigrowany na autoescapowany Jinja env
    # (finding #35). Escapowanie title/source jest tu domyślne.
    news_block = _render_news_block_html(r.top_news)
    rec_block = _recommendation_reason_html(r)
    council_block = (
        _render_council_section(r.council_verdict)
        if r.council_verdict is not None
        else ""
    )
    valuation_block = _render_valuation(r.valuation)
    _sector = _sector_label(r.symbol)
    sector_tag = (
        f' <span style="color: #6b7280; font-weight: 500;">· {_html(_sector)}</span>'
        if _sector
        else ""
    )
    return f"""
              <div style="margin-bottom: 10px; padding: 10px 12px; background: #fafafa; border-left: 3px solid {_trend_color(r.trend)}; border-radius: 4px;">
                <div style="font-weight: 600; font-size: 13px;">{_html(_company_label(r.symbol))}{sector_tag} <span style="color: {_trend_color(r.trend)};">{_html(_trend_label(r.trend))}</span>{badges_html}</div>
                {move_line}
                {rec_block}
                {f'<div style="font-size: 12px; color: #4b5563; margin-top: 6px;">{_html(r.reasoning)}</div>' if r.reasoning else ''}
                {attribution_block}
                {precedents_block}
                {news_block}
                {council_block}
                {valuation_block}
              </div>
            """


def _render_saved_section_html(saved: list[SymbolResult]) -> str:
    """🔮 Sekcja wygenerowanych predykcji: tabela + wykres prognozy +
    uzasadnienia. Pusty string, gdy w tym cyklu nic nie zapisano."""
    if not saved:
        return ""
    parts = [
        _section_h2("🔮 Wygenerowane predykcje"),
        "<table style='width: 100%; border-collapse: collapse; font-size: 13px;'>",
        """
          <tr style="background: #f3f4f6; text-align: left;">
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Symbol</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Cena</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Zmiana (cykl)</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Trend</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Prognoza (nast. cykl)</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Pewność</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Sentyment</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Newsy</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Rekomendacja</th>
          </tr>
        """,
    ]
    parts.extend(_render_prediction_row_html(r) for r in saved)
    parts.append("</table>")

    # Wykres prognozy
    forecast_chart_url = build_forecast_chart_url(saved)
    if forecast_chart_url:
        parts.append(f"""
      <div style="margin: 16px 0; text-align: center;">
        <img src="{_html(forecast_chart_url)}" alt="Prognoza zmian cen"
             style="max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;" />
      </div>
            """)

    # Uzasadnienia
    parts.append("<h3 style='font-size: 14px; margin: 16px 0 8px 0;'>💡 Uzasadnienia</h3>")
    parts.extend(_render_reasoning_block_html(r) for r in saved)
    return "".join(parts)


def _render_html(
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    started_at: datetime,
    duration_seconds: float,
    mood: dict[str, Any],
    session: dict[str, str],
    accuracy_stats: dict[str, Any] | None,
    trade_signals: list[TradeSignal],
    risk_signals: list[RiskSignal],
    resolved_predictions: list[ResolvedPrediction],
) -> str:
    sections = []

    # Header
    sections.append(f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; max-width: 720px; margin: 0 auto; color: #1f2937;">
      <h1 style="font-size: 22px; margin: 0 0 4px 0;">📊 StockAgent — raport cyklu</h1>
      <div style="font-size: 13px; color: #6b7280; margin-bottom: 24px;">
        Czas rozpoczęcia: {started_at.strftime("%Y-%m-%d %H:%M:%S UTC")}
        · Czas trwania: {duration_seconds:.1f}s
        · Liczba symboli: {len(results)}
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

    # Summary boxes
    sections.append(f"""
      <div style="display: flex; gap: 8px; margin-bottom: 16px;">
        <div style="flex: 1; padding: 12px; background: #f0fdf4; border-left: 3px solid #16a34a; border-radius: 4px;">
          <div style="font-size: 11px; color: #15803d; text-transform: uppercase;">Predykcje</div>
          <div style="font-size: 24px; font-weight: 600; color: #14532d;">{len(saved)}</div>
        </div>
        <div style="flex: 1; padding: 12px; background: #f9fafb; border-left: 3px solid #9ca3af; border-radius: 4px;">
          <div style="font-size: 11px; color: #4b5563; text-transform: uppercase;">Pominięte</div>
          <div style="font-size: 24px; font-weight: 600; color: #1f2937;">{len(ignored)}</div>
        </div>
        <div style="flex: 1; padding: 12px; background: {'#fef2f2' if errors else '#f9fafb'}; border-left: 3px solid {'#dc2626' if errors else '#9ca3af'}; border-radius: 4px;">
          <div style="font-size: 11px; color: {'#991b1b' if errors else '#4b5563'}; text-transform: uppercase;">Błędy</div>
          <div style="font-size: 24px; font-weight: 600; color: {'#7f1d1d' if errors else '#1f2937'};">{len(errors)}</div>
        </div>
      </div>
    """)

    # Slot na sekcję Risk Watch — wypełniany przez build_html_report,
    # gdy wywołujący przekaże MacroRiskReport. Bez slotu sekcja jest pomijana.
    sections.append(_RISK_WATCH_SLOT)
    # Slot na radar korelacji portfela (Q4) — wypełniany przez build_html_report.
    sections.append(_PORTFOLIO_SLOT)
    # Slot na panel historii głosów rady (U5).
    sections.append(_COUNCIL_HISTORY_SLOT)
    # Slot na ranking wiarygodności person (#3) — "kto z rady miał rację".
    # Zaraz po historii głosów: najpierw KTO jak głosował, potem KTO trafiał.
    sections.append(_PERSONA_LEADERBOARD_SLOT)
    sections.append(_CONSENSUS_SHIFT_SLOT)
    # Slot na sekcję Track Record (T1 equity curve, T2 calibration, T4 lessons).
    sections.append(_TRACK_RECORD_SLOT)
    sections.append(_MODEL_SCORECARD_SLOT)
    # Slot na sekcję Alpha Signals (Dane — insider/analitycy/opcje/social/FRED).
    sections.append(_ALPHA_SLOT)

    # 🪙 Krypto — osobna sekcja, widoczna też gdy cykl 'ignored' (próg 5%).
    crypto_html = _render_crypto_section_html(results)
    if crypto_html:
        sections.append(crypto_html)

    # 🎯 Trade ideas (najsilniejsze sygnały transakcyjne)
    sections.append(_render_trade_signals_html(trade_signals))

    # 🚨 Risk signals (anomalie / niespójności)
    sections.append(_render_risk_signals_html(risk_signals))

    # Portfolio mood box
    bullish_part = (
        f"<strong>{_html(_company_label(mood['most_bullish'].symbol))}</strong> "
        f"({mood['most_bullish'].sentiment_score:+.2f})"
        if mood["most_bullish"] else "—"
    )
    bearish_part = (
        f"<strong>{_html(_company_label(mood['most_bearish'].symbol))}</strong> "
        f"({mood['most_bearish'].sentiment_score:+.2f})"
        if mood["most_bearish"] else "—"
    )
    sections.append(f"""
      <div style="padding: 12px 14px; background: #eff6ff; border-left: 3px solid #2563eb;
                  border-radius: 4px; margin-bottom: 20px; font-size: 13px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: #1e3a8a;">
          📊 Nastroje portfela
        </div>
        <div style="color: #1e40af; line-height: 1.7;">
          Średni sentyment: <strong>{mood['avg_sentiment']:+.2f}</strong>
          ({mood['avg_sentiment_label']})<br/>
          Najbardziej pozytywny: {bullish_part} · Najbardziej negatywny: {bearish_part}<br/>
          Predykcje z wysoką pewnością (≥70%): {mood['high_confidence_count']} / {mood['saved_count']}
          · Łącznie newsów: {mood['total_news']}
        </div>
      </div>
    """)

    # 📊 Day-over-day — zamknięte predykcje z ostatnich 24h
    sections.append(_render_resolved_predictions_html(resolved_predictions))

    # Historia trafności
    sections.append(_render_accuracy_history_html(accuracy_stats))

    # Wykres zmiany cen
    chart_url = build_chart_url(results)
    if chart_url:
        sections.append(f"""
      <div style="margin-bottom: 16px; text-align: center;">
        <img src="{_html(chart_url)}" alt="Wykres zmiany cen (cykl)"
             style="max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;" />
      </div>
        """)

    # Self-Reflection — wnioski z poprzednich cykli
    sections.append(_render_reflections_html(saved))

    # Predykcje (saved)
    sections.append(_render_saved_section_html(saved))

    sections.append(_SUGGESTIONS_SLOT)

    # Scatter: sentyment vs cena (jeśli ≥3 punkty)
    scatter_url = build_correlation_chart_url(results)
    if scatter_url:
        sections.append(f"""
      <h3 style="font-size: 14px; margin: 20px 0 8px 0;">📈 Korelacja sentymentu z ruchem ceny</h3>
      <div style="margin-bottom: 16px; text-align: center;">
        <img src="{_html(scatter_url)}" alt="Scatter sentyment vs cena"
             style="max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;" />
      </div>
        """)

    # Pominięte
    sections.append(_render_ignored_html(ignored))

    # Błędy
    sections.append(_render_errors_html(errors))

    sections.append("""
      <div style="margin-top: 32px; padding-top: 16px; border-top: 1px solid #e5e7eb; font-size: 11px; color: #9ca3af;">
        StockAgent · Agentic RAG + Continual Learning · raport automatyczny
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
    for sig in trade_signals:
        size_txt = f"  [{sig.size_band.label}]" if sig.size_band else ""
        lines.append(
            f"  {sig.direction:9s} {_company_label(sig.symbol):40s} "
            f"pewność {sig.confidence * 100:.0f}%  "
            f"prognoza {_pct(sig.expected_change, signed=True):>8s}  "
            f"siła {sig.strength:.2f}{size_txt}"
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
) -> list[str]:
    """Sekcja 'Zamknięte predykcje' (plain). Pusta lista, gdy brak."""
    if not resolved_predictions:
        return []
    correct = sum(1 for p in resolved_predictions if p.is_correct)
    lines = ["ZAMKNIĘTE PREDYKCJE (ostatnie 24h)", "-" * 64]
    for p in resolved_predictions:
        mark = "✅" if p.is_correct else "❌"
        verdict = "Trafiona" if p.is_correct else "Błędna"
        lines.append(
            f"  {mark} {_company_label_with_sector(p.symbol)}  "
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
        f"  Suma: {correct}/{len(resolved_predictions)} "
        f"({correct / max(1, len(resolved_predictions)) * 100:.0f}% accuracy)"
    )
    lines.append("")
    return lines


def _render_accuracy_history_text(accuracy_stats: dict[str, Any] | None) -> list[str]:
    """Sekcja 'Historia trafności' (plain). Pusta lista, gdy brak danych."""
    if not (accuracy_stats and accuracy_stats.get("mean_accuracy") is not None):
        return []
    return [
        f"HISTORIA TRAFNOŚCI (ostatnie {accuracy_stats['days_window']} dni)",
        "-" * 64,
        f"  Średnia trafność: {accuracy_stats['mean_accuracy'] * 100:.1f}%"
        f"  ·  Predykcji ocenionych: {accuracy_stats['sample_count']}"
        f"  ·  Poprawnych: {accuracy_stats['correct_count']}",
        "",
    ]


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


def _render_prediction_text(r: SymbolResult) -> list[str]:
    """Linie jednej predykcji (plain): nagłówek + sentyment + rekomendacja +
    proweniencja + reasoning + analogi + newsy + wycena."""
    conf_part = (
        f" · pewność {r.confidence_score * 100:.0f}%"
        if r.confidence_score is not None
        else ""
    )
    forecast_part = (
        f"prognoza {_money(r.target_price)} "
        f"({_pct(r.expected_change, signed=True)})"
        if r.target_price is not None and r.expected_change is not None
        else "prognoza —"
    )
    sentiment_part = (
        f"sentyment {r.sentiment_score:+.2f}"
        if r.sentiment_score is not None
        else "sentyment —"
    )
    rec_text = _recommendation_reason_text(r)
    lines = [
        f"  {_company_label_with_sector(r.symbol):55s} {_money(r.current_price):>10s}  "
        f"Δ {_pct(r.delta, signed=True):>8s}  →  "
        f"{_trend_label(r.trend):11s} {forecast_part}{conf_part}",
        f"         {sentiment_part}",
        f"         {rec_text}",
    ]
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
            f" (rel={n.relevance:.2f}, sent={n.sentiment:+.2f})"
        )
    valuation_text = _render_valuation_text(r.valuation)
    if valuation_text:
        lines.append(valuation_text)
    return lines


def _render_saved_section_text(saved: list[SymbolResult]) -> list[str]:
    """Sekcja 'Predykcje' (plain). Pusta lista, gdy nic nie zapisano."""
    if not saved:
        return []
    lines = ["PREDYKCJE", "-" * 64]
    for r in saved:
        lines.extend(_render_prediction_text(r))
    lines.append("")
    return lines


def _render_ignored_text(ignored: list[SymbolResult]) -> list[str]:
    """Sekcja 'Pominięte' (plain). Pusta lista, gdy brak."""
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
    started_at: datetime,
    duration_seconds: float,
    mood: dict[str, Any],
    session: dict[str, str],
    accuracy_stats: dict[str, Any] | None,
    trade_signals: list[TradeSignal],
    risk_signals: list[RiskSignal],
    resolved_predictions: list[ResolvedPrediction],
) -> str:
    lines: list[str] = []
    lines.append(_QUOTA_BANNER_SLOT)
    lines.append(_LEAD_SLOT)
    lines.append(_ONBOARDING_SLOT)
    lines.append("=" * 64)
    lines.append("STOCKAGENT — RAPORT CYKLU")
    lines.append("=" * 64)
    lines.append(f"Czas rozpoczęcia:  {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Czas trwania:      {duration_seconds:.1f}s")
    lines.append(f"Sesja NYSE:        {session['label']} — {session['detail']}")
    lines.append(
        f"Symboli:           {len(results)} "
        f"(predykcji={len(saved)}, pominiętych={len(ignored)}, błędów={len(errors)})"
    )
    lines.append("")

    # Slot na sekcję Risk Watch w wariancie plain text — analogicznie do HTML.
    lines.append(_RISK_WATCH_SLOT)

    # Krypto — osobna sekcja (widoczna też gdy 'ignored').
    crypto_text = _render_crypto_section_text(results)
    if crypto_text:
        lines.append(crypto_text)
        lines.append("")

    # Trade ideas
    lines.extend(_render_trade_signals_text(trade_signals))

    # Risk signals
    lines.extend(_render_risk_signals_text(risk_signals))

    # Zamknięte predykcje
    lines.extend(_render_resolved_predictions_text(resolved_predictions))

    # Nastroje portfela
    lines.append("NASTROJE PORTFELA")
    lines.append("-" * 64)
    lines.append(
        f"  Średni sentyment: {mood['avg_sentiment']:+.2f} ({mood['avg_sentiment_label']})"
    )
    if mood["most_bullish"]:
        mb = mood["most_bullish"]
        lines.append(f"  Najbardziej pozytywny: {_company_label(mb.symbol)} ({mb.sentiment_score:+.2f})")
    if mood["most_bearish"]:
        mbe = mood["most_bearish"]
        lines.append(f"  Najbardziej negatywny: {_company_label(mbe.symbol)} ({mbe.sentiment_score:+.2f})")
    lines.append(
        f"  Wysoka pewność (≥70%): {mood['high_confidence_count']} / {mood['saved_count']}"
        f"  ·  Newsów: {mood['total_news']}"
    )
    lines.append("")

    lines.extend(_render_accuracy_history_text(accuracy_stats))

    # Wnioski z poprzednich cykli
    lines.extend(_render_reflections_text(saved))

    lines.extend(_render_saved_section_text(saved))

    lines.append(_SUGGESTIONS_SLOT)

    lines.extend(_render_ignored_text(ignored))

    lines.extend(_render_errors_text(errors))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: convert use case output → SymbolResult
# ---------------------------------------------------------------------------


def to_symbol_result(symbol: str, raw: dict[str, Any] | None, error: str | None = None) -> SymbolResult:
    """Mapper z wyjścia AnalyzeMarketUseCase.run() na SymbolResult."""
    if error is not None:
        return SymbolResult(symbol=symbol, status="error", error_message=error)
    if raw is None:
        return SymbolResult(symbol=symbol, status="error", error_message="No result")

    status = str(raw.get("status", "unknown"))
    sentiment = raw.get("sentiment") or {}
    llm = raw.get("llm_analysis") or {}

    # #4: cold-start jest widoczny w stanie końcowym jako previous_price == 0
    # (graf nie miał punktu odniesienia). Symbol odcięty bramką volatility ma
    # poprzednią cenę — to zupełnie inna historia niż pierwszy dzień deploymentu.
    skip_reason: SkipReason | None = None
    if status == "ignored":
        previous_price = raw.get("previous_price")
        skip_reason = (
            SkipReason.COLD_START
            if previous_price is None or Decimal(str(previous_price)) == 0
            else SkipReason.BELOW_THRESHOLD
        )

    # Klasa aktywa z domeny — graf trzyma Asset w stanie (main_agent klasyfikuje
    # STOCK/ETF/CRYPTO przed wywołaniem). Pozwala raportowi rozpoznać krypto
    # niezawodnie, niezależnie od statusu (saved/ignored).
    asset = raw.get("asset")
    asset_class = (
        asset.asset_type.value if asset is not None and hasattr(asset, "asset_type")
        else None
    )
    confidence = llm.get("confidence_score")
    av_agreement = llm.get("av_agreement")

    council_verdict = raw.get("council_verdict")

    valuation: ValuationSection | None = None
    verdict = raw.get("valuation_verdict")
    fundamentals = raw.get("fundamentals")
    if (
        isinstance(verdict, ValuationVerdict)
        and verdict is not ValuationVerdict.UNKNOWN
        and fundamentals is not None
    ):
        valuation = ValuationSection(
            trailing_pe=fundamentals.trailing_pe,
            forward_pe=fundamentals.forward_pe,
            peg_ratio=fundamentals.peg_ratio,
            eps_growth_yoy=fundamentals.eps_growth_yoy,
            verdict=verdict,
            fetched_at=fundamentals.fetched_at,
        )

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
        now=datetime.now(UTC),
        ttl_hours=FUNDAMENTALS_CACHE_TTL_HOURS,
    )

    return SymbolResult(
        symbol=symbol,
        status=status,
        skip_reason=skip_reason,
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
    )


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
