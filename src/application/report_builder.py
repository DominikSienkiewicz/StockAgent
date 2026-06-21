"""Buduje raport HTML+text z wyników cyklu Fast Loop.

Czysta logika prezentacji — bez I/O, łatwo testowalne.
"""

from __future__ import annotations

import html
import urllib.parse
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
from src.application.report_lessons import LessonsReport
from src.application.report_models import (
    ResolvedPrediction,
    RiskSignal,
    SimilarPrecedent,
    SymbolResult,
    TopNewsItem,
    TradeSignal,
    ValuationSection,
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
from src.domain.council import CouncilVerdict
from src.domain.equity_curve import EquityCurve
from src.domain.feature_attribution import FeatureContribution
from src.domain.macro_rates import YieldCurveSnapshot
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
        "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>🪙 Krypto</h2>"
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


def build_html_report(
    results: list[SymbolResult],
    started_at: datetime,
    duration_seconds: float,
    accuracy_stats: dict[str, Any] | None = None,
    resolved_predictions: list[ResolvedPrediction] | None = None,
    macro_risk_report: MacroRiskReport | None = None,
    portfolio_risk_report: PortfolioRiskReport | None = None,
    council_history: list[InvestorHistory] | None = None,
    quota_alerts: list[QuotaAlert] | None = None,
    symbols_filter: frozenset[str] | None = None,
    equity_curve: EquityCurve | None = None,
    calibration_buckets: list[CalibrationBucket] | None = None,
    lessons: LessonsReport | None = None,
    alpha_signals: dict[str, AlphaSignals] | None = None,
    yield_curve: YieldCurveSnapshot | None = None,
    alpha_prices: dict[str, Decimal] | None = None,
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
    trade_signals = build_trade_signals(results, hit_rate=hit_rate)
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
    # U5: panel historii głosów rady per inwestor (render zwraca "" gdy brak).
    council_history_html = render_council_history_html(council_history or [])
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
    if quota_html:
        html = html.replace(
            "<!-- QUOTA_BANNER_SLOT -->", quota_html, 1
        )
    if macro_risk_html:
        html = html.replace(
            "<!-- RISK_WATCH_SLOT -->", macro_risk_html, 1
        )
    # Slot portfela zawsze podmieniamy (też na "") — inaczej znacznik zostałby w treści.
    html = html.replace("<!-- PORTFOLIO_SLOT -->", portfolio_html, 1)
    html = html.replace("<!-- COUNCIL_HISTORY_SLOT -->", council_history_html, 1)
    # Slot Track Record zawsze podmieniamy (też na ""), by znacznik nie został.
    html = html.replace("<!-- TRACK_RECORD_SLOT -->", track_record_html, 1)
    # Slot Alpha Signals (Dane) — zawsze podmieniany (też na "").
    html = html.replace("<!-- ALPHA_SLOT -->", alpha_html, 1)
    # Slot zawsze podmieniamy (też na "") — inaczej znacznik zostałby w treści.
    html = html.replace("<!-- SUGGESTIONS_SLOT -->", suggestions_html, 1)
    text = _render_plain(
        results, saved, ignored, errors, started_at, duration_seconds,
        mood, session, accuracy_stats, trade_signals, risk_signals,
        resolved_predictions or [],
    )
    if quota_text:
        text = text.replace(
            "<!-- QUOTA_BANNER_SLOT -->", quota_text, 1
        )
    if macro_risk_text:
        text = text.replace(
            "<!-- RISK_WATCH_SLOT -->", macro_risk_text, 1
        )
    # Plain text: usuwamy znacznik (pusta sekcja) albo wstawiamy treść.
    text = text.replace(
        "<!-- SUGGESTIONS_SLOT -->\n", f"{suggestions_text}\n\n" if suggestions_text else "", 1
    ).replace("<!-- SUGGESTIONS_SLOT -->", suggestions_text, 1)
    return html, text


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


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
    sections.append("<!-- QUOTA_BANNER_SLOT -->")

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
    sections.append("<!-- RISK_WATCH_SLOT -->")
    # Slot na radar korelacji portfela (Q4) — wypełniany przez build_html_report.
    sections.append("<!-- PORTFOLIO_SLOT -->")
    # Slot na panel historii głosów rady (U5).
    sections.append("<!-- COUNCIL_HISTORY_SLOT -->")
    # Slot na sekcję Track Record (T1 equity curve, T2 calibration, T4 lessons).
    sections.append("<!-- TRACK_RECORD_SLOT -->")
    # Slot na sekcję Alpha Signals (Dane — insider/analitycy/opcje/social/FRED).
    sections.append("<!-- ALPHA_SLOT -->")

    # 🪙 Krypto — osobna sekcja, widoczna też gdy cykl 'ignored' (próg 5%).
    crypto_html = _render_crypto_section_html(results)
    if crypto_html:
        sections.append(crypto_html)

    # 🎯 Trade ideas (najsilniejsze sygnały transakcyjne)
    if trade_signals:
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>"
            "🎯 Najsilniejsze sygnały</h2>"
        )
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
            sections.append(f"""
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

    # 🚨 Risk signals (anomalie / niespójności)
    if risk_signals:
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0; color: #991b1b;'>"
            "🚨 Sygnały ostrzegawcze</h2>"
        )
        type_label = {
            "DIVERGENCE": "Rozbieżność cena ↔ sentyment",
            "AV_LLM_CONFLICT": "Niezgodność AV ↔ LLM",
            "LOW_SIGNAL": "Słaby sygnał",
        }
        for rs in risk_signals:
            sev_color = {"high": "#dc2626", "medium": "#f59e0b", "low": "#6b7280"}[rs.severity]
            sections.append(f"""
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
    if resolved_predictions:
        correct = [p for p in resolved_predictions if p.is_correct]
        wrong = [p for p in resolved_predictions if not p.is_correct]
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>"
            "📊 Zamknięte predykcje (ostatnie 24h)</h2>"
        )
        for p in resolved_predictions:
            sections.append(_render_resolved_item_html(p))
        sections.append(
            "<div style='font-size: 11px; color: #6b7280; margin: 4px 0 16px 0;'>"
            f"Suma: {len(correct)} trafionych / {len(wrong)} błędnych "
            f"({len(correct) / max(1, len(resolved_predictions)) * 100:.0f}% accuracy)"
            "</div>"
        )

    # Historia trafności
    if accuracy_stats and accuracy_stats.get("mean_accuracy") is not None:
        acc = accuracy_stats["mean_accuracy"]
        n = accuracy_stats["sample_count"]
        correct = accuracy_stats["correct_count"]
        days = accuracy_stats["days_window"]
        sections.append(f"""
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
        """)
    elif accuracy_stats is not None:
        sections.append("""
      <div style="padding: 10px 14px; background: #f9fafb; border-radius: 4px;
                  margin-bottom: 20px; font-size: 12px; color: #6b7280;">
        🎯 Historia trafności: brak ocenionych predykcji (potrzeba ≥1 zamkniętego cyklu).
      </div>
        """)

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
    reflections = [r for r in saved if r.reflection_insight]
    if reflections:
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>"
            "🧠 Wnioski z poprzednich cykli (Self-Reflection)</h2>"
        )
        for r in reflections:
            sections.append(f"""
              <div style="margin-bottom: 8px; padding: 10px 12px; background: #faf5ff;
                          border-left: 3px solid #9333ea; border-radius: 4px;
                          font-size: 12px;">
                <strong>{_html(_company_label(r.symbol))}:</strong>
                <span style="color: #581c87;">{_html(r.reflection_insight)}</span>
              </div>
            """)

    # Predykcje (saved)
    if saved:
        sections.append("<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>🔮 Wygenerowane predykcje</h2>")
        sections.append("<table style='width: 100%; border-collapse: collapse; font-size: 13px;'>")
        sections.append("""
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
        """)
        for r in saved:
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
            sections.append(f"""
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
            """)
        sections.append("</table>")

        # Wykres prognozy
        forecast_chart_url = build_forecast_chart_url(saved)
        if forecast_chart_url:
            sections.append(f"""
      <div style="margin: 16px 0; text-align: center;">
        <img src="{_html(forecast_chart_url)}" alt="Prognoza zmian cen"
             style="max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;" />
      </div>
            """)

        # Uzasadnienia
        sections.append("<h3 style='font-size: 14px; margin: 16px 0 8px 0;'>💡 Uzasadnienia</h3>")
        for r in saved:
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
                continue
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
            sections.append(f"""
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
            """)

    sections.append("<!-- SUGGESTIONS_SLOT -->")

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
    if ignored:
        sections.append("<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>⏸ Pominięte (poniżej progu zmienności)</h2>")
        rows = "".join(
            f"<span style='display: inline-block; padding: 2px 8px; margin: 2px; background: #f3f4f6; border-radius: 3px; font-size: 12px;'>"
            f"<strong>{_html(r.symbol)}</strong> <span style='color: {_delta_color(r.delta)}'>{_pct(r.delta)}</span></span>"
            for r in ignored
        )
        sections.append(f"<div>{rows}</div>")

    # Błędy
    if errors:
        sections.append("<h2 style='font-size: 16px; margin: 20px 0 8px 0; color: #991b1b;'>⚠️ Błędy</h2>")
        for r in errors:
            sections.append(f"""
              <div style="margin-bottom: 8px; padding: 10px; background: #fef2f2; border-left: 3px solid #dc2626; border-radius: 4px; font-size: 12px;">
                <strong>{_html(_company_label(r.symbol))}</strong>: {_html(r.error_message)}
              </div>
            """)

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
    lines.append("<!-- QUOTA_BANNER_SLOT -->")
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
    lines.append("<!-- RISK_WATCH_SLOT -->")

    # Krypto — osobna sekcja (widoczna też gdy 'ignored').
    crypto_text = _render_crypto_section_text(results)
    if crypto_text:
        lines.append(crypto_text)
        lines.append("")

    # Trade ideas
    if trade_signals:
        lines.append("NAJSILNIEJSZE SYGNAŁY")
        lines.append("-" * 64)
        for sig in trade_signals:
            size_txt = f"  [{sig.size_band.label}]" if sig.size_band else ""
            lines.append(
                f"  {sig.direction:9s} {_company_label(sig.symbol):40s} "
                f"pewność {sig.confidence * 100:.0f}%  "
                f"prognoza {_pct(sig.expected_change, signed=True):>8s}  "
                f"siła {sig.strength:.2f}{size_txt}"
            )
        lines.append("")

    # Risk signals
    if risk_signals:
        lines.append("SYGNAŁY OSTRZEGAWCZE")
        lines.append("-" * 64)
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

    # Zamknięte predykcje
    if resolved_predictions:
        correct = sum(1 for p in resolved_predictions if p.is_correct)
        lines.append("ZAMKNIĘTE PREDYKCJE (ostatnie 24h)")
        lines.append("-" * 64)
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

    if accuracy_stats and accuracy_stats.get("mean_accuracy") is not None:
        lines.append(f"HISTORIA TRAFNOŚCI (ostatnie {accuracy_stats['days_window']} dni)")
        lines.append("-" * 64)
        lines.append(
            f"  Średnia trafność: {accuracy_stats['mean_accuracy'] * 100:.1f}%"
            f"  ·  Predykcji ocenionych: {accuracy_stats['sample_count']}"
            f"  ·  Poprawnych: {accuracy_stats['correct_count']}"
        )
        lines.append("")

    # Wnioski z poprzednich cykli
    reflections = [r for r in saved if r.reflection_insight]
    if reflections:
        lines.append("WNIOSKI Z POPRZEDNICH CYKLI (Self-Reflection)")
        lines.append("-" * 64)
        for r in reflections:
            lines.append(f"  {_company_label(r.symbol)}: {r.reflection_insight}")
        lines.append("")

    if saved:
        lines.append("PREDYKCJE")
        lines.append("-" * 64)
        for r in saved:
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
            lines.append(
                f"  {_company_label_with_sector(r.symbol):55s} {_money(r.current_price):>10s}  "
                f"Δ {_pct(r.delta, signed=True):>8s}  →  "
                f"{_trend_label(r.trend):11s} {forecast_part}{conf_part}"
            )
            lines.append(f"         {sentiment_part}")
            lines.append(f"         {rec_text}")
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
                outcome = (
                    "trafił" if prec.is_trend_correct
                    else "chybił" if prec.is_trend_correct is False
                    else "wynik nieznany"
                )
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
        lines.append("")

    lines.append("<!-- SUGGESTIONS_SLOT -->")

    if ignored:
        lines.append("POMINIĘTE (poniżej progu zmienności)")
        lines.append("-" * 64)
        ignored_str = ", ".join(f"{r.symbol}({_pct(r.delta)})" for r in ignored)
        lines.append(f"  {ignored_str}")
        lines.append("")

    if errors:
        lines.append("BŁĘDY")
        lines.append("-" * 64)
        for r in errors:
            lines.append(f"  {_company_label(r.symbol)}: {r.error_message}")
        lines.append("")

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
