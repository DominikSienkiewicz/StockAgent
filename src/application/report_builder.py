"""Buduje raport HTML+text z wyników cyklu Fast Loop.

Czysta logika prezentacji — bez I/O, łatwo testowalne.
"""

from __future__ import annotations

import dataclasses
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from src.application.alpha_signals import AlphaSignals
from src.application.portfolio_fx import effective_pln_rates
from src.application.report_accuracy_history import ACCURACY_HISTORY_SECTION
from src.application.report_alpha import render_alpha_html, significant_buy_label
from src.application.report_charts import (
    build_correlation_chart_html,
    build_delta_chart_html,
    build_forecast_chart_html,
)
from src.application.report_consensus_shift import CONSENSUS_SHIFT_SECTION
from src.application.report_council_history import (
    InvestorHistory,
    render_council_history_html,
    render_council_history_text,
)
from src.application.report_crypto import CRYPTO_SECTION
from src.application.report_finops import render_finops_html, render_finops_text
from src.application.report_formatting import (
    DATA_SOURCES_FOOTER_PL,
    DEFAULT_DISPLAY_TIMEZONE,
    INVESTMENT_DISCLAIMER,
    PALETTE,
    direction_pct,
    format_local,
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
    provenance_badges_html as _provenance_badges_html,
)
from src.application.report_formatting import (
    section_h2 as _section_h2,
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
from src.application.report_header import (
    FIRST_SCREEN_SECTION,
    MASTHEAD_SECTION,
    SESSION_SECTION,
)
from src.application.report_lead import LEAD_SECTION, PREHEADER_SECTION
from src.application.report_lessons import LessonsReport
from src.application.report_model_scorecard import (
    ModelScorecardView,
    render_model_scorecard_html,
    render_model_scorecard_text,
)
from src.application.report_models import (
    PortfolioMood,
    ResolvedPrediction,
    RiskSignal,
    SimilarPrecedent,
    SymbolResult,
    TopNewsItem,
    TradeSignal,
    ValuationSection,
)
from src.application.report_onboarding import ONBOARDING_SECTION
from src.application.report_persona_leaderboard import (
    render_persona_leaderboard_html,
)
from src.application.report_portfolio import (
    render_correlation_html,
    render_correlation_text,
)
from src.application.report_quota_banner import QUOTA_BANNER_SECTION
from src.application.report_regime import (
    RegimeBanner,
)
from src.application.report_resolved import (
    RESOLVED_PREDICTIONS_SECTION,
    resolved_window,
)
from src.application.report_risk_watch import (
    render_risk_watch_html,
    render_risk_watch_text,
)
from src.application.report_signals import (
    build_portfolio_mood,
    build_trade_signals,
    detect_risk_signals,
    forecast_horizon_label,
    market_status,
    parse_resolved_predictions,
)
from src.application.report_skipped import (
    BUDGET_EXHAUSTED_SECTION,
    ERRORS_SECTION,
    IGNORED_SECTION,
    REBASELINED_SECTION,
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
from src.application.report_symbol_record import SymbolTrackRecordView, symbol_chip_labels
from src.application.report_templates import _env, render_template
from src.application.report_track_record import (
    TrackRecordStats,
    render_track_record_html,
)
from src.application.report_user_portfolio import (
    USER_PORTFOLIO_SECTION,
    portfolio_change_view,
)
from src.application.reporting.section import SectionInput
from src.application.run_context import RunContext
from src.application.use_cases.monitor_macro_risk import MacroRiskReport
from src.application.use_cases.portfolio_risk import PortfolioRiskReport, PositionRisk
from src.domain.asset import PriceDelta
from src.domain.calibration_curve import CalibrationBucket
from src.domain.conformal import MoveInterval
from src.domain.consensus_shift import ConsensusShift
from src.domain.council import CouncilVerdict
from src.domain.cycle_maturity import CycleMaturity, SkipReason
from src.domain.digest_lead import LeadItem, LeadSignal, build_lead
from src.domain.equity_curve import EquityCurve
from src.domain.feature_attribution import FeatureContribution
from src.domain.finops import CycleCost
from src.domain.insider_flow import InsiderLeadWindow
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.persona_track_record import PersonaTrackRecord
from src.domain.polish_labels import (
    MODEL_SIGNAL_TITLE,
    council_recommendation_label,
    model_signal_label,
    trend_forecast_label,
)
from src.domain.portfolio import (
    PlnRates,
    Portfolio,
)
from src.domain.provenance import ProvenanceLevel, build_provenance_badges
from src.domain.quota import QuotaAlert
from src.domain.sizing import PositionAction, SizeBand
from src.domain.value_objects import (
    FUNDAMENTALS_CACHE_TTL_HOURS,
    ValuationVerdict,
)

__all__ = [
    "PortfolioMood",
    "ResolvedPrediction",
    "RiskSignal",
    "SimilarPrecedent",
    "SymbolResult",
    "TopNewsItem",
    "TradeSignal",
    "ValuationSection",
    "build_correlation_chart_html",
    "build_cycle_lead",
    "build_cycle_signals",
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
_RISK_WATCH_SLOT = "<!-- RISK_WATCH_SLOT -->"
_PORTFOLIO_SLOT = "<!-- PORTFOLIO_SLOT -->"
_COUNCIL_HISTORY_SLOT = "<!-- COUNCIL_HISTORY_SLOT -->"
_PERSONA_LEADERBOARD_SLOT = "<!-- PERSONA_LEADERBOARD_SLOT -->"
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
# PORT-08: język ostrożny (MAR/KNF) — kierunek w kontekście portfela to nadal sygnał modelu.
_SIGNALS_POSITION_NOTE = (
    "Kierunek w kontekście Twojego portfela (tylko pozycje długie): spadek na "
    "trzymanej spółce to „ZREDUKUJ”, poza portfelem „UNIKAJ”, wzrost na już dużej "
    "pozycji „TRZYMAJ, nie dokładaj”. To sygnał modelu, nie rekomendacja."
)
# PORT-08: kolor kierunku po akcji pozycji — spadek czerwony, wzrost zielony.
_POSITION_ACTION_TONE = {
    PositionAction.BUY: "KUP",
    PositionAction.HOLD: "OBSERWUJ",
    PositionAction.REDUCE: "SPRZEDAJ",
    PositionAction.AVOID: "SPRZEDAJ",
    PositionAction.WATCH: "OBSERWUJ",
}


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
    "{% if failed %} · ⚠️ odpowiedziało {{ answered }}/{{ size }}{% endif %}"
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
        failed=verdict.failed_personas,
        answered=len(verdict.investor_opinions),
        size=verdict.council_size(),
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
    consensus_shifts: Sequence[tuple[str, ConsensusShift]] | None = None,
    alpha_signals: Mapping[str, AlphaSignals] | None = None,
    insider_lead_window: InsiderLeadWindow | None = None,
) -> list[LeadSignal]:
    """Mapuje DTO raportu na domenowe `LeadSignal` (#1).

    Granica architektury: `src/domain/digest_lead.py` bierze wyłącznie typy
    domenowe i prymitywy, więc `resolved_label` powstaje TU jako gotowy string.
    Symbole bez zmiany ceny nie wnoszą nic do rankingu i są pomijane.

    PORT-10: `portfolio_weights` (z `Portfolio.weights`) daje sygnałom wagę
    pozycji. Status "saved" znaczy, że ruch przeszedł bramkę volatility (graf
    poszedł do analizy) — tylko taki ruch pozycji może być leadem.

    MAIL-19: `consensus_shifts` (#8, z `ReportContext`) dają sygnałom zmianę
    nastawienia rady — odwrócenie (FLIP) ma w leadzie własny tier.
    DAT-21: `alpha_signals` daje znaczący zakup insidera (próg z kapitalizacji)
    jako gotowy tekst — ten sam co linia w sekcji alfy. `insider_lead_window`
    wpuszcza do leadu tylko filing z ostatnich N dni; None = bez okna.
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
    shifts = dict(consensus_shifts or ())
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
                consensus_shift=shifts.get(r.symbol),
                insider_buy_label=_insider_buy_label(
                    alpha_signals, r.symbol, insider_lead_window
                ),
            )
        )
    return signals


def _insider_buy_label(
    alpha_signals: Mapping[str, AlphaSignals] | None,
    symbol: str,
    window: InsiderLeadWindow | None,
) -> str | None:
    """DAT-21: tekst znaczącego zakupu insidera symbolu albo None — także gdy
    filing jest starszy niż okno leadu."""
    signal = alpha_signals.get(symbol) if alpha_signals else None
    buy = signal.significant_insider_buy() if signal is not None else None
    if buy is None or (window is not None and not window.admits(buy)):
        return None
    return significant_buy_label(buy)


def _weight_prices(
    portfolio: Portfolio, prices: Mapping[str, Decimal], rates: PlnRates | None
) -> Mapping[str, Decimal]:
    """Ceny do wag i ekspozycji: w złotych, gdy znamy kursy NBP (jedna waluta
    zamiast sumy dolarów z euro; pozycja bez kursu wypada), inaczej natywne."""
    if rates is None:
        return prices
    return portfolio.pln_prices(prices, rates).prices


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
        context.user_portfolio.weights(
            _weight_prices(
                context.user_portfolio, _report_prices(results, context), _pln_rates_of(context)
            )
        )
        if context.user_portfolio is not None
        else None
    )
    signals = to_lead_signals(
        results,
        context.resolved_predictions or [],
        weights,
        context.consensus_shifts,
        context.alpha_signals,
        context.insider_lead_window,
    )
    # ALR-05: zmiana reżimu (tylko z histerezy) tuż pod CRITICAL quota.
    regime_change = context.regime.change if context.regime is not None else None
    return build_lead(signals, context.quota_alerts or [], regime_change=regime_change)


# Tyle sygnałów pokazuje sekcja „🎯 Najsilniejsze sygnały” (domyślne `top_n`).
_TOP_SIGNALS = 5


def build_cycle_signals(
    results: list[SymbolResult], context: ReportContext, *, top_n: int = _TOP_SIGNALS
) -> list[TradeSignal]:
    """Ranking sygnałów cyklu — jedno źródło dla 🎯 sekcji maila i pushu (MAIL-20).

    Q7: hit-rate cyklu (z `accuracy_stats`) napędza pasmo wielkości pozycji.
    #9: kubełki kalibracji (flaga `calibrated_confidence_enabled`) liczą ranking
    na pewności skorygowanej historią; None → surowa pewność.
    #15 faza 2 / PORT-04 / PORT-08: gdy znamy realny portfel, sizing widzi
    istniejącą ekspozycję (symbolu i klastra) i ryzyko portfela, a sygnał niesie
    kierunek w kontekście pozycji. Bez portfela — jak przed tymi wpisami.
    """
    hit_rate, hit_sample = _sizing_track_record(context.accuracy_stats)
    portfolio = context.user_portfolio
    current_prices = _report_prices(results, context)
    weight_prices = (
        _weight_prices(portfolio, current_prices, _pln_rates_of(context))
        if portfolio is not None
        else current_prices
    )
    sizing_risk = (
        context.position_risk
        if portfolio is not None and context.position_risk is not None
        else PositionRisk()
    )
    return build_trade_signals(
        results,
        top_n=top_n,
        hit_rate=hit_rate,
        hit_sample=hit_sample,
        calibration_buckets=context.signal_calibration_buckets,
        portfolio_weights=portfolio.weights(weight_prices) if portfolio is not None else None,
        cluster_exposures=_cluster_exposure_per_symbol(
            portfolio, context.portfolio_clusters, weight_prices
        ),
        portfolio_var=sizing_risk.var,
        drawdown=sizing_risk.drawdown,
        hedge_effective=sizing_risk.hedge_effective,
    )


# Sloty pomijane przy pustej treści: nagłówek sekcji rysuje się w szablonie, więc
# podmiana na "" zostawiłaby pustą ramkę. Nieusunięty znacznik jest komentarzem
# HTML, czyli w mailu niewidoczny — to bezpieczniejszy z dwóch wyników.
_SKIP_WHEN_EMPTY_SLOTS = frozenset({_RISK_WATCH_SLOT})


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
    macro_risk_text: str,
    suggestions_text: str,
    portfolio_text: str = "",
    sections: Mapping[str, str] | None = None,
) -> str:
    """Wstawia treść w sloty plain-text. Pusty slot usuwa cały wiersz znacznika —
    tekst nie ma komentarzy HTML, więc znacznik byłby w nim dosłowną linią.
    W HTML RISK_WATCH zostaje przy pustej treści (`_SKIP_WHEN_EMPTY_SLOTS`).

    `sections` (MAIL-09): znacznik slotu → tekst sekcji z jej `render_*_text`, który
    kończy się już pustą linią odstępu."""
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
        f"{_SUGGESTIONS_SLOT}\n",
        f"{suggestions_text}\n\n" if suggestions_text else "",
        1,
    ).replace(_SUGGESTIONS_SLOT, suggestions_text, 1)
    for slot, content in (sections or {}).items():
        text = text.replace(f"{slot}\n", f"{content}\n" if content else "", 1).replace(
            slot, content, 1
        )
    return text


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


def _pln_rates_of(context: ReportContext) -> PlnRates | None:
    """PORT-11: kursy do „P/L zł”, wag i zmiany portfela (`effective_pln_rates`)."""
    report = context.macro_risk_report
    return effective_pln_rates(
        context.portfolio_fx, report.polish_macro if report is not None else None
    )


def _usd_pln_of(macro_risk_report: MacroRiskReport | None) -> Decimal | None:
    """Kurs USD/PLN z `MacroRiskReport.polish_macro` (wartość, nie port)."""
    if macro_risk_report is None or macro_risk_report.polish_macro is None:
        return None
    return macro_risk_report.polish_macro.usd_pln


def _build_model_scorecard_text(
    model_scorecard: ModelScorecardView | None,
    *,
    now: datetime,
    model_loaded: bool,
) -> str:
    """MAIL-09: karta kondycji modelu w wersji tekstowej (te same dane co HTML)."""
    if model_scorecard is None:
        return ""
    return render_model_scorecard_text(
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
        coverage=model_scorecard.coverage,
        drifted_features=model_scorecard.drifted_features,
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
    # PORT-11: kursy NBP do „P/L zł” (bieżące i z dni zakupu) — tylko z portfelem
    # właściciela; obcy subskrybent (BEZ-01) dostaje None jak resztę portfela.
    portfolio_fx: PlnRates | None = None
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
    # WIA-35: trafność kierunkowa per symbol z okna track recordu — chip przy
    # predykcji i sygnale. None = mail bez chipów.
    symbol_track_record: SymbolTrackRecordView | None = None
    # WIA-49: predykcje, które ten cykl oznaczył jako wygasłe bez oceny.
    expired_unverified: int = 0
    # MAIL-14: strefa (IANA) czasu startu cyklu w nagłówku HTML i tekstu.
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE
    # MAIL-42: przebieg, który wygenerował mail — stopka z numerem, krótkim SHA
    # i (tylko w GHA) linkiem do logów. None = stopka jak przed MAIL-42.
    run_context: RunContext | None = None
    # ALR-05: reżim rynku i mnożnik progu bramki w nagłówku. None = reżim off
    # albo brak raportu Risk Watch — nagłówek jak przed ALR-05.
    regime: RegimeBanner | None = None
    # DAT-21: zakup insidera prowadzi maila tylko przez N dni od filingu. None =
    # bez okna (wołający bez daty cyklu, testy).
    insider_lead_window: InsiderLeadWindow | None = None


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


def _section_input(
    results: list[SymbolResult],
    started_at: datetime,
    duration_seconds: float,
    context: ReportContext,
) -> SectionInput:
    """ARCH-12: wspólne wejście sekcji maila, liczone RAZ na render."""
    # U2: personalizacja — gdy podany filtr symboli (subskrybent), tniemy wyniki
    # do jego watchlisty. Analiza i tak poszła RAZ nad unią; tu tylko re-slicing.
    if context.symbols_filter is not None:
        results = [r for r in results if r.symbol in context.symbols_filter]
    saved, ignored, errors = _partition_by_status(results)
    current_prices = _report_prices(results, context)
    pln_rates = _pln_rates_of(context)
    return SectionInput(
        context=context,
        results=results,
        saved=saved,
        ignored=ignored,
        errors=errors,
        started_at=started_at,
        duration_seconds=duration_seconds,
        started_local=format_local(started_at, context.display_timezone),
        sector_map=context.sector_map or {},
        # #1: ranking leadu robi domena; TU jest jedyne miejsce, gdzie DTO warstwy
        # application (SymbolResult / ResolvedPrediction) są mapowane na typy
        # domenowe — `src/domain/digest_lead.py` ich nie importuje.
        lead_items=build_cycle_lead(results, context),
        resolved=context.resolved_predictions or [],
        resolved_window=resolved_window(context.resolved_this_cycle),
        current_prices=current_prices,
        pln_rates=pln_rates,
        usd_pln=_usd_pln_of(context.macro_risk_report),
        # PORT-06: nie trafia do tematu maila (decyzja 2026-10-19).
        portfolio_change=portfolio_change_view(
            results,
            context.user_portfolio,
            current_prices,
            now=started_at,
            display_timezone=context.display_timezone,
            rates=pln_rates,
        ),
    )


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
    data = _section_input(results, started_at, duration_seconds, context)
    results, saved, ignored, errors = data.results, data.saved, data.ignored, data.errors
    sector_map = data.sector_map
    mood = build_portfolio_mood(results)
    forecast_horizon = forecast_horizon_label(started_at)
    # MAIL-06: pełny ranking (pełna rada dla top N); 🎯 sekcja bierze jego czołówkę.
    ranked_signals = build_cycle_signals(results, context, top_n=len(results))
    trade_signals = ranked_signals[:_TOP_SIGNALS]
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
    # MAIL-06: pełna rada dla top N wg siły sygnału i symboli z leadu.
    full_detail = _full_detail_symbols(saved, ranked_signals, data.lead_items)
    # U5: panel historii głosów rady per inwestor (render zwraca "" gdy brak).
    council_history_html = render_council_history_html(context.council_history or [])
    # #3: ranking wiarygodności person — ranking (z progiem min_votes) przychodzi
    # gotowy z domeny; render zwraca "" gdy żadna persona nie ma dość głosów.
    persona_leaderboard_html = render_persona_leaderboard_html(
        context.persona_track_record or []
    )
    # FinOps: koszt cyklu. Cykl darmowy (0 wywołań) renderuje się JAWNIE —
    # to najciekawsza informacja w tej sekcji, nie brak danych.
    finops_html = render_finops_html(context.cycle_cost)
    model_scorecard_html = _build_model_scorecard_html(
        context.model_scorecard,
        now=started_at,
        model_loaded=context.ml_model_loaded,
    )
    # T1/T2/T4: sekcja Track Record (krzywa kapitału, kalibracja, lessons).
    # render zwraca "" gdy wszystkie trzy podsekcje puste. Duży dzień (próg
    # MAIL-06) dostaje wersję zwartą — zastrzeżenie musi zmieścić się w 100 KB.
    track_record_html = render_track_record_html(
        context.equity_curve, context.calibration_buckets, context.lessons,
        context.track_record_stats,
        compact=len(saved) > _FULL_DETAIL_TOP_N,
        expired_count=context.expired_unverified,
    )
    # Dane: sekcja Alpha Signals (insider/analitycy/opcje/social/earnings per
    # symbol + krzywa rentowności FRED). render zwraca "" gdy brak danych.
    alpha_html = render_alpha_html(
        context.alpha_signals or {}, context.yield_curve, context.alpha_prices
    )
    macro_risk_text = (
        render_risk_watch_text(context.macro_risk_report) if context.macro_risk_report else ""
    )
    suggestions = build_sector_suggestions(results, sector_map, context.sector_peers or {})
    suggestions_html = render_suggestions_html(suggestions)
    suggestions_text = render_suggestions_text(suggestions)

    started_local = data.started_local
    symbol_chips = symbol_chip_labels(context.symbol_track_record, (r.symbol for r in saved))
    html = _render_html(
        data,
        results, saved, ignored, errors, started_local, duration_seconds,
        mood, trade_signals, risk_signals,
        display_timezone=context.display_timezone,
        full_detail=full_detail,
        sector_map=sector_map,
        forecast_horizon=forecast_horizon,
        symbol_chips=symbol_chips,
    )
    # Kolejność jak w poprzedniej wersji — znaczniki są rozłączne, więc nie ma
    # znaczenia dla wyniku, ale trzyma diff czytelnym przy porównaniu z historią.
    html = _fill_html_slots(html, {
        _RISK_WATCH_SLOT: macro_risk_html,
        _PORTFOLIO_SLOT: portfolio_html,
        _COUNCIL_HISTORY_SLOT: council_history_html,
        _PERSONA_LEADERBOARD_SLOT: persona_leaderboard_html,
        _TRACK_RECORD_SLOT: track_record_html,
        _MODEL_SCORECARD_SLOT: model_scorecard_html,
        _FINOPS_SLOT: finops_html,
        _ALPHA_SLOT: alpha_html,
        _SUGGESTIONS_SLOT: suggestions_html,
        _RUN_FOOTER_SLOT: _render_run_footer_html(context.run_context),
    })
    html = _email_document(_compact_markup(html))
    text = _render_plain(
        data,
        results, saved, ignored, errors, started_local, duration_seconds,
        mood, trade_signals, risk_signals,
        display_timezone=context.display_timezone,
        sector_map=sector_map,
        forecast_horizon=forecast_horizon,
        symbol_chips=symbol_chips,
    )
    text = _fill_text_slots(
        text,
        macro_risk_text=macro_risk_text,
        portfolio_text=portfolio_text,
        suggestions_text=suggestions_text,
        # MAIL-09: sekcje z rendererem tekstu, których wersja tekstowa nie miała.
        sections={
            _MODEL_SCORECARD_SLOT: _build_model_scorecard_text(
                context.model_scorecard, now=started_at, model_loaded=context.ml_model_loaded
            ),
            _COUNCIL_HISTORY_SLOT: render_council_history_text(context.council_history or []),
            _FINOPS_SLOT: render_finops_text(context.cycle_cost),
        },
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


def _signal_label(sig: TradeSignal) -> str:
    """PORT-08: kierunek sygnału — przy znanym portfelu w kontekście pozycji
    („ZREDUKUJ (12% portfela)”), inaczej KUP / SPRZEDAJ / OBSERWUJ jak dotąd."""
    action = sig.position_action
    if action is None:
        return sig.direction
    held = action in (PositionAction.REDUCE, PositionAction.HOLD)
    if held and sig.position_weight is not None:
        return f"{action.label} ({_pct(sig.position_weight, decimals=0)} portfela)"
    return action.label


def _signal_tone(sig: TradeSignal) -> str:
    action = sig.position_action
    return sig.direction if action is None else _POSITION_ACTION_TONE[action]


def _signal_size_band(sig: TradeSignal) -> SizeBand | None:
    """Banda wielkości to sugestia wejścia — przy „UNIKAJ”, „ZREDUKUJ” i „TRZYMAJ,
    nie dokładaj” sprzeczna z kierunkiem, więc jej nie pokazujemy (PORT-08)."""
    if sig.position_action in (None, PositionAction.BUY):
        return sig.size_band
    return None


def _render_trade_signals_html(
    trade_signals: list[TradeSignal], symbol_chips: Mapping[str, str] | None = None
) -> str:
    """🎯 Najsilniejsze sygnały transakcyjne. Pusty string, gdy brak sygnałów.

    `symbol_chips` (WIA-35): trafność kierunkowa na danym symbolu pod wierszem."""
    if not trade_signals:
        return ""
    chips = symbol_chips or {}
    parts = [_section_h2("🎯 Najsilniejsze sygnały")]
    if any(sig.ml_cold_start for sig in trade_signals):
        parts.append(
            "<div style='font-size: 12px; color: #6b7280; margin-bottom: 8px;'>"
            f"{_html(_SIGNALS_COLD_START_NOTE)}</div>"
        )
    if any(sig.position_action is not None for sig in trade_signals):
        parts.append(
            "<div style='font-size: 12px; color: #6b7280; margin-bottom: 8px;'>"
            f"{_html(_SIGNALS_POSITION_NOTE)}</div>"
        )
    for rank, sig in enumerate(trade_signals, start=1):
        dir_color = {
            "KUP": PALETTE.positive,
            "SPRZEDAJ": PALETTE.negative,
            "OBSERWUJ": PALETTE.neutral,
        }[_signal_tone(sig)]
        dir_bg = {
            "KUP": "#f0fdf4",
            "SPRZEDAJ": "#fef2f2",
            "OBSERWUJ": "#f9fafb",
        }[_signal_tone(sig)]
        forecast_html = (
            f'<span style="color: #6b7280;">prognoza <strong>{_COLD_START_FORECAST}</strong></span>'
            if sig.ml_cold_start
            else (
                f'<span style="color: {_delta_color(sig.expected_change)};">'
                f"prognoza <strong>{direction_pct(sig.expected_change)}</strong></span>"
            )
        )
        # Q7: pasmo wielkości pozycji (Kelly-lite z konsensusu rady + hit-rate).
        size_band = _signal_size_band(sig)
        size_html = (
            f'<span style="color: #6b7280; font-size: 12px;">'
            f"{_html(size_band.label)}</span>"
            if size_band
            else ""
        )
        parts.append(f"""
              <div style="margin-bottom: 8px; padding: 10px 14px; background: {dir_bg};
                          border-left: 4px solid {dir_color}; border-radius: 4px;
                          font-size: 13px; display: flex; gap: 12px; align-items: center;">
                <span style="font-weight: 700; color: {dir_color}; min-width: 90px;">
                  {_html(_signal_label(sig))}
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
        chip = chips.get(sig.symbol)
        if chip:
            parts.append(
                "<div style='font-size: 12px; color: #6b7280; margin: -4px 0 8px 18px;'>"
                f"{_html(chip)}</div>"
            )
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


def _render_reasoning_block_html(
    r: SymbolResult, sector_map: Mapping[str, str], chip: str | None = None
) -> str:
    """Blok 'Uzasadnienia' dla jednej predykcji. Pusty string, gdy nie ma
    czego pokazać (brak reasoning / rady / odznak / analogów / atrybucji).
    `chip` (WIA-35) — trafność kierunkowa na symbolu, pod oczekiwanym ruchem."""
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
    if chip:
        move_line += f'<div class="rm">{_html(chip)}</div>'
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
    forecast_horizon: str = "",
    symbol_chips: Mapping[str, str] | None = None,
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

    chips = symbol_chips or {}
    parts.extend(
        _render_reasoning_block_html(r, sector_map, chips.get(r.symbol))
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
# MAIL-10: średnia bez ani jednego sentymentu to brak danych, nie „+0,00 (Neutralny)”.
_NO_SENTIMENT_LABEL = "brak danych"


def _mood_average(mood: PortfolioMood) -> tuple[str, str]:
    """(średni sentyment, etykieta) — „—” i „brak danych”, gdy cykl nie ma sentymentu."""
    if mood.avg_sentiment is None or mood.avg_sentiment_label is None:
        return "—", _NO_SENTIMENT_LABEL
    return _number(mood.avg_sentiment, decimals=2, signed=True), mood.avg_sentiment_label


def _render_mood_html(mood: PortfolioMood) -> str:
    """📊 Nastroje portfela — sentyment cyklu w Zapleczu. MAIL-10: cykl bez
    zapisanych predykcji (dzień zdegradowany, same pominięte) nie ma boxu."""
    if mood.saved_count == 0:
        return ""
    average, average_label = _mood_average(mood)
    bullish_part = (
        f"<strong>{_html(_company_label(mood.most_bullish.symbol))}</strong> "
        f"({_number(mood.most_bullish.sentiment_score, decimals=2, signed=True)})"
        if mood.most_bullish else "—"
    )
    bearish_part = (
        f"<strong>{_html(_company_label(mood.most_bearish.symbol))}</strong> "
        f"({_number(mood.most_bearish.sentiment_score, decimals=2, signed=True)})"
        if mood.most_bearish else "—"
    )
    return f"""
      <div style="padding: 12px 14px; background: #eff6ff; border-left: 3px solid #2563eb;
                  border-radius: 4px; margin-bottom: 20px; font-size: 13px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: #1e3a8a;">
          📊 Nastroje portfela
        </div>
        <div style="color: #1e40af; line-height: 1.7;">
          Średni sentyment: <strong>{average}</strong>
          ({average_label})<br/>
          Najbardziej pozytywny: {bullish_part} · Najbardziej negatywny: {bearish_part}<br/>
          Predykcje z wysoką pewnością (≥70%): {mood.high_confidence_count} / {mood.saved_count}
          · Łącznie newsów: {mood.total_news}
        </div>
      </div>
    """


def _render_html(
    data: SectionInput,
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    started_local: str,
    duration_seconds: float,
    mood: PortfolioMood,
    trade_signals: list[TradeSignal],
    risk_signals: list[RiskSignal],
    *,
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE,
    full_detail: frozenset[str] | None = None,
    sector_map: Mapping[str, str],
    forecast_horizon: str = "",
    symbol_chips: Mapping[str, str] | None = None,
) -> str:
    sections = [PREHEADER_SECTION.render_html(data)]

    sections.append(MASTHEAD_SECTION.render_html(data))
    sections.append(QUOTA_BANNER_SECTION.render_html(data))
    sections.append(LEAD_SECTION.render_html(data))
    sections.append(ONBOARDING_SECTION.render_html(data))
    sections.append(SESSION_SECTION.render_html(data))
    sections.append(FIRST_SCREEN_SECTION.render_html(data))
    sections.append(USER_PORTFOLIO_SECTION.render_html(data))
    # Zmiany od wczoraj: rozliczenia, zmiany nastawienia, nowy punkt odniesienia.
    sections.append(RESOLVED_PREDICTIONS_SECTION.render_html(data))
    sections.append(CONSENSUS_SHIFT_SECTION.render_html(data))
    sections.append(REBASELINED_SECTION.render_html(data))
    sections.append(_render_trade_signals_html(trade_signals, symbol_chips))
    sections.append(_render_risk_signals_html(risk_signals))
    sections.append(
        _render_saved_section_html(
            saved,
            full_detail,
            sector_map=sector_map,
            forecast_horizon=forecast_horizon,
            symbol_chips=symbol_chips,
        )
    )
    # 🪙 Krypto — osobna sekcja, widoczna też gdy cykl 'ignored' (próg 5%).
    sections.append(CRYPTO_SECTION.render_html(data))

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
    sections.append(ACCURACY_HISTORY_SECTION.render_html(data))
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
    sections.append(BUDGET_EXHAUSTED_SECTION.render_html(data))
    sections.append(IGNORED_SECTION.render_html(data))
    sections.append(ERRORS_SECTION.render_html(data))

    # MAIL-08: zastrzeżenie w stałej stopce, nie w sekcji warunkowej — sekcja,
    # która się chowa, zabrałaby je ze sobą.
    sections.append(f"""
      <div style="margin-top: 32px; padding-top: 16px; border-top: 1px solid #e5e7eb; font-size: 11px; color: {PALETTE.muted};">
        <div style="margin-bottom: 8px; color: #6b7280;">{_html(INVESTMENT_DISCLAIMER)}</div>
        {_render_technical_footer_html(results, saved, ignored, errors, duration_seconds)}
        <div style="margin-bottom: 8px;">{_html(DATA_SOURCES_FOOTER_PL)}</div>
        StockAgent · Agentic RAG + Continual Learning · raport automatyczny
        {_RUN_FOOTER_SLOT}
      </div>
    </div>
    """)

    return "".join(sections)


# ---------------------------------------------------------------------------
# Plain text fallback
# ---------------------------------------------------------------------------


def _render_trade_signals_text(
    trade_signals: list[TradeSignal], symbol_chips: Mapping[str, str] | None = None
) -> list[str]:
    """Sekcja 'Najsilniejsze sygnały' (plain). Pusta lista, gdy brak sygnałów."""
    if not trade_signals:
        return []
    chips = symbol_chips or {}
    lines = ["NAJSILNIEJSZE SYGNAŁY", "-" * 64]
    if any(sig.ml_cold_start for sig in trade_signals):
        lines.append(f"  {_SIGNALS_COLD_START_NOTE}")
    if any(sig.position_action is not None for sig in trade_signals):
        lines.append(f"  {_SIGNALS_POSITION_NOTE}")
    for rank, sig in enumerate(trade_signals, start=1):
        size_band = _signal_size_band(sig)
        size_txt = f"  [{size_band.label}]" if size_band else ""
        forecast = (
            _COLD_START_FORECAST if sig.ml_cold_start else _pct(sig.expected_change, signed=True)
        )
        lines.append(
            f"  {_signal_label(sig):9s} {_company_label(sig.symbol):40s} "
            f"pewność {_confidence_text(sig)}  "
            f"prognoza {forecast:>8s}  "
            f"{_strength_label(rank, sig)}{size_txt}"
        )
        chip = chips.get(sig.symbol)
        if chip:
            lines.append(f"            {chip}")
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


def _render_mood_text(mood: PortfolioMood) -> list[str]:
    """Sekcja 'Nastroje portfela' (plain) — w Zapleczu, jak w HTML (MAIL-10: bez
    zapisanych predykcji sekcji nie ma)."""
    if mood.saved_count == 0:
        return []
    average, average_label = _mood_average(mood)
    lines = [
        "NASTROJE PORTFELA",
        "-" * 64,
        f"  Średni sentyment: {average} ({average_label})",
    ]
    if mood.most_bullish:
        mb = mood.most_bullish
        lines.append(f"  Najbardziej pozytywny: {_company_label(mb.symbol)} ({_number(mb.sentiment_score, decimals=2, signed=True)})")
    if mood.most_bearish:
        mbe = mood.most_bearish
        lines.append(f"  Najbardziej negatywny: {_company_label(mbe.symbol)} ({_number(mbe.sentiment_score, decimals=2, signed=True)})")
    lines.append(
        f"  Wysoka pewność (≥70%): {mood.high_confidence_count} / {mood.saved_count}"
        f"  ·  Newsów: {mood.total_news}"
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
            f" (rel={_number(n.relevance, decimals=2)}, sent={_number(n.sentiment, decimals=2, signed=True)})"
        )
    valuation_text = _render_valuation_text(r.valuation)
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


def _render_plain(
    data: SectionInput,
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    started_local: str,
    duration_seconds: float,
    mood: PortfolioMood,
    trade_signals: list[TradeSignal],
    risk_signals: list[RiskSignal],
    *,
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE,
    sector_map: Mapping[str, str],
    forecast_horizon: str = "",
    symbol_chips: Mapping[str, str] | None = None,
) -> str:
    lines: list[str] = []
    lines.extend(QUOTA_BANNER_SECTION.render_text(data))
    lines.extend(LEAD_SECTION.render_text(data))
    lines.extend(ONBOARDING_SECTION.render_text(data))
    lines.extend(SESSION_SECTION.render_text(data))
    lines.extend(FIRST_SCREEN_SECTION.render_text(data))
    # MAIL-17: strefa „Na dziś”, kolejność jak w HTML.
    lines.extend(USER_PORTFOLIO_SECTION.render_text(data))
    # Zmiany od wczoraj: rozliczenia i nowy punkt odniesienia.
    lines.extend(RESOLVED_PREDICTIONS_SECTION.render_text(data))
    # MAIL-09: zmiany nastawienia rady — jak w HTML, za rozliczeniami.
    lines.extend(CONSENSUS_SHIFT_SECTION.render_text(data))
    lines.extend(REBASELINED_SECTION.render_text(data))
    lines.extend(_render_trade_signals_text(trade_signals, symbol_chips))
    lines.extend(_render_risk_signals_text(risk_signals))
    lines.extend(_render_saved_section_text(saved, sector_map, forecast_horizon, symbol_chips))
    # Krypto — osobna sekcja (widoczna też gdy 'ignored').
    lines.extend(CRYPTO_SECTION.render_text(data))

    # MAIL-17: „Zaplecze” — kontekst i kontrola za separatorem.
    lines.extend(["=" * 64, "ZAPLECZE", _BACKOFFICE_NOTE, "=" * 64, ""])
    # Slot na sekcję Risk Watch w wariancie plain text — analogicznie do HTML.
    lines.append(_RISK_WATCH_SLOT)
    # PORT-22: radar portfela (klastry, VaR, stress, „za mało historii”).
    lines.append(_PORTFOLIO_SLOT)
    lines.extend(ACCURACY_HISTORY_SECTION.render_text(data))
    lines.append(_MODEL_SCORECARD_SLOT)
    lines.append(_COUNCIL_HISTORY_SLOT)
    lines.extend(_render_mood_text(mood))
    # Wnioski z poprzednich cykli
    lines.extend(_render_reflections_text(saved))
    lines.append(_SUGGESTIONS_SLOT)
    lines.append(_FINOPS_SLOT)
    lines.extend(BUDGET_EXHAUSTED_SECTION.render_text(data))
    lines.extend(IGNORED_SECTION.render_text(data))
    lines.extend(ERRORS_SECTION.render_text(data))

    # MAIL-08: zastrzeżenie zamyka treść; stopka techniczna (MAIL-16) i stopka
    # przebiegu (MAIL-42) idą za nim.
    lines.extend([
        "-" * 64,
        INVESTMENT_DISCLAIMER,
        _technical_footer_text(results, saved, ignored, errors, duration_seconds),
        DATA_SOURCES_FOOTER_PL,
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
    # FIN-12: ruch przeszedł bramkę, odciął go budżet płatnych wywołań cyklu.
    if raw.get("budget_exhausted") is True:
        return SkipReason.BUDGET_EXHAUSTED
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
