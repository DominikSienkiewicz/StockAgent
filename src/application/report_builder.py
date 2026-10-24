"""Buduje raport HTML+text z wyników cyklu Fast Loop.

Czysta logika prezentacji — bez I/O, łatwo testowalne.
"""

from __future__ import annotations

import dataclasses
import re
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
    format_local,
    safe_http_url,
)
from src.application.report_formatting import (
    company_label as _company_label,
)
from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_formatting import (
    number as _number,
)
from src.application.report_formatting import (
    section_h2 as _section_h2,
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
from src.application.report_predictions import PREDICTIONS_SECTION, full_detail_symbols
from src.application.report_quota_banner import QUOTA_BANNER_SECTION
from src.application.report_regime import (
    RegimeBanner,
)
from src.application.report_resolved import (
    RESOLVED_PREDICTIONS_SECTION,
    resolved_window,
)
from src.application.report_risk_signals import RISK_SIGNALS_SECTION
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
from src.application.report_skipped import (
    BUDGET_EXHAUSTED_SECTION,
    ERRORS_SECTION,
    IGNORED_SECTION,
    REBASELINED_SECTION,
)
from src.application.report_styles import (
    EMAIL_STYLESHEET,
)
from src.application.report_suggestions import (
    build_sector_suggestions,
    render_suggestions_html,
    render_suggestions_text,
)
from src.application.report_symbol_record import SymbolTrackRecordView, symbol_chip_labels
from src.application.report_track_record import (
    TrackRecordStats,
    render_track_record_html,
)
from src.application.report_trade_signals import TRADE_SIGNALS_SECTION
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
from src.domain.finops import CycleCost
from src.domain.insider_flow import InsiderLeadWindow
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.persona_track_record import PersonaTrackRecord
from src.domain.polish_labels import (
    trend_forecast_label,
)
from src.domain.portfolio import (
    PlnRates,
    Portfolio,
)
from src.domain.provenance import build_provenance_badges
from src.domain.quota import QuotaAlert
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
    # #1: ranking leadu robi domena; TU jest jedyne miejsce, gdzie DTO warstwy
    # application (SymbolResult / ResolvedPrediction) są mapowane na typy
    # domenowe — `src/domain/digest_lead.py` ich nie importuje.
    lead_items = build_cycle_lead(results, context)
    # MAIL-06: pełny ranking (pełna rada dla top N); 🎯 sekcja bierze jego czołówkę.
    ranked_signals = build_cycle_signals(results, context, top_n=len(results))
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
        lead_items=lead_items,
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
        trade_signals=ranked_signals[:_TOP_SIGNALS],
        full_detail=full_detail_symbols(saved, ranked_signals, lead_items),
        symbol_chips=symbol_chip_labels(context.symbol_track_record, (r.symbol for r in saved)),
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
        compact=data.large_day,
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
    html = _render_html(
        data,
        results, saved, ignored, errors, started_local, duration_seconds,
        mood,
        display_timezone=context.display_timezone,
        sector_map=sector_map,
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
        mood,
        display_timezone=context.display_timezone,
        sector_map=sector_map,
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
    url = safe_http_url(run_context.run_url)
    logs = (
        f' · <a href="{_html(url)}" style="color: {PALETTE.muted};">logi</a>'
        if url is not None
        else ""
    )
    return f"<br/>{label}{logs}"


def _render_run_footer_text(run_context: RunContext | None) -> str:
    if run_context is None:
        return ""
    url = safe_http_url(run_context.run_url)
    logs = f" · logi: {url}" if url is not None else ""
    return "\n".join(["-" * 64, f"{_run_footer_label(run_context)}{logs}"])


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


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
    *,
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE,
    sector_map: Mapping[str, str],
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
    sections.append(TRADE_SIGNALS_SECTION.render_html(data))
    sections.append(RISK_SIGNALS_SECTION.render_html(data))
    sections.append(PREDICTIONS_SECTION.render_html(data))
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


def _render_plain(
    data: SectionInput,
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    started_local: str,
    duration_seconds: float,
    mood: PortfolioMood,
    *,
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE,
    sector_map: Mapping[str, str],
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
    lines.extend(TRADE_SIGNALS_SECTION.render_text(data))
    lines.extend(RISK_SIGNALS_SECTION.render_text(data))
    lines.extend(PREDICTIONS_SECTION.render_text(data))
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
