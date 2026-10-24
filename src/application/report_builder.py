"""Buduje raport HTML+text z wyników cyklu Fast Loop.

Czysta logika prezentacji — bez I/O, łatwo testowalne. ARCH-12: mail składa się
z sekcji rejestru (`src/application/reporting/registry.py`); tu żyje kontekst
raportu (`ReportContext`), wspólne wejście sekcji liczone raz na render
(`_section_input`), ranking leadu i sygnałów (też dla tematu i pushu) oraz
mapowanie wyjścia grafu na `SymbolResult`.
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
from src.application.ports import StoredShockAlert
from src.application.report_alpha import analyst_revision_label, significant_buy_label
from src.application.report_charts import (
    build_correlation_chart_html,
    build_delta_chart_html,
    build_forecast_chart_html,
)
from src.application.report_council_history import InvestorHistory
from src.application.report_formatting import (
    DEFAULT_DISPLAY_TIMEZONE,
    format_local,
)
from src.application.report_lessons import LessonsReport
from src.application.report_model_scorecard import ModelScorecardView
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
from src.application.report_predictions import full_detail_symbols
from src.application.report_regime import RegimeBanner
from src.application.report_resolved import resolved_window
from src.application.report_signals import (
    build_portfolio_mood,
    build_trade_signals,
    detect_risk_signals,
    market_status,
    parse_resolved_predictions,
)
from src.application.report_styles import EMAIL_STYLESHEET
from src.application.report_symbol_record import SymbolTrackRecordView, symbol_chip_labels
from src.application.report_track_record import TrackRecordStats
from src.application.report_user_portfolio import portfolio_change_view
from src.application.reporting.registry import REPORT_SECTIONS
from src.application.reporting.section import SectionInput
from src.application.run_context import RunContext
from src.application.source_health import SourceHealth
from src.application.use_cases.monitor_macro_risk import MacroRiskReport
from src.application.use_cases.portfolio_risk import PortfolioRiskReport, PositionRisk
from src.domain.asset import PriceDelta
from src.domain.calibration_curve import CalibrationBucket
from src.domain.conformal import MoveInterval
from src.domain.consensus_shift import ConsensusShift
from src.domain.council import CouncilVerdict
from src.domain.council_redundancy import CouncilRedundancy
from src.domain.cycle_maturity import CycleMaturity, SkipReason
from src.domain.digest_lead import LeadItem, LeadSignal, build_lead
from src.domain.earnings import EarningsEvent
from src.domain.equity_curve import EquityCurve
from src.domain.finops import CycleCost
from src.domain.insider_flow import InsiderLeadWindow
from src.domain.isotonic_calibration import IsotonicMap
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.persona_track_record import PersonaTrackRecord
from src.domain.polish_labels import trend_forecast_label
from src.domain.portfolio import (
    PlnRates,
    Portfolio,
)
from src.domain.provenance import build_provenance_badges
from src.domain.quota import QuotaAlert
from src.domain.value_objects import (
    FUNDAMENTALS_CACHE_TTL_HOURS,
    Percent,
    Ratio,
    ValuationVerdict,
    to_percent,
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
    shock_alerts: Sequence[StoredShockAlert] | None = None,
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
    ALR-06: `shock_alerts` (szoki z watcha od ostatniego maila) dają sygnałowi
    największy szok symbolu — na pozycji z portfela to kandydat leadu.
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
    shocks = _largest_shocks(shock_alerts or ())
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
                shock_pct=shocks.get(r.symbol),
                analyst_revision_label=_analyst_revision_label(alpha_signals, r.symbol),
            )
        )
    return signals


def _largest_shocks(alerts: Sequence[StoredShockAlert]) -> dict[str, Percent]:
    """ALR-06: symbol → największy |szok| od ostatniego maila jako PROCENT."""
    largest: dict[str, Decimal] = {}
    for alert in alerts:
        delta = Decimal(str(alert.delta))
        if alert.symbol not in largest or abs(delta) > abs(largest[alert.symbol]):
            largest[alert.symbol] = delta
    return {symbol: to_percent(Ratio(delta)) for symbol, delta in largest.items()}


def _analyst_revision_label(
    alpha_signals: Mapping[str, AlphaSignals] | None, symbol: str
) -> str | None:
    """DAT-13: tekst rewizji konsensusu m/m symbolu (ten sam co w sekcji alfy) albo None."""
    signal = alpha_signals.get(symbol) if alpha_signals else None
    if signal is None or signal.analyst is None:
        return None
    return analyst_revision_label(signal.analyst.revision())


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
        context.shock_alerts,
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
        isotonic_calibration=context.signal_isotonic_calibration,
        portfolio_weights=portfolio.weights(weight_prices) if portfolio is not None else None,
        cluster_exposures=_cluster_exposure_per_symbol(
            portfolio, context.portfolio_clusters, weight_prices
        ),
        portfolio_var=sizing_risk.var,
        drawdown=sizing_risk.drawdown,
        hedge_effective=sizing_risk.hedge_effective,
    )


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
    council_redundancy: CouncilRedundancy | None = None
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
    # ML-28: mapa izotoniczna, która wygrała poza próbą — zastępuje w rankingu
    # shrinkage kubełkowy (None = kubełki albo surowa pewność, jak dotąd).
    signal_isotonic_calibration: IsotonicMap | None = None
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
    # WIA-49: predykcje wygasłe bez oceny — `main_agent` podaje liczbę z tego przebiegu,
    # a `ReportAssembler` podnosi ją do liczby z okna track recordu.
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
    # MAIL-21: stan każdego źródła danych w cyklu (obwody HTTP, flagi jakości,
    # błędne odpowiedzi). None = sekcja 📡 bez danych, jak przed MAIL-21.
    source_health: tuple[SourceHealth, ...] | None = None
    # ALR-06: alerty szoku z watcha od ostatniego maila — sekcja ⚡ Między mailami
    # i kandydat leadu na pozycji z portfela. None = nie czytano (flaga watcha off).
    shock_alerts: list[StoredShockAlert] | None = None
    # MAIL-24: podpisany URL pełnego digestu w prywatnym archiwum (TTL 7 dni) —
    # link w nagłówku HTML i w bloku tytułowym tekstu. None = mail bez linku.
    archive_url: str | None = None
    # DAT-20: raporty wyników watchlisty do niedzieli — tylko w poniedziałek, z flagą
    # `finnhub_earnings_calendar_enabled`. None = sekcji „📅 Ten tydzień” nie ma.
    earnings_this_week: tuple[EarningsEvent, ...] | None = None


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
    return _email_document(_compact_markup(_render_html(data))), _render_text(data)


def _render_html(data: SectionInput) -> str:
    """ARCH-12: HTML maila — fragmenty sekcji rejestru w jego kolejności."""
    return "".join(section.render_html(data) for section in REPORT_SECTIONS)


def _render_text(data: SectionInput) -> str:
    """ARCH-12: wersja tekstowa — linie sekcji rejestru w tej samej kolejności."""
    return "\n".join(line for section in REPORT_SECTIONS for line in section.render_text(data))


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
        # MAIL-18: próg efektywny wyniku 'ignored' i mnożniki, które go podniosły.
        effective_threshold=_as_decimal(raw.get("effective_threshold")),
        regime_multiplier=_multiplier_of(raw, "regime_multiplier"),
        earnings_multiplier=_multiplier_of(raw, "earnings_multiplier"),
        data_quality_flags=tuple(str(flag) for flag in raw.get("data_quality_flags") or ()),
    )


def _multiplier_of(raw: dict[str, Any], key: str) -> float:
    """MAIL-18: mnożnik progu ze stanu grafu; brak albo zniekształcony = 1.0."""
    value = raw.get(key)
    return float(value) if isinstance(value, int | float) and value > 0 else 1.0


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
