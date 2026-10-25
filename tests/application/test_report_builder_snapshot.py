"""Golden snapshot maila (TEST-02): HTML i plain text trzech syntetycznych cykli.

Siatka bezpieczeństwa przed zmianami układu maila — każda różnica w treści
dowolnej sekcji wychodzi tu jako unified diff, nawet gdy żaden test
fragmentaryczny jej nie łapie. Cykle są w pełni deterministyczne: zamrożony
`started_at` (czwartek, premarket NYSE — bez gałęzi z nazwą dnia `%A`, zależną
od locale), stała strefa wyświetlania, zegar mappera wstrzyknięty (ARCH-26),
dane wyłącznie z literałów. Snapshot nie zależy od strefy czasu maszyny,
locale ani `PYTHONHASHSEED`.

Snapshoty: `tests/application/snapshots/<scenariusz>.html` i `.txt`. HTML jest
zapisany z jednym tagiem na linię, żeby diff w review był czytelny;
transformacja jest odwracalna, a test pilnuje tej odwracalności, więc
porównanie pozostaje dokładne co do znaku.

Zamierzona zmiana maila — odśwież snapshoty jawną komendą i przejrzyj diff:

    uv run pytest tests/application/test_report_builder_snapshot.py --update-golden
"""

from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from src.application.alpha_signals import AlphaSignals
from src.application.ports import StoredShockAlert
from src.application.report_builder import (
    ReportContext,
    ResolvedPrediction,
    SymbolResult,
    build_html_report,
    to_symbol_result,
)
from src.application.report_council_history import InvestorHistory
from src.application.report_evidence import PaperCosts
from src.application.report_formatting import build_sector_map
from src.application.report_lessons import LessonItem, LessonsReport
from src.application.report_model_scorecard import ModelScorecardView, snapshot_model_cards
from src.application.report_track_record import TrackRecordStats
from src.application.run_context import RunContext
from src.application.source_health import SOURCE_ORDER, SourceHealth, SourceStatus
from src.application.use_cases.monitor_macro_risk import MacroRiskReport
from src.application.use_cases.portfolio_risk import PortfolioRiskReport
from src.domain.analyst_consensus import AnalystConsensus
from src.domain.asset import Asset
from src.domain.benchmark import BenchmarkComparison
from src.domain.calibration_curve import CalibrationBucket
from src.domain.conformal import MoveInterval
from src.domain.consensus_shift import ConsensusShift, ShiftKind
from src.domain.council import CouncilVerdict, InvestorOpinion, Recommendation
from src.domain.cycle_maturity import CycleMaturity, SkipReason
from src.domain.drawdown import DrawdownSignal
from src.domain.equity_curve import DirectionalHitRate, EquityCurve, EquityPoint
from src.domain.feature_attribution import FeatureContribution
from src.domain.finops import BudgetBreach, BudgetWindow, estimate_cycle_cost
from src.domain.insider_flow import InsiderFlowSnapshot
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.macro_risk import MacroAlertLevel, MacroRiskInstrumentType, MacroRiskSignal
from src.domain.model_scorecard import summarize_evaluations, summarize_run
from src.domain.persona_track_record import PersonaTrackRecord
from src.domain.polish_macro import PolishMacroSnapshot
from src.domain.portfolio import Portfolio, Position
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.track_record import ScoredCall, assess_directional_hit_rate
from src.domain.value_objects import AssetType, Fundamentals, Money, Ratio, ValuationVerdict

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE_COMMAND = "uv run pytest tests/application/test_report_builder_snapshot.py --update-golden"

# Czwartek 2026-10-15 10:30 UTC = 06:30 ET (premarket) = 12:30 w Warszawie.
STARTED = datetime(2026, 10, 15, 10, 30, tzinfo=UTC)
DURATION_SECONDS = 74.2
DISPLAY_TIMEZONE = "Europe/Warsaw"

RUN = RunContext(
    run_id="18234567890",
    git_sha="6eff942115c0ffee",
    trigger="schedule",
    run_url="https://github.com/example/StockAgent/actions/runs/18234567890",
)
# PORT-05: sektory i peerzy idą z config.toml przez `ReportContext`. Fixture
# podaje dla symboli scenariuszy te same etykiety i peerów, co dawna mapa w kodzie
# (VOO i SPY nie miały w niej sektora, więc tu też go nie mają).
SECTOR_MAP = build_sector_map(
    {"Półprzewodniki/AI": ["NVDA", "AMD", "ASML"], "Chmura/Software": ["MSFT"]},
    etf_symbols=("XDWD.DE",),
    crypto_symbols=("BTC", "ETH"),
)
SECTOR_PEERS = {
    "Półprzewodniki/AI": ["AVGO", "TXN", "AMAT", "KLAC", "LRCX", "ARM", "MRVL"],
    "Chmura/Software": ["CRM", "ADBE", "NOW", "MDB", "WDAY", "HUBS"],
}


# ---------------------------------------------------------------------------
# Format pliku snapshotu
# ---------------------------------------------------------------------------


def _html_to_snapshot(html: str) -> str:
    """Jeden tag na linię: `><` → `>⏎<`, a `> <` → `>⏎ <` (spacja z maila zostaje
    na początku linii). Skompaktowany HTML nie ma `>` z białym znakiem innym niż
    pojedyncza spacja przed `<`, więc zapis da się odwrócić bez strat."""
    return html.replace("> <", ">\n <").replace("><", ">\n<") + "\n"


def _html_from_snapshot(snapshot: str) -> str:
    return snapshot.removesuffix("\n").replace(">\n <", "> <").replace(">\n<", "><")


def _text_to_snapshot(text: str) -> str:
    return text + "\n"


# ---------------------------------------------------------------------------
# Surowe wyjście grafu → SymbolResult (przez mapper, z zegarem cyklu)
# ---------------------------------------------------------------------------


def _fundamentals(age_hours: int) -> Fundamentals:
    return Fundamentals(
        trailing_pe=48.2,
        forward_pe=31.5,
        peg_ratio=1.4,
        eps_growth_yoy=0.62,
        fetched_at=STARTED - timedelta(hours=age_hours),
    )


def _council(
    recommendation: Recommendation, consensus: float, dissent: tuple[str, ...]
) -> CouncilVerdict:
    return CouncilVerdict(
        final_recommendation=recommendation,
        consensus_strength=consensus,
        summary=f"Rada skłania się ku {recommendation}: przewaga skali i rosnące marże.",
        dissenting_views=dissent,
        investor_opinions=(
            InvestorOpinion(
                investor_name="Buffett",
                recommendation=recommendation,
                confidence=0.74,
                reasoning="Trwała przewaga konkurencyjna i silne przepływy pieniężne.",
                key_factors=("moat", "FCF"),
            ),
            InvestorOpinion(
                investor_name="Lynch",
                recommendation=recommendation,
                confidence=0.68,
                reasoning="Wzrost zysków wyprzedza wycenę.",
                key_factors=("PEG", "wzrost EPS"),
            ),
            InvestorOpinion(
                investor_name="Soros",
                recommendation="HOLD",
                confidence=0.55,
                reasoning="Rynek wycenia już większość dobrych wiadomości.",
                key_factors=("refleksywność",),
            ),
        ),
    )


def _news(symbol: str) -> list[dict[str, Any]]:
    return [
        {
            "title": f"{symbol} beats quarterly estimates on data-center demand",
            "source": "Reuters",
            "url": f"https://news.example.com/{symbol.lower()}/earnings",
            "relevance_score": 0.91,
            "ticker_sentiment_score": 0.42,
        },
        {
            "title": f"Analysts lift {symbol} price targets after guidance",
            "source": "Bloomberg",
            "url": f"https://news.example.com/{symbol.lower()}/targets",
            "relevance_score": 0.77,
            "ticker_sentiment_score": 0.31,
        },
    ]


def _saved_stock_raw() -> dict[str, Any]:
    return {
        "status": "saved",
        "asset": Asset(symbol="NVDA", asset_type=AssetType.STOCK),
        "delta": Decimal("0.042"),
        "previous_price": Decimal("134.36"),
        "current_price": Decimal("140.00"),
        "ml_target_price": Decimal("143.50"),
        "llm_analysis": {
            "trend_direction": "BULLISH",
            "confidence_score": 0.82,
            "reasoning": "Popyt na akceleratory rośnie szybciej niż podaż; marże stabilne.",
            "av_agreement": 0.9,
        },
        "sentiment": {
            "av_sentiment_score": 0.41,
            "av_sentiment_label": "Bullish",
            "news_volume_24h": 18,
        },
        "data_quality_flags": [],
        "fundamentals": _fundamentals(age_hours=30),
        "valuation_verdict": ValuationVerdict.FAIR,
        "council_verdict": _council("BUY", 0.78, ("Soros: rynek przegrzany przed wynikami.",)),
        "feature_attribution": (
            FeatureContribution(feature="price_delta", contribution=0.0121),
            FeatureContribution(feature="av_sentiment_score", contribution=0.0064),
            FeatureContribution(feature="news_volume_24h", contribution=-0.0018),
        ),
        "move_interval": MoveInterval(low=-0.021, high=0.024, calibration_size=57),
        "similar_precedents": [
            {
                "summary": "NVDA +5% po wynikach, sentyment Bullish",
                "predicted_trend": "BULLISH",
                "is_trend_correct": True,
            },
        ],
        "reflection_context": "Poprzednia prognoza trafiona — popyt potwierdził tezę.",
        "news": _news("NVDA"),
    }


def _degraded_stock_raw() -> dict[str, Any]:
    return {
        "status": "saved",
        "asset": Asset(symbol="AMD", asset_type=AssetType.STOCK),
        "delta": Decimal("-0.035"),
        "previous_price": Decimal("165.80"),
        "current_price": Decimal("160.00"),
        "ml_target_price": Decimal("157.20"),
        "llm_analysis": {
            "trend_direction": "BEARISH",
            "confidence_score": 0.71,
            "reasoning": "Słabsze prognozy segmentu klienckiego i presja na ceny.",
            # Poniżej progu 0.3 — sygnał ostrzegawczy „konflikt AV vs LLM”.
            "av_agreement": 0.25,
        },
        "sentiment": {
            "av_sentiment_score": -0.18,
            "av_sentiment_label": "Somewhat-Bearish",
            "news_volume_24h": 9,
            "degraded_reason": "av_keys_exhausted",
        },
        "data_quality_flags": ["av_keys_exhausted", "stale_quote"],
        # 200 h względem startu cyklu — powyżej TTL 144 h, odznaka „Dane z cache”.
        "fundamentals": _fundamentals(age_hours=200),
        "council_verdict": _council("SELL", 0.61, ()),
        "news": _news("AMD"),
    }


def _crypto_saved_raw() -> dict[str, Any]:
    return {
        "status": "saved",
        "asset": Asset(symbol="BTC", asset_type=AssetType.CRYPTO),
        "delta": Decimal("0.061"),
        "previous_price": Decimal("61300"),
        "current_price": Decimal("65040"),
        "ml_target_price": Decimal("66100"),
        "llm_analysis": {
            "trend_direction": "BULLISH",
            "confidence_score": 0.64,
            "reasoning": "Napływy do ETF-ów spot i spadająca podaż na giełdach.",
        },
        "sentiment": {
            "av_sentiment_score": 0.22,
            "av_sentiment_label": "Somewhat-Bullish",
            # Mniej niż 3 newsy — sygnał ostrzegawczy „słaby sygnał”.
            "news_volume_24h": 2,
        },
        "data_quality_flags": [],
    }


def _ignored_raw(symbol: str, asset_type: AssetType, previous: str, delta: str) -> dict[str, Any]:
    return {
        "status": "ignored",
        "asset": Asset(symbol=symbol, asset_type=asset_type),
        "delta": Decimal(delta),
        "previous_price": Decimal(previous),
        "current_price": Decimal(previous) * (1 + Decimal(delta)),
    }


def _cold_start_raw(symbol: str, asset_type: AssetType, current: str) -> dict[str, Any]:
    """Pierwszy cykl: cena jest, punktu odniesienia jeszcze nie (previous_price 0)."""
    return {
        "status": "ignored",
        "asset": Asset(symbol=symbol, asset_type=asset_type),
        "delta": Decimal("0"),
        "previous_price": Decimal("0"),
        "current_price": Decimal(current),
    }


def _unsupported(symbol: str) -> SymbolResult:
    """Jak `main_agent._ignored_result` — ticker odcięty przed grafem."""
    return SymbolResult(
        symbol=symbol,
        status="ignored",
        skip_reason=SkipReason.UNSUPPORTED_PRICE,
        error_message="unsupported by current price adapter (Finnhub free)",
    )


def _mapped(symbol: str, raw: dict[str, Any]) -> SymbolResult:
    return to_symbol_result(symbol, raw, now=STARTED)


# ---------------------------------------------------------------------------
# Kontekst raportu
# ---------------------------------------------------------------------------


def _macro_risk_report() -> MacroRiskReport:
    def signal(symbol: str, kind: MacroRiskInstrumentType, prev: str, cur: str) -> MacroRiskSignal:
        return MacroRiskSignal(
            symbol=symbol,
            instrument_type=kind,
            current_price=Money(Decimal(cur)),
            previous_price=Money(Decimal(prev)),
        )

    return MacroRiskReport(
        signals=[
            signal("SH", MacroRiskInstrumentType.INVERSE_EQUITY, "40.00", "40.84"),
            signal("GLD", MacroRiskInstrumentType.SAFE_HAVEN, "245.00", "247.94"),
            signal("VIXY", MacroRiskInstrumentType.VOLATILITY, "12.00", "13.44"),
        ],
        polish_macro=PolishMacroSnapshot(
            eur_pln=Decimal("4.2710"),
            usd_pln=Decimal("3.6820"),
            eur_pln_30d_change_ratio=Ratio(Decimal("0.012")),
            usd_pln_30d_change_ratio=Ratio(Decimal("0.021")),
            fetched_at=STARTED - timedelta(hours=2),
        ),
        overall_alert=MacroAlertLevel.ELEVATED,
        drawdowns=[
            DrawdownSignal(symbol="AMD", current_price=Decimal("160.00"),
                           peak_price=Decimal("186.00")),
        ],
    )


def _portfolio_risk_report() -> PortfolioRiskReport:
    return PortfolioRiskReport(
        matrix={("AMD", "NVDA"): 0.86, ("BTC", "NVDA"): 0.31, ("AMD", "BTC"): 0.27},
        clusters=[("AMD", "NVDA")],
        verdict="Największy klaster: AMD + NVDA (2 z 3 symboli, 67% watchlisty)",
        watchlist_size=3,
        var=0.031,
        cvar=0.046,
    )


def _user_portfolio() -> Portfolio:
    def position(symbol: str, quantity: str, price: str) -> Position:
        return Position(
            symbol=symbol,
            quantity=Decimal(quantity),
            purchase_price=Decimal(price),
            purchase_date=date(2026, 3, 2),
        )

    return Portfolio(
        positions=(
            position("NVDA", "10", "118.00"),
            position("AMD", "15", "172.00"),
            position("XDWD.DE", "8", "110.00"),
        ),
        as_of=STARTED - timedelta(days=2),
    )


def _equity_curve() -> EquityCurve:
    return EquityCurve(
        points=(
            EquityPoint(index=0, period_return=0.012, equity=1.012),
            EquityPoint(index=1, period_return=-0.008, equity=1.0039),
            EquityPoint(index=2, period_return=0.015, equity=1.0190),
            EquityPoint(index=3, period_return=0.004, equity=1.0231),
        ),
        total_return=0.0231,
        win_rate=0.6,
        max_drawdown=0.008,
        n_trades=10,
    )


def _track_record_stats() -> TrackRecordStats:
    calls = [
        ScoredCall(
            cycle=f"2026-10-{day:02d}",
            predicted_trend="BULLISH" if day % 3 else "BEARISH",
            is_correct=day % 2 == 0,
            realized_change=0.012 if day % 2 == 0 else -0.009,
        )
        for day in range(1, 13)
    ]
    return TrackRecordStats(hit_rate=assess_directional_hit_rate(calls))


def _paper_costs() -> PaperCosts:
    """WIA-50: koszty jak w config.toml (10 bps, krypto 25 bps)."""
    return PaperCosts(cost_bps=10, crypto_cost_bps=25, crypto_symbols=frozenset({"BTC", "ETH"}))


def _benchmark() -> BenchmarkComparison:
    """WIA-21: agent z `_equity_curve()` obok koszyka i IVV na tych samych cyklach."""
    return BenchmarkComparison(
        agent=_equity_curve(),
        basket=EquityCurve(
            points=(
                EquityPoint(index=0, period_return=0.009, equity=1.009),
                EquityPoint(index=1, period_return=0.004, equity=1.0130),
                EquityPoint(index=2, period_return=0.011, equity=1.0242),
                EquityPoint(index=3, period_return=0.010, equity=1.0344),
            ),
            total_return=0.0344,
            win_rate=0.7,
            max_drawdown=0.0,
            n_trades=10,
        ),
        index=EquityCurve(
            points=(
                EquityPoint(index=0, period_return=0.006, equity=1.006),
                EquityPoint(index=1, period_return=0.003, equity=1.0090),
                EquityPoint(index=2, period_return=0.007, equity=1.0161),
                EquityPoint(index=3, period_return=0.005, equity=1.0212),
            ),
            total_return=0.0212,
            win_rate=1.0,
            max_drawdown=0.0,
            n_trades=4,
        ),
        index_symbol="IVV",
        market_hit_rate=DirectionalHitRate(hits=5, directional=9, sideways=1),
    )


def _model_scorecard() -> ModelScorecardView:
    return ModelScorecardView(
        summary=summarize_run(["trained_successfully"]),
        trained_at=STARTED - timedelta(days=4),
        candidate_rmse=0.0212,
        baseline_rmse=0.0240,
        directional_hit_rate=0.56,
        folds=5,
        metrics_symbol="ALL",
        metrics_n_samples=243,
        evaluations=summarize_evaluations(
            ["evaluated_passed_gate", "evaluated_passed_gate", "evaluated_failed_gate"]
        ),
    )


def _full_context() -> ReportContext:
    return ReportContext(
        accuracy_stats={
            "mean_accuracy": 0.62,
            "sample_count": 50,
            "correct_count": 31,
            "days_window": 30,
            "directional_count": 40,
            "directional_correct_count": 23,
        },
        resolved_predictions=[
            ResolvedPrediction(
                symbol="NVDA",
                predicted_trend="BULLISH",
                is_correct=True,
                reasoning="Popyt na akceleratory przewyższa podaż.",
                price_at_prediction=Decimal("131.00"),
                actual_price=Decimal("134.36"),
                predicted_at=STARTED - timedelta(days=1),
            ),
            ResolvedPrediction(
                symbol="AMD",
                predicted_trend="BULLISH",
                is_correct=False,
                reasoning="Odbicie po wyprzedaży.",
                insight="Rynek wycenił słabszy popyt w segmencie klienckim.",
                price_at_prediction=Decimal("170.00"),
                actual_price=Decimal("165.80"),
                predicted_at=STARTED - timedelta(days=1),
            ),
            ResolvedPrediction(
                symbol="SPY",
                predicted_trend="SIDEWAYS",
                is_correct=True,
                price_at_prediction=Decimal("580.00"),
                actual_price=Decimal("581.10"),
                predicted_at=STARTED - timedelta(days=1),
            ),
        ],
        macro_risk_report=_macro_risk_report(),
        portfolio_risk_report=_portfolio_risk_report(),
        council_history=[
            InvestorHistory(investor_name="Buffett", symbol="NVDA",
                            recommendations=("BUY", "BUY", "HOLD")),
            InvestorHistory(investor_name="Soros", symbol="NVDA",
                            recommendations=("HOLD", "SELL", "SELL")),
        ],
        persona_track_record=[
            PersonaTrackRecord(investor_name="Buffett", hit_rate=0.68, votes=25),
            PersonaTrackRecord(investor_name="Soros", hit_rate=0.52, votes=21),
        ],
        quota_alerts=[
            QuotaAlert(
                source="Alpha Vantage",
                severity=QuotaSeverity.WARNING,
                message="Klucz 1 z 3 wyczerpany — rotacja na kolejny.",
                action="Sprawdź limit dzienny kluczy AV.",
                occurred_at=STARTED - timedelta(minutes=50),
            ),
        ],
        equity_curve=_equity_curve(),
        calibration_buckets=[
            CalibrationBucket(lower=0.6, upper=0.7, count=14, mean_confidence=0.65,
                              hit_rate=0.57),
            CalibrationBucket(lower=0.8, upper=0.9, count=9, mean_confidence=0.84,
                              hit_rate=0.67),
        ],
        lessons=LessonsReport(
            items=(
                LessonItem(
                    symbol="AMD",
                    predicted_trend="BULLISH",
                    insight="Rynek wycenił słabszy popyt w segmencie klienckim.",
                    actual_change_ratio=Ratio(Decimal("-0.0247")),
                ),
            ),
            total_mistakes=1,
            recurring_theme="popyt",
        ),
        alpha_signals={
            "NVDA": AlphaSignals(
                symbol="NVDA",
                insider=InsiderFlowSnapshot(symbol="NVDA", net_shares=-12000.0,
                                            buy_count=1, sell_count=4, window_days=90),
                analyst=AnalystConsensus(symbol="NVDA", strong_buy=18, buy=22, hold=6,
                                         sell=1, price_target=165.0),
            ),
        },
        yield_curve=YieldCurveSnapshot(ten_year=4.12, two_year=3.94, fed_funds=4.33),
        alpha_prices={"NVDA": Decimal("140.00")},
        cycle_maturity=CycleMaturity.STEADY_STATE,
        model_scorecard=_model_scorecard(),
        ml_model_loaded=True,
        consensus_shifts=[
            (
                "NVDA",
                ConsensusShift(
                    kind=ShiftKind.FLIP,
                    previous_recommendation="SELL",
                    current_recommendation="BUY",
                    votes_changed=2,
                    sentiment_delta=0.31,
                    consensus_delta=0.12,
                    gap_hours=24.0,
                ),
            ),
        ],
        user_portfolio=_user_portfolio(),
        portfolio_clusters={"NVDA": "Półprzewodniki", "AMD": "Półprzewodniki"},
        cycle_cost=estimate_cycle_cost(
            {"llm": 3, "council_llm": 2, "sentiment": 3, "news": 3, "embedding": 6}
        ),
        track_record_stats=dataclasses.replace(
            _track_record_stats(), costs=_paper_costs(), benchmark=_benchmark()
        ),
        display_timezone=DISPLAY_TIMEZONE,
        run_context=RUN,
        sector_map=SECTOR_MAP,
        sector_peers=SECTOR_PEERS,
    )


# ---------------------------------------------------------------------------
# Scenariusze
# ---------------------------------------------------------------------------


def _full_cycle() -> tuple[str, str]:
    """Kilka statusów naraz: zapisane akcje (rada, newsy, odznaki), krypto,
    pominięte poniżej progu i nieobsługiwane cenowo, błąd źródła — plus pełny
    kontekst: Risk Watch, portfel, Track Record, scorecard, koszt, stopka."""
    results = [
        _mapped("NVDA", _saved_stock_raw()),
        _mapped("AMD", _degraded_stock_raw()),
        _mapped("BTC", _crypto_saved_raw()),
        _mapped("ETH", _ignored_raw("ETH", AssetType.CRYPTO, "3200", "0.012")),
        _mapped("MSFT", _ignored_raw("MSFT", AssetType.STOCK, "410.00", "0.006")),
        _unsupported("XDWD.DE"),
        to_symbol_result("ASML", None, error="HTTPError: 429 Too Many Requests", now=STARTED),
    ]
    return build_html_report(results, STARTED, DURATION_SECONDS, _full_context())


def _all_ignored_cycle() -> tuple[str, str]:
    """Spokojny dzień: bramka volatility odcina wszystko, cykl jest darmowy."""
    results = [
        _mapped("NVDA", _ignored_raw("NVDA", AssetType.STOCK, "140.00", "0.004")),
        _mapped("AMD", _ignored_raw("AMD", AssetType.STOCK, "160.00", "-0.007")),
        _mapped("VOO", _ignored_raw("VOO", AssetType.ETF, "530.00", "0.002")),
        _mapped("ETH", _ignored_raw("ETH", AssetType.CRYPTO, "3200", "-0.011")),
    ]
    context = ReportContext(
        accuracy_stats={
            "mean_accuracy": 0.62,
            "sample_count": 50,
            "correct_count": 31,
            "days_window": 30,
            "directional_count": 40,
            "directional_correct_count": 23,
        },
        cycle_maturity=CycleMaturity.STEADY_STATE,
        cycle_cost=estimate_cycle_cost({}),
        display_timezone=DISPLAY_TIMEZONE,
        run_context=RUN,
        sector_map=SECTOR_MAP,
        sector_peers=SECTOR_PEERS,
    )
    return build_html_report(results, STARTED, DURATION_SECONDS, context)


def _first_run_cycle() -> tuple[str, str]:
    """Dzień 1: żaden symbol nie ma punktu odniesienia (cold-start)."""
    results = [
        _mapped("NVDA", _cold_start_raw("NVDA", AssetType.STOCK, "140.00")),
        _mapped("AMD", _cold_start_raw("AMD", AssetType.STOCK, "160.00")),
        _mapped("BTC", _cold_start_raw("BTC", AssetType.CRYPTO, "65040")),
    ]
    context = ReportContext(
        cycle_maturity=CycleMaturity.FIRST_RUN,
        portfolio_sectors={"Półprzewodniki": 2, "Krypto": 1},
        cycle_cost=estimate_cycle_cost({}),
        display_timezone=DISPLAY_TIMEZONE,
        run_context=RunContext(),
        sector_map=SECTOR_MAP,
        sector_peers=SECTOR_PEERS,
    )
    return build_html_report(results, STARTED, DURATION_SECONDS, context)


def _degraded_sources_cycle() -> tuple[str, str]:
    """Fala 2: dzień z awarią źródła i szokiem z watcha — stan źródeł (MAIL-21),
    szoki między mailami i lead z szoku na pozycji (ALR-06), próg efektywny
    i „blisko progu” w ⏸ Pominięte (MAIL-18), karty modeli z `feature_snapshots`
    (ML-52) i link do prywatnego archiwum (MAIL-24)."""
    results = [
        _mapped(
            "NVDA",
            {
                **_ignored_raw("NVDA", AssetType.STOCK, "140.00", "-0.016"),
                "effective_threshold": Decimal("0.02"),
            },
        ),
        _mapped(
            "AMD",
            {
                **_ignored_raw("AMD", AssetType.STOCK, "160.00", "0.012"),
                "effective_threshold": Decimal("0.03"),
                "regime_multiplier": 1.5,
            },
        ),
        _mapped(
            "ETH",
            {
                **_ignored_raw("ETH", AssetType.CRYPTO, "3200", "0.011"),
                "effective_threshold": Decimal("0.05"),
            },
        ),
    ]
    scorecard = dataclasses.replace(
        _model_scorecard(),
        snapshot_models=snapshot_model_cards(
            [
                {"symbol": "ALL_FREE", "status": "trained_successfully",
                 "timestamp": "2026-10-11T03:36:00+00:00", "candidate_holdout_rmse": 0.021,
                 "baseline_rmse": 0.024, "n_samples": 812},
                {"symbol": "ALL_LLM", "status": "skipped",
                 "timestamp": "2026-10-11T03:37:00+00:00", "n_samples": 37},
            ]
        ),
    )
    context = ReportContext(
        cycle_maturity=CycleMaturity.STEADY_STATE,
        cycle_cost=estimate_cycle_cost({}),
        display_timezone=DISPLAY_TIMEZONE,
        run_context=RUN,
        sector_map=SECTOR_MAP,
        sector_peers=SECTOR_PEERS,
        user_portfolio=_user_portfolio(),
        model_scorecard=scorecard,
        source_health=tuple(
            {
                "Finnhub": SourceHealth("Finnhub", SourceStatus.UNAVAILABLE, affected=2),
                "Alpha Vantage": SourceHealth("Alpha Vantage", SourceStatus.DEGRADED, affected=3),
                "EDGAR": SourceHealth("EDGAR", SourceStatus.DISABLED),
                "Reddit": SourceHealth("Reddit", SourceStatus.DISABLED),
            }.get(source, SourceHealth(source, SourceStatus.OK))
            for source in SOURCE_ORDER
        ),
        shock_alerts=[
            StoredShockAlert(
                symbol="NVDA",
                delta=-0.062,
                direction="DOWN",
                created_at=STARTED - timedelta(hours=9),
            ),
        ],
        archive_url=(
            "https://proj.supabase.co/storage/v1/object/sign/digests/"
            "20261015T1030Z-18234567890.html?token=signed"
        ),
    )
    return build_html_report(results, STARTED, DURATION_SECONDS, context)


def _llm_budget_spent_cycle() -> tuple[str, str]:
    """FIN-25: dzienny limit USD z ledgera wyczerpany przed cyklem — banner limitu
    na górze, ruch ponad progiem w „wyczerpany budżet” (bez analizy), cykl darmowy."""
    results = [
        _mapped(
            "NVDA",
            {**_ignored_raw("NVDA", AssetType.STOCK, "140.00", "-0.035"), "budget_exhausted": True},
        ),
        _mapped("AMD", _ignored_raw("AMD", AssetType.STOCK, "160.00", "0.004")),
        _mapped("ETH", _ignored_raw("ETH", AssetType.CRYPTO, "3200", "-0.011")),
    ]
    context = ReportContext(
        cycle_maturity=CycleMaturity.STEADY_STATE,
        cycle_cost=estimate_cycle_cost({}),
        display_timezone=DISPLAY_TIMEZONE,
        run_context=RUN,
        sector_map=SECTOR_MAP,
        sector_peers=SECTOR_PEERS,
        llm_budget_breach=BudgetBreach(BudgetWindow.DAY, spent_usd=2.134, cap_usd=2.0),
    )
    return build_html_report(results, STARTED, DURATION_SECONDS, context)


def _quiet_day_compact_cycle() -> tuple[str, str]:
    """MAIL-35: cichy dzień z polityką `compact_when_quiet` — nagłówek, sesja, blok
    trzech linijek (największy ruch z progiem, nic za bramką, portfel właściciela)
    i pełna stopka ze źródłami i warunkami dostawców."""
    results = [
        _mapped(
            "NVDA",
            {
                **_ignored_raw("NVDA", AssetType.STOCK, "140.00", "0.004"),
                "effective_threshold": Decimal("0.02"),
            },
        ),
        _mapped(
            "AMD",
            {
                **_ignored_raw("AMD", AssetType.STOCK, "160.00", "-0.012"),
                "effective_threshold": Decimal("0.02"),
            },
        ),
        _mapped(
            "ETH",
            {
                **_ignored_raw("ETH", AssetType.CRYPTO, "3200", "-0.011"),
                "effective_threshold": Decimal("0.05"),
            },
        ),
    ]
    context = ReportContext(
        cycle_maturity=CycleMaturity.STEADY_STATE,
        cycle_cost=estimate_cycle_cost({}),
        display_timezone=DISPLAY_TIMEZONE,
        run_context=RUN,
        sector_map=SECTOR_MAP,
        sector_peers=SECTOR_PEERS,
        user_portfolio=_user_portfolio(),
        resolved_predictions=[],
    )
    return build_html_report(results, STARTED, DURATION_SECONDS, context, compact=True)


SCENARIOS: dict[str, Callable[[], tuple[str, str]]] = {
    "full_cycle": _full_cycle,
    "all_ignored": _all_ignored_cycle,
    "first_run": _first_run_cycle,
    "degraded_sources": _degraded_sources_cycle,
    "llm_budget_spent": _llm_budget_spent_cycle,
    "quiet_day_compact": _quiet_day_compact_cycle,
}


# ---------------------------------------------------------------------------
# Porównanie
# ---------------------------------------------------------------------------


def _golden_mismatch(path: Path, actual_snapshot: str) -> str | None:
    """None, gdy plik zgadza się co do znaku; inaczej opis różnicy (unified diff)."""
    if not path.exists():
        return f"Brak snapshotu snapshots/{path.name}."
    expected_snapshot = path.read_text(encoding="utf-8")
    if actual_snapshot == expected_snapshot:
        return None
    return "".join(
        difflib.unified_diff(
            expected_snapshot.splitlines(keepends=True),
            actual_snapshot.splitlines(keepends=True),
            fromfile=f"snapshots/{path.name} (zapisany)",
            tofile=f"snapshots/{path.name} (wygenerowany)",
        )
    )


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_report_matches_golden(scenario: str, request: pytest.FixtureRequest) -> None:
    html, text = SCENARIOS[scenario]()
    html_snapshot = _html_to_snapshot(html)
    # Format pliku musi dać się odwrócić bez strat — inaczej zgodność snapshotu
    # nie oznaczałaby zgodności maila co do znaku.
    assert _html_from_snapshot(html_snapshot) == html
    snapshots = {
        SNAPSHOT_DIR / f"{scenario}.html": html_snapshot,
        SNAPSHOT_DIR / f"{scenario}.txt": _text_to_snapshot(text),
    }

    if request.config.getoption("--update-golden"):
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for path, snapshot in snapshots.items():
            path.write_text(snapshot, encoding="utf-8", newline="\n")
        return

    mismatches = [
        mismatch
        for path, snapshot in snapshots.items()
        if (mismatch := _golden_mismatch(path, snapshot)) is not None
    ]
    if mismatches:
        pytest.fail(
            f"Mail scenariusza {scenario} różni się od snapshotu.\n"
            f"Zamierzona zmiana → odśwież: {UPDATE_COMMAND}\n\n" + "\n".join(mismatches),
            pytrace=False,
        )


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_report_is_identical_across_builds(scenario: str) -> None:
    assert SCENARIOS[scenario]() == SCENARIOS[scenario]()
