"""DX-01: fikcyjny, deterministyczny cykl Fast Loopa do renderu przykładowego maila.

Produkt to mail, a żeby go zobaczyć, trzeba było mieć 4–6 kont API i bazę
z kompletem migracji. Tu jest jeden cykl z literałów: akcje (zapisana z radą
i newsami, zapisana z odznaką DEGRADED, pominięta bramką), ETF (zapisany
i nieobsługiwany cenowo), krypto (zapisane i pominięte), błąd źródła, zamknięte
predykcje, alert limitu, karta modelu, koszt cyklu i Track Record.

Zasady danych: stały `SAMPLE_STARTED_AT` (zegar mappera też, ARCH-26), kwoty
wymyślone, zero adresów e-mail i zero prawdziwych pozycji. Moduł nie czyta
env, sieci ani bazy, nie składa `Settings` (ARCH-39) i nie importuje `tests/`.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from src.application.report_builder import (
    ReportContext,
    ResolvedPrediction,
    SymbolResult,
    to_symbol_result,
)
from src.application.report_evidence import PaperCosts
from src.application.report_formatting import build_sector_map
from src.application.report_lessons import LessonItem, LessonsReport
from src.application.report_model_scorecard import ModelScorecardView
from src.application.report_track_record import TrackRecordStats
from src.application.run_context import RunContext
from src.application.use_cases.monitor_macro_risk import MacroRiskReport
from src.domain.asset import Asset
from src.domain.calibration_curve import CalibrationBucket
from src.domain.conformal import MoveInterval
from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.cycle_maturity import CycleMaturity, SkipReason
from src.domain.equity_curve import EquityCurve, EquityPoint
from src.domain.feature_attribution import FeatureContribution
from src.domain.finops import estimate_cycle_cost
from src.domain.macro_risk import MacroAlertLevel, MacroRiskInstrumentType, MacroRiskSignal
from src.domain.model_scorecard import summarize_evaluations, summarize_run
from src.domain.polish_macro import PolishMacroSnapshot
from src.domain.portfolio import Portfolio, Position
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.track_record import ScoredCall, assess_directional_hit_rate
from src.domain.value_objects import AssetType, Fundamentals, Money, Ratio, ValuationVerdict

# Czwartek 2026-10-15 10:30 UTC = premarket NYSE, 12:30 w Warszawie.
SAMPLE_STARTED_AT = datetime(2026, 10, 15, 10, 30, tzinfo=UTC)
SAMPLE_DURATION_SECONDS = 71.5
SAMPLE_DISPLAY_TIMEZONE = "Europe/Warsaw"
# Stopka mówi wprost, że to nie przebieg produkcyjny.
SAMPLE_RUN = RunContext(run_id="sample", git_sha="sample", trigger="sample")

Recommendation = Literal["BUY", "SELL", "HOLD"]

_SECTOR_MAP = build_sector_map(
    {"Półprzewodniki/AI": ["NVDA", "AMD", "ASML"], "Chmura/Software": ["MSFT"]},
    etf_symbols=("VOO", "XDWD.DE"),
    crypto_symbols=("BTC", "ETH"),
)
_SECTOR_PEERS = {
    "Półprzewodniki/AI": ["AVGO", "TXN", "AMAT", "KLAC"],
    "Chmura/Software": ["CRM", "ADBE", "NOW"],
}


def _fundamentals(age_hours: int) -> Fundamentals:
    return Fundamentals(
        trailing_pe=44.0,
        forward_pe=30.0,
        peg_ratio=1.3,
        eps_growth_yoy=0.55,
        fetched_at=SAMPLE_STARTED_AT - timedelta(hours=age_hours),
    )


def _opinion(
    name: str, recommendation: Recommendation, confidence: float, reasoning: str
) -> InvestorOpinion:
    return InvestorOpinion(
        investor_name=name,
        recommendation=recommendation,
        confidence=confidence,
        reasoning=reasoning,
        key_factors=("przykład",),
    )


def _council(recommendation: Recommendation, consensus: float) -> CouncilVerdict:
    return CouncilVerdict(
        final_recommendation=recommendation,
        consensus_strength=consensus,
        summary=f"Przykładowa rada skłania się ku {recommendation}.",
        dissenting_views=(
            "Makro i refleksyjność: część dobrych wiadomości jest już w cenie.",
        ),
        # MAIL-38: przykład trafia do publicznych docs, więc persony jako archetypy
        # (`data/council_personas/archetypes.toml`), nie nazwiska.
        investor_opinions=(
            _opinion(
                "Jakość i przewaga konkurencyjna",
                recommendation,
                0.72,
                "Przewaga skali i stabilne przepływy.",
            ),
            _opinion(
                "Wzrost w rozsądnej cenie",
                recommendation,
                0.66,
                "Wzrost zysków szybszy niż wycena.",
            ),
            _opinion(
                "Makro i refleksyjność",
                "HOLD",
                0.54,
                "Rynek wycenia już większość dobrych wiadomości.",
            ),
        ),
    )


def _news(symbol: str) -> list[dict[str, Any]]:
    return [
        {
            "title": f"{symbol}: sample headline about quarterly results",
            "source": "Example Wire",
            "url": f"https://news.example.com/{symbol.lower()}/results",
            "relevance_score": 0.88,
            "ticker_sentiment_score": 0.35,
        },
    ]


def _saved(
    symbol: str,
    asset_type: AssetType,
    *,
    previous: str,
    current: str,
    target: str,
    trend: str,
    confidence: float,
    reasoning: str,
    av_agreement: float = 0.8,
    sentiment: Mapping[str, Any] | None = None,
    data_quality_flags: tuple[str, ...] = (),
    **graph_output: Any,
) -> SymbolResult:
    """Zapisany symbol przez ten sam mapper co produkcja, z zegarem cyklu.

    `graph_output` to dodatkowe klucze wyjścia grafu (rada, newsy, fundamenty…)."""
    raw: dict[str, Any] = {
        "status": "saved",
        "asset": Asset(symbol=symbol, asset_type=asset_type),
        "previous_price": Decimal(previous),
        "current_price": Decimal(current),
        "delta": (Decimal(current) - Decimal(previous)) / Decimal(previous),
        "ml_target_price": Decimal(target),
        "llm_analysis": {
            "trend_direction": trend,
            "confidence_score": confidence,
            "reasoning": reasoning,
            "av_agreement": av_agreement,
        },
        "sentiment": {
            "av_sentiment_score": 0.3,
            "av_sentiment_label": "Somewhat-Bullish",
            "news_volume_24h": 12,
            **(sentiment or {}),
        },
        "data_quality_flags": list(data_quality_flags),
        **graph_output,
    }
    return to_symbol_result(symbol, raw, now=SAMPLE_STARTED_AT)


def _ignored(symbol: str, asset_type: AssetType, previous: str, delta: str) -> SymbolResult:
    raw = {
        "status": "ignored",
        "asset": Asset(symbol=symbol, asset_type=asset_type),
        "delta": Decimal(delta),
        "previous_price": Decimal(previous),
        "current_price": Decimal(previous) * (1 + Decimal(delta)),
    }
    return to_symbol_result(symbol, raw, now=SAMPLE_STARTED_AT)


def sample_results() -> list[SymbolResult]:
    """Wyniki cyklu: każdy status i każda klasa aktywa, które raport rozróżnia."""
    return [
        _saved(
            "NVDA",
            AssetType.STOCK,
            previous="120.00",
            current="125.40",
            target="128.10",
            trend="BULLISH",
            confidence=0.8,
            reasoning="Przykład: popyt na akceleratory rośnie szybciej niż podaż.",
            fundamentals=_fundamentals(age_hours=20),
            valuation_verdict=ValuationVerdict.FAIR,
            council_verdict=_council("BUY", 0.76),
            feature_attribution=(
                FeatureContribution(feature="price_delta", contribution=0.011),
                FeatureContribution(feature="av_sentiment_score", contribution=0.005),
                FeatureContribution(feature="news_volume_24h", contribution=-0.002),
            ),
            move_interval=MoveInterval(low=-0.02, high=0.025, calibration_size=60),
            similar_precedents=[
                {
                    "summary": "NVDA +4% po wynikach, sentyment Bullish (przykład)",
                    "predicted_trend": "BULLISH",
                    "is_trend_correct": True,
                },
            ],
            reflection_context="Poprzednia prognoza trafiona (przykład).",
            news=_news("NVDA"),
        ),
        _saved(
            "AMD",
            AssetType.STOCK,
            previous="150.00",
            current="145.20",
            target="142.90",
            trend="BEARISH",
            confidence=0.69,
            reasoning="Przykład: słabsze prognozy segmentu klienckiego.",
            av_agreement=0.25,
            sentiment={"av_sentiment_score": -0.2, "av_sentiment_label": "Somewhat-Bearish",
                       "degraded_reason": "av_keys_exhausted"},
            data_quality_flags=("av_keys_exhausted",),
            fundamentals=_fundamentals(age_hours=200),
            council_verdict=_council("SELL", 0.6),
            news=_news("AMD"),
        ),
        _saved(
            "VOO",
            AssetType.ETF,
            previous="500.00",
            current="511.00",
            target="512.50",
            trend="BULLISH",
            confidence=0.61,
            reasoning="Przykład: szeroki rynek odbija po danych o inflacji.",
        ),
        _saved(
            "BTC",
            AssetType.CRYPTO,
            previous="60000",
            current="63600",
            target="64500",
            trend="BULLISH",
            confidence=0.62,
            reasoning="Przykład: napływy do funduszy spot.",
            sentiment={"news_volume_24h": 2},
        ),
        _ignored("ETH", AssetType.CRYPTO, "3000", "0.011"),
        _ignored("MSFT", AssetType.STOCK, "400.00", "0.005"),
        SymbolResult(
            symbol="XDWD.DE",
            status="ignored",
            skip_reason=SkipReason.UNSUPPORTED_PRICE,
            error_message="unsupported by current price adapter (Finnhub free)",
        ),
        to_symbol_result(
            "ASML", None, error="HTTPError: 429 Too Many Requests", now=SAMPLE_STARTED_AT
        ),
    ]


def _resolved() -> list[ResolvedPrediction]:
    yesterday = SAMPLE_STARTED_AT - timedelta(days=1)
    return [
        ResolvedPrediction(
            symbol="NVDA",
            predicted_trend="BULLISH",
            is_correct=True,
            reasoning="Przykład: popyt przewyższa podaż.",
            price_at_prediction=Decimal("117.00"),
            actual_price=Decimal("120.00"),
            predicted_at=yesterday,
        ),
        ResolvedPrediction(
            symbol="AMD",
            predicted_trend="BULLISH",
            is_correct=False,
            reasoning="Przykład: odbicie po wyprzedaży.",
            insight="Przykład: rynek wycenił słabszy popyt.",
            price_at_prediction=Decimal("154.00"),
            actual_price=Decimal("150.00"),
            predicted_at=yesterday,
        ),
    ]


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
            signal("GLD", MacroRiskInstrumentType.SAFE_HAVEN, "230.00", "232.30"),
            signal("VIXY", MacroRiskInstrumentType.VOLATILITY, "12.00", "13.20"),
        ],
        polish_macro=PolishMacroSnapshot(
            eur_pln=Decimal("4.2500"),
            usd_pln=Decimal("3.7000"),
            eur_pln_30d_change_ratio=Ratio(Decimal("0.010")),
            usd_pln_30d_change_ratio=Ratio(Decimal("0.015")),
            fetched_at=SAMPLE_STARTED_AT - timedelta(hours=2),
        ),
        overall_alert=MacroAlertLevel.ELEVATED,
    )


def _portfolio() -> Portfolio:
    """Fikcyjny portfel — kwoty z sufitu, żeby pokazać sekcję „Twój portfel”."""

    def position(symbol: str, quantity: str, price: str) -> Position:
        return Position(
            symbol=symbol,
            quantity=Decimal(quantity),
            purchase_price=Decimal(price),
            purchase_date=date(2026, 3, 2),
        )

    return Portfolio(
        positions=(position("NVDA", "5", "100.00"), position("VOO", "2", "480.00")),
        as_of=SAMPLE_STARTED_AT - timedelta(days=1),
    )


def _equity_curve() -> EquityCurve:
    return EquityCurve(
        points=(
            EquityPoint(index=0, period_return=0.01, equity=1.01),
            EquityPoint(index=1, period_return=-0.006, equity=1.00394),
            EquityPoint(index=2, period_return=0.012, equity=1.01599),
        ),
        total_return=0.01599,
        win_rate=0.6,
        max_drawdown=0.006,
        n_trades=10,
    )


def _track_record_stats() -> TrackRecordStats:
    calls = [
        ScoredCall(
            cycle=f"2026-10-{day:02d}",
            predicted_trend="BULLISH" if day % 3 else "BEARISH",
            is_correct=day % 2 == 0,
            realized_change=0.01 if day % 2 == 0 else -0.008,
        )
        for day in range(1, 13)
    ]
    return TrackRecordStats(
        hit_rate=assess_directional_hit_rate(calls),
        costs=PaperCosts(cost_bps=10, crypto_cost_bps=25, crypto_symbols=frozenset({"BTC", "ETH"})),
    )


def _model_scorecard() -> ModelScorecardView:
    return ModelScorecardView(
        summary=summarize_run(["trained_successfully"]),
        trained_at=SAMPLE_STARTED_AT - timedelta(days=4),
        candidate_rmse=0.021,
        baseline_rmse=0.024,
        directional_hit_rate=0.55,
        folds=5,
        metrics_symbol="ALL",
        metrics_n_samples=240,
        evaluations=summarize_evaluations(["evaluated_passed_gate", "evaluated_failed_gate"]),
    )


def sample_context() -> ReportContext:
    """Kontekst raportu: sekcje, które w produkcji dociąga orkiestrator z bazy."""
    return ReportContext(
        accuracy_stats={
            "mean_accuracy": 0.58,
            "sample_count": 40,
            "correct_count": 23,
            "days_window": 30,
            "directional_count": 32,
            "directional_correct_count": 18,
        },
        resolved_predictions=_resolved(),
        macro_risk_report=_macro_risk_report(),
        quota_alerts=[
            QuotaAlert(
                source="Alpha Vantage",
                severity=QuotaSeverity.WARNING,
                message="Klucz 1 z 3 wyczerpany — rotacja na kolejny (przykład).",
                action="Sprawdź limit dzienny kluczy AV.",
                occurred_at=SAMPLE_STARTED_AT - timedelta(minutes=40),
            ),
        ],
        equity_curve=_equity_curve(),
        calibration_buckets=[
            CalibrationBucket(lower=0.6, upper=0.7, count=12, mean_confidence=0.65,
                              hit_rate=0.58),
        ],
        lessons=LessonsReport(
            items=(
                LessonItem(
                    symbol="AMD",
                    predicted_trend="BULLISH",
                    insight="Przykład: rynek wycenił słabszy popyt.",
                    actual_change_ratio=Ratio(Decimal("-0.026")),
                ),
            ),
            total_mistakes=1,
            recurring_theme="popyt",
        ),
        cycle_maturity=CycleMaturity.STEADY_STATE,
        model_scorecard=_model_scorecard(),
        ml_model_loaded=True,
        user_portfolio=_portfolio(),
        cycle_cost=estimate_cycle_cost(
            {"llm": 4, "council_llm": 2, "sentiment": 4, "news": 4, "embedding": 8}
        ),
        track_record_stats=_track_record_stats(),
        display_timezone=SAMPLE_DISPLAY_TIMEZONE,
        run_context=SAMPLE_RUN,
        sector_map=_SECTOR_MAP,
        sector_peers=_SECTOR_PEERS,
    )
