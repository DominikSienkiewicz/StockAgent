from __future__ import annotations

from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from src.application.report_formatting import pct, score_to_pl_label
from src.application.report_models import (
    ResolvedPrediction,
    RiskSignal,
    SymbolResult,
    TradeSignal,
)
from src.domain.sizing import SizeBand, suggest_band

DIVERGENCE_PRICE_THRESHOLD = Decimal("0.02")
DIVERGENCE_SENTIMENT_THRESHOLD = 0.2
AV_LLM_CONFLICT_THRESHOLD = 0.3
LOW_SIGNAL_NEWS_THRESHOLD = 3

NYSE_TZ = ZoneInfo("America/New_York")
NYSE_OPEN = time(9, 30)
NYSE_CLOSE = time(16, 0)


def build_trade_signals(
    results: list[SymbolResult], top_n: int = 5, hit_rate: float | None = None
) -> list[TradeSignal]:
    """Top sygnały transakcyjne — posortowane po sile sygnału.

    Q7: gdy symbol ma werdykt rady (`council_verdict`), doklejamy sugerowaną
    bandę wielkości pozycji (Kelly-lite z konsensusu + dissentu rady i — gdy
    podany — historycznego hit-rate agenta). `hit_rate` jest wspólny dla
    całego cyklu (np. `accuracy_stats["mean_accuracy"]`); None = cold-start →
    sizing konserwatywny. Brak werdyktu rady → `size_band` None."""
    signals: list[TradeSignal] = []
    for r in results:
        if r.status != "saved":
            continue
        if r.confidence_score is None or r.expected_change is None:
            continue
        strength = r.confidence_score * float(abs(r.expected_change)) * 100
        signals.append(TradeSignal(
            symbol=r.symbol,
            direction=_trend_to_direction(r.trend),
            confidence=r.confidence_score,
            expected_change=r.expected_change,
            strength=strength,
            current_price=r.current_price,
            target_price=r.target_price,
            size_band=_size_band_for(r, hit_rate),
        ))
    signals.sort(key=lambda s: s.strength, reverse=True)
    return signals[:top_n]


def _size_band_for(r: SymbolResult, hit_rate: float | None) -> SizeBand | None:
    """Wylicza bandę pozycji z werdyktu rady symbolu; None gdy brak werdyktu."""
    verdict = r.council_verdict
    if verdict is None:
        return None
    return suggest_band(
        consensus_strength=verdict.consensus_strength,
        dissent_ratio=verdict.dissent_ratio(),
        hit_rate=hit_rate,
    )


def _divergence_signal(r: SymbolResult) -> RiskSignal | None:
    """Rozbieżność cena↔sentyment — możliwy pump albo niedoszacowane odbicie."""
    if r.delta is None or r.sentiment_score is None:
        return None
    if (
        r.delta > DIVERGENCE_PRICE_THRESHOLD
        and r.sentiment_score < -DIVERGENCE_SENTIMENT_THRESHOLD
    ):
        return RiskSignal(
            symbol=r.symbol, type="DIVERGENCE", severity="high",
            description=(
                f"cena rośnie {pct(r.delta)} ale sentyment "
                f"{r.sentiment_score:+.2f} -> możliwa korekta lub pump"
            ),
        )
    if (
        r.delta < -DIVERGENCE_PRICE_THRESHOLD
        and r.sentiment_score > DIVERGENCE_SENTIMENT_THRESHOLD
    ):
        return RiskSignal(
            symbol=r.symbol, type="DIVERGENCE", severity="high",
            description=(
                f"cena spada {pct(r.delta)} ale sentyment "
                f"{r.sentiment_score:+.2f} -> okazja lub odbicie?"
            ),
        )
    return None


def _av_llm_conflict_signal(r: SymbolResult) -> RiskSignal | None:
    """LLM kontra Alpha Vantage — potencjalny fake news / manipulacja."""
    if r.av_llm_agreement is None or r.av_llm_agreement >= AV_LLM_CONFLICT_THRESHOLD:
        return None
    return RiskSignal(
        symbol=r.symbol, type="AV_LLM_CONFLICT", severity="high",
        description=(
            f"LLM nie zgadza się z AV (zgoda={r.av_llm_agreement:.2f})"
            " -> potencjalny fake news / manipulacja"
        ),
    )


def _low_signal_signal(r: SymbolResult) -> RiskSignal | None:
    """Decyzja podjęta na zbyt małej liczbie relevantnych newsów."""
    if r.news_volume is None or r.news_volume >= LOW_SIGNAL_NEWS_THRESHOLD:
        return None
    return RiskSignal(
        symbol=r.symbol, type="LOW_SIGNAL", severity="medium",
        description=(
            f"tylko {r.news_volume} relevantnych newsów — decyzja"
            " na słabych danych"
        ),
    )


# Detektory uruchamiane per zapisany symbol; każdy zwraca sygnał albo None.
_RISK_DETECTORS = (_divergence_signal, _av_llm_conflict_signal, _low_signal_signal)


def detect_risk_signals(results: list[SymbolResult]) -> list[RiskSignal]:
    """Flagi ostrzegawcze per symbol."""
    out: list[RiskSignal] = []
    for r in results:
        if r.status != "saved":
            continue
        out.extend(sig for det in _RISK_DETECTORS if (sig := det(r)) is not None)
    return out


def parse_resolved_predictions(
    rows: list[dict[str, Any]],
) -> list[ResolvedPrediction]:
    """Mapuje rekordy z `prediction_logs` na DTO `ResolvedPrediction`.

    Trafność jedzie po `is_trend_correct` (zgodność kierunku) — NIE po
    `accuracy_score`, które przy cold-starcie jest ~100% także dla
    kierunkowo błędnych prognoz."""
    out: list[ResolvedPrediction] = []
    for row in rows:
        trend_correct = row.get("is_trend_correct")
        if trend_correct is None:
            continue
        out.append(ResolvedPrediction(
            symbol=str(row.get("symbol", "?")),
            predicted_trend=str(row.get("predicted_trend", "?")),
            is_correct=bool(trend_correct),
            reasoning=_clean_text(row.get("reasoning_text")),
            insight=_clean_text(row.get("correction_insights")),
            price_at_prediction=_to_decimal(row.get("price_at_prediction")),
            actual_price=_to_decimal(row.get("actual_price_after_12h")),
        ))
    return out


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return None


def build_portfolio_mood(results: list[SymbolResult]) -> dict[str, Any]:
    """Liczy agregaty dla całego portfela: avg sentiment, top mover, etc."""
    saved = [r for r in results if r.status == "saved"]
    with_sentiment = [r for r in results if r.sentiment_score is not None]
    sentiments = [r.sentiment_score for r in with_sentiment if r.sentiment_score is not None]

    avg = sum(sentiments) / len(sentiments) if sentiments else 0.0
    most_bullish = max(with_sentiment, key=lambda r: r.sentiment_score or 0, default=None)
    most_bearish = min(with_sentiment, key=lambda r: r.sentiment_score or 0, default=None)
    high_conf = sum(
        1 for r in saved if r.confidence_score is not None and r.confidence_score >= 0.7
    )
    total_news = sum(r.news_volume or 0 for r in results)

    return {
        "avg_sentiment": avg,
        "avg_sentiment_label": score_to_pl_label(avg),
        "most_bullish": most_bullish,
        "most_bearish": most_bearish,
        "high_confidence_count": high_conf,
        "saved_count": len(saved),
        "total_news": total_news,
    }


def market_status(now_utc: datetime) -> dict[str, str]:
    """Status sesji NYSE (uproszczony — bez świąt)."""
    nyc = now_utc.astimezone(NYSE_TZ)
    if nyc.weekday() >= 5:
        next_monday = nyc + timedelta(days=(7 - nyc.weekday()))
        return {
            "label": "🌙 Zamknięta (weekend)",
            "detail": f"Najbliższe otwarcie: {next_monday.strftime('%A %d.%m')} 09:30 ET",
        }
    cur = nyc.time()
    if NYSE_OPEN <= cur < NYSE_CLOSE:
        close_today = nyc.replace(hour=NYSE_CLOSE.hour, minute=NYSE_CLOSE.minute, second=0)
        delta = close_today - nyc
        return {
            "label": "🟢 Otwarta (regular hours)",
            "detail": f"Zamknięcie za {_humanize_delta(delta)}",
        }
    if cur < NYSE_OPEN:
        open_today = nyc.replace(hour=NYSE_OPEN.hour, minute=NYSE_OPEN.minute, second=0)
        delta = open_today - nyc
        return {
            "label": "🌅 Premarket",
            "detail": f"Otwarcie za {_humanize_delta(delta)}",
        }
    return {
        "label": "🌆 After-hours",
        "detail": "Otwarcie jutro 09:30 ET",
    }


def _trend_to_direction(trend: str | None) -> str:
    return {
        "BULLISH": "KUP",
        "BEARISH": "SPRZEDAJ",
        "SIDEWAYS": "OBSERWUJ",
    }.get(trend or "", "OBSERWUJ")


def _humanize_delta(delta: timedelta) -> str:
    total_min = max(0, int(delta.total_seconds() // 60))
    h, m = divmod(total_min, 60)
    if h > 0:
        return f"{h}h {m}min"
    return f"{m}min"
