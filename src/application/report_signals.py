from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from src.application.decision_receipt import read_decision_receipt
from src.application.report_formatting import format_local, number, pct, score_to_pl_label
from src.application.report_models import (
    ResolvedPrediction,
    RiskSignal,
    SymbolResult,
    TradeSignal,
)
from src.domain.calibration_curve import (
    CalibrationBucket,
    calibrated_confidence,
    find_bucket,
)
from src.domain.market_calendar import (
    TradingSession,
    holiday_name,
    next_open,
    session_on,
    to_nyse_time,
    upcoming_session,
)
from src.domain.mistake_taxonomy import parse_mistake_category
from src.domain.sizing import SizeBand, suggest_band

DIVERGENCE_PRICE_THRESHOLD = Decimal("0.02")
DIVERGENCE_SENTIMENT_THRESHOLD = 0.2
AV_LLM_CONFLICT_THRESHOLD = 0.3
LOW_SIGNAL_NEWS_THRESHOLD = 3

# Strefa nazw dni i godzin sesji w mailu — ta sama, w której NYSE publikuje kalendarz.
_NYSE_DISPLAY_TZ = "America/New_York"


def build_trade_signals(
    results: list[SymbolResult],
    top_n: int = 5,
    hit_rate: float | None = None,
    calibration_buckets: Sequence[CalibrationBucket] | None = None,
    portfolio_weights: Mapping[str, Decimal] | None = None,
    cluster_exposures: Mapping[str, Decimal] | None = None,
    hit_sample: float | None = None,
    portfolio_var: float | None = None,
    drawdown: float | None = None,
    hedge_effective: bool | None = None,
) -> list[TradeSignal]:
    """Top sygnały transakcyjne — posortowane po sile sygnału.

    Q7: gdy symbol ma werdykt rady (`council_verdict`), doklejamy sugerowaną
    bandę wielkości pozycji (Kelly-lite z konsensusu + dissentu rady i — gdy
    podany — historycznego hit-rate agenta). `hit_rate` jest wspólny dla
    całego cyklu (np. `accuracy_stats["mean_accuracy"]`); None = cold-start →
    sizing konserwatywny. Brak werdyktu rady → `size_band` None.
    PORT-01: `hit_sample` (n_eff próby hit-rate) trafia do `suggest_band` —
    próba poniżej `MIN_HIT_SAMPLE` nie odblokuje pełnej bandy.
    PORT-04: `portfolio_var`, `drawdown` i `hedge_effective` (ryzyko realnego
    portfela, wspólne dla cyklu) idą wprost do `suggest_band` i mogą bandę
    wyłącznie zwęzić. None → banda jak przed PORT-04.

    #9: gdy podano `calibration_buckets`, `strength` liczymy na pewności
    SKALIBROWANEJ historycznym hit-rate'em jej kubełka (shrinkage przy małej
    próbce). Kalibracja przestawia więc RANKING sygnałów, nie tylko etykietę:
    przepewny kubełek spada, uczciwy rośnie. Brak kubełków → funkcja domenowa
    zwraca surowe `raw` → zachowanie identyczne jak przed #9. Render-only, zero
    persystencji: kolumna `confidence_calibration` z migracji 016 ma INNĄ
    semantykę (post-hoc score sędziego LLM) i nie wolno jej nadpisać.

    WIA-13: przy cold-starcie ML (`r.ml_cold_start`) prognozy ruchu nie ma, więc
    `strength` = sama (skalibrowana) pewność. To inna skala niż pewność × ruch,
    dlatego sygnały z prognozą modelu stoją w rankingu przed sygnałami cold-start."""
    signals: list[TradeSignal] = []
    for r in results:
        if r.status != "saved":
            continue
        if r.confidence_score is None or r.expected_change is None:
            continue
        calibrated = (
            calibrated_confidence(r.confidence_score, calibration_buckets)
            if calibration_buckets
            else None
        )
        bucket_count = _calibration_bucket_count(r.confidence_score, calibration_buckets)
        effective = r.confidence_score if calibrated is None else calibrated
        strength = effective if r.ml_cold_start else effective * float(abs(r.expected_change)) * 100
        signals.append(TradeSignal(
            symbol=r.symbol,
            direction=_trend_to_direction(r.trend),
            confidence=r.confidence_score,
            expected_change=r.expected_change,
            strength=strength,
            current_price=r.current_price,
            target_price=r.target_price,
            size_band=_size_band_for(
                r,
                hit_rate,
                portfolio_weights,
                cluster_exposures,
                hit_sample,
                portfolio_var=portfolio_var,
                drawdown=drawdown,
                hedge_effective=hedge_effective,
            ),
            calibrated_confidence=calibrated,
            ml_cold_start=r.ml_cold_start,
            calibration_bucket_count=bucket_count,
        ))
    signals.sort(key=lambda s: (not s.ml_cold_start, s.strength), reverse=True)
    return signals[:top_n]


def _calibration_bucket_count(
    confidence: float, buckets: Sequence[CalibrationBucket] | None
) -> int | None:
    """n kubełka, którym kalibrujemy pewność; 0 = brak pasującego kubełka."""
    if not buckets:
        return None
    bucket = find_bucket(confidence, buckets)
    return 0 if bucket is None else bucket.count


def _size_band_for(
    r: SymbolResult,
    hit_rate: float | None,
    weights: Mapping[str, Decimal] | None = None,
    cluster_exposures: Mapping[str, Decimal] | None = None,
    hit_sample: float | None = None,
    *,
    portfolio_var: float | None = None,
    drawdown: float | None = None,
    hedge_effective: bool | None = None,
) -> SizeBand | None:
    """Wylicza bandę pozycji z werdyktu rady symbolu; None gdy brak werdyktu.

    #15 faza 2: gdy znamy realne wagi portfela, banda uwzględnia istniejącą
    ekspozycję (symbolu i jego klastra korelacji) i może się wyłącznie zwęzić.
    Brak wag → zachowanie sprzed fazy 2.
    """
    verdict = r.council_verdict
    if verdict is None:
        return None
    weight = weights.get(r.symbol) if weights else None
    cluster = cluster_exposures.get(r.symbol) if cluster_exposures else None
    return suggest_band(
        consensus_strength=verdict.consensus_strength,
        dissent_ratio=verdict.dissent_ratio(),
        hit_rate=hit_rate,
        current_weight=float(weight) if weight is not None else None,
        cluster_exposure=float(cluster) if cluster is not None else None,
        hit_sample=hit_sample,
        portfolio_var=portfolio_var,
        drawdown=drawdown,
        hedge_effective=hedge_effective,
    )


def divergence_direction(delta: Decimal | float | None, sentiment: float | None) -> int:
    """WIA-47: reguła DIVERGENCE — +1 cena rośnie mimo negatywnego sentymentu,
    -1 spada mimo pozytywnego, 0 bez rozbieżności. Brak danych też 0 (wołający
    rozróżnia go sam). Ta sama reguła dla maila i `evaluate --by risk_signal`."""
    if delta is None or sentiment is None:
        return 0
    move = Decimal(str(delta))
    if move > DIVERGENCE_PRICE_THRESHOLD and sentiment < -DIVERGENCE_SENTIMENT_THRESHOLD:
        return 1
    if move < -DIVERGENCE_PRICE_THRESHOLD and sentiment > DIVERGENCE_SENTIMENT_THRESHOLD:
        return -1
    return 0


def is_av_llm_conflict(agreement: float | None) -> bool:
    """WIA-47: reguła AV_LLM_CONFLICT — zgoda LLM z Alpha Vantage poniżej progu."""
    return agreement is not None and agreement < AV_LLM_CONFLICT_THRESHOLD


def is_low_signal(news_volume: int | None) -> bool:
    """WIA-47: reguła LOW_SIGNAL — decyzja na mniej niż progu relevantnych newsów."""
    return news_volume is not None and news_volume < LOW_SIGNAL_NEWS_THRESHOLD


def _divergence_signal(r: SymbolResult) -> RiskSignal | None:
    """Rozbieżność cena↔sentyment — możliwy pump albo niedoszacowane odbicie."""
    if r.delta is None or r.sentiment_score is None:
        return None
    direction = divergence_direction(r.delta, r.sentiment_score)
    if direction > 0:
        return RiskSignal(
            symbol=r.symbol, type="DIVERGENCE", severity="high",
            description=(
                f"cena rośnie {pct(r.delta)} ale sentyment "
                f"{number(r.sentiment_score, decimals=2, signed=True)} -> możliwa korekta lub pump"
            ),
        )
    if direction < 0:
        return RiskSignal(
            symbol=r.symbol, type="DIVERGENCE", severity="high",
            description=(
                f"cena spada {pct(r.delta)} ale sentyment "
                f"{number(r.sentiment_score, decimals=2, signed=True)} -> okazja lub odbicie?"
            ),
        )
    return None


def _av_llm_conflict_signal(r: SymbolResult) -> RiskSignal | None:
    """LLM kontra Alpha Vantage — rozbieżne odczyty sentymentu."""
    if r.av_llm_agreement is None or not is_av_llm_conflict(r.av_llm_agreement):
        return None
    return RiskSignal(
        symbol=r.symbol, type="AV_LLM_CONFLICT", severity="high",
        description=(
            f"LLM nie zgadza się z AV (zgoda={number(r.av_llm_agreement, decimals=2)})"
            " -> rozbieżne odczyty sentymentu"
        ),
    )


def _low_signal_signal(r: SymbolResult) -> RiskSignal | None:
    """Decyzja podjęta na zbyt małej liczbie relevantnych newsów."""
    if not is_low_signal(r.news_volume):
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
            predicted_at=_to_datetime(row.get("timestamp")),
            receipt=read_decision_receipt(row.get("decision_receipts")),
            correction_category=parse_mistake_category(row.get("correction_category")),
        ))
    return out


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_datetime(value: Any) -> datetime | None:
    """ISO-8601 z PostgREST → datetime; brak albo śmieć → None (bez wyjątku)."""
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


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
    """Status sesji NYSE z kalendarzem świąt i skróconych sesji (ALR-15).

    Święto i skrócona sesja są nazwane wprost w etykiecie, a „najbliższe otwarcie”
    pomija weekend i święta. Naiwny `datetime` = UTC."""
    nyc = to_nyse_time(now_utc)
    session = session_on(nyc.date())
    if session is None:
        holiday = holiday_name(nyc.date())
        label = f"📅 Zamknięta (święto: {holiday})" if holiday else "🌙 Zamknięta (weekend)"
        return {"label": label, "detail": f"Najbliższe otwarcie: {_format_et(next_open(nyc))}"}
    early_close = _early_close_note(session)
    if nyc < session.opens_at:
        return {
            "label": _with_note("🌅 Premarket", early_close),
            "detail": f"Otwarcie za {_humanize_delta(session.opens_at - nyc)}",
        }
    if nyc < session.closes_at:
        return {
            "label": f"🟢 Otwarta ({early_close or 'regular hours'})",
            "detail": f"Zamknięcie za {_humanize_delta(session.closes_at - nyc)}",
        }
    opening = next_open(nyc)
    detail = (
        "Otwarcie jutro 09:30 ET"
        if opening.date() == nyc.date() + timedelta(days=1)
        else f"Najbliższe otwarcie: {_format_et(opening)}"
    )
    return {
        "label": _with_note("🌆 After-hours", early_close),
        "detail": detail,
    }


def forecast_horizon_label(now_utc: datetime) -> str:
    """Najbliższe zamknięcie sesji NYSE po starcie cyklu (ALR-15), np.
    „najbliższe zamknięcie NYSE: pt 27.11 13:00 ET (skrócona sesja)”.

    Następny cykl rozlicza prognozę bieżącą ceną, więc przy dziennej kadencji przed
    otwarciem prognoza obejmuje właśnie tę sesję; święta i weekendy są pominięte."""
    session = upcoming_session(now_utc)
    suffix = " (skrócona sesja)" if session.is_early_close else ""
    return f"najbliższe zamknięcie NYSE: {_format_et(session.closes_at)}{suffix}"


def _early_close_note(session: TradingSession) -> str | None:
    if not session.is_early_close:
        return None
    return f"skrócona sesja do {session.closes_at:%H:%M} ET"


def _with_note(label: str, note: str | None) -> str:
    return f"{label} ({note})" if note else label


def _format_et(moment: datetime) -> str:
    """„pt 27.11 13:00 ET” — dzień tygodnia po polsku, niezależnie od locale."""
    return f"{format_local(moment, _NYSE_DISPLAY_TZ)} ET"


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
