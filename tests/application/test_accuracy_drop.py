"""NIEZ-20: detektor spadku trafności w Track Record i w bannerze maila.

Trafność per cykl z rozliczonych predykcji idzie do Page-Hinkleya; przy alarmie
Track Record mówi „trafność spadła istotnie od <data>”, a banner dostaje WARNING.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.application.report_evidence import (
    ACCURACY_DROP_ALERT_SOURCE,
    accuracy_drop_alert,
    accuracy_drop_evidence,
)
from src.application.report_models import ResolvedPrediction
from src.application.report_track_record import TrackRecordStats, render_track_record_html
from src.domain.change_detection import AccuracyDrop
from src.domain.equity_curve import EquityCurve, EquityPoint
from src.domain.quota import QuotaSeverity

START = datetime(2026, 6, 1, 5, 35, tzinfo=UTC)


def _cycle(day: int, hits: int, misses: int) -> list[ResolvedPrediction]:
    return [
        ResolvedPrediction(
            symbol=f"S{index}",
            predicted_trend="BULLISH",
            is_correct=index < hits,
            price_at_prediction=Decimal("100"),
            actual_price=Decimal("101") if index < hits else Decimal("99"),
            predicted_at=START + timedelta(days=day),
        )
        for index in range(hits + misses)
    ]


def _degrading_record() -> list[ResolvedPrediction]:
    resolved: list[ResolvedPrediction] = []
    for day in range(30):
        resolved += _cycle(day, 6, 4)
    for day in range(30, 70):
        resolved += _cycle(day, 4, 6)
    return resolved


def _drop() -> AccuracyDrop:
    return AccuracyDrop(
        since="2026-07-01", alarm_cycle="2026-07-20", before_rate=0.6, after_rate=0.4, cycles=70
    )


def _curve() -> EquityCurve:
    return EquityCurve(
        points=(EquityPoint(index=0, period_return=0.01, equity=1.01),),
        total_return=0.01,
        win_rate=0.5,
        max_drawdown=0.0,
        n_trades=1,
    )


def test_degrading_record_yields_the_date_of_the_drop() -> None:
    drop = accuracy_drop_evidence(_degrading_record())

    assert drop is not None
    assert drop.since == "2026-07-01"


def test_stable_or_empty_record_yields_nothing() -> None:
    stable = [p for day in range(60) for p in _cycle(day, 5, 5)]

    assert accuracy_drop_evidence(stable) is None
    assert accuracy_drop_evidence([]) is None


def test_track_record_says_accuracy_dropped_since_the_date() -> None:
    html = render_track_record_html(_curve(), None, None, TrackRecordStats(accuracy_drop=_drop()))

    assert "trafność spadła istotnie od 2026-07-01" in html


def test_compact_track_record_keeps_the_drop_line() -> None:
    html = render_track_record_html(
        _curve(), None, None, TrackRecordStats(accuracy_drop=_drop()), compact=True
    )

    assert "trafność spadła istotnie od 2026-07-01" in html


def test_without_a_drop_the_track_record_has_no_drop_line() -> None:
    html = render_track_record_html(_curve(), None, None, TrackRecordStats())

    assert "spadła istotnie" not in html


def test_drop_becomes_a_warning_for_the_banner() -> None:
    now = datetime(2026, 7, 21, tzinfo=UTC)

    alert = accuracy_drop_alert(_drop(), now)

    assert alert.source == ACCURACY_DROP_ALERT_SOURCE
    assert alert.severity is QuotaSeverity.WARNING
    assert "2026-07-01" in alert.message
    assert alert.occurred_at == now
