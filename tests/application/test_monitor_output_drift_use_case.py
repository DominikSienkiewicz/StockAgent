"""NIEZ-19: tygodniowy krok Slow Loopa porównujący rozkład wyjść LLM.

Okno bieżące = predykcje z ostatnich 7 dni, poprzednie = 7 dni wcześniej. Skok
bez zmiany przepisu daje WARNING (QuotaAlert → banner maila), zero płatnych wywołań.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from src.application.use_cases.monitor_output_drift import (
    DRIFT_ALERT_SOURCE,
    MonitorOutputDriftUseCase,
    drift_alert,
    output_samples,
)
from src.domain.quota import QuotaSeverity

NOW = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)


def _rows(
    days_ago: float, bullish: int, other: int, *, ups: int, recipe: str | None = None
) -> list[dict[str, Any]]:
    stamp = (NOW - timedelta(days=days_ago)).isoformat()
    rows: list[dict[str, Any]] = []
    for index in range(bullish + other):
        row: dict[str, Any] = {
            "predicted_trend": "BULLISH" if index < bullish else "BEARISH",
            "confidence_score": 0.6,
            "price_at_prediction": 100.0,
            "actual_price_after_12h": 101.0 if index < ups else 99.0,
            "timestamp": stamp,
        }
        if recipe is not None:
            row["recipe_version"] = recipe
        rows.append(row)
    return rows


class _Reader:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.days: list[int] = []

    def __call__(self, days: int) -> list[dict[str, Any]]:
        self.days.append(days)
        return self.rows


def test_jump_in_bullish_share_raises_a_warning_alert() -> None:
    reader = _Reader(_rows(10, 30, 70, ups=50) + _rows(2, 80, 20, ups=50))

    report = MonitorOutputDriftUseCase(reader).run(now=NOW)
    alert = drift_alert(report, NOW)

    assert reader.days == [14]
    assert report.previous.n == 100 and report.current.n == 100
    assert alert is not None
    assert alert.source == DRIFT_ALERT_SOURCE
    assert alert.severity is QuotaSeverity.WARNING
    assert "bez zmiany przepisu" in alert.message


def test_stable_weeks_raise_no_alert() -> None:
    reader = _Reader(_rows(10, 40, 60, ups=50) + _rows(2, 40, 60, ups=50))

    report = MonitorOutputDriftUseCase(reader).run(now=NOW)

    assert not report.flagged
    assert drift_alert(report, NOW) is None


def test_alert_after_recipe_change_names_it() -> None:
    reader = _Reader(
        _rows(10, 30, 70, ups=50, recipe="aaa111") + _rows(2, 80, 20, ups=50, recipe="bbb222")
    )

    alert = drift_alert(MonitorOutputDriftUseCase(reader).run(now=NOW), NOW)

    assert alert is not None
    assert "zmiana zachowania po zmianie przepisu bbb222" in alert.message


def test_rows_without_prices_or_trend_are_handled() -> None:
    samples = output_samples(
        [
            {"predicted_trend": "SIDEWAYS", "confidence_score": None, "price_at_prediction": None},
            {"predicted_trend": None, "confidence_score": 0.5},
            {
                "predicted_trend": "BULLISH",
                "confidence_score": "0.7",
                "price_at_prediction": "100",
                "actual_price_after_12h": "100",
            },
        ]
    )

    assert [s.trend for s in samples] == ["SIDEWAYS", "BULLISH"]
    assert samples[0].realized_up is None
    # Cena bez zmiany to nie wzrost.
    assert samples[1].realized_up is False
    assert samples[1].confidence == 0.7
