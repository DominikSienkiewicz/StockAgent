"""NIEZ-19: tygodniowy monitor rozkładu wyjść LLM (krok Slow Loopa).

Czyta rozliczone predykcje z dwóch ostatnich tygodni, dzieli je po dacie
utworzenia na okno poprzednie i bieżące, a porównanie robi domena
(`src/domain/output_drift.py`). Zero płatnych wywołań: sam odczyt bazy.

Skok udziałów (z biasem) albo pewności daje WARNING — `QuotaAlert`, który Slow
Loop zapisuje w `quota_alerts`, więc trafia do bannera najbliższego maila.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from src.domain.output_drift import OutputDriftReport, OutputSample, output_distribution_shift
from src.domain.quota import QuotaAlert, QuotaSeverity

WINDOW_DAYS = 7
DRIFT_ALERT_SOURCE = "Monitor wyjść LLM"
DRIFT_ALERT_ACTION = (
    "Sprawdź, czy dostawca nie podmienił modelu i czy prompt albo persony się nie zmieniły; "
    "porównaj trafność przepisów: `main_tools.py evaluate --by-recipe`."
)

RowsReader = Callable[[int], Sequence[Mapping[str, Any]]]


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _realized_up(row: Mapping[str, Any]) -> bool | None:
    before = _number(row.get("price_at_prediction"))
    after = _number(row.get("actual_price_after_12h"))
    if before is None or after is None:
        return None
    return after > before


def output_samples(rows: Sequence[Mapping[str, Any]]) -> list[OutputSample]:
    """Wiersze `prediction_logs` jako próbki domeny; wiersz bez trendu odpada."""
    samples: list[OutputSample] = []
    for row in rows:
        trend = row.get("predicted_trend")
        if not isinstance(trend, str) or not trend:
            continue
        recipe = row.get("recipe_version")
        samples.append(
            OutputSample(
                trend=trend,
                confidence=_number(row.get("confidence_score")),
                realized_up=_realized_up(row),
                recipe_version=recipe if isinstance(recipe, str) and recipe else None,
            )
        )
    return samples


def _created_at(row: Mapping[str, Any]) -> datetime | None:
    stamp = row.get("timestamp")
    if not isinstance(stamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class MonitorOutputDriftUseCase:
    """`read_rows(days)` — rozliczone predykcje z ostatnich `days` dni
    (`predicted_trend`, `confidence_score`, ceny, `timestamp`, opcjonalnie
    `recipe_version`); składa go DI Slow Loopa."""

    def __init__(self, read_rows: RowsReader) -> None:
        self._read_rows = read_rows

    def run(self, now: datetime) -> OutputDriftReport:
        cutoff = now - timedelta(days=WINDOW_DAYS)
        previous: list[Mapping[str, Any]] = []
        current: list[Mapping[str, Any]] = []
        for row in self._read_rows(2 * WINDOW_DAYS):
            created = _created_at(row)
            if created is None:
                continue
            (current if created >= cutoff else previous).append(row)
        return output_distribution_shift(output_samples(previous), output_samples(current))


def drift_alert(report: OutputDriftReport, now: datetime) -> QuotaAlert | None:
    """WARNING dla bannera maila, gdy monitor flaguje; None, gdy rozkład stabilny."""
    if not report.flagged:
        return None
    return QuotaAlert(
        source=DRIFT_ALERT_SOURCE,
        severity=QuotaSeverity.WARNING,
        message=report.describe_pl(),
        action=DRIFT_ALERT_ACTION,
        occurred_at=now,
    )
