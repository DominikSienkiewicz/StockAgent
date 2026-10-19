"""ML-34 — rozliczone predykcje → okna kalibracyjne przedziału ruchu i pokrycie.

Zero I/O: wejście to wiersze `RepositoryPort.get_settled_moves` (rosnąco po
czasie), wyjście to obiekty domeny `src.domain.conformal`. Okno kalibracyjne
to ROZLICZONE predykcje, nie wszystkie snapshoty ceny: bieżący przedział
dostaje tylko cykl za bramką volatility, więc kalibrujemy na tej samej selekcji
(dni z dużym ruchem), a nie na spokojnych dniach, które by go zawęziły.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from src.domain.conformal import (
    CalibrationPoint,
    IntervalCoverage,
    realized_coverage,
)
from src.domain.value_objects import AssetType

# Okno odczytu rozliczonych predykcji (dni). `move_interval` i tak bierze
# najnowsze `MAX_CALIBRATION_SIZE` punktów klasy aktywa.
CALIBRATION_LOOKBACK_DAYS = 90


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _calibration_point(row: Mapping[str, Any]) -> CalibrationPoint | None:
    """Zwrot przewidziany i zrealizowany względem ceny predykcji. Brak ceny
    docelowej = predyktor 0 (tak zapisywał cold-start)."""
    price = _number(row.get("price_at_prediction"))
    actual = _number(row.get("actual_price_after_12h"))
    if price is None or price <= 0 or actual is None:
        return None
    target = _number(row.get("predicted_target_price"))
    predicted = 0.0 if target is None else target / price - 1
    return CalibrationPoint(predicted_return=predicted, realized_return=actual / price - 1)


def calibration_windows(
    rows: Iterable[Any], asset_type_of: Callable[[str], AssetType]
) -> dict[AssetType, tuple[CalibrationPoint, ...]]:
    """Okna kalibracyjne per klasa aktywa, w kolejności wejścia (rosnąco po
    czasie). Wiersze bez symbolu, ceny albo rozliczenia są pomijane."""
    windows: dict[AssetType, list[CalibrationPoint]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        symbol = row.get("symbol")
        point = _calibration_point(row)
        if not isinstance(symbol, str) or point is None:
            continue
        windows.setdefault(asset_type_of(symbol), []).append(point)
    return {asset_type: tuple(points) for asset_type, points in windows.items()}


def interval_coverage(rows: Iterable[Any]) -> IntervalCoverage:
    """Pokrycie zrealizowane: rozliczona cena wewnątrz zapisanego przedziału.
    Wiersze bez przedziału (sprzed migracji 032 albo bez flagi) nie wchodzą."""
    outcomes: list[tuple[float, float, float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        low = _number(row.get("interval_low"))
        high = _number(row.get("interval_high"))
        actual = _number(row.get("actual_price_after_12h"))
        if low is None or high is None or actual is None:
            continue
        outcomes.append((low, high, actual))
    return realized_coverage(outcomes)
