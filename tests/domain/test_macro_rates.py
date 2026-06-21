"""Testy domeny krzywej rentowności — spread 10Y-2Y i klasyfikacja stanu.

Czysta logika bez I/O: sprawdzamy obliczenie spreadu, progi
INVERTED/FLAT/NORMAL oraz graceful obsługę braków (None)."""

from __future__ import annotations

from src.domain.macro_rates import YieldCurveSnapshot, YieldCurveState


class TestSpread10y2y:
    def test_computes_difference(self) -> None:
        snap = YieldCurveSnapshot(ten_year=4.25, two_year=4.80, fed_funds=5.33)
        # 4.25 - 4.80 == -0.55 (z tolerancją na błąd float)
        assert snap.spread_10y_2y is not None
        assert round(snap.spread_10y_2y, 2) == -0.55

    def test_positive_spread(self) -> None:
        snap = YieldCurveSnapshot(ten_year=4.50, two_year=4.00)
        assert snap.spread_10y_2y is not None
        assert round(snap.spread_10y_2y, 2) == 0.50

    def test_none_when_ten_year_missing(self) -> None:
        snap = YieldCurveSnapshot(ten_year=None, two_year=4.00)
        assert snap.spread_10y_2y is None

    def test_none_when_two_year_missing(self) -> None:
        snap = YieldCurveSnapshot(ten_year=4.00, two_year=None)
        assert snap.spread_10y_2y is None


class TestState:
    def test_inverted_when_spread_negative(self) -> None:
        snap = YieldCurveSnapshot(ten_year=4.25, two_year=4.80)
        assert snap.state() == YieldCurveState.INVERTED

    def test_flat_when_spread_small_positive(self) -> None:
        # spread 0.20 < flat_below (0.5) → FLAT
        snap = YieldCurveSnapshot(ten_year=4.20, two_year=4.00)
        assert snap.state() == YieldCurveState.FLAT

    def test_normal_when_spread_wide(self) -> None:
        snap = YieldCurveSnapshot(ten_year=5.00, two_year=4.00)
        assert snap.state() == YieldCurveState.NORMAL

    def test_normal_when_data_missing(self) -> None:
        """Brak danych traktowany neutralnie — NORMAL, nie wywala."""
        snap = YieldCurveSnapshot(ten_year=None, two_year=None)
        assert snap.state() == YieldCurveState.NORMAL
