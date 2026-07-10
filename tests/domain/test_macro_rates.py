"""Testy domeny krzywej rentowności — spread 10Y-2Y i klasyfikacja stanu.

Czysta logika bez I/O: sprawdzamy obliczenie spreadu, progi
INVERTED/FLAT/NORMAL oraz graceful obsługę braków (None)."""

from __future__ import annotations

import pytest

from src.domain.macro_rates import YieldCurveSnapshot, YieldCurveState


class TestSpread10y2y:
    def test_computes_difference(self) -> None:
        snap = YieldCurveSnapshot(ten_year=4.25, two_year=4.80, fed_funds=5.33)
        # 4.25 - 4.80 == -0.55 (z tolerancją na błąd float)
        assert snap.spread_10y_2y is not None
        assert round(snap.spread_10y_2y, 2) == pytest.approx(-0.55)

    def test_positive_spread(self) -> None:
        snap = YieldCurveSnapshot(ten_year=4.50, two_year=4.00)
        assert snap.spread_10y_2y is not None
        assert round(snap.spread_10y_2y, 2) == pytest.approx(0.50)

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


class TestYieldCurveAlertLevel:
    def test_inverted_maps_to_elevated(self) -> None:
        # Inwersja 10Y-2Y to klasyczny sygnał recesyjny → ELEVATED.
        from src.domain.macro_rates import yield_curve_alert_level
        from src.domain.macro_risk import MacroAlertLevel

        assert (
            yield_curve_alert_level(YieldCurveState.INVERTED)
            is MacroAlertLevel.ELEVATED
        )

    def test_flat_and_normal_map_to_normal(self) -> None:
        # Zdrowa/płaska krzywa nie jest sygnałem ryzyka → poziom neutralny.
        from src.domain.macro_rates import yield_curve_alert_level
        from src.domain.macro_risk import MacroAlertLevel

        assert (
            yield_curve_alert_level(YieldCurveState.FLAT)
            is MacroAlertLevel.NORMAL
        )
        assert (
            yield_curve_alert_level(YieldCurveState.NORMAL)
            is MacroAlertLevel.NORMAL
        )

    def test_covers_every_state(self) -> None:
        # Każdy wariant enuma ma zdefiniowane mapowanie (brak KeyError).
        from src.domain.macro_rates import yield_curve_alert_level
        from src.domain.macro_risk import MacroAlertLevel

        for state in YieldCurveState:
            assert isinstance(yield_curve_alert_level(state), MacroAlertLevel)

    def test_inversion_alone_is_not_risk_off(self) -> None:
        """Semantyka GŁOS-nie-WETO: sama inwersja zwraca JEDEN poziom alertu
        (ELEVATED), a NIE decyzję o reżimie.

        Reguła orkiestratora brzmi ">= 2 głosy ELEVATED → RISK_OFF". Funkcja
        dostarcza tylko pojedynczy głos, więc inwersja bez innych sygnałów
        ELEVATED przekłada się na NEUTRAL, nie RISK_OFF. To celowy tie-breaker
        mitygujący wielomiesięczne inwersje krzywej, które inaczej trzymałyby
        agenta w risk-off przez miesiące."""
        from src.domain.macro_rates import yield_curve_alert_level
        from src.domain.regime import MarketRegime, RegimeDetector

        vote = yield_curve_alert_level(YieldCurveState.INVERTED)
        detector = RegimeDetector()

        # Pojedynczy głos ELEVATED z krzywej, zero innych sygnałów.
        regime = detector.classify(macro_alert=vote, risk_signals=[])

        assert regime is MarketRegime.NEUTRAL
        assert regime is not MarketRegime.RISK_OFF
