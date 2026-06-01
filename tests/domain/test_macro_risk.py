"""Testy domeny Risk Watch — instrumenty proxy ryzyka makro.

Reguła kierunku alertu zależy od typu instrumentu:
- INVERSE_EQUITY / INVERSE_TREASURY / VOLATILITY / SAFE_HAVEN → wzrost ceny = alert
- SOVEREIGN_PROXY (np. EPOL) → spadek ceny = alert (wycofują się z PL equity)
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.macro_risk import (
    MacroAlertLevel,
    MacroRiskInstrumentType,
    MacroRiskSignal,
)
from src.domain.value_objects import Money


def _signal(
    instrument_type: MacroRiskInstrumentType,
    previous: str,
    current: str,
) -> MacroRiskSignal:
    return MacroRiskSignal(
        symbol="X",
        instrument_type=instrument_type,
        current_price=Money(Decimal(current)),
        previous_price=Money(Decimal(previous)),
    )


class TestChangePct:
    def test_change_pct_positive_when_price_rises(self) -> None:
        sig = _signal(MacroRiskInstrumentType.VOLATILITY, "100", "105")
        assert sig.change_pct == Decimal("0.05")

    def test_change_pct_negative_when_price_falls(self) -> None:
        sig = _signal(MacroRiskInstrumentType.SOVEREIGN_PROXY, "100", "97")
        assert sig.change_pct == Decimal("-0.03")

    def test_change_pct_zero_when_previous_is_zero_cold_start(self) -> None:
        sig = _signal(MacroRiskInstrumentType.VOLATILITY, "0", "10")
        # Cold-start guard — bez referencji nie można liczyć zmiany.
        assert sig.change_pct == Decimal("0")


class TestInverseAndSafeHavenAlerts:
    """Wzrost ceny = sygnał risk-off / spadek rynku bazowego."""

    @pytest.mark.parametrize(
        "instrument",
        [
            MacroRiskInstrumentType.INVERSE_EQUITY,
            MacroRiskInstrumentType.INVERSE_TREASURY,
            MacroRiskInstrumentType.VOLATILITY,
            MacroRiskInstrumentType.SAFE_HAVEN,
        ],
    )
    def test_small_move_is_normal(
        self, instrument: MacroRiskInstrumentType
    ) -> None:
        sig = _signal(instrument, "100", "100.5")  # +0.5%
        assert sig.evaluate_alert() is MacroAlertLevel.NORMAL

    @pytest.mark.parametrize(
        "instrument",
        [
            MacroRiskInstrumentType.INVERSE_EQUITY,
            MacroRiskInstrumentType.VOLATILITY,
            MacroRiskInstrumentType.SAFE_HAVEN,
        ],
    )
    def test_moderate_rise_is_elevated(
        self, instrument: MacroRiskInstrumentType
    ) -> None:
        sig = _signal(instrument, "100", "102")  # +2%
        assert sig.evaluate_alert() is MacroAlertLevel.ELEVATED

    @pytest.mark.parametrize(
        "instrument",
        [
            MacroRiskInstrumentType.INVERSE_EQUITY,
            MacroRiskInstrumentType.VOLATILITY,
        ],
    )
    def test_large_rise_is_critical(
        self, instrument: MacroRiskInstrumentType
    ) -> None:
        sig = _signal(instrument, "100", "104")  # +4%
        assert sig.evaluate_alert() is MacroAlertLevel.CRITICAL

    def test_inverse_falling_is_not_alert(self) -> None:
        # Inverse ETF spada → rynek bazowy rośnie → dobrze, nie alarm.
        sig = _signal(MacroRiskInstrumentType.INVERSE_EQUITY, "100", "95")
        assert sig.evaluate_alert() is MacroAlertLevel.NORMAL


class TestSovereignProxyAlerts:
    """Spadek ceny = inwestorzy wychodzą z polskiego equity."""

    def test_rising_epol_is_normal(self) -> None:
        sig = _signal(MacroRiskInstrumentType.SOVEREIGN_PROXY, "20", "20.5")
        assert sig.evaluate_alert() is MacroAlertLevel.NORMAL

    def test_moderate_fall_is_elevated(self) -> None:
        sig = _signal(MacroRiskInstrumentType.SOVEREIGN_PROXY, "20", "19.6")
        # -2% → ELEVATED
        assert sig.evaluate_alert() is MacroAlertLevel.ELEVATED

    def test_large_fall_is_critical(self) -> None:
        sig = _signal(MacroRiskInstrumentType.SOVEREIGN_PROXY, "20", "19.2")
        # -4% → CRITICAL
        assert sig.evaluate_alert() is MacroAlertLevel.CRITICAL


class TestCustomThresholds:
    def test_custom_thresholds_override_defaults(self) -> None:
        sig = _signal(MacroRiskInstrumentType.VOLATILITY, "100", "100.6")
        assert (
            sig.evaluate_alert(
                elevated_threshold=Decimal("0.005"),
                critical_threshold=Decimal("0.02"),
            )
            is MacroAlertLevel.ELEVATED
        )
