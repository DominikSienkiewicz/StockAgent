"""Testy domeny polskiego makro — proxy ryzyka suwerennego przez kurs PLN.

Założenie: rosnący EUR/PLN i USD/PLN (PLN się osłabia) = sygnał stresu
fiskalnego — inwestorzy zagraniczni redukują pozycje, presja na rating.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.domain.polish_macro import MacroStressLevel, PolishMacroSnapshot


def _snap(
    eur_change: str,
    usd_change: str,
    eur_pln: str = "4.30",
    usd_pln: str = "4.00",
) -> PolishMacroSnapshot:
    return PolishMacroSnapshot(
        eur_pln=Decimal(eur_pln),
        usd_pln=Decimal(usd_pln),
        eur_pln_30d_change_pct=Decimal(eur_change),
        usd_pln_30d_change_pct=Decimal(usd_change),
        fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
    )


class TestStressLevel:
    def test_stable_currencies_are_normal(self) -> None:
        snap = _snap(eur_change="0.005", usd_change="-0.002")  # ±0.5% / 0.2%
        assert snap.evaluate_stress_level() is MacroStressLevel.NORMAL

    def test_pln_strengthening_is_normal(self) -> None:
        # Spadek EUR/PLN = PLN się umacnia, brak stresu.
        snap = _snap(eur_change="-0.03", usd_change="-0.04")
        assert snap.evaluate_stress_level() is MacroStressLevel.NORMAL

    def test_moderate_pln_weakening_is_elevated(self) -> None:
        # +2.5% EUR/PLN w 30 dni = PLN osłabia się umiarkowanie.
        snap = _snap(eur_change="0.025", usd_change="0.01")
        assert snap.evaluate_stress_level() is MacroStressLevel.ELEVATED

    def test_strong_pln_weakening_is_critical(self) -> None:
        # +6% EUR/PLN w 30 dni = poważny sell-off PLN.
        snap = _snap(eur_change="0.06", usd_change="0.04")
        assert snap.evaluate_stress_level() is MacroStressLevel.CRITICAL

    def test_max_of_two_pairs_drives_alert(self) -> None:
        # Tylko USD/PLN poszedł mocno w górę — i tak alert critical.
        snap = _snap(eur_change="0.005", usd_change="0.07")
        assert snap.evaluate_stress_level() is MacroStressLevel.CRITICAL


class TestCustomThresholds:
    def test_custom_thresholds_override_defaults(self) -> None:
        snap = _snap(eur_change="0.012", usd_change="0.0")
        assert (
            snap.evaluate_stress_level(
                elevated_threshold=Decimal("0.01"),
                critical_threshold=Decimal("0.03"),
            )
            is MacroStressLevel.ELEVATED
        )


class TestInvariants:
    def test_negative_thresholds_are_rejected(self) -> None:
        snap = _snap(eur_change="0.0", usd_change="0.0")
        with pytest.raises(ValueError):
            snap.evaluate_stress_level(
                elevated_threshold=Decimal("-0.01"),
                critical_threshold=Decimal("0.03"),
            )

    def test_elevated_must_be_below_critical(self) -> None:
        snap = _snap(eur_change="0.0", usd_change="0.0")
        with pytest.raises(ValueError):
            snap.evaluate_stress_level(
                elevated_threshold=Decimal("0.05"),
                critical_threshold=Decimal("0.02"),
            )
