"""Testy domeny polskiego makro — proxy ryzyka suwerennego przez kurs PLN.

Założenie: rosnący EUR/PLN i USD/PLN (PLN się osłabia) = sygnał stresu
fiskalnego — inwestorzy zagraniczni redukują pozycje, presja na rating.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.domain.polish_macro import MacroStressLevel, PolishMacroSnapshot
from src.domain.value_objects import Ratio


def _snap(
    eur_change: str,
    usd_change: str,
    eur_pln: str = "4.30",
    usd_pln: str = "4.00",
) -> PolishMacroSnapshot:
    return PolishMacroSnapshot(
        eur_pln=Decimal(eur_pln),
        usd_pln=Decimal(usd_pln),
        eur_pln_30d_change_ratio=Ratio(Decimal(eur_change)),
        usd_pln_30d_change_ratio=Ratio(Decimal(usd_change)),
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
        elevated = Decimal("-0.01")
        critical = Decimal("0.03")
        with pytest.raises(ValueError):
            snap.evaluate_stress_level(elevated_threshold=elevated, critical_threshold=critical)

    def test_elevated_must_be_below_critical(self) -> None:
        snap = _snap(eur_change="0.0", usd_change="0.0")
        elevated = Decimal("0.05")
        critical = Decimal("0.02")
        with pytest.raises(ValueError):
            snap.evaluate_stress_level(elevated_threshold=elevated, critical_threshold=critical)


def _rates(*mids: str) -> list[Decimal]:
    return [Decimal(mid) for mid in mids]


class TestFromRates:
    """DAT-29: zmiana 30d i dzienna z notowań tabeli A (najstarsze → najświeższe)."""

    def test_snapshot_exposes_daily_change(self) -> None:
        snap = PolishMacroSnapshot.from_rates(
            eur=_rates("4.30", "4.25", "4.27"),
            usd=_rates("3.70", "3.64", "3.62"),
            fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
        )

        assert snap is not None
        # (3.62 - 3.64) / 3.64 ≈ −0,55% d/d
        assert snap.usd_pln_1d_change_ratio is not None
        assert float(snap.usd_pln_1d_change_ratio) == pytest.approx(-0.0055, abs=1e-4)
        # (4.27 - 4.25) / 4.25 ≈ +0,47% d/d
        assert snap.eur_pln_1d_change_ratio is not None
        assert float(snap.eur_pln_1d_change_ratio) == pytest.approx(0.0047, abs=1e-4)

    def test_month_change_is_from_the_oldest_rate(self) -> None:
        snap = PolishMacroSnapshot.from_rates(
            eur=_rates("4.30", "4.32", "4.35", "4.40"),
            usd=_rates("4.00", "3.95", "4.05", "4.10"),
            fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
        )

        assert snap is not None
        assert snap.eur_pln == Decimal("4.40")
        assert snap.usd_pln == Decimal("4.10")
        assert snap.usd_pln_30d_change_ratio == Decimal("0.025")

    @pytest.mark.parametrize(
        ("eur", "usd"),
        [
            (_rates("4.30"), _rates("3.70", "3.62")),
            (_rates("4.30", "4.27"), _rates()),
            (_rates("0", "4.27"), _rates("3.70", "3.62")),
            (_rates("4.30", "4.27"), _rates("3.70", "-1")),
        ],
        ids=["one_rate", "no_rates", "zero_baseline", "negative_previous"],
    )
    def test_without_two_positive_rates_there_is_no_snapshot(
        self, eur: list[Decimal], usd: list[Decimal]
    ) -> None:
        snap = PolishMacroSnapshot.from_rates(
            eur=eur, usd=usd, fetched_at=datetime(2026, 6, 1, tzinfo=UTC)
        )

        assert snap is None

    def test_daily_change_defaults_to_unknown(self) -> None:
        # Snapshot zbudowany wprost (fixture'y, sample mail) nie zna zmiany dziennej.
        assert _snap(eur_change="0.01", usd_change="0.01").usd_pln_1d_change_ratio is None
