"""Testy domeny Drawdown — śledzenie spadku od szczytu z historii snapshotów.

Drawdown liczony jest WYŁĄCZNIE z darmowych snapshotów ceny (price_snapshots),
bez żadnych płatnych portów. Reguła: szczyt = max z historii, drawdown =
(current - peak) / peak (wartość <= 0). Poziom alertu mapowany jest na ten sam
enum co macro-risk (MacroAlertLevel), żeby raport miał jedną wspólną skalę.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.drawdown import DrawdownSignal, peak_from_history
from src.domain.macro_risk import MacroAlertLevel


def _signal(peak: str, current: str) -> DrawdownSignal:
    return DrawdownSignal(
        symbol="X",
        current_price=Decimal(current),
        peak_price=Decimal(peak),
    )


class TestPeakFromHistory:
    def test_peak_is_max_of_prices(self) -> None:
        prices = [Decimal("100"), Decimal("120"), Decimal("90")]
        assert peak_from_history(prices) == Decimal("120")

    def test_peak_of_single_price(self) -> None:
        assert peak_from_history([Decimal("42")]) == Decimal("42")

    def test_peak_of_empty_history_is_zero(self) -> None:
        # Brak historii — zwracamy 0 zamiast rzucać, by use case mógł
        # gładko pominąć symbol bez snapshotów.
        assert peak_from_history([]) == Decimal("0")


class TestDrawdownFraction:
    def test_drawdown_fraction_below_peak(self) -> None:
        # 100 → szczyt 120 → drawdown = (100 - 120) / 120 = -0.1666...
        sig = _signal(peak="120", current="100")
        assert sig.drawdown_fraction == (
            Decimal("100") - Decimal("120")
        ) / Decimal("120")

    def test_drawdown_fraction_at_peak_is_zero(self) -> None:
        sig = _signal(peak="120", current="120")
        assert sig.drawdown_fraction == Decimal("0")

    def test_drawdown_fraction_zero_peak_is_zero_cold_start(self) -> None:
        # Cold-start guard — bez szczytu (0) nie da się policzyć spadku.
        sig = _signal(peak="0", current="50")
        assert sig.drawdown_fraction == Decimal("0")


class TestEvaluateDrawdown:
    def test_flat_history_is_normal(self) -> None:
        sig = _signal(peak="100", current="100")
        assert (
            sig.evaluate_drawdown(
                elevated_pct=Decimal("0.10"),
                critical_pct=Decimal("0.20"),
            )
            is MacroAlertLevel.NORMAL
        )

    def test_small_dip_is_normal(self) -> None:
        # -5% spadek < 10% próg ELEVATED.
        sig = _signal(peak="100", current="95")
        assert (
            sig.evaluate_drawdown(
                elevated_pct=Decimal("0.10"),
                critical_pct=Decimal("0.20"),
            )
            is MacroAlertLevel.NORMAL
        )

    def test_moderate_drawdown_is_elevated(self) -> None:
        # -12% → przekracza 10%, nie sięga 20% → ELEVATED.
        sig = _signal(peak="100", current="88")
        assert (
            sig.evaluate_drawdown(
                elevated_pct=Decimal("0.10"),
                critical_pct=Decimal("0.20"),
            )
            is MacroAlertLevel.ELEVATED
        )

    def test_large_drawdown_is_critical(self) -> None:
        # -25% → przekracza 20% → CRITICAL.
        sig = _signal(peak="100", current="75")
        assert (
            sig.evaluate_drawdown(
                elevated_pct=Decimal("0.10"),
                critical_pct=Decimal("0.20"),
            )
            is MacroAlertLevel.CRITICAL
        )

    def test_threshold_boundary_is_inclusive(self) -> None:
        # Dokładnie -20% → CRITICAL (próg włączający, jak w macro_risk).
        sig = _signal(peak="100", current="80")
        assert (
            sig.evaluate_drawdown(
                elevated_pct=Decimal("0.10"),
                critical_pct=Decimal("0.20"),
            )
            is MacroAlertLevel.CRITICAL
        )

    def test_zero_peak_is_normal_cold_start(self) -> None:
        sig = _signal(peak="0", current="50")
        assert (
            sig.evaluate_drawdown(
                elevated_pct=Decimal("0.10"),
                critical_pct=Decimal("0.20"),
            )
            is MacroAlertLevel.NORMAL
        )

    def test_negative_thresholds_raise(self) -> None:
        sig = _signal(peak="100", current="80")
        elevated_pct = Decimal("-0.1")
        critical_pct = Decimal("0.2")
        with pytest.raises(ValueError):
            sig.evaluate_drawdown(elevated_pct=elevated_pct, critical_pct=critical_pct)

    def test_inverted_thresholds_raise(self) -> None:
        sig = _signal(peak="100", current="80")
        elevated_pct = Decimal("0.2")
        critical_pct = Decimal("0.1")
        with pytest.raises(ValueError):
            sig.evaluate_drawdown(elevated_pct=elevated_pct, critical_pct=critical_pct)
