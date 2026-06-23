"""Testy domeny Equity Curve — krzywa kapitału z historii trafień predykcji.

Czysta domena (stdlib only). Każdy trade to (predicted_trend, realized_change_pct
jako ułamek, np. 0.03 dla +3%). Sygnał: BULLISH→+1, BEARISH→-1, SIDEWAYS→0.
period_return = signal * realized_change_pct. Kapitał składa się procentowo:
equity_i = equity_{i-1} * (1 + period_return).

Reguły metryk:
- total_return = final_equity / starting_equity - 1,
- win_rate = ułamek trade'ów KIERUNKOWYCH (signal != 0) z period_return > 0;
  trade'y SIDEWAYS / o zerowym sygnale są WYŁĄCZONE z mianownika,
- max_drawdown = największy spadek szczyt→dołek serii kapitału jako DODATNI ułamek,
- pusta historia → EquityCurve z samymi zerami.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.domain.equity_curve import (
    EquityCurve,
    EquityPoint,
    equity_curve,
    signal_of,
)


class TestSignalOf:
    def test_bullish_is_long(self) -> None:
        assert signal_of("BULLISH") == 1

    def test_bearish_is_short(self) -> None:
        assert signal_of("BEARISH") == -1

    def test_sideways_is_flat(self) -> None:
        assert signal_of("SIDEWAYS") == 0

    def test_unknown_trend_is_flat(self) -> None:
        # Nieznany trend traktujemy jak brak kierunku (fail-safe, zero ekspozycji).
        assert signal_of("WHATEVER") == 0

    def test_is_case_insensitive(self) -> None:
        assert signal_of("bullish") == 1
        assert signal_of("Bearish") == -1
        assert signal_of("sideways") == 0


class TestEquityCurveEmpty:
    def test_empty_history_is_all_zeros(self) -> None:
        curve = equity_curve([])
        assert curve == EquityCurve(
            points=(),
            total_return=0.0,
            win_rate=0.0,
            max_drawdown=0.0,
            n_trades=0,
        )


class TestEquityCurveCompounding:
    def test_three_correct_bullish_calls_compound(self) -> None:
        # Trzy trafione long-y po +5% — kapitał składa się: 1.05^3.
        trades = [
            ("BULLISH", 0.05),
            ("BULLISH", 0.05),
            ("BULLISH", 0.05),
        ]
        curve = equity_curve(trades)

        assert curve.n_trades == 3
        assert curve.total_return == 1.05**3 - 1
        assert curve.win_rate == pytest.approx(1.0)
        # Monotoniczny wzrost → brak obsunięcia.
        assert curve.max_drawdown == pytest.approx(0.0)

    def test_points_carry_running_equity(self) -> None:
        trades = [("BULLISH", 0.05), ("BULLISH", 0.05), ("BULLISH", 0.05)]
        curve = equity_curve(trades)

        assert curve.points == (
            EquityPoint(index=0, period_return=0.05, equity=1.05),
            EquityPoint(index=1, period_return=0.05, equity=1.05**2),
            EquityPoint(index=2, period_return=0.05, equity=1.05**3),
        )

    def test_custom_starting_equity_scales_curve(self) -> None:
        curve = equity_curve([("BULLISH", 0.10)], starting_equity=1000.0)
        assert curve.points[0].equity == pytest.approx(1100.0)
        # total_return jest niezależny od skali kapitału startowego
        # (drobny szum zmiennoprzecinkowy z dzielenia 1100/1000 - 1).
        assert abs(curve.total_return - 0.10) < 1e-12


class TestEquityCurveDirection:
    def test_correct_bearish_call_is_a_win(self) -> None:
        # Short trafiony: cena spadła o 4% → period_return = (-1) * (-0.04) = +0.04.
        curve = equity_curve([("BEARISH", -0.04)])
        assert curve.points[0].period_return == pytest.approx(0.04)
        assert curve.win_rate == pytest.approx(1.0)
        assert abs(curve.total_return - 0.04) < 1e-12

    def test_wrong_bullish_call_yields_negative_return(self) -> None:
        # Long, a cena spadła o 6% → period_return = (+1) * (-0.06) = -0.06.
        curve = equity_curve([("BULLISH", -0.06)])
        assert curve.points[0].period_return == pytest.approx(-0.06)
        assert abs(curve.total_return - (-0.06)) < 1e-12
        assert curve.win_rate == pytest.approx(0.0)
        # Spadek z 1.0 do 0.94 to 6% obsunięcia.
        assert abs(curve.max_drawdown - 0.06) < 1e-12


class TestWinRate:
    def test_sideways_excluded_from_denominator(self) -> None:
        # Dwa kierunkowe trade'y (jeden wygrany), jeden SIDEWAYS (bez sygnału).
        # win_rate liczony tylko z kierunkowych: 1 z 2 = 0.5.
        trades = [
            ("BULLISH", 0.05),  # wygrany kierunkowy
            ("BULLISH", -0.05),  # przegrany kierunkowy
            ("SIDEWAYS", 0.05),  # zerowy sygnał — poza mianownikiem
        ]
        curve = equity_curve(trades)
        assert curve.n_trades == 3
        assert curve.win_rate == pytest.approx(0.5)

    def test_all_sideways_win_rate_is_zero(self) -> None:
        # Brak trade'ów kierunkowych → mianownik 0 → win_rate 0.0 (bez dzielenia
        # przez zero).
        curve = equity_curve([("SIDEWAYS", 0.05), ("SIDEWAYS", -0.03)])
        assert curve.win_rate == pytest.approx(0.0)

    def test_zero_return_directional_trade_is_not_a_win(self) -> None:
        # Kierunkowy trade z dokładnie zerowym zwrotem nie jest wygraną
        # (period_return > 0 jest ścisłe), ale liczy się do mianownika.
        curve = equity_curve([("BULLISH", 0.0), ("BULLISH", 0.05)])
        assert curve.win_rate == pytest.approx(0.5)


class TestSidewaysReturn:
    def test_sideways_contributes_zero_return(self) -> None:
        # SIDEWAYS → signal 0 → period_return 0 → kapitał bez zmian.
        curve = equity_curve([("SIDEWAYS", 0.05)])
        assert curve.points[0].period_return == pytest.approx(0.0)
        assert curve.points[0].equity == pytest.approx(1.0)
        assert curve.total_return == pytest.approx(0.0)
        assert curve.max_drawdown == pytest.approx(0.0)


class TestMaxDrawdown:
    def test_up_down_up_series_drawdown_is_hand_computed(self) -> None:
        # Seria up-down-up na kapitale startowym 1.0:
        #   t0: +20% → equity 1.20 (szczyt)
        #   t1: -25% → equity 0.90 (dołek)
        #   t2: +10% → equity 0.99
        # Największe obsunięcie szczyt→dołek: (1.20 - 0.90) / 1.20 = 0.25.
        trades = [
            ("BULLISH", 0.20),
            ("BULLISH", -0.25),
            ("BULLISH", 0.10),
        ]
        curve = equity_curve(trades)
        assert curve.points[0].equity == pytest.approx(1.20)
        assert abs(curve.points[1].equity - 0.90) < 1e-12
        assert abs(curve.points[2].equity - 0.99) < 1e-12
        assert abs(curve.max_drawdown - 0.25) < 1e-12

    def test_monotonic_rise_has_no_drawdown(self) -> None:
        curve = equity_curve([("BULLISH", 0.01), ("BULLISH", 0.02)])
        assert curve.max_drawdown == pytest.approx(0.0)


class TestImmutability:
    def test_equity_point_is_frozen(self) -> None:
        point = EquityPoint(index=0, period_return=0.05, equity=1.05)
        with pytest.raises(FrozenInstanceError):
            point.equity = 2.0  # type: ignore[misc]

    def test_equity_curve_is_frozen(self) -> None:
        curve = equity_curve([("BULLISH", 0.05)])
        with pytest.raises(FrozenInstanceError):
            curve.total_return = 0.0  # type: ignore[misc]
