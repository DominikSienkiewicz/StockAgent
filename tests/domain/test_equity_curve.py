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
from datetime import UTC, datetime, timedelta

import pytest

from src.domain.benchmark import index_curve, window_change
from src.domain.equity_curve import (
    DirectionalHitRate,
    EquityCurve,
    EquityPoint,
    PaperTrade,
    buy_and_hold_curve,
    directional_hit_rate,
    equity_curve,
    excess_return,
    market_relative_hit_rate,
    paper_portfolio_curve,
    portfolio_equity_curve,
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

    def test_max_drawdown_within_unit_interval_on_wipeout(self) -> None:
        # Short przy +150% to strata ponad cały kapitał: kapitał kończy na 0
        # (ruina), a nie na -0.5, więc obsunięcie to 100%, nie 150%.
        curve = equity_curve([("BEARISH", 1.5)])

        assert curve.points[0].equity == 0.0
        assert curve.max_drawdown == 1.0

    def test_wipeout_ends_the_curve_at_zero_equity(self) -> None:
        # Po ruinie kolejne zyski nie wskrzeszają kapitału ani nie zmieniają obsunięcia.
        curve = equity_curve([("BULLISH", 0.10), ("BEARISH", 1.5), ("BULLISH", 0.20)])

        assert [point.equity for point in curve.points][1:] == [0.0, 0.0]
        assert 0.0 <= curve.max_drawdown <= 1.0
        assert curve.max_drawdown == 1.0

    def test_portfolio_wipeout_stays_within_unit_interval(self) -> None:
        curve = portfolio_equity_curve([[("BEARISH", 3.0)], [("BULLISH", 0.5)]])

        assert all(point.equity >= 0.0 for point in curve.points)
        assert curve.max_drawdown == 1.0


class TestImmutability:
    def test_equity_point_is_frozen(self) -> None:
        point = EquityPoint(index=0, period_return=0.05, equity=1.05)
        with pytest.raises(FrozenInstanceError):
            point.equity = 2.0  # type: ignore[misc]

    def test_equity_curve_is_frozen(self) -> None:
        curve = equity_curve([("BULLISH", 0.05)])
        with pytest.raises(FrozenInstanceError):
            curve.total_return = 0.0  # type: ignore[misc]


class TestDirectionalHitRate:
    """WIA-05: jedna domenowa definicja hit-rate kierunkowego — SIDEWAYS (sygnał 0)
    nie stawia zakładu, więc nie wchodzi do mianownika."""

    def test_sideways_calls_are_excluded_from_the_denominator(self) -> None:
        result = directional_hit_rate(
            [
                ("BULLISH", True),
                ("BEARISH", False),
                ("SIDEWAYS", True),
                ("SIDEWAYS", True),
            ]
        )

        assert result == DirectionalHitRate(hits=1, directional=2, sideways=2)
        assert result.rate == pytest.approx(0.5)

    def test_unknown_trend_counts_as_no_bet(self) -> None:
        # Spójnie z `signal_of`: nieznana etykieta = brak ekspozycji.
        result = directional_hit_rate([("WHATEVER", True), ("bullish", True)])

        assert result.directional == 1
        assert result.sideways == 1
        assert result.rate == pytest.approx(1.0)

    def test_no_directional_calls_has_no_rate(self) -> None:
        # Brak zakładów to brak wyniku — nie 0% i nie 100%.
        assert directional_hit_rate([("SIDEWAYS", True)]).rate is None
        assert directional_hit_rate([]).rate is None


class TestPortfolioEquityCurve:
    """WIA-17: jeden punkt na cykl, równa waga trade'ów w cyklu, SIDEWAYS = gotówka.
    Szeregowe składanie równoległych trade'ów na 100% kapitału nie ma sensu."""

    def test_portfolio_curve_averages_trades_within_cycle(self) -> None:
        cycles = [[("BULLISH", 0.10), ("BULLISH", 0.10)]]

        curve = portfolio_equity_curve(cycles)

        assert curve.total_return == pytest.approx(0.10)
        # Ten sam zestaw szeregowo dałby 1.1 * 1.1 - 1 = 0.21.
        assert equity_curve(cycles[0]).total_return == pytest.approx(0.21)

    def test_one_point_per_cycle(self) -> None:
        cycles = [
            [("BULLISH", 0.10), ("BEARISH", 0.02)],
            [("BEARISH", -0.04)],
        ]

        curve = portfolio_equity_curve(cycles)

        # Cykl 1: (0.10 + (-1 * 0.02)) / 2 = 0.04; cykl 2: 0.04.
        assert [p.index for p in curve.points] == [0, 1]
        assert curve.points[0].period_return == pytest.approx(0.04)
        assert curve.points[1].equity == pytest.approx(1.04 * 1.04)
        assert curve.n_trades == 3

    def test_sideways_is_cash_with_equal_weight(self) -> None:
        # SIDEWAYS nie stawia zakładu, ale zajmuje swoją 1/N kapitału jako gotówka.
        curve = portfolio_equity_curve([[("BULLISH", 0.10), ("SIDEWAYS", 0.30)]])

        assert curve.total_return == pytest.approx(0.05)

    def test_win_rate_and_drawdown_use_the_same_rules_as_serial_curve(self) -> None:
        cycles = [
            [("BULLISH", 0.20)],
            [("BULLISH", -0.25), ("SIDEWAYS", 0.0)],
        ]

        curve = portfolio_equity_curve(cycles)

        assert curve.win_rate == pytest.approx(0.5)
        # 1.2 → 1.2 * (1 - 0.125) = 1.05; obsunięcie (1.2 - 1.05) / 1.2 = 0.125.
        assert curve.max_drawdown == pytest.approx(0.125)

    def test_empty_cycles_are_skipped_and_empty_input_is_all_zeros(self) -> None:
        assert portfolio_equity_curve([]) == equity_curve([])
        assert portfolio_equity_curve([[], [("BULLISH", 0.1)]]).points[0].index == 0


def test_round_trip_cost_is_charged_on_directional_entry() -> None:
    """WIA-50: wejście z gotówki płaci nogę wejścia round tripu (10 bps → -0.001);
    noga wyjścia przychodzi dopiero przy zmianie pozycji."""
    curve = equity_curve([("BULLISH", 0.0)], cost_bps=10)

    assert curve.points[0].period_return == pytest.approx(-0.001)
    assert curve.total_return == pytest.approx(-0.001)


class TestSerialCurveCosts:
    """WIA-50: w szeregowej krzywej poprzedni trade to poprzednia pozycja."""

    def test_holding_the_same_direction_pays_only_once(self) -> None:
        curve = equity_curve([("BULLISH", 0.01), ("BULLISH", 0.01)], cost_bps=10)

        assert curve.points[0].period_return == pytest.approx(0.009)
        assert curve.points[1].period_return == pytest.approx(0.01)

    def test_sideways_exit_and_flip_are_charged(self) -> None:
        curve = equity_curve(
            [("BULLISH", 0.0), ("SIDEWAYS", 0.05), ("BEARISH", 0.0), ("BULLISH", 0.0)],
            cost_bps=10,
        )

        returns = [p.period_return for p in curve.points]
        assert returns == [
            pytest.approx(-0.001),  # wejście
            pytest.approx(-0.001),  # wyjście do gotówki
            pytest.approx(-0.001),  # wejście w short
            pytest.approx(-0.002),  # odwrócenie short → long
        ]

    def test_default_cost_is_zero(self) -> None:
        trades = [("BULLISH", 0.02), ("BEARISH", 0.01)]

        assert equity_curve(trades) == equity_curve(trades, cost_bps=0)

    def test_win_rate_stays_directional_and_gross(self) -> None:
        # Koszt zmienia PnL, nie trafność: +0.05% brutto to nadal trafiony long.
        curve = equity_curve([("BULLISH", 0.0005)], cost_bps=10)

        assert curve.win_rate == pytest.approx(1.0)
        assert curve.total_return < 0


class TestPaperPortfolioCurve:
    """WIA-50: portfel o równej wadze per cykl z kosztem od obrotu per symbol."""

    def test_entry_cost_is_weighted_by_the_trade_share(self) -> None:
        cycles = [
            [
                PaperTrade("AAA", "BULLISH", 0.0, cost_bps=10),
                PaperTrade("BBB", "SIDEWAYS", 0.0, cost_bps=10),
            ]
        ]

        curve = paper_portfolio_curve(cycles)

        # AAA wchodzi (−0.001) na połowie kapitału, BBB zostaje w gotówce.
        assert curve.total_return == pytest.approx(-0.0005)

    def test_turnover_is_tracked_per_symbol_across_cycles(self) -> None:
        cycles = [
            [PaperTrade("AAA", "BULLISH", 0.01, 10), PaperTrade("BBB", "BEARISH", 0.0, 10)],
            [PaperTrade("AAA", "BULLISH", 0.01, 10), PaperTrade("BBB", "BULLISH", 0.0, 10)],
        ]

        curve = paper_portfolio_curve(cycles)

        # Cykl 1: AAA i BBB wchodzą (po −0.001). Cykl 2: AAA trzyma (0), BBB
        # odwraca short → long (−0.002).
        assert curve.points[0].period_return == pytest.approx((0.009 - 0.001) / 2)
        assert curve.points[1].period_return == pytest.approx((0.01 - 0.002) / 2)

    def test_sideways_exit_is_charged_to_the_symbol(self) -> None:
        cycles = [
            [PaperTrade("AAA", "BULLISH", 0.0, 10)],
            [PaperTrade("AAA", "SIDEWAYS", 0.03, 10)],
        ]

        curve = paper_portfolio_curve(cycles)

        assert curve.points[1].period_return == pytest.approx(-0.001)

    def test_each_trade_carries_its_own_cost(self) -> None:
        # Krypto ma wyższy koszt niż akcje — koszt jedzie z trade'em.
        cycles = [
            [
                PaperTrade("AAA", "BULLISH", 0.0, cost_bps=10),
                PaperTrade("BTC", "BULLISH", 0.0, cost_bps=25),
            ]
        ]

        curve = paper_portfolio_curve(cycles)

        assert curve.total_return == pytest.approx(-(0.001 + 0.0025) / 2)

    def test_zero_cost_matches_the_cost_free_portfolio_curve(self) -> None:
        cycles = [
            [("BULLISH", 0.10), ("BEARISH", 0.02)],
            [("BEARISH", -0.04), ("SIDEWAYS", 0.3)],
        ]
        paper = [
            [PaperTrade(f"S{i}", trend, change) for i, (trend, change) in enumerate(cycle)]
            for cycle in cycles
        ]

        assert paper_portfolio_curve(paper) == portfolio_equity_curve(cycles)


def test_benchmark_curve_compounds_long_only_changes() -> None:
    """WIA-21: koszyk to buy&hold tych samych symboli w tych samych oknach —
    long niezależnie od prognozy agenta."""
    cycles = [
        [PaperTrade("AAA", "BEARISH", 0.02)],
        [PaperTrade("AAA", "SIDEWAYS", -0.01)],
    ]

    basket = buy_and_hold_curve(cycles)

    assert [p.period_return for p in basket.points] == [
        pytest.approx(0.02),
        pytest.approx(-0.01),
    ]
    assert basket.total_return == pytest.approx(1.02 * 0.99 - 1)
    # Agent na tych samych oknach: short traci 2%, gotówka stoi.
    assert paper_portfolio_curve(cycles).total_return == pytest.approx(-0.02)


def test_bullish_counts_as_market_relative_hit_when_stock_falls_less_than_benchmark() -> None:
    """WIA-21: akcja −1% przy IVV −3% to zwrot nadwyżkowy +2% — BULLISH trafił
    względem rynku, choć kierunkowo chybił."""
    result = market_relative_hit_rate([("BULLISH", -0.01, -0.03)])

    assert result == DirectionalHitRate(hits=1, directional=1, sideways=0)
    assert excess_return(-0.01, -0.03) == pytest.approx(0.02)


def test_benchmark_curve_aligned_to_same_days() -> None:
    """WIA-21: okno benchmarku zaczyna się snapshotem z TEGO SAMEGO cyklu co
    predykcja i kończy w cyklu rozliczenia (≥ 6 h później) — nie od wczorajszego
    snapshotu, nawet gdy benchmark zapisano kilka minut po predykcji."""
    history = [
        (datetime(2026, 10, 1, 5, 31, tzinfo=UTC), 100.0),
        (datetime(2026, 10, 2, 5, 40, tzinfo=UTC), 102.0),  # po predykcji z 05:35
        (datetime(2026, 10, 3, 5, 31, tzinfo=UTC), 101.0),
        (datetime(2026, 10, 4, 5, 31, tzinfo=UTC), 104.0),
    ]
    stamps = [datetime(2026, 10, day, 5, 35, tzinfo=UTC) for day in (1, 2, 3)]

    changes = [window_change(history, stamp, timedelta(hours=6)) for stamp in stamps]
    curve = index_curve(
        [[PaperTrade("AAA", "BULLISH", 0.0, benchmark_change=change)] for change in changes],
        symbol="IVV",
        cost_bps=0,
    )

    assert changes == [
        pytest.approx(0.02),
        pytest.approx(101 / 102 - 1),
        pytest.approx(104 / 101 - 1),
    ]
    assert curve is not None
    assert [p.index for p in curve.points] == [0, 1, 2]
    assert curve.total_return == pytest.approx(0.04)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_change_raises(bad: float) -> None:
    # PORT-09: NaN zmiany ceny zatruwał cały kapitał krzywej bez śladu.
    with pytest.raises(ValueError, match="finite"):
        equity_curve([("BULLISH", 0.01), ("BEARISH", bad)])
    with pytest.raises(ValueError, match="finite"):
        portfolio_equity_curve([[("BULLISH", bad)]])
