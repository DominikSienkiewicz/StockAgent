"""Testy domeny benchmarku krzywej kapitału (WIA-21).

Benchmark (IVV) mierzy się tymi samymi snapshotami cyklu co predykcje: start =
snapshot z tego samego cyklu (dzień UTC) najbliższy chwili predykcji, koniec =
pierwszy snapshot co najmniej `settle_after` później (cykl rozliczenia). Snapshot
o 05:30 UTC to zamknięcie poprzedniej sesji po obu stronach porównania.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.domain.benchmark import (
    BenchmarkComparison,
    compare_with_benchmarks,
    index_curve,
    window_change,
)
from src.domain.equity_curve import DirectionalHitRate, PaperTrade, paper_portfolio_curve

SETTLE = timedelta(hours=6)


def _at(day: int, hour: int = 5, minute: int = 31) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC)


class TestWindowChange:
    def test_missing_snapshot_in_the_prediction_cycle_gives_no_change(self) -> None:
        # Wczorajszy snapshot to inny cykl — nie może udawać startu okna.
        history = [(_at(1), 100.0), (_at(3), 103.0)]

        assert window_change(history, _at(2, minute=35), SETTLE) is None

    def test_no_settlement_snapshot_yet_gives_no_change(self) -> None:
        history = [(_at(1), 100.0), (_at(1, hour=9), 101.0)]

        assert window_change(history, _at(1, minute=35), SETTLE) is None

    def test_end_is_the_first_snapshot_after_the_settlement_age(self) -> None:
        # Drugi przebieg tego samego dnia (< 6 h) jeszcze nie rozlicza predykcji.
        history = [(_at(1), 100.0), (_at(1, hour=9), 150.0), (_at(2), 101.0)]

        assert window_change(history, _at(1, minute=35), SETTLE) == pytest.approx(0.01)

    def test_start_is_the_nearest_snapshot_of_the_same_cycle(self) -> None:
        # Dwa przebiegi jednego dnia: predykcja z 12:00 startuje od snapshotu 12:02.
        history = [(_at(1), 100.0), (_at(1, hour=12, minute=2), 110.0), (_at(2), 121.0)]

        assert window_change(history, _at(1, hour=12, minute=0), SETTLE) == pytest.approx(0.10)

    def test_unsorted_history_and_naive_utc_timestamps(self) -> None:
        history = [(datetime(2026, 10, 2, 5, 31), 105.0), (datetime(2026, 10, 1, 5, 31), 100.0)]

        assert window_change(history, _at(1, minute=35), SETTLE) == pytest.approx(0.05)

    def test_zero_start_price_gives_no_change(self) -> None:
        assert window_change([(_at(1), 0.0), (_at(2), 1.0)], _at(1), SETTLE) is None


class TestIndexCurve:
    def test_entry_cost_is_paid_once(self) -> None:
        cycles = [
            [PaperTrade("AAA", "BULLISH", 0.0, benchmark_change=0.01)],
            [PaperTrade("AAA", "BEARISH", 0.0, benchmark_change=0.01)],
        ]

        curve = index_curve(cycles, symbol="IVV", cost_bps=10)

        assert curve is not None
        assert [p.period_return for p in curve.points] == [
            pytest.approx(0.009),
            pytest.approx(0.01),
        ]

    def test_one_point_per_agent_cycle_from_the_trades_with_data(self) -> None:
        cycles = [
            [
                PaperTrade("AAA", "BULLISH", 0.0, benchmark_change=0.02),
                PaperTrade("BBB", "BULLISH", 0.0, benchmark_change=None),
            ]
        ]

        curve = index_curve(cycles, symbol="IVV", cost_bps=0)

        assert curve is not None
        assert curve.total_return == pytest.approx(0.02)

    def test_a_cycle_without_benchmark_data_gives_no_curve(self) -> None:
        # Brak cyklu w benchmarku rozjechałby krzywe — lepiej jawne „brak danych”.
        cycles = [
            [PaperTrade("AAA", "BULLISH", 0.0, benchmark_change=0.02)],
            [PaperTrade("AAA", "BULLISH", 0.0)],
        ]

        assert index_curve(cycles, symbol="IVV", cost_bps=0) is None


class TestCompareWithBenchmarks:
    def _cycles(self) -> list[list[PaperTrade]]:
        return [
            [
                PaperTrade("AAA", "BULLISH", 0.02, 10, benchmark_change=0.01),
                PaperTrade("BBB", "BEARISH", 0.01, 10, benchmark_change=0.01),
            ],
            [
                PaperTrade("AAA", "BULLISH", -0.01, 10, benchmark_change=-0.03),
                PaperTrade("BBB", "SIDEWAYS", 0.04, 10, benchmark_change=-0.03),
            ],
        ]

    def test_all_three_curves_share_the_costs_and_the_cycles(self) -> None:
        comparison = compare_with_benchmarks(self._cycles(), index_symbol="IVV", index_cost_bps=10)

        assert isinstance(comparison, BenchmarkComparison)
        assert comparison.agent == paper_portfolio_curve(self._cycles())
        # Koszyk: oba symbole long, wejście po 10 bps tylko w pierwszym cyklu.
        assert comparison.basket.points[0].period_return == pytest.approx((0.019 + 0.009) / 2)
        assert comparison.basket.points[1].period_return == pytest.approx((-0.01 + 0.04) / 2)
        assert comparison.index is not None
        assert [p.period_return for p in comparison.index.points] == [
            pytest.approx(0.009),
            pytest.approx(-0.03),
        ]
        assert comparison.index_symbol == "IVV"

    def test_excess_returns_are_differences_of_total_returns(self) -> None:
        comparison = compare_with_benchmarks(self._cycles(), index_symbol="IVV", index_cost_bps=10)

        assert comparison.excess_vs_basket == pytest.approx(
            comparison.agent.total_return - comparison.basket.total_return
        )
        assert comparison.index is not None
        assert comparison.excess_vs_index == pytest.approx(
            comparison.agent.total_return - comparison.index.total_return
        )

    def test_market_hit_rate_scores_the_sign_of_the_excess_return(self) -> None:
        comparison = compare_with_benchmarks(self._cycles(), index_symbol="IVV", index_cost_bps=10)

        # AAA +2% vs +1% (trafienie), BBB short +1% vs +1% (zero ≠ trafienie),
        # AAA −1% vs −3% (trafienie), BBB SIDEWAYS poza mianownikiem.
        assert comparison.market_hit_rate == DirectionalHitRate(hits=2, directional=3, sideways=1)

    def test_without_benchmark_data_the_index_side_is_empty(self) -> None:
        cycles = [[PaperTrade("AAA", "BULLISH", 0.02, 10)]]

        comparison = compare_with_benchmarks(cycles, index_symbol="IVV", index_cost_bps=10)

        assert comparison.index is None
        assert comparison.excess_vs_index is None
        assert comparison.market_hit_rate.rate is None
        assert comparison.basket.total_return == pytest.approx(0.019)

    def test_predictions_of_the_index_itself_skip_the_market_hit_rate(self) -> None:
        # IVV vs IVV to zawsze zerowy zwrot nadwyżkowy — nie pudło, tylko brak zakładu.
        cycles = [
            [
                PaperTrade("IVV", "BULLISH", 0.01, benchmark_change=0.01),
                PaperTrade("AAA", "BULLISH", 0.02, benchmark_change=0.01),
            ]
        ]

        comparison = compare_with_benchmarks(cycles, index_symbol="IVV", index_cost_bps=0)

        assert comparison.market_hit_rate == DirectionalHitRate(hits=1, directional=1, sideways=0)
        # Do krzywych agenta i koszyka predykcja IVV wchodzi normalnie.
        assert comparison.agent.n_trades == 2
