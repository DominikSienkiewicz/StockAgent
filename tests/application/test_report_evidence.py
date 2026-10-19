"""Testy wspólnego dowodu trafności — krzywa portfela i założenia kosztowe (WIA-50).

Mapowanie `ResolvedPrediction` → trade'y domeny: chronologicznie, z kosztem
dobranym do symbolu (krypto ma własny koszt), plus copy założeń dla maila
(polski zapis liczb) i publicznego scorecardu (kropka)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.application.report_evidence import (
    BENCHMARK_SYMBOL,
    PORTFOLIO_ASSUMPTIONS_PL,
    PaperCosts,
    benchmark_comparison,
    benchmark_detail,
    benchmark_line,
    paper_pnl_label,
    portfolio_assumptions,
    portfolio_curve,
)
from src.application.report_formatting import DOT_NUMBERS
from src.application.report_models import ResolvedPrediction
from src.domain.benchmark import BenchmarkComparison

COSTS = PaperCosts(cost_bps=10, crypto_cost_bps=25, crypto_symbols=frozenset({"BTC"}))


def _call(symbol: str, trend: str, day: int, actual: str = "100") -> ResolvedPrediction:
    return ResolvedPrediction(
        symbol=symbol,
        predicted_trend=trend,
        is_correct=True,
        price_at_prediction=Decimal("100"),
        actual_price=Decimal(actual),
        predicted_at=datetime(2026, 10, day, 5, 35, tzinfo=UTC),
    )


class TestPaperCosts:
    def test_crypto_symbols_use_the_crypto_cost(self) -> None:
        assert COSTS.for_symbol("BTC") == 25
        assert COSTS.for_symbol("AAPL") == 10

    def test_default_costs_are_zero(self) -> None:
        assert PaperCosts().for_symbol("BTC") == 0


class TestPortfolioCurveCosts:
    def test_without_costs_the_curve_is_gross(self) -> None:
        curve = portfolio_curve([_call("AAPL", "BULLISH", 1, "101")])

        assert curve is not None
        assert curve.total_return == pytest.approx(0.01)

    def test_costs_are_charged_once_per_position_change(self) -> None:
        # Repozytorium zwraca od najnowszej — krzywa i obrót idą chronologicznie.
        resolved = [
            _call("AAPL", "SIDEWAYS", 3),
            _call("AAPL", "BULLISH", 2, "101"),
            _call("AAPL", "BULLISH", 1, "101"),
        ]

        curve = portfolio_curve(resolved, COSTS)

        assert curve is not None
        assert [p.period_return for p in curve.points] == [
            pytest.approx(0.01 - 0.001),  # wejście
            pytest.approx(0.01),  # trzymanie
            pytest.approx(-0.001),  # wyjście do gotówki
        ]

    def test_crypto_trade_pays_the_crypto_cost(self) -> None:
        curve = portfolio_curve([_call("BTC", "BULLISH", 1)], COSTS)

        assert curve is not None
        assert curve.total_return == pytest.approx(-0.0025)


class TestCostCopy:
    def test_label_names_the_cost_in_bps(self) -> None:
        assert paper_pnl_label(COSTS) == "paper-PnL (koszt 10 bps)"

    def test_fractional_bps_follow_the_number_style(self) -> None:
        costs = PaperCosts(cost_bps=12.5)

        assert paper_pnl_label(costs) == "paper-PnL (koszt 12,5 bps)"
        assert paper_pnl_label(costs, numbers=DOT_NUMBERS) == "paper-PnL (koszt 12.5 bps)"

    def test_assumptions_state_the_turnover_rule_and_the_crypto_cost(self) -> None:
        text = portfolio_assumptions(COSTS)

        assert "równa waga" in text
        assert "gotówka" in text
        assert "10 bps" in text
        assert "krypto 25 bps" in text
        assert "tylko przy zmianie pozycji" in text
        assert "bez kosztów transakcyjnych" not in text

    def test_assumptions_skip_crypto_without_crypto_symbols(self) -> None:
        assert "krypto" not in portfolio_assumptions(PaperCosts(cost_bps=10, crypto_cost_bps=25))

    def test_without_costs_the_legacy_assumptions_stay(self) -> None:
        assert portfolio_assumptions(None) == PORTFOLIO_ASSUMPTIONS_PL


class TestBenchmarkComparison:
    """WIA-21: koszyk i IVV z tych samych okien co predykcje, z tymi samymi
    kosztami; okno IVV z jego snapshotów cyklu predykcji i cyklu rozliczenia."""

    _HISTORY = [
        (datetime(2026, 10, 1, 5, 31, tzinfo=UTC), 100.0),
        (datetime(2026, 10, 2, 5, 31, tzinfo=UTC), 101.0),
        (datetime(2026, 10, 3, 5, 31, tzinfo=UTC), 102.01),
    ]

    def test_comparison_uses_the_same_windows_and_costs(self) -> None:
        resolved = [_call("AAPL", "BULLISH", 2, "102"), _call("AAPL", "BULLISH", 1, "99")]

        comparison = benchmark_comparison(
            resolved, COSTS, self._HISTORY, settle_after=timedelta(hours=6)
        )

        assert comparison is not None
        assert comparison.agent == portfolio_curve(resolved, COSTS)
        assert comparison.index_symbol == BENCHMARK_SYMBOL == "IVV"
        assert comparison.index is not None
        assert [p.period_return for p in comparison.index.points] == [
            pytest.approx(0.01 - 0.001),
            pytest.approx(0.01),
        ]
        # −1% przy IVV +1% to pudło; +2% przy IVV +1% to trafienie względem rynku.
        assert (comparison.market_hit_rate.hits, comparison.market_hit_rate.directional) == (1, 2)

    def test_without_benchmark_history_only_the_basket_remains(self) -> None:
        comparison = benchmark_comparison(
            [_call("AAPL", "BULLISH", 1, "101")], COSTS, [], settle_after=timedelta(hours=6)
        )

        assert comparison is not None
        assert comparison.index is None
        assert comparison.basket.total_return == pytest.approx(0.009)

    def test_no_price_moves_no_comparison(self) -> None:
        comparison = benchmark_comparison(
            [], COSTS, self._HISTORY, settle_after=timedelta(hours=6)
        )

        assert comparison is None


class TestBenchmarkCopy:
    def _comparison(self, *, with_index: bool = True) -> BenchmarkComparison:
        history = [
            (datetime(2026, 10, 1, 5, 31, tzinfo=UTC), 100.0),
            (datetime(2026, 10, 2, 5, 31, tzinfo=UTC), 102.1),
        ]
        comparison = benchmark_comparison(
            [_call("AAPL", "BEARISH", 1, "103.4")],
            PaperCosts(),
            history if with_index else [],
            settle_after=timedelta(hours=6),
        )
        assert comparison is not None
        return comparison

    def test_line_puts_agent_basket_and_index_side_by_side(self) -> None:
        assert benchmark_line(self._comparison()) == "agent -3,4% · koszyk +3,4% · IVV +2,1%"

    def test_scorecard_line_keeps_the_decimal_point(self) -> None:
        line = benchmark_line(self._comparison(), numbers=DOT_NUMBERS)

        assert line == "agent -3.4% · koszyk +3.4% · IVV +2.1%"

    def test_missing_index_says_so_instead_of_a_blank_number(self) -> None:
        line = benchmark_line(self._comparison(with_index=False))

        assert line == "agent -3,4% · koszyk +3,4% · IVV: brak danych benchmarku"

    def test_detail_states_the_excess_return_and_the_market_hit_rate(self) -> None:
        detail = benchmark_detail(self._comparison())

        assert "Zwrot nadwyżkowy vs koszyk: -6,8 p.p." in detail
        assert "hit-rate względem rynku (IVV): 0,0% (n=1)" in detail

    def test_detail_without_index_data(self) -> None:
        detail = benchmark_detail(self._comparison(with_index=False))

        assert "hit-rate względem rynku (IVV): brak danych benchmarku" in detail
