"""Testy wspólnego dowodu trafności — krzywa portfela i założenia kosztowe (WIA-50).

Mapowanie `ResolvedPrediction` → trade'y domeny: chronologicznie, z kosztem
dobranym do symbolu (krypto ma własny koszt), plus copy założeń dla maila
(polski zapis liczb) i publicznego scorecardu (kropka)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.application.report_evidence import (
    PORTFOLIO_ASSUMPTIONS_PL,
    PaperCosts,
    paper_pnl_label,
    portfolio_assumptions,
    portfolio_curve,
)
from src.application.report_formatting import DOT_NUMBERS
from src.application.report_models import ResolvedPrediction

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
