import math
from dataclasses import FrozenInstanceError

import pytest

from src.domain.scenarios import (
    Scenario,
    ScenarioImpact,
    beta_to_portfolio,
    stress_test,
)


class TestBetaToPortfolio:
    """Beta = cov(asset, market) / var(market), populacyjne kowariancja/wariancja
    (stdlib). Flat market → 0.0, nierówne długości → truncacja do krótszej."""

    def test_identical_series_beta_one(self):
        market = [0.01, -0.02, 0.03, -0.01, 0.02]
        assert beta_to_portfolio(market, market) == pytest.approx(1.0)

    def test_scaled_series_beta_two(self):
        market = [0.01, -0.02, 0.03, -0.01, 0.02]
        asset = [2 * r for r in market]
        assert beta_to_portfolio(asset, market) == pytest.approx(2.0)

    def test_inverse_series_beta_negative_one(self):
        market = [0.01, -0.02, 0.03, -0.01, 0.02]
        asset = [-r for r in market]
        assert beta_to_portfolio(asset, market) == pytest.approx(-1.0)

    def test_flat_market_returns_zero(self):
        # Wariancja rynku = 0 → beta 0.0 zamiast dzielenia przez zero.
        market = [0.0, 0.0, 0.0, 0.0]
        asset = [0.01, -0.02, 0.03, -0.01]
        assert beta_to_portfolio(asset, market) == 0.0

    def test_unequal_length_truncates_to_shorter(self):
        # Asset dłuższy o jeden punkt — liczone tylko na wspólnym prefiksie.
        market = [0.01, -0.02, 0.03, -0.01, 0.02]
        asset = [2 * r for r in market] + [0.5]
        assert beta_to_portfolio(asset, market) == pytest.approx(2.0)

    def test_too_few_paired_points_returns_zero(self):
        assert beta_to_portfolio([0.01], [0.02]) == 0.0
        assert beta_to_portfolio([], []) == 0.0

    def test_result_is_finite_float(self):
        beta = beta_to_portfolio([0.02, -0.01, 0.03], [0.01, -0.02, 0.03])
        assert isinstance(beta, float)
        assert math.isfinite(beta)


class TestScenarioDataclasses:
    """Scenario i ScenarioImpact to mrożone value objects."""

    def test_scenario_is_frozen(self):
        scenario = Scenario(name="krach", market_shock=-0.10)
        assert scenario.name == "krach"
        assert scenario.market_shock == -0.10
        with pytest.raises(FrozenInstanceError):
            scenario.market_shock = -0.20  # type: ignore[misc]

    def test_scenario_impact_is_frozen(self):
        impact = ScenarioImpact(
            scenario_name="krach",
            market_shock=-0.10,
            portfolio_pnl=-0.15,
            per_symbol={"NVDA": -0.20, "AAPL": -0.10},
            worst_symbol="NVDA",
            worst_pnl=-0.20,
        )
        assert impact.scenario_name == "krach"
        assert impact.worst_symbol == "NVDA"
        with pytest.raises(FrozenInstanceError):
            impact.portfolio_pnl = 0.0  # type: ignore[misc]


class TestStressTest:
    """Dla każdego symbolu liczymy betę, dla każdego scenariusza
    per_symbol = beta*shock, portfolio_pnl = średnia równoważona, worst_symbol
    = najbardziej ujemny per_symbol."""

    def test_per_symbol_pnl_is_beta_times_shock(self):
        market = [0.01, -0.02, 0.03, -0.01, 0.02]
        returns_by_symbol = {
            "ONE": list(market),  # beta 1.0
            "TWO": [2 * r for r in market],  # beta 2.0
        }
        scenarios = [Scenario(name="krach", market_shock=-0.10)]
        impacts = stress_test(returns_by_symbol, market, scenarios)
        assert len(impacts) == 1
        impact = impacts[0]
        assert impact.per_symbol["ONE"] == pytest.approx(-0.10)
        assert impact.per_symbol["TWO"] == pytest.approx(-0.20)

    def test_portfolio_pnl_is_equal_weight_mean(self):
        market = [0.01, -0.02, 0.03, -0.01, 0.02]
        returns_by_symbol = {
            "ONE": list(market),  # beta 1.0 → -0.10
            "TWO": [2 * r for r in market],  # beta 2.0 → -0.20
        }
        scenarios = [Scenario(name="krach", market_shock=-0.10)]
        impact = stress_test(returns_by_symbol, market, scenarios)[0]
        # Średnia (-0.10, -0.20) = -0.15.
        assert impact.portfolio_pnl == pytest.approx(-0.15)

    def test_worst_symbol_is_most_negative(self):
        market = [0.01, -0.02, 0.03, -0.01, 0.02]
        returns_by_symbol = {
            "ONE": list(market),  # beta 1.0
            "TWO": [2 * r for r in market],  # beta 2.0 → najgorszy przy spadku
        }
        scenarios = [Scenario(name="krach", market_shock=-0.10)]
        impact = stress_test(returns_by_symbol, market, scenarios)[0]
        assert impact.worst_symbol == "TWO"
        assert impact.worst_pnl == pytest.approx(-0.20)

    def test_positive_shock_flips_worst_symbol(self):
        # Przy szoku dodatnim najbardziej ujemny per_symbol ma symbol z ujemną betą.
        market = [0.01, -0.02, 0.03, -0.01, 0.02]
        returns_by_symbol = {
            "LONG": list(market),  # beta 1.0
            "SHORT": [-r for r in market],  # beta -1.0
        }
        scenarios = [Scenario(name="rajd", market_shock=0.10)]
        impact = stress_test(returns_by_symbol, market, scenarios)[0]
        # LONG: +0.10, SHORT: -0.10 → najgorszy SHORT.
        assert impact.worst_symbol == "SHORT"
        assert impact.worst_pnl == pytest.approx(-0.10)

    def test_multiple_scenarios_one_impact_each(self):
        market = [0.01, -0.02, 0.03, -0.01, 0.02]
        returns_by_symbol = {"ONE": list(market)}
        scenarios = [
            Scenario(name="krach", market_shock=-0.10),
            Scenario(name="rajd", market_shock=0.10),
        ]
        impacts = stress_test(returns_by_symbol, market, scenarios)
        assert len(impacts) == 2
        assert [i.scenario_name for i in impacts] == ["krach", "rajd"]

    def test_scenario_metadata_propagated(self):
        market = [0.01, -0.02, 0.03, -0.01, 0.02]
        returns_by_symbol = {"ONE": list(market)}
        scenarios = [Scenario(name="krach", market_shock=-0.10)]
        impact = stress_test(returns_by_symbol, market, scenarios)[0]
        assert impact.scenario_name == "krach"
        assert impact.market_shock == -0.10

    def test_empty_symbols_yields_empty_per_symbol_and_none_worst(self):
        market = [0.01, -0.02, 0.03]
        scenarios = [Scenario(name="krach", market_shock=-0.10)]
        impacts = stress_test({}, market, scenarios)
        assert len(impacts) == 1
        impact = impacts[0]
        assert impact.per_symbol == {}
        assert impact.worst_symbol is None
        assert impact.portfolio_pnl == 0.0

    def test_no_scenarios_yields_empty_list(self):
        market = [0.01, -0.02, 0.03]
        returns_by_symbol = {"ONE": list(market)}
        assert stress_test(returns_by_symbol, market, []) == []

    def test_empty_inputs_graceful(self):
        assert stress_test({}, [], []) == []
