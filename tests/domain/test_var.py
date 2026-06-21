import math

import pytest

from src.domain.var import conditional_var, historical_var


class TestHistoricalVar:
    """Empiryczny Value-at-Risk: strata w (1-confidence) dolnym ogonie rozkładu
    zwrotów, z liniową interpolacją między statystykami pozycyjnymi, zwracana
    jako DODATNIA magnituda straty."""

    def test_hand_computed_var_95(self):
        # Zwroty posortowane: [-0.10, -0.05, 0.0, 0.02, 0.08], n=5.
        # Pozycja interpolacji h = (1-0.95)*(n-1) = 0.05*4 = 0.2.
        # Kwantyl = sorted[0] + 0.2*(sorted[1]-sorted[0])
        #         = -0.10 + 0.2*(0.05) = -0.09.
        # VaR = max(0, -(-0.09)) = 0.09.
        returns = [-0.10, -0.05, 0.0, 0.02, 0.08]
        assert historical_var(returns, 0.95) == pytest.approx(0.09)

    def test_unsorted_input_is_sorted_internally(self):
        # Ta sama próbka, ale w losowej kolejności — wynik identyczny.
        returns = [0.08, -0.05, 0.02, -0.10, 0.0]
        assert historical_var(returns, 0.95) == pytest.approx(0.09)

    def test_var_is_positive_loss_magnitude(self):
        # Same straty → VaR dodatni.
        returns = [-0.03, -0.07, -0.12, -0.20]
        var = historical_var(returns, 0.95)
        assert var > 0.0

    def test_all_gains_clamps_to_zero(self):
        # Brak strat w ogonie — kwantyl dodatni, więc VaR = max(0, -kwantyl) = 0.
        returns = [0.01, 0.02, 0.03, 0.05, 0.10]
        assert historical_var(returns, 0.95) == 0.0

    def test_empty_returns_zero(self):
        assert historical_var([], 0.95) == 0.0

    def test_single_element_returns_zero(self):
        assert historical_var([-0.5], 0.95) == 0.0

    def test_lower_confidence_gives_smaller_or_equal_tail(self):
        # 90% sięga płycej w ogon niż 99% → strata nie większa.
        returns = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
        var_90 = historical_var(returns, 0.90)
        var_99 = historical_var(returns, 0.99)
        assert var_90 <= var_99

    def test_default_confidence_is_95(self):
        returns = [-0.10, -0.05, 0.0, 0.02, 0.08]
        assert historical_var(returns) == pytest.approx(historical_var(returns, 0.95))

    def test_result_is_finite_float(self):
        var = historical_var([-0.10, -0.05, 0.0, 0.02, 0.08], 0.95)
        assert isinstance(var, float)
        assert math.isfinite(var)

    def test_confidence_out_of_range_clamped(self):
        # confidence poza (0,1) jest clampowane do sensownego zakresu — nie wybucha.
        returns = [-0.10, -0.05, 0.0, 0.02, 0.08]
        assert math.isfinite(historical_var(returns, 0.0))
        assert math.isfinite(historical_var(returns, 1.0))
        assert math.isfinite(historical_var(returns, 1.5))
        assert math.isfinite(historical_var(returns, -0.2))


class TestConditionalVar:
    """CVaR / expected shortfall: DODATNIA średnia magnituda straty w najgorszym
    (1-confidence) ogonie. Musi spełniać CVaR >= VaR."""

    def test_cvar_at_least_var(self):
        returns = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
        var = historical_var(returns, 0.95)
        cvar = conditional_var(returns, 0.95)
        assert cvar >= var

    def test_cvar_at_least_var_for_fixed_sample(self):
        returns = [-0.10, -0.05, 0.0, 0.02, 0.08]
        assert conditional_var(returns, 0.95) >= historical_var(returns, 0.95)

    def test_cvar_is_positive_loss_magnitude(self):
        returns = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05]
        assert conditional_var(returns, 0.90) > 0.0

    def test_cvar_averages_worst_tail(self):
        # Przy 90% i n=10 ogon to najgorszy 1 zwrot — CVaR ≈ |najgorszy|.
        returns = [-0.50, -0.10, -0.05, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        cvar = conditional_var(returns, 0.90)
        assert cvar == pytest.approx(0.50, abs=0.05)

    def test_all_gains_clamps_to_zero(self):
        returns = [0.01, 0.02, 0.03, 0.05, 0.10]
        assert conditional_var(returns, 0.95) == 0.0

    def test_empty_returns_zero(self):
        assert conditional_var([], 0.95) == 0.0

    def test_single_element_returns_zero(self):
        assert conditional_var([-0.5], 0.95) == 0.0

    def test_default_confidence_is_95(self):
        returns = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
        assert conditional_var(returns) == pytest.approx(conditional_var(returns, 0.95))

    def test_result_is_finite_float(self):
        cvar = conditional_var([-0.30, -0.20, -0.10, 0.0, 0.05], 0.90)
        assert isinstance(cvar, float)
        assert math.isfinite(cvar)
