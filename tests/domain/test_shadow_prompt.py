"""EXP-26: shadow mode wariantu promptu albo modelu — reguły domeny.

Stała podpróbka symboli (ta sama w każdym cyklu, bez ręcznego wyboru), kontrakt
prognozy cienia zgodny z `shadow_predictions` i rozliczenie TĄ SAMĄ regułą co
produkcja (`Prediction.is_trend_correct` z pasmem klasy aktywa, jak reflect).
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from src.domain.prediction import Prediction, TrendDirection
from src.domain.shadow_prompt import (
    SHADOW_PROMPT_MAX_PER_CYCLE,
    ShadowTrendForecast,
    in_shadow_sample,
    variant_version,
)
from src.domain.value_objects import AssetType

_VERSION = "a" * 64
_SYMBOLS = [f"S{index:03d}" for index in range(400)]


def _forecast(trend: TrendDirection, confidence: float = 0.7) -> ShadowTrendForecast:
    return ShadowTrendForecast("llm_variant", _VERSION, trend, confidence)


class TestShadowSample:
    def test_membership_is_stable_across_calls(self) -> None:
        first = [in_shadow_sample("llm_variant", symbol, 0.5) for symbol in _SYMBOLS]
        second = [in_shadow_sample("llm_variant", symbol, 0.5) for symbol in _SYMBOLS]

        assert first == second

    def test_share_sets_the_size_of_the_sample(self) -> None:
        picked = sum(in_shadow_sample("llm_variant", symbol, 0.5) for symbol in _SYMBOLS)

        assert 150 <= picked <= 250

    def test_full_share_takes_every_symbol(self) -> None:
        assert all(in_shadow_sample("llm_variant", symbol, 1.0) for symbol in _SYMBOLS)

    def test_variant_name_draws_an_independent_sample(self) -> None:
        one = {s for s in _SYMBOLS if in_shadow_sample("variant_a", s, 0.5)}
        other = {s for s in _SYMBOLS if in_shadow_sample("variant_b", s, 0.5)}

        assert one != other

    @pytest.mark.parametrize("share", [0.0, -0.1, 1.5, math.nan])
    def test_share_outside_the_unit_interval_is_rejected(self, share: float) -> None:
        with pytest.raises(ValueError, match="share"):
            in_shadow_sample("llm_variant", "AAPL", share)


def test_hard_cap_is_five_calls_per_cycle() -> None:
    assert SHADOW_PROMPT_MAX_PER_CYCLE == 5


class TestVariantVersion:
    def test_is_a_sha256_of_the_identity_independent_of_key_order(self) -> None:
        one = variant_version({"model": "gpt-5-mini", "prompt_sha256": "b" * 64})
        same = variant_version({"prompt_sha256": "b" * 64, "model": "gpt-5-mini"})

        assert one == same
        assert len(one) == 64 and int(one, 16) >= 0

    def test_changes_with_any_part_of_the_identity(self) -> None:
        base = variant_version({"model": "gpt-5-mini", "prompt_sha256": "b" * 64})

        assert base != variant_version({"model": "gpt-5", "prompt_sha256": "b" * 64})


class TestShadowTrendForecastContract:
    def test_accepts_a_trend_with_unit_confidence(self) -> None:
        forecast = _forecast(TrendDirection.BULLISH, 0.0)

        assert forecast.predicted_trend is TrendDirection.BULLISH

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, math.nan, math.inf])
    def test_rejects_confidence_outside_the_unit_interval(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _forecast(TrendDirection.BULLISH, confidence)

    @pytest.mark.parametrize(
        ("name", "version"), [("", _VERSION), ("llm_variant", "cold_start"), ("x", "A" * 64)]
    )
    def test_rejects_a_missing_name_or_a_version_that_is_not_a_sha256(
        self, name: str, version: str
    ) -> None:
        with pytest.raises(ValueError):
            ShadowTrendForecast(name, version, TrendDirection.SIDEWAYS, 0.5)


class TestSettlementMatchesProduction:
    """Wariant rozlicza się tą samą regułą, co predykcja produkcyjna w reflect."""

    @pytest.mark.parametrize(
        ("trend", "actual", "asset_type"),
        [
            (TrendDirection.BULLISH, "101", AssetType.STOCK),
            (TrendDirection.BULLISH, "99", AssetType.STOCK),
            (TrendDirection.BEARISH, "99", None),
            (TrendDirection.SIDEWAYS, "100.4", AssetType.STOCK),
            (TrendDirection.SIDEWAYS, "102", AssetType.STOCK),
            (TrendDirection.SIDEWAYS, "102", AssetType.CRYPTO),
        ],
    )
    def test_same_verdict_as_a_production_prediction_with_that_trend(
        self, trend: TrendDirection, actual: str, asset_type: AssetType | None
    ) -> None:
        from src.domain.prediction import sideways_tolerance_for

        price = Decimal("100")
        production = Prediction("AAPL", trend, price, Decimal("100.5"))

        expected = production.is_trend_correct(
            Decimal(actual), tolerance=sideways_tolerance_for(asset_type)
        )

        assert (
            _forecast(trend).is_correct(
                price_at_prediction=price, actual_price=Decimal(actual), asset_type=asset_type
            )
            is expected
        )

    def test_a_hit_and_a_miss_are_told_apart(self) -> None:
        forecast = _forecast(TrendDirection.BULLISH)

        assert forecast.is_correct(
            price_at_prediction=Decimal("100"), actual_price=Decimal("101"), asset_type=None
        )
        assert not forecast.is_correct(
            price_at_prediction=Decimal("100"), actual_price=Decimal("99"), asset_type=None
        )
