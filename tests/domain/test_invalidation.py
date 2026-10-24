"""ML-27: poziom unieważnienia tezy — „co by zmieniło zdanie”.

Teza BULLISH upada poniżej poziomu pod ceną, BEARISH — powyżej poziomu nad nią;
SIDEWAYS ma jeden poziom po dowolnej stronie. Poziom dalej niż 15% od ceny albo
po złej stronie to zmyślone „wsparcie”, nie falsyfikowalna część predykcji."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.invalidation import (
    INVALIDATION_BAND,
    is_thesis_invalidated,
    is_valid_invalidation,
)
from src.domain.prediction import TrendDirection

PRICE = Decimal("120")


class TestValidInvalidation:
    @pytest.mark.parametrize(
        ("trend", "level", "valid"),
        [
            (TrendDirection.BULLISH, "118", True),
            (TrendDirection.BULLISH, "125", False),
            (TrendDirection.BEARISH, "125", True),
            (TrendDirection.BEARISH, "118", False),
            (TrendDirection.SIDEWAYS, "118", True),
            (TrendDirection.SIDEWAYS, "125", True),
        ],
    )
    def test_level_sits_on_the_side_that_breaks_the_thesis(
        self, trend: TrendDirection, level: str, valid: bool
    ) -> None:
        assert is_valid_invalidation(trend, PRICE, Decimal(level)) is valid

    @pytest.mark.parametrize("level", ["101", "139", "0", "-5", "120"])
    def test_level_outside_the_band_or_at_the_price_is_invalid(self, level: str) -> None:
        trend = TrendDirection.BULLISH if Decimal(level) < PRICE else TrendDirection.BEARISH

        assert is_valid_invalidation(trend, PRICE, Decimal(level)) is False

    def test_band_edge_is_still_valid(self) -> None:
        edge = PRICE * (1 - INVALIDATION_BAND)

        assert is_valid_invalidation(TrendDirection.BULLISH, PRICE, edge) is True

    def test_no_reference_price_no_valid_level(self) -> None:
        assert is_valid_invalidation(TrendDirection.BULLISH, Decimal(0), Decimal("1")) is False


class TestThesisInvalidated:
    def test_level_below_the_entry_breaks_on_a_drop_to_it(self) -> None:
        assert is_thesis_invalidated(PRICE, Decimal("118"), Decimal("118")) is True
        assert is_thesis_invalidated(PRICE, Decimal("118"), Decimal("118.5")) is False

    def test_level_above_the_entry_breaks_on_a_rise_to_it(self) -> None:
        assert is_thesis_invalidated(PRICE, Decimal("125"), Decimal("126")) is True
        assert is_thesis_invalidated(PRICE, Decimal("125"), Decimal("121")) is False
