from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from src.domain.value_objects import Money, Threshold


class TestMoney:
    def test_creates_money_with_decimal_amount(self):
        money = Money(Decimal("100.50"))
        assert money.amount == Decimal("100.50")

    def test_is_immutable(self):
        money = Money(Decimal("100.00"))
        with pytest.raises(FrozenInstanceError):
            money.amount = Decimal("200.00")  # type: ignore[misc]

    def test_two_money_with_same_amount_are_equal(self):
        assert Money(Decimal("50.0")) == Money(Decimal("50.0"))

    def test_different_amounts_are_not_equal(self):
        assert Money(Decimal("50.0")) != Money(Decimal("51.0"))


class TestThreshold:
    def test_creates_threshold_with_decimal_value(self):
        threshold = Threshold(Decimal("0.02"))
        assert threshold.value == Decimal("0.02")

    def test_is_immutable(self):
        threshold = Threshold(Decimal("0.02"))
        with pytest.raises(FrozenInstanceError):
            threshold.value = Decimal("0.05")  # type: ignore[misc]
