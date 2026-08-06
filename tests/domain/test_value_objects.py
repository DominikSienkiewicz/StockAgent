from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.domain.value_objects import (
    FUNDAMENTALS_CACHE_TTL_HOURS,
    AssetType,
    Fundamentals,
    Money,
    Threshold,
    ValuationVerdict,
)


class TestMoney:
    def test_creates_money_with_decimal_amount(self) -> None:
        money = Money(Decimal("100.50"))
        assert money.amount == Decimal("100.50")

    def test_is_immutable(self) -> None:
        money = Money(Decimal("100.00"))
        with pytest.raises(FrozenInstanceError):
            money.amount = Decimal("200.00")  # type: ignore[misc]

    def test_two_money_with_same_amount_are_equal(self) -> None:
        # Dwie ODRĘBNE instancje o tej samej kwocie — równość ma iść po
        # wartości, nie po tożsamości obiektu.
        money = Money(Decimal("50.0"))
        same_amount = Money(Decimal("50.0"))
        assert money == same_amount

    def test_different_amounts_are_not_equal(self) -> None:
        assert Money(Decimal("50.0")) != Money(Decimal("51.0"))

    def test_allows_negative_amount(self) -> None:
        # Money reprezentuje też delty/zmiany cen — ujemne są legalne.
        assert Money(Decimal("-12.34")).amount == Decimal("-12.34")

    def test_rejects_nan(self) -> None:
        nan = Decimal("NaN")
        with pytest.raises(ValueError, match="NaN|finite"):
            Money(nan)

    def test_rejects_infinity(self) -> None:
        infinity = Decimal("Infinity")
        with pytest.raises(ValueError, match="Inf|finite"):
            Money(infinity)


class TestThreshold:
    def test_creates_threshold_with_decimal_value(self) -> None:
        threshold = Threshold(Decimal("0.02"))
        assert threshold.value == Decimal("0.02")

    def test_is_immutable(self) -> None:
        threshold = Threshold(Decimal("0.02"))
        with pytest.raises(FrozenInstanceError):
            threshold.value = Decimal("0.05")  # type: ignore[misc]

    def test_allows_zero(self) -> None:
        # 0 = "always run" — legalne, tylko ujemne jest footgunem.
        assert Threshold(Decimal("0")).value == Decimal("0")

    def test_rejects_negative_value(self) -> None:
        # Ujemny próg → abs(delta) >= value ZAWSZE prawdziwe → bramka
        # volatility cicho wyłączona.
        negative = Decimal("-1")
        with pytest.raises(ValueError, match="non-negative|negative"):
            Threshold(negative)

    def test_rejects_nan(self) -> None:
        nan = Decimal("NaN")
        with pytest.raises(ValueError, match="NaN|finite"):
            Threshold(nan)

    def test_rejects_infinity(self) -> None:
        infinity = Decimal("Infinity")
        with pytest.raises(ValueError, match="Inf|finite"):
            Threshold(infinity)


def test_asset_type_members() -> None:
    assert AssetType.STOCK.value == "STOCK"
    assert AssetType.ETF.value == "ETF"
    assert AssetType.OTHER.value == "OTHER"


def test_valuation_verdict_members() -> None:
    assert {v.name for v in ValuationVerdict} == {
        "UNDERVALUED", "FAIR", "OVERVALUED", "UNKNOWN"
    }


def test_fundamentals_is_frozen_dataclass() -> None:
    f = Fundamentals(
        trailing_pe=25.0,
        forward_pe=20.0,
        peg_ratio=1.2,
        eps_growth_yoy=0.15,
        fetched_at=datetime(2026, 5, 17, tzinfo=UTC),
    )
    with pytest.raises(FrozenInstanceError):
        f.trailing_pe = 30.0  # type: ignore[misc]


def test_fundamentals_allows_none_fields() -> None:
    f = Fundamentals(
        trailing_pe=None,
        forward_pe=None,
        peg_ratio=None,
        eps_growth_yoy=None,
        fetched_at=datetime(2026, 5, 17, tzinfo=UTC),
    )
    assert f.trailing_pe is None


def test_ttl_constant() -> None:
    assert FUNDAMENTALS_CACHE_TTL_HOURS == 24
