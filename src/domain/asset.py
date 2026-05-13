from dataclasses import dataclass
from decimal import Decimal

from src.domain.value_objects import Money, Threshold


@dataclass(frozen=True)
class PriceDelta:
    percentage: Decimal

    @staticmethod
    def calculate(previous: Money, current: Money) -> "PriceDelta":
        if previous.amount == 0:
            return PriceDelta(Decimal("0"))
        delta = (current.amount - previous.amount) / previous.amount
        return PriceDelta(delta)


@dataclass(frozen=True)
class Asset:
    symbol: str

    def evaluate_volatility(self, delta: PriceDelta, threshold: Threshold) -> bool:
        return abs(delta.percentage) >= threshold.value
