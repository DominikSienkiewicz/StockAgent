from dataclasses import dataclass, field
from decimal import Decimal

from src.domain.value_objects import (
    AssetType,
    Fundamentals,
    Money,
    Threshold,
    ValuationVerdict,
)

# Progi heurystyki wyceny — jawnie nazwane, łatwe do strojenia w testach.
PEG_UNDERVALUED_THRESHOLD = 1.0
PEG_OVERVALUED_THRESHOLD = 2.0
FORWARD_PE_HIGH = 30.0
EPS_GROWTH_LOW = 0.10


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
    asset_type: AssetType = field(default=AssetType.STOCK)

    def evaluate_volatility(self, delta: PriceDelta, threshold: Threshold) -> bool:
        return abs(delta.percentage) >= threshold.value

    def evaluate_valuation(
        self, fundamentals: Fundamentals | None
    ) -> ValuationVerdict:
        if self.asset_type is not AssetType.STOCK:
            return ValuationVerdict.UNKNOWN
        if fundamentals is None or fundamentals.peg_ratio is None:
            return ValuationVerdict.UNKNOWN

        peg = fundamentals.peg_ratio
        forward = fundamentals.forward_pe
        trailing = fundamentals.trailing_pe
        growth = fundamentals.eps_growth_yoy or 0.0

        if (
            peg < PEG_UNDERVALUED_THRESHOLD
            and forward is not None
            and trailing is not None
            and forward < trailing
        ):
            return ValuationVerdict.UNDERVALUED

        if peg > PEG_OVERVALUED_THRESHOLD or (
            forward is not None
            and forward > FORWARD_PE_HIGH
            and growth < EPS_GROWTH_LOW
        ):
            return ValuationVerdict.OVERVALUED

        return ValuationVerdict.FAIR
