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
    # `fraction` trzyma UŁAMEK zmiany ceny (np. 0.02 dla +2%), NIE procent.
    # Dawna nazwa `percentage` była myląca i groziła 100x błędem (finding #33).
    fraction: Decimal

    @property
    def percent(self) -> Decimal:
        """Zmiana wyrażona w procentach (fraction * 100). Wygodny przelicznik
        dla warstwy prezentacji — nie używać do bramki volatility, która
        operuje na ułamku."""
        return self.fraction * 100

    @property
    def percentage(self) -> Decimal:
        """DEPRECATED alias na `fraction` (zwraca UŁAMEK, nie procent).

        Zachowany wyłącznie dla zewnętrznych czytelników poza tym pakietem
        (np. `application/agent_graph.py` czyta `delta.percentage` jako ułamek).
        Nowy kod powinien używać `fraction` (ułamek) lub `percent` (procent).
        """
        return self.fraction

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
        # Bramka operuje na UŁAMKU (fraction), nie procencie — próg też jest
        # ułamkiem (np. 0.02 = 2%).
        return abs(delta.fraction) >= threshold.value

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
