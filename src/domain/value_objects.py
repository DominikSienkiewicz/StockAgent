from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Money:
    amount: Decimal


@dataclass(frozen=True)
class Threshold:
    value: Decimal
