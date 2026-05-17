from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


@dataclass(frozen=True)
class Money:
    amount: Decimal


@dataclass(frozen=True)
class Threshold:
    value: Decimal


class AssetType(Enum):
    STOCK = "STOCK"
    ETF = "ETF"
    OTHER = "OTHER"


class ValuationVerdict(Enum):
    UNDERVALUED = "UNDERVALUED"
    FAIR = "FAIR"
    OVERVALUED = "OVERVALUED"
    UNKNOWN = "UNKNOWN"


# TTL danych fundamentalnych w cache (slow loop odświeża raz dziennie).
FUNDAMENTALS_CACHE_TTL_HOURS = 24


@dataclass(frozen=True)
class Fundamentals:
    trailing_pe: float | None
    forward_pe: float | None
    peg_ratio: float | None
    eps_growth_yoy: float | None
    fetched_at: datetime
