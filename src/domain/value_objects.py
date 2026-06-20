from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


@dataclass(frozen=True)
class Money:
    amount: Decimal

    def __post_init__(self) -> None:
        """Money reprezentuje ceny i delty — dowolna skończona wartość (także
        ujemna) jest legalna. Odrzucamy tylko NaN/Inf, których poprawny kod
        nigdy nie produkuje, a które cicho zatruwają porównania w domenie.
        """
        if self.amount.is_nan() or self.amount.is_infinite():
            raise ValueError(
                f"Money amount must be a finite number (got {self.amount})"
            )


@dataclass(frozen=True)
class Threshold:
    value: Decimal

    def __post_init__(self) -> None:
        """Próg volatility musi być nieujemny i skończony.

        Ujemna wartość sprawia, że `abs(delta) >= value` jest ZAWSZE
        prawdziwe → bramka volatility cicho znika i płatne porty odpalają
        co cykl. `0` jest dozwolone ("always run"). NaN/Inf to błąd
        konfiguracji — zgłaszamy `ValueError` zamiast cicho psuć porównanie.
        """
        if self.value.is_nan() or self.value.is_infinite():
            raise ValueError(
                f"Threshold value must be a finite number (got {self.value})"
            )
        if self.value < 0:
            raise ValueError(
                f"Threshold value must be non-negative (got {self.value})"
            )


class AssetType(Enum):
    STOCK = "STOCK"
    ETF = "ETF"
    # Krypto: 24/7 market, brak EPS/P/E. Wycena fundamentalna nie ma sensu —
    # `Asset.evaluate_valuation` zwraca UNKNOWN tak samo jak dla ETF.
    CRYPTO = "CRYPTO"
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
