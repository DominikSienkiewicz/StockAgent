import math
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


def clamp_unit(value: float) -> tuple[float, bool]:
    """Przycina wartość do skali jednostkowej [0, 1] (pewność, zgoda).

    Zwraca `(wartość_w_skali, czy_była_poza_skalą)`. LLM dryfujący na skalę
    0–100 zwraca np. 85 — bez przycięcia „8500%” trafia do bazy i na szczyt
    rankingu sygnałów. Nieskończoność przycinamy do granicy; NaN nie ma
    pozycji na osi, więc to `ValueError` (wołający odsiewa go wcześniej).
    """
    if math.isnan(value):
        raise ValueError("clamp_unit requires a number, got NaN")
    clamped = min(max(value, 0.0), 1.0)
    return clamped, clamped != value


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


# TTL danych fundamentalnych w cache. Slow Loop odświeża je raz w tygodniu, więc
# 6 × 24 h: Fast Loop widzi fundamenty przez sześć dni, a niedzielny przebieg
# zawsze trafia na nieświeży cache i odświeża (TTL ≥ 7 × 24 h odświeżałby co dwa tygodnie).
FUNDAMENTALS_CACHE_TTL_HOURS = 144


@dataclass(frozen=True)
class Fundamentals:
    trailing_pe: float | None
    forward_pe: float | None
    peg_ratio: float | None
    eps_growth_yoy: float | None
    fetched_at: datetime
