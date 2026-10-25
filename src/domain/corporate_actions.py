"""Domena: podejrzenie splitu z `pc` notowania (DAT-23).

Split 10:1 bez korekty to dla systemu ruch −90%: reflect oceniłby go jako trafny
BEARISH, krzywa kapitału pokazałaby zysk, a cecha `price_delta` = −0,9 zatrułaby
trening. Cykl Fast Loopa idzie przed otwarciem, więc `pc` z /quote to zamknięcie
sesji sprzed ostatniej — dokładnie cena poprzedniego snapshotu. Stosunek
`pc / snapshot` daleko od 1 nie jest ruchem ceny (ten widać w `c / pc`), tylko
korektą historii przez dostawcę: przy stosunku bliskim stosunkowi splitu mamy
podejrzenie splitu.

`ratio` to mnożnik ceny po splicie względem ceny przed nim (10:1 → 0,1, odwrotny
1:2 → 2). Zero importów zewnętrznych.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum

# Stosunki splitów spotykane na watchliście (katalog DAT-23): 2:1, 3:1, 4:1, 5:1,
# 10:1 i odwrotne 1:2, 1:3.
SPLIT_RATIOS: tuple[Decimal, ...] = (
    *(Decimal(1) / Decimal(shares) for shares in (2, 3, 4, 5, 10)),
    Decimal(2),
    Decimal(3),
)
# Względna tolerancja dopasowania `pc / snapshot` do stosunku splitu (±2%).
SPLIT_TOLERANCE = Decimal("0.02")


class CorporateActionKind(StrEnum):
    """Rodzaj zdarzenia w `corporate_actions.kind` (migracja 062)."""

    SPLIT_SUSPECTED = "split_suspected"


@dataclass(frozen=True)
class SplitSuspicion:
    """Podejrzenie splitu: `prev_close` z notowania i `reference_price` z ostatniego
    snapshotu dały stosunek bliski `ratio`. Chwila wykrycia jest aware (UTC)."""

    symbol: str
    ratio: Decimal
    prev_close: Decimal
    reference_price: Decimal
    detected_at: datetime
    kind: CorporateActionKind = CorporateActionKind.SPLIT_SUSPECTED

    @property
    def detected_on(self) -> date:
        """Dzień UTC wykrycia — klucz idempotencji zapisu (powtórka cyklu go nie dubluje)."""
        moment = self.detected_at
        aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
        return aware.astimezone(UTC).date()


def detect_split(
    symbol: str,
    prev_close: Decimal,
    reference_price: Decimal,
    *,
    at: datetime,
    tolerance: Decimal = SPLIT_TOLERANCE,
) -> SplitSuspicion | None:
    """Podejrzenie splitu, gdy `prev_close / reference_price` leży w ±`tolerance`
    (względnie) od stosunku z `SPLIT_RATIOS`; inaczej None. Cena niedodatnia nie
    daje stosunku, więc nie flaguje."""
    if prev_close <= 0 or reference_price <= 0:
        return None
    observed = prev_close / reference_price
    for ratio in SPLIT_RATIOS:
        if abs(observed / ratio - 1) <= tolerance:
            return SplitSuspicion(
                symbol=symbol,
                ratio=ratio,
                prev_close=prev_close,
                reference_price=reference_price,
                detected_at=at,
            )
    return None


def to_pre_split_price(price: Decimal, ratio: Decimal) -> Decimal:
    """Cena po splicie w jednostkach sprzed niego (`price / ratio`).

    Rozliczenie predykcji sprzed splitu tą ceną jest równoważne przeskalowaniu
    `price_at_prediction` i celu przez `ratio`, a zapisana cena rozliczenia zostaje
    w jednostkach `price_at_prediction` — zwrot liczony z kolumn bazy jest poprawny."""
    return price / ratio
