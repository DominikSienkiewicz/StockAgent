"""Kurs waluty do PLN z danego dnia (PORT-11) — wartość, którą zwraca `FxRatePort`.

`on_date` to dzień, o który pytano (np. data zakupu), `effective_date` — dzień
tabeli NBP, z której pochodzi kurs. W weekend, święto i przed publikacją tabeli
(ok. 12:00) źródło cofa się do poprzedniego dnia roboczego, więc
`effective_date <= on_date`. Zero I/O, tylko stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from src.domain.portfolio import is_iso_currency


@dataclass(frozen=True)
class FxRate:
    """Kurs średni: PLN za 1 jednostkę `currency` (kod ISO 4217)."""

    currency: str
    on_date: date
    effective_date: date
    rate: Decimal

    def __post_init__(self) -> None:
        if not is_iso_currency(self.currency):
            raise ValueError(f"currency must be an ISO 4217 code (got {self.currency!r})")
        if not self.rate.is_finite() or self.rate <= 0:
            raise ValueError(f"rate must be a positive finite number (got {self.rate})")
        if self.effective_date > self.on_date:
            raise ValueError(
                f"effective_date {self.effective_date} is after on_date {self.on_date}"
            )
