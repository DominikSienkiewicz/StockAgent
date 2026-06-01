"""Domena Risk Watch — sygnały ryzyka makro z instrumentów proxy.

Inverse ETF / VIX / złoto rosną gdy rynek bazowy spada — wzrost ich ceny
oznacza pogarszające się warunki. Sovereign proxy (EPOL dla PL) działa
odwrotnie: spadek wartości = inwestorzy uciekają z kraju.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from src.domain.value_objects import Money


class MacroRiskInstrumentType(Enum):
    INVERSE_EQUITY = "INVERSE_EQUITY"
    INVERSE_TREASURY = "INVERSE_TREASURY"
    SAFE_HAVEN = "SAFE_HAVEN"
    VOLATILITY = "VOLATILITY"
    SOVEREIGN_PROXY = "SOVEREIGN_PROXY"


class MacroAlertLevel(Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


# Domyślne progi — strojone pod naturalną dzienną zmienność hedge'ów (VIX/SH).
_DEFAULT_ELEVATED = Decimal("0.01")
_DEFAULT_CRITICAL = Decimal("0.03")


@dataclass(frozen=True)
class MacroRiskSignal:
    symbol: str
    instrument_type: MacroRiskInstrumentType
    current_price: Money
    previous_price: Money

    @property
    def change_pct(self) -> Decimal:
        """Procentowa zmiana ceny vs poprzedni snapshot.

        Cold-start guard: gdy previous=0 zwraca 0 (brak referencji), żeby
        nie wywoływać alertu pierwszego cyklu.
        """
        if self.previous_price.amount == 0:
            return Decimal("0")
        return (
            self.current_price.amount - self.previous_price.amount
        ) / self.previous_price.amount

    def evaluate_alert(
        self,
        elevated_threshold: Decimal = _DEFAULT_ELEVATED,
        critical_threshold: Decimal = _DEFAULT_CRITICAL,
    ) -> MacroAlertLevel:
        """Mapuje zmianę ceny na poziom alertu z uwzględnieniem kierunku.

        Dla SOVEREIGN_PROXY interpretujemy spadek jako alert (inwersja znaku).
        """
        change = self.change_pct
        if self.instrument_type is MacroRiskInstrumentType.SOVEREIGN_PROXY:
            change = -change

        if change >= critical_threshold:
            return MacroAlertLevel.CRITICAL
        if change >= elevated_threshold:
            return MacroAlertLevel.ELEVATED
        return MacroAlertLevel.NORMAL
