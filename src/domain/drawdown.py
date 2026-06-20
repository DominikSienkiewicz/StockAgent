"""Domena Drawdown — spadek ceny symbolu od jego szczytu z historii snapshotów.

Liczony WYŁĄCZNIE z darmowych snapshotów ceny (`price_snapshots`), bez żadnych
płatnych portów (LLM, Alpha Vantage). Szczyt = max z historii; drawdown =
(current - peak) / peak (wartość <= 0). Poziom alertu mapowany jest na ten sam
enum co macro-risk (`MacroAlertLevel`), żeby raport miał jedną wspólną skalę.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from src.domain.macro_risk import MacroAlertLevel


def peak_from_history(prices: Sequence[Decimal]) -> Decimal:
    """Szczyt (max) z historii cen. Pusta historia → 0 (brak referencji).

    Zwracamy 0 zamiast rzucać, żeby warstwa use case mogła gładko pominąć
    symbol bez snapshotów — `evaluate_drawdown` traktuje peak=0 jak cold-start.
    """
    if not prices:
        return Decimal("0")
    return max(prices)


@dataclass(frozen=True)
class DrawdownSignal:
    symbol: str
    current_price: Decimal
    peak_price: Decimal

    @property
    def drawdown_fraction(self) -> Decimal:
        """Procentowy spadek od szczytu: (current - peak) / peak (<= 0).

        Cold-start guard: gdy peak=0 zwraca 0 (brak referencji), żeby nie
        wywoływać alertu, kiedy symbol nie ma jeszcze historii snapshotów.
        """
        if self.peak_price == 0:
            return Decimal("0")
        return (self.current_price - self.peak_price) / self.peak_price

    def evaluate_drawdown(
        self,
        elevated_pct: Decimal,
        critical_pct: Decimal,
    ) -> MacroAlertLevel:
        """Mapuje głębokość spadku od szczytu na poziom alertu.

        Progi to DODATNIE ułamki (np. 0.10 = -10%). Porównujemy je z
        głębokością spadku (`-drawdown_fraction`, bo drawdown jest <= 0).
        Negatywne progi i odwrócona kolejność (elevated >= critical) to błąd
        konfiguracji — zgłaszamy `ValueError`, analogicznie do `polish_macro`.
        """
        if elevated_pct < 0 or critical_pct < 0:
            raise ValueError(
                "drawdown thresholds must be non-negative "
                f"(elevated={elevated_pct}, critical={critical_pct})"
            )
        if elevated_pct >= critical_pct:
            raise ValueError(
                "elevated_pct must be strictly below critical_pct "
                f"({elevated_pct} >= {critical_pct})"
            )

        depth = -self.drawdown_fraction
        if depth >= critical_pct:
            return MacroAlertLevel.CRITICAL
        if depth >= elevated_pct:
            return MacroAlertLevel.ELEVATED
        return MacroAlertLevel.NORMAL
