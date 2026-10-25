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
from src.domain.value_objects import Ratio


def peak_from_history(prices: Sequence[Decimal]) -> Decimal:
    """Szczyt (max) z historii cen. Pusta historia → 0 (brak referencji).

    Zwracamy 0 zamiast rzucać, żeby warstwa use case mogła gładko pominąć
    symbol bez snapshotów — `evaluate_drawdown` traktuje peak=0 jak cold-start.
    """
    if not prices:
        return Decimal("0")
    return max(prices)


def drawdown_from_returns(returns: Sequence[float]) -> float:
    """PORT-04: bieżące obsunięcie od szczytu krzywej kapitału ze zwrotów.

    Krzywa startuje od 1.0 (start to pierwszy szczyt) i rośnie o kolejne
    zwroty. Wynik to DODATNIA głębokość ostatniego punktu pod szczytem
    (0.2 = 20% niżej), 0.0 na nowym szczycie albo bez zwrotów.
    """
    equity = 1.0
    peak = 1.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
    return 1.0 - equity / peak


@dataclass(frozen=True)
class DrawdownSignal:
    symbol: str
    current_price: Decimal
    peak_price: Decimal

    @property
    def drawdown_fraction(self) -> Ratio:
        """Spadek od szczytu jako UŁAMEK (`Ratio`, ARCH-04):
        (current - peak) / peak (<= 0), np. -0.12 = 12% pod szczytem.

        Cold-start guard: gdy peak=0 zwraca 0 (brak referencji), żeby nie
        wywoływać alertu, kiedy symbol nie ma jeszcze historii snapshotów.
        """
        if self.peak_price == 0:
            return Ratio(Decimal("0"))
        return Ratio((self.current_price - self.peak_price) / self.peak_price)

    def evaluate_drawdown(
        self,
        elevated_threshold: Ratio,
        critical_threshold: Ratio,
    ) -> MacroAlertLevel:
        """Mapuje głębokość spadku od szczytu na poziom alertu.

        Progi to DODATNIE ułamki (`Ratio`, np. 0.10 = -10%). Porównujemy je z
        głębokością spadku (`-drawdown_fraction`, bo drawdown jest <= 0).
        Negatywne progi i odwrócona kolejność (elevated >= critical) to błąd
        konfiguracji — zgłaszamy `ValueError`, analogicznie do `polish_macro`.
        """
        if elevated_threshold < 0 or critical_threshold < 0:
            raise ValueError(
                "drawdown thresholds must be non-negative "
                f"(elevated={elevated_threshold}, critical={critical_threshold})"
            )
        if elevated_threshold >= critical_threshold:
            raise ValueError(
                "elevated_threshold must be strictly below critical_threshold "
                f"({elevated_threshold} >= {critical_threshold})"
            )

        depth = -self.drawdown_fraction
        if depth >= critical_threshold:
            return MacroAlertLevel.CRITICAL
        if depth >= elevated_threshold:
            return MacroAlertLevel.ELEVATED
        return MacroAlertLevel.NORMAL
