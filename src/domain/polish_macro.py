"""Domena polskiego makro — proxy ryzyka suwerennego przez kurs PLN.

Wzrost EUR/PLN i USD/PLN w 30 dni = osłabienie PLN = sygnał odpływu
kapitału zagranicznego i potencjalna presja na rating. Skoncentrowane
na 30-dniowym oknie, bo to horyzont reagowania funduszy macro, nie
dzienna fluktuacja.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class MacroStressLevel(Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


# Domyślne progi 30-dniowej deprecjacji PLN — kalibracja konserwatywna,
# ELEVATED przy ~2% (zwykły szum miesięczny), CRITICAL przy ~5% (sell-off).
_DEFAULT_ELEVATED = Decimal("0.02")
_DEFAULT_CRITICAL = Decimal("0.05")


@dataclass(frozen=True)
class PolishMacroSnapshot:
    eur_pln: Decimal
    usd_pln: Decimal
    eur_pln_30d_change_pct: Decimal
    usd_pln_30d_change_pct: Decimal
    fetched_at: datetime

    def evaluate_stress_level(
        self,
        elevated_threshold: Decimal = _DEFAULT_ELEVATED,
        critical_threshold: Decimal = _DEFAULT_CRITICAL,
    ) -> MacroStressLevel:
        """Zwraca poziom stresu na podstawie max(EUR/PLN, USD/PLN) Δ30d.

        Negatywne progi i odwrócona kolejność (elevated > critical) to
        błąd konfiguracji — zgłaszamy `ValueError` zamiast cicho zwracać
        mylący wynik.
        """
        if elevated_threshold < 0 or critical_threshold < 0:
            raise ValueError(
                "stress thresholds must be non-negative "
                f"(elevated={elevated_threshold}, critical={critical_threshold})"
            )
        if elevated_threshold >= critical_threshold:
            raise ValueError(
                "elevated_threshold must be strictly below critical_threshold "
                f"({elevated_threshold} >= {critical_threshold})"
            )

        worst = max(self.eur_pln_30d_change_pct, self.usd_pln_30d_change_pct)
        if worst >= critical_threshold:
            return MacroStressLevel.CRITICAL
        if worst >= elevated_threshold:
            return MacroStressLevel.ELEVATED
        return MacroStressLevel.NORMAL
