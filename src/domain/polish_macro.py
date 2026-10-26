"""Domena polskiego makro — proxy ryzyka suwerennego przez kurs PLN.

Wzrost EUR/PLN i USD/PLN w 30 dni = osłabienie PLN = sygnał odpływu
kapitału zagranicznego i potencjalna presja na rating. Skoncentrowane
na 30-dniowym oknie, bo to horyzont reagowania funduszy macro, nie
dzienna fluktuacja.

DAT-29: dzienna zmiana kursu (dwa ostatnie notowania tabeli A) idzie obok —
dla inwestora w USD ruch kursu jest równie ważny jak ruch akcji. Tylko do
prezentacji (Risk Watch): stres fiskalny liczy wyłącznie okno 30 dni.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from src.domain.value_objects import Ratio


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
    """Kursy referencyjne NBP i ich zmiana w oknie 30 dni.

    `eur_pln_30d_change_ratio` i `usd_pln_30d_change_ratio` to zmiany kursu jako
    UŁAMEK (`Ratio`: 0.02 = +2%, ARCH-04) — tak jak progi stresu.
    `*_1d_change_ratio` (DAT-29) — zmiana od poprzedniego notowania tabeli A;
    None = nieznana (snapshot zbudowany bez szeregu kursów)."""

    eur_pln: Decimal
    usd_pln: Decimal
    eur_pln_30d_change_ratio: Ratio
    usd_pln_30d_change_ratio: Ratio
    fetched_at: datetime
    eur_pln_1d_change_ratio: Ratio | None = None
    usd_pln_1d_change_ratio: Ratio | None = None

    @classmethod
    def from_rates(
        cls, *, eur: Sequence[Decimal], usd: Sequence[Decimal], fetched_at: datetime
    ) -> PolishMacroSnapshot | None:
        """Snapshot z kursów średnich tabeli A, od najstarszego do najświeższego.

        Zmiana 30d liczy się od pierwszego notowania okna, dzienna — od
        przedostatniego. Tabela A wychodzi w dni robocze, więc „d/d” w poniedziałek
        to zmiana od piątku. Mniej niż dwa notowania którejś pary albo kurs
        niedodatni → None (brak punktu odniesienia zamiast dzielenia przez zero)."""
        if len(eur) < 2 or len(usd) < 2 or min(*eur, *usd) <= 0:
            return None
        return cls(
            eur_pln=eur[-1],
            usd_pln=usd[-1],
            eur_pln_30d_change_ratio=_change(eur[0], eur[-1]),
            usd_pln_30d_change_ratio=_change(usd[0], usd[-1]),
            fetched_at=fetched_at,
            eur_pln_1d_change_ratio=_change(eur[-2], eur[-1]),
            usd_pln_1d_change_ratio=_change(usd[-2], usd[-1]),
        )

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

        worst = max(self.eur_pln_30d_change_ratio, self.usd_pln_30d_change_ratio)
        if worst >= critical_threshold:
            return MacroStressLevel.CRITICAL
        if worst >= elevated_threshold:
            return MacroStressLevel.ELEVATED
        return MacroStressLevel.NORMAL


def _change(earlier: Decimal, later: Decimal) -> Ratio:
    """Zmiana kursu jako ułamek (`Ratio`, ARCH-04)."""
    return Ratio((later - earlier) / earlier)
