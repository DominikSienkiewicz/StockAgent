"""Domena: kalendarz wyników (earnings) — okno zdarzeń wokół raportu spółki.

Czysta logika: bliskość najbliższego raportu wyników klasyfikuje, czy jesteśmy
w „oknie zdarzenia". Może REDUKOWAĆ spend (np. wstrzymać kosztowną analizę tuż
przed earnings, gdy ruch i tak będzie zdominowany przez zaskoczenie wynikami)
albo podbić niepewność prognozy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EarningsProximity(Enum):
    """Jak blisko najbliższego raportu wyników jesteśmy."""

    IMMINENT = "IMMINENT"   # raport w ciągu paru dni
    NEAR = "NEAR"           # raport w bieżącym tygodniu/oknie
    FAR = "FAR"             # daleko albo brak danych


@dataclass(frozen=True)
class EarningsEvent:
    """Najbliższe znane zdarzenie wyników dla symbolu.

    `days_until` to liczba dni do raportu (None = brak danych / brak nadchodzącego
    raportu). `report_date` to surowa data ISO (do prezentacji)."""

    symbol: str
    days_until: int | None
    report_date: str | None = None

    def proximity(
        self, imminent_days: int = 2, near_days: int = 7
    ) -> EarningsProximity:
        """Klasyfikuje bliskość raportu. Brak daty / przeszłość → FAR."""
        if self.days_until is None or self.days_until < 0:
            return EarningsProximity.FAR
        if self.days_until <= imminent_days:
            return EarningsProximity.IMMINENT
        if self.days_until <= near_days:
            return EarningsProximity.NEAR
        return EarningsProximity.FAR
