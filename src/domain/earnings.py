"""Domena: kalendarz wyników (earnings) — okno zdarzeń wokół raportu spółki.

Czysta logika: bliskość najbliższego raportu wyników klasyfikuje, czy jesteśmy
w „oknie zdarzenia". Może REDUKOWAĆ spend (np. wstrzymać kosztowną analizę tuż
przed earnings, gdy ruch i tak będzie zdominowany przez zaskoczenie wynikami)
albo podbić niepewność prognozy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

# FIN-11: ważność wpisu `earnings_cache`. Slow Loop odświeża cache w niedzielę
# o 03:00 UTC, Fast Loop biegnie pon–pt: 6 dni pokrywa cały tydzień cykli, a wpis
# z poprzedniej niedzieli jest już stary, więc kolejne odświeżenie pyta AV od nowa.
EARNINGS_CACHE_TTL_HOURS = 144
# `date.weekday()` niedzieli — koniec tygodnia sekcji „📅 Ten tydzień” (DAT-20).
_SUNDAY = 6


class EarningsProximity(Enum):
    """Jak blisko najbliższego raportu wyników jesteśmy."""

    IMMINENT = "IMMINENT"   # raport w ciągu paru dni
    NEAR = "NEAR"           # raport w bieżącym tygodniu/oknie
    FAR = "FAR"             # daleko albo brak danych


class EarningsTiming(Enum):
    """DAT-20: pora publikacji wyników względem sesji (pole `hour` Finnhuba)."""

    BEFORE_OPEN = "bmo"
    AFTER_CLOSE = "amc"
    DURING_MARKET = "dmh"


# DAT-20: pora publikacji po polsku — sekcja „📅 Ten tydzień” i lead maila.
_TIMING_PL: dict[EarningsTiming, str] = {
    EarningsTiming.BEFORE_OPEN: "przed sesją",
    EarningsTiming.AFTER_CLOSE: "po sesji",
    EarningsTiming.DURING_MARKET: "w trakcie sesji",
}


def earnings_timing_label(timing: EarningsTiming) -> str:
    """„przed sesją” / „po sesji” / „w trakcie sesji”."""
    return _TIMING_PL[timing]


@dataclass(frozen=True)
class EarningsEvent:
    """Najbliższe znane zdarzenie wyników dla symbolu.

    `days_until` to liczba dni do raportu (None = brak danych / brak nadchodzącego
    raportu). `report_date` to surowa data ISO (do prezentacji). DAT-20: `timing`
    (przed / po sesji) i `eps_estimate` niesie kalendarz Finnhub; None = nieznane."""

    symbol: str
    days_until: int | None
    report_date: str | None = None
    timing: EarningsTiming | None = None
    eps_estimate: float | None = None

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


@dataclass(frozen=True)
class CachedEarningsDate:
    """FIN-11: data najbliższego raportu z cache Slow Loopa.

    Cache trzyma datę, nie `days_until` — liczba dni zmienia się codziennie, więc
    liczy ją dopiero cykl, który czyta wpis (`event`)."""

    symbol: str
    report_date: date
    fetched_at: datetime

    def event(self, today: date) -> EarningsEvent | None:
        """Zdarzenie dla cyklu z dnia `today`; None, gdy raport już się odbył."""
        days_until = (self.report_date - today).days
        if days_until < 0:
            return None
        return EarningsEvent(
            symbol=self.symbol,
            days_until=days_until,
            report_date=self.report_date.isoformat(),
        )


# Mnożniki progu volatility per bliskość earnings. Im bliżej raportu, tym
# mocniej zacieśniamy bramkę (mniej płatnych wywołań na ruchu, który i tak
# zdominuje zaskoczenie wynikami). FAR = neutralnie (brak wpływu).
_THRESHOLD_MULTIPLIER = {
    EarningsProximity.IMMINENT: 1.5,
    EarningsProximity.NEAR: 1.15,
    EarningsProximity.FAR: 1.0,
}


def earnings_threshold_multiplier(proximity: EarningsProximity) -> float:
    """Mnożnik progu volatility dla bliskości earnings.

    Niezmiennik FinOps (kontrakt jak `RegimeDetector.threshold_multiplier`):
    wynik jest ZAWSZE >= 1.0 — reguła earnings może tylko ZACIEŚNIĆ bramkę
    volatility (podnieść próg, zredukować spend), nigdy jej poluzować. Typ
    `float` dla spójności z `regime_multiplier` (iloczyn w `_pick_threshold`).
    """
    return _THRESHOLD_MULTIPLIER[proximity]


def earnings_this_week(
    events: Iterable[EarningsEvent | None], today: date
) -> tuple[EarningsEvent, ...]:
    """DAT-20: raporty od `today` do niedzieli jego tygodnia, wg dnia i symbolu.
    Zdarzenie bez daty, raport miniony i brak zdarzenia (None) odpadają."""
    days_left = _SUNDAY - today.weekday()
    week = [
        event
        for event in events
        if event is not None
        and event.days_until is not None
        and 0 <= event.days_until <= days_left
    ]
    return tuple(sorted(week, key=lambda event: (event.days_until or 0, event.symbol)))


# DAT-20 (decyzja 2026-10-25): lead „wyniki spółki, którą masz” obejmuje raporty
# dziś (mail idzie przed otwarciem NYSE) i jutro.
HELD_EARNINGS_LEAD_MAX_DAYS = 1


def earnings_due_within(
    events: Iterable[EarningsEvent | None], max_days_until: int = HELD_EARNINGS_LEAD_MAX_DAYS
) -> tuple[EarningsEvent, ...]:
    """DAT-20: raporty od dziś do `max_days_until` dni naprzód, wg dnia i symbolu.
    Zdarzenie bez daty, raport miniony i brak zdarzenia (None) odpadają."""
    due = [
        event
        for event in events
        if event is not None
        and event.days_until is not None
        and 0 <= event.days_until <= max_days_until
    ]
    return tuple(sorted(due, key=lambda event: (event.days_until or 0, event.symbol)))
