"""Reguła deadmana (NIEZ-11): czy job, który powinien chodzić, dał znak życia.

Brak maila wygląda identycznie jak spokojny dzień, więc o cichej śmierci jobu mówi
osobny watchdog. Tu żyje wyłącznie reguła — skąd brać ślady życia i jak alarmować,
decyduje entry-point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LivenessExpectation:
    """Oczekiwanie wobec jobu: maksymalna przerwa między śladami życia w godzinach.
    `max_gap_hours <= 0` oznacza job świadomie wstrzymany — nigdy nie jest spóźniony."""

    job: str
    max_gap_hours: float


def is_overdue(last_seen: datetime | None, now: datetime, max_gap_hours: float) -> bool:
    """Spóźniony = oczekiwany job bez śladu albo z ostatnim śladem starszym niż przerwa."""
    if max_gap_hours <= 0:
        return False
    if last_seen is None:
        return True
    return (now - last_seen).total_seconds() / 3600.0 > max_gap_hours


def overdue_jobs(
    expectations: Sequence[LivenessExpectation],
    last_seen: Mapping[str, datetime | None],
    now: datetime,
) -> tuple[str, ...]:
    """Nazwy spóźnionych jobów, w kolejności oczekiwań."""
    return tuple(
        e.job for e in expectations if is_overdue(last_seen.get(e.job), now, e.max_gap_hours)
    )
