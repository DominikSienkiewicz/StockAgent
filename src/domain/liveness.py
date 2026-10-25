"""Reguła deadmana (NIEZ-11): czy job, który powinien chodzić, dał znak życia.

Brak maila wygląda identycznie jak spokojny dzień, więc o cichej śmierci jobu mówi
osobny watchdog. Tu żyje wyłącznie reguła — skąd brać ślady życia i jak alarmować,
decyduje entry-point.

Dwie reguły: godzinowa (maksymalna przerwa między śladami, np. tygodniowy Slow
Loop) i dni handlowych NYSE (job dzienny, który w weekend i święto nie chodzi —
Fast Loop po wznowieniu). Reguła godzinowa przy dziennym Fast Loopie alarmowałaby
po każdym weekendzie i święcie.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum

from src.domain.market_calendar import NYSE_TZ, is_trading_day, to_nyse_time

# NIEZ-11: Fast Loop chodzi w dni handlowe przed otwarciem (cron 05:30 UTC = 00:30 EST
# / 01:30 EDT), więc ślad dnia handlowego liczymy od północy ET tego dnia, a
# `liveness_fast_loop_max_gap_h` to zapas od tej chwili. Weekend i święto NYSE nie
# mają terminu. Tę samą regułę stosuje deadman i publiczna strona statusu (NIEZ-33).
FAST_LOOP_TRADING_DAY_SLOT = time(0, 0)


@dataclass(frozen=True)
class LivenessExpectation:
    """Oczekiwanie wobec jobu: maksymalna przerwa między śladami życia w godzinach.
    `max_gap_hours <= 0` oznacza job świadomie wstrzymany — nigdy nie jest spóźniony.

    `trading_day_slot` (NIEZ-11) przełącza na regułę dni handlowych NYSE: job ma
    zostawić ślad w każdy dzień handlowy nie wcześniej niż o tej godzinie (czas ET),
    a `max_gap_hours` to zapas liczony od niej. None = reguła godzinowa."""

    job: str
    max_gap_hours: float
    trading_day_slot: time | None = None


def is_overdue(last_seen: datetime | None, now: datetime, max_gap_hours: float) -> bool:
    """Spóźniony = oczekiwany job bez śladu albo z ostatnim śladem starszym niż przerwa."""
    if max_gap_hours <= 0:
        return False
    if last_seen is None:
        return True
    return (now - last_seen).total_seconds() / 3600.0 > max_gap_hours


def is_overdue_on_trading_days(
    last_run_at: datetime | None, now: datetime, slot: time, grace: timedelta
) -> bool:
    """Spóźniony job dni handlowych NYSE (NIEZ-11).

    Każdy dzień handlowy D ma termin: `slot` dnia D (czas ET) plus `grace`. Job jest
    spóźniony, gdy termin ostatniego dnia handlowego już minął, a ostatni ślad jest
    starszy niż `slot` tego dnia (albo śladu nie ma). Weekend i święto NYSE nie mają
    terminu, więc nie alarmują. Ślad z późniejszego dnia pokrywa wcześniejszy termin:
    przy `grace` ≥ 24 h jeden opuszczony dzień nie alarmuje, jeśli następny cykl
    przyszedł — alarm oznacza job martwy, nie pojedynczy pominięty przebieg.
    Naiwny `datetime` = UTC."""
    due_slot = _latest_due_slot(to_nyse_time(now), slot, grace)
    return last_run_at is None or to_nyse_time(last_run_at) < due_slot


def _latest_due_slot(now: datetime, slot: time, grace: timedelta) -> datetime:
    """`slot` ostatniego dnia handlowego, którego termin (`slot + grace`) już minął.
    Zapas dodajemy w czasie bezwzględnym (UTC), nie na zegarze ściennym ET — dzień
    zmiany czasu ma 23 albo 25 godzin."""
    day = now.date()
    while True:
        if is_trading_day(day):
            slot_at = datetime.combine(day, slot, tzinfo=NYSE_TZ)
            if slot_at.astimezone(UTC) + grace <= now:
                return slot_at
        day -= timedelta(days=1)


def _expectation_overdue(
    expectation: LivenessExpectation, last_seen: datetime | None, now: datetime
) -> bool:
    if expectation.max_gap_hours <= 0:
        return False
    if expectation.trading_day_slot is None:
        return is_overdue(last_seen, now, expectation.max_gap_hours)
    return is_overdue_on_trading_days(
        last_seen,
        now,
        expectation.trading_day_slot,
        timedelta(hours=expectation.max_gap_hours),
    )


def overdue_jobs(
    expectations: Sequence[LivenessExpectation],
    last_seen: Mapping[str, datetime | None],
    now: datetime,
) -> tuple[str, ...]:
    """Nazwy spóźnionych jobów, w kolejności oczekiwań."""
    return tuple(
        e.job for e in expectations if _expectation_overdue(e, last_seen.get(e.job), now)
    )


class JobState(StrEnum):
    """NIEZ-33 — stan jobu na publicznej stronie statusu, z tej samej reguły co deadman."""

    RUNNING = "running"
    PAUSED = "paused"
    OVERDUE = "overdue"


def job_state(
    expectation: LivenessExpectation, last_seen: datetime | None, now: datetime
) -> JobState:
    """Przerwa ≤ 0 = świadomie wstrzymany; spóźniony dokładnie wtedy, gdy deadman by
    zaalarmował — strona publiczna nie może mówić „działa”, gdy właściciel dostaje alarm."""
    if expectation.max_gap_hours <= 0:
        return JobState.PAUSED
    if _expectation_overdue(expectation, last_seen, now):
        return JobState.OVERDUE
    return JobState.RUNNING
