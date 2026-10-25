"""Reguła deadmana (NIEZ-11): job jest „spóźniony”, gdy od jego ostatniego śladu w
bazie minęło więcej niż dopuszczalna przerwa. Przerwa 0 = job świadomie wstrzymany."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from src.domain.liveness import (
    JobState,
    LivenessExpectation,
    is_overdue,
    is_overdue_on_trading_days,
    job_state,
    overdue_jobs,
)

NOW = datetime(2026, 10, 18, 8, 0, tzinfo=UTC)


def test_gap_longer_than_allowed_is_overdue() -> None:
    assert is_overdue(NOW - timedelta(hours=26), NOW, max_gap_hours=24)


def test_recent_heartbeat_is_not_overdue() -> None:
    assert not is_overdue(NOW - timedelta(hours=2), NOW, max_gap_hours=24)


def test_paused_job_is_never_overdue() -> None:
    assert not is_overdue(None, NOW, max_gap_hours=0)
    assert not is_overdue(NOW - timedelta(days=60), NOW, max_gap_hours=0)


def test_missing_heartbeat_is_overdue_for_an_expected_job() -> None:
    assert is_overdue(None, NOW, max_gap_hours=192)


def test_overdue_jobs_lists_only_late_expected_jobs() -> None:
    expectations = (
        LivenessExpectation(job="Fast Loop", max_gap_hours=0),
        LivenessExpectation(job="Slow Loop", max_gap_hours=192),
    )
    last_seen = {"Fast Loop": None, "Slow Loop": NOW - timedelta(days=9)}

    assert overdue_jobs(expectations, last_seen, NOW) == ("Slow Loop",)


# NIEZ-11 (dni handlowe): Fast Loop po wznowieniu chodzi w dni handlowe NYSE przed
# otwarciem (cron 05:30 UTC), deadman sprawdza o 09:17 UTC. Termin dnia handlowego
# liczymy od północy ET; weekend i święto NYSE nie mają terminu.
START_OF_TRADING_DAY = time(0, 0)
GRACE = timedelta(hours=26)
FRIDAY_CYCLE = datetime(2026, 10, 16, 5, 35, tzinfo=UTC)
MONDAY_DEADMAN = datetime(2026, 10, 19, 9, 17, tzinfo=UTC)


def test_monday_morning_after_friday_cycle_is_not_overdue() -> None:
    assert not is_overdue_on_trading_days(FRIDAY_CYCLE, MONDAY_DEADMAN, START_OF_TRADING_DAY, GRACE)
    # Reguła godzinowa z tą samą przerwą alarmowałaby po każdym weekendzie.
    assert is_overdue(FRIDAY_CYCLE, MONDAY_DEADMAN, max_gap_hours=26)


def test_weekend_check_after_friday_cycle_is_not_overdue() -> None:
    sunday_deadman = datetime(2026, 10, 18, 9, 17, tzinfo=UTC)

    assert not is_overdue_on_trading_days(FRIDAY_CYCLE, sunday_deadman, START_OF_TRADING_DAY, GRACE)


def test_two_missed_trading_days_is_overdue() -> None:
    # Ostatni cykl we wtorek; środa i czwartek bez śladu.
    tuesday_cycle = datetime(2026, 10, 13, 5, 35, tzinfo=UTC)
    thursday_deadman = datetime(2026, 10, 15, 9, 17, tzinfo=UTC)

    assert is_overdue_on_trading_days(tuesday_cycle, thursday_deadman, START_OF_TRADING_DAY, GRACE)


def test_holiday_is_not_a_missed_trading_day() -> None:
    # Środa przed Thanksgiving (czw 26.11.2026), sprawdzenie w piątek rano.
    wednesday_cycle = datetime(2026, 11, 25, 5, 35, tzinfo=UTC)
    friday_deadman = datetime(2026, 11, 27, 9, 17, tzinfo=UTC)

    assert not is_overdue_on_trading_days(
        wednesday_cycle, friday_deadman, START_OF_TRADING_DAY, GRACE
    )
    assert is_overdue(wednesday_cycle, friday_deadman, max_gap_hours=26)


def test_short_grace_flags_a_single_missed_trading_day() -> None:
    tuesday_cycle = datetime(2026, 10, 13, 5, 35, tzinfo=UTC)
    wednesday_deadman = datetime(2026, 10, 14, 9, 17, tzinfo=UTC)

    assert is_overdue_on_trading_days(
        tuesday_cycle, wednesday_deadman, START_OF_TRADING_DAY, timedelta(hours=4)
    )


def test_trading_day_job_without_any_trace_is_overdue() -> None:
    assert is_overdue_on_trading_days(None, MONDAY_DEADMAN, START_OF_TRADING_DAY, GRACE)


def test_overdue_jobs_uses_trading_days_for_a_trading_day_expectation() -> None:
    expectations = (
        LivenessExpectation(
            job="Fast Loop", max_gap_hours=26, trading_day_slot=START_OF_TRADING_DAY
        ),
        LivenessExpectation(job="Slow Loop", max_gap_hours=192),
    )
    last_seen = {"Fast Loop": FRIDAY_CYCLE, "Slow Loop": MONDAY_DEADMAN - timedelta(days=1)}

    assert overdue_jobs(expectations, last_seen, MONDAY_DEADMAN) == ()


def test_paused_trading_day_job_is_never_overdue() -> None:
    paused = LivenessExpectation(
        job="Fast Loop", max_gap_hours=0, trading_day_slot=START_OF_TRADING_DAY
    )

    assert overdue_jobs((paused,), {"Fast Loop": None}, MONDAY_DEADMAN) == ()


# --- NIEZ-33: stan jobu dla publicznej strony statusu (ta sama reguła co deadman) ---


def test_job_with_a_zero_gap_is_paused_on_purpose() -> None:
    paused = LivenessExpectation("Fast Loop", 0.0)

    assert job_state(paused, NOW - timedelta(days=60), NOW) is JobState.PAUSED


def test_job_the_deadman_would_alert_on_is_overdue() -> None:
    fast = LivenessExpectation("Fast Loop", 26.0, trading_day_slot=time(0, 0))

    assert job_state(fast, NOW - timedelta(days=9), NOW) is JobState.OVERDUE
    assert job_state(fast, None, NOW) is JobState.OVERDUE


def test_job_inside_its_gap_is_running() -> None:
    slow = LivenessExpectation("Slow Loop", 192.0)

    assert job_state(slow, NOW - timedelta(days=2), NOW) is JobState.RUNNING
