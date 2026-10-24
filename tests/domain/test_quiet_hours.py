"""ALR-09 — okno ciszy nocnej pushy w strefie odbiorcy."""

from __future__ import annotations

from datetime import UTC, datetime, time

from src.domain.quiet_hours import is_quiet

WARSAW = "Europe/Warsaw"


def _utc(month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


def test_window_wraps_midnight_in_local_time() -> None:
    # Październik (CEST, UTC+2): 01:30 UTC = 03:30 w Warszawie, 06:30 UTC = 08:30.
    assert is_quiet(_utc(10, 15, 1, 30), WARSAW, time(22, 0), time(7, 0)) is True
    assert is_quiet(_utc(10, 15, 6, 30), WARSAW, time(22, 0), time(7, 0)) is False


def test_start_belongs_to_the_window_and_end_does_not() -> None:
    # 20:00 UTC = 22:00 CEST (początek ciszy), 05:00 UTC = 07:00 CEST (pobudka).
    assert is_quiet(_utc(10, 15, 20, 0), WARSAW) is True
    assert is_quiet(_utc(10, 15, 5, 0), WARSAW) is False
    assert is_quiet(_utc(10, 15, 19, 59), WARSAW) is False


def test_winter_time_shifts_the_utc_hours() -> None:
    # Styczeń (CET, UTC+1): 21:30 UTC = 22:30 lokalnie (cisza), 20:30 UTC = 21:30
    # (jeszcze nie) — okno idzie za czasem lokalnym, nie za godziną UTC.
    assert is_quiet(_utc(1, 15, 21, 30), WARSAW) is True
    assert is_quiet(_utc(1, 15, 20, 30), WARSAW) is False


def test_window_within_one_day_does_not_wrap() -> None:
    assert is_quiet(_utc(10, 15, 12, 0), WARSAW, time(13, 0), time(15, 0)) is True
    assert is_quiet(_utc(10, 15, 14, 0), WARSAW, time(13, 0), time(15, 0)) is False


def test_equal_bounds_mean_no_quiet_hours() -> None:
    assert is_quiet(_utc(10, 15, 1, 30), WARSAW, time(0, 0), time(0, 0)) is False


def test_naive_datetime_is_read_as_utc() -> None:
    assert is_quiet(datetime(2026, 10, 15, 1, 30), WARSAW) is True
