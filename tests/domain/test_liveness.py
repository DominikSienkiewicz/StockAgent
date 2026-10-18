"""Reguła deadmana (NIEZ-11): job jest „spóźniony”, gdy od jego ostatniego śladu w
bazie minęło więcej niż dopuszczalna przerwa. Przerwa 0 = job świadomie wstrzymany."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.domain.liveness import LivenessExpectation, is_overdue, overdue_jobs

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
