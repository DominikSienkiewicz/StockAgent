"""FIN-11: read-through cache kalendarza wyników.

Slow Loop odświeża `earnings_cache` z Alpha Vantage, Fast Loop dostaje ten sam
adapter z delegatem Null — czyta wyłącznie cache i nigdy nie woła AV przed
bramką volatility."""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import Mock

from src.application.ports import EarningsCalendarPort, RepositoryPort
from src.domain.earnings import CachedEarningsDate, EarningsEvent
from src.infrastructure.adapters.alpha_vantage_earnings import NullEarningsCalendarAdapter
from src.infrastructure.adapters.cached_earnings import CachedEarningsAdapter

TODAY = date(2026, 10, 27)


def _adapter(
    repo: Mock, delegate: EarningsCalendarPort | None = None
) -> CachedEarningsAdapter:
    return CachedEarningsAdapter(
        repo=repo,
        delegate=delegate or Mock(spec=EarningsCalendarPort),
        today=lambda: TODAY,
        clock=lambda: datetime(2026, 10, 25, 3, 0, tzinfo=UTC),
    )


def test_fresh_cache_answers_without_the_delegate() -> None:
    repo = Mock(spec=RepositoryPort)
    repo.get_cached_earnings.return_value = CachedEarningsDate(
        "AAPL", date(2026, 10, 30), datetime(2026, 10, 25, tzinfo=UTC)
    )
    delegate = Mock(spec=EarningsCalendarPort)

    event = _adapter(repo, delegate).get_next_earnings("AAPL")

    assert event == EarningsEvent("AAPL", days_until=3, report_date="2026-10-30")
    repo.get_cached_earnings.assert_called_once_with("AAPL")
    delegate.get_next_earnings.assert_not_called()


def test_cache_miss_asks_the_delegate_and_stores_the_report_date() -> None:
    repo = Mock(spec=RepositoryPort)
    repo.get_cached_earnings.return_value = None
    delegate = Mock(spec=EarningsCalendarPort)
    delegate.get_next_earnings.return_value = EarningsEvent(
        "AAPL", days_until=4, report_date="2026-10-29"
    )

    event = _adapter(repo, delegate).get_next_earnings("AAPL")

    assert event == EarningsEvent("AAPL", days_until=4, report_date="2026-10-29")
    repo.save_earnings.assert_called_once_with(
        CachedEarningsDate("AAPL", date(2026, 10, 29), datetime(2026, 10, 25, 3, 0, tzinfo=UTC))
    )


def test_no_upcoming_report_is_not_cached() -> None:
    # Adapter AV oddaje None także przy błędzie — zapis „brak raportu” na tydzień
    # utrwaliłby awarię. Fast Loop i tak dostaje None przy pustym cache.
    repo = Mock(spec=RepositoryPort)
    repo.get_cached_earnings.return_value = None
    delegate = Mock(spec=EarningsCalendarPort)
    delegate.get_next_earnings.return_value = None

    assert _adapter(repo, delegate).get_next_earnings("AAPL") is None
    repo.save_earnings.assert_not_called()


def test_event_without_a_report_date_is_not_cached() -> None:
    repo = Mock(spec=RepositoryPort)
    repo.get_cached_earnings.return_value = None
    delegate = Mock(spec=EarningsCalendarPort)
    delegate.get_next_earnings.return_value = EarningsEvent("AAPL", days_until=None)

    _adapter(repo, delegate).get_next_earnings("AAPL")

    repo.save_earnings.assert_not_called()


def test_fast_loop_wiring_reads_the_cache_only() -> None:
    repo = Mock(spec=RepositoryPort)
    repo.get_cached_earnings.return_value = None
    adapter = _adapter(repo, NullEarningsCalendarAdapter())

    assert adapter.get_next_earnings("AAPL") is None
    repo.save_earnings.assert_not_called()


def test_defaults_use_the_real_calendar_and_clock() -> None:
    repo = Mock(spec=RepositoryPort)
    repo.get_cached_earnings.return_value = None

    adapter = CachedEarningsAdapter(repo=repo, delegate=NullEarningsCalendarAdapter())

    assert adapter.get_next_earnings("AAPL") is None
