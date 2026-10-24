"""DAT-20: kalendarz wyników z Finnhub /calendar/earnings — jedno zapytanie na
watchlistę (lista wszystkich spółek z okna, filtrowana do watchlisty), wspólny
limiter Finnhub i odpowiedź trzymana na czas życia adaptera (jeden cykl)."""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock

import pytest
import requests

from src.application.ports import EarningsCalendarPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.earnings import EarningsTiming
from src.domain.quota import QuotaSeverity
from src.infrastructure.adapters.finnhub_earnings import FinnhubEarningsCalendarAdapter

TODAY = date(2026, 10, 26)


def _response(payload: object, status: int = 200) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


def _calendar(*rows: dict[str, object]) -> dict[str, object]:
    return {"earningsCalendar": list(rows)}


def _row(symbol: str, day: str, hour: str = "", eps: float | None = None) -> dict[str, object]:
    return {"symbol": symbol, "date": day, "hour": hour, "epsEstimate": eps, "quarter": 4}


def _adapter(
    *symbols: str, quota_monitor: QuotaMonitor | None = None, rate_limiter: Mock | None = None
) -> FinnhubEarningsCalendarAdapter:
    return FinnhubEarningsCalendarAdapter(
        "fh-secret-token-1",
        symbols,
        today=lambda: TODAY,
        quota_monitor=quota_monitor,
        rate_limiter=rate_limiter,
    )


def test_is_an_earnings_calendar_port() -> None:
    assert isinstance(_adapter("AAPL"), EarningsCalendarPort)


def test_single_request_covers_all_symbols(mocker) -> None:
    get = mocker.patch(
        "requests.Session.get",
        return_value=_response(
            _calendar(
                _row("AAPL", "2026-10-29", "amc", 1.5),
                _row("MSFT", "2026-10-28", "bmo"),
                _row("NVDA", "2026-11-19", ""),
            )
        ),
    )
    adapter = _adapter("AAPL", "MSFT", "NVDA")

    events = [adapter.get_next_earnings(s) for s in ("AAPL", "MSFT", "NVDA")]

    assert get.call_count == 1
    assert [(e.symbol, e.report_date, e.days_until, e.timing) for e in events if e] == [
        ("AAPL", "2026-10-29", 3, EarningsTiming.AFTER_CLOSE),
        ("MSFT", "2026-10-28", 2, EarningsTiming.BEFORE_OPEN),
        ("NVDA", "2026-11-19", 24, None),
    ]
    assert events[0] is not None and events[0].eps_estimate == pytest.approx(1.5)
    params = get.call_args.kwargs["params"]
    assert (params["from"], params["to"]) == ("2026-10-26", "2026-11-25")
    assert "symbol" not in params


def test_maps_calendar_rows_to_events_for_watchlist_only(mocker) -> None:
    mocker.patch(
        "requests.Session.get",
        return_value=_response(
            _calendar(
                _row("ZZZZ", "2026-10-27"),
                _row("AAPL", "2026-11-20"),
                _row("AAPL", "2026-10-30"),
                _row("AAPL", "2026-10-20"),
                _row("AAPL", "not-a-date"),
            )
        ),
    )
    adapter = _adapter("AAPL")

    event = adapter.get_next_earnings("AAPL")

    assert adapter.get_next_earnings("ZZZZ") is None
    assert event is not None and event.report_date == "2026-10-30"


@pytest.mark.parametrize(
    "response",
    [
        _response({}, status=403),
        _response({"earningsCalendar": None}),
        _response([]),
    ],
    ids=["premium-403", "null-calendar", "not-object"],
)
def test_unusable_answer_gives_no_event_and_no_retry(mocker, response: Mock) -> None:
    get = mocker.patch("requests.Session.get", return_value=response)
    adapter = _adapter("AAPL", "MSFT")

    assert adapter.get_next_earnings("AAPL") is None
    assert adapter.get_next_earnings("MSFT") is None
    assert get.call_count == 1


def test_provider_failure_is_logged_without_the_token(mocker, caplog) -> None:
    import logging

    mocker.patch(
        "requests.Session.get",
        side_effect=requests.ConnectionError(
            "Max retries: /calendar/earnings?from=x&token=fh-secret-token-1"
        ),
    )

    with caplog.at_level(logging.WARNING):
        assert _adapter("AAPL").get_next_earnings("AAPL") is None

    assert "Finnhub earnings calendar" in caplog.text
    assert "fh-secret-token-1" not in caplog.text


def test_rate_limit_is_a_quota_alert_and_uses_the_shared_limiter(mocker) -> None:
    mocker.patch("requests.Session.get", return_value=_response({}, status=429))
    limiter = Mock()
    monitor = QuotaMonitor()

    _adapter("AAPL", quota_monitor=monitor, rate_limiter=limiter).get_next_earnings("AAPL")

    limiter.acquire.assert_called_once()
    assert [alert.source for alert in monitor.alerts] == ["Finnhub earnings"]
    assert monitor.alerts[0].severity in {QuotaSeverity.WARNING, QuotaSeverity.CRITICAL}
