"""Testy adaptera AlphaVantage EARNINGS_CALENDAR.

Mockujemy `self._session.get` (konwencja repo) — odpowiedź AV to CSV.
Wstrzykujemy `today`, żeby testy były deterministyczne niezależnie od
realnej daty."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.adapters.alpha_vantage_earnings import (
    AlphaVantageEarningsAdapter,
    NullEarningsCalendarAdapter,
)

# AV zwraca CSV z nagłówkiem; dwie przyszłe daty + jedna przeszła.
CSV_TWO_FUTURE = (
    "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
    "AAPL,Apple Inc,2026-08-01,2026-06-30,1.50,USD\n"
    "AAPL,Apple Inc,2026-07-15,2026-06-30,1.50,USD\n"
)

CSV_PAST_ONLY = (
    "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
    "AAPL,Apple Inc,2026-01-01,2025-12-31,1.40,USD\n"
)

CSV_EMPTY = "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"

# AV przy rate-limit / błędzie zwraca JSON zamiast CSV.
RATE_LIMIT_NOTE = (
    '{"Information": "Thank you for using Alpha Vantage! Our standard '
    "API rate limit is 25 requests per day.\"}"
)


def _mock_response(text: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


def test_picks_earliest_future_report_date() -> None:
    today = date(2026, 6, 21)
    adapter = AlphaVantageEarningsAdapter(api_key="KEY", today=today)
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.return_value = _mock_response(CSV_TWO_FUTURE)
        event = adapter.get_next_earnings("AAPL")

    assert event is not None
    assert event.symbol == "AAPL"
    # Najwcześniejsza przyszła data to 2026-07-15, nie 2026-08-01.
    assert event.report_date == "2026-07-15"
    assert event.days_until == (date(2026, 7, 15) - today).days


def test_calls_earnings_calendar_endpoint() -> None:
    adapter = AlphaVantageEarningsAdapter(api_key="KEY", today=date(2026, 6, 21))
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.return_value = _mock_response(CSV_TWO_FUTURE)
        adapter.get_next_earnings("AAPL")

    _, kwargs = mock_get.call_args
    params = kwargs["params"]
    assert params["function"] == "EARNINGS_CALENDAR"
    assert params["symbol"] == "AAPL"
    assert params["horizon"] == "3month"
    assert params["apikey"] == "KEY"
    assert kwargs["timeout"] == pytest.approx(10.0)


def test_past_only_returns_none() -> None:
    today = date(2026, 6, 21)
    adapter = AlphaVantageEarningsAdapter(api_key="KEY", today=today)
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.return_value = _mock_response(CSV_PAST_ONLY)
        assert adapter.get_next_earnings("AAPL") is None


def test_empty_csv_returns_none() -> None:
    adapter = AlphaVantageEarningsAdapter(api_key="KEY", today=date(2026, 6, 21))
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.return_value = _mock_response(CSV_EMPTY)
        assert adapter.get_next_earnings("AAPL") is None


def test_rate_limit_json_returns_none() -> None:
    # AV zwrócił JSON note zamiast CSV — nie parsujemy, zwracamy None.
    adapter = AlphaVantageEarningsAdapter(api_key="KEY", today=date(2026, 6, 21))
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.return_value = _mock_response(RATE_LIMIT_NOTE)
        assert adapter.get_next_earnings("AAPL") is None


def test_non_200_returns_none() -> None:
    adapter = AlphaVantageEarningsAdapter(api_key="KEY", today=date(2026, 6, 21))
    with patch.object(adapter._session, "get") as mock_get:
        resp = _mock_response("", status=500)
        resp.raise_for_status.side_effect = Exception("boom")
        mock_get.return_value = resp
        assert adapter.get_next_earnings("AAPL") is None


def test_network_error_returns_none() -> None:
    adapter = AlphaVantageEarningsAdapter(api_key="KEY", today=date(2026, 6, 21))
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.side_effect = Exception("connection reset")
        assert adapter.get_next_earnings("AAPL") is None


def test_malformed_date_row_skipped() -> None:
    # Wiersz z nieparsowalną datą jest pomijany; bierzemy następny ważny.
    csv = (
        "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
        "AAPL,Apple Inc,not-a-date,2026-06-30,1.50,USD\n"
        "AAPL,Apple Inc,2026-07-15,2026-06-30,1.50,USD\n"
    )
    today = date(2026, 6, 21)
    adapter = AlphaVantageEarningsAdapter(api_key="KEY", today=today)
    with patch.object(adapter._session, "get") as mock_get:
        mock_get.return_value = _mock_response(csv)
        event = adapter.get_next_earnings("AAPL")

    assert event is not None
    assert event.report_date == "2026-07-15"


def test_today_defaults_without_injection() -> None:
    # Bez wstrzykniętego today adapter nie wybucha przy konstrukcji
    # (data.today() liczona leniwie, nie na imporcie).
    adapter = AlphaVantageEarningsAdapter(api_key="KEY")
    # Brak wstrzyknięcia => today NIE jest zamrożone na starcie (liczone leniwie).
    assert adapter._today is None


def test_null_adapter_returns_none() -> None:
    adapter = NullEarningsCalendarAdapter()
    assert adapter.get_next_earnings("AAPL") is None
