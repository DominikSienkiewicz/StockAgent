"""Limit klucza ponawia bieżące zapytanie kolejnym kluczem aż do końca listy."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Literal

import pytest
import requests
from pytest_mock import MockerFixture
from requests.adapters import HTTPAdapter

from src.application.errors import PortError
from src.application.quota_monitor import QuotaMonitor
from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient
from src.infrastructure.adapters.alpha_vantage_earnings import AlphaVantageEarningsAdapter
from src.infrastructure.adapters.alpha_vantage_fundamentals import AlphaVantageFundamentalsAdapter

AdapterName = Literal["news", "fundamentals", "earnings"]
LimitKind = Literal["note", "information", "daily", "http429"]
KEYS = ["rotation-test-key-1", "rotation-test-key-2", "rotation-test-key-3"]


def _response(body: object, status: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = "https://www.alphavantage.co/query"
    response._content = (body if isinstance(body, str) else json.dumps(body)).encode()
    return response


def _limited(kind: LimitKind, key: str) -> requests.Response:
    if kind == "http429":
        return _response("Too many requests " + key, 429)
    marker = "Note" if kind == "note" else "Information"
    limit = "25 requests per day" if kind == "daily" else "60 calls per minute"
    return _response({marker: f"API key {key}: our standard API rate limit is {limit}."})


def _success(adapter: AdapterName) -> requests.Response:
    if adapter == "news":
        return _response({"feed": []})
    if adapter == "fundamentals":
        return _response({"Symbol": "AAPL", "PERatio": "25", "quarterlyEarnings": []})
    return _response(
        "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n"
        "AAPL,Apple,2026-12-01,2026-09-30,1,USD\n"
        "MSFT,Microsoft,2026-12-02,2026-09-30,1,USD\n"
    )


def _fetcher(adapter: AdapterName, monitor: QuotaMonitor) -> Callable[[str], object]:
    if adapter == "news":
        return AlphaVantageClient(
            api_keys=KEYS,
            symbols=["AAPL", "MSFT"],
            lazy_per_symbol=True,
            quota_monitor=monitor,
        ).sentiment_for
    if adapter == "fundamentals":
        return AlphaVantageFundamentalsAdapter(KEYS, quota_monitor=monitor).get_fundamentals
    return AlphaVantageEarningsAdapter(
        KEYS[0], reserve_keys=KEYS[1:], today=date(2026, 11, 3), quota_monitor=monitor
    ).get_next_earnings


@pytest.mark.parametrize("adapter", ["news", "fundamentals", "earnings"])
@pytest.mark.parametrize("limit", ["note", "information", "daily", "http429"])
class TestSequentialKeyRotation:
    def test_retries_pending_request_then_keeps_the_successful_key(
        self,
        adapter: AdapterName,
        limit: LimitKind,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monitor = QuotaMonitor()
        mocker.patch("time.sleep")
        get = mocker.patch(
            "requests.Session.get",
            side_effect=[
                _limited(limit, KEYS[0]),
                _limited(limit, KEYS[1]),
                *[_success(adapter) for _ in range(4)],
            ],
        )
        fetch = _fetcher(adapter, monitor)

        first = fetch("AAPL")
        second = fetch("MSFT")

        expected_count = {"news": 4, "fundamentals": 6, "earnings": 3}[adapter]
        keys = [call.kwargs["params"]["apikey"] for call in get.call_args_list]
        assert keys == [*KEYS[:2], *[KEYS[2]] * (expected_count - 2)]
        params = [call.kwargs["params"] for call in get.call_args_list[:3]]
        assert [{k: v for k, v in p.items() if k != "apikey"} for p in params] == [
            {k: v for k, v in params[0].items() if k != "apikey"}
        ] * 3
        assert first is not None and second is not None
        if adapter == "news":
            assert isinstance(first, dict) and "degraded_reason" not in first
            assert isinstance(second, dict) and "degraded_reason" not in second
        assert monitor.alerts == []
        assert all(key not in caplog.text for key in KEYS)

    def test_exhausts_each_key_once_and_stops_later_requests(
        self, adapter: AdapterName, limit: LimitKind, mocker: MockerFixture
    ) -> None:
        monitor = QuotaMonitor()
        mocker.patch("time.sleep")
        get = mocker.patch("requests.Session.get", return_value=_limited(limit, KEYS[0]))
        fetch = _fetcher(adapter, monitor)

        first = fetch("AAPL")
        fetch("MSFT")
        fetch("AAPL")

        assert [call.kwargs["params"]["apikey"] for call in get.call_args_list] == KEYS
        assert len(monitor.alerts) == 1
        if limit == "http429":
            alert = monitor.alerts[0]
            assert "częstotliwości" not in alert.message
            assert "dzienny" not in alert.message
            assert "HTTP 429" in alert.message
            assert "ALPHA_VANTAGE_API_KEYS" in alert.action
        if adapter == "news":
            assert isinstance(first, dict) and first["degraded_reason"] == "av_keys_exhausted"
        else:
            assert first is None


def test_transport_leaves_http_429_to_key_rotation() -> None:
    adapters: list[
        AlphaVantageClient | AlphaVantageFundamentalsAdapter | AlphaVantageEarningsAdapter
    ] = [
        AlphaVantageClient(KEYS, ["AAPL"]),
        AlphaVantageFundamentalsAdapter(KEYS),
        AlphaVantageEarningsAdapter(KEYS[0], reserve_keys=KEYS[1:]),
    ]
    for adapter in adapters:
        transport = adapter._session.get_adapter("https://")
        assert isinstance(transport, HTTPAdapter)
        assert 429 not in transport.max_retries.status_forcelist
        assert not transport.max_retries.respect_retry_after_header


@pytest.mark.parametrize("adapter", ["news", "fundamentals", "earnings"])
@pytest.mark.parametrize("status", [401, 500])
def test_other_http_errors_do_not_consume_reserve_keys(
    adapter: AdapterName, status: int, mocker: MockerFixture
) -> None:
    monitor = QuotaMonitor()
    get = mocker.patch("requests.Session.get", return_value=_response("Failure", status))
    fetch = _fetcher(adapter, monitor)

    if adapter == "news":
        with pytest.raises(PortError):
            fetch("AAPL")
    else:
        assert fetch("AAPL") is None

    assert get.call_count == 1
    assert monitor.alerts == []


@pytest.mark.parametrize("lazy", [False, True])
def test_news_preserves_completed_feed_when_next_symbol_exhausts_keys(
    lazy: bool, mocker: MockerFixture
) -> None:
    monitor = QuotaMonitor()
    mocker.patch("time.sleep")
    feed = {
        "feed": [
            {
                "title": "Apple",
                "url": "https://example.com/apple",
                "ticker_sentiment": [
                    {"ticker": "AAPL", "relevance_score": "1", "ticker_sentiment_score": "0.5"}
                ],
            }
        ]
    }
    get = mocker.patch(
        "requests.Session.get",
        side_effect=[_response(feed), *[_limited("http429", k) for k in KEYS]],
    )
    client = AlphaVantageClient(KEYS, ["AAPL", "MSFT"], lazy_per_symbol=lazy, quota_monitor=monitor)

    assert len(client.articles_for("AAPL")) == 1
    assert client.articles_for("MSFT") == []
    assert len(client.articles_for("AAPL")) == 1
    assert get.call_count == 4
    assert len(monitor.alerts) == 1


def test_concurrent_http_429_does_not_skip_the_next_key(mocker: MockerFixture) -> None:
    monitor = QuotaMonitor()
    both_in_flight = threading.Barrier(2, timeout=5)
    used_keys: list[str] = []
    mocker.patch("time.sleep")

    def get(_url: str, *, params: dict[str, str], timeout: int) -> requests.Response:
        key = params["apikey"]
        used_keys.append(key)
        if key == KEYS[0]:
            both_in_flight.wait()
            return _limited("http429", key)
        return _success("news")

    mocker.patch("requests.Session.get", side_effect=get)
    client = AlphaVantageClient(KEYS, ["AAPL", "MSFT"], lazy_per_symbol=True, quota_monitor=monitor)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(client.sentiment_for, ["AAPL", "MSFT"]))

    assert used_keys.count(KEYS[0]) == 2
    assert used_keys.count(KEYS[1]) == 2
    assert KEYS[2] not in used_keys
    assert all("degraded_reason" not in result for result in results)
    assert monitor.alerts == []
