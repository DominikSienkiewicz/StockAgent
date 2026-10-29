"""Odmowa dostępu AV nie jest limitem dobowym ani pustym poprawnym feedem."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, Literal
from unittest.mock import Mock

import pytest
import requests
from pytest_mock import MockerFixture

from src.application.quota_monitor import QuotaMonitor
from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient
from src.infrastructure.adapters.alpha_vantage_earnings import AlphaVantageEarningsAdapter
from src.infrastructure.adapters.alpha_vantage_fundamentals import AlphaVantageFundamentalsAdapter
from tests.infrastructure.provider_fixtures import load_json

AdapterName = Literal["news", "fundamentals", "earnings"]


def _response(payload: dict[str, Any]) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.text = json.dumps(payload)
    return response


def _premium_payload() -> dict[str, Any]:
    payload = load_json("av_premium_endpoint_truncated.json")
    assert isinstance(payload, dict)
    return payload


def _fetcher(name: AdapterName, monitor: QuotaMonitor) -> Callable[[str], object]:
    if name == "news":
        return AlphaVantageClient(
            api_keys=["free-key-1", "free-key-2"],
            symbols=["AAPL", "MSFT"],
            quota_monitor=monitor,
            lazy_per_symbol=True,
        ).sentiment_for
    if name == "fundamentals":
        return AlphaVantageFundamentalsAdapter(
            api_keys=["free-key-1", "free-key-2"], quota_monitor=monitor
        ).get_fundamentals
    return AlphaVantageEarningsAdapter(
        api_key="free-key-1",
        reserve_keys=["free-key-2"],
        quota_monitor=monitor,
        today=date(2026, 10, 29),
    ).get_next_earnings


class TestAlphaVantageResponseCauses:
    @pytest.mark.parametrize("adapter", ["news", "fundamentals", "earnings"])
    def test_premium_denial_is_not_daily_quota_and_is_cached(
        self, adapter: AdapterName, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        monitor = QuotaMonitor()
        mock_get = mocker.patch("requests.Session.get", return_value=_response(_premium_payload()))
        mocker.patch("time.sleep")
        fetch = _fetcher(adapter, monitor)

        first = fetch("AAPL")
        fetch("MSFT")
        fetch("AAPL")

        assert len(monitor.alerts) == 1
        alert = monitor.alerts[0]
        assert "premium" in alert.message.lower(), "odmowa planu ma wskazywać dostęp premium"
        assert "dzienny limit" not in alert.message
        assert "północy" not in alert.action
        assert "darmowy" not in alert.action
        assert "uprawnienia" in alert.action.lower()
        assert mock_get.call_count == 2
        assert "daily quota" not in caplog.text
        assert "rate-limited" not in caplog.text
        if adapter == "news":
            assert isinstance(first, dict)
            assert first["degraded_reason"] == "av_keys_exhausted"

    @pytest.mark.parametrize("adapter", ["news", "fundamentals", "earnings"])
    @pytest.mark.parametrize("information", ["Contact support.", "This dataset is updated daily."])
    def test_unknown_information_does_not_invent_daily_quota(
        self, adapter: AdapterName, information: str, mocker: MockerFixture
    ) -> None:
        monitor = QuotaMonitor()
        mocker.patch("requests.Session.get", return_value=_response({"Information": information}))
        mocker.patch("time.sleep")

        _fetcher(adapter, monitor)("AAPL")

        alert = monitor.alerts[0]
        assert "dzienny limit" not in alert.message
        assert "północy" not in alert.action
        assert "darmowy" not in alert.action
        assert "niejednoznaczn" in alert.message

    def test_premium_api_function_message_reports_access_denial(
        self, mocker: MockerFixture
    ) -> None:
        monitor = QuotaMonitor()
        mocker.patch(
            "requests.Session.get",
            return_value=_response({"Information": "This is a premium API function."}),
        )
        mocker.patch("time.sleep")

        _fetcher("news", monitor)("AAPL")

        assert "premium" in monitor.alerts[0].message

    @pytest.mark.parametrize("adapter", ["news", "fundamentals", "earnings"])
    def test_daily_quota_in_note_does_not_trigger_frequency_retry(
        self, adapter: AdapterName, mocker: MockerFixture
    ) -> None:
        monitor = QuotaMonitor()
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_response({"Note": "Our standard API rate limit is 25 requests per day."}),
        )
        mocker.patch("time.sleep")

        _fetcher(adapter, monitor)("AAPL")

        assert mock_get.call_count == 2
        assert "dzienny limit" in monitor.alerts[0].message
        assert "częstotliwości" not in monitor.alerts[0].message

    @pytest.mark.parametrize("adapter", ["news", "fundamentals", "earnings"])
    def test_frequency_limit_in_information_retries_each_key_once(
        self, adapter: AdapterName, mocker: MockerFixture
    ) -> None:
        monitor = QuotaMonitor()
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_response(
                {"Information": "Our standard API call frequency is 5 calls per minute."}
            ),
        )
        mocker.patch("time.sleep")

        _fetcher(adapter, monitor)("AAPL")

        assert mock_get.call_count == 4
        assert "częstotliwości" in monitor.alerts[0].message
        assert "dzienny limit" not in monitor.alerts[0].message
        assert "północy" not in monitor.alerts[0].action

    def test_news_can_fall_back_to_an_entitled_key(self, mocker: MockerFixture) -> None:
        monitor = QuotaMonitor()
        mock_get = mocker.patch(
            "requests.Session.get",
            side_effect=[
                _response(_premium_payload()),
                _response({"feed": []}),
                _response({"feed": []}),
            ],
        )
        mocker.patch("time.sleep")
        fetch = _fetcher("news", monitor)

        assert fetch("AAPL") == fetch("MSFT")
        assert monitor.alerts == []
        assert [call.kwargs["params"]["apikey"] for call in mock_get.call_args_list] == [
            "free-key-1",
            "free-key-2",
            "free-key-2",
        ]

    def test_news_mixed_denials_do_not_claim_every_key_hit_daily_quota(
        self, mocker: MockerFixture
    ) -> None:
        monitor = QuotaMonitor()
        mocker.patch(
            "requests.Session.get",
            side_effect=[
                _response(_premium_payload()),
                _response({"Information": "Our standard API rate limit is 25 requests per day."}),
            ],
        )
        mocker.patch("time.sleep")

        _fetcher("news", monitor)("AAPL")

        alert = monitor.alerts[0]
        assert "dzienny limit" not in alert.message
        assert "północy" not in alert.action

    def test_data_with_information_is_still_a_success(self, mocker: MockerFixture) -> None:
        monitor = QuotaMonitor()
        mocker.patch(
            "requests.Session.get", return_value=_response({**_premium_payload(), "feed": []})
        )
        client = AlphaVantageClient(api_keys="key", symbols=["AAPL"], quota_monitor=monitor)

        client.sentiment_for("AAPL")

        assert client.degraded_reason is None
        assert monitor.alerts == []

    def test_premium_log_redacts_the_entire_key_before_truncation(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        key = "private-premium-key-that-crosses-the-log-truncation-boundary"
        marker = "This is a premium endpoint. " + "x" * 120 + key
        mocker.patch("requests.Session.get", return_value=_response({"Information": marker}))
        monitor = QuotaMonitor()
        client = AlphaVantageClient(api_keys=key, symbols=["AAPL"], quota_monitor=monitor)

        client.articles_for("AAPL")

        assert "premium" in monitor.alerts[0].message
        assert key not in caplog.text
        assert "private-premium" not in caplog.text

    def test_concurrent_premium_denials_preserve_one_alert_and_stop_later_symbols(
        self, mocker: MockerFixture
    ) -> None:
        monitor = QuotaMonitor()
        both_in_flight = threading.Barrier(2, timeout=5)

        def get(*_args: object, **_kwargs: object) -> Mock:
            both_in_flight.wait()
            return _response(_premium_payload())

        mock_get = mocker.patch("requests.Session.get", side_effect=get)
        mocker.patch("time.sleep")
        client = AlphaVantageClient(
            api_keys="free-key",
            symbols=["AAPL", "MSFT", "NVDA"],
            quota_monitor=monitor,
            lazy_per_symbol=True,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(client.sentiment_for, ["AAPL", "MSFT"]))
        results.append(client.sentiment_for("NVDA"))

        assert mock_get.call_count == 2
        assert len(monitor.alerts) == 1
        assert "premium" in monitor.alerts[0].message
        assert all(result["degraded_reason"] == "av_keys_exhausted" for result in results)
