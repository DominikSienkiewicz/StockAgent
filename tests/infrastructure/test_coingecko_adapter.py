"""Testy CoinGeckoAdapter — MarketDataPort dla krypto.

CoinGecko REST: `/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd`
Free tier — bez klucza, bez rate-limitu znaczącego dla pojedynczych zapytań.

Adapter mapuje wewnętrznie tickery (BTC, ETH) na nazwy CoinGecko
(bitcoin, ethereum) — użytkownik trzyma w configu czyste tickery.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from src.application.ports import MarketDataPort
from src.domain.value_objects import Money
from src.infrastructure.adapters.coingecko import CoinGeckoAdapter


def _ok_response(payload: dict[str, Any]) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


def _error_response(status_code: int) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    return response


@pytest.fixture
def adapter() -> CoinGeckoAdapter:
    return CoinGeckoAdapter()


class TestGetCurrentPrice:
    def test_returns_money_for_btc(self, adapter: CoinGeckoAdapter, mocker: Any) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"bitcoin": {"usd": 67234.55}}),
        )

        price = adapter.get_current_price("BTC")

        assert isinstance(price, Money)
        assert price.amount == Decimal("67234.55")

    def test_returns_money_for_eth(self, adapter: CoinGeckoAdapter, mocker: Any) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"ethereum": {"usd": 3142.10}}),
        )

        price = adapter.get_current_price("ETH")

        assert price.amount == Decimal("3142.10")

    def test_maps_ticker_to_coingecko_id(self, adapter: CoinGeckoAdapter, mocker: Any) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"bitcoin": {"usd": 50000}}),
        )

        adapter.get_current_price("BTC")

        call = mock_get.call_args
        # Sprawdzamy że request mówi do CG `ids=bitcoin`, nie `ids=BTC`.
        params = call.kwargs.get("params", {})
        assert params.get("ids") == "bitcoin"
        assert params.get("vs_currencies") == "usd"

    def test_request_url_targets_simple_price_endpoint(
        self, adapter: CoinGeckoAdapter, mocker: Any
    ) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"bitcoin": {"usd": 50000}}),
        )

        adapter.get_current_price("BTC")

        url = mock_get.call_args.args[0]
        assert url.endswith("/simple/price")

    def test_sets_timeout(self, adapter: CoinGeckoAdapter, mocker: Any) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"bitcoin": {"usd": 50000}}),
        )

        adapter.get_current_price("BTC")

        assert mock_get.call_args.kwargs.get("timeout", 0) > 0


class TestErrorHandling:
    def test_raises_value_error_for_unknown_ticker(self, adapter: CoinGeckoAdapter) -> None:
        # Tylko BTC i ETH na start — nieznane tickery rzucają jawnie.
        with pytest.raises(ValueError, match="DOGE"):
            adapter.get_current_price("DOGE")

    def test_raises_on_http_error(self, adapter: CoinGeckoAdapter, mocker: Any) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_error_response(503),
        )

        with pytest.raises(requests.HTTPError):
            adapter.get_current_price("BTC")

    def test_raises_on_empty_response(self, adapter: CoinGeckoAdapter, mocker: Any) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({}),
        )

        with pytest.raises(ValueError, match="BTC"):
            adapter.get_current_price("BTC")

    def test_raises_on_missing_usd_field(self, adapter: CoinGeckoAdapter, mocker: Any) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"bitcoin": {}}),
        )

        with pytest.raises(ValueError, match="BTC"):
            adapter.get_current_price("BTC")


class TestExtensibility:
    def test_custom_mapping_supports_additional_tickers(self, mocker: Any) -> None:
        """User może dorzucić własne tickery przez custom mapping."""
        adapter = CoinGeckoAdapter(ticker_to_id={"BTC": "bitcoin", "SOL": "solana"})
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"solana": {"usd": 145.20}}),
        )

        price = adapter.get_current_price("SOL")

        assert price.amount == Decimal("145.20")


class TestAdapterImplementsPort:
    def test_is_market_data_port(self) -> None:
        assert issubclass(CoinGeckoAdapter, MarketDataPort)


def _rate_limited_response(retry_after: str | None = None) -> Mock:
    """CoinGecko free tier zwraca 429 przy przekroczeniu ~5-15 req/min."""
    response = Mock(spec=requests.Response)
    response.status_code = 429
    response.headers = {"Retry-After": retry_after} if retry_after else {}
    response.raise_for_status.side_effect = requests.HTTPError("429 error")
    return response


class TestQuotaAlertEmit:
    """429 z CoinGecko → QuotaAlert do monitora (mirror FinnhubAdapter)."""

    def test_emits_critical_alert_on_429(self, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity

        monitor = QuotaMonitor()
        adapter = CoinGeckoAdapter(quota_monitor=monitor)
        mocker.patch(
            "requests.Session.get",
            return_value=_rate_limited_response(),
        )

        with pytest.raises(requests.HTTPError):
            adapter.get_current_price("BTC")

        assert len(monitor.alerts) == 1
        alert = monitor.alerts[0]
        assert alert.source == "CoinGecko"
        assert alert.severity is QuotaSeverity.CRITICAL
        assert alert.message
        assert alert.action

    def test_retry_after_header_surfaced_in_action(self, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor

        monitor = QuotaMonitor()
        adapter = CoinGeckoAdapter(quota_monitor=monitor)
        mocker.patch(
            "requests.Session.get",
            return_value=_rate_limited_response(retry_after="42"),
        )

        with pytest.raises(requests.HTTPError):
            adapter.get_current_price("BTC")

        assert "42" in monitor.alerts[0].action

    def test_happy_path_emits_no_alert(self, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor

        monitor = QuotaMonitor()
        adapter = CoinGeckoAdapter(quota_monitor=monitor)
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"bitcoin": {"usd": 50000}}),
        )

        adapter.get_current_price("BTC")

        assert monitor.alerts == []

    def test_no_monitor_means_no_crash_on_429(self, mocker: Any) -> None:
        # Backward-compat: bez monitora 429 dalej raisuje HTTPError, brak emitu.
        adapter = CoinGeckoAdapter()
        mocker.patch(
            "requests.Session.get",
            return_value=_rate_limited_response(),
        )

        with pytest.raises(requests.HTTPError):
            adapter.get_current_price("BTC")


def test_rate_limit_alert_is_polish_and_logged_in_english(
    mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # MAIL-15: komunikat alertu idzie do maila po polsku, diagnoza do logów po angielsku.
    import logging

    from src.application.quota_monitor import QuotaMonitor

    monitor = QuotaMonitor()
    adapter = CoinGeckoAdapter(quota_monitor=monitor)
    mocker.patch("requests.Session.get", return_value=_rate_limited_response(retry_after="42"))

    with caplog.at_level(logging.WARNING), pytest.raises(requests.HTTPError):
        adapter.get_current_price("BTC")

    alert = monitor.alerts[0]
    assert "wyczerpany darmowy limit" in alert.message
    assert "exhausted" not in alert.message
    assert alert.action.startswith("Spowolnij")
    assert "CoinGecko returned 429 for 'BTC'" in caplog.text


class TestGetDailyChange:
    """ALR-02: zmiana 24 h i chwila aktualizacji z tego samego /simple/price."""

    def test_asks_for_24h_change_and_update_time(
        self, adapter: CoinGeckoAdapter, mocker: Any
    ) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(
                {"bitcoin": {"usd": 61000, "usd_24h_change": -8.0, "last_updated_at": 1760630400}}
            ),
        )

        adapter.get_daily_change("BTC")

        params = mock_get.call_args.kwargs["params"]
        assert params["ids"] == "bitcoin"
        assert params["include_24hr_change"] == "true"
        assert params["include_last_updated_at"] == "true"

    def test_returns_fraction_and_update_time(self, adapter: CoinGeckoAdapter, mocker: Any) -> None:
        from datetime import UTC, datetime

        from src.domain.shock import DailyChange

        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(
                {"bitcoin": {"usd": 61000, "usd_24h_change": -8.0, "last_updated_at": 1760630400}}
            ),
        )

        assert adapter.get_daily_change("BTC") == DailyChange(
            fraction=Decimal("-0.08"),
            as_of=datetime.fromtimestamp(1760630400, tz=UTC),
        )

    def test_raises_value_error_without_change_field(
        self, adapter: CoinGeckoAdapter, mocker: Any
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"bitcoin": {"usd": 61000, "last_updated_at": 1760630400}}),
        )

        with pytest.raises(ValueError, match="24h"):
            adapter.get_daily_change("BTC")

    def test_raises_value_error_for_unmapped_ticker(self, adapter: CoinGeckoAdapter) -> None:
        with pytest.raises(ValueError, match="not mapped"):
            adapter.get_daily_change("DOGE")

    def test_rate_limit_emits_a_quota_alert(self, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor

        monitor = QuotaMonitor()
        adapter = CoinGeckoAdapter(quota_monitor=monitor)
        mocker.patch("requests.Session.get", return_value=_rate_limited_response())

        with pytest.raises(requests.HTTPError):
            adapter.get_daily_change("BTC")

        assert len(monitor.alerts) == 1

    def test_is_daily_change_port(self) -> None:
        from src.application.ports import DailyChangePort

        assert issubclass(CoinGeckoAdapter, DailyChangePort)


class TestPortErrorTaxonomy:
    """ARCH-33: błędy CoinGecko wychodzą jako taksonomia portów, a przetłumaczony
    wyjątek nadal jest typem `requests` / `ValueError` (wywołujący działają jak dotąd)."""

    def test_network_error_is_provider_unavailable(
        self, adapter: CoinGeckoAdapter, mocker: Any
    ) -> None:
        from src.application.errors import ProviderUnavailable

        mocker.patch("requests.Session.get", side_effect=requests.ConnectionError("reset"))

        with pytest.raises(ProviderUnavailable) as excinfo:
            adapter.get_current_price("BTC")

        assert isinstance(excinfo.value, requests.ConnectionError)

    @pytest.mark.parametrize(
        ("status", "kind_name"), [(503, "ProviderUnavailable"), (429, "QuotaExhausted")]
    )
    def test_error_status_maps_to_its_kind(
        self, adapter: CoinGeckoAdapter, mocker: Any, status: int, kind_name: str
    ) -> None:
        from src.application import errors

        mocker.patch("requests.Session.get", return_value=_error_response(status))

        with pytest.raises(getattr(errors, kind_name)) as excinfo:
            adapter.get_daily_change("BTC")

        assert isinstance(excinfo.value, requests.HTTPError)

    def test_unmapped_and_unknown_coins_are_not_found(
        self, adapter: CoinGeckoAdapter, mocker: Any
    ) -> None:
        from src.application.errors import NotFound

        mocker.patch("requests.Session.get", return_value=_ok_response({}))

        with pytest.raises(NotFound):
            adapter.get_current_price("DOGE")
        with pytest.raises(NotFound) as excinfo:
            adapter.get_current_price("BTC")

        assert isinstance(excinfo.value, ValueError)

    def test_price_block_without_the_currency_is_malformed_payload(
        self, adapter: CoinGeckoAdapter, mocker: Any
    ) -> None:
        from src.application.errors import MalformedPayload

        mocker.patch("requests.Session.get", return_value=_ok_response({"bitcoin": {}}))

        with pytest.raises(MalformedPayload):
            adapter.get_current_price("BTC")

    def test_change_block_without_the_change_is_malformed_payload(
        self, adapter: CoinGeckoAdapter, mocker: Any
    ) -> None:
        from src.application.errors import MalformedPayload

        mocker.patch("requests.Session.get", return_value=_ok_response({"bitcoin": {"usd": 1.0}}))

        with pytest.raises(MalformedPayload):
            adapter.get_daily_change("BTC")

    def test_unparseable_body_is_malformed_payload(
        self, adapter: CoinGeckoAdapter, mocker: Any
    ) -> None:
        from src.application.errors import MalformedPayload

        response = _ok_response({})
        response.json.side_effect = requests.JSONDecodeError("Expecting value", "<html>", 0)
        mocker.patch("requests.Session.get", return_value=response)

        with pytest.raises(MalformedPayload):
            adapter.get_current_price("BTC")
