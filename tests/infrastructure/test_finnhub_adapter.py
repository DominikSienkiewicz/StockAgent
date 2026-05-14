from decimal import Decimal
from unittest.mock import Mock

import pytest
import requests

from src.domain.value_objects import Money
from src.infrastructure.adapters.finnhub_api import FinnhubAdapter


@pytest.fixture
def adapter() -> FinnhubAdapter:
    return FinnhubAdapter(api_key="test-token-123")


def _ok_response(payload: dict, status_code: int = 200) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


def _error_response(status_code: int) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.raise_for_status.side_effect = requests.HTTPError(
        f"{status_code} Client Error"
    )
    return response


class TestGetCurrentPrice:
    def test_returns_money_from_finnhub_quote(self, adapter, mocker):
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"c": 192.53, "d": 1.2, "dp": 0.63}),
        )

        price = adapter.get_current_price("AAPL")

        assert isinstance(price, Money)
        assert price.amount == Decimal("192.53")
        mock_get.assert_called_once()

    def test_passes_symbol_and_token_as_query_params(self, adapter, mocker):
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"c": 100.0}),
        )

        adapter.get_current_price("VOO")

        call = mock_get.call_args
        assert call.kwargs["params"] == {"symbol": "VOO", "token": "test-token-123"}
        assert call.kwargs["url"].endswith("/quote") if "url" in call.kwargs else \
               call.args[0].endswith("/quote")

    def test_sets_request_timeout(self, adapter, mocker):
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"c": 100.0}),
        )

        adapter.get_current_price("AAPL")

        assert "timeout" in mock_get.call_args.kwargs
        assert mock_get.call_args.kwargs["timeout"] > 0

    def test_raises_on_http_error(self, adapter, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=_error_response(401),
        )

        with pytest.raises(requests.HTTPError):
            adapter.get_current_price("AAPL")

    def test_raises_value_error_for_403_unsupported_ticker(self, adapter, mocker):
        # Finnhub free tier returns 403 for LSE tickers like CSPX.L
        mocker.patch(
            "requests.Session.get",
            return_value=_error_response(403),
        )

        with pytest.raises(ValueError, match="not supported"):
            adapter.get_current_price("CSPX.L")

    def test_raises_value_error_when_response_lacks_price(self, adapter, mocker):
        # Finnhub przy nieznanym tickerze zwraca 200 + {"c": 0, ...} albo brak klucza
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"d": 0, "dp": 0}),
        )

        with pytest.raises(ValueError, match="price"):
            adapter.get_current_price("UNKNOWN")

    def test_raises_value_error_when_price_is_zero(self, adapter, mocker):
        # Finnhub: 0 oznacza brak danych
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"c": 0, "d": 0, "dp": 0}),
        )

        with pytest.raises(ValueError, match="price"):
            adapter.get_current_price("DELISTED")


class TestAdapterImplementsPort:
    def test_is_market_data_port(self):
        from src.application.ports import MarketDataPort

        assert issubclass(FinnhubAdapter, MarketDataPort)


@pytest.mark.integration
class TestFinnhubLive:
    """Integracyjne — wymaga FINNHUB_API_KEY w środowisku. Pomijane domyślnie."""

    def test_fetches_aapl_real_price(self):
        import os

        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            pytest.skip("FINNHUB_API_KEY not set")

        adapter = FinnhubAdapter(api_key=api_key)
        price = adapter.get_current_price("AAPL")
        assert price.amount > 0
