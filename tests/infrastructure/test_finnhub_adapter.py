from decimal import Decimal
from typing import Any
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
    def test_returns_money_from_finnhub_quote(self, adapter: FinnhubAdapter, mocker: Any) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"c": 192.53, "d": 1.2, "dp": 0.63}),
        )

        price = adapter.get_current_price("AAPL")

        assert isinstance(price, Money)
        assert price.amount == Decimal("192.53")
        mock_get.assert_called_once()

    def test_passes_symbol_and_token_as_query_params(
        self, adapter: FinnhubAdapter, mocker: Any
    ) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"c": 100.0}),
        )

        adapter.get_current_price("VOO")

        call = mock_get.call_args
        assert call.kwargs["params"] == {"symbol": "VOO", "token": "test-token-123"}
        assert call.kwargs["url"].endswith("/quote") if "url" in call.kwargs else \
               call.args[0].endswith("/quote")

    def test_sets_request_timeout(self, adapter: FinnhubAdapter, mocker: Any) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"c": 100.0}),
        )

        adapter.get_current_price("AAPL")

        assert "timeout" in mock_get.call_args.kwargs
        assert mock_get.call_args.kwargs["timeout"] > 0

    def test_raises_on_http_error(self, adapter: FinnhubAdapter, mocker: Any) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_error_response(401),
        )

        with pytest.raises(requests.HTTPError):
            adapter.get_current_price("AAPL")

    def test_raises_value_error_for_403_unsupported_ticker(
        self, adapter: FinnhubAdapter, mocker: Any
    ) -> None:
        # Finnhub free tier returns 403 for LSE tickers like CSPX.L
        mocker.patch(
            "requests.Session.get",
            return_value=_error_response(403),
        )

        with pytest.raises(ValueError, match="not supported"):
            adapter.get_current_price("CSPX.L")

    def test_raises_value_error_when_response_lacks_price(
        self, adapter: FinnhubAdapter, mocker: Any
    ) -> None:
        # Finnhub przy nieznanym tickerze zwraca 200 + {"c": 0, ...} albo brak klucza
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"d": 0, "dp": 0}),
        )

        with pytest.raises(ValueError, match="price"):
            adapter.get_current_price("UNKNOWN")

    def test_raises_value_error_when_price_is_zero(
        self, adapter: FinnhubAdapter, mocker: Any
    ) -> None:
        # Finnhub: 0 oznacza brak danych
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"c": 0, "d": 0, "dp": 0}),
        )

        with pytest.raises(ValueError, match="price"):
            adapter.get_current_price("DELISTED")

    def test_http_error_message_omits_token(self, adapter: FinnhubAdapter, mocker: Any) -> None:
        # BEZ-07: prawdziwy Response — `requests` wkleja URL z `&token=` do komunikatu,
        # który trafia do maila (błąd symbolu) i publicznych logów GHA.
        import traceback

        response = requests.Response()
        response.status_code = 401
        response.reason = "Unauthorized"
        response.url = "https://finnhub.io/api/v1/quote?symbol=AAPL&token=test-token-123"
        mocker.patch("requests.Session.get", return_value=response)

        with pytest.raises(requests.HTTPError) as excinfo:
            adapter.get_current_price("AAPL")

        assert "401 Client Error" in str(excinfo.value)
        assert "test-token-123" not in str(excinfo.value)
        assert "test-token-123" not in "".join(traceback.format_exception(excinfo.value))

    def test_network_error_message_omits_token(self, adapter: FinnhubAdapter, mocker: Any) -> None:
        # BEZ-07: ConnectionError po wyczerpaniu retry też niesie URL z kluczem.
        mocker.patch(
            "requests.Session.get",
            side_effect=requests.ConnectionError(
                "Max retries exceeded with url: /api/v1/quote?symbol=AAPL&token=test-token-123"
            ),
        )

        with pytest.raises(requests.ConnectionError) as excinfo:
            adapter.get_current_price("AAPL")

        assert "test-token-123" not in str(excinfo.value)
        assert excinfo.value.__cause__ is None


class TestAdapterImplementsPort:
    def test_is_market_data_port(self) -> None:
        from src.application.ports import MarketDataPort

        assert issubclass(FinnhubAdapter, MarketDataPort)


@pytest.mark.integration
class TestFinnhubLive:
    """Integracyjne — wymaga FINNHUB_API_KEY w środowisku. Pomijane domyślnie."""

    def test_fetches_aapl_real_price(self) -> None:
        import os

        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            pytest.skip("FINNHUB_API_KEY not set")

        adapter = FinnhubAdapter(api_key=api_key)
        price = adapter.get_current_price("AAPL")
        assert price.amount > 0


class TestQuotaAlertEmit:
    """429 → QuotaAlert.CRITICAL do monitora."""

    def test_emits_critical_on_429(self, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity
        from src.infrastructure.adapters.finnhub_api import FinnhubAdapter

        monitor = QuotaMonitor()
        adapter = FinnhubAdapter(api_key="x", quota_monitor=monitor)

        response = Mock(spec=requests.Response)
        response.status_code = 429
        response.raise_for_status.side_effect = requests.HTTPError("429")
        mocker.patch("requests.Session.get", return_value=response)

        with pytest.raises(requests.HTTPError):
            adapter.get_current_price("AAPL")

        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].source == "Finnhub"
        assert monitor.alerts[0].severity is QuotaSeverity.CRITICAL

    def test_no_alert_on_200(self, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.infrastructure.adapters.finnhub_api import FinnhubAdapter

        monitor = QuotaMonitor()
        adapter = FinnhubAdapter(api_key="x", quota_monitor=monitor)

        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.json.return_value = {"c": 100.0}
        response.raise_for_status = Mock()
        mocker.patch("requests.Session.get", return_value=response)

        adapter.get_current_price("AAPL")

        assert monitor.alerts == []


def test_rate_limit_alert_is_polish_and_logged_in_english(
    mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # MAIL-15: komunikat alertu idzie do maila po polsku, diagnoza do logów po angielsku.
    import logging

    from src.application.quota_monitor import QuotaMonitor

    monitor = QuotaMonitor()
    adapter = FinnhubAdapter(api_key="x", quota_monitor=monitor)
    response = Mock(spec=requests.Response)
    response.status_code = 429
    response.raise_for_status.side_effect = requests.HTTPError("429")
    mocker.patch("requests.Session.get", return_value=response)

    with caplog.at_level(logging.WARNING), pytest.raises(requests.HTTPError):
        adapter.get_current_price("AAPL")

    alert = monitor.alerts[0]
    assert "wyczerpany darmowy limit" in alert.message
    assert "exhausted" not in alert.message
    assert "SYMBOL_THROTTLE_SECONDS" in alert.action
    assert "Finnhub returned 429 for 'AAPL'" in caplog.text


class TestGetDailyChange:
    """ALR-02: `dp` (procent od poprzedniego zamknięcia) i `t` (chwila notowania)
    z tego samego darmowego /quote — watch nie potrzebuje snapshotu Fast Loopa."""

    def test_returns_fraction_and_quote_time(self, adapter: FinnhubAdapter, mocker: Any) -> None:
        from datetime import UTC, datetime

        from src.domain.shock import DailyChange

        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(
                {"c": 182.0, "d": -18.0, "dp": -9.0, "pc": 200.0, "t": 1760630400}
            ),
        )

        reading = adapter.get_daily_change("AAPL")

        assert reading == DailyChange(
            fraction=Decimal("-0.09"),
            as_of=datetime.fromtimestamp(1760630400, tz=UTC),
        )

    def test_raises_value_error_without_quote_time(
        self, adapter: FinnhubAdapter, mocker: Any
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"c": 182.0, "dp": -9.0, "t": 0}),
        )

        with pytest.raises(ValueError, match="daily change"):
            adapter.get_daily_change("AAPL")

    def test_raises_value_error_for_unknown_ticker(
        self, adapter: FinnhubAdapter, mocker: Any
    ) -> None:
        # Finnhub: nieznany ticker to same zera i `dp: null`.
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"c": 0, "d": None, "dp": None, "t": 0}),
        )

        with pytest.raises(ValueError):
            adapter.get_daily_change("NOPE")

    def test_http_error_message_omits_token(self, adapter: FinnhubAdapter, mocker: Any) -> None:
        response = requests.Response()
        response.status_code = 401
        response.reason = "Unauthorized"
        response.url = "https://finnhub.io/api/v1/quote?symbol=AAPL&token=test-token-123"
        mocker.patch("requests.Session.get", return_value=response)

        with pytest.raises(requests.HTTPError) as excinfo:
            adapter.get_daily_change("AAPL")

        assert "test-token-123" not in str(excinfo.value)

    def test_is_daily_change_port(self) -> None:
        from src.application.ports import DailyChangePort

        assert issubclass(FinnhubAdapter, DailyChangePort)


def test_quote_request_takes_a_token_from_the_shared_limiter(mocker: Any) -> None:
    # DAT-12: wszystkie adaptery Finnhub dzielą jeden limiter procesu.
    limiter = Mock()
    adapter = FinnhubAdapter(api_key="test-key", rate_limiter=limiter)
    mocker.patch("requests.Session.get", return_value=_ok_response({"c": 189.5}))

    adapter.get_current_price("AAPL")

    limiter.acquire.assert_called_once()


class TestGetQuote:
    """DAT-14: /quote niesie chwilę notowania (`t`) — `get_current_price` czyta
    tylko `c`, więc w święto NYSE wczorajsze zamknięcie udawało bieżącą cenę."""

    def test_quote_carries_timestamp_from_t_field(
        self, adapter: FinnhubAdapter, mocker: Any
    ) -> None:
        from datetime import UTC, datetime

        from src.domain.value_objects import Quote

        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(
                {"c": 182.5, "pc": 180.0, "o": 181.0, "dp": 1.39, "t": 1760630400}
            ),
        )

        quote = adapter.get_quote("AAPL")

        assert quote == Quote(
            price=Money(Decimal("182.5")),
            quoted_at=datetime.fromtimestamp(1760630400, tz=UTC),
            prev_close=Money(Decimal("180.0")),
            open_price=Money(Decimal("181.0")),
        )

    def test_missing_previous_close_and_open_are_none(
        self, adapter: FinnhubAdapter, mocker: Any
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"c": 182.5, "pc": 0, "t": 1760630400}),
        )

        quote = adapter.get_quote("AAPL")

        assert quote.prev_close is None
        assert quote.open_price is None

    @pytest.mark.parametrize("payload", [{"c": 182.5, "t": 0}, {"c": 0, "t": 1760630400}])
    def test_raises_value_error_without_price_or_time(
        self, adapter: FinnhubAdapter, mocker: Any, payload
    ) -> None:
        mocker.patch("requests.Session.get", return_value=_ok_response(payload))

        with pytest.raises(ValueError, match="quote"):
            adapter.get_quote("AAPL")

    def test_is_quote_port(self) -> None:
        from src.application.ports import QuotePort

        assert issubclass(FinnhubAdapter, QuotePort)


class TestPortErrorTaxonomy:
    """ARCH-33: adapter tłumaczy błędy Finnhub na taksonomię portów, a przetłumaczony
    wyjątek nadal jest typem `requests` (dotychczasowi wywołujący działają)."""

    @staticmethod
    def _real_response(status: int) -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response.reason = "Service Unavailable"
        response.url = "https://finnhub.io/api/v1/quote?symbol=AAPL&token=test-token-123"
        return response

    def test_5xx_maps_to_provider_unavailable(self, adapter: FinnhubAdapter, mocker: Any) -> None:
        from src.application.errors import ProviderUnavailable

        mocker.patch("requests.Session.get", return_value=self._real_response(503))

        with pytest.raises(ProviderUnavailable) as excinfo:
            adapter.get_quote("AAPL")

        assert isinstance(excinfo.value, requests.HTTPError)
        assert "test-token-123" not in str(excinfo.value)

    def test_network_error_maps_to_provider_unavailable(
        self, adapter: FinnhubAdapter, mocker: Any
    ) -> None:
        from src.application.errors import ProviderUnavailable

        mocker.patch(
            "requests.Session.get",
            side_effect=requests.ReadTimeout(
                "Read timed out: /quote?symbol=AAPL&token=test-token-123"
            ),
        )

        with pytest.raises(ProviderUnavailable) as excinfo:
            adapter.get_current_price("AAPL")

        assert isinstance(excinfo.value, requests.ReadTimeout)
        assert "test-token-123" not in str(excinfo.value)
        assert excinfo.value.__cause__ is None

    def test_429_maps_to_quota_exhausted(self, adapter: FinnhubAdapter, mocker: Any) -> None:
        from src.application.errors import QuotaExhausted

        mocker.patch("requests.Session.get", return_value=_error_response(429))

        with pytest.raises(QuotaExhausted):
            adapter.get_daily_change("AAPL")

    def test_other_client_error_is_a_port_error(self, adapter: FinnhubAdapter, mocker: Any) -> None:
        from src.application.errors import PortError, ProviderUnavailable, QuotaExhausted

        mocker.patch("requests.Session.get", return_value=_error_response(401))

        with pytest.raises(PortError) as excinfo:
            adapter.get_current_price("AAPL")

        assert not isinstance(excinfo.value, ProviderUnavailable | QuotaExhausted)

    @pytest.mark.parametrize(
        "response",
        [_error_response(403), _ok_response({"c": 0, "t": 1760630400})],
        ids=["free-tier-403", "zero-price"],
    )
    def test_unknown_ticker_maps_to_not_found(
        self, adapter: FinnhubAdapter, mocker: Any, response
    ) -> None:
        from src.application.errors import NotFound

        mocker.patch("requests.Session.get", return_value=response)

        with pytest.raises(NotFound):
            adapter.get_quote("CSPX.L")

    def test_quote_without_time_is_malformed_payload(
        self, adapter: FinnhubAdapter, mocker: Any
    ) -> None:
        from src.application.errors import MalformedPayload

        mocker.patch("requests.Session.get", return_value=_ok_response({"c": 182.5, "t": 0}))

        with pytest.raises(MalformedPayload, match="quote"):
            adapter.get_quote("AAPL")


def test_open_circuit_fails_fast_without_network_and_without_token(mocker: Any) -> None:
    # NIEZ-31: po 5 porażkach finnhub.io kolejny symbol dostaje ProviderUnavailable
    # od razu — bez 4 prób z timeoutem — a komunikat nadal nie niesie tokenu.
    from requests.adapters import HTTPAdapter

    from src.application.errors import ProviderUnavailable

    transport = mocker.patch.object(
        HTTPAdapter, "send", side_effect=requests.ConnectionError("down")
    )
    adapter = FinnhubAdapter(api_key="test-token-123")
    for _ in range(5):
        with pytest.raises(ProviderUnavailable):
            adapter.get_quote("AAPL")

    with pytest.raises(ProviderUnavailable) as excinfo:
        adapter.get_quote("MSFT")

    assert transport.call_count == 5
    assert "test-token-123" not in str(excinfo.value)
