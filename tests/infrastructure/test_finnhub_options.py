"""Testy adaptera Finnhub option-chain → OptionsFlowSnapshot.

option-chain to PREMIUM endpoint Finnhuba — na free tier zwraca 403 / pustą
listę. Adapter ma wtedy graceful no-op (None), nigdy nie rzuca. Mockujemy
`requests.Session.get`, nie gołe `requests`.
"""

from typing import Any
from unittest.mock import Mock

import pytest
import requests

from src.application.ports import OptionsFlowPort
from src.domain.options_flow import OptionsFlowSnapshot, OptionsSentiment
from src.infrastructure.adapters.finnhub_options import (
    FinnhubOptionsAdapter,
    NullOptionsFlowAdapter,
)


@pytest.fixture
def adapter() -> FinnhubOptionsAdapter:
    return FinnhubOptionsAdapter(api_key="test-token-123")


def _response(payload: object, status_code: int = 200) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


def _chain(calls: list[dict[str, Any]], puts: list[dict[str, Any]]) -> dict[str, Any]:
    """Buduje payload option-chain Finnhuba z jednym wygaśnięciem."""
    return {"data": [{"options": {"CALL": calls, "PUT": puts}}]}


class TestGetOptionsFlow:
    def test_aggregates_put_call_ratio(self, adapter: FinnhubOptionsAdapter, mocker: Any) -> None:
        # PUT vol 120, CALL vol 200 → ratio 0.6 → BULLISH.
        payload = _chain(
            calls=[
                {"volume": 150, "impliedVolatility": 0.40},
                {"volume": 50, "impliedVolatility": 0.50},
            ],
            puts=[
                {"volume": 80, "impliedVolatility": 0.42},
                {"volume": 40, "impliedVolatility": 0.48},
            ],
        )
        mocker.patch("requests.Session.get", return_value=_response(payload))

        snapshot = adapter.get_options_flow("AAPL")

        assert isinstance(snapshot, OptionsFlowSnapshot)
        assert snapshot.symbol == "AAPL"
        assert snapshot.put_call_ratio == pytest.approx(0.6)
        assert snapshot.sentiment() is OptionsSentiment.BULLISH

    def test_averages_implied_vol(self, adapter: FinnhubOptionsAdapter, mocker: Any) -> None:
        payload = _chain(
            calls=[{"volume": 100, "impliedVolatility": 0.40}],
            puts=[{"volume": 100, "impliedVolatility": 0.60}],
        )
        mocker.patch("requests.Session.get", return_value=_response(payload))

        snapshot = adapter.get_options_flow("AAPL")

        assert snapshot is not None
        # Średnia z {0.40, 0.60} = 0.50.
        assert snapshot.implied_vol == pytest.approx(0.50)

    def test_passes_symbol_and_token_as_query_params(
        self, adapter: FinnhubOptionsAdapter, mocker: Any
    ) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_response(
                _chain(
                    calls=[{"volume": 100, "impliedVolatility": 0.3}],
                    puts=[{"volume": 50, "impliedVolatility": 0.3}],
                )
            ),
        )

        adapter.get_options_flow("TSLA")

        call = mock_get.call_args
        assert call.kwargs["params"] == {"symbol": "TSLA", "token": "test-token-123"}

    def test_sets_request_timeout(self, adapter: FinnhubOptionsAdapter, mocker: Any) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_response(
                _chain(
                    calls=[{"volume": 100, "impliedVolatility": 0.3}],
                    puts=[{"volume": 50, "impliedVolatility": 0.3}],
                )
            ),
        )

        adapter.get_options_flow("AAPL")

        assert "timeout" in mock_get.call_args.kwargs
        assert mock_get.call_args.kwargs["timeout"] > 0

    def test_returns_none_on_403_premium(self, adapter: FinnhubOptionsAdapter, mocker: Any) -> None:
        # Free tier: option-chain jest płatne → 403, graceful None.
        response = Mock(spec=requests.Response)
        response.status_code = 403
        response.raise_for_status.side_effect = requests.HTTPError("403")
        response.json.return_value = {}
        mocker.patch("requests.Session.get", return_value=response)

        assert adapter.get_options_flow("AAPL") is None

    def test_returns_none_on_empty_data(self, adapter: FinnhubOptionsAdapter, mocker: Any) -> None:
        mocker.patch("requests.Session.get", return_value=_response({"data": []}))

        assert adapter.get_options_flow("AAPL") is None

    def test_returns_none_on_missing_data_field(
        self, adapter: FinnhubOptionsAdapter, mocker: Any
    ) -> None:
        mocker.patch("requests.Session.get", return_value=_response({}))

        assert adapter.get_options_flow("AAPL") is None

    def test_returns_none_on_zero_call_volume_no_div0(
        self, adapter: FinnhubOptionsAdapter, mocker: Any
    ) -> None:
        # CALL vol = 0 → unikamy dzielenia przez zero, zwracamy None.
        payload = _chain(
            calls=[{"volume": 0, "impliedVolatility": 0.4}],
            puts=[{"volume": 100, "impliedVolatility": 0.4}],
        )
        mocker.patch("requests.Session.get", return_value=_response(payload))

        assert adapter.get_options_flow("AAPL") is None

    def test_returns_none_on_network_error(
        self, adapter: FinnhubOptionsAdapter, mocker: Any
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=requests.ConnectionError("boom"),
        )

        assert adapter.get_options_flow("AAPL") is None

    def test_returns_none_on_malformed_json(
        self, adapter: FinnhubOptionsAdapter, mocker: Any
    ) -> None:
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.raise_for_status = Mock()
        response.json.side_effect = ValueError("no json")
        mocker.patch("requests.Session.get", return_value=response)

        assert adapter.get_options_flow("AAPL") is None

    def test_tolerates_missing_volume_fields(
        self, adapter: FinnhubOptionsAdapter, mocker: Any
    ) -> None:
        # Kontrakty bez "volume" liczymy jako 0 wolumenu, ale nie wywalamy się.
        payload = _chain(
            calls=[{"impliedVolatility": 0.4}, {"volume": 200, "impliedVolatility": 0.4}],
            puts=[{"volume": 100, "impliedVolatility": 0.4}],
        )
        mocker.patch("requests.Session.get", return_value=_response(payload))

        snapshot = adapter.get_options_flow("AAPL")

        assert snapshot is not None
        # CALL vol = 0 + 200 = 200, PUT vol = 100 → 0.5.
        assert snapshot.put_call_ratio == pytest.approx(0.5)


class TestNullAdapter:
    def test_returns_none_without_io(self, mocker: Any) -> None:
        mock_get = mocker.patch("requests.Session.get")
        null = NullOptionsFlowAdapter()

        assert null.get_options_flow("AAPL") is None
        mock_get.assert_not_called()


class TestAdaptersImplementPort:
    def test_finnhub_options_is_options_flow_port(self) -> None:
        assert issubclass(FinnhubOptionsAdapter, OptionsFlowPort)

    def test_null_is_options_flow_port(self) -> None:
        assert issubclass(NullOptionsFlowAdapter, OptionsFlowPort)


def test_network_error_debug_log_omits_token(mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
    # BEZ-07: traceback w logu DEBUG też nie może nieść URL-a z `&token=`.
    import logging

    adapter = FinnhubOptionsAdapter(api_key="options-secret-token")
    mocker.patch(
        "requests.Session.get",
        side_effect=requests.ConnectionError(
            "Max retries exceeded with url: "
            "/api/v1/stock/option-chain?symbol=AAPL&token=options-secret-token"
        ),
    )

    with caplog.at_level(logging.DEBUG):
        assert adapter.get_options_flow("AAPL") is None

    assert "Finnhub option-chain failed for AAPL" in caplog.text
    assert "options-secret-token" not in caplog.text


# --- DAT-12: wspólny limiter i alerty 401/403/429 -----------------------------


def test_429_on_option_chain_emits_quota_alert(mocker: Any) -> None:
    from src.application.quota_monitor import QuotaMonitor
    from src.domain.quota import QuotaSeverity

    monitor = QuotaMonitor()
    adapter = FinnhubOptionsAdapter(api_key="test-token-123", quota_monitor=monitor)
    mocker.patch("requests.Session.get", return_value=_response({}, status_code=429))

    assert adapter.get_options_flow("AAPL") is None

    assert [a.severity for a in monitor.alerts] == [QuotaSeverity.CRITICAL]
    assert "429" in monitor.alerts[0].message


def test_premium_403_warns_once_per_adapter(mocker: Any, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    adapter = FinnhubOptionsAdapter(api_key="test-token-123")
    mocker.patch("requests.Session.get", return_value=_response({}, status_code=403))

    with caplog.at_level(logging.WARNING):
        assert adapter.get_options_flow("AAPL") is None
        assert adapter.get_options_flow("MSFT") is None

    warnings = [r for r in caplog.records if "403" in r.getMessage()]
    assert len(warnings) == 1
    assert "test-token-123" not in caplog.text


def test_option_chain_request_takes_a_token_from_the_shared_limiter(mocker: Any) -> None:
    limiter = Mock()
    adapter = FinnhubOptionsAdapter(api_key="test-token-123", rate_limiter=limiter)
    mocker.patch("requests.Session.get", return_value=_response({"data": []}))

    adapter.get_options_flow("AAPL")

    limiter.acquire.assert_called_once()


class TestPortErrorTaxonomy:
    """ARCH-33: adapter łapie wyłącznie przetłumaczone błędy dostawcy (sieć, status,
    śmieciowa treść → None); błąd programisty nie znika jako „brak opcji”."""

    def test_programming_error_is_not_swallowed(
        self, adapter: FinnhubOptionsAdapter, mocker: Any
    ) -> None:
        mocker.patch("requests.Session.get", side_effect=TypeError("bug in the adapter"))

        with pytest.raises(TypeError, match="bug in the adapter"):
            adapter.get_options_flow("AAPL")
