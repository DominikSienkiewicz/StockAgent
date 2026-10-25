"""DAT-18: Deribit DVOL (publiczne API, bez klucza) → implikowana zmienność krypto.

DVOL to 30-dniowa annualizowana IV w punktach procentowych (55.0 = 55%); port
opcji niesie ją jako ułamek (0.55). Mockujemy `requests.Session.get`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from src.application.ports import OptionsFlowPort
from src.infrastructure.adapters.deribit_dvol import DeribitDvolAdapter

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


def _ok(payload: object) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


def _dvol(*closes: float) -> dict[str, object]:
    """Odpowiedź get_volatility_index_data: [timestamp ms, open, high, low, close]."""
    start = 1_761_390_000_000
    data = [[start + i * 3_600_000, c, c, c, c] for i, c in enumerate(closes)]
    return {"jsonrpc": "2.0", "result": {"data": data, "continuation": None}}


@pytest.fixture
def adapter() -> DeribitDvolAdapter:
    return DeribitDvolAdapter(clock=lambda: NOW)


def test_is_an_options_flow_port(adapter: DeribitDvolAdapter) -> None:
    assert isinstance(adapter, OptionsFlowPort)


def test_maps_latest_dvol_to_annualized_implied_vol(
    adapter: DeribitDvolAdapter, mocker: Any
) -> None:
    mocker.patch("requests.Session.get", return_value=_ok(_dvol(52.0, 55.0)))

    snapshot = adapter.get_options_flow("BTC")

    assert snapshot is not None
    assert snapshot.symbol == "BTC"
    assert snapshot.implied_vol == pytest.approx(0.55)
    assert snapshot.put_call_ratio is None


def test_asks_for_the_currency_index_over_the_last_hours(
    adapter: DeribitDvolAdapter, mocker: Any
) -> None:
    get = mocker.patch("requests.Session.get", return_value=_ok(_dvol(60.0)))

    adapter.get_options_flow("eth")

    url = get.call_args.args[0]
    params = get.call_args.kwargs["params"]
    assert url == "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    assert params["currency"] == "ETH"
    assert params["resolution"] == "3600"
    assert params["end_timestamp"] == int(NOW.timestamp() * 1000)
    assert params["start_timestamp"] < params["end_timestamp"]
    assert get.call_args.kwargs["timeout"] > 0


def test_symbol_without_a_dvol_index_makes_no_request(
    adapter: DeribitDvolAdapter, mocker: Any
) -> None:
    get = mocker.patch("requests.Session.get")

    assert adapter.get_options_flow("SOL") is None
    assert adapter.get_options_flow("AAPL") is None
    get.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {"result": {"data": []}},
        {"result": {}},
        {"error": {"code": 10001, "message": "bad currency"}},
        [],
        {"result": {"data": [[1, 2, 3]]}},
        {"result": {"data": [[1, 0, 0, 0, 0.0]]}},
        {"result": {"data": [[1, 0, 0, 0, "x"]]}},
    ],
    ids=["empty", "no-data", "rpc-error", "not-object", "short-row", "zero", "text"],
)
def test_unusable_answer_is_none(adapter: DeribitDvolAdapter, mocker: Any, payload: object) -> None:
    mocker.patch("requests.Session.get", return_value=_ok(payload))

    assert adapter.get_options_flow("BTC") is None


def test_provider_failure_is_none(adapter: DeribitDvolAdapter, mocker: Any) -> None:
    mocker.patch("requests.Session.get", side_effect=requests.ConnectionError("down"))

    assert adapter.get_options_flow("BTC") is None


def test_programming_error_is_not_swallowed(adapter: DeribitDvolAdapter, mocker: Any) -> None:
    mocker.patch("requests.Session.get", side_effect=TypeError("bug"))

    with pytest.raises(TypeError):
        adapter.get_options_flow("BTC")
