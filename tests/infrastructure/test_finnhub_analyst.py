"""Testy FinnhubAnalystAdapter — konsensus analityków z Finnhub.

Dwa wywołania GET: /stock/recommendation (lista miesięcznych snapshotów) oraz
/stock/price-target (mediana/średnia celu — bywa płatny → 403). Mockujemy
requests.Session.get z side_effect (najpierw recommendation, potem price-target),
jak inne adaptery HTTP w projekcie. Adapter jest graceful: każdy błąd → None.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from src.application.ports import AnalystConsensusPort
from src.domain.analyst_consensus import AnalystConsensus
from src.infrastructure.adapters.finnhub_analyst import (
    FinnhubAnalystAdapter,
    NullAnalystConsensusAdapter,
)


def _ok_response(payload: object) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


def _status_response(status_code: int) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = {}
    response.raise_for_status = Mock()
    return response


# Dwa snapshoty: starszy (kwiecień) i nowszy (czerwiec). Adapter ma wybrać
# najświeższy po polu "period".
_RECOMMENDATION_PAYLOAD = [
    {
        "strongBuy": 2,
        "buy": 3,
        "hold": 1,
        "sell": 0,
        "strongSell": 0,
        "period": "2024-04-01",
    },
    {
        "strongBuy": 13,
        "buy": 18,
        "hold": 5,
        "sell": 1,
        "strongSell": 0,
        "period": "2024-06-01",
    },
]


@pytest.fixture
def adapter() -> FinnhubAnalystAdapter:
    return FinnhubAnalystAdapter(api_key="test-token-123")


class TestGetConsensus:
    def test_picks_most_recent_recommendation_snapshot(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=[
                _ok_response(_RECOMMENDATION_PAYLOAD),
                _ok_response({"targetMedian": 220.0, "targetMean": 215.0}),
            ],
        )

        consensus = adapter.get_consensus("AAPL")

        assert isinstance(consensus, AnalystConsensus)
        assert consensus.symbol == "AAPL"
        # Najświeższy snapshot to czerwcowy, nie kwietniowy.
        assert consensus.strong_buy == 13
        assert consensus.buy == 18
        assert consensus.hold == 5
        assert consensus.sell == 1
        assert consensus.strong_sell == 0

    def test_sets_price_target_from_target_median(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=[
                _ok_response(_RECOMMENDATION_PAYLOAD),
                _ok_response({"targetMedian": 220.0, "targetMean": 215.0}),
            ],
        )

        consensus = adapter.get_consensus("AAPL")

        assert consensus is not None
        assert consensus.price_target == 220.0

    def test_price_target_403_premium_leaves_target_none(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        # price-target bywa płatny → 403. Konsensus dalej zwracany, tylko bez celu.
        mocker.patch(
            "requests.Session.get",
            side_effect=[
                _ok_response(_RECOMMENDATION_PAYLOAD),
                _status_response(403),
            ],
        )

        consensus = adapter.get_consensus("AAPL")

        assert consensus is not None
        assert consensus.price_target is None
        assert consensus.strong_buy == 13

    def test_empty_price_target_leaves_target_none(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        # 200 + pusty payload → brak celu, konsensus dalej zwracany.
        mocker.patch(
            "requests.Session.get",
            side_effect=[
                _ok_response(_RECOMMENDATION_PAYLOAD),
                _ok_response({}),
            ],
        )

        consensus = adapter.get_consensus("AAPL")

        assert consensus is not None
        assert consensus.price_target is None

    def test_falls_back_to_target_mean_when_median_missing(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=[
                _ok_response(_RECOMMENDATION_PAYLOAD),
                _ok_response({"targetMean": 215.0}),
            ],
        )

        consensus = adapter.get_consensus("AAPL")

        assert consensus is not None
        assert consensus.price_target == 215.0

    def test_empty_recommendation_list_returns_none(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        mocker.patch("requests.Session.get", side_effect=[_ok_response([])])

        assert adapter.get_consensus("AAPL") is None

    def test_non_200_on_recommendation_returns_none(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get", side_effect=[_status_response(403)]
        )

        assert adapter.get_consensus("AAPL") is None

    def test_network_error_returns_none(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=requests.ConnectionError("boom"),
        )

        assert adapter.get_consensus("AAPL") is None

    def test_passes_symbol_and_token_and_timeout(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            side_effect=[
                _ok_response(_RECOMMENDATION_PAYLOAD),
                _ok_response({"targetMedian": 220.0}),
            ],
        )

        adapter.get_consensus("AAPL")

        first = mock_get.call_args_list[0]
        assert first.kwargs["params"]["symbol"] == "AAPL"
        assert first.kwargs["params"]["token"] == "test-token-123"
        assert first.kwargs["timeout"] > 0
        url = first.args[0] if first.args else first.kwargs.get("url", "")
        assert "/stock/recommendation" in url


class TestAdapterImplementsPort:
    def test_is_analyst_consensus_port(self) -> None:
        assert issubclass(FinnhubAnalystAdapter, AnalystConsensusPort)


class TestNullAdapter:
    def test_null_returns_none_with_no_io(self, mocker) -> None:
        mock_get = mocker.patch("requests.Session.get")

        null = NullAnalystConsensusAdapter()

        assert null.get_consensus("AAPL") is None
        mock_get.assert_not_called()

    def test_null_implements_port(self) -> None:
        assert issubclass(NullAnalystConsensusAdapter, AnalystConsensusPort)
