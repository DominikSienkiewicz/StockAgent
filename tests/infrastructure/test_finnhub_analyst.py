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
        assert consensus.price_target == pytest.approx(220.0)

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
        assert consensus.price_target == pytest.approx(215.0)

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


def test_network_error_log_omits_token(mocker, caplog) -> None:
    # BEZ-07: `logger.exception` drukuje komunikat i traceback błędu sieci, a ten
    # niesie URL z `&token=` — logi GHA są publiczne.
    import logging

    adapter = FinnhubAnalystAdapter(api_key="analyst-secret-token")
    mocker.patch(
        "requests.Session.get",
        side_effect=requests.ConnectionError(
            "Max retries exceeded with url: "
            "/api/v1/stock/recommendation?symbol=AAPL&token=analyst-secret-token"
        ),
    )

    with caplog.at_level(logging.WARNING):
        assert adapter.get_consensus("AAPL") is None

    assert "Finnhub analyst consensus failed for AAPL" in caplog.text
    assert "analyst-secret-token" not in caplog.text


# --- DAT-12: wspólny limiter i alerty 401/403/429 -----------------------------


def test_429_on_recommendation_emits_quota_alert() -> None:
    # 429 był połykany jak brak danych — wspólny limit Finnhub znikał bez śladu.
    from unittest.mock import patch

    from src.application.quota_monitor import QuotaMonitor
    from src.domain.quota import QuotaSeverity

    monitor = QuotaMonitor()
    adapter = FinnhubAnalystAdapter(api_key="test-token-123", quota_monitor=monitor)
    with patch.object(adapter._session, "get", return_value=_status_response(429)):
        assert adapter.get_consensus("AAPL") is None

    assert len(monitor.alerts) == 1
    alert = monitor.alerts[0]
    assert alert.severity is QuotaSeverity.WARNING
    assert alert.source.startswith("Finnhub")
    assert "429" in alert.message
    assert "test-token-123" not in alert.message + alert.action


@pytest.mark.parametrize("status", [401, 403])
def test_auth_error_on_recommendation_warns_once(status: int, caplog) -> None:
    import logging
    from unittest.mock import patch

    from src.application.quota_monitor import QuotaMonitor

    monitor = QuotaMonitor()
    adapter = FinnhubAnalystAdapter(api_key="test-token-123", quota_monitor=monitor)
    with (
        caplog.at_level(logging.WARNING),
        patch.object(adapter._session, "get", return_value=_status_response(status)),
    ):
        assert adapter.get_consensus("AAPL") is None
        assert adapter.get_consensus("MSFT") is None

    warnings = [r for r in caplog.records if str(status) in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert monitor.alerts == []


def test_every_request_takes_a_token_from_the_shared_limiter(mocker) -> None:
    limiter = Mock()
    adapter = FinnhubAnalystAdapter(api_key="test-token-123", rate_limiter=limiter)
    mocker.patch(
        "requests.Session.get",
        side_effect=[
            _ok_response(_RECOMMENDATION_PAYLOAD),
            _ok_response({"targetMedian": 220.0}),
        ],
    )

    adapter.get_consensus("AAPL")

    assert limiter.acquire.call_count == 2


class TestPortErrorTaxonomy:
    """ARCH-33: adapter łapie wyłącznie przetłumaczone błędy dostawcy (sieć, status,
    śmieciowa treść → None); błąd programisty nie znika jako „brak konsensusu”."""

    def test_unparseable_recommendation_body_degrades_to_none(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        broken = _ok_response([])
        broken.json.side_effect = requests.JSONDecodeError("Expecting value", "<html>", 0)
        mocker.patch("requests.Session.get", return_value=broken)

        assert adapter.get_consensus("AAPL") is None

    def test_programming_error_is_not_swallowed(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        mocker.patch("requests.Session.get", side_effect=TypeError("bug in the adapter"))

        with pytest.raises(TypeError, match="bug in the adapter"):
            adapter.get_consensus("AAPL")


class TestRevision:
    """DAT-13: dwa najświeższe okresy z tej samej odpowiedzi /stock/recommendation
    (deduplikacja po period) — rewizja m/m bez nowego requestu."""

    @staticmethod
    def _period(period: str, **counts: int) -> dict[str, object]:
        return {"period": period, **counts}

    def test_revision_up_when_latest_period_has_more_buy_votes(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        from src.domain.analyst_consensus import RevisionDirection

        payload = [
            self._period("2026-09-01", buy=10, hold=10),
            self._period("2026-10-01", strongBuy=3, buy=12, hold=5),
            self._period("2026-08-01", buy=12, hold=8),
        ]
        get = mocker.patch(
            "requests.Session.get", side_effect=[_ok_response(payload), _ok_response({})]
        )

        consensus = adapter.get_consensus("AAPL")

        assert consensus is not None
        assert consensus.revision().direction is RevisionDirection.UPGRADE
        assert consensus.previous is not None and consensus.previous.buy == 10
        assert get.call_count == 2

    def test_duplicate_latest_period_is_not_its_own_previous(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        payload = [
            self._period("2026-10-01", strongBuy=3, buy=12, hold=5),
            self._period("2026-10-01", strongBuy=3, buy=12, hold=5),
            self._period("2026-09-01", buy=10, hold=10),
        ]
        mocker.patch(
            "requests.Session.get", side_effect=[_ok_response(payload), _ok_response({})]
        )

        consensus = adapter.get_consensus("AAPL")

        assert consensus is not None
        assert consensus.previous is not None and consensus.previous.buy == 10

    def test_single_period_has_no_previous(
        self, adapter: FinnhubAnalystAdapter, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=[
                _ok_response([self._period("2026-10-01", buy=4)]),
                _ok_response({}),
            ],
        )

        consensus = adapter.get_consensus("AAPL")

        assert consensus is not None and consensus.previous is None
