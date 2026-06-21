"""Testy FredClient — adapter MacroRatesPort dla api.stlouisfed.org/fred.

Mocujemy requests.Session.get (jak inne adaptery HTTP w projekcie).
FRED wymaga klucza API. Adapter jest cienki: pobiera najświeższą obserwację
trzech serii (DGS10, DGS2, DFF) i mapuje na YieldCurveSnapshot. FRED oznacza
brak wartości znakiem "." — wtedy dana seria jest None. Adapter jest graceful:
nigdy nie rzuca, zwraca None gdy wszystkie serie padną.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from src.domain.macro_rates import YieldCurveSnapshot, YieldCurveState
from src.infrastructure.adapters.fred_client import (
    FredClient,
    NullMacroRatesAdapter,
)


def _ok_response(value: str) -> Mock:
    """Odpowiedź FRED z jedną obserwacją o danej wartości."""
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = {
        "observations": [{"date": "2026-06-20", "value": value}]
    }
    response.raise_for_status = Mock()
    return response


def _error_response(status_code: int) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.raise_for_status.side_effect = requests.HTTPError(
        f"{status_code} error"
    )
    return response


def _by_series(values: dict[str, Mock]) -> object:
    """Zwraca side_effect mapujący series_id z URL/params na odpowiedź."""

    def get(url: str, **kwargs: object) -> Mock:
        params = kwargs.get("params") or {}
        series_id = ""
        if isinstance(params, dict):
            series_id = str(params.get("series_id", ""))
        if not series_id:
            for sid in values:
                if sid in url:
                    series_id = sid
                    break
        return values[series_id]

    return get


@pytest.fixture
def client() -> FredClient:
    return FredClient(api_key="test-key")


class TestFetchYieldCurve:
    def test_returns_snapshot_inverted(self, client: FredClient, mocker) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=_by_series(
                {
                    "DGS10": _ok_response("4.25"),
                    "DGS2": _ok_response("4.80"),
                    "DFF": _ok_response("5.33"),
                }
            ),
        )

        snap = client.fetch_yield_curve()

        assert isinstance(snap, YieldCurveSnapshot)
        assert snap.ten_year == 4.25
        assert snap.two_year == 4.80
        assert snap.fed_funds == 5.33
        assert snap.spread_10y_2y is not None
        assert round(snap.spread_10y_2y, 2) == -0.55
        assert snap.state() == YieldCurveState.INVERTED

    def test_sets_timeout_on_every_call(
        self, client: FredClient, mocker
    ) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            side_effect=_by_series(
                {
                    "DGS10": _ok_response("4.25"),
                    "DGS2": _ok_response("4.80"),
                    "DFF": _ok_response("5.33"),
                }
            ),
        )

        client.fetch_yield_curve()

        assert mock_get.call_count == 3
        for call in mock_get.call_args_list:
            assert "timeout" in call.kwargs
            assert call.kwargs["timeout"] > 0

    def test_passes_api_key_and_series_params(
        self, client: FredClient, mocker
    ) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            side_effect=_by_series(
                {
                    "DGS10": _ok_response("4.25"),
                    "DGS2": _ok_response("4.80"),
                    "DFF": _ok_response("5.33"),
                }
            ),
        )

        client.fetch_yield_curve()

        # Każde wywołanie idzie na endpoint series/observations z kluczem.
        for call in mock_get.call_args_list:
            url = call.args[0] if call.args else call.kwargs.get("url", "")
            assert "fred/series/observations" in url
            params = call.kwargs.get("params") or {}
            assert isinstance(params, dict)
            assert params.get("api_key") == "test-key"

        sent_series = {
            (call.kwargs.get("params") or {}).get("series_id")
            for call in mock_get.call_args_list
        }
        assert sent_series == {"DGS10", "DGS2", "DFF"}

    def test_missing_value_dot_becomes_none(
        self, client: FredClient, mocker
    ) -> None:
        """FRED zwraca "." dla braku — ta seria ma być None, reszta bez zmian."""
        mocker.patch(
            "requests.Session.get",
            side_effect=_by_series(
                {
                    "DGS10": _ok_response("4.25"),
                    "DGS2": _ok_response("."),
                    "DFF": _ok_response("5.33"),
                }
            ),
        )

        snap = client.fetch_yield_curve()

        assert isinstance(snap, YieldCurveSnapshot)
        assert snap.ten_year == 4.25
        assert snap.two_year is None
        assert snap.fed_funds == 5.33
        # Brak 2Y → spread nieobliczalny, stan neutralny.
        assert snap.spread_10y_2y is None
        assert snap.state() == YieldCurveState.NORMAL

    def test_one_series_non_200_becomes_none(
        self, client: FredClient, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=_by_series(
                {
                    "DGS10": _ok_response("4.25"),
                    "DGS2": _error_response(500),
                    "DFF": _ok_response("5.33"),
                }
            ),
        )

        snap = client.fetch_yield_curve()

        assert isinstance(snap, YieldCurveSnapshot)
        assert snap.ten_year == 4.25
        assert snap.two_year is None
        assert snap.fed_funds == 5.33

    def test_unparseable_value_becomes_none(
        self, client: FredClient, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=_by_series(
                {
                    "DGS10": _ok_response("not-a-number"),
                    "DGS2": _ok_response("4.80"),
                    "DFF": _ok_response("5.33"),
                }
            ),
        )

        snap = client.fetch_yield_curve()

        assert isinstance(snap, YieldCurveSnapshot)
        assert snap.ten_year is None
        assert snap.two_year == 4.80
        assert snap.fed_funds == 5.33

    def test_all_series_non_200_returns_none(
        self, client: FredClient, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_error_response(503),
        )

        snap = client.fetch_yield_curve()

        assert snap is None

    def test_network_error_returns_none(
        self, client: FredClient, mocker
    ) -> None:
        """Wyjątek sieciowy nie może wyciec — to opcjonalne źródło alpha."""
        mocker.patch(
            "requests.Session.get",
            side_effect=requests.ConnectionError("boom"),
        )

        snap = client.fetch_yield_curve()

        assert snap is None

    def test_empty_observations_returns_none(
        self, client: FredClient, mocker
    ) -> None:
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.json.return_value = {"observations": []}
        response.raise_for_status = Mock()
        mocker.patch("requests.Session.get", return_value=response)

        snap = client.fetch_yield_curve()

        assert snap is None

    def test_non_dict_payload_returns_none_without_raising(
        self, client: FredClient, mocker
    ) -> None:
        # FRED za proxy/CDN może zwrócić listę/null zamiast dicta — .get()
        # rzuciłby AttributeError. Optional source NIE MOŻE wywalić cyklu.
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.json.return_value = ["nieoczekiwana lista"]
        response.raise_for_status = Mock()
        mocker.patch("requests.Session.get", return_value=response)

        assert client.fetch_yield_curve() is None

    def test_non_dict_observation_element_returns_none_without_raising(
        self, client: FredClient, mocker
    ) -> None:
        # Element observations może być nie-dictem (string/null) → .get() na nim
        # rzuciłby AttributeError.
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.json.return_value = {"observations": ["bare string"]}
        response.raise_for_status = Mock()
        mocker.patch("requests.Session.get", return_value=response)

        assert client.fetch_yield_curve() is None


class TestNullMacroRatesAdapter:
    def test_returns_none_without_io(self, mocker) -> None:
        mock_get = mocker.patch("requests.Session.get")

        adapter = NullMacroRatesAdapter()

        assert adapter.fetch_yield_curve() is None
        mock_get.assert_not_called()
