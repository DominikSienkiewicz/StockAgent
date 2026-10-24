"""Testy NbpClient — adapter MacroIndicatorsPort dla api.nbp.pl.

Mocujemy requests.Session.get tak jak inne adaptery HTTP w projekcie.
NBP nie wymaga klucza API, free tier bez limitu — adapter jest cienki,
liczy się tylko poprawność parsowania JSONa i mapowanie na domenę.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import Mock

import pytest
import requests

from src.domain.fx import FxRate
from src.domain.polish_macro import PolishMacroSnapshot
from src.infrastructure.adapters.nbp_client import (
    MAX_LOOKBACK_DAYS,
    NbpClient,
    NbpFxRateClient,
)


def _ok_response(payload: dict) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


def _error_response(status_code: int) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.raise_for_status.side_effect = requests.HTTPError(
        f"{status_code} error"
    )
    return response


def _nbp_rates_payload(currency: str, mids: list[float]) -> dict:
    """Format zgodny z api.nbp.pl/exchangerates/rates/A/{curr}/last/N/."""
    return {
        "table": "A",
        "currency": currency,
        "code": currency.upper(),
        "rates": [
            {"no": f"{i:03d}/A/NBP/2026", "effectiveDate": f"2026-05-{i:02d}", "mid": m}
            for i, m in enumerate(mids, start=1)
        ],
    }


@pytest.fixture
def client() -> NbpClient:
    return NbpClient()


class TestFetchPolishMacro:
    def test_returns_snapshot_with_latest_and_change_pct(
        self, client: NbpClient, mocker
    ) -> None:
        eur = _nbp_rates_payload("euro", [4.30, 4.32, 4.35, 4.40])
        usd = _nbp_rates_payload("dolar amerykański", [4.00, 3.95, 4.05, 4.10])

        def get(url: str, **_: object) -> Mock:
            if "EUR" in url:
                return _ok_response(eur)
            return _ok_response(usd)

        mocker.patch("requests.Session.get", side_effect=get)

        snap = client.fetch_polish_macro()

        assert isinstance(snap, PolishMacroSnapshot)
        assert snap.eur_pln == Decimal("4.40")
        assert snap.usd_pln == Decimal("4.10")
        # (4.40 - 4.30) / 4.30 ≈ 0.0233
        assert snap.eur_pln_30d_change_pct.quantize(Decimal("0.0001")) == Decimal(
            "0.0233"
        )
        # (4.10 - 4.00) / 4.00 == 0.025
        assert snap.usd_pln_30d_change_pct == Decimal("0.025")

    def test_calls_both_currencies(self, client: NbpClient, mocker) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_nbp_rates_payload("x", [4.0, 4.1])),
        )

        client.fetch_polish_macro()

        urls = [
            call.args[0] if call.args else call.kwargs.get("url", "")
            for call in mock_get.call_args_list
        ]
        assert any("EUR" in u for u in urls)
        assert any("USD" in u for u in urls)

    def test_sets_timeout(self, client: NbpClient, mocker) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_nbp_rates_payload("x", [4.0, 4.1])),
        )

        client.fetch_polish_macro()

        for call in mock_get.call_args_list:
            assert "timeout" in call.kwargs
            assert call.kwargs["timeout"] > 0

    def test_returns_none_on_http_error(self, client: NbpClient, mocker) -> None:
        """Brak NBP nie powinien wywalić raportu — port pozwala na None."""
        mocker.patch(
            "requests.Session.get",
            return_value=_error_response(503),
        )

        snap = client.fetch_polish_macro()

        assert snap is None

    def test_returns_none_on_empty_rates(self, client: NbpClient, mocker) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"rates": []}),
        )

        snap = client.fetch_polish_macro()

        assert snap is None

    def test_returns_none_on_single_rate_no_baseline(
        self, client: NbpClient, mocker
    ) -> None:
        """Jeden punkt = brak referencji do liczenia zmiany 30d."""
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_nbp_rates_payload("x", [4.0])),
        )

        snap = client.fetch_polish_macro()

        assert snap is None

    def test_url_path_uses_table_a_and_30_days(
        self, client: NbpClient, mocker
    ) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_nbp_rates_payload("x", [4.0, 4.1])),
        )

        client.fetch_polish_macro()

        for call in mock_get.call_args_list:
            url = call.args[0] if call.args else call.kwargs.get("url", "")
            assert "/exchangerates/rates/A/" in url
            assert "last/30" in url


# --- PORT-11: kurs z dnia (FxRatePort) ----------------------------------------


def _day_payload(code: str, effective: str, mid: float) -> dict:
    """Format api.nbp.pl/api/exchangerates/rates/a/{code}/{date}/."""
    return {
        "table": "A",
        "currency": "dolar amerykański",
        "code": code,
        "rates": [{"no": "043/A/NBP/2025", "effectiveDate": effective, "mid": mid}],
    }


def _not_found() -> Mock:
    return _error_response(404)


def _urls(mock_get: Mock) -> list[str]:
    return [call.args[0] for call in mock_get.call_args_list]


class TestFxRateForDay:
    def test_returns_the_mid_rate_of_the_requested_day(self, mocker) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_day_payload("USD", "2025-03-03", 3.9725)),
        )

        rate = NbpFxRateClient().get_rate("USD", date(2025, 3, 3))

        assert rate == FxRate(
            currency="USD",
            on_date=date(2025, 3, 3),
            effective_date=date(2025, 3, 3),
            rate=Decimal("3.9725"),
        )
        assert _urls(mock_get) == [
            "https://api.nbp.pl/api/exchangerates/rates/a/usd/2025-03-03/?format=json"
        ]
        assert mock_get.call_args.kwargs["timeout"] > 0

    def test_steps_back_to_the_previous_business_day(self, mocker) -> None:
        # Niedziela 2025-03-02 i sobota 03-01 bez tabeli → kurs z piątku 02-28.
        mock_get = mocker.patch(
            "requests.Session.get",
            side_effect=[
                _not_found(),
                _not_found(),
                _ok_response(_day_payload("EUR", "2025-02-28", 4.1512)),
            ],
        )

        rate = NbpFxRateClient().get_rate("eur", date(2025, 3, 2))

        assert rate is not None
        assert (rate.currency, rate.on_date, rate.effective_date, rate.rate) == (
            "EUR", date(2025, 3, 2), date(2025, 2, 28), Decimal("4.1512"),
        )
        assert [url.split("/")[-2] for url in _urls(mock_get)] == [
            "2025-03-02", "2025-03-01", "2025-02-28",
        ]

    def test_gives_up_after_the_lookback_window(self, mocker) -> None:
        mock_get = mocker.patch("requests.Session.get", return_value=_not_found())

        assert NbpFxRateClient().get_rate("XYZ", date(2025, 3, 3)) is None
        assert mock_get.call_count == MAX_LOOKBACK_DAYS + 1

    def test_network_error_is_none_and_stops_stepping_back(self, mocker) -> None:
        mock_get = mocker.patch(
            "requests.Session.get", side_effect=requests.ConnectionError("down")
        )

        assert NbpFxRateClient().get_rate("USD", date(2025, 3, 3)) is None
        assert mock_get.call_count == 1

    def test_server_error_is_none(self, mocker) -> None:
        mocker.patch("requests.Session.get", return_value=_error_response(503))

        assert NbpFxRateClient().get_rate("USD", date(2025, 3, 3)) is None

    @pytest.mark.parametrize(
        "payload",
        [
            {"rates": []},
            {"rates": [{"effectiveDate": "2025-03-03"}]},
            {"rates": [{"effectiveDate": "zły", "mid": 3.9}]},
            {"rates": [{"effectiveDate": "2025-03-03", "mid": 0}]},
        ],
    )
    def test_malformed_payload_is_none(self, mocker, payload: dict) -> None:
        mocker.patch("requests.Session.get", return_value=_ok_response(payload))

        assert NbpFxRateClient().get_rate("USD", date(2025, 3, 3)) is None

    def test_pln_needs_no_request(self, mocker) -> None:
        mock_get = mocker.patch("requests.Session.get")

        rate = NbpFxRateClient().get_rate("PLN", date(2025, 3, 3))

        assert rate is not None and rate.rate == Decimal("1")
        mock_get.assert_not_called()

    def test_after_a_network_error_the_cycle_stops_asking_nbp(self, mocker) -> None:
        # Jeden klient na cykl: przy leżącym NBP każde zapytanie to timeout z retry,
        # a dziesiątki dat zakupu wysadziłyby limit czasu joba.
        mock_get = mocker.patch(
            "requests.Session.get", side_effect=requests.ConnectTimeout("timeout")
        )
        client = NbpFxRateClient()

        assert client.get_rate("USD", date(2025, 3, 3)) is None
        assert client.get_rate("EUR", date(2024, 6, 3)) is None
        assert mock_get.call_count == 1


class TestMalformedBodyIsGraceful:
    """DAT-26: `MacroIndicatorsPort` zwraca None, gdy źródło nie daje danych —
    nieparsowalny JSON i wpis kursu bez `mid` też, tak jak 5xx albo pusta tabela
    (ARCH-33 zostawia klasę błędu w logu degradacji, zob. `TestPortErrorTaxonomy`)."""

    def test_malformed_json_returns_empty(self, client: NbpClient, mocker, caplog) -> None:
        broken = _ok_response({})
        broken.json.side_effect = requests.JSONDecodeError("Expecting value", "<html>", 0)
        mocker.patch("requests.Session.get", return_value=broken)

        assert client._fetch_rates("EUR") == []
        assert any(record.exc_info for record in caplog.records)

    def test_json_that_is_not_an_object_returns_empty(self, client: NbpClient, mocker) -> None:
        mocker.patch("requests.Session.get", return_value=_ok_response(["rates"]))

        assert client.fetch_polish_macro() is None


class TestPortErrorTaxonomy:
    """ARCH-33: błędy NBP w taksonomii portów — awaria logowana jako klasa
    taksonomii, śmieciowa treść tabeli kursów wychodzi jako `MalformedPayload`."""

    def test_server_error_is_logged_as_provider_unavailable(
        self, client: NbpClient, mocker, caplog
    ) -> None:
        from src.application.errors import ProviderUnavailable

        mocker.patch("requests.Session.get", return_value=_error_response(503))

        assert client.fetch_polish_macro() is None
        exc_info = caplog.records[-1].exc_info
        assert exc_info is not None and exc_info[0] is not None
        assert issubclass(exc_info[0], ProviderUnavailable)

    def test_unparseable_rates_body_is_malformed_payload(
        self, client: NbpClient, mocker, caplog
    ) -> None:
        from src.application.errors import MalformedPayload

        broken = _ok_response({})
        broken.json.side_effect = requests.JSONDecodeError("Expecting value", "<html>", 0)
        mocker.patch("requests.Session.get", return_value=broken)

        assert client.fetch_polish_macro() is None
        exc_info = caplog.records[-1].exc_info
        assert exc_info is not None and exc_info[0] is not None
        assert issubclass(exc_info[0], MalformedPayload)

    def test_rate_without_mid_is_malformed_payload(
        self, client: NbpClient, mocker, caplog
    ) -> None:
        from src.application.errors import MalformedPayload

        payload = {"rates": [{"effectiveDate": "2026-05-01"}, {"effectiveDate": "2026-05-02"}]}
        mocker.patch("requests.Session.get", return_value=_ok_response(payload))

        assert client.fetch_polish_macro() is None
        exc_info = caplog.records[-1].exc_info
        assert exc_info is not None and exc_info[0] is not None
        assert issubclass(exc_info[0], MalformedPayload)

    def test_fx_rate_network_error_is_logged_as_provider_unavailable(
        self, mocker, caplog
    ) -> None:
        from src.application.errors import ProviderUnavailable

        mocker.patch("requests.Session.get", side_effect=requests.ConnectionError("down"))

        assert NbpFxRateClient().get_rate("USD", date(2026, 5, 4)) is None
        exc_info = caplog.records[-1].exc_info
        assert exc_info is not None and exc_info[0] is not None
        assert issubclass(exc_info[0], ProviderUnavailable)
