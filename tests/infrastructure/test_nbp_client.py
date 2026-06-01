"""Testy NbpClient — adapter MacroIndicatorsPort dla api.nbp.pl.

Mocujemy requests.Session.get tak jak inne adaptery HTTP w projekcie.
NBP nie wymaga klucza API, free tier bez limitu — adapter jest cienki,
liczy się tylko poprawność parsowania JSONa i mapowanie na domenę.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest
import requests

from src.domain.polish_macro import PolishMacroSnapshot
from src.infrastructure.adapters.nbp_client import NbpClient


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
