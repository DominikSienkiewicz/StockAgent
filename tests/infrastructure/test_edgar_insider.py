"""Testy EdgarInsiderAdapter — adapter InsiderFlowPort dla SEC EDGAR.

Mockujemy requests.Session.get tak jak inne adaptery HTTP w projekcie.
EDGAR nie wymaga klucza, ale WYMAGA nagłówka User-Agent na każdym żądaniu.
Pipeline ma trzy kroki sieciowe:
  1) company_tickers.json  — ticker → CIK,
  2) submissions/CIK{cik}.json — równoległe tablice filingów,
  3) Form-4 XML — parsowanie transakcji nonDerivative.
Adapter jest „graceful": każdy błąd na dowolnym kroku → None, nigdy raise.
"""

from __future__ import annotations

import os
import tomllib
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from src.application.ports import InsiderFlowPort
from src.domain.insider_flow import InsiderFlowSnapshot, InsiderSignal
from src.infrastructure.adapters.edgar_insider import (
    EdgarInsiderAdapter,
    NullInsiderFlowAdapter,
)

USER_AGENT = "StockAgent test test@example.com"
CONFIG_TOML = Path(__file__).resolve().parents[2] / "config.toml"


def _ok_json(payload: dict) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


def _ok_text(text: str) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.text = text
    response.raise_for_status = Mock()
    return response


def _error_response(status_code: int) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    return response


def _company_tickers() -> dict:
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }


def _recent_date(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _submissions(
    *,
    forms: list[str],
    dates: list[str],
    accessions: list[str],
    primary_docs: list[str],
) -> dict:
    return {
        "cik": "320193",
        "filings": {
            "recent": {
                "form": forms,
                "filingDate": dates,
                "accessionNumber": accessions,
                "primaryDocument": primary_docs,
            }
        },
    }


def _form4_xml(*, code: str, shares: float) -> str:
    """Minimalny Form-4 z jedną transakcją nonDerivative.

    transactionCode 'P' = purchase (kupno), 'S' = sale (sprzedaż).
    Celowo z namespace, żeby sprawdzić tolerancję parsera na prefiksy.
    """
    return f"""<?xml version="1.0"?>
<ownershipDocument xmlns="http://www.sec.gov/edgar/ownership">
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
      </transactionAmounts>
      <transactionCoding>
        <transactionCode>{code}</transactionCode>
      </transactionCoding>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def _two_form4_submissions() -> dict:
    # Dwa Form 4 w oknie + jeden inny formularz (10-K) który należy pominąć.
    return _submissions(
        forms=["4", "10-K", "4"],
        dates=[_recent_date(5), _recent_date(10), _recent_date(20)],
        accessions=[
            "0000320193-26-000001",
            "0000320193-26-000002",
            "0000320193-26-000003",
        ],
        primary_docs=["form4_buy.xml", "aapl-10k.htm", "form4_sell.xml"],
    )


@pytest.fixture
def adapter() -> EdgarInsiderAdapter:
    return EdgarInsiderAdapter(user_agent=USER_AGENT, window_days=90)


class TestGetInsiderFlow:
    def test_aggregates_buy_and_sell(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        buy_xml = _form4_xml(code="P", shares=1000.0)
        sell_xml = _form4_xml(code="S", shares=400.0)

        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            if "form4_buy.xml" in url:
                return _ok_text(buy_xml)
            if "form4_sell.xml" in url:
                return _ok_text(sell_xml)
            raise AssertionError(f"unexpected url {url}")

        mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        assert isinstance(snap, InsiderFlowSnapshot)
        assert snap.symbol == "AAPL"
        assert snap.buy_count == 1
        assert snap.sell_count == 1
        # +1000 kupno, -400 sprzedaż → +600 netto.
        assert snap.net_shares == pytest.approx(600.0)
        assert snap.window_days == 90
        assert snap.evaluate_signal() is InsiderSignal.NET_BUYING

    def test_net_selling_when_sales_dominate(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        buy_xml = _form4_xml(code="P", shares=100.0)
        sell_xml = _form4_xml(code="S", shares=900.0)

        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            if "form4_buy.xml" in url:
                return _ok_text(buy_xml)
            return _ok_text(sell_xml)

        mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert snap.net_shares == pytest.approx(-800.0)
        assert snap.evaluate_signal() is InsiderSignal.NET_SELLING

    def test_skips_filings_outside_window(self, mocker) -> None:
        # window_days=15: filing sprzed 20 dni wypada poza okno.
        adapter = EdgarInsiderAdapter(user_agent=USER_AGENT, window_days=15)
        buy_xml = _form4_xml(code="P", shares=1000.0)

        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            return _ok_text(buy_xml)

        mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        # Tylko Form 4 sprzed 5 dni mieści się w oknie 15d.
        assert snap.buy_count == 1
        assert snap.sell_count == 0

    def test_sets_user_agent_header_on_every_call(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        buy_xml = _form4_xml(code="P", shares=1.0)

        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            return _ok_text(buy_xml)

        mocker.patch("requests.Session.get", side_effect=get)

        adapter.get_insider_flow("AAPL")

        # User-Agent ustawiony na sesji (SEC wymaga go na każdym żądaniu).
        assert USER_AGENT in adapter._session.headers.get("User-Agent", "")

    def test_sets_timeout_on_every_call(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        buy_xml = _form4_xml(code="P", shares=1.0)

        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            return _ok_text(buy_xml)

        mock_get = mocker.patch("requests.Session.get", side_effect=get)

        adapter.get_insider_flow("AAPL")

        for call in mock_get.call_args_list:
            assert "timeout" in call.kwargs
            assert call.kwargs["timeout"] > 0

    def test_zero_pads_cik_in_submissions_url(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        buy_xml = _form4_xml(code="P", shares=1.0)

        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            return _ok_text(buy_xml)

        mock_get = mocker.patch("requests.Session.get", side_effect=get)

        adapter.get_insider_flow("AAPL")

        urls = [
            c.args[0] if c.args else c.kwargs.get("url", "")
            for c in mock_get.call_args_list
        ]
        # CIK 320193 → 0000320193 (10 cyfr) w URL submissions.
        assert any("CIK0000320193.json" in u for u in urls)

    def test_unknown_ticker_returns_none(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_json(_company_tickers()),
        )

        snap = adapter.get_insider_flow("ZZZZ")

        assert snap is None

    def test_ticker_lookup_http_error_returns_none(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get", return_value=_error_response(503)
        )

        snap = adapter.get_insider_flow("AAPL")

        assert snap is None

    def test_submissions_http_error_returns_none(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            return _error_response(500)

        mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        assert snap is None

    def test_malformed_xml_is_skipped_partial_aggregation(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        # Jeden Form 4 ma zepsuty XML — pomijamy go, drugi liczy się normalnie.
        # form4_buy.xml jest zepsuty; form4_sell.xml to poprawna sprzedaż.
        good_sell_xml = _form4_xml(code="S", shares=500.0)

        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            if "form4_buy.xml" in url:
                return _ok_text("<<<not xml at all>>>")
            return _ok_text(good_sell_xml)

        mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        # Tylko sprzedaż (500) sparsowana; kupno pominięte z powodu zepsutego XML.
        assert snap.sell_count == 1
        assert snap.buy_count == 0
        assert snap.net_shares == pytest.approx(-500.0)

    def test_all_xml_failing_still_returns_snapshot_with_counts(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        # Wszystkie Form-4 XML padają na fetchu → snapshot z zerowymi licznikami.
        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            return _error_response(404)

        mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        assert isinstance(snap, InsiderFlowSnapshot)
        assert snap.net_shares == pytest.approx(0.0)
        assert snap.buy_count == 0
        assert snap.sell_count == 0
        assert snap.evaluate_signal() is InsiderSignal.NEUTRAL

    def test_no_form4_filings_returns_zero_snapshot(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        only_10k = _submissions(
            forms=["10-K", "8-K"],
            dates=[_recent_date(3), _recent_date(4)],
            accessions=["0000320193-26-000010", "0000320193-26-000011"],
            primary_docs=["a.htm", "b.htm"],
        )

        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            return _ok_json(only_10k)

        mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert snap.buy_count == 0
        assert snap.sell_count == 0
        assert snap.evaluate_signal() is InsiderSignal.NEUTRAL

    def test_respects_max_form4_cap(self, mocker) -> None:
        # max_form4=1: bierzemy tylko najnowszy Form 4 (kupno sprzed 5 dni).
        adapter = EdgarInsiderAdapter(
            user_agent=USER_AGENT, window_days=90, max_form4=1
        )
        buy_xml = _form4_xml(code="P", shares=1000.0)
        sell_xml = _form4_xml(code="S", shares=400.0)

        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            if "form4_buy.xml" in url:
                return _ok_text(buy_xml)
            return _ok_text(sell_xml)

        mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert snap.buy_count == 1
        assert snap.sell_count == 0
        assert snap.net_shares == pytest.approx(1000.0)

    def test_strips_xsl_rendering_prefix(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        # Kształt z prawdziwego submissions API: primaryDocument Form 4 wskazuje
        # widok renderowany przez XSL (HTML), a surowy XML leży w tym samym
        # katalogu accession pod ostatnim segmentem ścieżki.
        submissions = _submissions(
            forms=["4"],
            dates=[_recent_date(5)],
            accessions=["0000320193-26-000001"],
            primary_docs=["xslF345X05/wk-form4_1760000000.xml"],
        )
        raw_xml = _form4_xml(code="P", shares=1000.0)
        rendered_html = "<html><body><table><tr><td>Form 4<br></td></tr></table>"

        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(submissions)
            if "xslF345X05" in url:
                return _ok_text(rendered_html)
            if url.endswith("/000032019326000001/wk-form4_1760000000.xml"):
                return _ok_text(raw_xml)
            raise AssertionError(f"unexpected url {url}")

        mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert snap.buy_count == 1
        assert snap.net_shares == pytest.approx(1000.0)

    def test_keeps_non_xsl_subdirectory_in_primary_document(
        self, adapter: EdgarInsiderAdapter, mocker
    ) -> None:
        # Zdejmujemy wyłącznie prefiks renderera XSL; inny podkatalog zostaje.
        submissions = _submissions(
            forms=["4"],
            dates=[_recent_date(5)],
            accessions=["0000320193-26-000001"],
            primary_docs=["exhibits/form4.xml"],
        )
        raw_xml = _form4_xml(code="S", shares=300.0)

        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(submissions)
            if url.endswith("/000032019326000001/exhibits/form4.xml"):
                return _ok_text(raw_xml)
            raise AssertionError(f"unexpected url {url}")

        mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert snap.sell_count == 1

    def test_is_insider_flow_port(self, adapter: EdgarInsiderAdapter) -> None:
        assert isinstance(adapter, InsiderFlowPort)


class TestNullInsiderFlowAdapter:
    def test_returns_none_without_io(self, mocker) -> None:
        mock_get = mocker.patch("requests.Session.get")
        null = NullInsiderFlowAdapter()

        assert null.get_insider_flow("AAPL") is None
        mock_get.assert_not_called()

    def test_is_insider_flow_port(self) -> None:
        assert isinstance(NullInsiderFlowAdapter(), InsiderFlowPort)


def _configured_user_agent() -> str:
    """User-Agent z kontaktem tak jak w produkcji: env `EDGAR_USER_AGENT` > config.toml.

    Polityka SEC wymaga realnego kontaktu — placeholder z testów jednostkowych
    skończyłby się 403 i fałszywym wynikiem testu na żywo."""
    override = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if override:
        return override
    config = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    return str(config["edgar_user_agent"])


@pytest.mark.integration
class TestEdgarLive:
    """Integracyjne — prawdziwe, publiczne EDGAR (bez klucza). Pomijane domyślnie.

    Weryfikuje założenie, na którym stoi cały adapter: `primaryDocument` z
    submissions wskazuje surowy XML Form 4. Gdy SEC podaje ścieżkę renderowaną
    przez XSL (`xslF345X05/...`), pod URL-em jest HTML i żaden Form 4 się nie
    sparsuje — adapter po cichu zwraca NEUTRAL dla każdej spółki."""

    def test_live_form4_primary_document_parses(self) -> None:
        adapter = EdgarInsiderAdapter(
            user_agent=_configured_user_agent(), window_days=180, max_form4=5
        )

        cik = adapter._resolve_cik("AAPL")
        if cik is None:
            pytest.skip("EDGAR unreachable (ticker lookup failed)")
        filings = adapter._recent_form4_filings(cik)
        if not filings:
            pytest.skip("EDGAR returned no recent Form 4 filings for AAPL")

        # Nagrywamy treść pobranych dokumentów Form 4, żeby sprawdzić nie tylko
        # „sparsowało się", ale że to faktycznie XML ownershipDocument, nie HTML.
        archive_bodies: list[str] = []
        real_get = adapter._session.get

        def recording_get(url: str, **kwargs: object) -> requests.Response:
            response = real_get(url, **kwargs)
            if "/Archives/" in url:
                archive_bodies.append(response.text)
            return response

        adapter._session.get = recording_get  # type: ignore[method-assign]

        parsed = [adapter._parse_form4(cik, acc, doc) for acc, doc in filings]

        roots = [_xml_root_name(body) for body in archive_bodies]
        assert "ownershipDocument" in roots, (
            f"żaden Form 4 nie jest surowym XML-em; primaryDocument={filings}, "
            f"root={roots}"
        )
        assert any(result is not None for result in parsed)


def _xml_root_name(body: str) -> str | None:
    """Lokalna nazwa korzenia XML albo None, gdy treść nie jest XML-em (np. HTML)."""
    try:
        return ET.fromstring(body).tag.rsplit("}", 1)[-1]
    except ET.ParseError:
        return None
