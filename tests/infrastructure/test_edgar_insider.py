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
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any
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
        self, adapter: EdgarInsiderAdapter, mocker: Any
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
        self, adapter: EdgarInsiderAdapter, mocker: Any
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

    def test_skips_filings_outside_window(self, mocker: Any) -> None:
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
        self, adapter: EdgarInsiderAdapter, mocker: Any
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
        self, adapter: EdgarInsiderAdapter, mocker: Any
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
        self, adapter: EdgarInsiderAdapter, mocker: Any
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
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_json(_company_tickers()),
        )

        snap = adapter.get_insider_flow("ZZZZ")

        assert snap is None

    def test_ticker_lookup_http_error_returns_none(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        mocker.patch(
            "requests.Session.get", return_value=_error_response(503)
        )

        snap = adapter.get_insider_flow("AAPL")

        assert snap is None

    def test_submissions_http_error_returns_none(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            return _error_response(500)

        mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        assert snap is None

    def test_malformed_xml_is_skipped_partial_aggregation(
        self, adapter: EdgarInsiderAdapter, mocker: Any
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
        self, adapter: EdgarInsiderAdapter, mocker: Any
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
        self, adapter: EdgarInsiderAdapter, mocker: Any
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

    def test_respects_max_form4_cap(self, mocker: Any) -> None:
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
        self, adapter: EdgarInsiderAdapter, mocker: Any
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
        self, adapter: EdgarInsiderAdapter, mocker: Any
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


class TestFilingEvidence:
    """DAT-03: adapter liczy Form 4 widziane w oknie i sparsowane, żeby domena
    odróżniła awarię parsowania od braku transakcji."""

    @staticmethod
    def _patch(mocker: Any, form4_body: dict[str, str]) -> None:
        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            for doc, body in form4_body.items():
                if doc in url:
                    return _ok_text(body)
            return _error_response(404)

        mocker.patch("requests.Session.get", side_effect=get)

    def test_counts_seen_and_parsed_filings(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        self._patch(
            mocker,
            {
                "form4_buy.xml": _form4_xml(code="P", shares=1000.0),
                "form4_sell.xml": _form4_xml(code="S", shares=400.0),
            },
        )

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert (snap.filings_seen, snap.filings_parsed) == (2, 2)
        assert snap.has_evidence()

    def test_all_failing_filings_are_seen_but_not_parsed(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        self._patch(mocker, {})

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert (snap.filings_seen, snap.filings_parsed) == (2, 0)
        assert not snap.has_evidence()

    def test_partial_parse_counts_only_the_parsed_filing(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        self._patch(
            mocker,
            {
                "form4_buy.xml": "<<<not xml at all>>>",
                "form4_sell.xml": _form4_xml(code="S", shares=500.0),
            },
        )

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert (snap.filings_seen, snap.filings_parsed) == (2, 1)
        assert snap.has_evidence()


class TestTickerMapAndPacing:
    """DAT-10: company_tickers.json (kilka MB) pobierany raz na instancję zamiast
    per symbol, a requesty do sec.gov mają minimalny odstęp (limit SEC 10/s)."""

    @staticmethod
    def _get(tickers_status: list[int]) -> tuple[list[str], object]:
        urls: list[str] = []
        statuses = iter(tickers_status)

        def get(url: str, **_: object) -> Mock:
            urls.append(url)
            if "company_tickers.json" in url:
                status = next(statuses, 200)
                return _ok_json(_company_tickers()) if status == 200 else _error_response(status)
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            return _ok_text(_form4_xml(code="P", shares=10.0))

        return urls, get

    def test_ticker_map_fetched_once_per_instance(self, mocker: Any) -> None:
        urls, get = self._get([200])
        mocker.patch("requests.Session.get", side_effect=get)
        adapter = EdgarInsiderAdapter(user_agent=USER_AGENT)

        assert adapter.get_insider_flow("AAPL") is not None
        assert adapter.get_insider_flow("MSFT") is not None

        assert sum("company_tickers.json" in url for url in urls) == 1

    def test_failed_ticker_map_fetch_is_not_cached(self, mocker: Any) -> None:
        urls, get = self._get([503, 200])
        mocker.patch("requests.Session.get", side_effect=get)
        adapter = EdgarInsiderAdapter(user_agent=USER_AGENT)

        assert adapter.get_insider_flow("AAPL") is None
        assert adapter.get_insider_flow("AAPL") is not None

        assert sum("company_tickers.json" in url for url in urls) == 2

    def test_requests_to_sec_keep_the_minimum_interval(self, mocker: Any) -> None:
        _, get = self._get([200])
        mocker.patch("requests.Session.get", side_effect=get)
        clock = [100.0]
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock[0] += seconds

        adapter = EdgarInsiderAdapter(
            user_agent=USER_AGENT, clock=lambda: clock[0], sleep=sleep
        )

        adapter.get_insider_flow("AAPL")

        # 4 requesty (mapa, submissions, 2 × Form 4) przy zamrożonym zegarze:
        # pierwszy bez czekania, każdy kolejny czeka pełny odstęp 0.12 s.
        assert sleeps == pytest.approx([0.12, 0.12, 0.12])

    def test_no_wait_when_requests_are_already_spaced(self, mocker: Any) -> None:
        _, get = self._get([200])
        clock = [0.0]

        def slow_get(url: str, **kwargs: object) -> Mock:
            clock[0] += 0.5  # sieć sama rozkłada requesty
            return get(url, **kwargs)

        mocker.patch("requests.Session.get", side_effect=slow_get)
        sleep = Mock()
        adapter = EdgarInsiderAdapter(user_agent=USER_AGENT, clock=lambda: clock[0], sleep=sleep)

        adapter.get_insider_flow("AAPL")

        sleep.assert_not_called()


def _price_xml(price: float | None) -> str:
    # Cena bywa pusta (sam przypis) — wtedy element nie ma <value>.
    return "" if price is None else f"<value>{price}</value>"


def _form4_owner_xml(
    *,
    transactions: list[tuple[str, float, float | None]],
    officer_title: str | None = None,
    is_officer: bool = False,
    is_director: bool = False,
    ten_percent_owner: bool = False,
    plan: str | None = None,
) -> str:
    """Form 4 z właścicielem i transakcjami `(kod, akcje, cena)` — kształt jak
    w EDGAR (ownershipDocument z reportingOwnerRelationship i aff10b5One)."""
    title = f"<officerTitle>{officer_title}</officerTitle>" if officer_title else ""
    plan_xml = f"<aff10b5One>{plan}</aff10b5One>" if plan is not None else ""
    rows = "".join(
        f"""
    <nonDerivativeTransaction>
      <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare>{_price_xml(price)}</transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>"""
        for code, shares, price in transactions
    )
    return f"""<?xml version="1.0"?>
<ownershipDocument xmlns="http://www.sec.gov/edgar/ownership">
  {plan_xml}
  <reportingOwner>
    <reportingOwnerRelationship>
      <isDirector>{int(is_director)}</isDirector>
      <isOfficer>{int(is_officer)}</isOfficer>
      {title}
      <isTenPercentOwner>{int(ten_percent_owner)}</isTenPercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>{rows}
  </nonDerivativeTable>
</ownershipDocument>"""


class TestDollarValueAndRole:
    """DAT-21: wartość transakcji w USD, rola insidera, wykluczenie planów
    10b5-1 z przepływu wartości i link do Form 4 — z tego samego XML-a."""

    @staticmethod
    def _patch(mocker: Any, buy_xml: str, sell_xml: str) -> None:
        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            if "form4_buy.xml" in url:
                return _ok_text(buy_xml)
            return _ok_text(sell_xml)

        mocker.patch("requests.Session.get", side_effect=get)

    def test_parses_officer_purchase_value(self, adapter: EdgarInsiderAdapter, mocker: Any) -> None:
        buy = _form4_owner_xml(
            transactions=[("P", 10_000.0, 120.0)], officer_title="CEO", is_officer=True
        )
        self._patch(mocker, buy, _form4_owner_xml(transactions=[]))

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert snap.net_value_usd == pytest.approx(1_200_000.0)
        assert snap.largest_buy is not None
        assert snap.largest_buy.role == "CEO"
        assert snap.largest_buy.value_usd == pytest.approx(1_200_000.0)
        assert snap.largest_buy.filing_url == (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/form4_buy.xml"
        )
        assert snap.significant_buy() == snap.largest_buy

    def test_10b5_1_sale_is_excluded_from_net_flow(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        buy = _form4_owner_xml(
            transactions=[("P", 100.0, 50.0)], officer_title="CFO", is_officer=True
        )
        planned_sale = _form4_owner_xml(
            transactions=[("S", 5_000.0, 200.0)],
            officer_title="CEO",
            is_officer=True,
            plan="1",
        )
        self._patch(mocker, buy, planned_sale)

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        # Planowa sprzedaż CEO nie zasila przepływu wartości...
        assert snap.net_value_usd == pytest.approx(5_000.0)
        # ...a net_shares (waga w Alpha Fusion) zostaje jak dotąd — bez A/B bez zmian.
        assert snap.net_shares == pytest.approx(-4_900.0)

    def test_filing_without_plan_marker_counts_as_discretionary(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        # Starsze Form 4 nie mają aff10b5One — brak znacznika to nie dowód planu.
        sale = _form4_owner_xml(transactions=[("S", 1_000.0, 10.0)], is_director=True)
        self._patch(mocker, _form4_owner_xml(transactions=[]), sale)

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert snap.net_value_usd == pytest.approx(-10_000.0)

    def test_ten_percent_owner_purchase_is_not_an_insider_buy(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        fund = _form4_owner_xml(transactions=[("P", 50_000.0, 100.0)], ten_percent_owner=True)
        self._patch(mocker, fund, _form4_owner_xml(transactions=[]))

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert snap.largest_buy is None
        assert snap.net_value_usd == pytest.approx(5_000_000.0)

    def test_grants_and_unpriced_rows_do_not_add_value(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        buy = _form4_owner_xml(
            transactions=[("A", 9_999.0, 0.0), ("P", 10.0, None)],
            officer_title="CEO",
            is_officer=True,
        )
        self._patch(mocker, buy, _form4_owner_xml(transactions=[]))

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert snap.net_value_usd == pytest.approx(0.0)
        assert snap.largest_buy is None
        assert snap.buy_count == 1

    def test_largest_buy_carries_its_filing_date(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        # DAT-21: lead zakupu insidera ma okno N dni od filingu — data z submissions.
        buy = _form4_owner_xml(
            transactions=[("P", 10_000.0, 120.0)], officer_title="CEO", is_officer=True
        )
        self._patch(mocker, buy, _form4_owner_xml(transactions=[]))

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None and snap.largest_buy is not None
        assert snap.largest_buy.filed_on == date.fromisoformat(_recent_date(5))

    def test_latest_filing_url_points_to_the_newest_form4(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        self._patch(mocker, _form4_owner_xml(transactions=[]), _form4_owner_xml(transactions=[]))

        snap = adapter.get_insider_flow("AAPL")

        assert snap is not None
        assert snap.latest_filing_url == (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/form4_buy.xml"
        )


class TestSecFairAccess:
    """Polityka SEC (sec.gov/os/accessing-edgar-data): najwyżej 10 req/s na
    użytkownika, a dostęp wraca dopiero po 10 minutach poniżej progu. Adapter
    żyje jeden cykl, więc 403 kończy EDGAR do końca cyklu — każde kolejne
    żądanie tylko przedłuża blokadę (od 2026-07 każdy symbol dostawał 403)."""

    def test_403_on_ticker_lookup_stops_requests_for_later_symbols(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        mock_get = mocker.patch(
            "requests.Session.get", return_value=_error_response(403)
        )

        snaps = [adapter.get_insider_flow(s) for s in ("AAPL", "MSFT", "NVDA")]

        assert snaps == [None, None, None]
        assert mock_get.call_count == 1

    def test_403_on_form4_returns_none_without_fetching_the_rest(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        def get(url: str, **_: object) -> Mock:
            if "company_tickers.json" in url:
                return _ok_json(_company_tickers())
            if "submissions" in url:
                return _ok_json(_two_form4_submissions())
            return _error_response(403)

        mock_get = mocker.patch("requests.Session.get", side_effect=get)

        snap = adapter.get_insider_flow("AAPL")

        # Snapshot z połową filingów udawałby awarię parsowania (DAT-03).
        assert snap is None
        archive_calls = [c for c in mock_get.call_args_list if "/Archives/" in c.args[0]]
        assert len(archive_calls) == 1

    @staticmethod
    def _no_form4_get(url: str, **_: object) -> Mock:
        if "company_tickers.json" in url:
            return _ok_json(_company_tickers())
        return _ok_json(_submissions(forms=[], dates=[], accessions=[], primary_docs=[]))

    @staticmethod
    def _ticker_downloads(mock_get: Mock) -> int:
        return sum("company_tickers.json" in c.args[0] for c in mock_get.call_args_list)

    def test_ticker_map_is_downloaded_once_per_cycle(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        mock_get = mocker.patch("requests.Session.get", side_effect=self._no_form4_get)

        snaps = [adapter.get_insider_flow(s) for s in ("AAPL", "MSFT", "ZZZZ")]

        assert [s is not None for s in snaps] == [True, True, False]
        assert self._ticker_downloads(mock_get) == 1

    def test_parallel_symbols_share_one_ticker_download(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        import threading
        import time as real_time

        def slow_get(url: str, **kwargs: object) -> Mock:
            real_time.sleep(0.02)  # poszerza okno race'a
            return self._no_form4_get(url, **kwargs)

        mock_get = mocker.patch("requests.Session.get", side_effect=slow_get)
        symbols = ("AAPL", "MSFT", "AAPL", "MSFT")
        barrier = threading.Barrier(len(symbols))

        def worker(symbol: str) -> None:
            barrier.wait()
            adapter.get_insider_flow(symbol)

        threads = [threading.Thread(target=worker, args=(s,)) for s in symbols]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert self._ticker_downloads(mock_get) == 1

    def test_requests_stay_under_the_sec_rate_limit(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        clock = {"now": 0.0}

        def sleep(seconds: float) -> None:
            clock["now"] += seconds

        mocker.patch("time.monotonic", side_effect=lambda: clock["now"])
        mocker.patch("time.sleep", side_effect=sleep)
        sent_at: list[float] = []

        def get(url: str, **kwargs: object) -> Mock:
            sent_at.append(clock["now"])
            return self._no_form4_get(url, **kwargs)

        mocker.patch("requests.Session.get", side_effect=get)

        for symbol in ("AAPL", "MSFT"):
            adapter.get_insider_flow(symbol)

        # Ticker map + 2× submissions; SEC: najwyżej 10 req/s → odstęp ≥ 0.1 s.
        gaps = [later - earlier for earlier, later in pairwise(sent_at)]
        assert len(gaps) == 2
        assert min(gaps) >= 0.1


class TestNullInsiderFlowAdapter:
    def test_returns_none_without_io(self, mocker: Any) -> None:
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


class TestPortErrorTaxonomy:
    """ARCH-33: awaria EDGAR na każdym kroku jest logowana jako klasa taksonomii
    portów (telemetria liczy błędy per klasa), a adapter nadal degraduje do None."""

    @staticmethod
    def _logged_error_class(caplog: pytest.LogCaptureFixture) -> type[BaseException]:
        records = [record for record in caplog.records if record.exc_info is not None]
        assert records, "brak zalogowanego wyjątku"
        exc_info = records[-1].exc_info
        assert exc_info is not None and exc_info[0] is not None
        return exc_info[0]

    def test_ticker_map_network_error_is_logged_as_provider_unavailable(
        self, adapter: EdgarInsiderAdapter, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        from src.application.errors import ProviderUnavailable

        mocker.patch("requests.Session.get", side_effect=requests.ConnectionError("reset"))

        assert adapter.get_insider_flow("AAPL") is None
        assert issubclass(self._logged_error_class(caplog), ProviderUnavailable)

    def test_submissions_garbage_is_logged_as_malformed_payload(
        self, adapter: EdgarInsiderAdapter, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        from src.application.errors import MalformedPayload

        broken = _ok_json({})
        broken.json.side_effect = requests.JSONDecodeError("Expecting value", "<html>", 0)
        mocker.patch(
            "requests.Session.get", side_effect=[_ok_json(_company_tickers()), broken]
        )

        assert adapter.get_insider_flow("AAPL") is None
        assert issubclass(self._logged_error_class(caplog), MalformedPayload)

    def test_form4_server_error_is_logged_as_provider_unavailable(
        self, adapter: EdgarInsiderAdapter, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        from src.application.errors import ProviderUnavailable

        mocker.patch(
            "requests.Session.get",
            side_effect=[
                _ok_json(_company_tickers()),
                _ok_json(_two_form4_submissions()),
                _error_response(503),
                _error_response(503),
            ],
        )

        adapter.get_insider_flow("AAPL")

        assert issubclass(self._logged_error_class(caplog), ProviderUnavailable)


class TestCorporateEvents:
    """DAT-17: raporty 8-K z tej samej odpowiedzi submissions co Form 4 — itemy,
    dzień filingu i link, bez nowego requestu."""

    @staticmethod
    def _submissions_with_8k() -> dict:
        payload = _submissions(
            forms=["8-K", "4", "8-K", "8-K"],
            dates=[_recent_date(1), _recent_date(2), _recent_date(5), _recent_date(40)],
            accessions=[
                "0000320193-26-000010",
                "0000320193-26-000009",
                "0000320193-26-000008",
                "0000320193-26-000007",
            ],
            primary_docs=["aapl-8k.htm", "form4.xml", "aapl-8k-ceo.htm", "aapl-8k-old.htm"],
        )
        payload["filings"]["recent"]["items"] = ["2.02,9.01", "", "5.02", "1.01"]
        return payload

    def test_extracts_8k_items_from_submissions_fixture(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_json(_company_tickers()), _ok_json(self._submissions_with_8k())],
        )

        snapshot = adapter.get_corporate_events("AAPL")

        assert snapshot is not None
        assert [(event.form, event.items) for event in snapshot.events] == [
            ("8-K", ("2.02", "9.01")),
            ("8-K", ("5.02",)),
        ]
        latest = snapshot.latest()
        assert latest is not None and latest.is_earnings()
        assert latest.filing_url == (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019326000010/aapl-8k.htm"
        )

    def test_events_reuse_the_submissions_of_the_insider_flow(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        get = mocker.patch(
            "requests.Session.get",
            side_effect=[
                _ok_json(_company_tickers()),
                _ok_json(self._submissions_with_8k()),
                _ok_text(_form4_xml(code="P", shares=100.0)),
            ],
        )

        adapter.get_insider_flow("AAPL")
        requests_for_flow = get.call_count
        snapshot = adapter.get_corporate_events("AAPL")

        assert snapshot is not None and len(snapshot.events) == 2
        assert get.call_count == requests_for_flow

    def test_unknown_ticker_has_no_events(self, adapter: EdgarInsiderAdapter, mocker: Any) -> None:
        mocker.patch("requests.Session.get", side_effect=[_ok_json(_company_tickers())])

        assert adapter.get_corporate_events("ZZZZ") is None

    def test_submissions_without_items_column_keep_the_events(
        self, adapter: EdgarInsiderAdapter, mocker: Any
    ) -> None:
        payload = _submissions(
            forms=["8-K"],
            dates=[_recent_date(1)],
            accessions=["0000320193-26-000010"],
            primary_docs=["aapl-8k.htm"],
        )
        mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_json(_company_tickers()), _ok_json(payload)],
        )

        snapshot = adapter.get_corporate_events("AAPL")

        assert snapshot is not None
        assert [event.items for event in snapshot.events] == [()]

    def test_failed_submissions_give_none(self, adapter: EdgarInsiderAdapter, mocker: Any) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_json(_company_tickers()), _error_response(503)],
        )

        assert adapter.get_corporate_events("AAPL") is None
