"""TEST-16: payloady dostawców z `tests/fixtures/providers/` przez prawdziwe adaptery.

Testy adapterów (test_finnhub_adapter.py, test_edgar_insider.py…) budują payloady
ręcznie, więc zmiana formatu u dostawcy przechodziła przez zielone testy — tak było
z prefiksem `xsl…/` w `primaryDocument` EDGAR (G42) i z polem `Note` Alpha Vantage
(G13). Tu każdy plik fixture'a przechodzi przez publiczne API adaptera, a oczekiwane
wartości liczymy z samego pliku: po odświeżeniu nagraniem testy dalej obowiązują.

HTTP mockujemy przez `requests.Session.get`, jak w pozostałych testach adapterów.
`TestLiveProviderShapes` (marker `integration`) pobiera świeże odpowiedzi i porównuje
ich kształt (klucze i typy) z fixture'ami; cron w `integration_smoke.yml` puszcza ją
raz w tygodniu.
"""

from __future__ import annotations

import os
import re
import time
import tomllib
from datetime import UTC, date, datetime
from decimal import Decimal
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from src.application.ports import UserMessage
from src.domain.macro_series import SeriesObservation
from src.infrastructure.adapters import (
    alpha_vantage_client,
    coingecko,
    edgar_insider,
    finnhub_api,
    fred_client,
    nbp_client,
)
from src.infrastructure.adapters._alpha_vantage_common import is_frequency_limit, is_soft_limit
from src.infrastructure.adapters._http import build_session, raise_for_status_redacted, safe_get
from src.infrastructure.adapters.alpha_vantage_client import (
    DEFAULT_RELEVANCE_THRESHOLD,
    AlphaVantageClient,
)
from src.infrastructure.adapters.coingecko import CoinGeckoAdapter
from src.infrastructure.adapters.edgar_insider import TICKERS_URL, EdgarInsiderAdapter
from src.infrastructure.adapters.finnhub_api import FinnhubAdapter
from src.infrastructure.adapters.fred_client import FredClient
from src.infrastructure.adapters.nbp_client import NbpClient, NbpFxRateClient
from src.infrastructure.adapters.rekor_publisher import RekorClient, SigstoreKeylessSigner
from src.infrastructure.adapters.telegram_command_source import TelegramCommandSource
from tests.infrastructure.provider_fixtures import (
    PROVIDERS_DIR,
    JsonValue,
    load_json,
    load_text,
    shape_drift,
    xml_element_paths,
)
from tests.sigstore_support import identity_token

# Każdy plik katalogu ma tu swoje miejsce: albo kształt sprawdza cotygodniowy test na
# żywym API, albo jest powód, dla którego się nie da. Nowy plik bez wpisu czerwieni
# `test_every_fixture_file_is_classified`.
LIVE_CHECKED_FIXTURES = frozenset(
    {
        "av_news_sentiment.json",
        "coingecko_price.json",
        "edgar_company_tickers.json",
        "edgar_form4.xml",
        "edgar_form4_submissions.json",
        "finnhub_quote.json",
        "fred_series.json",
        "nbp_table.json",
    }
)
NOT_LIVE_CHECKABLE_FIXTURES = {
    "av_premium_endpoint_truncated.json": (
        "marker premium z logu runu 40867144144, przycięty do 160 znaków; nie pełny payload HTTP"
    ),
    "av_note_limit.json": "limitu częstotliwości nie da się wywołać na żądanie",
    # WIA-73: kształt z API Fulcio v2 i Rekora v1, wartości syntetyczne, ale spójne
    # kryptograficznie (tests/sigstore_support.py): podpis rootu, inclusion proof ciała.
    "fulcio_signing_cert.json": "certyfikat wydaje tylko POST z tokenem OIDC joba GitHub Actions",
    "rekor_log_entry.json": (
        "odpowiedź POST-a, który tworzy publiczny na zawsze wpis — test niczego nie loguje"
    ),
    # MAIL-48: `getUpdates` z offsetem potwierdza aktualizacje bota właściciela —
    # żywe wywołanie ze smoke'u zjadłoby jego komendy, a bez offsetu i tak zwraca [].
    "telegram_get_updates.json": "żywe getUpdates potwierdziłoby komendy właściciela",
}

# Elementy Form 4, które czyta `EdgarInsiderAdapter` (transakcje, cena, rola, plan
# 10b5-1). Ich zniknięcie z prawdziwych filingów to cichy zerowy przepływ insiderów.
FORM4_PARSED_PATHS = frozenset(
    {
        "/ownershipDocument/aff10b5One",
        "/ownershipDocument/reportingOwner/reportingOwnerRelationship/isDirector",
        "/ownershipDocument/reportingOwner/reportingOwnerRelationship/isOfficer",
        "/ownershipDocument/reportingOwner/reportingOwnerRelationship/officerTitle",
        "/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/transactionCoding"
        "/transactionCode",
        "/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/transactionAmounts"
        "/transactionShares/value",
        "/ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/transactionAmounts"
        "/transactionPricePerShare/value",
    }
)

# Fragmenty, po których poznać klucz w query stringu albo w treści — fixture'y są
# w repo, więc nie mogą ich nieść.
_SECRET_MARKERS = re.compile(r"(?i)(apikey|api_key|token)=|\bsk-[A-Za-z0-9]")
_XSL_RENDERED_FORM4 = "<html><head><title>SEC FORM 4</title></head><body><table></table></body>"


def _json_response(payload: JsonValue) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


def _text_response(text: str) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.text = text
    response.raise_for_status = Mock()
    return response


def _object(name: str) -> dict[str, Any]:
    payload = load_json(name)
    assert isinstance(payload, dict), name
    return payload


class TestShapeDrift:
    def test_extra_live_fields_are_not_drift(self) -> None:
        assert shape_drift({"c": 1.5}, {"c": 2.5, "new_field": "x"}) == []

    def test_missing_field_is_drift(self) -> None:
        assert shape_drift({"c": 1.5, "t": 1}, {"c": 2.5}) == ["$.t: number (jest: brak)"]

    def test_changed_type_is_drift(self) -> None:
        # AV podaje relevance_score jako tekst; liczba to już inny kontrakt parsera.
        expected = {"feed": [{"relevance_score": "0.71"}]}
        actual = {"feed": [{"relevance_score": 0.71}]}

        assert shape_drift(expected, actual) == ["$.feed[].relevance_score: string (jest: number)"]

    def test_integer_and_float_are_one_number_type(self) -> None:
        assert shape_drift({"c": 262.24}, {"c": 262}) == []

    def test_list_elements_merge_into_one_shape(self) -> None:
        # Jeden artykuł bez obrazka (null) nie jest dryfem, gdy inny go ma.
        expected = {"feed": [{"banner_image": "https://example.com/a.jpg"}]}
        actual: JsonValue = {
            "feed": [{"banner_image": None}, {"banner_image": "https://example.com/b"}]
        }

        assert shape_drift(expected, actual) == []

    def test_bool_is_not_a_number(self) -> None:
        assert shape_drift({"flag": True}, {"flag": 1}) == ["$.flag: bool (jest: number)"]

    def test_xml_paths_ignore_namespaces(self) -> None:
        document = '<a xmlns="urn:x"><b><c>1</c></b></a>'

        assert xml_element_paths(document) == {"/a", "/a/b", "/a/b/c"}


class TestFixtureCatalog:
    def test_every_fixture_file_is_classified(self) -> None:
        files = {path.name for path in PROVIDERS_DIR.iterdir() if path.is_file()}

        assert files == LIVE_CHECKED_FIXTURES | set(NOT_LIVE_CHECKABLE_FIXTURES)

    @pytest.mark.parametrize(
        "name", sorted(LIVE_CHECKED_FIXTURES | set(NOT_LIVE_CHECKABLE_FIXTURES))
    )
    def test_fixture_carries_no_credentials(self, name: str) -> None:
        assert not _SECRET_MARKERS.search(load_text(name)), name

    def test_form4_fixture_has_every_parsed_element(self) -> None:
        assert FORM4_PARSED_PATHS.issubset(xml_element_paths(load_text("edgar_form4.xml")))


class TestFinnhubQuoteFixture:
    def test_quote_fixture_parses_to_quote(self, mocker: Any) -> None:
        payload = _object("finnhub_quote.json")
        mocker.patch("requests.Session.get", return_value=_json_response(payload))

        quote = FinnhubAdapter(api_key="test-token").get_quote("AAPL")

        assert quote.price.amount == Decimal(str(payload["c"]))
        assert quote.quoted_at == datetime.fromtimestamp(payload["t"], tz=UTC)
        assert quote.prev_close is not None
        assert quote.prev_close.amount == Decimal(str(payload["pc"]))
        assert quote.open_price is not None
        assert quote.open_price.amount == Decimal(str(payload["o"]))

    def test_quote_fixture_parses_to_daily_change(self, mocker: Any) -> None:
        payload = _object("finnhub_quote.json")
        mocker.patch("requests.Session.get", return_value=_json_response(payload))

        change = FinnhubAdapter(api_key="test-token").get_daily_change("AAPL")

        assert change.fraction == Decimal(str(payload["dp"])) / 100


class TestAlphaVantageFixtures:
    def test_captured_premium_marker_reports_access_denial(self, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor

        payload = _object("av_premium_endpoint_truncated.json")
        mocker.patch("requests.Session.get", return_value=_json_response(payload))
        monitor = QuotaMonitor()
        client = AlphaVantageClient(api_keys=["av-test"], symbols=["AAPL"], quota_monitor=monitor)

        assert client.articles_for("AAPL") == []
        assert client.degraded_reason == "av_keys_exhausted"
        assert "premium" in monitor.alerts[0].message
        assert "dzienny limit" not in monitor.alerts[0].message

    @staticmethod
    def _aapl_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Bloki `ticker_sentiment` AAPL powyżej progu relevance klienta."""
        return [
            block
            for article in payload["feed"]
            for block in article["ticker_sentiment"]
            if block["ticker"] == "AAPL"
            and float(block["relevance_score"]) >= DEFAULT_RELEVANCE_THRESHOLD
        ]

    def test_news_fixture_parses_to_articles(self, mocker: Any) -> None:
        payload = _object("av_news_sentiment.json")
        mocker.patch("requests.Session.get", return_value=_json_response(payload))
        expected = self._aapl_blocks(payload)

        articles = AlphaVantageClient(api_keys=["av-test"], symbols=["AAPL"]).articles_for("AAPL")

        assert expected, "fixture bez artykułu AAPL nad progiem nic nie sprawdza"
        assert [article["relevance_score"] for article in articles] == [
            float(block["relevance_score"]) for block in expected
        ]
        assert [article["ticker_sentiment_score"] for article in articles] == [
            float(block["ticker_sentiment_score"]) for block in expected
        ]

    def test_news_fixture_parses_to_sentiment_features(self, mocker: Any) -> None:
        payload = _object("av_news_sentiment.json")
        mocker.patch("requests.Session.get", return_value=_json_response(payload))

        client = AlphaVantageClient(api_keys=["av-test"], symbols=["AAPL"])
        features = client.sentiment_for("AAPL")

        assert features["news_volume_24h"] == len(self._aapl_blocks(payload))
        assert "degraded_reason" not in features
        assert client.degraded_reason is None

    def test_note_fixture_is_a_frequency_soft_limit(self) -> None:
        payload = _object("av_note_limit.json")

        assert is_soft_limit(payload, ("feed",))
        assert is_frequency_limit(payload)

    def test_note_fixture_on_every_attempt_degrades_the_feed(self, mocker: Any) -> None:
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        mocker.patch(
            "requests.Session.get", return_value=_json_response(_object("av_note_limit.json"))
        )
        client = AlphaVantageClient(api_keys=["av-test"], symbols=["AAPL"])

        assert client.articles_for("AAPL") == []
        assert client.degraded_reason == "av_keys_exhausted"


class TestCoinGeckoPriceFixture:
    def test_price_fixture_parses_to_price_and_daily_change(self, mocker: Any) -> None:
        payload = _object("coingecko_price.json")
        block = payload["bitcoin"]
        mocker.patch("requests.Session.get", return_value=_json_response(payload))
        adapter = CoinGeckoAdapter()

        price = adapter.get_current_price("BTC")
        change = adapter.get_daily_change("BTC")

        assert price.amount == Decimal(str(block["usd"]))
        assert change.fraction == Decimal(str(block["usd_24h_change"])) / 100
        assert change.as_of == datetime.fromtimestamp(block["last_updated_at"], tz=UTC)


class TestEdgarForm4Fixture:
    @staticmethod
    def _window_covering(submissions: dict[str, Any]) -> int:
        """Okno adaptera (dni od dziś) obejmujące wszystkie Form 4 z fixture'a."""
        recent = submissions["filings"]["recent"]
        oldest = min(
            date.fromisoformat(filed)
            for form, filed in zip(recent["form"], recent["filingDate"], strict=True)
            if form == "4"
        )
        return (datetime.now(UTC).date() - oldest).days + 1

    def _run_adapter(self, mocker: Any) -> tuple[Any, list[str]]:
        submissions = _object("edgar_form4_submissions.json")
        tickers = _object("edgar_company_tickers.json")
        form4 = load_text("edgar_form4.xml")
        requested: list[str] = []

        def get(url: str, **_: object) -> Mock:
            requested.append(url)
            if url == TICKERS_URL:
                return _json_response(tickers)
            if "/submissions/" in url:
                return _json_response(submissions)
            if "/Archives/" in url and "/xsl" in url.lower():
                # Jak SEC: pod ścieżką z prefiksem `xsl…/` leży XML przepuszczony
                # przez arkusz XSL, czyli HTML.
                return _text_response(_XSL_RENDERED_FORM4)
            if "/Archives/" in url:
                return _text_response(form4)
            raise AssertionError(f"nieoczekiwany URL: {url}")

        mocker.patch("requests.Session.get", side_effect=get)
        adapter = EdgarInsiderAdapter(
            user_agent="StockAgent test test@example.com",
            window_days=self._window_covering(submissions),
            min_request_interval=0.0,
        )
        return adapter.get_insider_flow("AAPL"), requested

    def test_edgar_fixture_parses_to_transactions(self, mocker: Any) -> None:
        # primaryDocument z prefiksem `xslF345X05/` wskazuje widok HTML; adapter
        # musi pobrać surowy XML, inaczej żaden Form 4 się nie sparsuje (G42).
        snapshot, _ = self._run_adapter(mocker)

        assert snapshot is not None
        assert snapshot.filings_seen == 2
        assert snapshot.filings_parsed == snapshot.filings_seen
        assert snapshot.buy_count + snapshot.sell_count >= 1

    def test_edgar_fixture_requests_raw_xml_not_the_xsl_render(self, mocker: Any) -> None:
        _, requested = self._run_adapter(mocker)

        archive_urls = [url for url in requested if "/Archives/" in url]
        assert len(archive_urls) == 2
        assert not [url for url in archive_urls if "/xsl" in url.lower()], archive_urls

    def test_edgar_fixture_reads_the_officer_purchase(self, mocker: Any) -> None:
        snapshot, _ = self._run_adapter(mocker)

        assert snapshot is not None
        assert snapshot.largest_buy is not None
        assert snapshot.largest_buy.is_officer
        assert snapshot.largest_buy.role == "Chief Operating Officer"


class TestFredSeriesFixture:
    def test_series_fixture_parses_to_yield_curve(self, mocker: Any) -> None:
        payload = _object("fred_series.json")
        mocker.patch("requests.Session.get", return_value=_json_response(payload))
        value = float(payload["observations"][0]["value"])

        curve = FredClient(api_key="test-key").fetch_yield_curve()

        assert curve is not None
        assert (curve.ten_year, curve.two_year, curve.fed_funds) == (value, value, value)

    def test_series_fixture_parses_to_macro_observations(self, mocker: Any) -> None:
        payload = _object("fred_series.json")
        mocker.patch("requests.Session.get", return_value=_json_response(payload))
        observation = payload["observations"][0]
        expected = SeriesObservation(
            date.fromisoformat(observation["date"]), float(observation["value"])
        )

        snapshot = FredClient(api_key="test-key").fetch_macro_series(date(2026, 10, 25))

        assert snapshot is not None
        assert snapshot.term_spread == expected
        assert snapshot.vix == (expected,)


class TestNbpTableFixture:
    def test_table_fixture_parses_to_fx_rate(self, mocker: Any) -> None:
        payload = _object("nbp_table.json")
        latest = payload["rates"][-1]
        effective = date.fromisoformat(latest["effectiveDate"])
        mocker.patch("requests.Session.get", return_value=_json_response(payload))

        rate = NbpFxRateClient().get_rate(payload["code"], effective)

        assert rate is not None
        assert rate.rate == Decimal(str(latest["mid"]))
        assert rate.effective_date == effective

    def test_table_fixture_parses_to_polish_macro(self, mocker: Any) -> None:
        payload = _object("nbp_table.json")
        rates = payload["rates"]
        mocker.patch("requests.Session.get", return_value=_json_response(payload))

        snapshot = NbpClient().fetch_polish_macro()

        assert snapshot is not None
        latest, baseline = Decimal(str(rates[-1]["mid"])), Decimal(str(rates[0]["mid"]))
        assert snapshot.usd_pln == latest
        assert snapshot.usd_pln_30d_change_ratio == (latest - baseline) / baseline


class TestSigstoreFixtures:
    """WIA-73: odpowiedzi Fulcio i Rekora przez publiczne API adaptera Rekora."""

    def test_signing_cert_fixture_gives_the_leaf_of_its_chain(self, mocker: Any) -> None:
        payload = _object("fulcio_signing_cert.json")
        chain = payload["signedCertificateEmbeddedSct"]["chain"]["certificates"]
        mocker.patch("requests.Session.post", return_value=_json_response(payload))
        signer = SigstoreKeylessSigner(build_session(), identity_token)

        signed = signer.sign(b"\x00" * 32)

        assert signed.certificate_pem == chain[0]

    def test_log_entry_fixture_parses_to_the_ledger_field(self, mocker: Any) -> None:
        payload = _object("rekor_log_entry.json")
        ((uuid, entry),) = payload.items()
        mocker.patch("requests.Session.post", return_value=_json_response(payload))

        logged = RekorClient(build_session()).log({"kind": "hashedrekord"})

        assert logged == {"uuid": uuid, **entry}
        assert isinstance(logged["integratedTime"], int)


class TestTelegramGetUpdatesFixture:
    def test_updates_fixture_passes_only_owner_text_messages(self, mocker: Any) -> None:
        payload = _object("telegram_get_updates.json")
        updates = payload["result"]
        owner = str(updates[0]["message"]["chat"]["id"])
        mocker.patch("requests.Session.get", return_value=_json_response(payload))

        source = TelegramCommandSource(bot_token="1:x", allowed_chat_ids=[owner])

        batch = source.fetch_messages(None)

        assert batch.messages == tuple(
            UserMessage(update["update_id"], owner, update["message"]["text"])
            for update in updates
            if str(update["message"]["chat"]["id"]) == owner and "text" in update["message"]
        )
        assert batch.next_offset == max(update["update_id"] for update in updates) + 1


# ---------------------------------------------------------------------------
# Kształt na żywym API (marker `integration`, cron tygodniowy w integration_smoke)
# ---------------------------------------------------------------------------

CONFIG_TOML = Path(__file__).resolve().parents[2] / "config.toml"
_LIVE_TIMEOUT = 30.0
# 403/429 to odmowa albo limit (SEC blokuje sec.gov od 2026-07, CoinGecko dławi
# współdzielone IP runnerów), nie zmiana kształtu — skip z powodem w podsumowaniu.
_REFUSALS = frozenset({HTTPStatus.FORBIDDEN, HTTPStatus.TOO_MANY_REQUESTS})
_LIVE_FORM4_SAMPLE = 3


def _env_key(name: str) -> str:
    """Pierwszy klucz ze zmiennej (`ALPHA_VANTAGE_API_KEYS` bywa listą CSV)."""
    keys = [key.strip() for key in os.environ.get(name, "").split(",") if key.strip()]
    if not keys:
        pytest.skip(f"{name} not set")
    return keys[0]


def _edgar_user_agent() -> str:
    """User-Agent z kontaktem jak w produkcji: env `EDGAR_USER_AGENT` > config.toml."""
    override = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if override:
        return override
    config = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    return str(config["edgar_user_agent"])


def _live_get(
    url: str, *, params: dict[str, str] | None = None, headers: dict[str, str] | None = None
) -> requests.Response:
    """GET przez sesję adapterów (retry, circuit breaker); klucz tylko w `params`,
    więc ani URL w skipie, ani wyjątek `safe_get` go nie niosą."""
    session = build_session()
    if headers:
        session.headers.update(headers)
    response = safe_get(session, url, params=params, timeout=_LIVE_TIMEOUT)
    if response.status_code in _REFUSALS:
        pytest.skip(f"{url}: HTTP {response.status_code} — odmowa albo limit, nie kształt")
    raise_for_status_redacted(response)
    return response


def _live_json(
    url: str, *, params: dict[str, str] | None = None, headers: dict[str, str] | None = None
) -> JsonValue:
    payload: JsonValue = _live_get(url, params=params, headers=headers).json()
    return payload


@pytest.mark.integration
class TestLiveProviderShapes:
    """Świeże odpowiedzi dostawców mają co najmniej kształt fixture'ów.

    Brak pola albo zmiana typu (np. `relevance_score` z tekstu na liczbę) czerwieni
    test z listą ścieżek; nowe pola u dostawcy nie są dryfem. Treści odpowiedzi nie
    wypisujemy — same ścieżki i typy. Koszt: jedno zapytanie Alpha Vantage z dziennego
    limitu klucza smoke, reszta darmowa."""

    def test_finnhub_quote_keeps_its_shape(self) -> None:
        payload = _live_json(
            f"{finnhub_api.DEFAULT_BASE_URL}/quote",
            params={"symbol": "AAPL", "token": _env_key("FINNHUB_API_KEY")},
        )

        assert shape_drift(load_json("finnhub_quote.json"), payload) == []

    def test_alpha_vantage_news_keeps_its_shape(self) -> None:
        payload = _live_json(
            f"{alpha_vantage_client.DEFAULT_BASE_URL}/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": "AAPL",
                "limit": str(alpha_vantage_client.DEFAULT_LIMIT),
                "sort": "LATEST",
                "apikey": _env_key("ALPHA_VANTAGE_API_KEYS"),
            },
        )
        assert isinstance(payload, dict)
        if is_soft_limit(payload, ("feed",)):
            pytest.skip("Alpha Vantage soft limit (Information/Note) — feed not compared")

        assert shape_drift(load_json("av_news_sentiment.json"), payload) == []

    def test_coingecko_price_keeps_its_shape(self) -> None:
        payload = _live_json(
            f"{coingecko.DEFAULT_BASE_URL}/simple/price",
            params={
                "ids": "bitcoin",
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_last_updated_at": "true",
            },
        )

        assert shape_drift(load_json("coingecko_price.json"), payload) == []

    def test_edgar_submissions_and_form4_keep_their_shape(self) -> None:
        headers = {"User-Agent": _edgar_user_agent()}
        tickers = _live_json(edgar_insider.TICKERS_URL, headers=headers)
        assert shape_drift(load_json("edgar_company_tickers.json"), tickers) == []
        assert isinstance(tickers, dict)
        cik = next(
            str(entry["cik_str"]).zfill(10)
            for entry in tickers.values()
            if isinstance(entry, dict) and entry.get("ticker") == "AAPL"
        )

        time.sleep(edgar_insider.SEC_MIN_REQUEST_INTERVAL_SECONDS)
        submissions = _live_json(edgar_insider.SUBMISSIONS_URL.format(cik=cik), headers=headers)
        assert shape_drift(load_json("edgar_form4_submissions.json"), submissions) == []

        form4_paths: set[str] = set()
        for accession, document in self._recent_form4(submissions):
            time.sleep(edgar_insider.SEC_MIN_REQUEST_INTERVAL_SECONDS)
            url = edgar_insider.ARCHIVE_URL.format(
                cik=int(cik),
                accession=accession.replace("-", ""),
                doc=edgar_insider._raw_form4_doc(document),
            )
            paths = xml_element_paths(_live_get(url, headers=headers).text)
            # Korzeń inny niż ownershipDocument = HTML albo inny dokument (G42).
            assert "/ownershipDocument" in paths, (document, sorted(paths)[:3])
            form4_paths |= paths

        assert form4_paths, "EDGAR returned no recent Form 4 for AAPL"
        assert sorted(FORM4_PARSED_PATHS - form4_paths) == []

    @staticmethod
    def _recent_form4(submissions: JsonValue) -> list[tuple[str, str]]:
        assert isinstance(submissions, dict)
        recent = submissions["filings"]["recent"]
        rows = zip(
            recent["form"], recent["accessionNumber"], recent["primaryDocument"], strict=True
        )
        form4 = [(accession, doc) for form, accession, doc in rows if form == "4"]
        return form4[:_LIVE_FORM4_SAMPLE]

    def test_fred_series_keeps_its_shape(self) -> None:
        payload = _live_json(
            f"{fred_client.DEFAULT_BASE_URL}/series/observations",
            params={
                "series_id": fred_client.SERIES_TEN_YEAR,
                "api_key": _env_key("FRED_API_KEY"),
                "file_type": "json",
                "sort_order": "desc",
                "limit": "1",
            },
        )

        assert shape_drift(load_json("fred_series.json"), payload) == []

    def test_nbp_table_keeps_its_shape(self) -> None:
        payload = _live_json(f"{nbp_client.DEFAULT_BASE_URL}/USD/last/5/?format=json")

        assert shape_drift(load_json("nbp_table.json"), payload) == []
