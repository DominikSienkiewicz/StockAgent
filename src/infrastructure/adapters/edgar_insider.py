"""Adapter SEC EDGAR — przepływy insiderów z Form 4 (alpha „Dane").

SEC EDGAR jest darmowe i bez klucza, ale **wymaga** nagłówka `User-Agent`
identyfikującego aplikację na każdym żądaniu (inaczej 403). Pipeline:

  1) ticker → CIK przez GET https://www.sec.gov/files/company_tickers.json
     (mapa {"0": {"cik_str": 320193, "ticker": "AAPL", ...}}; CIK pad do 10 cyfr),
  2) recent filings przez GET https://data.sec.gov/submissions/CIK{cik}.json
     (równoległe tablice form[]/filingDate[]/accessionNumber[]/primaryDocument[]),
  3) dla każdego Form 4 w oknie — fetch surowego XML (primaryDocument bez
     prefiksu renderera `xsl…/`, patrz `_raw_form4_doc`) i parsowanie transakcji
     nonDerivative: transactionCode 'P' = kupno, 'S' = sprzedaż.

Adapter jest cienki i „graceful": każdy błąd na dowolnym kroku → None, nigdy
nie podnosi wyjątku (to opcjonalne źródło alpha). Logika klasyfikacji sygnału
żyje w domenie (`InsiderFlowSnapshot.evaluate_signal`). Częściowa agregacja jest
ok — jeśli część Form-4 XML nie sparsuje się, pomijamy je; jeśli nie sparsuje
się żaden, zwracamy snapshot z zerowymi licznikami (net_shares = 0.0). Snapshot
niesie liczbę widzianych i sparsowanych Form 4 (DAT-03), więc domena
(`has_evidence`) odróżnia taką awarię od okna bez transakcji.

DAT-10: mapa ticker → CIK (company_tickers.json, kilka MB) jest pobierana raz na
instancję i trzymana w pamięci — nieudane pobranie nie jest cache'owane, więc
kolejny symbol spróbuje ponownie. Każdy request do sec.gov przechodzi przez
`_RequestPacer` z minimalnym odstępem `SEC_MIN_REQUEST_INTERVAL_SECONDS` (polityka
SEC: najwyżej 10 requestów na sekundę). 403 z SEC nie trafia do globalnego
`RETRY_STATUS_FORCELIST` — to odmowa, nie chwilowa awaria.
"""

from __future__ import annotations

import logging
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import requests

from src.application.ports import InsiderFlowPort
from src.domain.insider_flow import InsiderFlowSnapshot
from src.infrastructure.adapters._http import build_session

DEFAULT_TIMEOUT = 10.0
DEFAULT_WINDOW_DAYS = 90
DEFAULT_MAX_FORM4 = 20
# Polityka SEC: ≤ 10 requestów/s na aplikację — 0.12 s to zapas na drgania zegara.
SEC_MIN_REQUEST_INTERVAL_SECONDS = 0.12

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"

# Kody transakcji Form 4 (tabela nonDerivative).
BUY_CODE = "P"  # purchase — kupno
SELL_CODE = "S"  # sale — sprzedaż

logger = logging.getLogger(__name__)


def _local_name(tag: str) -> str:
    """Zwraca nazwę bez namespace'u (`{ns}foo` → `foo`)."""
    return tag.rsplit("}", 1)[-1]


def _find_local(element: ET.Element, name: str) -> ET.Element | None:
    """Pierwszy potomek (na dowolnej głębokości) o lokalnej nazwie `name`."""
    for descendant in element.iter():
        if _local_name(descendant.tag) == name:
            return descendant
    return None


def _findall_local(element: ET.Element, name: str) -> list[ET.Element]:
    """Wszyscy potomkowie o lokalnej nazwie `name` (tolerancja na namespace)."""
    return [d for d in element.iter() if _local_name(d.tag) == name]


def _padded_cik(cik_raw: int | str | None) -> str | None:
    """CIK bywa intem lub stringiem; brak albo nie-numeryczny → None (graceful,
    zamiast ValueError wyciekającego z mapowania)."""
    if cik_raw is None:
        return None
    try:
        return str(int(cik_raw)).zfill(10)
    except (TypeError, ValueError):
        return None


def _raw_form4_doc(primary_doc: str) -> str:
    """Ścieżka surowego XML Form 4 zamiast widoku renderowanego przez XSL.

    Submissions API podaje dla Form 4 `primaryDocument` w postaci
    `xslF345X05/wk-form4_….xml` — pod nią SEC serwuje HTML (XML przepuszczony
    przez arkusz XSL), którego parser XML nie przeczyta, więc każdy filing
    wypadałby z agregacji. Surowy XML leży w tym samym katalogu accession pod
    ostatnim segmentem. Ścieżki bez prefiksu `xsl` (w tym gołe nazwy) bez zmian.
    """
    first_segment, separator, _ = primary_doc.partition("/")
    if separator and first_segment.lower().startswith("xsl"):
        return primary_doc.rsplit("/", 1)[-1]
    return primary_doc


class _RequestPacer:
    """Minimalny odstęp między kolejnymi requestami jednej instancji (DAT-10).

    Lock trzyma odstęp także przy równoległych symbolach: wątek czeka na swoją
    kolej, a pierwszy request nie czeka wcale."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last is not None:
                remaining = self._last + self._min_interval - now
                if remaining > 0:
                    self._sleep(remaining)
                    now = self._clock()
            self._last = now


class EdgarInsiderAdapter(InsiderFlowPort):
    """Adapter InsiderFlowPort oparty o publiczne EDGAR Form 4."""

    def __init__(
        self,
        user_agent: str,
        window_days: int = DEFAULT_WINDOW_DAYS,
        max_form4: int = DEFAULT_MAX_FORM4,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        min_request_interval: float = SEC_MIN_REQUEST_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._user_agent = user_agent
        self._window_days = window_days
        self._max_form4 = max_form4
        self._timeout = timeout
        self._session = session if session is not None else build_session()
        # SEC wymaga User-Agent na KAŻDYM żądaniu — ustawiamy raz na sesji.
        self._session.headers.update({"User-Agent": user_agent})
        self._pacer = _RequestPacer(min_request_interval, clock, sleep)
        # DAT-10: mapa ticker → CIK (None = wpis bez poprawnego CIK) na czas
        # życia instancji; None w polu = jeszcze nie pobrana albo ostatnia próba padła.
        self._cik_by_ticker: dict[str, str | None] | None = None
        self._cik_lock = threading.Lock()

    def _get(self, url: str) -> requests.Response:
        """Każdy request do sec.gov przez pacer (polityka SEC ≤ 10/s)."""
        self._pacer.wait()
        return self._session.get(url, timeout=self._timeout)

    def get_insider_flow(self, symbol: str) -> InsiderFlowSnapshot | None:
        try:
            cik = self._resolve_cik(symbol)
            if cik is None:
                return None
            filings = self._recent_form4_filings(cik)
            if filings is None:
                return None

            net_shares = 0.0
            buy_count = 0
            sell_count = 0
            filings_parsed = 0
            for accession, primary_doc in filings:
                parsed = self._parse_form4(cik, accession, primary_doc)
                if parsed is None:
                    # Pojedynczy zepsuty filing nie wywala agregacji.
                    continue
                delta_net, buys, sells = parsed
                net_shares += delta_net
                buy_count += buys
                sell_count += sells
                filings_parsed += 1

            return InsiderFlowSnapshot(
                symbol=symbol,
                net_shares=net_shares,
                buy_count=buy_count,
                sell_count=sell_count,
                window_days=self._window_days,
                filings_seen=len(filings),
                filings_parsed=filings_parsed,
            )
        except Exception:
            # Opcjonalne źródło alpha — żaden błąd nie może wywalić cyklu.
            logger.exception("EDGAR insider flow failed for %s", symbol)
            return None

    def _resolve_cik(self, symbol: str) -> str | None:
        """Mapuje ticker → 10-cyfrowy CIK (z paddingiem) z mapy instancji."""
        cik_by_ticker = self._ticker_map()
        if cik_by_ticker is None:
            return None
        return cik_by_ticker.get(symbol.upper())

    def _ticker_map(self) -> dict[str, str | None] | None:
        """Mapa ticker → CIK, pobierana przy pierwszym użyciu (DAT-10).

        Porażka nie jest cache'owana — następny symbol pobiera ponownie. Lock
        sprawia, że równoległe symbole nie ściągają pliku kilka razy."""
        with self._cik_lock:
            if self._cik_by_ticker is None:
                self._cik_by_ticker = self._fetch_ticker_map()
            return self._cik_by_ticker

    def _fetch_ticker_map(self) -> dict[str, str | None] | None:
        try:
            response = self._get(TICKERS_URL)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            logger.exception("EDGAR ticker lookup failed")
            return None

        if not isinstance(payload, dict):
            return None

        cik_by_ticker: dict[str, str | None] = {}
        for entry in payload.values():
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker", "")).upper()
            # Pierwszy wpis tickera wygrywa — jak przy liniowym wyszukiwaniu.
            if ticker and ticker not in cik_by_ticker:
                cik_by_ticker[ticker] = _padded_cik(entry.get("cik_str"))
        return cik_by_ticker

    def _recent_form4_filings(self, cik: str) -> list[tuple[str, str]] | None:
        """Zwraca [(accessionNumber, primaryDocument)] dla Form 4 w oknie.

        None gdy submissions niedostępne; pusta lista gdy brak Form 4."""
        url = SUBMISSIONS_URL.format(cik=cik)
        try:
            response = self._get(url)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            logger.exception("EDGAR submissions fetch failed for CIK %s", cik)
            return None

        recent = (
            payload.get("filings", {}).get("recent", {})
            if isinstance(payload, dict)
            else {}
        )
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        if not (
            isinstance(forms, list)
            and isinstance(dates, list)
            and isinstance(accessions, list)
            and isinstance(primary_docs, list)
        ):
            return None

        cutoff = datetime.now(UTC).date() - timedelta(days=self._window_days)
        result: list[tuple[str, str]] = []
        for form, date_str, accession, doc in zip(
            forms, dates, accessions, primary_docs, strict=False
        ):
            if str(form) != "4":
                continue
            filing_date = self._parse_date(date_str)
            if filing_date is None or filing_date < cutoff:
                continue
            result.append((str(accession), str(doc)))
            if len(result) >= self._max_form4:
                break
        return result

    @staticmethod
    def _parse_date(date_str: object) -> date | None:
        try:
            return datetime.strptime(str(date_str), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None

    def _parse_form4(
        self, cik: str, accession: str, primary_doc: str
    ) -> tuple[float, int, int] | None:
        """Fetch + parsuje pojedynczy Form 4 → (net_shares, buy_count, sell_count).

        None gdy fetch lub parse XML padnie (filing pomijany w agregacji)."""
        accession_no_dashes = accession.replace("-", "")
        # CIK w ścieżce Archives bez zer wiodących (int).
        url = ARCHIVE_URL.format(
            cik=int(cik),
            accession=accession_no_dashes,
            doc=_raw_form4_doc(primary_doc),
        )
        try:
            response = self._get(url)
            response.raise_for_status()
            xml_text = response.text
        except requests.RequestException:
            logger.exception("EDGAR Form 4 fetch failed (%s)", accession)
            return None

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            logger.warning("EDGAR Form 4 XML parse failed (%s)", accession)
            return None

        net_shares = 0.0
        buy_count = 0
        sell_count = 0
        for transaction in _findall_local(root, "nonDerivativeTransaction"):
            code = self._transaction_code(transaction)
            shares = self._transaction_shares(transaction)
            if shares is None:
                continue
            if code == BUY_CODE:
                net_shares += shares
                buy_count += 1
            elif code == SELL_CODE:
                net_shares -= shares
                sell_count += 1
        return net_shares, buy_count, sell_count

    @staticmethod
    def _transaction_code(transaction: ET.Element) -> str | None:
        coding = _find_local(transaction, "transactionCoding")
        if coding is None:
            return None
        code_el = _find_local(coding, "transactionCode")
        if code_el is None or code_el.text is None:
            return None
        return code_el.text.strip()

    @staticmethod
    def _transaction_shares(transaction: ET.Element) -> float | None:
        shares_el = _find_local(transaction, "transactionShares")
        if shares_el is None:
            return None
        # <transactionShares> ma zwykle wewnętrzny <value> — szukamy go,
        # a w razie braku bierzemy bezpośredni tekst węzła.
        value_el = _find_local(shares_el, "value")
        text = value_el.text if value_el is not None else shares_el.text
        if text is None:
            return None
        try:
            return float(text.strip())
        except ValueError:
            return None


class NullInsiderFlowAdapter(InsiderFlowPort):
    """Domyślny adapter Fast-Loop / brak konfiguracji — zero I/O, zawsze None."""

    def get_insider_flow(self, symbol: str) -> InsiderFlowSnapshot | None:
        return None
