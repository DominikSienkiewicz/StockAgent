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
"""

from __future__ import annotations

import logging
import threading
import time
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from http import HTTPStatus

import requests

from src.application.ports import InsiderFlowPort
from src.domain.insider_flow import InsiderFlowSnapshot
from src.infrastructure.adapters._http import build_session

DEFAULT_TIMEOUT = 10.0
DEFAULT_WINDOW_DAYS = 90
DEFAULT_MAX_FORM4 = 20
# SEC: „Current max request rate: 10 requests/second" na użytkownika, niezależnie
# od liczby maszyn (sec.gov/os/accessing-edgar-data). 8 zostawia zapas na retry
# urllib3 z `build_session`, które idą poza tym pacerem.
MAX_REQUESTS_PER_SECOND = 8
MIN_REQUEST_INTERVAL_SECONDS = 1.0 / MAX_REQUESTS_PER_SECOND

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


def _padded_cik(raw: object) -> str | None:
    """CIK bywa intem lub stringiem → 10 cyfr; brak albo nie-numeryczny → None."""
    try:
        return str(int(str(raw))).zfill(10)
    except ValueError:
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


class EdgarInsiderAdapter(InsiderFlowPort):
    """Adapter InsiderFlowPort oparty o publiczne EDGAR Form 4."""

    def __init__(
        self,
        user_agent: str,
        window_days: int = DEFAULT_WINDOW_DAYS,
        max_form4: int = DEFAULT_MAX_FORM4,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._user_agent = user_agent
        self._window_days = window_days
        self._max_form4 = max_form4
        self._timeout = timeout
        self._session = session if session is not None else build_session()
        # SEC wymaga User-Agent na KAŻDYM żądaniu — ustawiamy raz na sesji.
        self._session.headers.update({"User-Agent": user_agent})
        # Adapter żyje jeden cykl: po 403 SEC dalsze żądania tylko przedłużają
        # blokadę (dostęp wraca po 10 min poniżej progu), więc milkniemy do końca.
        self._blocked = False
        self._ciks: dict[str, str] | None = None
        self._ciks_lock = threading.Lock()
        self._last_request_at: float | None = None
        self._pace_lock = threading.Lock()

    def get_insider_flow(self, symbol: str) -> InsiderFlowSnapshot | None:
        if self._blocked:
            return None
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

            if self._blocked:
                # Snapshot z częścią filingów wyglądałby jak awaria parsowania.
                return None
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

    def _get(self, url: str) -> requests.Response | None:
        """GET do SEC; None, gdy SEC odrzucił nas w tym cyklu (403)."""
        if self._blocked:
            return None
        self._pace()
        response = self._session.get(url, timeout=self._timeout)
        if response.status_code == HTTPStatus.FORBIDDEN:
            self._blocked = True
            logger.warning(
                "SEC EDGAR answered 403 for %s — no more EDGAR requests this cycle "
                "(see docs/configuration.md, insider flow)",
                url,
            )
            return None
        return response

    def _pace(self) -> None:
        """Odstęp ≥ `MIN_REQUEST_INTERVAL_SECONDS` między żądaniami WSZYSTKICH
        wątków. Bez niego równoległe symbole (`symbol_concurrency` > 1), każdy
        z kilkunastoma Form 4, strzelałyby burstem ponad limit SEC."""
        with self._pace_lock:
            if self._last_request_at is not None:
                wait = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
                if wait > 0:
                    time.sleep(wait)
            self._last_request_at = time.monotonic()

    def _resolve_cik(self, symbol: str) -> str | None:
        """Mapuje ticker → 10-cyfrowy CIK (z paddingiem)."""
        ciks = self._ticker_ciks()
        return None if ciks is None else ciks.get(symbol.upper())

    def _ticker_ciks(self) -> dict[str, str] | None:
        """Mapa ticker → CIK pobierana raz na cykl: plik ma ~1 MB i jest wspólny
        dla wszystkich symboli. Lock, bo symbole idą równolegle
        (`symbol_concurrency`); nieudane pobranie nie trafia do cache'u."""
        with self._ciks_lock:
            if self._ciks is None:
                self._ciks = self._fetch_ticker_ciks()
            return self._ciks

    def _fetch_ticker_ciks(self) -> dict[str, str] | None:
        try:
            response = self._get(TICKERS_URL)
            if response is None:
                return None
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            logger.exception("EDGAR ticker lookup failed")
            return None

        if not isinstance(payload, dict):
            return None

        ciks: dict[str, str] = {}
        for entry in payload.values():
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker", "")).upper()
            cik = _padded_cik(entry.get("cik_str"))
            if ticker and cik is not None:
                ciks.setdefault(ticker, cik)
        return ciks

    def _recent_form4_filings(self, cik: str) -> list[tuple[str, str]] | None:
        """Zwraca [(accessionNumber, primaryDocument)] dla Form 4 w oknie.

        None gdy submissions niedostępne; pusta lista gdy brak Form 4."""
        url = SUBMISSIONS_URL.format(cik=cik)
        try:
            response = self._get(url)
            if response is None:
                return None
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
            if response is None:
                return None
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
