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
`RETRY_STATUS_FORCELIST` — to odmowa, nie chwilowa awaria. Instancja żyje jeden
cykl, więc pierwsza 403 kończy EDGAR do końca cyklu: kolejne żądania tylko
przedłużałyby blokadę.

DAT-21: z tego samego XML-a czytamy też cenę (`transactionPricePerShare`),
relację właściciela (`isOfficer`/`officerTitle`/`isDirector`; sam 10% owner to
fundusz, nie insider) i znacznik planu 10b5-1 (`aff10b5One`, Form 4 od 2023 —
pole dotyczy całego filingu). Transakcje planowe nie zasilają `net_value_usd`
ani `largest_buy`; brak znacznika (starsze Form 4) to transakcja uznaniowa.
`net_shares` i liczniki liczą się jak dotąd. Kody inne niż P/S (przydziały A,
opcje M, podatek F) są pomijane. Zakup niesie datę filingu z submissions
(`filingDate`) — od niej liczy się okno, w którym może prowadzić maila.
"""

from __future__ import annotations

import logging
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from http import HTTPStatus

import requests

from src.application.ports import InsiderFlowPort
from src.domain.insider_flow import InsiderBuy, InsiderFlowSnapshot
from src.infrastructure.adapters._http import build_session

DEFAULT_TIMEOUT = 10.0
DEFAULT_WINDOW_DAYS = 90
DEFAULT_MAX_FORM4 = 20
# Polityka SEC: ≤ 10 requestów/s na użytkownika, niezależnie od liczby maszyn
# (sec.gov/os/accessing-edgar-data). 0.12 s (~8 req/s) to zapas na drgania zegara
# i na retry urllib3 z `build_session`, które idą poza pacerem.
SEC_MIN_REQUEST_INTERVAL_SECONDS = 0.12

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"

# Kody transakcji Form 4 (tabela nonDerivative).
BUY_CODE = "P"  # purchase — kupno
SELL_CODE = "S"  # sale — sprzedaż
# Wartości „zaznaczonego” pola logicznego Form 4 (`1` albo `true`).
_TRUE_FLAGS = frozenset({"1", "true"})

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


@dataclass(frozen=True)
class _Form4Filing:
    """Form 4 z okna: accession, `primaryDocument` i data filingu (DAT-21)."""

    accession: str
    primary_doc: str
    filed_on: date


@dataclass(frozen=True)
class _Form4Summary:
    """Wynik jednego Form 4: przepływ akcji (jak przed DAT-21) i wartości."""

    net_shares: float
    buy_count: int
    sell_count: int
    net_value_usd: float
    largest_buy: InsiderBuy | None


def _stronger_buy(current: InsiderBuy | None, candidate: InsiderBuy | None) -> InsiderBuy | None:
    """Zakup officera wygrywa z zakupem członka rady, potem większa kwota."""
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate, key=lambda buy: (buy.is_officer, buy.value_usd))


def _filed_on(buy: InsiderBuy | None, filed_on: date) -> InsiderBuy | None:
    """DAT-21: zakup z datą filingu z submissions — okno leadu liczy się od niej."""
    return replace(buy, filed_on=filed_on) if buy is not None else None


def _is_flag_set(element: ET.Element | None) -> bool:
    """Pole logiczne Form 4 (np. `<isOfficer>1</isOfficer>`) — brak = fałsz."""
    if element is None:
        return False
    value_el = _find_local(element, "value")
    text = value_el.text if value_el is not None else element.text
    return (text or "").strip().lower() in _TRUE_FLAGS


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


def _monotonic() -> float:
    """Zegar domyślnego pacera — `time.monotonic` czytany przy wywołaniu, nie
    przy definicji klasy, więc pacer idzie za zegarem, który ma moduł `time`."""
    return time.monotonic()


def _sleep(seconds: float) -> None:
    """Czekanie domyślnego pacera — `time.sleep` czytany przy wywołaniu."""
    time.sleep(seconds)


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
        clock: Callable[[], float] = _monotonic,
        sleep: Callable[[float], None] = _sleep,
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
        # Adapter żyje jeden cykl: po 403 SEC dalsze żądania tylko przedłużają
        # blokadę (dostęp wraca po 10 min poniżej progu), więc milkniemy do końca.
        self._blocked = False

    def _get(self, url: str) -> requests.Response | None:
        """Każdy request do sec.gov przez pacer (polityka SEC ≤ 10/s); None, gdy
        SEC odrzucił nas w tym cyklu (403) — wtedy nie wysyłamy już nic."""
        if self._blocked:
            return None
        self._pacer.wait()
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

    def get_insider_flow(self, symbol: str) -> InsiderFlowSnapshot | None:
        if self._blocked:
            return None
        try:
            cik = self._resolve_cik(symbol)
            if cik is None:
                return None
            filings = self._recent_form4(cik)
            if filings is None:
                return None

            net_shares = 0.0
            buy_count = 0
            sell_count = 0
            filings_parsed = 0
            net_value_usd = 0.0
            largest_buy: InsiderBuy | None = None
            for filing in filings:
                parsed = self._parse_form4(cik, filing.accession, filing.primary_doc)
                if parsed is None:
                    # Pojedynczy zepsuty filing nie wywala agregacji.
                    continue
                net_shares += parsed.net_shares
                buy_count += parsed.buy_count
                sell_count += parsed.sell_count
                net_value_usd += parsed.net_value_usd
                largest_buy = _stronger_buy(
                    largest_buy, _filed_on(parsed.largest_buy, filing.filed_on)
                )
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
                net_value_usd=net_value_usd,
                largest_buy=largest_buy,
                latest_filing_url=(
                    _filing_url(cik, filings[0].accession, filings[0].primary_doc)
                    if filings
                    else None
                ),
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
            if response is None:
                return None
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
        """Zwraca [(accessionNumber, primaryDocument)] dla Form 4 w oknie — kształt
        sprzed DAT-21, który czyta test integracyjny z prawdziwym EDGAR."""
        filings = self._recent_form4(cik)
        if filings is None:
            return None
        return [(filing.accession, filing.primary_doc) for filing in filings]

    def _recent_form4(self, cik: str) -> list[_Form4Filing] | None:
        """Form 4 w oknie z datą filingu, od najnowszego (kolejność submissions).

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
        result: list[_Form4Filing] = []
        for form, date_str, accession, doc in zip(
            forms, dates, accessions, primary_docs, strict=False
        ):
            if str(form) != "4":
                continue
            filing_date = self._parse_date(date_str)
            if filing_date is None or filing_date < cutoff:
                continue
            result.append(_Form4Filing(str(accession), str(doc), filing_date))
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
    ) -> _Form4Summary | None:
        """Fetch + parsuje pojedynczy Form 4 (akcje, wartość USD, rola).

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

        planned = _is_flag_set(_find_local(root, "aff10b5One"))
        net_shares = 0.0
        buy_count = 0
        sell_count = 0
        net_value = 0.0
        bought_value = 0.0
        for transaction in _findall_local(root, "nonDerivativeTransaction"):
            code = self._transaction_code(transaction)
            shares = self._transaction_shares(transaction)
            if shares is None:
                continue
            if code not in (BUY_CODE, SELL_CODE):
                continue
            sign = 1.0 if code == BUY_CODE else -1.0
            net_shares += sign * shares
            if code == BUY_CODE:
                buy_count += 1
            else:
                sell_count += 1
            price = self._transaction_price(transaction)
            if planned or price is None:
                continue
            net_value += sign * shares * price
            if code == BUY_CODE:
                bought_value += shares * price
        filing_url = _filing_url(cik, accession, primary_doc)
        return _Form4Summary(
            net_shares=net_shares,
            buy_count=buy_count,
            sell_count=sell_count,
            net_value_usd=net_value,
            largest_buy=self._insider_buy(root, bought_value, filing_url),
        )

    @staticmethod
    def _insider_buy(root: ET.Element, value_usd: float, filing_url: str) -> InsiderBuy | None:
        """Zakup z filingu jako `InsiderBuy` — tylko officer albo członek rady
        (sam 10% owner to zwykle fundusz, nie insider znający spółkę)."""
        if value_usd <= 0:
            return None
        relationship = _find_local(root, "reportingOwnerRelationship")
        if relationship is None:
            return None
        is_officer = _is_flag_set(_find_local(relationship, "isOfficer"))
        if not is_officer and not _is_flag_set(_find_local(relationship, "isDirector")):
            return None
        title_el = _find_local(relationship, "officerTitle")
        title = (title_el.text or "").strip() if title_el is not None else ""
        return InsiderBuy(
            role=title if is_officer else "",
            value_usd=value_usd,
            is_officer=is_officer,
            filing_url=filing_url,
        )

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
        return _amount(transaction, "transactionShares")

    @staticmethod
    def _transaction_price(transaction: ET.Element) -> float | None:
        """Cena za akcję; brak (sam przypis) albo niedodatnia → None."""
        price = _amount(transaction, "transactionPricePerShare")
        return price if price is not None and price > 0 else None


def _amount(transaction: ET.Element, name: str) -> float | None:
    """Liczba z pola kwoty transakcji. Pole ma zwykle wewnętrzny <value> —
    szukamy go, a w razie braku bierzemy bezpośredni tekst węzła."""
    amount_el = _find_local(transaction, name)
    if amount_el is None:
        return None
    value_el = _find_local(amount_el, "value")
    text = value_el.text if value_el is not None else amount_el.text
    if text is None or not text.strip():
        return None
    try:
        return float(text.strip())
    except ValueError:
        return None


def _filing_url(cik: str, accession: str, primary_doc: str) -> str:
    """Link do Form 4 dla człowieka — `primaryDocument` bez zmian, więc
    z prefiksem `xsl…/` to widok HTML renderowany przez SEC."""
    return ARCHIVE_URL.format(cik=int(cik), accession=accession.replace("-", ""), doc=primary_doc)


class NullInsiderFlowAdapter(InsiderFlowPort):
    """Domyślny adapter Fast-Loop / brak konfiguracji — zero I/O, zawsze None."""

    def get_insider_flow(self, symbol: str) -> InsiderFlowSnapshot | None:
        return None
