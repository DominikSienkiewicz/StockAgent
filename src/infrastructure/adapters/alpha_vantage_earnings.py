"""Adapter AlphaVantage EARNINGS_CALENDAR — najbliższy raport wyników per symbol.

AV zwraca kalendarz wyników jako **CSV** (nie JSON) z nagłówkiem:
    symbol,name,reportDate,fiscalDateEnding,estimate,currency

FIN-09: zapytanie BEZ parametru `symbol` zwraca pełną listę spółek z horyzontu,
więc adapter pobiera kalendarz raz na proces — leniwie, przy pierwszym pytaniu,
pod `threading.Lock` (symbole idą równolegle, `symbol_concurrency`) — i obsługuje
każdy symbol z pamięci: najwcześniejsza PRZYSZŁA data raportu (reportDate >= today)
i `days_until`. Zamiast ~39 zapytań na cykl jedno. Wynik nieudanej próby (sieć,
wyczerpane klucze, odpowiedź bez kalendarza) też jest zapamiętany — kolejne symbole
nie ponawiają zapytania, które właśnie zawiodło. `today` jest wstrzykiwalne dla
deterministycznych testów — `date.today()` liczymy leniwie (przy wywołaniu).

Opcjonalne źródło alpha — błędy są łapane wewnętrznie i dają `None`; adapter nigdy
nie rzuca wyjątkiem.

FIN-11: `reserve_keys` to dalsze klucze z ALPHA_VANTAGE_API_KEYS. Soft-limit AV
(JSON z `Information`/`Note` zamiast CSV) rotuje klucz wg reguł DAT-06 — `Note`
dostaje jedno odczekanie na tym samym kluczu, `Information` od razu przesuwa na
kolejny. FIN-09: gdy soft-limit odbiją wszystkie klucze, `QuotaMonitor` dostaje
jeden WARNING (kalendarz w tym przebiegu jest pusty, bramka earnings bez mnożnika).
ARCH-33: sieć, status i treść idą przez pomocniki `_http` w taksonomii portów."""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

import requests

from src.application.ports import EarningsCalendarPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.earnings import EarningsEvent
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.infrastructure.adapters._alpha_vantage_common import (
    NOTE_BACKOFF_SECONDS,
    is_frequency_limit,
    is_soft_limit,
    soft_limit_text,
)
from src.infrastructure.adapters._http import (
    build_session,
    raise_for_status_translated,
    translated_get,
)

DEFAULT_BASE_URL = "https://www.alphavantage.co"
DEFAULT_TIMEOUT = 10.0
DEFAULT_HORIZON = "3month"

logger = logging.getLogger(__name__)


class AlphaVantageEarningsAdapter(EarningsCalendarPort):
    """Najbliższe zdarzenie wyników z jednego AV EARNINGS_CALENDAR (CSV) na proces."""

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        today: date | None = None,
        base_url: str = DEFAULT_BASE_URL,
        *,
        reserve_keys: Sequence[str] = (),
        quota_monitor: QuotaMonitor | None = None,
    ) -> None:
        self._api_keys = [api_key, *reserve_keys]
        # Indeks aktywnego klucza — przesuwany wyłącznie przy soft-limicie (jak
        # w adapterze fundamentów), więc rezerwa zostaje na resztę przebiegu.
        self._key_index = 0
        self._session = session or build_session()
        self._timeout = timeout
        # Wstrzyknięte `today` zamrażamy; bez niego liczymy leniwie przy wywołaniu.
        self._today = today
        self._base_url = base_url.rstrip("/")
        self._quota_monitor = quota_monitor
        # FIN-09: symbol → daty raportów z jednego pobrania; None = jeszcze nie pytano.
        self._reports: dict[str, list[date]] | None = None
        self._lock = threading.Lock()

    def get_next_earnings(self, symbol: str) -> EarningsEvent | None:
        today = self._today or date.today()
        upcoming = [day for day in self._calendar(symbol).get(symbol.upper(), []) if day >= today]
        if not upcoming:
            return None
        earliest = min(upcoming)
        return EarningsEvent(
            symbol=symbol,
            days_until=(earliest - today).days,
            report_date=earliest.isoformat(),
        )

    def _calendar(self, asking_symbol: str) -> dict[str, list[date]]:
        with self._lock:
            if self._reports is None:
                self._reports = self._fetch_reports(asking_symbol)
            return self._reports

    def _fetch_reports(self, asking_symbol: str) -> dict[str, list[date]]:
        try:
            body = self._fetch_csv()
        # Opcjonalne źródło — nigdy nie rzuca w górę; każdy błąd degradujemy do pustego
        # kalendarza. Log niesie symbol, który pierwszy zapytał (pobranie jest jedno).
        except Exception:
            logger.exception("AlphaVantage earnings fetch failed for %s", asking_symbol)
            return {}
        return _parse_reports(body) if body is not None else {}

    def _fetch_csv(self) -> str | None:
        backed_off = False
        while self._key_index < len(self._api_keys):
            response = translated_get(
                self._session,
                f"{self._base_url}/query",
                params={
                    "function": "EARNINGS_CALENDAR",
                    "horizon": DEFAULT_HORIZON,
                    "apikey": self._api_keys[self._key_index],
                },
                timeout=self._timeout,
            )
            raise_for_status_translated(response)
            text = response.text
            # AV przy rate-limit / błędzie zwraca JSON zamiast CSV — wykrywamy to
            # po wiodącym `{` i odmawiamy parsowania jako CSV.
            if not text.lstrip().startswith("{"):
                return text
            payload = _json_object(text)
            marker = soft_limit_text(payload, self._api_keys)
            if not is_soft_limit(payload, ()):
                # Np. `Error Message` (zły klucz, zła funkcja) — inny klucz nic nie zmieni.
                logger.warning("AlphaVantage returned a JSON error instead of the earnings CSV")
                return None
            if is_frequency_limit(payload) and not backed_off:
                logger.warning(
                    "AlphaVantage earnings key #%d hit the frequency limit: %s — "
                    "retrying the same key in %.1f s.",
                    self._key_index + 1,
                    marker,
                    NOTE_BACKOFF_SECONDS,
                )
                backed_off = True
                time.sleep(NOTE_BACKOFF_SECONDS)
                continue
            logger.warning(
                "AlphaVantage earnings key #%d exhausted: %s — rotating.",
                self._key_index + 1,
                marker,
            )
            self._key_index += 1
            backed_off = False
        self._report_keys_exhausted()
        return None

    def _report_keys_exhausted(self) -> None:
        """Log po angielsku + jeden WARNING po polsku (MAIL-15): kalendarz w tym
        przebiegu jest pusty, bramka earnings działa bez mnożnika."""
        logger.warning("AlphaVantage earnings: every key is exhausted — no earnings dates")
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source="Alpha Vantage earnings",
                severity=QuotaSeverity.WARNING,
                message=(
                    f"Kalendarz wyników Alpha Vantage odbił limit na wszystkich kluczach "
                    f"({len(self._api_keys)}); w tym przebiegu bramka earnings działa "
                    "bez mnożnika przed raportami."
                ),
                action=(
                    "Dodaj kolejny klucz w ALPHA_VANTAGE_API_KEYS (darmowy na "
                    "alphavantage.co) albo poczekaj na reset limitu o północy UTC."
                ),
                occurred_at=datetime.now(UTC),
            )
        )


def _parse_reports(body: str) -> dict[str, list[date]]:
    """CSV kalendarza → symbol (wielkie litery) → daty raportów. Wiersz bez symbolu
    albo z nieparsowalną datą jest pomijany."""
    reports: dict[str, list[date]] = {}
    for row in csv.DictReader(io.StringIO(body)):
        symbol = (row.get("symbol") or "").strip().upper()
        raw = (row.get("reportDate") or "").strip()
        if not symbol or not raw:
            continue
        try:
            reports.setdefault(symbol, []).append(date.fromisoformat(raw))
        except ValueError:
            continue
    return reports


def _json_object(text: str) -> dict[str, Any]:
    """Treść JSON odpowiedzi AV jako słownik; nieparsowalna albo nie-obiekt → {}."""
    try:
        payload = json.loads(text)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


class NullEarningsCalendarAdapter(EarningsCalendarPort):
    """Default Fast-Loop / brak klucza — zero I/O, zawsze None."""

    def get_next_earnings(self, symbol: str) -> EarningsEvent | None:
        return None
