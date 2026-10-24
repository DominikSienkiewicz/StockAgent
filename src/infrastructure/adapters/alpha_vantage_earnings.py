"""Adapter AlphaVantage EARNINGS_CALENDAR — najbliższy raport wyników per symbol.

AV zwraca kalendarz wyników jako **CSV** (nie JSON) z nagłówkiem:
    symbol,name,reportDate,fiscalDateEnding,estimate,currency

Wybieramy najwcześniejszą PRZYSZŁĄ datę raportu (reportDate >= today) i liczymy
`days_until`. `today` jest wstrzykiwalne dla deterministycznych testów —
`date.today()` liczymy leniwie (przy wywołaniu), nie na imporcie.

Opcjonalne źródło alpha — wszystkie błędy (sieć, parse, brak danych, non-200,
JSON note przy rate-limit) są łapane wewnętrznie i dają `None`. Adapter nigdy
nie rzuca wyjątkiem.

FIN-11: `reserve_keys` to dalsze klucze z ALPHA_VANTAGE_API_KEYS. Soft-limit AV
(JSON z `Information`/`Note` zamiast CSV) rotuje klucz wg reguł DAT-06 — `Note`
dostaje jedno odczekanie na tym samym kluczu, `Information` od razu przesuwa na
kolejny — a po ostatnim kluczu adapter przestaje pytać do końca przebiegu."""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from collections.abc import Sequence
from datetime import date
from typing import Any

import requests

from src.application.ports import EarningsCalendarPort
from src.domain.earnings import EarningsEvent
from src.infrastructure.adapters._alpha_vantage_common import (
    NOTE_BACKOFF_SECONDS,
    is_frequency_limit,
    is_soft_limit,
    soft_limit_text,
)
from src.infrastructure.adapters._http import (
    build_session,
    raise_for_status_redacted,
    safe_get,
)

DEFAULT_BASE_URL = "https://www.alphavantage.co"
DEFAULT_TIMEOUT = 10.0
DEFAULT_HORIZON = "3month"

logger = logging.getLogger(__name__)


class AlphaVantageEarningsAdapter(EarningsCalendarPort):
    """Pobiera najbliższe zdarzenie wyników z AV EARNINGS_CALENDAR (CSV)."""

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        today: date | None = None,
        base_url: str = DEFAULT_BASE_URL,
        *,
        reserve_keys: Sequence[str] = (),
    ) -> None:
        self._api_keys = [api_key, *reserve_keys]
        # Indeks aktywnego klucza — przesuwany wyłącznie przy soft-limicie (jak
        # w adapterze fundamentów), więc rezerwa zostaje na dalsze symbole.
        self._key_index = 0
        self._session = session or build_session()
        self._timeout = timeout
        # Wstrzyknięte `today` zamrażamy; bez niego liczymy leniwie przy wywołaniu.
        self._today = today
        self._base_url = base_url.rstrip("/")

    def get_next_earnings(self, symbol: str) -> EarningsEvent | None:
        try:
            body = self._fetch_csv(symbol)
            if body is None:
                return None
            return self._parse(symbol, body)
        # Opcjonalne źródło — nigdy nie rzuca w górę; każdy błąd degradujemy do None.
        except Exception:  # noqa: BLE001
            logger.exception("AlphaVantage earnings fetch failed for %s", symbol)
            return None

    def _fetch_csv(self, symbol: str) -> str | None:
        backed_off = False
        while self._key_index < len(self._api_keys):
            response = safe_get(
                self._session,
                f"{self._base_url}/query",
                params={
                    "function": "EARNINGS_CALENDAR",
                    "symbol": symbol,
                    "horizon": DEFAULT_HORIZON,
                    "apikey": self._api_keys[self._key_index],
                },
                timeout=self._timeout,
            )
            raise_for_status_redacted(response)
            text = response.text
            # AV przy rate-limit / błędzie zwraca JSON zamiast CSV — wykrywamy to
            # po wiodącym `{` i odmawiamy parsowania jako CSV.
            if not text.lstrip().startswith("{"):
                return text
            payload = _json_object(text)
            marker = soft_limit_text(payload, self._api_keys)
            if not is_soft_limit(payload, ()):
                # Np. `Error Message` dla nieznanego symbolu — inny klucz nic nie zmieni.
                logger.warning(
                    "AlphaVantage returned a JSON error instead of CSV for %s", symbol
                )
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
                "AlphaVantage earnings key #%d exhausted for %s: %s — rotating.",
                self._key_index + 1,
                symbol,
                marker,
            )
            self._key_index += 1
            backed_off = False
        logger.warning("AlphaVantage earnings: every key is exhausted — %s skipped", symbol)
        return None

    def _parse(self, symbol: str, body: str) -> EarningsEvent | None:
        today = self._today or date.today()
        reader = csv.DictReader(io.StringIO(body))
        earliest: date | None = None
        for row in reader:
            raw = (row.get("reportDate") or "").strip()
            if not raw:
                continue
            try:
                report = date.fromisoformat(raw)
            except ValueError:
                # Wiersz z nieparsowalną datą pomijamy.
                continue
            if report < today:
                continue
            if earliest is None or report < earliest:
                earliest = report

        if earliest is None:
            return None

        return EarningsEvent(
            symbol=symbol,
            days_until=(earliest - today).days,
            report_date=earliest.isoformat(),
        )


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
