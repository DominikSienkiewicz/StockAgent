"""Adapter AlphaVantage EARNINGS_CALENDAR — najbliższy raport wyników per symbol.

AV zwraca kalendarz wyników jako **CSV** (nie JSON) z nagłówkiem:
    symbol,name,reportDate,fiscalDateEnding,estimate,currency

Wybieramy najwcześniejszą PRZYSZŁĄ datę raportu (reportDate >= today) i liczymy
`days_until`. `today` jest wstrzykiwalne dla deterministycznych testów —
`date.today()` liczymy leniwie (przy wywołaniu), nie na imporcie.

Opcjonalne źródło alpha — wszystkie błędy (sieć, parse, brak danych, non-200,
JSON note przy rate-limit) są łapane wewnętrznie i dają `None`. Adapter nigdy
nie rzuca wyjątkiem."""

from __future__ import annotations

import csv
import io
import logging
from datetime import date

import requests

from src.application.ports import EarningsCalendarPort
from src.domain.earnings import EarningsEvent
from src.infrastructure.adapters._http import build_session

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
    ) -> None:
        self._api_key = api_key
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
        response = self._session.get(
            f"{self._base_url}/query",
            params={
                "function": "EARNINGS_CALENDAR",
                "symbol": symbol,
                "horizon": DEFAULT_HORIZON,
                "apikey": self._api_key,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        text = response.text
        # AV przy rate-limit / błędzie zwraca JSON note zamiast CSV — wykrywamy
        # to po wiodącym `{` i odmawiamy parsowania jako CSV.
        if text.lstrip().startswith("{"):
            logger.warning(
                "AlphaVantage returned a JSON note instead of CSV for %s "
                "(rate-limit?): %s",
                symbol,
                text.strip()[:160],
            )
            return None
        return text

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


class NullEarningsCalendarAdapter(EarningsCalendarPort):
    """Default Fast-Loop / brak klucza — zero I/O, zawsze None."""

    def get_next_earnings(self, symbol: str) -> EarningsEvent | None:
        return None
