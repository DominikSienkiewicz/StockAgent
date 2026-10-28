"""Adapter Finnhub /calendar/earnings — kalendarz wyników całej watchlisty (DAT-20).

    GET /calendar/earnings?from=YYYY-MM-DD&to=YYYY-MM-DD (token w nagłówku, BEZ-15)
    → {"earningsCalendar": [{"symbol": "AAPL", "date": "2026-10-29",
                             "hour": "amc", "epsEstimate": 1.5, ...}, ...]}

Bez parametru `symbol` Finnhub zwraca wszystkie spółki z okna, więc JEDNO zapytanie
na proces (leniwie, pod lockiem — symbole idą równolegle) obsługuje całą
watchlistę; odpowiedź filtrujemy do watchlisty i trzymamy na czas życia adaptera
(jeden cykl). Nieudana albo bezużyteczna odpowiedź też jest zapamiętana: kolejne
symbole nie ponawiają zapytania, które właśnie zawiodło (limit 60/min jest
wspólny z /quote, analitykami i opcjami — `FINNHUB_RATE_LIMITER`).

Darmowy plan może ograniczać zakres dat albo pola (do sprawdzenia sondą przed
włączeniem); `hour` bywa pusty — wtedy pora raportu jest nieznana. Źródło
opcjonalne: błąd dostawcy → brak zdarzeń; 401/403/429 mówi `FinnhubStatusWatcher`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import requests

from src.application.errors import PortError
from src.application.ports import EarningsCalendarPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.earnings import EarningsEvent, EarningsTiming
from src.infrastructure.adapters._finnhub_common import (
    FinnhubStatusWatcher,
    build_finnhub_session,
    finnhub_auth_headers,
)
from src.infrastructure.adapters._http import (
    RateLimiter,
    json_payload,
    translated_get,
)

DEFAULT_BASE_URL = "https://finnhub.io/api/v1"
DEFAULT_TIMEOUT = 10.0
# Okno kalendarza: sekcja tygodnia potrzebuje 7 dni, bramka earnings — najbliższego
# raportu; 30 dni łapie kolejny sezon wyników bez pytania o kwartał naprzód.
DEFAULT_HORIZON_DAYS = 30
_ENDPOINT = "calendar/earnings"
_WATCHLIST = "watchlista"

logger = logging.getLogger(__name__)


def _utc_today() -> date:
    return datetime.now(UTC).date()


@dataclass(frozen=True)
class _ScheduledReport:
    report_date: date
    timing: EarningsTiming | None
    eps_estimate: float | None


class FinnhubEarningsCalendarAdapter(EarningsCalendarPort):
    """`EarningsCalendarPort` z jednego zapytania Finnhub na watchlistę."""

    def __init__(
        self,
        api_key: str,
        symbols: Iterable[str],
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: str = DEFAULT_BASE_URL,
        *,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
        quota_monitor: QuotaMonitor | None = None,
        rate_limiter: RateLimiter | None = None,
        today: Callable[[], date] = _utc_today,
    ) -> None:
        self._api_key = api_key
        self._symbols = frozenset(symbol.upper() for symbol in symbols)
        self._session = session if session is not None else build_finnhub_session()
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")
        self._horizon_days = horizon_days
        self._rate_limiter = rate_limiter
        self._today = today
        self._status = FinnhubStatusWatcher("Finnhub earnings", "kalendarz wyników", quota_monitor)
        self._reports: dict[str, _ScheduledReport] | None = None
        self._lock = threading.Lock()

    def get_next_earnings(self, symbol: str) -> EarningsEvent | None:
        report = self._calendar().get(symbol.upper())
        if report is None:
            return None
        days_until = (report.report_date - self._today()).days
        if days_until < 0:
            return None
        return EarningsEvent(
            symbol=symbol,
            days_until=days_until,
            report_date=report.report_date.isoformat(),
            timing=report.timing,
            eps_estimate=report.eps_estimate,
        )

    def _calendar(self) -> dict[str, _ScheduledReport]:
        with self._lock:
            if self._reports is None:
                self._reports = self._fetch()
            return self._reports

    def _fetch(self) -> dict[str, _ScheduledReport]:
        today = self._today()
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            response = translated_get(
                self._session,
                f"{self._base_url}/{_ENDPOINT}",
                params={
                    "from": today.isoformat(),
                    "to": (today + timedelta(days=self._horizon_days)).isoformat(),
                },
                headers=finnhub_auth_headers(self._api_key),
                timeout=self._timeout,
            )
            self._status.observe(_ENDPOINT, _WATCHLIST, response.status_code)
            if response.status_code != 200:
                return {}
            payload = json_payload(response)
        except PortError:
            logger.warning(
                "Finnhub earnings calendar unavailable — no earnings dates this cycle",
                exc_info=True,
            )
            return {}
        rows = payload.get("earningsCalendar") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return {}
        return self._nearest_reports(rows, today)

    def _nearest_reports(self, rows: list[object], today: date) -> dict[str, _ScheduledReport]:
        """Najbliższy raport od `today` per symbol watchlisty."""
        nearest: dict[str, _ScheduledReport] = {}
        for row in rows:
            report = _scheduled_report(row)
            symbol = str(row.get("symbol", "")).upper() if isinstance(row, dict) else ""
            if report is None or symbol not in self._symbols or report.report_date < today:
                continue
            current = nearest.get(symbol)
            if current is None or report.report_date < current.report_date:
                nearest[symbol] = report
        return nearest


def _scheduled_report(row: object) -> _ScheduledReport | None:
    if not isinstance(row, dict):
        return None
    try:
        report_date = date.fromisoformat(str(row.get("date")))
    except ValueError:
        return None
    return _ScheduledReport(
        report_date=report_date,
        timing=_timing(row.get("hour")),
        eps_estimate=_eps(row.get("epsEstimate")),
    )


def _eps(raw: object) -> float | None:
    """Szacunek EPS; brak (null) albo nie-liczba → None."""
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return float(raw)


def _timing(raw: object) -> EarningsTiming | None:
    try:
        return EarningsTiming(str(raw).strip().lower())
    except ValueError:
        return None
