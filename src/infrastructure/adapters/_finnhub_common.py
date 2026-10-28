"""Wspólne reguły adapterów Finnhub (DAT-12): budżet zapytań i reakcja na statusy.

Darmowy plan Finnhub to 60 zapytań na minutę na KLUCZ, a proces ma kilka
adapterów na tym samym kluczu (/quote, analitycy, opcje). DI tworzy więc jeden
`RateLimiter` na proces i wstrzykuje go wszystkim; Fast Loop i Shock Watch dzielą
klucz, więc każdy dostaje swój udział w limicie. `FinnhubStatusWatcher` mówi głośno
o statusach, które adaptery opcjonalnych źródeł alpha dotąd połykały jak „brak danych”.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

import requests

from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.infrastructure.adapters._http import build_session

# Fast Loop i Shock Watch to osobne procesy na jednym kluczu i mogą chodzić naraz:
# kolejka GHA ich nie rozdziela, bo godzinny watch wypychałby z niej czekający Fast Loop.
# Dzielą więc 60 zapytań na minutę: 35 + 20 = 55, zapas 5 na drgania zegara.
FAST_LOOP_FINNHUB_REQUESTS_PER_MINUTE = 35
SHOCK_WATCH_FINNHUB_REQUESTS_PER_MINUTE = 20

# Akcja każdego alertu 429 Finnhub (/quote i źródła alpha): żaden proces nie przekracza
# swojego udziału, więc 429 znaczy trzeciego użytkownika klucza, nie za szybką pętlę.
FINNHUB_RATE_LIMITED_ACTION = (
    "Fast Loop i Shock Watch dzielą klucz po "
    f"{FAST_LOOP_FINNHUB_REQUESTS_PER_MINUTE} i {SHOCK_WATCH_FINNHUB_REQUESTS_PER_MINUTE} "
    "zapytań/min, więc w tej minucie FINNHUB_API_KEY brał jeszcze ktoś inny: zwykle "
    "lokalny przebieg z kluczem produkcyjnym albo nowy job bez własnego udziału w limicie. "
    "Nie uruchamiaj lokalnie z tym kluczem w trakcie cronów."
)

# BEZ-15: token jedzie nagłówkiem, nie `&token=` w URL-u — URL trafia do treści
# wyjątków `requests`, logów urllib3 i tracebacków.
FINNHUB_TOKEN_HEADER = "X-Finnhub-Token"

_AUTH_STATUSES = frozenset({401, 403})
_RATE_LIMITED = 429

logger = logging.getLogger(__name__)


def finnhub_auth_headers(api_key: str) -> dict[str, str]:
    """Nagłówki uwierzytelnienia żądania Finnhub (BEZ-15)."""
    return {FINNHUB_TOKEN_HEADER: api_key}


def build_finnhub_session() -> requests.Session:
    """Sesja adaptera Finnhub: 429 wraca od razu do adaptera. Ponowienia urllib3
    szłyby obok wspólnego limitera, a minutowe okno Finnhub trwa dłużej niż ich
    backoff — każde 429 kosztowało cztery zapytania i przedłużało blokadę klucza."""
    return build_session(retry_rate_limited=False)


class FinnhubStatusWatcher:
    """429 → jeden CRITICAL `QuotaAlert` na endpoint na czas życia adaptera (jeden
    cykl, decyzja 2026-10-24): przy wspólnym limiterze 429 znaczy wyczerpany limit
    planu, więc dane źródła znikną do końca cyklu — kolejne 429 tylko w logu.
    401/403 → jeden WARNING w logu na endpoint (403 bywa planem premium, więc
    powtórka per symbol byłaby szumem). Bez tokenu w komunikatach."""

    def __init__(self, source: str, label_pl: str, quota_monitor: QuotaMonitor | None) -> None:
        self._source = source
        self._label_pl = label_pl
        self._quota_monitor = quota_monitor
        self._warned: set[tuple[str, int]] = set()
        self._alerted_endpoints: set[str] = set()
        self._lock = threading.Lock()

    def observe(self, endpoint: str, symbol: str, status: int) -> None:
        if status == _RATE_LIMITED:
            self._rate_limited(endpoint, symbol)
        elif status in _AUTH_STATUSES:
            self._warn_once(endpoint, status)

    def _warn_once(self, endpoint: str, status: int) -> None:
        with self._lock:
            first = (endpoint, status) not in self._warned
            self._warned.add((endpoint, status))
        if first:
            logger.warning(
                "%s %s returned %d (key rejected or premium endpoint) — "
                "further %d responses from this endpoint are not logged",
                self._source,
                endpoint,
                status,
                status,
            )

    def _rate_limited(self, endpoint: str, symbol: str) -> None:
        logger.warning(
            "%s %s returned 429 for '%s' — shared Finnhub budget (60 req/min) exhausted",
            self._source,
            endpoint,
            symbol,
        )
        with self._lock:
            first = endpoint not in self._alerted_endpoints
            self._alerted_endpoints.add(endpoint)
        if self._quota_monitor is None or not first:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source=self._source,
                severity=QuotaSeverity.CRITICAL,
                message=(
                    f"Finnhub zwrócił 429 dla '{symbol}' ({self._label_pl}) — wyczerpany "
                    "wspólny limit 60 zapytań/min; temu źródłu może brakować danych do "
                    "końca cyklu (kolejne 429 tylko w logu)."
                ),
                action=FINNHUB_RATE_LIMITED_ACTION,
                occurred_at=datetime.now(UTC),
            )
        )
