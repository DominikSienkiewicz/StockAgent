"""Wspólne reguły adapterów Finnhub (DAT-12): budżet zapytań i reakcja na statusy.

Darmowy plan Finnhub to 60 zapytań na minutę na KLUCZ, a proces ma kilka
adapterów na tym samym kluczu (/quote, analitycy, opcje). DI tworzy więc jeden
`RateLimiter` z `FINNHUB_REQUESTS_PER_MINUTE` (zapas pod 60) i wstrzykuje go
wszystkim. `FinnhubStatusWatcher` mówi głośno o statusach, które adaptery
opcjonalnych źródeł alpha dotąd połykały jak „brak danych”.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity

# 55 z 60 dozwolonych na minutę — zapas na drgania zegara i requesty spoza procesu.
FINNHUB_REQUESTS_PER_MINUTE = 55

_AUTH_STATUSES = frozenset({401, 403})
_RATE_LIMITED = 429

logger = logging.getLogger(__name__)


class FinnhubStatusWatcher:
    """429 → `QuotaAlert` (WARNING — źródło jest opcjonalne), 401/403 → jeden
    WARNING na endpoint na czas życia adaptera (403 bywa planem premium, więc
    powtórka per symbol byłaby szumem). Bez tokenu w komunikatach."""

    def __init__(self, source: str, label_pl: str, quota_monitor: QuotaMonitor | None) -> None:
        self._source = source
        self._label_pl = label_pl
        self._quota_monitor = quota_monitor
        self._warned: set[tuple[str, int]] = set()
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
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source=self._source,
                severity=QuotaSeverity.WARNING,
                message=(
                    f"Finnhub zwrócił 429 dla '{symbol}' ({self._label_pl}) — wyczerpany "
                    "wspólny limit 60 zapytań/min; ten sygnał dla symbolu pominięty."
                ),
                action=(
                    "Wyłącz część źródeł alpha Finnhub albo zmniejsz liczbę SYMBOLS; "
                    "większy limit daje płatny plan Finnhub."
                ),
                occurred_at=datetime.now(UTC),
            )
        )
