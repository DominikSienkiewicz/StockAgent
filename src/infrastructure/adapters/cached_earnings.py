"""FIN-11: read-through cache kalendarza wyników (`earnings_cache`, migracja 053).

Bramka earnings potrzebuje daty raportu PRZED bramką volatility, więc w Fast Loopie
każdy symbol płacił request Alpha Vantage w każdym cyklu, jednym kluczem i bez
licznika. Teraz Slow Loop odświeża cache raz w tygodniu z prawdziwym adapterem AV
(lista kluczy), a Fast Loop dostaje ten sam dekorator z delegatem Null — czyta
wyłącznie cache.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

from src.application.ports import EarningsCalendarPort, RepositoryPort
from src.domain.earnings import CachedEarningsDate, EarningsEvent


def _utc_today() -> date:
    return datetime.now(UTC).date()


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CachedEarningsAdapter(EarningsCalendarPort):
    """Repo najpierw (wpis młodszy niż TTL), delegat przy braku wpisu.

    Do cache trafia wyłącznie znaleziony raport z datą. Adapter AV oddaje None
    także przy błędzie, więc zapisany „brak raportu” utrwaliłby awarię na tydzień;
    przy pustym cache Fast Loop i tak dostaje None (bramka bez mnożnika)."""

    def __init__(
        self,
        repo: RepositoryPort,
        delegate: EarningsCalendarPort,
        *,
        today: Callable[[], date] = _utc_today,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._repo = repo
        self._delegate = delegate
        self._today = today
        self._clock = clock

    def get_next_earnings(self, symbol: str) -> EarningsEvent | None:
        cached = self._repo.get_cached_earnings(symbol)
        if cached is not None:
            return cached.event(self._today())
        fresh = self._delegate.get_next_earnings(symbol)
        if fresh is not None and fresh.report_date is not None:
            self._repo.save_earnings(
                CachedEarningsDate(
                    symbol=symbol,
                    report_date=date.fromisoformat(fresh.report_date),
                    fetched_at=self._clock(),
                )
            )
        return fresh
