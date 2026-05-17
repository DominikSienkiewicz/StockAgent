"""Dekorator cache'ujący dla FundamentalsPort oraz Null implementation.

CachedFundamentalsAdapter pyta repo o świeży snapshot (TTL po stronie repo);
jeśli pusto, deleguje do prawdziwego źródła i zapisuje wynik.

NullFundamentalsAdapter zawsze zwraca None — fast loop wstrzykuje go jako
delegate, dzięki czemu fast loop nigdy nie woła płatnego API Alpha Vantage
(czyta wyłącznie cache).
"""

from __future__ import annotations

from src.application.ports import FundamentalsPort, RepositoryPort
from src.domain.value_objects import Fundamentals


class NullFundamentalsAdapter(FundamentalsPort):
    """No-op źródło danych — używane jako delegate w fast loop."""

    def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        return None


class CachedFundamentalsAdapter(FundamentalsPort):
    """Read-through cache: repo najpierw, delegate na fallback."""

    def __init__(
        self, repo: RepositoryPort, delegate: FundamentalsPort
    ) -> None:
        self._repo = repo
        self._delegate = delegate

    def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        cached = self._repo.get_cached_fundamentals(symbol)
        if cached is not None:
            return cached
        fresh = self._delegate.get_fundamentals(symbol)
        if fresh is not None:
            self._repo.save_fundamentals(symbol, fresh)
        return fresh
