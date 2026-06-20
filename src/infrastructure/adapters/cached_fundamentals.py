"""Dekorator cache'ujący dla FundamentalsPort oraz Null implementation.

CachedFundamentalsAdapter pyta repo o świeży snapshot (TTL po stronie repo);
jeśli pusto, deleguje do prawdziwego źródła i zapisuje wynik.

NullFundamentalsAdapter zawsze zwraca None — fast loop wstrzykuje go jako
delegate, dzięki czemu fast loop nigdy nie woła płatnego API Alpha Vantage
(czyta wyłącznie cache).
"""

from __future__ import annotations

import threading

from src.application.ports import FundamentalsPort, RepositoryPort
from src.domain.value_objects import Fundamentals


class NullFundamentalsAdapter(FundamentalsPort):
    """No-op źródło danych — używane jako delegate w fast loop."""

    def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        return None


class CachedFundamentalsAdapter(FundamentalsPort):
    """Read-through cache: repo najpierw, delegate na fallback.

    Single-flight per symbol: dwa nakładające się cykle dla tego samego
    symbolu (read cache → miss → płatny delegate → save) bez koordynacji
    obie spaliłyby request AV. Per-symbol lock serializuje tylko ten sam
    symbol; po wejściu w lock cache jest re-sprawdzany, więc drugi wołający
    dostaje świeżo wypełniony wpis zamiast ponownie płacić.

    UWAGA: lock jest in-process (`threading.Lock`) — chroni tylko przed
    równoległością W TYM SAMYM procesie. Dwa osobne procesy/agenty na tej
    samej bazie mogą wciąż się ścigać; to akceptowalny kompromis dla
    pojedynczego procesu main_agent na cykl.
    """

    def __init__(
        self, repo: RepositoryPort, delegate: FundamentalsPort
    ) -> None:
        self._repo = repo
        self._delegate = delegate
        # Rejestr locków per symbol + lock chroniący sam rejestr przed
        # wyścigiem przy tworzeniu wpisu dla nowego symbolu.
        self._symbol_locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def _lock_for(self, symbol: str) -> threading.Lock:
        with self._registry_lock:
            lock = self._symbol_locks.get(symbol)
            if lock is None:
                lock = threading.Lock()
                self._symbol_locks[symbol] = lock
            return lock

    def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        # Szybka ścieżka: trafienie w cache bez wchodzenia w per-symbol lock.
        cached = self._repo.get_cached_fundamentals(symbol)
        if cached is not None:
            return cached

        with self._lock_for(symbol):
            # Re-check po zdobyciu locka: pierwszy wołający mógł właśnie
            # wypełnić cache, gdy czekaliśmy — wtedy nie płacimy drugi raz.
            cached = self._repo.get_cached_fundamentals(symbol)
            if cached is not None:
                return cached
            fresh = self._delegate.get_fundamentals(symbol)
            if fresh is not None:
                self._repo.save_fundamentals(symbol, fresh)
            return fresh
