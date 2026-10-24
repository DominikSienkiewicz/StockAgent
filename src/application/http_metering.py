"""NIEZ-24: licznik żądań HTTP per host dla jednego przebiegu.

Limity dostawców (Finnhub 60/min, Alpha Vantage 25/dobę/klucz, SEC) były
pilnowane po omacku — nikt nie wiedział, ile żądań idzie na cykl. Miernik liczy
per host żądania, klasy statusów, retry urllib3 i sumaryczny czas odpowiedzi.
Zasila go hook sesji `requests` z `infrastructure/adapters/_http.py`; SDK na
httpx (OpenAI, Anthropic, supabase-py) pokrywa ledger `llm_calls`, nie ten licznik.

Klucz to sam host: ścieżka i query (klucze API, G23) nigdy tu nie trafiają.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict

REQUESTS_KEY = "requests"
RETRIES_KEY = "retries"
# 429 osobno od reszty 4xx: to sygnał limitu dostawcy, nie błąd żądania.
RATE_LIMITED = "429"


def status_class(status: int) -> str:
    """Klasa statusu HTTP: "2xx", "3xx", "4xx" (bez 429), "429", "5xx" albo "other"."""
    if status == 429:
        return RATE_LIMITED
    if 200 <= status < 600:
        return f"{status // 100}xx"
    return "other"


class HttpCallMeter:
    """Wątkowo bezpieczny licznik żądań per host — jeden na przebieg."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self._elapsed: defaultdict[str, float] = defaultdict(float)

    def record(self, host: str, status: int, *, retries: int = 0, elapsed_s: float = 0.0) -> None:
        with self._lock:
            counts = self._counts[host]
            counts[REQUESTS_KEY] += 1
            counts[status_class(status)] += 1
            if retries:
                counts[RETRIES_KEY] += retries
            self._elapsed[host] += elapsed_s

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Kopia liczników per host, bez zerowych pozycji."""
        with self._lock:
            return {
                host: {key: value for key, value in counts.items() if value}
                for host, counts in self._counts.items()
            }

    def elapsed_seconds(self) -> dict[str, float]:
        """Suma czasu odpowiedzi per host (sekundy)."""
        with self._lock:
            return dict(self._elapsed)
