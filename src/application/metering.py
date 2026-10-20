"""FIN-02: liczenie płatnych wywołań per REALNY request do dostawcy LLM.

Rada doradcza robi w jednym przebiegu N wywołań person, chairmana i (w trybie
debaty) rundę rewizji. Węzeł grafu widzi z tego jedno `council_port.analyze`,
więc licznik w węźle zaniżał radę 8–15 razy. Dekorator portu LLM liczy każde
wywołanie u źródła — zanim oddamy je dostawcy, więc request, który potem padnie
albo nie zdąży przed limitem czasu rady, też jest policzony (poszedł i kosztuje).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from src.application.cycle_telemetry import CycleTelemetry
from src.application.ports import LLMPort
from src.application.usage_ledger import usage_scope

logger = logging.getLogger(__name__)


class MeteredLLMPort(LLMPort):
    """Port LLM, który rejestruje każde wywołanie w telemetrii cyklu pod
    kluczem cennika `source` (np. `council_llm`) i deleguje do `inner`.

    FIN-13: wywołanie `inner` idzie w zakresie `usage_scope(source=...)`, więc
    tokeny zgłoszone przez adapter trafiają do ledgera pod tym samym kluczem
    cennika co licznik. `node` (opcjonalny) nadaje węzeł wywołaniom spoza grafu
    (joby Slow Loopa); bez niego węzeł i symbol dziedziczą z zakresu wołającego."""

    def __init__(
        self,
        inner: LLMPort,
        telemetry: CycleTelemetry,
        source: str,
        node: str | None = None,
    ) -> None:
        self._inner = inner
        self._telemetry = telemetry
        self._source = source
        self._node = node

    def analyze(self, prompt: str) -> dict[str, Any]:
        self._telemetry.record_paid_call(self._source)
        with usage_scope(source=self._source, node=self._node):
            return self._inner.analyze(prompt)

    def analyze_mistake(self, prompt: str) -> str:
        self._telemetry.record_paid_call(self._source)
        with usage_scope(source=self._source, node=self._node):
            return self._inner.analyze_mistake(prompt)


class PaidBudgetExhausted(RuntimeError):
    """FIN-30: wywołanie LLM ponad przydział budżetu — odrzucone przed dostawcą."""


class BudgetedLLMPort(LLMPort):
    """Port LLM z twardym przydziałem `allowed` wywołań dla jednego joba.

    Wywołanie ponad przydział rzuca `PaidBudgetExhausted` ZANIM dotrze do `inner`
    (i do licznika `MeteredLLMPort` pod spodem), więc use case z deterministycznym
    fallbackiem idzie swoją ścieżką awaryjną bez płatnego requestu."""

    def __init__(self, inner: LLMPort, allowed: int, job: str) -> None:
        self._inner = inner
        self._remaining = allowed
        self._job = job
        self._lock = threading.Lock()

    def analyze(self, prompt: str) -> dict[str, Any]:
        self._spend()
        return self._inner.analyze(prompt)

    def analyze_mistake(self, prompt: str) -> str:
        self._spend()
        return self._inner.analyze_mistake(prompt)

    def _spend(self) -> None:
        with self._lock:
            if self._remaining <= 0:
                logger.warning(
                    "paid LLM budget exhausted for job %s — call refused before "
                    "the provider, the job falls back without LLM",
                    self._job,
                )
                raise PaidBudgetExhausted(f"paid LLM budget exhausted for job {self._job}")
            self._remaining -= 1
