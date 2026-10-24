"""ARCH-11: jeden wątkowo bezpieczny kolektor metryk cyklu Fast Loopa.

Licznik płatnych wywołań, czasy węzłów i decyzje bramki volatility zbiera JEDEN
obiekt na cykl, wstrzykiwany polem `AgentGraphDeps.telemetry`. Konsumenci (mail,
step summary, `cycle_runs`, budżet wywołań) czytają zamrożone `CycleMetrics`
ze `snapshot()`, zamiast parsować logi INFO.

Lock jest konieczny: rada woła persony w wątkach, a symbole mogą iść równolegle
(`symbol_concurrency`), więc `Counter[k] += 1` bez locka gubiłby inkrementy.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from types import MappingProxyType

from src.application.http_metering import HttpCallMeter


@dataclass(frozen=True)
class CycleMetrics:
    """Zamrożony stan metryk cyklu w chwili `CycleTelemetry.snapshot()`.

    - `paid_calls`    — płatne wywołania per klucz cennika `src/domain/finops.py`.
    - `node_p50_ms` / `node_p95_ms` / `node_max_ms` — czasy węzłów (nearest-rank).
    - `node_failures` — wywołania węzła zakończone wyjątkiem; `failures` to ich suma.
    - `gated` / `passed` — decyzje bramki volatility; `gate_reasons` — ich powody.
    - `http_calls` — żądania HTTP per host z `HttpCallMeter` (NIEZ-24); puste bez miernika.
    """

    paid_calls: Mapping[str, int]
    node_p50_ms: Mapping[str, float]
    node_p95_ms: Mapping[str, float]
    node_max_ms: Mapping[str, float]
    node_failures: Mapping[str, int]
    gated: int
    passed: int
    gate_reasons: Mapping[str, int]
    http_calls: Mapping[str, Mapping[str, int]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def failures(self) -> int:
        return sum(self.node_failures.values())

    @property
    def total_paid_calls(self) -> int:
        return sum(self.paid_calls.values())


def _nearest_rank(sorted_values: Sequence[float], quantile: float) -> float:
    """Percentyl metodą nearest-rank: wartość z próby, bez interpolacji."""
    rank = max(1, math.ceil(quantile * len(sorted_values)))
    return sorted_values[rank - 1]


class CycleTelemetry:
    """Kolektor metryk JEDNEGO cyklu. Wszystkie zapisy i `snapshot()` idą pod
    jednym lockiem, więc snapshot jest spójny między licznikami."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paid_calls: Counter[str] = Counter()
        self._timings: defaultdict[str, list[float]] = defaultdict(list)
        self._node_failures: Counter[str] = Counter()
        self._gate_reasons: Counter[str] = Counter()
        self._gated = 0
        self._passed = 0
        self._http_meter: HttpCallMeter | None = None

    def bind_http_meter(self, meter: HttpCallMeter) -> None:
        """NIEZ-24: miernik żądań HTTP przebiegu; `snapshot()` oddaje jego stan."""
        with self._lock:
            self._http_meter = meter

    @property
    def paid_calls(self) -> Counter[str]:
        """Żywy licznik płatnych wywołań (ten sam obiekt przez cały cykl).

        Zostaje dla czytelników, którzy znają goły Counter (`use_case.paid_calls`);
        nowi konsumenci biorą spójny odczyt ze `snapshot()`."""
        return self._paid_calls

    def bind_paid_calls(self, counter: Counter[str]) -> None:
        """Zgodność wstecz z `AgentGraphDeps(paid_call_meter=...)`: licznik
        wołającego staje się magazynem płatnych wywołań, z dotychczasowymi
        wpisami dopisanymi do niego."""
        with self._lock:
            if counter is self._paid_calls:
                return
            counter.update(self._paid_calls)
            self._paid_calls = counter

    def paid_calls_of(self, sources: Set[str]) -> int:
        """FIN-12: suma płatnych wywołań wskazanych źródeł, odczytana pod lockiem
        (symbole idą równolegle, a budżet cyklu czyta licznik w trakcie zapisów)."""
        with self._lock:
            return sum(self._paid_calls[source] for source in sources)

    def record_paid_call(self, source: str) -> None:
        with self._lock:
            self._paid_calls[source] += 1

    def record_node_timing(
        self, node: str, symbol: str, ms: float, *, failed: bool = False
    ) -> None:
        """Czas węzła dla symbolu; `failed` = węzeł zakończył się wyjątkiem.
        Symbol niesie log `_timed_node`; agregat jest per węzeł."""
        with self._lock:
            self._timings[node].append(ms)
            if failed:
                self._node_failures[node] += 1

    def record_gate(self, symbol: str, passed: bool, reason: str) -> None:
        """Decyzja bramki volatility dla symbolu (`reason` np. `below_threshold`)."""
        with self._lock:
            if passed:
                self._passed += 1
            else:
                self._gated += 1
            self._gate_reasons[reason] += 1

    def snapshot(self) -> CycleMetrics:
        with self._lock:
            http_calls = {} if self._http_meter is None else self._http_meter.snapshot()
            timings = {node: sorted(values) for node, values in self._timings.items()}
            return CycleMetrics(
                paid_calls=MappingProxyType(dict(self._paid_calls)),
                node_p50_ms=MappingProxyType(
                    {node: _nearest_rank(values, 0.50) for node, values in timings.items()}
                ),
                node_p95_ms=MappingProxyType(
                    {node: _nearest_rank(values, 0.95) for node, values in timings.items()}
                ),
                node_max_ms=MappingProxyType(
                    {node: values[-1] for node, values in timings.items()}
                ),
                node_failures=MappingProxyType(dict(self._node_failures)),
                gated=self._gated,
                passed=self._passed,
                gate_reasons=MappingProxyType(dict(self._gate_reasons)),
                http_calls=MappingProxyType(http_calls),
            )
