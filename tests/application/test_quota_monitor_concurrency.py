"""Testy QuotaMonitor — agregacja alertów i thread-safety debounce (U4).

Race check-then-act na `_alerted_sources`: dwa wątki z tym samym source mogą
oba przejść warunek `in` przed dodaniem → podwójny push. Rada już dziś woła
record() z 7 wątków; zrównoleglenie symboli mnoży kontencję.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity


def _critical(source: str) -> QuotaAlert:
    return QuotaAlert(
        source=source,
        severity=QuotaSeverity.CRITICAL,
        message="m",
        action="a",
        occurred_at=datetime.now(UTC),
    )


class _SlowContainsSet(set):  # type: ignore[type-arg]
    """`__contains__` blokuje się na `_release` Event, a `add` śpi przed
    dodaniem elementu. Razem deterministycznie poszerzają okno check→add
    w _maybe_push_realtime.

    Mechanizm (bez locka):
    1. Wszystkie N wątków startują jednocześnie (bariera w worker).
    2. Każdy wątek wchodzi do `__contains__` i czeka na `_release.wait()`.
    3. Main śpi 0.05 s (czas na dojście wszystkich wątków), potem ustawia Event.
    4. Wszystkie N wątków wychodzą z `__contains__` z wynikiem False (set pusty).
    5. `add` śpi 0.05 s przed dodaniem elementu — okno pozwala wszystkim wątkom
       minąć check i przejść do send_alert zanim pierwszy wątek doda do seta.
    6. Bez locka: call_count == N (błąd). Z lockiem: call_count == 1 (poprawnie).

    Mechanizm (z lockiem — zielony):
    1. Tylko jeden wątek wchodzi do record() naraz.
    2. `_release` jest już ustawiony (main ustawia go bezwarunkowo po 0.05 s).
    3. Wątek 1: __contains__ → False, add (śpi 0.05 s), send_alert.
    4. Wątek 2: __contains__ → True (element już jest), pomija send_alert.
    5. call_count == 1 (poprawnie).
    """

    def __init__(self) -> None:
        super().__init__()
        self._release = threading.Event()

    def __contains__(self, item: object) -> bool:
        self._release.wait()  # czekaj aż main zwolni wszystkie wątki naraz
        return super().__contains__(item)

    def add(self, item: object) -> None:  # type: ignore[override]
        # Śpij dłużej niż czas od zwolnienia _release do zakończenia __contains__
        # we wszystkich wątkach — gwarantuje, że wszystkie N wątków zdążą
        # zobaczyć False zanim pierwszy wątek doda element do seta.
        time.sleep(0.05)
        super().add(item)  # type: ignore[arg-type]


def test_concurrent_record_same_source_pushes_once() -> None:
    """N wątków, ten sam source. _SlowContainsSet wymusza deterministyczny
    interleaving: bez locka wszystkie N przejdą check zanim którykolwiek
    skończy add → N pushy. Z lockiem record() jest serializowane → 1 push."""
    notifier = MagicMock()
    monitor = QuotaMonitor(alert_notifier=notifier)
    n = 8
    slow_set: _SlowContainsSet = _SlowContainsSet()
    monitor._alerted_sources = slow_set  # type: ignore[assignment]

    start_barrier = threading.Barrier(n)

    def worker() -> None:
        start_barrier.wait()
        monitor.record(_critical("Alpha Vantage"))

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()

    # Odczekaj aż wszystkie wątki są w __contains__.wait(), potem odblokuj.
    time.sleep(0.05)
    slow_set._release.set()

    for t in threads:
        t.join()

    assert notifier.send_alert.call_count == 1
    # Wszystkie alerty trafiają do raportu dobowego (append nic nie gubi).
    assert len(monitor.alerts) == n


def test_concurrent_record_distinct_sources_all_retained() -> None:
    monitor = QuotaMonitor(alert_notifier=MagicMock())
    n = 16
    barrier = threading.Barrier(n)

    def worker(i: int) -> None:
        barrier.wait()
        monitor.record(_critical(f"src-{i}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(monitor.alerts) == n
