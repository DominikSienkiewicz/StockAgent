"""ARCH-11: `CycleTelemetry` — jeden wątkowo bezpieczny kolektor metryk cyklu.

Licznik płatnych wywołań, czasy węzłów i decyzje bramki volatility zbiera jeden
obiekt na cykl; `snapshot()` oddaje zamrożone `CycleMetrics`, więc mail, step
summary i przyszłe `cycle_runs` nie muszą parsować logów.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import Counter

import pytest

from src.application.cycle_telemetry import CycleMetrics, CycleTelemetry


def test_snapshot_aggregates_paid_calls_timings_and_gates() -> None:
    telemetry = CycleTelemetry()

    telemetry.record_paid_call("llm")
    telemetry.record_paid_call("llm")
    telemetry.record_node_timing("council", "NVDA", 1200.0)
    telemetry.record_gate("AAPL", False, "below_threshold")

    metrics = telemetry.snapshot()

    assert isinstance(metrics, CycleMetrics)
    assert metrics.paid_calls["llm"] == 2
    assert metrics.node_max_ms["council"] == 1200.0
    assert metrics.gated == 1
    assert metrics.passed == 0


def test_empty_cycle_is_free_and_has_no_timings() -> None:
    metrics = CycleTelemetry().snapshot()

    assert metrics.total_paid_calls == 0
    assert dict(metrics.paid_calls) == {}
    assert dict(metrics.node_max_ms) == {}
    assert metrics.gated == metrics.passed == metrics.failures == 0


def test_node_percentiles_use_nearest_rank() -> None:
    telemetry = CycleTelemetry()
    for ms in range(1, 21):
        telemetry.record_node_timing("predict", f"S{ms}", float(ms * 10))

    metrics = telemetry.snapshot()

    assert metrics.node_p50_ms["predict"] == 100.0
    assert metrics.node_p95_ms["predict"] == 190.0
    assert metrics.node_max_ms["predict"] == 200.0


def test_gate_counts_passed_gated_and_reasons() -> None:
    telemetry = CycleTelemetry()
    telemetry.record_gate("AAPL", True, "above_threshold")
    telemetry.record_gate("MSFT", False, "below_threshold")
    telemetry.record_gate("NVDA", False, "stale_reference")
    telemetry.record_gate("AMD", False, "below_threshold")

    metrics = telemetry.snapshot()

    assert metrics.passed == 1
    assert metrics.gated == 3
    assert dict(metrics.gate_reasons) == {
        "above_threshold": 1,
        "below_threshold": 2,
        "stale_reference": 1,
    }


def test_failed_node_timing_counts_as_failure() -> None:
    telemetry = CycleTelemetry()
    telemetry.record_node_timing("council", "NVDA", 30_000.0, failed=True)
    telemetry.record_node_timing("predict", "NVDA", 900.0)

    metrics = telemetry.snapshot()

    assert metrics.failures == 1
    assert dict(metrics.node_failures) == {"council": 1}
    assert metrics.node_max_ms["council"] == 30_000.0


def test_snapshot_is_frozen_and_detached_from_later_records() -> None:
    telemetry = CycleTelemetry()
    telemetry.record_paid_call("news")
    metrics = telemetry.snapshot()

    telemetry.record_paid_call("news")

    assert metrics.paid_calls["news"] == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.gated = 5  # type: ignore[misc]
    with pytest.raises(TypeError):
        metrics.paid_calls["news"] = 9  # type: ignore[index]


def test_paid_calls_is_the_live_counter() -> None:
    # `use_case.paid_calls` i dotychczasowe testy czytają goły Counter.
    telemetry = CycleTelemetry()
    live = telemetry.paid_calls

    telemetry.record_paid_call("sentiment")

    assert isinstance(live, Counter)
    assert live["sentiment"] == 1
    assert telemetry.paid_calls is live


def test_bind_paid_calls_adopts_a_caller_counter_and_keeps_counts() -> None:
    telemetry = CycleTelemetry()
    telemetry.record_paid_call("llm")
    external: Counter[str] = Counter({"news": 2})

    telemetry.bind_paid_calls(external)
    telemetry.record_paid_call("llm")

    assert telemetry.paid_calls is external
    assert external == Counter({"llm": 2, "news": 2})


def test_concurrent_records_lose_no_increment() -> None:
    # Rada woła persony w wątkach, a symbole mogą iść równolegle
    # (`symbol_concurrency`) — `Counter[k] += 1` bez locka gubi inkrementy.
    telemetry = CycleTelemetry()
    workers, per_worker = 8, 2_000
    barrier = threading.Barrier(workers)

    def hammer() -> None:
        barrier.wait()
        for _ in range(per_worker):
            telemetry.record_paid_call("council_llm")
            telemetry.record_node_timing("council", "NVDA", 1.0)

    threads = [threading.Thread(target=hammer) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    metrics = telemetry.snapshot()
    assert metrics.paid_calls["council_llm"] == workers * per_worker
