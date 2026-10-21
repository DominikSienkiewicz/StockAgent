"""NIEZ-24: `HttpCallMeter` — wątkowo bezpieczny licznik żądań HTTP per host.

Symbole idą równolegle (`symbol_concurrency`), a rada woła persony w wątkach,
więc licznik bez locka gubiłby inkrementy. Snapshot trafia do `CycleTelemetry`
i do step summary cyklu.
"""

from __future__ import annotations

import threading

from src.application.cycle_telemetry import CycleTelemetry
from src.application.http_metering import HttpCallMeter, status_class


def test_status_classes_keep_rate_limit_apart_from_other_4xx() -> None:
    assert status_class(200) == "2xx"
    assert status_class(304) == "3xx"
    assert status_class(404) == "4xx"
    assert status_class(429) == "429"
    assert status_class(503) == "5xx"
    assert status_class(99) == "other"


def test_snapshot_omits_zero_counters() -> None:
    meter = HttpCallMeter()

    meter.record("finnhub.io", 200)
    meter.record("finnhub.io", 404, retries=1, elapsed_s=0.25)

    assert meter.snapshot() == {"finnhub.io": {"requests": 2, "2xx": 1, "4xx": 1, "retries": 1}}
    assert meter.elapsed_seconds() == {"finnhub.io": 0.25}


def test_snapshot_is_a_copy() -> None:
    meter = HttpCallMeter()
    meter.record("finnhub.io", 200)

    meter.snapshot()["finnhub.io"]["requests"] = 99

    assert meter.snapshot()["finnhub.io"]["requests"] == 1


def test_concurrent_records_are_not_lost() -> None:
    meter = HttpCallMeter()

    def hammer() -> None:
        for _ in range(500):
            meter.record("finnhub.io", 200)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert meter.snapshot()["finnhub.io"]["requests"] == 4000


def test_cycle_telemetry_snapshot_carries_bound_http_calls() -> None:
    telemetry = CycleTelemetry()
    meter = HttpCallMeter()
    telemetry.bind_http_meter(meter)

    meter.record("www.alphavantage.co", 429)

    assert dict(telemetry.snapshot().http_calls) == {
        "www.alphavantage.co": {"requests": 1, "429": 1}
    }


def test_cycle_telemetry_without_meter_has_no_http_calls() -> None:
    assert dict(CycleTelemetry().snapshot().http_calls) == {}
