"""NIEZ-31: otwarty obwód hosta HTTP jako jeden WARNING w bannerze kwot.

Obwód otwiera się w infrastrukturze (`_http.CircuitBreakerRegistry`), ale liczbę
pominiętych symboli zna dopiero koniec cyklu — dlatego entry-point czyta stan przez
`CircuitStatePort` raz, po przetworzeniu symboli, a przed raportem.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from src.application.ports import CircuitState, CircuitStatePort
from src.application.quota_monitor import QuotaMonitor
from src.application.report_formatting import plural_pl
from src.domain.quota import QuotaAlert, QuotaSeverity


def open_circuit_alerts(states: Iterable[CircuitState], occurred_at: datetime) -> list[QuotaAlert]:
    """Jeden WARNING na każdy otwarty obwód, np. „Finnhub niedostępny — 43 symbole
    pominięte.”; zamknięte obwody nie dają alertu."""
    return [
        QuotaAlert(
            source=state.source,
            severity=QuotaSeverity.WARNING,
            message=f"{state.source} niedostępny — {_skipped_phrase(state)}.",
            action=(
                f"Obwód {state.host} otwarty po {state.consecutive_failures} kolejnych błędach "
                "(5xx, timeout, połączenie) — do końca przebiegu bez zapytań do tego hosta. "
                "Sprawdź stronę statusu dostawcy; następny przebieg spróbuje od nowa."
            ),
            occurred_at=occurred_at,
        )
        for state in states
        if state.is_open
    ]


def record_open_circuit_alerts(
    monitor: QuotaMonitor, circuits: CircuitStatePort, occurred_at: datetime
) -> None:
    """Zapisuje alerty otwartych obwodów procesu do `monitor` (banner i `quota_alerts`)."""
    for alert in open_circuit_alerts(circuits.circuit_states(), occurred_at):
        monitor.record(alert)


def _skipped_phrase(state: CircuitState) -> str:
    symbols = len(state.skipped_symbols)
    if symbols == 0 and state.skipped_calls > 0:
        calls = state.skipped_calls
        noun = plural_pl(calls, "zapytanie pominięte", "zapytania pominięte", "zapytań pominiętych")
        return f"{calls} {noun}"
    noun = plural_pl(symbols, "symbol pominięty", "symbole pominięte", "symboli pominiętych")
    return f"{symbols} {noun}"
