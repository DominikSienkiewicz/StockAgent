"""NIEZ-31: otwarty obwód hosta to jeden WARNING w bannerze kwot na koniec cyklu."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock

from src.application.circuit_alerts import open_circuit_alerts, record_open_circuit_alerts
from src.application.ports import CircuitState, CircuitStatePort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaSeverity

NOW = datetime(2026, 10, 24, 6, 0, tzinfo=UTC)


def _state(
    *, is_open: bool = True, symbols: int = 0, calls: int | None = None, source: str = "Finnhub"
) -> CircuitState:
    skipped = frozenset(f"S{i}" for i in range(symbols))
    return CircuitState(
        source=source,
        host="finnhub.io",
        is_open=is_open,
        consecutive_failures=5,
        skipped_calls=len(skipped) if calls is None else calls,
        skipped_symbols=skipped,
    )


def test_open_circuit_is_one_warning_with_skipped_symbols() -> None:
    [alert] = open_circuit_alerts([_state(symbols=43)], NOW)

    assert alert.source == "Finnhub"
    assert alert.severity is QuotaSeverity.WARNING
    assert alert.message == "Finnhub niedostępny — 43 symbole pominięte."
    assert "finnhub.io" in alert.action
    assert alert.occurred_at == NOW


def test_symbol_count_uses_polish_plural_forms() -> None:
    messages = [open_circuit_alerts([_state(symbols=n)], NOW)[0].message for n in (1, 5)]

    assert messages == [
        "Finnhub niedostępny — 1 symbol pominięty.",
        "Finnhub niedostępny — 5 symboli pominiętych.",
    ]


def test_calls_outside_a_symbol_are_counted_as_requests() -> None:
    [alert] = open_circuit_alerts([_state(symbols=0, calls=2, source="FRED")], NOW)

    assert alert.message == "FRED niedostępny — 2 zapytania pominięte."


def test_closed_circuits_raise_no_alert() -> None:
    assert open_circuit_alerts([_state(is_open=False)], NOW) == []


def test_alerts_are_recorded_in_the_quota_monitor() -> None:
    circuits = Mock(spec=CircuitStatePort)
    circuits.circuit_states.return_value = (_state(symbols=3), _state(is_open=False))
    monitor = QuotaMonitor()

    record_open_circuit_alerts(monitor, circuits, NOW)

    assert [alert.message for alert in monitor.alerts] == [
        "Finnhub niedostępny — 3 symbole pominięte."
    ]
