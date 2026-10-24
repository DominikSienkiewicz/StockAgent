"""MAIL-21: werdykt stanu każdego dostawcy z sygnałów, które cykl już ma."""

from __future__ import annotations

from decimal import Decimal

from src.application.ports import CircuitState
from src.application.report_builder import SymbolResult, to_symbol_result
from src.application.source_health import (
    SOURCE_ORDER,
    SourceHealth,
    SourceStatus,
    assess_source_health,
    degraded_symbols,
)

ALL_ON = {source: True for source in SOURCE_ORDER}


def _circuit(source: str, *, is_open: bool, skipped: frozenset[str] = frozenset()) -> CircuitState:
    return CircuitState(
        source=source,
        host=f"{source.lower()}.example",
        is_open=is_open,
        consecutive_failures=5 if is_open else 0,
        skipped_calls=len(skipped),
        skipped_symbols=skipped,
    )


def _by_source(rows: tuple[SourceHealth, ...]) -> dict[str, SourceHealth]:
    return {row.source: row for row in rows}


def test_quiet_cycle_reports_every_source_ok_in_mail_order() -> None:
    rows = assess_source_health(enabled=ALL_ON, circuits=(), degraded={}, failed_requests={})

    assert tuple(row.source for row in rows) == SOURCE_ORDER
    assert {row.status for row in rows} == {SourceStatus.OK}


def test_open_circuit_makes_the_source_unavailable_with_skipped_symbols() -> None:
    rows = assess_source_health(
        enabled=ALL_ON,
        circuits=(
            _circuit("Finnhub", is_open=True, skipped=frozenset({"AAPL", "MSFT"})),
            _circuit("NBP", is_open=False),
        ),
        degraded={},
        failed_requests={"Finnhub": 5},
    )

    finnhub = _by_source(rows)["Finnhub"]
    assert finnhub.status is SourceStatus.UNAVAILABLE
    assert finnhub.affected == 2
    assert _by_source(rows)["NBP"].status is SourceStatus.OK


def test_degraded_symbols_and_failed_requests_degrade_the_source() -> None:
    rows = _by_source(
        assess_source_health(
            enabled=ALL_ON,
            circuits=(),
            degraded={"Alpha Vantage": frozenset({"AAPL", "NVDA", "AMD"})},
            failed_requests={"EDGAR": 4},
        )
    )

    assert rows["Alpha Vantage"] == SourceHealth("Alpha Vantage", SourceStatus.DEGRADED, affected=3)
    assert rows["EDGAR"] == SourceHealth("EDGAR", SourceStatus.DEGRADED, failed_requests=4)


def test_source_disabled_by_flag_is_reported_as_disabled() -> None:
    enabled = {**ALL_ON, "Reddit": False}

    rows = _by_source(
        assess_source_health(
            enabled=enabled, circuits=(), degraded={}, failed_requests={"Reddit": 1}
        )
    )

    assert rows["Reddit"].status is SourceStatus.DISABLED


def test_quality_flags_map_degradation_to_providers() -> None:
    results = [
        SymbolResult(symbol="AAPL", status="saved", data_quality_flags=("av_keys_exhausted",)),
        SymbolResult(symbol="NVDA", status="saved", data_quality_flags=("llm_analysis_failed",)),
        SymbolResult(symbol="AMD", status="saved", data_quality_flags=("stale_quote",)),
    ]

    assert degraded_symbols(results, llm_source="Anthropic") == {
        "Alpha Vantage": frozenset({"AAPL"}),
        "Anthropic": frozenset({"NVDA"}),
    }


def test_mapper_carries_quality_flags_of_the_graph_state() -> None:
    raw = {
        "status": "saved",
        "delta": Decimal("0.03"),
        "data_quality_flags": ["av_fetch_error", "llm_failover"],
    }

    assert to_symbol_result("AAPL", raw).data_quality_flags == ("av_fetch_error", "llm_failover")
