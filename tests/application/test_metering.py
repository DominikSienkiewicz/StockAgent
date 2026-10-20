"""FIN-02: `MeteredLLMPort` — dekorator portu LLM, który liczy KAŻDY realny
request w telemetrii cyklu, zanim odda go dostawcy (także request, który
potem padnie albo nie zdąży przed limitem czasu rady)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.application.cycle_telemetry import CycleTelemetry
from src.application.metering import MeteredLLMPort
from src.application.ports import LLMPort


@pytest.fixture
def inner() -> Mock:
    port = Mock(spec=LLMPort)
    port.analyze.return_value = {"recommendation": "BUY"}
    port.analyze_mistake.return_value = "lekcja"
    return port


def test_is_an_llm_port(inner: Mock) -> None:
    assert isinstance(MeteredLLMPort(inner, CycleTelemetry(), "council_llm"), LLMPort)


def test_analyze_counts_and_delegates(inner: Mock) -> None:
    telemetry = CycleTelemetry()
    port = MeteredLLMPort(inner, telemetry, "council_llm")

    assert port.analyze("prompt") == {"recommendation": "BUY"}
    port.analyze("prompt 2")

    inner.analyze.assert_called_with("prompt 2")
    assert telemetry.snapshot().paid_calls["council_llm"] == 2


def test_analyze_mistake_counts_and_delegates(inner: Mock) -> None:
    telemetry = CycleTelemetry()
    port = MeteredLLMPort(inner, telemetry, "llm")

    assert port.analyze_mistake("diagnoza") == "lekcja"
    assert telemetry.snapshot().paid_calls["llm"] == 1


def test_failed_request_is_counted_and_the_error_propagates(inner: Mock) -> None:
    inner.analyze.side_effect = RuntimeError("429")
    telemetry = CycleTelemetry()
    port = MeteredLLMPort(inner, telemetry, "council_llm")

    with pytest.raises(RuntimeError, match="429"):
        port.analyze("prompt")

    assert telemetry.snapshot().paid_calls["council_llm"] == 1


class TestBudgetedLLMPort:
    """FIN-30: przycięcie budżetem Slow Loopa działa PRZED wywołaniem dostawcy —
    wywołanie ponad przydział rzuca, więc use case idzie deterministycznym
    fallbackiem (recap bez narracji, kalibracja bez wniosku)."""

    def test_is_an_llm_port(self, inner: Mock) -> None:
        from src.application.metering import BudgetedLLMPort

        assert isinstance(BudgetedLLMPort(inner, allowed=1, job="recap"), LLMPort)

    def test_delegates_up_to_the_allowance(self, inner: Mock) -> None:
        from src.application.metering import BudgetedLLMPort

        port = BudgetedLLMPort(inner, allowed=2, job="backfill")

        assert port.analyze("a") == {"recommendation": "BUY"}
        assert port.analyze_mistake("b") == "lekcja"
        assert inner.analyze.call_count + inner.analyze_mistake.call_count == 2

    def test_call_past_the_allowance_never_reaches_the_provider(
        self, inner: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from src.application.metering import BudgetedLLMPort, PaidBudgetExhausted

        port = BudgetedLLMPort(inner, allowed=0, job="judge")

        with caplog.at_level(logging.WARNING, logger="src.application.metering"):
            with pytest.raises(PaidBudgetExhausted, match="judge"):
                port.analyze_mistake("prompt")
            with pytest.raises(PaidBudgetExhausted):
                port.analyze("prompt")

        inner.analyze_mistake.assert_not_called()
        inner.analyze.assert_not_called()
        assert "judge" in caplog.text

    def test_refused_call_is_not_metered(self, inner: Mock) -> None:
        from src.application.metering import BudgetedLLMPort, PaidBudgetExhausted

        telemetry = CycleTelemetry()
        port = BudgetedLLMPort(MeteredLLMPort(inner, telemetry, "llm"), allowed=1, job="recap")

        port.analyze("pierwsze")
        with pytest.raises(PaidBudgetExhausted):
            port.analyze("drugie")

        assert telemetry.snapshot().paid_calls["llm"] == 1


class TestUsageAttribution:
    """FIN-13: dekorator zna klucz cennika, więc zgłoszenie tokenów z adaptera
    pod nim trafia do ledgera z tym źródłem (rada w wątkach person, Slow Loop
    bez grafu), z węzłem i symbolem z zakresu wołającego."""

    def test_usage_recorded_inside_the_call_carries_the_source(self) -> None:
        from src.application.usage_ledger import UsageLedger, usage_scope
        from src.domain.finops import LLMUsage

        ledger = UsageLedger()
        usage = LLMUsage(provider="openai", model="gpt-5-mini", input_tokens=5, output_tokens=5)
        inner = Mock(spec=LLMPort)
        inner.analyze.side_effect = lambda _prompt: ledger.record(usage) or {}
        inner.analyze_mistake.side_effect = lambda _prompt: ledger.record(usage) or ""
        port = MeteredLLMPort(inner, CycleTelemetry(), "council_llm")

        with usage_scope(node="council", symbol="NVDA"):
            port.analyze("persona")
        port.analyze_mistake("diagnoza")

        sources = [(r.source, r.node, r.symbol) for r in ledger.records()]
        assert sources == [
            ("council_llm", "council", "NVDA"),
            ("council_llm", "unattributed", None),
        ]
