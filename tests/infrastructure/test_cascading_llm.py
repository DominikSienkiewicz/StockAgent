"""Testy CascadingLLMAdapter — confidence-gated model routing (Feature #2).

Reguła: analizę domyślnie robi tani LLM; eskalujemy do frontier dopiero,
gdy wynik taniego jest mało pewny (`confidence_score < routing_floor`) i
mamy jeszcze budżet eskalacji. Mockujemy dwa `Mock(spec=LLMPort)`, żeby
testować WYŁĄCZNIE logikę routingu (bez sieci, bez konkretnych adapterów).
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from unittest.mock import Mock

import pytest

from src.application.ports import LLM_ESCALATION_FAILED_KEY, LLM_ROUTED_MODEL_KEY, LLMPort
from src.infrastructure.llm.cascading import CascadingLLMAdapter


def _result(confidence: Any, tag: str) -> dict[str, Any]:
    """Wynik analyze() z osadzonym `confidence_score` i znacznikiem źródła,
    by test mógł odróżnić odpowiedź taniego od frontier."""
    return {"confidence_score": confidence, "_source": tag}


@pytest.fixture
def cheap() -> Mock:
    return Mock(spec=LLMPort)


@pytest.fixture
def frontier() -> Mock:
    return Mock(spec=LLMPort)


class TestPortContract:
    def test_is_llm_port(self, cheap: Mock, frontier: Mock) -> None:
        adapter = CascadingLLMAdapter(cheap=cheap, frontier=frontier)
        assert isinstance(adapter, LLMPort)


class TestNoEscalation:
    def test_high_confidence_returns_cheap_without_calling_frontier(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        cheap.analyze.return_value = _result(0.9, "cheap")
        adapter = CascadingLLMAdapter(
            cheap=cheap, frontier=frontier, routing_floor=0.6
        )

        result = adapter.analyze("prompt")

        assert result == _result(0.9, "cheap")
        cheap.analyze.assert_called_once_with("prompt")
        frontier.analyze.assert_not_called()

    def test_confidence_equal_to_floor_does_not_escalate(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        # Próg jest WŁĄCZAJĄCY: confidence == floor liczymy jako wystarczające.
        cheap.analyze.return_value = _result(0.6, "cheap")
        adapter = CascadingLLMAdapter(
            cheap=cheap, frontier=frontier, routing_floor=0.6
        )

        result = adapter.analyze("p")

        assert result["_source"] == "cheap"
        frontier.analyze.assert_not_called()


class TestEscalation:
    def test_low_confidence_escalates_and_returns_frontier_result(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        cheap.analyze.return_value = _result(0.3, "cheap")
        frontier.analyze.return_value = _result(0.95, "frontier")
        adapter = CascadingLLMAdapter(
            cheap=cheap, frontier=frontier, routing_floor=0.6
        )

        result = adapter.analyze("prompt")

        assert result == _result(0.95, "frontier")
        cheap.analyze.assert_called_once_with("prompt")
        frontier.analyze.assert_called_once_with("prompt")

    def test_escalation_is_logged(
        self, cheap: Mock, frontier: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        cheap.analyze.return_value = _result(0.1, "cheap")
        frontier.analyze.return_value = _result(0.9, "frontier")
        adapter = CascadingLLMAdapter(cheap=cheap, frontier=frontier)

        with caplog.at_level(logging.INFO):
            adapter.analyze("p")

        assert any(
            "escalat" in record.message.lower() for record in caplog.records
        )


class TestMissingOrInvalidConfidence:
    def test_missing_confidence_treated_as_low_and_escalates(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        cheap.analyze.return_value = {"_source": "cheap"}  # brak confidence_score
        frontier.analyze.return_value = _result(0.9, "frontier")
        adapter = CascadingLLMAdapter(cheap=cheap, frontier=frontier)

        result = adapter.analyze("p")

        assert result["_source"] == "frontier"
        frontier.analyze.assert_called_once()

    def test_non_numeric_confidence_treated_as_low_and_escalates(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        cheap.analyze.return_value = _result("oops", "cheap")
        frontier.analyze.return_value = _result(0.9, "frontier")
        adapter = CascadingLLMAdapter(cheap=cheap, frontier=frontier)

        result = adapter.analyze("p")

        assert result["_source"] == "frontier"
        frontier.analyze.assert_called_once()

    def test_none_confidence_treated_as_low_and_escalates(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        cheap.analyze.return_value = _result(None, "cheap")
        frontier.analyze.return_value = _result(0.9, "frontier")
        adapter = CascadingLLMAdapter(cheap=cheap, frontier=frontier)

        result = adapter.analyze("p")

        assert result["_source"] == "frontier"
        frontier.analyze.assert_called_once()


class TestEscalationBudget:
    def test_budget_caps_number_of_escalations(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        # Wszystkie wyniki taniego są mało pewne → każde wywołanie CHCE eskalować,
        # ale budżet = 2 pozwala tylko na 2 realne eskalacje.
        cheap.analyze.return_value = _result(0.1, "cheap")
        frontier.analyze.return_value = _result(0.9, "frontier")
        adapter = CascadingLLMAdapter(
            cheap=cheap, frontier=frontier, max_escalations=2
        )

        # 2 pierwsze wywołania eskalują (zwracają frontier).
        assert adapter.analyze("p1")["_source"] == "frontier"
        assert adapter.analyze("p2")["_source"] == "frontier"
        # 3. i 4. — budżet wyczerpany → zwraca tani wynik, frontier NIE wołany.
        assert adapter.analyze("p3")["_source"] == "cheap"
        assert adapter.analyze("p4")["_source"] == "cheap"

        assert frontier.analyze.call_count == 2
        assert cheap.analyze.call_count == 4

    def test_zero_budget_never_escalates(self, cheap: Mock, frontier: Mock) -> None:
        cheap.analyze.return_value = _result(0.0, "cheap")
        frontier.analyze.return_value = _result(0.9, "frontier")
        adapter = CascadingLLMAdapter(
            cheap=cheap, frontier=frontier, max_escalations=0
        )

        result = adapter.analyze("p")

        assert result["_source"] == "cheap"
        frontier.analyze.assert_not_called()

    def test_budget_exhaustion_logged_once(
        self, cheap: Mock, frontier: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        cheap.analyze.return_value = _result(0.0, "cheap")
        frontier.analyze.return_value = _result(0.9, "frontier")
        adapter = CascadingLLMAdapter(
            cheap=cheap, frontier=frontier, max_escalations=1
        )

        with caplog.at_level(logging.WARNING):
            adapter.analyze("p1")  # eskaluje (ostatnia z budżetu)
            adapter.analyze("p2")  # chce eskalować, budżet pusty → log raz
            adapter.analyze("p3")  # chce eskalować, budżet pusty → BEZ kolejnego logu

        exhaustion_logs = [
            r for r in caplog.records if "budget" in r.message.lower()
            or "exhaust" in r.message.lower()
        ]
        assert len(exhaustion_logs) == 1

    def test_high_confidence_does_not_consume_budget(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        # Pewny wynik nie zużywa budżetu — kolejne niepewne wciąż mogą eskalować.
        adapter = CascadingLLMAdapter(
            cheap=cheap, frontier=frontier, max_escalations=1
        )
        frontier.analyze.return_value = _result(0.9, "frontier")

        cheap.analyze.return_value = _result(0.9, "cheap")
        assert adapter.analyze("p1")["_source"] == "cheap"  # bez eskalacji

        cheap.analyze.return_value = _result(0.1, "cheap")
        assert adapter.analyze("p2")["_source"] == "frontier"  # budżet wciąż jest

        frontier.analyze.assert_called_once()


class TestAnalyzeMistake:
    def test_delegates_to_cheap_only(self, cheap: Mock, frontier: Mock) -> None:
        cheap.analyze_mistake.return_value = "diagnoza"
        adapter = CascadingLLMAdapter(cheap=cheap, frontier=frontier)

        result = adapter.analyze_mistake("why wrong?")

        assert result == "diagnoza"
        cheap.analyze_mistake.assert_called_once_with("why wrong?")
        frontier.analyze_mistake.assert_not_called()


class TestFrontierFailure:
    """NIEZ-17: opłacony wynik taniego modelu nie przepada, gdy frontier padnie —
    adapter zwraca go z WARNING-iem i alertem, żeby niedziałający routing nie
    schował się za cichym fallbackiem."""

    def test_frontier_failure_falls_back_to_cheap_result(
        self, cheap: Mock, frontier: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity

        cheap.analyze.return_value = _result(0.2, "cheap")
        frontier.analyze.side_effect = RuntimeError("frontier 400: model not found")
        monitor = QuotaMonitor()
        adapter = CascadingLLMAdapter(
            cheap=cheap, frontier=frontier, quota_monitor=monitor
        )

        with caplog.at_level(logging.WARNING):
            result = adapter.analyze("p")

        assert result == {**_result(0.2, "cheap"), LLM_ESCALATION_FAILED_KEY: True}
        frontier.analyze.assert_called_once_with("p")
        assert any("RuntimeError" in r.getMessage() for r in caplog.records)
        assert [alert.severity for alert in monitor.alerts] == [QuotaSeverity.WARNING]

    def test_paid_cheap_answer_after_a_failed_escalation_is_marked(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        # FIN-21 (G50): tani model zwraca confidence 0.4, frontier rzuca wyjątek →
        # odpowiedź taniego ze znacznikiem `escalation_failed`, żeby predykcja niosła
        # flagę jakości zamiast udawać zwykłą odpowiedź taniego modelu.
        cheap.analyze.return_value = _result(0.4, "cheap")
        frontier.analyze.side_effect = RuntimeError("frontier 500")
        adapter = CascadingLLMAdapter(cheap=cheap, frontier=frontier)

        result = adapter.analyze("p")

        assert result.get(LLM_ESCALATION_FAILED_KEY) is True
        assert result["_source"] == "cheap"

    def test_answers_without_a_failed_escalation_carry_no_marker(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        cheap.analyze.return_value = _result(0.1, "cheap")
        frontier.analyze.return_value = _result(0.9, "frontier")
        adapter = CascadingLLMAdapter(cheap=cheap, frontier=frontier, max_escalations=1)

        escalated, capped = adapter.analyze("p1"), adapter.analyze("p2")

        assert LLM_ESCALATION_FAILED_KEY not in escalated
        assert LLM_ESCALATION_FAILED_KEY not in capped

    def test_repeated_frontier_failures_alert_once_per_instance(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        from src.application.quota_monitor import QuotaMonitor

        cheap.analyze.return_value = _result(0.2, "cheap")
        frontier.analyze.side_effect = RuntimeError("down")
        monitor = QuotaMonitor()
        adapter = CascadingLLMAdapter(
            cheap=cheap, frontier=frontier, max_escalations=3, quota_monitor=monitor
        )

        for prompt in ("p1", "p2", "p3"):
            assert adapter.analyze(prompt)["_source"] == "cheap"

        assert frontier.analyze.call_count == 3
        assert len(monitor.alerts) == 1

    def test_frontier_failure_without_monitor_still_returns_cheap(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        cheap.analyze.return_value = _result(0.2, "cheap")
        frontier.analyze.side_effect = TimeoutError("frontier timeout")
        adapter = CascadingLLMAdapter(cheap=cheap, frontier=frontier)

        assert adapter.analyze("p")["_source"] == "cheap"


class TestRoutedModelMarker:
    """FIN-21: wynik kaskady niesie model, który faktycznie odpowiedział
    (`LLM_ROUTED_MODEL_KEY`), żeby kwit predykcji pozwolił porównać trafność per
    model. Bez tożsamości modeli (`cheap_model` / `frontier_model`) adapter nie ma
    czego wpisać i zwraca wynik bez znacznika."""

    CHEAP = "openai:gpt-5-mini"
    FRONTIER = "anthropic:claude-sonnet-4-6"

    def _adapter(self, cheap: Mock, frontier: Mock, **kwargs: Any) -> CascadingLLMAdapter:
        return CascadingLLMAdapter(
            cheap=cheap,
            frontier=frontier,
            cheap_model=self.CHEAP,
            frontier_model=self.FRONTIER,
            **kwargs,
        )

    def test_confident_cheap_answer_is_marked_with_the_cheap_model(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        cheap.analyze.return_value = _result(0.9, "cheap")

        result = self._adapter(cheap, frontier).analyze("p")

        assert result.get(LLM_ROUTED_MODEL_KEY) == self.CHEAP
        assert result["_source"] == "cheap"

    def test_escalated_answer_is_marked_with_the_frontier_model(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        cheap.analyze.return_value = _result(0.3, "cheap")
        frontier.analyze.return_value = _result(0.9, "frontier")

        result = self._adapter(cheap, frontier).analyze("p")

        assert result.get(LLM_ROUTED_MODEL_KEY) == self.FRONTIER
        assert result["_source"] == "frontier"

    def test_cheap_answer_after_exhausted_budget_is_marked_with_the_cheap_model(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        cheap.analyze.return_value = _result(0.1, "cheap")

        result = self._adapter(cheap, frontier, max_escalations=0).analyze("p")

        assert result.get(LLM_ROUTED_MODEL_KEY) == self.CHEAP
        frontier.analyze.assert_not_called()

    def test_marker_does_not_mutate_the_answer_of_the_inner_port(
        self, cheap: Mock, frontier: Mock
    ) -> None:
        answer = _result(0.9, "cheap")
        cheap.analyze.return_value = answer

        self._adapter(cheap, frontier).analyze("p")

        assert answer == _result(0.9, "cheap")


class _RendezvousCap(int):
    """Sufit eskalacji, który zatrzymuje każde porównanie z licznikiem, aż do niego
    dojdą wszystkie wątki (albo minie `patience`).

    Na CPython 3.12 z GIL między porównaniem licznika a jego inkrementacją nie ma
    punktu przełączenia wątku, więc wyścig nie odtwarza się sam. Ten int wymusza
    przeplot: bez locka wszystkie wątki porównują stary licznik i każdy eskaluje;
    z lockiem porównania idą po kolei, a każde czeka najwyżej `patience`."""

    _parties: int
    _patience: float
    _arrived: int
    _guard: threading.Lock
    _everyone_here: threading.Event

    def __new__(cls, value: int, *, parties: int, patience: float) -> _RendezvousCap:
        cap = super().__new__(cls, value)
        cap._parties = parties
        cap._patience = patience
        cap._arrived = 0
        cap._guard = threading.Lock()
        cap._everyone_here = threading.Event()
        return cap

    def __le__(self, other: object) -> bool:  # `used >= cap` woła odbicie `cap <= used`
        with self._guard:
            self._arrived += 1
            if self._arrived >= self._parties:
                self._everyone_here.set()
        self._everyone_here.wait(self._patience)
        return int(self) <= other  # type: ignore[operator]


class TestEscalationConcurrency:
    def test_escalation_cap_holds_under_concurrency(self, cheap: Mock, frontier: Mock) -> None:
        from concurrent.futures import ThreadPoolExecutor

        threads = 4
        cheap.analyze.return_value = _result(0.1, "cheap")
        frontier.analyze.return_value = _result(0.9, "frontier")
        adapter = CascadingLLMAdapter(
            cheap=cheap,
            frontier=frontier,
            max_escalations=_RendezvousCap(1, parties=threads, patience=0.2),
        )

        with ThreadPoolExecutor(max_workers=threads) as pool:
            sources = list(pool.map(lambda p: adapter.analyze(p)["_source"], ["p"] * threads))

        assert frontier.analyze.call_count == 1
        assert sorted(sources) == ["cheap", "cheap", "cheap", "frontier"]
