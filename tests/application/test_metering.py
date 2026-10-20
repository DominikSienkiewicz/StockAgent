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
