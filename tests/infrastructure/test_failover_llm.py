"""NIEZ-44: FailoverLLMAdapter — drugi dostawca LLM po wyczerpaniu retry głównego.

Kompozyt dwóch `LLMPort` (wzorem `CascadingLLMAdapter`). Zapasowy jest wołany
WYŁĄCZNIE, gdy główny rzucił — po swoich własnych retry. Błąd klienta 4xx inny
niż 429 (zły prompt, zły klucz) nie przełącza: drugi dostawca dostałby ten sam
zły request. Wynik od zapasowego niesie jego dostawcę i model (klucze
`LLM_FAILOVER_PROVIDER_KEY` / `LLM_FAILOVER_MODEL_KEY`), żeby graf dopisał flagę
jakości `llm_failover` i faktyczny przepis predykcji.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from src.application.ports import (
    LLM_FAILOVER_MODEL_KEY,
    LLM_FAILOVER_PROVIDER_KEY,
    LLMPort,
)
from src.infrastructure.llm.failover import FailoverLLMAdapter


class _StatusError(Exception):
    """Kształt błędu HTTP obu SDK (`APIStatusError.status_code`)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.fixture
def primary() -> Mock:
    return Mock(spec=LLMPort)


@pytest.fixture
def secondary() -> Mock:
    port = Mock(spec=LLMPort)
    port.analyze.return_value = {"trend_direction": "BULLISH", "confidence_score": 0.7}
    port.analyze_mistake.return_value = "insight from the fallback"
    return port


@pytest.fixture
def adapter(primary: Mock, secondary: Mock) -> FailoverLLMAdapter:
    return FailoverLLMAdapter(primary, secondary, provider="openai", model="gpt-4o")


def test_is_an_llm_port(adapter: FailoverLLMAdapter) -> None:
    assert isinstance(adapter, LLMPort)


def test_secondary_used_only_after_primary_raises(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock
) -> None:
    primary.analyze.side_effect = RuntimeError("overloaded")

    result = adapter.analyze("prompt")

    secondary.analyze.assert_called_once_with("prompt")
    assert result[LLM_FAILOVER_PROVIDER_KEY] == "openai"
    assert result[LLM_FAILOVER_MODEL_KEY] == "gpt-4o"
    assert result["trend_direction"] == "BULLISH"


def test_primary_success_never_touches_the_secondary(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock
) -> None:
    primary.analyze.return_value = {"trend_direction": "BEARISH"}

    result = adapter.analyze("prompt")

    secondary.analyze.assert_not_called()
    assert result == {"trend_direction": "BEARISH"}


@pytest.mark.parametrize("status", [429, 500, 529])
def test_rate_limit_and_server_errors_fail_over(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock, status: int
) -> None:
    primary.analyze.side_effect = _StatusError(status)

    adapter.analyze("prompt")

    secondary.analyze.assert_called_once()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_other_than_429_do_not_fail_over(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock, status: int
) -> None:
    primary.analyze.side_effect = _StatusError(status)

    with pytest.raises(_StatusError):
        adapter.analyze("prompt")

    secondary.analyze.assert_not_called()


def test_secondary_failure_propagates(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock
) -> None:
    # Graf degraduje wtedy do neutralnego fallbacku z flagą `llm_analysis_failed`.
    primary.analyze.side_effect = RuntimeError("overloaded")
    secondary.analyze.side_effect = TimeoutError("fallback timed out")

    with pytest.raises(TimeoutError):
        adapter.analyze("prompt")


def test_mistake_diagnosis_fails_over_too(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock
) -> None:
    primary.analyze_mistake.side_effect = RuntimeError("overloaded")

    assert adapter.analyze_mistake("prompt") == "insight from the fallback"
    secondary.analyze_mistake.assert_called_once_with("prompt")


def test_mistake_diagnosis_success_stays_on_the_primary(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock
) -> None:
    primary.analyze_mistake.return_value = "primary insight"

    assert adapter.analyze_mistake("prompt") == "primary insight"
    secondary.analyze_mistake.assert_not_called()


def test_failover_is_logged_with_the_error_type_only(
    adapter: FailoverLLMAdapter, primary: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    primary.analyze.side_effect = RuntimeError("overloaded at https://api.example.test?key=x")

    with caplog.at_level(logging.WARNING, logger="src.infrastructure.llm.failover"):
        adapter.analyze("prompt")

    assert "RuntimeError" in caplog.text
    assert "openai" in caplog.text and "gpt-4o" in caplog.text
    assert "key=x" not in caplog.text
