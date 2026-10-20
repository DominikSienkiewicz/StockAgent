"""NIEZ-44: FailoverLLMAdapter — drugi dostawca LLM po wyczerpaniu retry głównego.

Kompozyt dwóch `LLMPort` (wzorem `CascadingLLMAdapter`). Zapasowy jest wołany
WYŁĄCZNIE przy awarii dostawcy głównego — 429, 5xx, timeout, błąd sieci albo
połączenia — po jego własnych retry (decyzja właściciela 2026-10-20). Pusta albo
zła odpowiedź JSON głównego nie przełącza: kończy się neutralnym fallbackiem grafu
jak dotąd. Błąd klienta 4xx inny niż 429 (zły prompt, zły klucz) też nie
przełącza. Wynik od zapasowego niesie jego dostawcę i model
(`LLM_FAILOVER_PROVIDER_KEY` / `LLM_FAILOVER_MODEL_KEY`), żeby graf dopisał flagę
jakości `llm_failover` i faktyczny przepis predykcji.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import anthropic
import httpx
import openai
import pytest

from src.application.ports import (
    LLM_FAILOVER_MODEL_KEY,
    LLM_FAILOVER_PROVIDER_KEY,
    LLMPort,
)
from src.infrastructure.llm.failover import FailoverLLMAdapter

_REQUEST = httpx.Request("POST", "https://api.example.test/v1/messages")


class _StatusError(Exception):
    """Kształt błędu HTTP obu SDK (`APIStatusError.status_code`)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _sdk_status_error(sdk, status: int) -> Exception:
    return sdk.InternalServerError(
        f"HTTP {status}", response=httpx.Response(status, request=_REQUEST), body=None
    )


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
    # 529 Overloaded — Anthropic po wyczerpaniu własnych retry.
    primary.analyze.side_effect = _StatusError(529)

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


@pytest.mark.parametrize("status", [429, 500, 503, 529])
def test_rate_limit_and_server_errors_fail_over(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock, status: int
) -> None:
    primary.analyze.side_effect = _StatusError(status)

    adapter.analyze("prompt")

    secondary.analyze.assert_called_once()


@pytest.mark.parametrize("sdk", [anthropic, openai], ids=["anthropic", "openai"])
def test_sdk_503_fails_over(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock, sdk
) -> None:
    primary.analyze.side_effect = _sdk_status_error(sdk, 503)

    adapter.analyze("prompt")

    secondary.analyze.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APITimeoutError(request=_REQUEST),
        openai.APITimeoutError(request=_REQUEST),
        TimeoutError("read timed out"),
    ],
    ids=["anthropic", "openai", "builtin"],
)
def test_timeouts_fail_over(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock, error: Exception
) -> None:
    primary.analyze.side_effect = error

    adapter.analyze("prompt")

    secondary.analyze.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APIConnectionError(request=_REQUEST),
        openai.APIConnectionError(request=_REQUEST),
        ConnectionResetError("connection reset by peer"),
    ],
    ids=["anthropic", "openai", "builtin"],
)
def test_connection_errors_fail_over(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock, error: Exception
) -> None:
    primary.analyze.side_effect = error

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


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Anthropic returned invalid JSON: 'I cannot answer that.'"),
        ValueError("OpenAI returned empty content for analyze() (finish_reason='stop')."),
    ],
    ids=["invalid_json", "empty_content"],
)
def test_bad_answer_does_not_fail_over(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock, error: Exception
) -> None:
    # Decyzja właściciela: dostawca odpowiedział — zła treść kończy się neutralnym
    # fallbackiem grafu, a nie drugim płatnym requestem.
    primary.analyze.side_effect = error

    with pytest.raises(ValueError):
        adapter.analyze("prompt")

    secondary.analyze.assert_not_called()


def test_unclassified_error_does_not_fail_over(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock
) -> None:
    primary.analyze.side_effect = RuntimeError("unexpected")

    with pytest.raises(RuntimeError):
        adapter.analyze("prompt")

    secondary.analyze.assert_not_called()


def test_secondary_failure_propagates(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock
) -> None:
    # Graf degraduje wtedy do neutralnego fallbacku z flagą `llm_analysis_failed`.
    primary.analyze.side_effect = _StatusError(503)
    secondary.analyze.side_effect = TimeoutError("fallback timed out")

    with pytest.raises(TimeoutError):
        adapter.analyze("prompt")


def test_mistake_diagnosis_fails_over_too(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock
) -> None:
    primary.analyze_mistake.side_effect = _StatusError(503)

    assert adapter.analyze_mistake("prompt") == "insight from the fallback"
    secondary.analyze_mistake.assert_called_once_with("prompt")


def test_mistake_diagnosis_with_a_bad_answer_stays_on_the_primary(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock
) -> None:
    primary.analyze_mistake.side_effect = ValueError("empty content")

    with pytest.raises(ValueError):
        adapter.analyze_mistake("prompt")

    secondary.analyze_mistake.assert_not_called()


def test_mistake_diagnosis_success_stays_on_the_primary(
    adapter: FailoverLLMAdapter, primary: Mock, secondary: Mock
) -> None:
    primary.analyze_mistake.return_value = "primary insight"

    assert adapter.analyze_mistake("prompt") == "primary insight"
    secondary.analyze_mistake.assert_not_called()


def test_failover_is_logged_with_the_error_type_only(
    adapter: FailoverLLMAdapter, primary: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    primary.analyze.side_effect = TimeoutError("timed out at https://api.example.test?key=x")

    with caplog.at_level(logging.WARNING, logger="src.infrastructure.llm.failover"):
        adapter.analyze("prompt")

    assert "TimeoutError" in caplog.text
    assert "openai" in caplog.text and "gpt-4o" in caplog.text
    assert "key=x" not in caplog.text


@pytest.mark.parametrize("method", ["analyze", "analyze_mistake"])
def test_fallback_refused_by_the_budget_reraises_the_primary_error(
    adapter: FailoverLLMAdapter,
    primary: Mock,
    secondary: Mock,
    method: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Slow Loop (decyzja 2026-10-20): zapasowy idzie przez przydział joba FIN-30.
    # Wyczerpany przydział = failover nie startuje, a błąd głównego leci jak dotąd.
    from src.application.metering import PaidBudgetExhausted

    getattr(primary, method).side_effect = _StatusError(503)
    getattr(secondary, method).side_effect = PaidBudgetExhausted("paid LLM budget exhausted")

    with (
        caplog.at_level(logging.WARNING, logger="src.infrastructure.llm.failover"),
        pytest.raises(_StatusError),
    ):
        getattr(adapter, method)("prompt")

    assert "budget" in caplog.text
