"""NIEZ-25: sonda dostępności modeli LLM przez Models API (Slow Loop).

`validate_model_id` sprawdza tylko prefiks, więc wycofany id kończy się cichym
total outage dopiero przy pierwszym wywołaniu. Sonda pyta Models API (bez
tokenów) o każdy skonfigurowany model; brak = CRITICAL, błąd sondy = WARNING.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock

from src.application.ports import ModelCatalogPort
from src.application.use_cases.probe_models import (
    PROBE_ERROR_SOURCE,
    ModelProbe,
    ProbeModelsUseCase,
)
from src.domain.quota import QuotaSeverity

NOW = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)


def _catalog(missing: set[str] | None = None, error: Exception | None = None) -> Mock:
    catalog = Mock(spec=ModelCatalogPort)
    if error is not None:
        catalog.is_available.side_effect = error
    else:
        catalog.is_available.side_effect = lambda model_id: model_id not in (missing or set())
    return catalog


def test_missing_model_emits_critical_alert() -> None:
    anthropic = _catalog(missing={"claude-sonnet-4-6"})

    alerts = ProbeModelsUseCase({"anthropic": anthropic}).run(
        [ModelProbe("anthropic", "claude-sonnet-4-6", "główny LLM")], now=NOW
    )

    (alert,) = alerts
    assert alert.severity is QuotaSeverity.CRITICAL
    assert alert.source == "Anthropic"
    assert "claude-sonnet-4-6" in alert.message
    assert "główny LLM" in alert.message
    assert alert.occurred_at == NOW
    anthropic.is_available.assert_called_once_with("claude-sonnet-4-6")


def test_available_models_emit_nothing() -> None:
    alerts = ProbeModelsUseCase({"anthropic": _catalog(), "openai": _catalog()}).run(
        [
            ModelProbe("anthropic", "claude-sonnet-4-6", "główny LLM"),
            ModelProbe("openai", "gpt-5-mini", "rada"),
        ],
        now=NOW,
    )

    assert alerts == []


def test_probe_error_is_a_warning_not_a_missing_model() -> None:
    # Błąd sieci albo klucz bez dostępu do listy modeli to nie brak modelu.
    error = ConnectionError("connect to https://api.openai.com failed key=sk-secret")

    alerts = ProbeModelsUseCase({"openai": _catalog(error=error)}).run(
        [ModelProbe("openai", "gpt-5-mini", "rada")], now=NOW
    )

    (alert,) = alerts
    assert alert.severity is QuotaSeverity.WARNING
    assert alert.source == PROBE_ERROR_SOURCE
    assert "ConnectionError" in alert.message
    assert "sk-secret" not in alert.message


def test_each_model_is_checked_once() -> None:
    openai = _catalog(missing={"gpt-5-mini"})

    alerts = ProbeModelsUseCase({"openai": openai}).run(
        [
            ModelProbe("openai", "gpt-5-mini", "rada"),
            ModelProbe("openai", "gpt-5-mini", "router (tani)"),
        ],
        now=NOW,
    )

    openai.is_available.assert_called_once_with("gpt-5-mini")
    (alert,) = alerts
    assert alert.source == "OpenAI"
    assert "rada" in alert.message and "router (tani)" in alert.message


def test_provider_without_a_catalog_is_skipped() -> None:
    alerts = ProbeModelsUseCase({}).run(
        [ModelProbe("anthropic", "claude-sonnet-4-6", "główny LLM")], now=NOW
    )

    assert alerts == []
