"""NIEZ-25: sonda modeli w Slow Loopie — które modele, jakie katalogi, push.

Lokalnie (bez `PAID_CALLS_ENABLED`) katalogi są atrapą bez requestów; w GHA
pytamy Models API Anthropic i OpenAI (bez tokenów).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

import main_trainer
from src.application.ports import ModelCatalogPort, ReportNotifierPort
from src.application.quota_monitor import QuotaMonitor
from src.application.use_cases.probe_models import ModelProbe
from src.config import Settings
from src.domain.quota import QuotaSeverity
from src.infrastructure.llm.model_catalog import (
    AnthropicModelCatalog,
    LocalStubModelCatalog,
    OpenAIModelCatalog,
)

NOW = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test",
        anthropic_api_key="sk-ant-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="sb_secret_test",
        llm_provider="anthropic",
        council_llm_provider="openai",
        council_llm_model="gpt-5-mini",
        model_probe_enabled=True,
        paid_calls_enabled=True,
    )


def test_production_like_config_probes_main_council_and_embeddings(settings: Settings) -> None:
    assert main_trainer._probed_models(settings) == [
        ModelProbe("anthropic", "claude-sonnet-4-6", "główny LLM"),
        ModelProbe("openai", "gpt-5-mini", "rada"),
        ModelProbe("openai", "text-embedding-3-small", "embeddingi"),
    ]


def test_routing_and_fallback_models_are_probed(settings: Settings) -> None:
    settings.model_routing_enabled = True
    settings.council_llm_provider = None
    settings.council_llm_model = None
    settings.llm_fallback_provider = "openai"

    probes = main_trainer._probed_models(settings)

    assert ModelProbe("openai", "gpt-5-mini", "router (tani)") in probes
    assert ModelProbe("anthropic", "claude-sonnet-4-6", "router (frontier)") in probes
    assert ModelProbe("openai", "gpt-4o", "zapasowy LLM") in probes
    assert all(probe.role != "główny LLM" for probe in probes)


def test_local_run_uses_stub_catalogs(settings: Settings) -> None:
    settings.paid_calls_enabled = False

    catalogs = main_trainer._model_catalogs(settings)

    assert all(isinstance(c, LocalStubModelCatalog) for c in catalogs.values())


def test_production_run_builds_real_catalogs_for_keys_present(
    settings: Settings, mocker: Any
) -> None:
    mocker.patch("src.infrastructure.llm.model_catalog.Anthropic")
    mocker.patch("src.infrastructure.llm.model_catalog.OpenAI")
    settings.anthropic_api_key = None

    catalogs = main_trainer._model_catalogs(settings)

    assert set(catalogs) == {"openai"}
    assert isinstance(catalogs["openai"], OpenAIModelCatalog)
    settings.anthropic_api_key = "sk-ant-test"
    assert isinstance(main_trainer._model_catalogs(settings)["anthropic"], AnthropicModelCatalog)


def _missing(model_id: str) -> Mock:
    catalog = Mock(spec=ModelCatalogPort)
    catalog.is_available.side_effect = lambda probed: probed != model_id
    return catalog


def test_missing_model_is_recorded_as_critical(settings: Settings) -> None:
    monitor = QuotaMonitor()
    catalogs: dict[str, ModelCatalogPort] = {
        "anthropic": _missing("claude-sonnet-4-6"),
        "openai": _missing("none"),
    }

    main_trainer._run_model_probe(settings, monitor, now=NOW, catalogs=catalogs)

    (alert,) = monitor.alerts
    assert alert.severity is QuotaSeverity.CRITICAL
    assert alert.source == "Anthropic"


def test_flag_off_skips_the_probe(settings: Settings) -> None:
    settings.model_probe_enabled = False
    catalog = _missing("claude-sonnet-4-6")

    main_trainer._run_model_probe(
        settings, QuotaMonitor(), now=NOW, catalogs={"anthropic": catalog}
    )

    catalog.is_available.assert_not_called()


def test_critical_is_pushed_when_realtime_alerts_are_on(settings: Settings, mocker: Any) -> None:
    settings.realtime_alerts_enabled = True
    channel = MagicMock(spec=ReportNotifierPort)
    mocker.patch("main_trainer.build_push_channels", return_value=[channel])

    main_trainer._run_model_probe(
        settings,
        QuotaMonitor(),
        now=NOW,
        catalogs={"anthropic": _missing("claude-sonnet-4-6"), "openai": _missing("none")},
    )

    channel.send_report.assert_called_once()
    assert "claude-sonnet-4-6" in str(channel.send_report.call_args)


def test_no_push_while_realtime_alerts_are_off(settings: Settings, mocker: Any) -> None:
    settings.realtime_alerts_enabled = False
    channels = mocker.patch("main_trainer.build_push_channels")

    main_trainer._run_model_probe(
        settings,
        QuotaMonitor(),
        now=NOW,
        catalogs={"anthropic": _missing("claude-sonnet-4-6"), "openai": _missing("none")},
    )

    channels.assert_not_called()
