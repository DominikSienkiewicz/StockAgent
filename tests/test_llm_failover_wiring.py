"""NIEZ-44 (b): DI owija główny LLM w `FailoverLLMAdapter`, gdy jest zapasowy dostawca.

Domyślnie wyłączone (`llm_fallback_provider = None`). Owinięcie wymaga klucza
drugiego dostawcy — bez niego WARNING i główny adapter jak dotąd. Wywołanie
u zapasowego jest płatnym requestem: liczy się w telemetrii cyklu pod `llm`
(więc widzi je sufit FIN-12 i sekcja kosztu), a tokeny trafiają do tego samego
`UsageSink` z dostawcą i modelem zapasowego. Atrapy FIN-35 nie mają failovera.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from unittest.mock import MagicMock, Mock, call

import pytest

import main_agent
import main_trainer
from src.application.cycle_telemetry import CycleTelemetry
from src.application.ports import LLM_FAILOVER_PROVIDER_KEY, LLMPort
from src.application.quota_monitor import QuotaMonitor
from src.application.usage_ledger import UsageLedger
from src.config import Settings
from src.domain.finops import PAID_LLM_SOURCES
from src.infrastructure.llm.failover import FailoverLLMAdapter
from src.infrastructure.llm.local_stub import LocalStubLLMAdapter


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test",
        anthropic_api_key="sk-ant-test",
        llm_provider="anthropic",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1", "av2"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL"],
        volatility_threshold=Decimal("0.02"),
        ml_model_path="data/models/missing.ubj",
        db_writes_enabled=True,
        paid_calls_enabled=True,
    )


@pytest.fixture
def failing_primary(mocker) -> Mock:
    # Awaria dostawcy po retry (decyzja: failover tylko przy 429/5xx/timeout/sieci).
    primary = Mock(spec=LLMPort)
    primary.analyze.side_effect = TimeoutError("read timed out")
    primary.analyze_mistake.side_effect = TimeoutError("read timed out")
    mocker.patch("src.infrastructure.llm.anthropic_client.AnthropicAdapter", return_value=primary)
    return primary


@pytest.fixture
def secondary(mocker) -> Mock:
    port = Mock(spec=LLMPort)
    port.analyze.return_value = {"trend_direction": "BULLISH", "confidence_score": 0.7}
    port.analyze_mistake.return_value = "fallback insight"
    return port


class TestBuildLLMAdapter:
    def test_fallback_is_off_by_default(self, settings, mocker) -> None:
        mocker.patch("src.infrastructure.llm.anthropic_client.Anthropic")

        adapter = main_agent.build_llm_adapter(settings)

        assert not isinstance(adapter, FailoverLLMAdapter)

    def test_configured_fallback_wraps_the_main_adapter(
        self, settings, mocker, failing_primary, secondary
    ) -> None:
        settings.llm_fallback_provider = "openai"
        settings.llm_fallback_model = "gpt-4o"
        ledger = UsageLedger()
        monitor = QuotaMonitor()
        build_single = mocker.patch("main_agent._build_single_llm", return_value=secondary)

        adapter = main_agent.build_llm_adapter(settings, quota_monitor=monitor, usage_sink=ledger)

        assert isinstance(adapter, FailoverLLMAdapter)
        # Zapasowy dostaje ten sam monitor kwot i ten sam ledger tokenów.
        assert build_single.call_args == call(settings, "openai", "gpt-4o", monitor, ledger)
        assert adapter.analyze("prompt")[LLM_FAILOVER_PROVIDER_KEY] == "openai"

    def test_fallback_model_defaults_to_the_provider_default(
        self, settings, mocker, failing_primary, secondary
    ) -> None:
        from src.infrastructure.llm.openai_client import DEFAULT_MODEL

        settings.llm_fallback_provider = "openai"
        build_single = mocker.patch("main_agent._build_single_llm", return_value=secondary)

        main_agent.build_llm_adapter(settings)

        assert build_single.call_args.args[2] == DEFAULT_MODEL

    def test_missing_fallback_key_keeps_the_main_adapter(self, settings, mocker, caplog) -> None:
        settings.llm_provider = "openai"
        settings.llm_fallback_provider = "anthropic"
        settings.anthropic_api_key = None
        mocker.patch("src.infrastructure.llm.openai_client.OpenAI")

        with caplog.at_level(logging.WARNING, logger="main_agent"):
            adapter = main_agent.build_llm_adapter(settings)

        assert not isinstance(adapter, FailoverLLMAdapter)
        assert "ANTHROPIC_API_KEY" in caplog.text

    def test_fallback_call_is_metered_under_llm(
        self, settings, mocker, failing_primary, secondary
    ) -> None:
        settings.llm_fallback_provider = "openai"
        mocker.patch("main_agent._build_single_llm", return_value=secondary)
        telemetry = CycleTelemetry()

        adapter = main_agent.build_llm_adapter(settings, telemetry=telemetry)
        adapter.analyze("prompt")
        adapter.analyze_mistake("prompt")

        # Request do głównego liczy graf (`_log_paid_call`); tu tylko zapasowy.
        assert dict(telemetry.snapshot().paid_calls) == {"llm": 2}

    def test_fallback_allowance_refuses_the_secondary_before_the_provider(
        self, settings, mocker, failing_primary, secondary
    ) -> None:
        from src.application.metering import PaidCallAllowance

        settings.llm_fallback_provider = "openai"
        mocker.patch("main_agent._build_single_llm", return_value=secondary)
        telemetry = CycleTelemetry()

        adapter = main_agent.build_llm_adapter(
            settings, telemetry=telemetry, fallback_allowance=PaidCallAllowance(0, "recap")
        )
        with pytest.raises(TimeoutError):
            adapter.analyze("prompt")

        secondary.analyze.assert_not_called()
        assert dict(telemetry.snapshot().paid_calls) == {}

    def test_primary_success_meters_nothing_extra(self, settings, mocker, secondary) -> None:
        primary = Mock(spec=LLMPort)
        primary.analyze.return_value = {"trend_direction": "SIDEWAYS"}
        mocker.patch(
            "src.infrastructure.llm.anthropic_client.AnthropicAdapter", return_value=primary
        )
        settings.llm_fallback_provider = "openai"
        mocker.patch("main_agent._build_single_llm", return_value=secondary)
        telemetry = CycleTelemetry()

        main_agent.build_llm_adapter(settings, telemetry=telemetry).analyze("prompt")

        assert dict(telemetry.snapshot().paid_calls) == {}
        secondary.analyze.assert_not_called()


class TestFastLoopWiring:
    def _deps(self, settings, mocker) -> MagicMock:
        mocker.patch("src.infrastructure.adapters.supabase_repo.create_client")
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        main_agent.build_use_case(settings)
        return use_case_cls.call_args.args[0]

    def test_fallback_call_reaches_the_cycle_budget(
        self, settings, mocker, failing_primary, secondary
    ) -> None:
        # FIN-12: sufit liczy wydane wywołania z tej samej telemetrii co graf.
        settings.llm_fallback_provider = "openai"
        mocker.patch("main_agent._build_single_llm", return_value=secondary)
        mocker.patch("src.infrastructure.llm.openai_client.OpenAI")

        deps = self._deps(settings, mocker)
        deps.llm_port.analyze("prompt")

        assert isinstance(deps.llm_port, FailoverLLMAdapter)
        assert deps.telemetry.paid_calls_of(PAID_LLM_SOURCES) == 1

    def test_local_stub_run_has_no_failover(self, settings, mocker) -> None:
        # FIN-35: atrapy nie wysyłają requestów, więc nie ma czego przełączać.
        settings.paid_calls_enabled = False
        settings.llm_fallback_provider = "openai"
        build_single = mocker.patch("main_agent._build_single_llm")

        deps = self._deps(settings, mocker)

        assert isinstance(deps.llm_port, LocalStubLLMAdapter)
        build_single.assert_not_called()


class TestSlowLoopWiring:
    """Decyzje 2026-10-20: request do zapasowego przechodzi przez tygodniowy
    przydział joba (FIN-30) i zużywa go; przy wyczerpanym przydziale failover nie
    startuje, a błąd głównego idzie dalej jak dotąd. Z działającym fallbackiem każdy
    job prosi `slow_loop_paid_budget` o +1 wywołanie (recap 2, sędzia 2) — tylko na
    request do zapasowego; sufit tygodniowy zostaje twardy."""

    @staticmethod
    def _paid(settings, caplog) -> main_trainer._SlowLoopPaidLLM:
        with caplog.at_level(logging.INFO, logger="main_trainer"):
            return main_trainer._SlowLoopPaidLLM(settings, QuotaMonitor(), UsageLedger())

    def test_with_a_fallback_recap_gets_two_calls_and_fails_over(
        self, settings, mocker, failing_primary, secondary, caplog
    ) -> None:
        from src.application.metering import PaidBudgetExhausted

        settings.llm_fallback_provider = "openai"
        settings.weekly_recap_enabled = True
        settings.calibration_judge_enabled = True
        mocker.patch("main_agent._build_single_llm", return_value=secondary)
        paid = self._paid(settings, caplog)
        port = paid.port_for(main_trainer.RECAP_JOB)

        assert "allowed={'recap': 2, 'judge': 2}" in caplog.text
        assert port.analyze("prompt")["trend_direction"] == "BULLISH"
        # Jeden request do głównego (meter joba) + jeden do zapasowego = cały przydział.
        assert dict(paid.telemetry.snapshot().paid_calls) == {"llm": 2}
        with pytest.raises(PaidBudgetExhausted):
            port.analyze("prompt")
        assert failing_primary.analyze.call_count == 1
        assert secondary.analyze.call_count == 1

    def test_without_a_fallback_each_job_keeps_one_call(self, settings, mocker, caplog) -> None:
        from src.application.metering import PaidBudgetExhausted

        primary = Mock(spec=LLMPort)
        primary.analyze.return_value = {"trend_direction": "SIDEWAYS"}
        mocker.patch(
            "src.infrastructure.llm.anthropic_client.AnthropicAdapter", return_value=primary
        )
        settings.weekly_recap_enabled = True
        settings.calibration_judge_enabled = True
        paid = self._paid(settings, caplog)
        port = paid.port_for(main_trainer.RECAP_JOB)

        assert "allowed={'recap': 1, 'judge': 1}" in caplog.text
        port.analyze("prompt")
        with pytest.raises(PaidBudgetExhausted):
            port.analyze("prompt")
        assert primary.analyze.call_count == 1

    def test_fallback_without_its_key_adds_no_call(self, settings, caplog) -> None:
        # Adapter zapasowy nie powstanie (brak klucza), więc nie ma czego budżetować.
        settings.llm_provider = "openai"
        settings.llm_fallback_provider = "anthropic"
        settings.anthropic_api_key = None
        settings.weekly_recap_enabled = True

        self._paid(settings, caplog)

        assert "allowed={'recap': 1}" in caplog.text

    def test_local_stub_run_adds_no_call(self, settings, caplog) -> None:
        settings.paid_calls_enabled = False
        settings.llm_fallback_provider = "openai"
        settings.weekly_recap_enabled = True

        self._paid(settings, caplog)

        assert "allowed={'recap': 1}" in caplog.text

    def test_weekly_ceiling_stays_hard(
        self, settings, mocker, failing_primary, secondary, caplog
    ) -> None:
        # Sufit 3: recap ma pierwszeństwo (2), sędzia dostaje resztę (1) — jego jedyne
        # wywołanie zużywa request do głównego, więc failover sędziego nie startuje.
        settings.llm_fallback_provider = "openai"
        settings.weekly_recap_enabled = True
        settings.calibration_judge_enabled = True
        settings.slow_loop_max_paid_llm_calls_per_week = 3
        mocker.patch("main_agent._build_single_llm", return_value=secondary)
        paid = self._paid(settings, caplog)

        assert "allowed={'recap': 2, 'judge': 1}" in caplog.text
        with pytest.raises(TimeoutError):
            paid.port_for(main_trainer.JUDGE_JOB).analyze("prompt")
        secondary.analyze.assert_not_called()
        assert dict(paid.telemetry.snapshot().paid_calls) == {"llm": 1}

    def test_jobs_keep_separate_allowances(
        self, settings, mocker, failing_primary, secondary, caplog
    ) -> None:
        settings.llm_fallback_provider = "openai"
        settings.weekly_recap_enabled = True
        settings.calibration_judge_enabled = True
        mocker.patch("main_agent._build_single_llm", return_value=secondary)
        paid = self._paid(settings, caplog)

        paid.port_for(main_trainer.RECAP_JOB).analyze("prompt")
        paid.port_for(main_trainer.JUDGE_JOB).analyze("prompt")

        # Failover recapu nie zjada przydziału sędziego.
        assert secondary.analyze.call_count == 2

    def test_local_stub_run_has_no_failover(self, settings, mocker) -> None:
        settings.paid_calls_enabled = False
        settings.db_writes_enabled = False
        settings.llm_fallback_provider = "openai"
        settings.weekly_recap_enabled = True
        build_single = mocker.patch("main_agent._build_single_llm")
        paid = main_trainer._SlowLoopPaidLLM(settings, QuotaMonitor(), UsageLedger())

        paid.port_for(main_trainer.RECAP_JOB).analyze("prompt")

        build_single.assert_not_called()


class TestFallbackSettings:
    def test_defaults_off(self) -> None:
        settings = Settings(
            _env_file=None,
            openai_api_key="sk-test",
            finnhub_api_key="fh",
            alpha_vantage_api_keys=["av1"],
            supabase_url="https://test.supabase.co",
            supabase_key="anon",
        )

        assert (settings.llm_fallback_provider, settings.llm_fallback_model) == (None, None)

    def test_fallback_fields_are_outside_the_recipe_hash(self, settings) -> None:
        # Decyzja właściciela: NON_RECIPE — predykcja od zapasowego niesie własny
        # przepis (flaga `llm_failover` + faktyczny model w kwicie).
        from src.config import NON_RECIPE_SETTINGS_FIELDS

        baseline = settings.config_hash()
        settings.llm_fallback_provider = "openai"
        settings.llm_fallback_model = "gpt-4o"

        assert {"llm_fallback_provider", "llm_fallback_model"} <= NON_RECIPE_SETTINGS_FIELDS
        assert settings.config_hash() == baseline

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="llm_fallback_provider"):
            Settings(
                _env_file=None,
                openai_api_key="sk-test",
                finnhub_api_key="fh",
                alpha_vantage_api_keys=["av1"],
                supabase_url="https://test.supabase.co",
                supabase_key="anon",
                llm_fallback_provider="gemini",
            )
