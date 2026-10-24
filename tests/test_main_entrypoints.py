"""Smoke testy entry pointów — DI wiring oraz top-level main() z mockami sieci."""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, call

import pytest

import main_agent
import main_backtest
import main_trainer
from src.application.alpha_signals import AlphaSignals
from src.application.ports import PreviousVerdict, RepositoryPort
from src.application.use_cases import run_cycle
from src.application.use_cases.assemble_report import AssembledReport
from src.config import Settings
from src.domain.consensus_shift import ShiftKind
from src.domain.council import CouncilVerdict
from src.domain.cycle_maturity import CycleMaturity, SkipReason
from src.domain.digest_lead import LeadItem
from src.domain.earnings import EarningsEvent
from src.domain.insider_flow import InsiderFlowSnapshot
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.macro_risk import MacroAlertLevel
from src.domain.macro_series import MacroSeriesSnapshot, SeriesObservation
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.value_objects import Threshold

# ARCH-06: sekcje raportu składa `ReportAssembler` — tam patchujemy odczyty.
ASSEMBLY = "src.application.use_cases.assemble_report"
# ARCH-07: przebieg symboli prowadzi `RunFastLoopCycle` — tam pula wątków i contagion.
CYCLE = "src.application.use_cases.run_cycle"


@pytest.fixture
def settings() -> Settings:
    # _env_file=None odcina pydantic-settings od loadowania .env (developerzy
    # mają w nim klucze, risk_symbols itp. — testy muszą być deterministyczne).
    return Settings(
        _env_file=None,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1", "av2"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL", "VOO"],
        volatility_threshold=Decimal("0.02"),
        ml_model_path="data/models/missing.ubj",
        db_writes_enabled=True,  # BEZ-08: smoke testy odwzorowują workflowy GHA
        paid_calls_enabled=True,  # FIN-35: jak wyżej — workflowy GHA płacą jawnie
    )


@pytest.fixture
def mock_external_clients(mocker):
    """Mockuje wszystkie I/O na poziomie bibliotek zewnętrznych — żaden test
    entry pointa nie strzela do sieci ani na dysk po model XGBoost."""
    mocker.patch("src.infrastructure.adapters.supabase_repo.create_client")
    mocker.patch("src.infrastructure.llm.openai_client.OpenAI")


class TestLLMProviderFactory:
    def test_defaults_to_openai_adapter(self, settings, mocker):
        mocker.patch("src.infrastructure.llm.openai_client.OpenAI")
        from src.infrastructure.llm.openai_client import OpenAIAdapter

        llm = main_agent.build_llm_adapter(settings)

        assert isinstance(llm, OpenAIAdapter)

    def test_uses_anthropic_when_configured(self, settings, mocker):
        mocker.patch("src.infrastructure.llm.anthropic_client.Anthropic")
        settings.llm_provider = "anthropic"
        settings.anthropic_api_key = "sk-ant-test"
        from src.infrastructure.llm.anthropic_client import AnthropicAdapter

        llm = main_agent.build_llm_adapter(settings)

        assert isinstance(llm, AnthropicAdapter)

    def test_raises_when_anthropic_chosen_without_key(self, settings):
        settings.llm_provider = "anthropic"
        settings.anthropic_api_key = None

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            main_agent.build_llm_adapter(settings)


class TestModelRoutingWiring:
    """NIEZ-17: sufit eskalacji kaskady przychodzi z Settings, a awaria frontiera
    trafia alertem do monitora kwot cyklu."""

    def test_cascade_takes_the_escalation_cap_and_the_quota_monitor(
        self, settings, mocker
    ) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.infrastructure.llm.cascading import CascadingLLMAdapter

        mocker.patch("src.infrastructure.llm.openai_client.OpenAI")
        mocker.patch("src.infrastructure.llm.anthropic_client.Anthropic")
        settings.model_routing_enabled = True
        settings.anthropic_api_key = "sk-ant-test"
        settings.routing_max_escalations = 2
        monitor = QuotaMonitor()

        llm = main_agent.build_llm_adapter(settings, quota_monitor=monitor)

        assert isinstance(llm, CascadingLLMAdapter)
        assert llm._max_escalations == 2
        assert llm._quota_monitor is monitor


class TestCouncilLLMFactory:
    """Heterogeniczna strategia LLM — rada doradcza (15+1 wywołań × 22 symbole
    = ~94% wszystkich callsów) może działać na tańszym modelu niż główna
    analiza, bo persona-acting + JSON nie wymaga frontier reasoningu.

    Gdy obie council_llm_* są None → rada używa głównego LLM (wstecznie
    kompatybilne). Gdy ustawione → osobny adapter z innym providerem/modelem.
    """

    def test_falls_back_to_main_llm_when_council_overrides_none(
        self, settings, mocker
    ):
        mocker.patch("src.infrastructure.llm.openai_client.OpenAI")
        # Brak override'u — Settings ma defaulty None.
        main_llm = main_agent.build_llm_adapter(settings)
        council_llm = main_agent.build_council_llm_adapter(settings, main_llm)

        # Identyczna instancja — brak duplikatu klienta SDK.
        assert council_llm is main_llm

    def test_uses_separate_openai_adapter_when_council_model_set(
        self, settings, mocker
    ):
        mocker.patch("src.infrastructure.llm.openai_client.OpenAI")
        from src.infrastructure.llm.openai_client import OpenAIAdapter

        settings.council_llm_model = "gpt-4o-mini"
        main_llm = main_agent.build_llm_adapter(settings)
        council_llm = main_agent.build_council_llm_adapter(settings, main_llm)

        assert council_llm is not main_llm
        assert isinstance(council_llm, OpenAIAdapter)
        assert council_llm._model == "gpt-4o-mini"  # type: ignore[attr-defined]

    def test_uses_anthropic_when_council_provider_anthropic(
        self, settings, mocker
    ):
        mocker.patch("src.infrastructure.llm.openai_client.OpenAI")
        mocker.patch("src.infrastructure.llm.anthropic_client.Anthropic")
        from src.infrastructure.llm.anthropic_client import AnthropicAdapter

        settings.anthropic_api_key = "sk-ant-test"
        settings.council_llm_provider = "anthropic"
        settings.council_llm_model = "claude-haiku-4-5"
        main_llm = main_agent.build_llm_adapter(settings)
        council_llm = main_agent.build_council_llm_adapter(settings, main_llm)

        assert isinstance(council_llm, AnthropicAdapter)
        assert council_llm._model == "claude-haiku-4-5"  # type: ignore[attr-defined]

    def test_raises_when_council_anthropic_without_key(self, settings):
        settings.anthropic_api_key = None
        settings.council_llm_provider = "anthropic"
        # main pozostaje openai — error musi być specyficzny dla council
        main_llm_stub = MagicMock()

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            main_agent.build_council_llm_adapter(settings, main_llm_stub)


class TestEmbeddingAdapterFactory:
    def test_raises_naming_the_key_when_openai_key_is_missing(self, settings):
        # ARCH-05: klucz jest opcjonalny w Settings, więc żąda go fabryka.
        settings.openai_api_key = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            main_agent.build_embedding_adapter(settings)

    def test_returns_openai_embeddings_for_openai_provider(self, settings, mocker):
        mocker.patch("src.infrastructure.llm.openai_embeddings.OpenAI")
        from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingAdapter

        adapter = main_agent.build_embedding_adapter(settings)

        assert isinstance(adapter, OpenAIEmbeddingAdapter)

    def test_returns_openai_embeddings_even_when_anthropic_provider(
        self, settings, mocker
    ):
        # Anthropic nie ma natywnego embeddings API. OPENAI_API_KEY jest
        # wymagane field w Settings (zawsze obecne) — embeddingi działają
        # niezależnie od wybranego LLM providera. Bez tej zmiany pgvector
        # był NULL dla 50% deploymentów (LLM_PROVIDER=anthropic).
        mocker.patch("src.infrastructure.llm.openai_embeddings.OpenAI")
        from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingAdapter

        settings.llm_provider = "anthropic"
        settings.anthropic_api_key = "sk-ant-test"

        adapter = main_agent.build_embedding_adapter(settings)

        assert isinstance(adapter, OpenAIEmbeddingAdapter)


class TestMainAgent:
    def test_build_use_case_builds_without_error(self, settings, mock_external_clients):
        use_case = main_agent.build_use_case(settings)
        assert use_case is not None

    def test_main_returns_zero_on_success(self, settings, mock_external_clients, mocker):
        # Mock całego use case'a, żeby nie odpalać prawdziwych adapterów
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)

        exit_code = main_agent.main(settings)

        assert exit_code == 0
        assert fake_uc.run.call_count == len(settings.symbols)

    def test_main_returns_zero_on_partial_failure(
        self, settings, mock_external_clients, mocker
    ):
        """Pojedyncze błędy per-symbol (np. niewspierany ticker) → exit 0.
        Agent zrobił swoją robotę dla pozostałych, raport poszedł."""
        fake_uc = MagicMock()
        # 1 symbol pada, 1 przechodzi → częściowy sukces
        fake_uc.run.side_effect = [
            RuntimeError("CSPX.L unsupported"),
            {"status": "ignored", "delta": Decimal("0")},
        ]
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)

        exit_code = main_agent.main(settings)
        assert exit_code == 0

    def test_main_returns_one_when_all_symbols_fail(
        self, settings, mock_external_clients, mocker
    ):
        """Catastrophic failure (np. bad credentials, network down) → exit 1."""
        fake_uc = MagicMock()
        fake_uc.run.side_effect = RuntimeError("API down")
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)

        exit_code = main_agent.main(settings)
        assert exit_code == 1

    def test_main_throttles_between_symbols(
        self, settings, mock_external_clients, mocker
    ):
        """SYMBOL_THROTTLE_SECONDS > 0 → sleep między symbolami."""
        s = settings.model_copy(update={"symbol_throttle_seconds": 0.05})
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mock_sleep = mocker.patch("main_agent.time.sleep")

        main_agent.main(s)

        # Throttle między symbolami (NIE po ostatnim) → n-1 wywołań sleep.
        expected_sleeps = len(s.symbols) - 1
        sleep_calls = [
            c for c in mock_sleep.call_args_list
            if c.args and c.args[0] == pytest.approx(0.05)
        ]
        assert len(sleep_calls) == expected_sleeps

    def test_main_no_throttle_when_zero(
        self, settings, mock_external_clients, mocker
    ):
        """SYMBOL_THROTTLE_SECONDS=0 → brak sleepa (default)."""
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mock_sleep = mocker.patch("main_agent.time.sleep")

        main_agent.main(settings)

        sleep_with_zero = [
            c for c in mock_sleep.call_args_list
            if c.args and c.args[0] > 0
        ]
        assert sleep_with_zero == []

    def test_done_log_denominator_counts_eligible_not_settings_symbols(
        self, settings, mock_external_clients, mocker, caplog
    ):
        """#34: log 'Fast Loop done — failures=d/d' MUSI dzielić przez liczbę
        FAKTYCZNIE przetworzonych symboli (eligible = symbols + crypto −
        unsupported), nie przez len(settings.symbols). Inaczej przy obecności
        krypto / unsupported mianownik jest zły (failures mogą nawet przekroczyć
        mianownik), co rozjeżdża się z logiką exit-code używającą len(eligible)."""
        import logging

        # symbols=2 (AAPL, VOO), crypto=1 (BTC), unsupported=1 (VOO) →
        # eligible = [AAPL, BTC] (2). len(settings.symbols)=2 też jest 2, więc
        # dobieramy liczby tak, by oba mianowniki RÓŻNIŁY się: bez unsupported,
        # z krypto → eligible=3, ale settings.symbols=2.
        s = settings.model_copy(update={
            "symbols": ["AAPL", "VOO"],
            "crypto_symbols": ["BTC"],
            "symbols_unsupported_price": [],
        })
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)

        with caplog.at_level(logging.INFO, logger="main_agent"):
            main_agent.main(s)

        done_logs = [
            rec.message for rec in caplog.records
            if "Fast Loop done" in rec.message
        ]
        assert done_logs, "spodziewany log 'Fast Loop done'"
        # eligible = AAPL + VOO + BTC = 3 (nic nie unsupported), a
        # len(settings.symbols) = 2 → mianownik MUSI być 3.
        assert "failures=0/3" in done_logs[0], done_logs[0]
        assert "/2" not in done_logs[0], done_logs[0]

    def test_main_skips_unsupported_symbols_as_ignored(
        self, settings, mock_external_clients, mocker
    ):
        """Tickery z SYMBOLS_UNSUPPORTED_PRICE nie wchodzą w use_case.run()."""
        s = settings.model_copy(update={
            "symbols": ["AAPL", "CSPX.L", "XDWD.DE"],
            "symbols_unsupported_price": ["CSPX.L", "XDWD.DE"],
        })
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)

        exit_code = main_agent.main(s)

        # Tylko AAPL dostała się do use_case.
        assert exit_code == 0
        assert fake_uc.run.call_count == 1
        called_symbol = fake_uc.run.call_args_list[0].args[0]
        assert called_symbol == "AAPL"

    def test_main_maps_results_with_the_cycle_start(
        self, settings, mock_external_clients, mocker
    ):
        """ARCH-26: odznaki proweniencji i nagłówek maila biorą ten sam zegar
        — `started_at` cyklu, przekazany do mappera wyników."""
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        analyze = mocker.spy(main_agent, "_analyze_symbols")
        dispatch = mocker.spy(main_agent, "_dispatch_reports")

        main_agent.main(settings)

        assert analyze.call_args.kwargs["cycle_at"] == dispatch.call_args.kwargs["started_at"]


class TestBuildAgentGraphDeps:
    """TEST-09 (G56): fabryka deps grafu sprawdzana na polach, nie przez `is not None`.

    `test_build_use_case_builds_without_error` dowodzi tylko, że DI nie rzuca —
    port podpięty pod złe pole albo atrapa w płatnym przebiegu przeszłyby."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_sigma_target_flag_reaches_the_predict_node(
        self, settings, mock_external_clients, enabled
    ):
        # ML-11: serving mnoży prognozę modelu w jednostkach σ przez σ symbolu.
        settings.ml_sigma_target_enabled = enabled

        deps = main_agent.build_agent_graph_deps(settings)

        assert deps.ml_sigma_target_enabled is enabled

    @pytest.mark.parametrize("enabled", [True, False])
    def test_calendar_edge_horizon_flag_reaches_the_predict_node(
        self, settings, mock_external_clients, enabled
    ):
        # WIA-28: horyzont edge z kalendarza NYSE za flagą.
        settings.edge_calendar_horizon_enabled = enabled

        deps = main_agent.build_agent_graph_deps(settings)

        assert deps.edge_calendar_horizon_enabled is enabled

    @pytest.mark.parametrize("enabled", [True, False])
    def test_split_detection_flag_reaches_the_graph(
        self, settings, mock_external_clients, enabled
    ):
        # DAT-23: detekcja splitu z `pc` za flagą (zapis w `corporate_actions`, 062).
        settings.split_detection_enabled = enabled

        deps = main_agent.build_agent_graph_deps(settings)

        assert deps.split_detection_enabled is enabled

    @pytest.mark.parametrize("enabled", [True, False])
    def test_price_outlier_guard_flag_reaches_the_graph(
        self, settings, mock_external_clients, enabled
    ):
        # DAT-15: odporny detektor anomalii ceny za flagą.
        settings.price_outlier_guard_enabled = enabled

        deps = main_agent.build_agent_graph_deps(settings)

        assert deps.price_outlier_guard_enabled is enabled

    @pytest.mark.parametrize("enabled", [True, False])
    def test_prompt_taint_fence_flag_reaches_the_graph_and_the_council(
        self, settings, mock_external_clients, enabled
    ):
        # BEZ-14: fence lekcji (prompt predykcji) i uzasadnień person (przewodniczący).
        settings.prompt_taint_fence_enabled = enabled

        deps = main_agent.build_agent_graph_deps(settings)

        assert deps.prompt_taint_fence_enabled is enabled
        assert deps.council_port._fence_opinion_reasoning is enabled

    def test_tool_use_disabled_wires_no_tool_use_port(self, settings, mock_external_clients):
        settings.tool_use_enabled = False

        deps = main_agent.build_agent_graph_deps(settings)

        assert deps.tool_use_port is None
        assert list(deps.research_tools) == []
        assert deps.tool_use_threshold is None

    def test_tool_use_enabled_wires_port_tools_and_their_threshold(
        self, settings, mock_external_clients, mocker
    ):
        mocker.patch("src.infrastructure.llm.tool_calling_openai.OpenAI")
        from src.infrastructure.llm.tool_calling_openai import ToolCallingOpenAIAdapter

        settings.tool_use_enabled = True

        deps = main_agent.build_agent_graph_deps(settings)

        assert isinstance(deps.tool_use_port, ToolCallingOpenAIAdapter)
        assert deps.research_tools
        assert deps.tool_use_threshold == Threshold(settings.tool_use_volatility_threshold)

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [(None, None), (Decimal("0.10"), Threshold(Decimal("0.10")))],
    )
    def test_crypto_tool_use_threshold_reaches_the_tool_use_gate(
        self, settings, mock_external_clients, mocker, configured, expected
    ):
        # FIN-08: osobny próg tool-use krypto; brak = wspólny próg (None w deps).
        mocker.patch("src.infrastructure.llm.tool_calling_openai.OpenAI")
        settings.tool_use_enabled = True
        settings.crypto_tool_use_volatility_threshold = configured

        deps = main_agent.build_agent_graph_deps(settings)

        assert deps.crypto_tool_use_threshold == expected

    def test_crypto_tool_use_threshold_needs_the_tool_use_port(
        self, settings, mock_external_clients
    ):
        settings.tool_use_enabled = False
        settings.crypto_tool_use_volatility_threshold = Decimal("0.10")

        deps = main_agent.build_agent_graph_deps(settings)

        assert deps.crypto_tool_use_threshold is None

    def test_paid_run_wires_the_alpha_vantage_feeds_council_and_embeddings(
        self, settings, mock_external_clients
    ):
        from src.infrastructure.adapters.advisory_council import LLMAdvisoryCouncil
        from src.infrastructure.adapters.alpha_vantage_adapters import (
            AlphaVantageNewsAdapter,
            AlphaVantageSentimentAdapter,
        )
        from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingAdapter

        deps = main_agent.build_agent_graph_deps(settings)

        assert isinstance(deps.sentiment_port, AlphaVantageSentimentAdapter)
        assert isinstance(deps.news_port, AlphaVantageNewsAdapter)
        assert isinstance(deps.council_port, LLMAdvisoryCouncil)
        assert isinstance(deps.embedding_port, OpenAIEmbeddingAdapter)
        assert deps.threshold == Threshold(settings.volatility_threshold)
        assert deps.paid_call_cap == settings.max_paid_llm_calls_per_cycle

    def test_stub_run_wires_no_council_embeddings_or_tool_use(
        self, settings, mock_external_clients
    ):
        from src.infrastructure.llm.local_stub import LocalStubLLMAdapter

        settings.paid_calls_enabled = False
        settings.db_writes_enabled = False
        settings.tool_use_enabled = True

        deps = main_agent.build_agent_graph_deps(settings)

        assert isinstance(deps.llm_port, LocalStubLLMAdapter)
        assert deps.council_port is None
        assert deps.embedding_port is None
        assert deps.tool_use_port is None

    def test_zero_council_threshold_leaves_no_council_gate(self, settings, mock_external_clients):
        settings.council_volatility_threshold = Decimal("0")

        assert main_agent.build_agent_graph_deps(settings).council_threshold is None

    def test_fundamentals_port_reads_only_the_cache(self, settings, mocker):
        get = mocker.patch("requests.Session.get")
        repository = MagicMock(spec=RepositoryPort)
        repository.get_cached_fundamentals.return_value = None
        mocker.patch("src.infrastructure.llm.openai_client.OpenAI")

        deps = main_agent.build_agent_graph_deps(settings, repository=repository)

        assert deps.repository_port is repository
        assert deps.fundamentals_port is not None
        assert deps.fundamentals_port.get_fundamentals("AAPL") is None
        get.assert_not_called()

    def test_build_use_case_runs_on_the_factory_deps(self, settings, mock_external_clients, mocker):
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        factory = mocker.spy(main_agent, "build_agent_graph_deps")

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0] is factory.spy_return


class TestProductionConfigWiring:
    """TEST-04 (G64): produkcyjna kombinacja flag z `config.toml` przechodzi przez DI.

    Fixture `settings` ma czyste defaulty (conftest wyłącza źródło TOML), więc
    włączone w produkcji flagi nigdy razem nie trafiały do `build_use_case`. Tu
    źródło TOML jest aktywne, sekrety idą jako kwargs, a SDK są zmockowane na
    poziomie modułów adapterów — żadnego `importorskip`, bo brak SDK to błąd P0
    (G05), a nie powód do pominięcia testu."""

    _SDK_CLIENTS = (
        "src.infrastructure.adapters.supabase_repo.create_client",
        "src.infrastructure.llm.openai_client.OpenAI",
        "src.infrastructure.llm.openai_embeddings.OpenAI",
        "src.infrastructure.llm.tool_calling_openai.OpenAI",
        "src.infrastructure.llm.anthropic_client.Anthropic",
        "src.infrastructure.llm.tool_calling_anthropic.Anthropic",
    )

    def test_build_use_case_with_production_config_toml(self, monkeypatch, mocker):
        from src.application.use_cases.analyze_market import AnalyzeMarketUseCase

        monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
        monkeypatch.delenv("STOCKAGENT_TOML_FILE", raising=False)
        for target in self._SDK_CLIENTS:
            mocker.patch(target)
        # Przełączniki płatnego przebiegu jak w workflow Fast Loopa (env GHA).
        production = Settings(
            _env_file=None,
            openai_api_key="sk-test",
            anthropic_api_key="sk-ant-test",
            finnhub_api_key="fh",
            alpha_vantage_api_keys=["av1", "av2"],
            supabase_url="https://test.supabase.co",
            supabase_key="service",
            db_writes_enabled=True,
            paid_calls_enabled=True,
        )
        assert production.llm_provider == "anthropic", "config.toml nie został wczytany"

        use_case = main_agent.build_use_case(production)

        assert isinstance(use_case, AnalyzeMarketUseCase)


class TestFastLoopFundamentalsPort:
    """TEST-06 (G70): Fast Loop czyta fundamenty wyłącznie z cache Supabase.

    Port fundamentów idzie przez graf PRZED bramką volatility, więc delegat
    z prawdziwym klientem Alpha Vantage kosztowałby request na każdy symbol
    w każdym cyklu. Cache miss ma skończyć się brakiem danych, nie HTTP."""

    def test_fast_loop_fundamentals_cache_miss_makes_no_http_call(self, mocker):
        get = mocker.patch("requests.Session.get")
        post = mocker.patch("requests.Session.post")
        repository = MagicMock(spec=RepositoryPort)
        repository.get_cached_fundamentals.return_value = None

        port = main_agent.build_fast_loop_fundamentals_port(repository)

        assert port.get_fundamentals("AAPL") is None
        get.assert_not_called()
        post.assert_not_called()
        repository.save_fundamentals.assert_not_called()


class TestMainTrainer:
    def test_build_use_case_builds_without_error(self, settings, mock_external_clients):
        use_case = main_trainer.build_use_case(settings)
        assert use_case is not None

    def test_refresh_fetches_only_selected_batch(self, settings, mocker) -> None:
        # DAT-19: przy 1 kluczu (budżet 12 spółek) refresh bierze spółki bez wpisu
        # w cache i najstarsze, a nie zawsze pierwsze 12 z listy.
        from src.application.quota_monitor import QuotaMonitor

        settings.alpha_vantage_api_keys = ["av1"]
        settings.symbols_etf = []
        settings.symbols = [f"S{i:02d}" for i in range(1, 16)]
        now = datetime.now(UTC)
        repository = MagicMock(spec=RepositoryPort)
        repository.get_fundamentals_fetched_at.return_value = {
            f"S{i:02d}": now - timedelta(days=i) for i in range(3, 16)
        }
        mocker.patch("main_trainer.AlphaVantageFundamentalsAdapter")
        cached = mocker.patch("main_trainer.CachedFundamentalsAdapter")

        main_trainer._refresh_fundamentals_cache(settings, repository, QuotaMonitor())

        fetched = [c.args[0] for c in cached.return_value.get_fundamentals.call_args_list]
        assert fetched == ["S01", "S02", *[f"S{i:02d}" for i in range(15, 5, -1)]]
        repository.get_fundamentals_fetched_at.assert_called_once_with(settings.symbols)

    def test_main_returns_zero_on_success(self, settings, mock_external_clients, mocker):
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped", "reason": "not enough data"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        exit_code = main_trainer.main(settings)

        assert exit_code == 0
        fake_uc.refresh_feature_store.assert_called_once()
        assert fake_uc.run.call_count == len(settings.symbols)
        assert fake_uc.run.call_args_list[0].kwargs == {"refresh_view": False}
        assert fake_uc.run.call_args_list[1].kwargs == {"refresh_view": False}

    def test_main_returns_one_on_training_failure(
        self, settings, mock_external_clients, mocker
    ):
        fake_uc = MagicMock()
        fake_uc.run.side_effect = RuntimeError("DB down")
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        exit_code = main_trainer.main(settings)

        assert exit_code == 1

    def test_main_returns_one_when_feature_store_refresh_fails(
        self, settings, mock_external_clients, mocker
    ):
        fake_uc = MagicMock()
        fake_uc.refresh_feature_store.side_effect = RuntimeError("DB down")
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        exit_code = main_trainer.main(settings)

        assert exit_code == 1
        fake_uc.run.assert_not_called()


class TestMainBacktest:
    """Offline walk-forward backtest harness (T3) — DI + top-level main()."""

    def test_build_use_case_builds_without_error(
        self, settings, mock_external_clients
    ):
        use_case = main_backtest.build_use_case(settings)
        assert use_case is not None

    def test_main_runs_backtest_per_symbol(
        self, settings, mock_external_clients, mocker
    ):
        from src.domain.backtest import BacktestSummary

        fake_uc = MagicMock()
        fake_uc.run.return_value = BacktestSummary(
            n_folds=4,
            mean_candidate_rmse=0.03,
            mean_baseline_rmse=0.05,
            mean_hit_rate=0.6,
            improvement_pct=40.0,
            beats_baseline=True,
            verdict="model bije baseline o 40.0%",
        )
        mocker.patch("main_backtest.BacktestUseCase", return_value=fake_uc)

        exit_code = main_backtest.main(settings)

        assert exit_code == 0
        assert fake_uc.run.call_count == len(settings.symbols)
        # Widok odświeżamy raz globalnie → per-symbol refresh_view=False.
        assert fake_uc.run.call_args_list[0].kwargs == {"refresh_view": False}

    def test_main_survives_single_symbol_failure(
        self, settings, mock_external_clients, mocker
    ):
        from src.domain.backtest import BacktestSummary

        fake_uc = MagicMock()
        fake_uc.run.side_effect = [
            RuntimeError("XGBoost boom"),
            BacktestSummary(
                n_folds=4,
                mean_candidate_rmse=0.03,
                mean_baseline_rmse=0.05,
                mean_hit_rate=0.6,
                improvement_pct=40.0,
                beats_baseline=True,
                verdict="model bije baseline o 40.0%",
            ),
        ]
        mocker.patch("main_backtest.BacktestUseCase", return_value=fake_uc)

        # Pojedynczy błąd per-symbol nie wywala harnessa.
        assert main_backtest.main(settings) == 0

    def test_backtest_builds_no_paid_adapter(
        self, settings, mock_external_clients, mocker
    ):
        # DOC-21: „no retroactive LLM replays” — backtest odtwarza zebrane cechy
        # XGBoostem i nigdy nie pyta LLM o historię, której model mógł się nauczyć.
        import anthropic
        import openai

        from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient

        paid_clients = [
            mocker.patch.object(client, "__init__", side_effect=AssertionError(client.__name__))
            for client in (openai.OpenAI, anthropic.Anthropic, AlphaVantageClient)
        ]
        fake_uc = MagicMock()
        fake_uc.run.side_effect = RuntimeError("bez danych")

        main_backtest.build_use_case(settings)
        mocker.patch("main_backtest.BacktestUseCase", return_value=fake_uc)
        # NIEZ-18: wszystkie symbole padły → exit 1; test pilnuje tylko, że żaden
        # płatny klient nie powstał.
        assert main_backtest.main(settings) == 1

        for client in paid_clients:
            client.assert_not_called()

    @pytest.mark.parametrize("enabled", [False, True])
    def test_backtest_follows_the_feature_set_filter(
        self, settings, mock_external_clients, mocker, enabled: bool
    ) -> None:
        # ML-51: offline werdykt na tej samej próbce co trening (`main_trainer`).
        from src.application.ml_features import FEATURE_SET_VERSION

        use_case_cls = mocker.patch("main_backtest.BacktestUseCase")
        use_case_cls.return_value.run.side_effect = RuntimeError("bez foldów")
        settings.feature_set_filter_enabled = enabled

        main_backtest.build_use_case(settings)
        main_backtest.main(settings)

        expected = FEATURE_SET_VERSION if enabled else None
        versions = [c.kwargs["feature_set_version"] for c in use_case_cls.call_args_list]
        assert versions == [expected, expected]

    def test_main_returns_one_when_all_symbols_fail(
        self, settings, mock_external_clients, mocker
    ):
        # NIEZ-18: jak Fast Loop — exit 1 tylko, gdy padły wszystkie symbole.
        fake_uc = MagicMock()
        fake_uc.run.side_effect = RuntimeError("XGBoost boom")
        mocker.patch("main_backtest.BacktestUseCase", return_value=fake_uc)

        assert main_backtest.main(settings) == 1


class TestBacktestCoverage:
    """NIEZ-18: backtest iterował same `symbols`, więc krypto nigdy nie dostawało
    oceny offline. Krypto idzie przez ten sam `build_use_case`, z kosztem obrotu
    krypto (WIA-50)."""

    @staticmethod
    def _summary():
        from src.domain.backtest import BacktestSummary

        return BacktestSummary(
            n_folds=4,
            mean_candidate_rmse=0.03,
            mean_baseline_rmse=0.05,
            mean_hit_rate=0.6,
            improvement_pct=40.0,
            beats_baseline=True,
            verdict="model bije baseline o 40.0%",
        )

    def test_backtest_covers_crypto_symbols(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.crypto_symbols = ["BTC"]
        by_cost: dict[float, MagicMock] = {}

        def use_case(*, cost_bps: float, **_: object) -> MagicMock:
            fake = by_cost.setdefault(cost_bps, MagicMock())
            fake.run.return_value = self._summary()
            return fake

        mocker.patch("main_backtest.BacktestUseCase", side_effect=use_case)

        main_backtest.main(settings)

        stock = by_cost[settings.paper_cost_bps]
        crypto = by_cost[settings.crypto_paper_cost_bps]
        assert [c.args[0] for c in stock.run.call_args_list] == ["AAPL", "VOO"]
        assert [c.args[0] for c in crypto.run.call_args_list] == ["BTC"]

    def test_main_builds_the_backtest_through_build_use_case(
        self, settings, mock_external_clients, mocker
    ) -> None:
        build = mocker.patch("main_backtest.build_use_case")
        build.return_value.run.return_value = self._summary()

        main_backtest.main(settings)

        build.assert_called_once()
        assert build.call_args.args == (settings,)


class TestBuildMacroRiskUseCase:
    """Factory MonitorMacroRisk — DI Risk Watch w main_agent.py."""

    def _settings(self, **overrides) -> Settings:
        from src.domain.macro_risk import MacroRiskInstrumentType

        base = {
            "openai_api_key": "sk-test",
            "finnhub_api_key": "fh",
            "alpha_vantage_api_keys": ["av1"],
            "supabase_url": "https://test.supabase.co",
            "supabase_key": "anon",
            "symbols": ["AAPL"],
            "volatility_threshold": Decimal("0.02"),
            "ml_model_path": "data/models/missing.ubj",
            "risk_symbols": ["SH", "GLD"],
            "risk_symbol_types": {
                "SH": MacroRiskInstrumentType.INVERSE_EQUITY,
                "GLD": MacroRiskInstrumentType.SAFE_HAVEN,
            },
            "nbp_enabled": False,
        }
        base.update(overrides)
        return Settings(**base)

    def test_returns_none_when_no_risk_symbols(self):
        s = self._settings(risk_symbols=[])
        repo = MagicMock()
        assert main_agent.build_macro_risk_use_case(s, repo) is None

    def test_returns_none_when_no_symbol_types(self):
        s = self._settings(risk_symbol_types={})
        repo = MagicMock()
        assert main_agent.build_macro_risk_use_case(s, repo) is None

    def test_returns_use_case_without_macro_port_when_nbp_disabled(self):
        s = self._settings(nbp_enabled=False)
        repo = MagicMock()
        uc = main_agent.build_macro_risk_use_case(s, repo)
        assert uc is not None
        assert uc._macro is None  # type: ignore[attr-defined]

    def test_returns_use_case_with_nbp_client_when_enabled(self):
        from src.infrastructure.adapters.nbp_client import NbpClient

        s = self._settings(nbp_enabled=True)
        repo = MagicMock()
        uc = main_agent.build_macro_risk_use_case(s, repo)
        assert uc is not None
        assert isinstance(uc._macro, NbpClient)  # type: ignore[attr-defined]


class TestNotifierFactory:
    """build_notifier — gdy mail wyłączony, log MUSI wskazać który z trzech
    warunków zawiódł (regresja 2026-06-03: 'Notifications disabled' nie mówił
    czego brakuje → zgadywanka, gdy DIGEST_TO_EMAIL nie był repo Secret)."""

    def _ready(self, settings: Settings) -> Settings:
        settings.notifications_enabled = True
        settings.resend_api_key = "re_test"
        settings.digest_to_email = "you@example.com"
        return settings

    def test_returns_resend_when_all_present(self, settings):
        from src.infrastructure.adapters.resend_notifier import ResendNotifier

        n = main_agent.build_notifier(self._ready(settings))
        assert isinstance(n, ResendNotifier)

    def test_warns_naming_missing_digest_to_email(self, settings, caplog):
        import logging

        from src.infrastructure.adapters.resend_notifier import NullNotifier

        s = self._ready(settings)
        s.digest_to_email = None  # włączone, ale brak odbiorcy
        with caplog.at_level(logging.WARNING):
            n = main_agent.build_notifier(s)
        assert isinstance(n, NullNotifier)
        assert "DIGEST_TO_EMAIL" in caplog.text
        assert "RESEND_API_KEY" not in caplog.text  # ten jest obecny

    def test_warns_naming_missing_resend_key(self, settings, caplog):
        import logging

        s = self._ready(settings)
        s.resend_api_key = None
        with caplog.at_level(logging.WARNING):
            main_agent.build_notifier(s)
        assert "RESEND_API_KEY" in caplog.text

    def test_no_warning_when_intentionally_disabled(self, settings, caplog):
        import logging

        from src.infrastructure.adapters.resend_notifier import NullNotifier

        settings.notifications_enabled = False
        with caplog.at_level(logging.WARNING):
            n = main_agent.build_notifier(settings)
        assert isinstance(n, NullNotifier)
        # Świadome wyłączenie → bez WARNING (najwyżej INFO).
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestProcessOneSymbol:
    def _kwargs(self, settings, use_case):
        return {
            "use_case": use_case,
            "crypto_set": set(),
            "etf_set": set(),
            "regime_context": "",
            "regime_multiplier": 1.0,
            "collect_alpha": False,
            "alpha_use_case": MagicMock(),
        }

    def test_success_returns_result_not_failure(self, settings):
        uc = MagicMock()
        uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        proc = main_agent._process_one_symbol(
            "AAPL", peer_ctx=(), **self._kwargs(settings, uc)
        )
        assert proc.is_failure is False
        assert proc.result.symbol == "AAPL"
        uc.run.assert_called_once()

    def test_exception_returns_failure_with_error(self, settings):
        uc = MagicMock()
        uc.run.side_effect = RuntimeError("boom")
        proc = main_agent._process_one_symbol(
            "AAPL", peer_ctx=(), **self._kwargs(settings, uc)
        )
        assert proc.is_failure is True
        assert proc.result.status == "error"

    def test_collect_alpha_enriches_on_success(self, settings):
        uc = MagicMock()
        uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        kwargs = self._kwargs(settings, uc)
        kwargs["collect_alpha"] = True
        proc = main_agent._process_one_symbol("AAPL", peer_ctx=(), **kwargs)
        kwargs["alpha_use_case"].enrich.assert_called_once_with("AAPL")
        assert proc.alpha_signal is not None

    def test_provenance_is_measured_against_the_cycle_clock(self, settings):
        # ARCH-26: fundamenty sprzed 24 h względem startu cyklu są świeże, choć
        # wobec zegara rzeczywistego dawno przekroczyły TTL 144 h.
        from src.domain.value_objects import Fundamentals

        cycle_at = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
        uc = MagicMock()
        uc.run.return_value = {
            "status": "saved",
            "delta": Decimal("0.03"),
            "fundamentals": Fundamentals(
                trailing_pe=None, forward_pe=None, peg_ratio=None, eps_growth_yoy=None,
                fetched_at=cycle_at - timedelta(hours=24),
            ),
        }
        proc = main_agent._process_one_symbol(
            "AAPL", peer_ctx=(), cycle_at=cycle_at, **self._kwargs(settings, uc)
        )
        assert [b.level.value for b in proc.result.provenance_badges] == ["FRESH"]

    def test_error_message_omits_secret_query_parameter(self, settings):
        # BEZ-07: błąd symbolu idzie do maila — URL z kluczem nie może w nim zostać.
        import requests

        uc = MagicMock()
        uc.run.side_effect = requests.HTTPError(
            "401 Client Error: Unauthorized for url: "
            "https://finnhub.io/api/v1/quote?symbol=AAPL&token=symbol-secret-token"
        )
        proc = main_agent._process_one_symbol(
            "AAPL", peer_ctx=(), **self._kwargs(settings, uc)
        )
        assert proc.is_failure is True
        assert proc.result.error_message is not None
        assert "401 Client Error" in proc.result.error_message
        assert "symbol-secret-token" not in proc.result.error_message

    def test_port_error_is_one_warning_without_traceback(self, settings, caplog):
        # ARCH-33: awaria dostawcy przetłumaczona na taksonomię to stan znany — przy
        # otwartym obwodzie (NIEZ-31) 43 tracebacki tego samego błędu to szum.
        import logging

        from src.application.errors import ProviderUnavailable

        uc = MagicMock()
        uc.run.side_effect = ProviderUnavailable("finnhub.io circuit open")
        with caplog.at_level(logging.WARNING):
            proc = main_agent._process_one_symbol(
                "AAPL", peer_ctx=(), **self._kwargs(settings, uc)
            )

        assert proc.is_failure is True
        assert proc.result.error_type == "ProviderUnavailable"
        failures = [r for r in caplog.records if "AAPL" in r.getMessage()]
        assert [(r.levelno, r.exc_info) for r in failures] == [(logging.WARNING, None)]
        assert "finnhub.io circuit open" in failures[0].getMessage()

    def test_unexpected_error_keeps_the_traceback(self, settings, caplog):
        import logging

        uc = MagicMock()
        uc.run.side_effect = RuntimeError("bug")
        with caplog.at_level(logging.WARNING):
            main_agent._process_one_symbol("AAPL", peer_ctx=(), **self._kwargs(settings, uc))

        failures = [r for r in caplog.records if "AAPL" in r.getMessage()]
        assert [r.levelno for r in failures] == [logging.ERROR]
        assert failures[0].exc_info is not None


class TestAnalyzeSymbolsParallel:
    def _kwargs(self, settings, use_case):
        return {
            "use_case": use_case,
            "settings": settings,
            "crypto_set": set(),
            "etf_set": set(),
            "regime_context": "",
            "regime_multiplier": 1.0,
            "collect_alpha": False,
            "alpha_use_case": MagicMock(),
        }

    def test_parallel_processes_all_in_eligible_order(self, settings):
        s = settings.model_copy(update={"symbol_concurrency": 4})
        uc = MagicMock()
        uc.run.side_effect = lambda sym, **kw: {
            "status": "ignored", "delta": Decimal("0"), "symbol": sym,
        }
        eligible = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"]
        results, failures, _signals, _prices = main_agent._analyze_symbols(
            eligible, **self._kwargs(s, uc)
        )
        assert failures == 0
        assert [r.symbol for r in results] == eligible  # kolejność zachowana
        assert uc.run.call_count == len(eligible)

    def test_parallel_single_failure_isolated(self, settings):
        s = settings.model_copy(update={"symbol_concurrency": 4})
        uc = MagicMock()

        def _run(sym, **kw):
            if sym == "MSFT":
                raise RuntimeError("boom")
            return {"status": "ignored", "delta": Decimal("0")}

        uc.run.side_effect = _run
        eligible = ["AAPL", "MSFT", "NVDA"]
        results, failures, _s, _p = main_agent._analyze_symbols(
            eligible, **self._kwargs(s, uc)
        )
        assert failures == 1
        assert len(results) == 3  # reszta przetworzona mimo błędu jednego

    def test_parallel_does_not_build_peer_context(self, settings, mocker):
        s = settings.model_copy(update={
            "symbol_concurrency": 4, "contagion_enabled": False,
        })
        uc = MagicMock()
        uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        spy = mocker.patch(f"{CYCLE}.build_peer_context")
        main_agent._analyze_symbols(["AAPL", "MSFT"], **self._kwargs(s, uc))
        spy.assert_not_called()

    def test_contagion_forces_sequential_despite_concurrency(self, settings, mocker):
        s = settings.model_copy(update={
            "symbol_concurrency": 4, "contagion_enabled": True,
        })
        uc = MagicMock()
        uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        spy = mocker.patch(f"{CYCLE}.build_peer_context", return_value=())
        main_agent._analyze_symbols(["AAPL", "MSFT"], **self._kwargs(s, uc))
        # Sekwencyjna ścieżka (peer_context budowany) mimo concurrency=4.
        assert spy.call_count == 2

    def test_parallel_uses_thread_pool_executor(self, settings, mocker):
        s = settings.model_copy(update={"symbol_concurrency": 3})
        uc = MagicMock()
        uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        spy = mocker.spy(run_cycle, "ThreadPoolExecutor")
        main_agent._analyze_symbols(["AAPL", "MSFT"], **self._kwargs(s, uc))
        spy.assert_called_once()
        assert spy.call_args.kwargs.get("max_workers") == 3

    def test_sequential_path_no_thread_pool(self, settings, mocker):
        s = settings.model_copy(update={"symbol_concurrency": 1})
        uc = MagicMock()
        uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        spy = mocker.spy(run_cycle, "ThreadPoolExecutor")
        main_agent._analyze_symbols(["AAPL", "MSFT"], **self._kwargs(s, uc))
        spy.assert_not_called()

    @pytest.mark.parametrize("concurrency", [1, 4])
    def test_cycle_clock_reaches_every_symbol(self, settings, mocker, concurrency):
        # ARCH-26: obie ścieżki (sekwencyjna i równoległa) mapują wynik z zegarem cyklu.
        s = settings.model_copy(update={"symbol_concurrency": concurrency})
        uc = MagicMock()
        uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        spy = mocker.spy(run_cycle, "to_symbol_result")
        cycle_at = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
        main_agent._analyze_symbols(
            ["AAPL", "MSFT"], cycle_at=cycle_at, **self._kwargs(s, uc)
        )
        assert [c.kwargs.get("now") for c in spy.call_args_list] == [cycle_at, cycle_at]


class TestFetchReportContextPersonaTrackRecord:
    """#3 — orkiestrator dociąga track record person i przepuszcza go przez
    domenowy `rank_personas` (próg min_votes) zanim trafi do raportu."""

    @staticmethod
    def _repo(**overrides: object) -> MagicMock:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_accuracy_stats.return_value = None
        repo.get_recently_resolved_predictions.return_value = []
        repo.get_recent_quota_alerts.return_value = []
        repo.get_council_vote_history.return_value = []
        repo.get_persona_track_record.return_value = {}
        for name, value in overrides.items():
            getattr(repo, name).return_value = value
        return repo

    @staticmethod
    def _monitor() -> MagicMock:
        monitor = MagicMock()
        monitor.alerts = []
        return monitor

    def test_ranks_personas_from_repository_stats(self) -> None:
        repo = self._repo(
            get_persona_track_record={"Wood": (0.41, 18), "Buffett": (0.68, 22)}
        )

        *_, track_record = main_agent._fetch_report_context(
            repo, self._monitor(), ["AAPL"]
        )

        assert [r.investor_name for r in track_record] == ["Buffett", "Wood"]

    def test_applies_domain_min_votes_threshold(self) -> None:
        # Persona z 1 głosem i 100% trafności nie może wejść do rankingu.
        repo = self._repo(
            get_persona_track_record={"Nowicjusz": (1.0, 1), "Buffett": (0.68, 22)}
        )

        *_, track_record = main_agent._fetch_report_context(
            repo, self._monitor(), ["AAPL"]
        )

        assert [r.investor_name for r in track_record] == ["Buffett"]

    def test_rpc_failure_does_not_break_the_cycle(self) -> None:
        repo = self._repo()
        repo.get_persona_track_record.side_effect = Exception("RPC missing")

        *_, track_record = main_agent._fetch_report_context(
            repo, self._monitor(), ["AAPL"]
        )

        assert track_record == []


class TestResolveRegimeYieldCurveVote:
    """#7 — inwersja krzywej wchodzi do RegimeDetector jako GŁOS w liczniku
    ELEVATED, nie jako weto i nie przez `_compute_overall_alert`."""

    @staticmethod
    def _settings(**overrides: object) -> MagicMock:
        settings = MagicMock()
        settings.regime_aware_enabled = True
        settings.yield_curve_enabled = True
        settings.regime_threshold_max_multiplier = 2.0
        for name, value in overrides.items():
            setattr(settings, name, value)
        return settings

    @staticmethod
    def _macro(level: MacroAlertLevel) -> MagicMock:
        report = MagicMock()
        report.overall_alert = level
        return report

    def test_inversion_alone_stays_neutral(self) -> None:
        # Sama inwersja (makro NORMAL) = 1 głos ELEVATED → NEUTRAL, nie RISK_OFF.
        # To mitygacja wielomiesięcznych inwersji: głos, nie weto.
        inverted = YieldCurveSnapshot(ten_year=3.0, two_year=4.0)

        context, multiplier = main_agent._resolve_regime(
            self._settings(), self._macro(MacroAlertLevel.NORMAL), inverted
        )

        assert multiplier == 1.0
        assert "NEUTRAL" in context.upper() or context != ""

    def test_inversion_plus_elevated_macro_triggers_risk_off(self) -> None:
        # Dwa głosy ELEVATED (makro + krzywa) → RISK_OFF → próg zacieśniony.
        inverted = YieldCurveSnapshot(ten_year=3.0, two_year=4.0)

        _, multiplier = main_agent._resolve_regime(
            self._settings(), self._macro(MacroAlertLevel.ELEVATED), inverted
        )

        assert multiplier > 1.0

    def test_elevated_macro_alone_stays_neutral(self) -> None:
        # Kontrola: bez inwersji ten sam alert makro daje tylko 1 głos.
        normal_curve = YieldCurveSnapshot(ten_year=4.5, two_year=3.0)

        _, multiplier = main_agent._resolve_regime(
            self._settings(), self._macro(MacroAlertLevel.ELEVATED), normal_curve
        )

        assert multiplier == 1.0

    def test_yield_curve_disabled_casts_no_vote(self) -> None:
        inverted = YieldCurveSnapshot(ten_year=3.0, two_year=4.0)
        settings = self._settings(yield_curve_enabled=False)

        _, multiplier = main_agent._resolve_regime(
            settings, self._macro(MacroAlertLevel.ELEVATED), inverted
        )

        assert multiplier == 1.0

    def test_missing_snapshot_casts_no_vote(self) -> None:
        _, multiplier = main_agent._resolve_regime(
            self._settings(), self._macro(MacroAlertLevel.ELEVATED), None
        )

        assert multiplier == 1.0

    def test_regime_off_returns_neutral_defaults(self) -> None:
        settings = self._settings(regime_aware_enabled=False)

        macro = self._macro(MacroAlertLevel.CRITICAL)

        assert main_agent._resolve_regime(settings, macro, None) == ("", 1.0)


class TestResolveRegimeMacroSeriesVotes:
    """DAT-22 — percentyl VIX z roku i zmiana spreadu HY wchodzą do RegimeDetector
    jako głosy, tylko z `macro_series_enabled` (Settings jako MagicMock: brak
    snapshotu nie może głosować nawet przy prawdziwej fladze)."""

    @staticmethod
    def _settings(**overrides: object) -> MagicMock:
        settings = MagicMock()
        settings.regime_aware_enabled = True
        settings.yield_curve_enabled = False
        settings.macro_series_enabled = True
        settings.regime_threshold_max_multiplier = 2.0
        for name, value in overrides.items():
            setattr(settings, name, value)
        return settings

    @staticmethod
    def _stressed_vix() -> MacroSeriesSnapshot:
        start = date(2025, 10, 1)
        values = [15.0 + (i % 7) for i in range(250)] + [35.0]
        return MacroSeriesSnapshot(
            vix=tuple(
                SeriesObservation(start + timedelta(days=i), v) for i, v in enumerate(values)
            )
        )

    @staticmethod
    def _macro(level: MacroAlertLevel) -> MagicMock:
        report = MagicMock()
        report.overall_alert = level
        return report

    def test_vix_in_the_top_decile_forces_risk_off(self) -> None:
        _, multiplier = main_agent._resolve_regime(
            self._settings(),
            self._macro(MacroAlertLevel.NORMAL),
            None,
            macro_series=self._stressed_vix(),
        )

        assert multiplier == 2.0

    def test_flag_off_casts_no_macro_vote(self) -> None:
        _, multiplier = main_agent._resolve_regime(
            self._settings(macro_series_enabled=False),
            self._macro(MacroAlertLevel.NORMAL),
            None,
            macro_series=self._stressed_vix(),
        )

        assert multiplier == 1.0

    def test_missing_snapshot_casts_no_vote(self) -> None:
        _, multiplier = main_agent._resolve_regime(
            self._settings(), self._macro(MacroAlertLevel.NORMAL), None
        )

        assert multiplier == 1.0


class TestMacroSeriesWiring:
    """DAT-22 — port FRED tylko z flagą i kluczem; snapshot cyklu trafia do reżimu
    i do fazy cech rynkowych."""

    def test_port_is_null_without_the_flag(self, settings) -> None:
        from src.infrastructure.adapters.fred_client import NullMacroSeriesAdapter

        settings.fred_api_key = "fred-key"

        assert isinstance(main_agent.build_macro_series_port(settings), NullMacroSeriesAdapter)

    def test_port_is_null_without_a_key(self, settings) -> None:
        from src.infrastructure.adapters.fred_client import NullMacroSeriesAdapter

        settings.macro_series_enabled = True
        settings.fred_api_key = None

        assert isinstance(main_agent.build_macro_series_port(settings), NullMacroSeriesAdapter)

    def test_port_is_fred_with_the_flag_and_a_key(self, settings) -> None:
        from src.infrastructure.adapters.fred_client import FredClient

        settings.macro_series_enabled = True
        settings.fred_api_key = "fred-key"

        assert isinstance(main_agent.build_macro_series_port(settings), FredClient)

    def test_cycle_snapshot_reaches_the_regime_and_the_feature_phase(
        self, settings, mock_external_clients, mocker
    ) -> None:
        snapshot = MacroSeriesSnapshot(nfci=SeriesObservation(date(2026, 10, 16), -0.5))
        port = MagicMock()
        port.fetch_macro_series.return_value = snapshot
        mocker.patch("main_agent.build_macro_series_port", return_value=port)
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        resolve = mocker.patch("main_agent._resolve_regime", return_value=("", 1.0))

        main_agent.main(settings)

        assert resolve.call_args.kwargs["macro_series"] is snapshot
        macro = fake_uc.prepare_market_features.call_args.kwargs["macro_features"]
        assert macro["nfci"] == -0.5
        assert macro["macro_observed_on"] == {"NFCI": "2026-10-16"}


class TestProcessOneSymbolEarningsGate:
    """#6 — mnożnik earnings dociera do grafu, a płatny port AV jest wołany
    dokładnie RAZ na symbol (prefetch + wstrzyknięcie do enrich)."""

    @staticmethod
    def _alpha(event: object | None) -> MagicMock:
        alpha = MagicMock()
        alpha.fetch_earnings.return_value = event
        alpha.enrich.return_value = None
        return alpha

    @staticmethod
    def _use_case() -> MagicMock:
        use_case = MagicMock()
        use_case.run.return_value = {"status": "ignored", "delta": "0.001"}
        return use_case

    def _run(self, alpha: MagicMock, use_case: MagicMock) -> None:
        main_agent._process_one_symbol(
            "AAPL",
            peer_ctx=(),
            use_case=use_case,
            crypto_set=set(),
            etf_set=set(),
            regime_context="",
            regime_multiplier=1.0,
            collect_alpha=True,
            alpha_use_case=alpha,
            earnings_gate_enabled=True,
        )

    def test_gate_disabled_never_touches_the_earnings_port(self) -> None:
        # Wsteczna kompatybilność: z wyłączonym kalendarzem ścieżka jest
        # bit-w-bit taka jak przed #6 — zero prefetchu, `enrich(symbol)` gołe.
        alpha = self._alpha(EarningsEvent("AAPL", days_until=1))
        use_case = self._use_case()

        main_agent._process_one_symbol(
            "AAPL",
            peer_ctx=(),
            use_case=use_case,
            crypto_set=set(),
            etf_set=set(),
            regime_context="",
            regime_multiplier=1.0,
            collect_alpha=True,
            alpha_use_case=alpha,
        )

        alpha.fetch_earnings.assert_not_called()
        alpha.enrich.assert_called_once_with("AAPL")
        assert use_case.run.call_args.kwargs["earnings_multiplier"] == 1.0

    def test_imminent_earnings_tighten_the_gate(self) -> None:
        alpha = self._alpha(EarningsEvent("AAPL", days_until=1))
        use_case = self._use_case()

        self._run(alpha, use_case)

        assert use_case.run.call_args.kwargs["earnings_multiplier"] == 1.5

    def test_far_earnings_leave_the_gate_untouched(self) -> None:
        alpha = self._alpha(EarningsEvent("AAPL", days_until=60))
        use_case = self._use_case()

        self._run(alpha, use_case)

        assert use_case.run.call_args.kwargs["earnings_multiplier"] == 1.0

    def test_no_earnings_data_leaves_the_gate_untouched(self) -> None:
        alpha = self._alpha(None)
        use_case = self._use_case()

        self._run(alpha, use_case)

        assert use_case.run.call_args.kwargs["earnings_multiplier"] == 1.0

    def test_earnings_port_is_paid_for_only_once(self) -> None:
        # FinOps: bramka nie ma prawa dołożyć ani jednego płatnego wywołania.
        event = EarningsEvent("AAPL", days_until=1)
        alpha = self._alpha(event)

        self._run(alpha, self._use_case())

        alpha.fetch_earnings.assert_called_once_with("AAPL")
        assert alpha.enrich.call_args.kwargs["earnings_prefetched"] is True
        assert alpha.enrich.call_args.kwargs["earnings"] == event


class TestBuildSignalCalibration:
    """#9 — kubełki pod ranking sygnałów są niezależne od flagi sekcji
    Track Record i degradują gracefully."""

    @staticmethod
    def _settings(enabled: bool) -> MagicMock:
        settings = MagicMock()
        settings.calibrated_confidence_enabled = enabled
        settings.track_record_days = 30
        return settings

    def test_flag_off_reads_nothing(self) -> None:
        repo = MagicMock(spec=RepositoryPort)

        assert main_agent._build_signal_calibration(self._settings(False), repo) is None
        repo.get_resolved_for_calibration.assert_not_called()

    def test_flag_on_builds_buckets_from_resolved_samples(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = [
            {"confidence_score": 0.85, "is_trend_correct": True},
            {"confidence_score": 0.85, "is_trend_correct": False},
        ]

        buckets = main_agent._build_signal_calibration(self._settings(True), repo)

        assert buckets
        repo.get_resolved_for_calibration.assert_called_once_with(30)

    def test_repository_error_degrades_to_raw_confidence(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.side_effect = Exception("supabase down")

        assert main_agent._build_signal_calibration(self._settings(True), repo) is None

    def test_no_usable_samples_yields_no_buckets(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = [
            {"confidence_score": None, "is_trend_correct": True},
        ]

        assert main_agent._build_signal_calibration(self._settings(True), repo) is None


class TestDynamicSubject:
    """#1 — temat maila niesie najmocniejszy sygnał dnia zamiast suchych
    liczników. Liczony nad PEŁNYMI danymi cyklu, bo subject jest wspólny
    dla wszystkich subskrybentów (sekcje są re-slicowane, temat nie)."""

    def test_headline_replaces_the_counter_subject(self) -> None:
        item = LeadItem(icon="🚨", headline="NVDA -4.2%, rada PODZIELONA", detail="")

        subject = main_agent._build_subject(
            lead_items=[item],
            analyzed=3,
            failures=0,
            started_at=datetime(2026, 5, 14, 9, 30, tzinfo=UTC),
            prefix="",
        )

        assert "NVDA -4.2%, rada PODZIELONA" in subject

    def test_falls_back_to_counters_without_a_lead(self) -> None:
        subject = main_agent._build_subject(
            lead_items=[],
            analyzed=3,
            failures=1,
            started_at=datetime(2026, 5, 14, 9, 30, tzinfo=UTC),
            prefix="",
        )

        assert "3 predykcji" in subject
        assert "1 błędów" in subject

    def test_critical_quota_prefix_survives_the_headline(self) -> None:
        # Prefix [QUOTA] to sygnał operacyjny — nie może zniknąć pod nagłówkiem.
        item = LeadItem(icon="🚨", headline="NVDA -4.2%", detail="")

        subject = main_agent._build_subject(
            lead_items=[item],
            analyzed=1,
            failures=0,
            started_at=datetime(2026, 5, 14, 9, 30, tzinfo=UTC),
            prefix="⚠️ [QUOTA] ",
        )

        assert subject.startswith("⚠️ [QUOTA] ")
        assert "NVDA -4.2%" in subject


class TestPolishNumbersInSubjectAndPush:
    """MAIL-58 etap 1: temat maila i push niosą headline leadu z polskim zapisem
    liczby („-4,2%”), a nie angielskim („-4.2%”)."""

    def test_email_and_push_subjects_use_a_decimal_comma(self, settings, mocker) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.council import InvestorOpinion

        settings.messenger_digest_enabled = True
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch("main_agent._build_user_portfolio", return_value=None)
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        mocker.patch("main_agent._publish_public_scorecard")
        push = MagicMock()
        mocker.patch("main_agent.build_push_notifier", return_value=push)
        email = MagicMock()
        split = CouncilVerdict(
            final_recommendation="HOLD", consensus_strength=0.5, summary="1 KUP / 1 SPRZEDAJ",
            dissenting_views=(),
            investor_opinions=tuple(
                InvestorOpinion(investor_name=name, recommendation=rec,
                                confidence=0.8, reasoning="x", key_factors=())
                for name, rec in (("A", "BUY"), ("B", "SELL"))
            ),
        )

        main_agent._dispatch_reports(
            results=[
                main_agent.SymbolResult(
                    symbol="NVDA", status="saved", delta=Decimal("-0.042"),
                    current_price=Decimal("180"), council_verdict=split,
                ),
            ],
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=MagicMock(spec=RepositoryPort),
            quota_monitor=QuotaMonitor(),
            notifier=email,
            subscribers=[],
            all_symbols=["NVDA"],
            context=main_agent.ReportContext(),
        )

        assert "NVDA -4,2%, rada PODZIELONA" in email.send_report.call_args.kwargs["subject"]
        assert "NVDA -4,2%, rada PODZIELONA" in push.send_report.call_args.kwargs["subject"]


class TestMessengerDigestSplit:
    """#5 — push (Telegram/Slack) dostaje 5-linijkowy skrót, nie pełny raport.

    Realny failure mode: pełny plain-text przekracza limit 4096 znaków Telegrama,
    API zwraca 400 i push ginie po cichu. Dodatkowo gałąź U2 (fan-out per
    subskrybent) w ogóle pomijała `notifier.send_report`, więc push nie szedł.
    """

    @staticmethod
    def _push_settings(settings: Settings, enabled: bool) -> Settings:
        settings.notifications_enabled = True
        settings.resend_api_key = "re_test"
        settings.digest_to_email = "you@example.com"
        settings.telegram_bot_token = "tg-token"
        settings.telegram_chat_id = "42"
        settings.messenger_digest_enabled = enabled
        return settings

    def test_flag_on_keeps_push_out_of_the_email_notifier(self, settings) -> None:
        from src.infrastructure.adapters.resend_notifier import ResendNotifier

        notifier = main_agent.build_notifier(self._push_settings(settings, True))

        # Sam Resend — push idzie osobnym kanałem, ze skróconą treścią.
        assert isinstance(notifier, ResendNotifier)

    def test_flag_off_preserves_the_legacy_composite(self, settings) -> None:
        from src.infrastructure.adapters.composite_notifier import CompositeNotifier

        notifier = main_agent.build_notifier(self._push_settings(settings, False))

        assert isinstance(notifier, CompositeNotifier)

    def test_build_push_notifier_is_null_without_secrets(self, settings) -> None:
        from src.infrastructure.adapters.resend_notifier import NullNotifier

        settings.telegram_bot_token = None
        settings.telegram_chat_id = None
        settings.slack_webhook_url = None

        assert isinstance(main_agent.build_push_notifier(settings), NullNotifier)

    def test_push_digest_is_sent_once_and_stays_under_the_limit(self) -> None:
        from src.application.report_messenger import MAX_DIGEST_CHARS

        push = MagicMock()
        results = [
            main_agent.SymbolResult(symbol=f"SYM{i}", status="ignored")
            for i in range(60)
        ]

        main_agent._dispatch_push_digest(
            push_notifier=push,
            results=results,
            started_at=datetime(2026, 5, 14, 9, 30, tzinfo=UTC),
            resolved=[],
            subject="StockAgent — cokolwiek",
        )

        push.send_report.assert_called_once()
        sent = push.send_report.call_args.kwargs["plain_text"]
        assert len(sent) <= MAX_DIGEST_CHARS

    def test_dispatch_sends_the_lead_and_the_ranked_signals_to_the_push(
        self, settings, mocker
    ) -> None:
        # MAIL-20: push mówi, co najważniejsze — lead i ranking sygnałów, nie
        # pierwsze symbole watchlisty w jej kolejności.
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.council import InvestorOpinion

        settings.messenger_digest_enabled = True
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch("main_agent._build_user_portfolio", return_value=None)
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        mocker.patch("main_agent._publish_public_scorecard")
        push = MagicMock()
        mocker.patch("main_agent.build_push_notifier", return_value=push)
        split = CouncilVerdict(
            final_recommendation="HOLD", consensus_strength=0.5, summary="1 KUP / 1 SPRZEDAJ",
            dissenting_views=(),
            investor_opinions=(
                InvestorOpinion(investor_name="A", recommendation="BUY",
                                confidence=0.8, reasoning="x", key_factors=()),
                InvestorOpinion(investor_name="B", recommendation="SELL",
                                confidence=0.8, reasoning="x", key_factors=()),
            ),
        )
        results = [
            main_agent.SymbolResult(
                symbol="AAPL", status="saved", delta=Decimal("0.03"), trend="BULLISH",
                confidence_score=0.5, current_price=Decimal("100"),
                target_price=Decimal("101"),
            ),
            main_agent.SymbolResult(
                symbol="NVDA", status="saved", delta=Decimal("-0.042"), trend="BEARISH",
                confidence_score=0.8, current_price=Decimal("100"),
                target_price=Decimal("95"), council_verdict=split,
            ),
        ]

        main_agent._dispatch_reports(
            results=results,
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=MagicMock(spec=RepositoryPort),
            quota_monitor=QuotaMonitor(),
            notifier=MagicMock(),
            subscribers=[],
            all_symbols=["AAPL", "NVDA"],
            context=main_agent.ReportContext(),
        )

        digest = push.send_report.call_args.kwargs["plain_text"]
        assert "⚖️ NVDA -4,2%, rada PODZIELONA" in digest
        signals = next(line for line in digest.splitlines() if line.startswith("Sygnały:"))
        assert signals.index("NVDA") < signals.index("AAPL")

    def test_push_failure_never_breaks_the_cycle(self) -> None:
        push = MagicMock()
        push.send_report.side_effect = Exception("Telegram 400")

        main_agent._dispatch_push_digest(
            push_notifier=push,
            results=[],
            started_at=datetime(2026, 5, 14, 9, 30, tzinfo=UTC),
            resolved=[],
            subject="StockAgent",
        )


class TestCycleMaturityWiring:
    """#4 — klasyfikacja dojrzałości cyklu z policzonych przyczyn pominięcia."""

    @staticmethod
    def _r(symbol: str, status: str, reason: SkipReason | None = None) -> "main_agent.SymbolResult":
        return main_agent.SymbolResult(symbol=symbol, status=status, skip_reason=reason)

    def test_all_cold_start_is_first_run(self) -> None:
        results = [self._r(f"S{i}", "ignored", SkipReason.COLD_START) for i in range(5)]

        assert main_agent._classify_cycle_maturity(results) is CycleMaturity.FIRST_RUN

    def test_mass_source_outage_is_not_first_run(self) -> None:
        # Największe ryzyko pozycji: awaria NIE może udawać powitania.
        results = [self._r(f"S{i}", "error") for i in range(5)]

        assert main_agent._classify_cycle_maturity(results) is CycleMaturity.STEADY_STATE

    def test_unsupported_tickers_do_not_make_a_first_run(self) -> None:
        results = [self._r(f"S{i}", "ignored", SkipReason.UNSUPPORTED_PRICE) for i in range(4)]

        assert main_agent._classify_cycle_maturity(results) is CycleMaturity.STEADY_STATE

    def test_unsupported_tickers_do_not_dilute_a_cold_start(self) -> None:
        # Nieobsługiwane cenowo symbole nigdy nie dostaną punktu odniesienia,
        # więc muszą wypaść z mianownika ZANIM domena policzy przewagę
        # cold-startu. Domieszka balastu nie może zepsuć powitania Dnia 1.
        results = [self._r(f"C{i}", "ignored", SkipReason.COLD_START) for i in range(4)]
        results += [self._r(f"U{i}", "ignored", SkipReason.UNSUPPORTED_PRICE) for i in range(10)]

        assert main_agent._classify_cycle_maturity(results) is CycleMaturity.FIRST_RUN

    def test_below_threshold_means_the_agent_already_has_a_baseline(self) -> None:
        results = [self._r("AAPL", "ignored", SkipReason.BELOW_THRESHOLD)]

        assert main_agent._classify_cycle_maturity(results) is CycleMaturity.STEADY_STATE

    def test_unsupported_result_carries_its_skip_reason(self) -> None:
        assert main_agent._ignored_result("CSPX.L").skip_reason is SkipReason.UNSUPPORTED_PRICE


class TestOnboardingSubject:
    """#4 — pierwszy cykl ma powitalny temat zamiast '0 predykcji, 0 błędów'."""

    def test_first_run_gets_a_welcome_subject(self) -> None:
        subject = main_agent._build_subject(
            lead_items=[],
            analyzed=0,
            failures=0,
            started_at=datetime(2026, 5, 14, 9, 30, tzinfo=UTC),
            prefix="",
            maturity=CycleMaturity.FIRST_RUN,
            analyzed_total=45,
        )

        assert "0 predykcji" not in subject
        assert "45" in subject

    def test_steady_state_keeps_the_counter_subject(self) -> None:
        subject = main_agent._build_subject(
            lead_items=[],
            analyzed=3,
            failures=0,
            started_at=datetime(2026, 5, 14, 9, 30, tzinfo=UTC),
            prefix="",
            maturity=CycleMaturity.STEADY_STATE,
            analyzed_total=45,
        )

        assert "3 predykcji" in subject


class TestModelScorecardPersistence:
    """#12 — każdy przebieg treningu zapisuje scorecard per symbol, best-effort."""

    def test_scorecard_saved_per_symbol(self, settings, mock_external_clients, mocker) -> None:
        settings.model_scorecard_enabled = True
        settings.symbols = ["AAPL", "MSFT"]
        fake_uc = MagicMock()
        fake_uc.run.return_value = {
            "status": "trained_successfully",
            "candidate_holdout_rmse": 0.02,
            "baseline_rmse": 0.03,
            "candidate_holdout_directional_hit_rate": 0.6,
            "n_folds": 7,
            "n_samples": 1000,
        }
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        repo = MagicMock(spec=RepositoryPort)
        mocker.patch("main_trainer.SupabaseRepository", return_value=repo)

        main_trainer.main(settings)

        saved = [c.args[0] for c in repo.save_model_scorecard.call_args_list]
        assert saved == ["AAPL", "MSFT"]

    def test_scorecard_write_failure_does_not_fail_the_run(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.model_scorecard_enabled = True
        settings.symbols = ["AAPL"]
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        repo = MagicMock(spec=RepositoryPort)
        repo.save_model_scorecard.side_effect = Exception("no table")
        mocker.patch("main_trainer.SupabaseRepository", return_value=repo)

        assert main_trainer.main(settings) == 0

    def test_flag_off_writes_nothing(self, settings, mock_external_clients, mocker) -> None:
        settings.model_scorecard_enabled = False
        settings.symbols = ["AAPL"]
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        repo = MagicMock(spec=RepositoryPort)
        mocker.patch("main_trainer.SupabaseRepository", return_value=repo)

        main_trainer.main(settings)

        repo.save_model_scorecard.assert_not_called()


class TestModelScorecardRunId:
    """ML-24 — main_trainer losuje jeden uuid4 na przebieg; niesie go każdy wiersz
    karty (per symbol i model wspólny), więc re-run tego samego dnia to osobny przebieg."""

    def test_one_run_id_for_every_scorecard_row_of_a_run(
        self, settings, mock_external_clients, mocker
    ) -> None:
        import uuid

        settings.model_scorecard_enabled = True
        settings.symbols = ["AAPL", "MSFT"]
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "evaluated_passed_gate"}
        fake_uc.run_pooled.return_value = {"status": "skipped"}
        build = mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        repo = MagicMock(spec=RepositoryPort)
        mocker.patch("main_trainer.SupabaseRepository", return_value=repo)

        main_trainer.main(settings)
        main_trainer.main(settings)

        first, second = (c.kwargs["run_id"] for c in build.call_args_list)
        saved = [c.kwargs["run_id"] for c in repo.save_model_scorecard.call_args_list]
        assert saved == [first, first, second, second]
        assert first != second
        assert str(uuid.UUID(first)) == first

    @pytest.mark.parametrize("enabled", [False, True])
    def test_trainer_repository_follows_the_run_id_flag(
        self, settings, mocker, enabled: bool
    ) -> None:
        supabase_cls = mocker.patch("main_trainer.SupabaseRepository")
        settings = settings.model_copy(update={"model_scorecard_run_id_enabled": enabled})

        main_trainer.build_repository(settings)

        assert supabase_cls.call_args.kwargs["scorecard_run_id_enabled"] is enabled

    @pytest.mark.parametrize("enabled", [False, True])
    def test_trainer_repository_follows_the_feature_stats_flag(
        self, settings, mocker, enabled: bool
    ) -> None:
        # ML-21: kolumna `feature_stats` (migracja 052) tylko za flagą.
        supabase_cls = mocker.patch("main_trainer.SupabaseRepository")
        settings = settings.model_copy(
            update={"model_scorecard_feature_stats_enabled": enabled}
        )

        main_trainer.build_repository(settings)

        assert supabase_cls.call_args.kwargs["scorecard_feature_stats_enabled"] is enabled

    def test_build_use_case_hands_the_run_id_to_the_use_case(self, settings, mocker) -> None:
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        use_case_cls = mocker.patch("main_trainer.TrainModelUseCase")

        main_trainer.build_use_case(settings, run_id="run-1")

        assert use_case_cls.call_args.kwargs["run_id"] == "run-1"


class TestModelScorecardWindow:
    """ML-15 — karta czyta 2 × `MODEL_STALE_AFTER_DAYS`, a nie `track_record_days`:
    przy oknie 30 dni odznaka STALE (po 21 dniach) była widoczna tylko w dniach 22–30."""

    def test_card_older_than_the_track_record_window_shows_as_stale(self, settings) -> None:
        from src.domain.model_scorecard import MODEL_STALE_AFTER_DAYS, is_stale

        now = datetime.now(UTC)
        trained_at = now - timedelta(days=35)
        settings.model_scorecard_enabled = True
        settings.track_record_days = 30
        row = {
            "symbol": "ALL",
            "status": "trained_successfully",
            "timestamp": trained_at.isoformat(),
        }
        repo = MagicMock(spec=RepositoryPort)
        # Okno odczytu jak w adapterze: wiersz wraca tylko, gdy mieści się w `days`.
        repo.get_recent_model_scorecards.side_effect = lambda days: [row] if days >= 35 else []

        view = main_agent._build_model_scorecard(settings, repo)

        assert view is not None
        assert is_stale(view.trained_at, now)
        repo.get_recent_model_scorecards.assert_called_once_with(2 * MODEL_STALE_AFTER_DAYS)


class TestConsensusShiftWiring:
    """#8 — flipy rady. Graceful w callerze, tłumienie stęchłych porównań."""

    @staticmethod
    def _verdict(rec: str, consensus: float = 0.8) -> CouncilVerdict:
        return CouncilVerdict(
            final_recommendation=rec,  # type: ignore[arg-type]
            consensus_strength=consensus,
            summary="",
            dissenting_views=(),
            investor_opinions=(),
        )

    def _result(self, rec: str) -> "main_agent.SymbolResult":
        return main_agent.SymbolResult(
            symbol="NVDA",
            status="saved",
            sentiment_score=0.1,
            council_verdict=self._verdict(rec),
        )

    def _repo(self, previous: object) -> MagicMock:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_previous_verdict.return_value = previous
        return repo

    def test_flip_is_detected(self) -> None:
        previous = PreviousVerdict(
            verdict=self._verdict("BUY"),
            sentiment_score=0.4,
            timestamp=datetime(2026, 7, 10, 6, 0, tzinfo=UTC),
        )
        shifts = main_agent._build_consensus_shifts(
            self._repo(previous),
            [self._result("SELL")],
            now=datetime(2026, 7, 10, 18, 0, tzinfo=UTC),
        )

        assert shifts
        assert shifts[0][0] == "NVDA"
        assert shifts[0][1].kind is ShiftKind.FLIP

    def test_stale_comparison_is_marked_not_dramatised(self) -> None:
        # Cron bywa zakomentowany — "wczoraj" potrafi być sprzed tygodnia.
        previous = PreviousVerdict(
            verdict=self._verdict("BUY"),
            sentiment_score=0.4,
            timestamp=datetime(2026, 7, 3, 6, 0, tzinfo=UTC),
        )
        shifts = main_agent._build_consensus_shifts(
            self._repo(previous),
            [self._result("SELL")],
            now=datetime(2026, 7, 10, 18, 0, tzinfo=UTC),
        )

        assert shifts[0][1].kind is ShiftKind.STALE_COMPARISON

    def test_repository_error_degrades_to_no_section(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_previous_verdict.side_effect = Exception("supabase down")

        shifts = main_agent._build_consensus_shifts(
            repo, [self._result("SELL")], now=datetime(2026, 7, 10, tzinfo=UTC)
        )

        assert shifts == []

    def test_symbol_without_current_verdict_is_skipped(self) -> None:
        # Symbol odcięty bramką volatility nie ma świeżego werdyktu —
        # nie wolno mu pokazywać starego flipu drugi raz.
        repo = self._repo(None)
        result = main_agent.SymbolResult(symbol="NVDA", status="ignored")

        now = datetime(2026, 7, 10, tzinfo=UTC)

        assert main_agent._build_consensus_shifts(repo, [result], now=now) == []
        repo.get_previous_verdict.assert_not_called()


class TestAlphaEnrichmentRunsBeforeTheGraph:
    """#14 — korekta fundamentalna: enrichment działał PO grafie, więc LLM
    i cechy XGBoost nigdy nie widziały sygnałów alfa. Musi biec PRZED."""

    @staticmethod
    def _alpha(signals: object) -> MagicMock:
        alpha = MagicMock()
        alpha.fetch_earnings.return_value = None
        alpha.enrich.return_value = signals
        return alpha

    @staticmethod
    def _use_case() -> MagicMock:
        use_case = MagicMock()
        use_case.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        return use_case

    def _run(self, alpha: MagicMock, use_case: MagicMock) -> None:
        main_agent._process_one_symbol(
            "AAPL",
            peer_ctx=(),
            use_case=use_case,
            crypto_set=set(),
            etf_set=set(),
            regime_context="",
            regime_multiplier=1.0,
            collect_alpha=True,
            alpha_use_case=alpha,
            alpha_fusion_enabled=True,
        )

    def test_enrich_is_called_before_the_graph_runs(self) -> None:
        order: list[str] = []
        signals = AlphaSignals(symbol="AAPL")
        alpha = self._alpha(signals)
        alpha.enrich.side_effect = lambda *a, **k: (order.append("enrich"), signals)[1]
        use_case = self._use_case()
        use_case.run.side_effect = lambda *a, **k: (
            order.append("graph"),
            {"status": "ignored", "delta": Decimal("0")},
        )[1]

        self._run(alpha, use_case)

        assert order == ["enrich", "graph"]

    def test_fusion_score_is_injected_into_the_graph(self) -> None:
        signals = AlphaSignals(
            symbol="AAPL",
            insider=InsiderFlowSnapshot(
                "AAPL", net_shares=50_000.0, buy_count=5, sell_count=0, window_days=90
            ),
        )
        use_case = self._use_case()

        self._run(self._alpha(signals), use_case)

        fusion = use_case.run.call_args.kwargs["alpha_fusion"]
        assert fusion is not None
        assert fusion.score > 0

    def test_no_alpha_collection_means_no_fusion(self) -> None:
        use_case = self._use_case()
        alpha = self._alpha(None)

        main_agent._process_one_symbol(
            "AAPL",
            peer_ctx=(),
            use_case=use_case,
            crypto_set=set(),
            etf_set=set(),
            regime_context="",
            regime_multiplier=1.0,
            collect_alpha=False,
            alpha_use_case=alpha,
        )

        assert use_case.run.call_args.kwargs["alpha_fusion"] is None
        alpha.enrich.assert_not_called()


def test_alpha_fusion_flag_off_leaves_the_graph_untouched() -> None:
    """#14 — z flagą off graf dostaje `alpha_fusion=None`, więc prompt LLM
    i cechy ML są identyczne jak przed wprowadzeniem fuzji."""
    alpha = MagicMock()
    alpha.fetch_earnings.return_value = None
    alpha.enrich.return_value = AlphaSignals(symbol="AAPL")
    use_case = MagicMock()
    use_case.run.return_value = {"status": "ignored", "delta": Decimal("0")}

    main_agent._process_one_symbol(
        "AAPL",
        peer_ctx=(),
        use_case=use_case,
        crypto_set=set(),
        etf_set=set(),
        regime_context="",
        regime_multiplier=1.0,
        collect_alpha=True,
        alpha_use_case=alpha,
    )

    assert use_case.run.call_args.kwargs["alpha_fusion"] is None


class TestWeeklyRecapWiring:
    """#17 — niedzielny recap. Slow Loop chodzi co tydzień i dotąd nie wysyłał
    użytkownikowi NIC. Koszt: dokładnie 1 wywołanie LLM tygodniowo."""

    def _trainer(self, settings, mocker, recap_status: str) -> tuple[MagicMock, MagicMock]:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        recap_uc = MagicMock()
        recap_uc.run.return_value = (
            {"status": recap_status, "recap": MagicMock(), "narrative": "opowieść"}
            if recap_status == "recap_ready"
            else {"status": recap_status, "n_closed": 2}
        )
        mocker.patch("main_trainer.WeeklyRecapUseCase", return_value=recap_uc)
        notifier = MagicMock()
        mocker.patch("main_trainer.build_notifier", return_value=notifier)
        mocker.patch("main_agent.build_llm_adapter", return_value=MagicMock())
        # Renderery są testowane osobno; tu sprawdzamy wyłącznie orkiestrację.
        mocker.patch("main_trainer.render_weekly_recap_html", return_value="<h2>ok</h2>")
        mocker.patch("main_trainer.render_weekly_recap_text", return_value="ok")
        return recap_uc, notifier

    def test_recap_email_is_sent_when_ready(self, settings, mock_external_clients, mocker) -> None:
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        _, notifier = self._trainer(settings, mocker, "recap_ready")

        main_trainer.main(settings)

        notifier.send_report.assert_called_once()

    def test_below_threshold_no_email_goes_out(
        self, settings, mock_external_clients, mocker
    ) -> None:
        # Twardy próg próbki: przy chudym tygodniu recap robiłby dramat z szumu.
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        _, notifier = self._trainer(settings, mocker, "skipped_below_threshold")

        main_trainer.main(settings)

        notifier.send_report.assert_not_called()

    def test_recap_knows_the_crypto_symbols(
        self, settings, mock_external_clients, mocker
    ) -> None:
        # Decyzja 2026-10-19 (a): pasmo FLAT recapu zależy od klasy aktywa.
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        settings.crypto_symbols = ["BTC", "ETH"]
        self._trainer(settings, mocker, "recap_ready")

        main_trainer.main(settings)

        call = main_trainer.WeeklyRecapUseCase.call_args
        assert list(call.kwargs["crypto_symbols"]) == ["BTC", "ETH"]

    @pytest.mark.parametrize("enabled", [False, True])
    def test_recap_gets_the_council_redundancy_flag(
        self, settings, mock_external_clients, mocker, enabled: bool
    ) -> None:
        # WIA-45: linia redundancji w recapie tygodniowym tylko za flagą.
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        settings.council_redundancy_enabled = enabled
        self._trainer(settings, mocker, "recap_ready")

        main_trainer.main(settings)

        call = main_trainer.WeeklyRecapUseCase.call_args
        assert call.kwargs["council_redundancy_enabled"] is enabled

    def test_flag_off_never_builds_the_recap(self, settings, mock_external_clients, mocker) -> None:
        settings.weekly_recap_enabled = False
        settings.symbols = ["AAPL"]
        recap_uc, notifier = self._trainer(settings, mocker, "recap_ready")

        main_trainer.main(settings)

        recap_uc.run.assert_not_called()
        notifier.send_report.assert_not_called()

    def test_recap_failure_does_not_fail_the_slow_loop(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        recap_uc, _ = self._trainer(settings, mocker, "recap_ready")
        recap_uc.run.side_effect = Exception("LLM down")

        assert main_trainer.main(settings) == 0

    def test_recap_that_could_not_be_sent_is_annotated_and_persisted(
        self, settings, mock_external_clients, mocker, capsys
    ) -> None:
        # NIEZ-12: niedostarczony raport widać w UI Actions, a exit code zostaje 0.
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        _, notifier = self._trainer(settings, mocker, "recap_ready")
        notifier.send_report.side_effect = RuntimeError("Resend 403")
        repository = MagicMock(spec=RepositoryPort)
        mocker.patch("main_trainer.SupabaseRepository", return_value=repository)

        exit_code = main_trainer.main(settings)

        assert exit_code == 0
        assert UNDELIVERED_ANNOTATION in capsys.readouterr().out.splitlines()
        alerts = [c.args[0] for c in repository.save_quota_alert.call_args_list]
        assert [a.severity for a in alerts if "nie dotarł" in a.message] == [
            QuotaSeverity.CRITICAL
        ]

    def test_recap_without_a_failure_leaves_no_annotation(
        self, settings, mock_external_clients, mocker, capsys
    ) -> None:
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        self._trainer(settings, mocker, "recap_ready")

        main_trainer.main(settings)

        assert "::error" not in capsys.readouterr().out

    def test_a_skipped_recap_is_not_an_undelivered_report(
        self, settings, mock_external_clients, mocker, capsys
    ) -> None:
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        self._trainer(settings, mocker, "skipped_below_threshold")

        main_trainer.main(settings)

        assert "::error" not in capsys.readouterr().out

    def test_deterministic_recap_sent_when_llm_adapter_cannot_be_built(
        self, settings, mock_external_clients, mocker
    ) -> None:
        # NIEZ-10: brak SDK albo klucza wywracał cały recap przy BUDOWIE adaptera,
        # choć use case umie wysłać recap bez narracji, gdy samo wywołanie LLM padnie.
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        recap_uc, notifier = self._trainer(settings, mocker, "recap_ready")
        mocker.patch(
            "main_agent.build_llm_adapter",
            side_effect=ModuleNotFoundError("No module named 'anthropic'"),
        )

        main_trainer.main(settings)

        recap_uc.run.assert_called_once()
        notifier.send_report.assert_called_once()

    def test_resend_alert_raised_during_recap_send_is_persisted(
        self, settings, mock_external_clients, mocker
    ) -> None:
        # NIEZ-03: alerty kwot zapisujemy przed recapem, więc alert z samej wysyłki
        # recapu (np. Resend 429) przepadał zamiast trafić do poniedziałkowego bannera.
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        _, notifier = self._trainer(settings, mocker, "recap_ready")
        repository = MagicMock(spec=RepositoryPort)
        mocker.patch("main_trainer.SupabaseRepository", return_value=repository)
        alert = _quota_alert()
        monitors: list[object] = []

        def build_notifier(_settings: object, quota_monitor: object = None) -> MagicMock:
            monitors.append(quota_monitor)
            return notifier

        mocker.patch("main_trainer.build_notifier", side_effect=build_notifier)
        notifier.send_report.side_effect = lambda **_: monitors[-1].record(alert)

        main_trainer.main(settings)

        repository.save_quota_alert.assert_any_call(alert)

    def test_calibration_judge_runs_when_llm_adapter_cannot_be_built(
        self, settings, mock_external_clients, mocker
    ) -> None:
        # Ten sam wzorzec co recap: sędzia zapisuje kalibrację bez wniosku LLM.
        settings.calibration_judge_enabled = True
        settings.weekly_recap_enabled = False
        settings.symbols = ["AAPL"]
        self._trainer(settings, mocker, "recap_ready")
        judge_uc = MagicMock()
        judge_uc.run.return_value = {"status": "calibrated"}
        mocker.patch(
            "src.application.use_cases.calibrate_confidence.CalibrateConfidenceUseCase",
            return_value=judge_uc,
        )
        mocker.patch(
            "main_agent.build_llm_adapter",
            side_effect=ModuleNotFoundError("No module named 'anthropic'"),
        )

        main_trainer.main(settings)

        judge_uc.run.assert_called_once()


class TestModelScorecardCoverage:
    """DAT-16: karta kondycji niesie pokrycie danych okna z symbolami bez rozliczeń."""

    def test_scorecard_view_carries_the_data_coverage(self, settings, mocker) -> None:
        from src.application.report_model_scorecard import ModelScorecardView
        from src.domain.model_scorecard import summarize_run

        settings.model_scorecard_enabled = True
        settings.symbols = ["AAPL", "SSNLF"]
        settings.crypto_symbols = []
        view = ModelScorecardView(
            summary=summarize_run(["trained_successfully"]),
            trained_at=datetime(2026, 10, 18, tzinfo=UTC),
        )
        mocker.patch(f"{ASSEMBLY}.build_scorecard_view", return_value=view)
        repository = MagicMock(spec=RepositoryPort)
        repository.get_price_history.return_value = []
        repository.get_resolved_predictions.return_value = [{"symbol": "AAPL"}]

        scorecard = main_agent._build_model_scorecard(settings, repository)

        assert scorecard is not None and scorecard.coverage is not None
        assert scorecard.coverage.unsettled_symbols == ("SSNLF",)

    def test_coverage_failure_keeps_the_scorecard(self, settings, mocker) -> None:
        from src.application.report_model_scorecard import ModelScorecardView
        from src.domain.model_scorecard import summarize_run

        settings.model_scorecard_enabled = True
        view = ModelScorecardView(
            summary=summarize_run(["trained_successfully"]),
            trained_at=datetime(2026, 10, 18, tzinfo=UTC),
        )
        mocker.patch(f"{ASSEMBLY}.build_scorecard_view", return_value=view)
        repository = MagicMock(spec=RepositoryPort)
        repository.get_resolved_predictions.side_effect = RuntimeError("supabase down")

        scorecard = main_agent._build_model_scorecard(settings, repository)

        assert scorecard == view


class TestPublicScorecard:
    """#10 — publiczny scorecard. DRUGA instancja publishera (nie ta od
    prywatnego digestu!) i metryki liczone niezależnie od flag track-recordu."""

    @staticmethod
    def _settings(enabled: bool, **overrides: object) -> MagicMock:
        settings = MagicMock()
        settings.public_scorecard_enabled = enabled
        settings.public_scorecard_path = "public/scorecard/index.html"
        settings.track_record_days = 30
        # Flagi sekcji track-recordu w mailu — scorecard NIE MOŻE od nich zależeć.
        settings.equity_curve_enabled = False
        settings.calibration_curve_enabled = False
        for name, value in overrides.items():
            setattr(settings, name, value)
        return settings

    @staticmethod
    def _repo() -> MagicMock:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.return_value = [
            {
                "symbol": "NVDA",
                "predicted_trend": "BULLISH",
                "is_trend_correct": True,
                "price_at_prediction": "100",
                "actual_price_after_12h": "110",
            }
        ]
        repo.get_resolved_for_calibration.return_value = [
            {"confidence_score": 0.8, "is_trend_correct": True},
        ]
        return repo

    def test_flag_off_publishes_nothing(self) -> None:
        publisher = MagicMock()

        main_agent._publish_public_scorecard(
            self._settings(False), self._repo(), publisher
        )

        publisher.publish.assert_not_called()

    def test_metrics_are_computed_despite_track_record_flags_being_off(self) -> None:
        # Korekta weryfikatora: scorecard liczy z resolved/cal_rows niezależnie
        # od `equity_curve_enabled` / `calibration_curve_enabled`.
        publisher = MagicMock()

        main_agent._publish_public_scorecard(
            self._settings(True), self._repo(), publisher
        )

        publisher.publish.assert_called_once()

    def test_published_page_never_leaks_a_watchlist_symbol(self) -> None:
        publisher = MagicMock()

        main_agent._publish_public_scorecard(
            self._settings(True), self._repo(), publisher
        )

        page = publisher.publish.call_args.args[0]
        assert "NVDA" not in page

    def test_page_states_data_coverage_without_naming_symbols(self) -> None:
        # DAT-16: luka i symbol bez rozliczeń są na stronie tylko jako liczby.
        settings = self._settings(True)
        settings.symbols = ["NVDA", "SSNLF"]
        settings.crypto_symbols = []
        repo = self._repo()
        repo.get_price_history.return_value = []
        publisher = MagicMock()

        main_agent._publish_public_scorecard(settings, repo, publisher)

        page = publisher.publish.call_args.args[0]
        assert "Pokrycie danych: 0% sesji" in page
        assert "1 symbol bez rozliczeń w oknie" in page
        assert "SSNLF" not in page

    def test_repository_error_never_breaks_the_cycle(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.side_effect = Exception("supabase down")

        main_agent._publish_public_scorecard(
            self._settings(True), repo, MagicMock()
        )


class TestCryptoInTrainingLoop:
    """Bug: main_trainer trenował tylko settings.symbols, więc BTC/ETH NIGDY
    nie przechodziły retrainu — model krypto zostawał na zawsze cold-start."""

    def test_crypto_symbols_are_trained_too(self, settings, mock_external_clients, mocker) -> None:
        settings.symbols = ["AAPL"]
        settings.crypto_symbols = ["BTC", "ETH"]
        settings.model_scorecard_enabled = False
        settings.weekly_recap_enabled = False
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))

        main_trainer.main(settings)

        trained = [c.args[0] for c in fake_uc.run.call_args_list]
        assert trained == ["AAPL", "BTC", "ETH"]

    def test_no_duplicate_when_a_symbol_is_in_both_lists(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.symbols = ["AAPL", "BTC"]
        settings.crypto_symbols = ["BTC"]
        settings.model_scorecard_enabled = False
        settings.weekly_recap_enabled = False
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))

        main_trainer.main(settings)

        trained = [c.args[0] for c in fake_uc.run.call_args_list]
        assert trained == ["AAPL", "BTC"]


def _quota_alert(source: str = "Resend") -> QuotaAlert:
    return QuotaAlert(
        source=source,
        severity=QuotaSeverity.CRITICAL,
        message=f"{source} returned 503",
        action="sprawdź dostawcę",
        occurred_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
    )


class TestDispatchReportsDelivery:
    """NIEZ-02/MAIL-02/NIEZ-03 — awaria wysyłki maila nie może po cichu wyciąć
    dalszych kroków (scorecard, push), a alert z samej wysyłki musi dotrwać do
    następnego raportu."""

    @staticmethod
    def _dispatch(settings: Settings, mocker, *, notifier, quota_monitor, repository) -> MagicMock:
        settings.messenger_digest_enabled = True
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch("main_agent._build_user_portfolio", return_value=None)
        mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        publish_scorecard = mocker.patch("main_agent._publish_public_scorecard")
        push = MagicMock()
        mocker.patch("main_agent.build_push_notifier", return_value=push)
        main_agent._dispatch_reports(
            results=[],
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=repository,
            quota_monitor=quota_monitor,
            notifier=notifier,
            subscribers=[],
            all_symbols=["AAPL"],
            context=main_agent.ReportContext(),
        )
        push.publish_scorecard = publish_scorecard
        return push

    def test_push_digest_still_sent_when_resend_fails(self, settings, mocker) -> None:
        from src.application.quota_monitor import QuotaMonitor

        notifier = MagicMock()
        notifier.send_report.side_effect = RuntimeError("Resend 503")

        push = self._dispatch(
            settings, mocker, notifier=notifier, quota_monitor=QuotaMonitor(),
            repository=MagicMock(spec=RepositoryPort),
        )

        push.send_report.assert_called_once()
        push.publish_scorecard.assert_called_once()

    def test_push_digest_says_the_email_was_not_delivered(self, settings, mocker) -> None:
        from src.application.quota_monitor import QuotaMonitor

        notifier = MagicMock()
        notifier.send_report.side_effect = RuntimeError("Resend 503")

        push = self._dispatch(
            settings, mocker, notifier=notifier, quota_monitor=QuotaMonitor(),
            repository=MagicMock(spec=RepositoryPort),
        )

        assert "nie został dostarczony" in push.send_report.call_args.kwargs["plain_text"]

    def test_resend_failure_alert_is_persisted_for_next_report(self, settings, mocker) -> None:
        from src.application.quota_monitor import QuotaMonitor

        monitor = QuotaMonitor()
        alert = _quota_alert()
        repository = MagicMock(spec=RepositoryPort)
        notifier = MagicMock()

        def failing_send(**_: object) -> None:
            monitor.record(alert)
            raise RuntimeError("Resend 503")

        notifier.send_report.side_effect = failing_send

        self._dispatch(
            settings, mocker, notifier=notifier, quota_monitor=monitor, repository=repository
        )

        repository.save_quota_alert.assert_called_once_with(alert)


class TestQuotaAlertBannerLookback:
    def test_banner_reaches_back_past_a_weekly_slow_loop_alert(self) -> None:
        # NIEZ-01: niedzielny CRITICAL Alpha Vantage ze Slow Loopa ma być widoczny
        # w najbliższym mailu, a 24 h okna gubiło go już w poniedziałek rano.
        from src.application.quota_monitor import QuotaMonitor

        repository = MagicMock(spec=RepositoryPort)
        repository.get_recently_resolved_predictions.return_value = []

        main_agent._fetch_report_context(repository, QuotaMonitor(), ["AAPL"])

        hours = repository.get_recent_quota_alerts.call_args.kwargs["hours"]
        assert hours >= 7 * 24 + 3


class TestPooledTrainingLoop:
    """ML-13 (decyzja 4): po ewaluacji per symbol Slow Loop trenuje JEDEN wspólny
    model na wszystkich symbolach — jeden zapis wag zamiast last-writer-wins."""

    def _trainer(self, settings, mocker) -> MagicMock:
        settings.symbols = ["AAPL", "BTC"]
        settings.crypto_symbols = ["BTC", "ETH"]
        settings.weekly_recap_enabled = False
        settings.calibration_judge_enabled = False
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped", "n_samples": 17}
        fake_uc.run_pooled.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        return fake_uc

    def test_pooled_model_trains_once_over_every_trainable_symbol(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.model_scorecard_enabled = True
        fake_uc = self._trainer(settings, mocker)

        assert main_trainer.main(settings) == 0

        fake_uc.run_pooled.assert_called_once_with(
            ["AAPL", "BTC", "ETH"], record_scorecard=True
        )

    def test_scorecard_flag_off_reaches_the_pooled_run(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.model_scorecard_enabled = False
        fake_uc = self._trainer(settings, mocker)

        main_trainer.main(settings)

        assert fake_uc.run_pooled.call_args.kwargs == {"record_scorecard": False}

    def test_pooled_training_failure_fails_the_run(
        self, settings, mock_external_clients, mocker
    ) -> None:
        fake_uc = self._trainer(settings, mocker)
        fake_uc.run_pooled.side_effect = RuntimeError("XGBoost boom")

        assert main_trainer.main(settings) == 1


class TestSnapshotModelsLoop:
    """ML-52: po modelu wspólnym Slow Loop ewaluuje dwa modele z `feature_snapshots`
    (bez zapisu wag). Pomocniczy krok: jego błąd nie zmienia exit code'u."""

    def _trainer(self, settings, mocker) -> MagicMock:
        settings.symbols = ["AAPL", "BTC"]
        settings.crypto_symbols = ["BTC", "ETH"]
        settings.weekly_recap_enabled = False
        settings.calibration_judge_enabled = False
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped", "n_samples": 17}
        fake_uc.run_pooled.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        return fake_uc

    @pytest.mark.parametrize("cards", [True, False])
    def test_snapshot_models_run_over_every_trainable_symbol(
        self, settings, mock_external_clients, mocker, cards
    ) -> None:
        settings.model_scorecard_enabled = cards
        fake_uc = self._trainer(settings, mocker)

        assert main_trainer.main(settings) == 0

        fake_uc.run_snapshot_models.assert_called_once_with(
            ["AAPL", "BTC", "ETH"], record_scorecard=cards
        )

    def test_snapshot_model_failure_does_not_fail_the_run(
        self, settings, mock_external_clients, mocker
    ) -> None:
        fake_uc = self._trainer(settings, mocker)
        fake_uc.run_snapshot_models.side_effect = RuntimeError("feature_snapshots missing")

        assert main_trainer.main(settings) == 0

    @pytest.mark.parametrize("enabled", [True, False])
    def test_snapshot_training_flag_reaches_the_use_case(
        self, settings, mock_external_clients, enabled
    ) -> None:
        settings.snapshot_training_enabled = enabled

        use_case = main_trainer.build_use_case(settings)

        assert use_case._snapshot_source is enabled

    @pytest.mark.parametrize("enabled", [True, False])
    def test_sigma_target_flag_reaches_the_trainer(
        self, settings, mock_external_clients, enabled
    ) -> None:
        settings.ml_sigma_target_enabled = enabled

        use_case = main_trainer.build_use_case(settings)

        assert use_case._sigma_target is enabled

    def test_use_case_reads_training_data_from_supabase(
        self, settings, mock_external_clients
    ) -> None:
        from src.application.ports import TrainingDataPort

        use_case = main_trainer.build_use_case(settings)

        assert isinstance(use_case._training_data, TrainingDataPort)


def test_backtest_log_reports_ic_next_to_rmse(settings, mock_external_clients, mocker, caplog):
    # ML-20: IC i jego t-stat obok RMSE, zawsze z liczbą foldów (4 foldy = mało).
    from src.domain.backtest import BacktestSummary

    fake_uc = MagicMock()
    fake_uc.run.return_value = BacktestSummary(
        n_folds=4,
        mean_candidate_rmse=0.03,
        mean_baseline_rmse=0.05,
        mean_hit_rate=0.6,
        improvement_pct=40.0,
        beats_baseline=True,
        verdict="model bije baseline o 40.0%",
        mean_ic=0.12,
        ic_std=0.05,
        ic_t_stat=4.8,
    )
    mocker.patch("main_backtest.BacktestUseCase", return_value=fake_uc)

    with caplog.at_level("INFO", logger="main_backtest"):
        main_backtest.main(settings)

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("AAPL"))
    assert "ic=0.120" in line
    assert "t=4.80" in line
    assert "folds=4" in line


def _same_day_rows() -> list[dict[str, object]]:
    """Dwie równoległe predykcje BULLISH +10% w jednym cyklu (dniu UTC)."""
    return [
        {
            "symbol": symbol,
            "predicted_trend": "BULLISH",
            "is_trend_correct": True,
            "price_at_prediction": "100",
            "actual_price_after_12h": "110",
            "timestamp": "2026-10-17T14:00:00+00:00",
        }
        for symbol in ("NVDA", "AMD")
    ]


class TestHonestTrackRecordWiring:
    """WIA-17/33/36: orkiestrator buduje krzywą portfela (jeden punkt na cykl),
    ujawnia okno scorecardu i liczy metryki pewności z surowych próbek."""

    @staticmethod
    def _settings(**overrides: object) -> MagicMock:
        settings = MagicMock()
        settings.public_scorecard_enabled = True
        settings.track_record_days = 30
        settings.equity_curve_enabled = True
        settings.lessons_enabled = False
        settings.calibration_curve_enabled = True
        for name, value in overrides.items():
            setattr(settings, name, value)
        return settings

    @staticmethod
    def _repo() -> MagicMock:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.return_value = _same_day_rows()
        repo.get_resolved_for_calibration.return_value = [
            {"confidence_score": 0.6, "is_trend_correct": True},
            {"confidence_score": 0.6, "is_trend_correct": False},
        ]
        return repo

    def test_track_record_equity_is_one_equal_weight_point_per_cycle(self) -> None:
        equity, _, _ = main_agent._build_track_record(self._settings(), self._repo())

        assert equity is not None
        assert len(equity.points) == 1
        assert equity.total_return == pytest.approx(0.10)

    def test_scorecard_uses_the_portfolio_curve_and_discloses_its_window(self) -> None:
        publisher = MagicMock()

        main_agent._publish_public_scorecard(self._settings(), self._repo(), publisher)

        page = publisher.publish.call_args.args[0]
        assert "+10.0%" in page
        assert "+21.0%" not in page
        assert "Okno: ostatnie 30 dni" in page

    def test_track_record_stats_carry_hit_rate_evidence_and_brier(self) -> None:
        stats = main_agent._build_track_record_stats(self._settings(), self._repo())

        assert stats is not None
        assert stats.hit_rate is not None
        assert stats.hit_rate.hit_rate.directional == 2
        assert stats.hit_rate.cycles == 1
        assert stats.scores is not None
        assert stats.scores.brier == pytest.approx(0.26)

    def test_track_record_stats_follow_the_section_flags(self) -> None:
        settings = self._settings(equity_curve_enabled=False, calibration_curve_enabled=False)

        assert main_agent._build_track_record_stats(settings, self._repo()) is None

    def test_track_record_stats_survive_a_repository_error(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.side_effect = Exception("supabase down")
        repo.get_resolved_for_calibration.side_effect = Exception("supabase down")

        assert main_agent._build_track_record_stats(self._settings(), repo) is None


class TestBuildTrackRecord:
    """WIA-06 (G25): repozytorium zwraca zamknięte predykcje od najnowszej
    (`ORDER BY timestamp DESC`), a sparkline „ostatnie N” czyta ogon serii.
    Krzywa musi iść chronologicznie niezależnie od ORDER BY adaptera."""

    @staticmethod
    def _row(day: str, actual: str) -> dict[str, object]:
        return {
            "symbol": "NVDA",
            "predicted_trend": "BULLISH",
            "is_trend_correct": Decimal(actual) > Decimal("100"),
            "price_at_prediction": "100",
            "actual_price_after_12h": actual,
            "timestamp": f"{day}T14:00:00+00:00",
        }

    def test_equity_points_are_chronological_when_repo_returns_newest_first(self) -> None:
        settings = MagicMock()
        settings.track_record_days = 30
        settings.equity_curve_enabled = True
        settings.lessons_enabled = False
        settings.calibration_curve_enabled = False
        repo = MagicMock(spec=RepositoryPort)
        # Najnowszy cykl to strata, najstarszy — zysk; kolejność jak z adaptera.
        repo.get_resolved_predictions.return_value = [
            self._row("2026-10-17", "90"),
            self._row("2026-10-16", "105"),
            self._row("2026-10-15", "110"),
        ]

        equity, _, _ = main_agent._build_track_record(settings, repo)

        assert equity is not None
        assert [p.period_return for p in equity.points] == [
            pytest.approx(0.10),
            pytest.approx(0.05),
            pytest.approx(-0.10),
        ]


class TestReflectionAgeWindowWiring:
    """WIA-04: `reflection_max_age_hours` z Settings trafia do AgentGraphDeps;
    0 wyłącza granicę i wtedy deps dostają None (kwarg nie idzie do repo)."""

    def test_build_use_case_passes_max_age_to_graph_deps(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")

        main_agent.build_use_case(settings)

        deps = use_case_cls.call_args.args[0]
        assert deps.reflection_max_age_hours == 96

    def test_zero_max_age_leaves_the_bound_off(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.reflection_max_age_hours = 0

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].reflection_max_age_hours is None


class TestInsiderFlowCoversStocksOnly:
    """Form 4 składają insiderzy spółek. Krypto nie ma insiderów SEC, a w mapie
    tickerów SEC `BTC`/`ETH` mogą oznaczać trusty ETF, więc wiersz krypto dostałby
    cudze dane. ETF-y to fundusze otwarte bez Form 4: puste okno głosowałoby
    w Alpha Fusion neutralnie z największą wagą. Jak fundamenty — tylko STOCK."""

    @staticmethod
    def _enrichment(settings):
        settings.insider_flow_enabled = True
        settings.crypto_symbols = ["BTC"]
        settings.symbols_etf = ["VOO"]
        return main_agent.build_alpha_enrichment(settings)

    @pytest.mark.parametrize("symbol", ["BTC", "VOO"])
    def test_non_stock_symbol_never_reaches_edgar(self, settings, mocker, symbol) -> None:
        mock_get = mocker.patch("requests.Session.get")

        signals = self._enrichment(settings).enrich(symbol)

        assert signals.insider is None
        mock_get.assert_not_called()

    def test_stock_symbol_still_reaches_edgar(self, settings, mocker) -> None:
        mock_get = mocker.patch("requests.Session.get")

        self._enrichment(settings).enrich("AAPL")

        assert mock_get.called


class TestAlphaVantageNewsWindowWiring:
    """DAT-09: Fast Loop pyta AV o newsy z ostatnich 24 h (`time_from`), żeby
    `news_volume_24h` znaczyło to, co mówi nazwa — bez dodatkowych requestów."""

    def test_build_use_case_bounds_news_feed_to_24h(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from datetime import timedelta

        from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient

        client_cls = mocker.patch(
            "main_agent.AlphaVantageClient", wraps=AlphaVantageClient
        )

        main_agent.build_use_case(settings)

        assert client_cls.call_args.kwargs["news_window"] == timedelta(hours=24)

    def test_build_use_case_fetches_news_lazily_per_symbol(
        self, settings, mock_external_clients, mocker
    ) -> None:
        # FIN-04: symbol pod bramką volatility nie pociąga requestu AV za innych.
        from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient

        client_cls = mocker.patch(
            "main_agent.AlphaVantageClient", wraps=AlphaVantageClient
        )

        main_agent.build_use_case(settings)

        assert client_cls.call_args.kwargs["lazy_per_symbol"] is True


class TestAlphaVantageBudgetBanner:
    """FIN-32: najgorszy przypadek doby AV ponad pojemność kluczy to WARNING w bannerze
    maila Fast Loopa — zanim `av_keys_exhausted` wyjdzie w środku cyklu."""

    def _alerts(self, settings, mocker) -> list[QuotaAlert]:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        dispatch = mocker.patch("main_agent._dispatch_reports")

        main_agent.main(settings)

        return list(dispatch.call_args.kwargs["quota_monitor"].alerts)

    def test_worst_case_above_capacity_lands_in_the_banner(
        self, settings, mock_external_clients, mocker
    ) -> None:
        over = settings.model_copy(update={
            "symbols": [f"S{i:02d}" for i in range(30)],
            "alpha_vantage_api_keys": ["av1"],
        })

        alerts = self._alerts(over, mocker)

        budget = [a for a in alerts if a.source == "Alpha Vantage"]
        assert [a.severity for a in budget] == [QuotaSeverity.WARNING]
        assert "30" in budget[0].message
        assert "25" in budget[0].message
        assert "ALPHA_VANTAGE_API_KEYS" in budget[0].action

    def test_budget_within_capacity_adds_no_alert(
        self, settings, mock_external_clients, mocker
    ) -> None:
        assert self._alerts(settings, mocker) == []


class TestPortfolioRadarMinimumSamples:
    """PORT-22: produkcyjny radar egzekwuje minimalne próby — VaR z dziewięciu
    zwrotów nie trafia do maila jako „VaR 95%”."""

    def test_short_history_reports_missing_var_instead_of_a_number(self, settings) -> None:
        from datetime import timedelta

        from src.domain.value_objects import Money

        base = datetime(2026, 6, 1, tzinfo=UTC)

        def history(symbol: str, days: int) -> list:
            scale = {"AAPL": 1.0, "VOO": 2.0}[symbol]
            return [
                (base + timedelta(days=i), Money(Decimal(str(100 * scale + i * (i % 3)))))
                for i in range(10)
            ]

        repository = MagicMock(spec=RepositoryPort)
        repository.get_price_history.side_effect = history
        settings.portfolio_var_enabled = True

        report = main_agent._build_portfolio_radar(settings, repository, ["AAPL", "VOO"])

        assert report is not None
        assert report.var is None
        assert ("var", 9, 60) in report.insufficient


class TestHeldPositionsOutsideWatchlist:
    """PORT-07: pozycja w spółce spoza config.toml nie miała ceny i znikała z P/L,
    a dopisanie jej do `symbols` uruchamiało płatne wywołania przy każdym ruchu
    ≥ 2%. Teraz dostaje SAM darmowy /quote — bez snapshotu, grafu, LLM, AV
    i embeddingów."""

    _PAID_FACTORIES = (
        "build_use_case",
        "build_llm_adapter",
        "build_council_adapter",
        "build_council_llm_adapter",
        "build_embedding_adapter",
        "build_alpha_enrichment",
        "build_tool_use_adapter",
    )

    @staticmethod
    def _portfolio(*symbols: str) -> "main_agent.Portfolio":
        from datetime import date

        from src.domain.portfolio import Position

        return main_agent.Portfolio(
            positions=tuple(
                Position(
                    symbol=symbol,
                    quantity=Decimal("2"),
                    purchase_price=Decimal("100"),
                    purchase_date=date(2026, 10, 1),
                )
                for symbol in symbols
            ),
            as_of=datetime(2026, 10, 18, 5, 0, tzinfo=UTC),
        )

    def _dispatch(
        self, settings: Settings, mocker, portfolio, market,
        unsupported: tuple[str, ...] = ("XDWD.DE",),
    ) -> tuple[MagicMock, MagicMock, dict[str, MagicMock], MagicMock]:
        from src.application.ports import LLMPort
        from src.application.quota_monitor import QuotaMonitor

        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        settings.symbols_unsupported_price = list(unsupported)
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch("main_agent._build_user_portfolio", return_value=portfolio)
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        mocker.patch("main_agent._publish_public_scorecard")
        report = mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
        market_factory = mocker.patch("main_agent.build_market_port", return_value=market)
        llm = MagicMock(spec=LLMPort)
        paid = {
            name: mocker.patch(f"main_agent.{name}", return_value=llm)
            for name in self._PAID_FACTORIES
        }
        repository = MagicMock(spec=RepositoryPort)
        main_agent._dispatch_reports(
            results=[
                main_agent.SymbolResult(
                    symbol="AAPL", status="saved", current_price=Decimal("230")
                )
            ],
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=repository,
            quota_monitor=QuotaMonitor(),
            notifier=MagicMock(),
            subscribers=[],
            all_symbols=["AAPL", "VOO", "XDWD.DE"],
            context=main_agent.ReportContext(),
        )
        paid["llm"] = llm
        return report, repository, paid, market_factory

    def test_held_symbol_outside_watchlist_gets_quote_only(self, settings, mocker) -> None:
        from src.application.ports import MarketDataPort
        from src.domain.value_objects import Money

        market = MagicMock(spec=MarketDataPort)
        market.get_current_price.return_value = Money(Decimal("410.50"))

        report, repository, paid, _ = self._dispatch(
            settings, mocker, self._portfolio("AAPL", "MSFT"), market
        )

        market.get_current_price.assert_called_once_with("MSFT")
        repository.save_price_snapshot.assert_not_called()
        for name in self._PAID_FACTORIES:
            paid[name].assert_not_called()
        assert paid["llm"].mock_calls == []
        context = report.call_args_list[0].args[3]
        assert context.held_prices == {"MSFT": Decimal("410.50")}

    def test_watchlist_and_unsupported_symbols_are_not_quoted(self, settings, mocker) -> None:
        from src.application.ports import MarketDataPort

        market = MagicMock(spec=MarketDataPort)

        # AAPL i XDWD.DE są w watchliście (XDWD.DE także w unsupported), IUSN.DE
        # tylko w unsupported — Finnhub free i tak go nie wyceni.
        self._dispatch(
            settings, mocker, self._portfolio("AAPL", "XDWD.DE", "IUSN.DE", "MSFT"),
            market, unsupported=("XDWD.DE", "IUSN.DE"),
        )

        market.get_current_price.assert_called_once_with("MSFT")

    def test_quote_failure_leaves_the_symbol_unpriced(self, settings, mocker) -> None:
        from src.application.ports import MarketDataPort
        from src.domain.value_objects import Money

        market = MagicMock(spec=MarketDataPort)

        def quote(symbol: str) -> Money:
            if symbol == "MSFT":
                raise RuntimeError("Finnhub 429")
            return Money(Decimal("50"))

        market.get_current_price.side_effect = quote

        report, _, _, _ = self._dispatch(
            settings, mocker, self._portfolio("MSFT", "TSLA"), market
        )

        assert report.call_args_list[0].args[3].held_prices == {"TSLA": Decimal("50")}

    def test_no_portfolio_builds_no_market_port(self, settings, mocker) -> None:
        from src.application.ports import MarketDataPort

        _, _, _, market_factory = self._dispatch(
            settings, mocker, None, MagicMock(spec=MarketDataPort)
        )

        market_factory.assert_not_called()


class TestPortfolioAwareSubject:
    """PORT-10: temat maila dziedziczy po leadzie — także po tierach portfela."""

    def test_subject_carries_the_signal_against_your_position(self, settings, mocker) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.council import InvestorOpinion

        verdict = CouncilVerdict(
            final_recommendation="SELL", consensus_strength=0.9, summary="Rada: SELL.",
            dissenting_views=(),
            investor_opinions=(
                InvestorOpinion(investor_name="A", recommendation="SELL",
                                confidence=0.9, reasoning="x", key_factors=()),
            ),
        )
        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch(
            "main_agent._build_user_portfolio",
            return_value=TestHeldPositionsOutsideWatchlist._portfolio("AAPL"),
        )
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        mocker.patch("main_agent._publish_public_scorecard")
        mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
        notifier = MagicMock()

        main_agent._dispatch_reports(
            results=[
                main_agent.SymbolResult(
                    symbol="AAPL", status="saved", delta=Decimal("-0.05"),
                    current_price=Decimal("95"), council_verdict=verdict,
                ),
            ],
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=MagicMock(spec=RepositoryPort),
            quota_monitor=QuotaMonitor(),
            notifier=notifier,
            subscribers=[],
            all_symbols=["AAPL", "VOO"],
            context=main_agent.ReportContext(),
        )

        subject = notifier.send_report.call_args.kwargs["subject"]
        assert "Sygnał przeciw Twojej pozycji: rada SPRZEDAJ na AAPL (100% portfela)" in subject


class TestLegacyPushWithoutPortfolio:
    """Decyzja 2026-10-19: mail tekstowy niesie sekcję portfela, ale stary wariant
    pushu (pełny tekst do Telegram/Slack przy wyłączonym `messenger_digest_enabled`)
    jej nie dostaje — kwoty P/L nie trafiają do komunikatorów."""

    @staticmethod
    def _push_settings(settings: Settings, *, messenger_digest: bool) -> Settings:
        settings.notifications_enabled = True
        settings.resend_api_key = "re_test"
        settings.digest_to_email = "you@example.com"
        settings.telegram_bot_token = "tg-token"
        settings.telegram_chat_id = "42"
        settings.messenger_digest_enabled = messenger_digest
        return settings

    def test_email_text_has_the_portfolio_and_the_full_push_does_not(
        self, settings, mocker
    ) -> None:
        from src.application.quota_monitor import QuotaMonitor

        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch(
            "main_agent._build_user_portfolio",
            return_value=TestHeldPositionsOutsideWatchlist._portfolio("AAPL"),
        )
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        mocker.patch("main_agent._publish_public_scorecard")
        email = MagicMock()
        legacy_push = MagicMock()

        main_agent._dispatch_reports(
            results=[
                main_agent.SymbolResult(
                    symbol="AAPL", status="saved", current_price=Decimal("230")
                ),
            ],
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=MagicMock(spec=RepositoryPort),
            quota_monitor=QuotaMonitor(),
            notifier=email,
            subscribers=[],
            all_symbols=["AAPL", "VOO"],
            context=main_agent.ReportContext(),
            legacy_push_notifier=legacy_push,
        )

        email_text = email.send_report.call_args.kwargs["plain_text"]
        push_text = legacy_push.send_report.call_args.kwargs["plain_text"]
        assert "Twój portfel" in email_text
        assert "$460,00" in email_text
        assert "Twój portfel" not in push_text
        assert "$460,00" not in push_text
        assert (
            legacy_push.send_report.call_args.kwargs["subject"]
            == email.send_report.call_args.kwargs["subject"]
        )

    def test_legacy_push_notifier_exists_only_without_the_messenger_digest(
        self, settings
    ) -> None:
        from src.infrastructure.adapters.telegram_notifier import TelegramNotifier

        off = main_agent.build_legacy_push_notifier(
            self._push_settings(settings, messenger_digest=False)
        )
        on = main_agent.build_legacy_push_notifier(
            self._push_settings(settings, messenger_digest=True)
        )

        assert isinstance(off, TelegramNotifier)
        assert on is None

    def test_legacy_push_notifier_is_none_without_push_secrets(self, settings) -> None:
        settings.messenger_digest_enabled = False
        settings.telegram_bot_token = None
        settings.telegram_chat_id = None
        settings.slack_webhook_url = None

        assert main_agent.build_legacy_push_notifier(settings) is None

    def test_email_notifier_can_leave_push_out(self, settings) -> None:
        from src.infrastructure.adapters.resend_notifier import ResendNotifier

        notifier = main_agent.build_notifier(
            self._push_settings(settings, messenger_digest=False), include_push=False
        )

        assert isinstance(notifier, ResendNotifier)

    def test_main_sends_push_apart_from_the_email(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.infrastructure.adapters.resend_notifier import ResendNotifier
        from src.infrastructure.adapters.telegram_notifier import TelegramNotifier

        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        dispatch = mocker.patch("main_agent._dispatch_reports")

        main_agent.main(self._push_settings(settings, messenger_digest=False))

        kwargs = dispatch.call_args.kwargs
        assert isinstance(kwargs["notifier"], ResendNotifier)
        assert isinstance(kwargs["legacy_push_notifier"], TelegramNotifier)


class TestDisplayTimezone:
    """MAIL-14: temat maila i nagłówek pushu w strefie `display_timezone`, nie UTC."""

    def test_subject_stamp_is_local_time(self) -> None:
        subject = main_agent._build_subject(
            lead_items=[],
            analyzed=3,
            failures=0,
            started_at=datetime(2026, 10, 16, 5, 30, tzinfo=UTC),
            prefix="",
            display_timezone="Europe/Warsaw",
        )

        assert subject.endswith("(pt 16.10 07:30)")
        assert "UTC" not in subject

    def test_dispatch_uses_the_configured_zone_everywhere(self, settings, mocker) -> None:
        from src.application.quota_monitor import QuotaMonitor

        # Strefa inna niż domyślna — dowód, że idzie z ustawień, nie ze stałej.
        settings.display_timezone = "America/New_York"
        settings.messenger_digest_enabled = True
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch("main_agent._build_user_portfolio", return_value=None)
        report = mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        mocker.patch("main_agent._publish_public_scorecard")
        push = MagicMock()
        mocker.patch("main_agent.build_push_notifier", return_value=push)
        email = MagicMock()

        main_agent._dispatch_reports(
            results=[],
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=MagicMock(spec=RepositoryPort),
            quota_monitor=QuotaMonitor(),
            notifier=email,
            subscribers=[],
            all_symbols=["AAPL"],
            context=main_agent.ReportContext(),
        )

        assert report.call_args.args[3].display_timezone == "America/New_York"
        assert email.send_report.call_args.kwargs["subject"].endswith("(nd 18.10 01:30)")
        digest = push.send_report.call_args.kwargs["plain_text"]
        assert digest.splitlines()[0] == "📊 StockAgent — nd 18.10 01:30"


class TestDispatchReports:
    """BEZ-01 (G22): fan-out per subskrybent nie wysyła portfela właściciela
    (`DIGEST_TO_EMAIL`) nikomu poza nim — ani w HTML, ani w tekście."""

    OWNER = "owner@example.com"
    FRIEND = "friend@example.com"

    def _fan_out(
        self,
        settings: Settings,
        mocker,
        results: list | None = None,
        *,
        notifications_enabled: bool = True,
    ) -> dict[str, dict[str, str]]:
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.subscriber import Subscriber

        settings.notifications_enabled = notifications_enabled
        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        settings.resend_api_key = "re_test"
        settings.digest_to_email = self.OWNER
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch(
            "main_agent._build_user_portfolio",
            return_value=TestHeldPositionsOutsideWatchlist._portfolio("AAPL"),
        )
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        mocker.patch("main_agent._publish_public_scorecard")
        sent: dict[str, dict[str, str]] = {}

        def resend(*, recipient: str, **_: object) -> MagicMock:
            channel = MagicMock()
            channel.send_report.side_effect = lambda **mail: sent.__setitem__(recipient, mail)
            return channel

        mocker.patch("main_agent.ResendNotifier", side_effect=resend)

        main_agent._dispatch_reports(
            results=results or [
                main_agent.SymbolResult(
                    symbol="AAPL", status="saved", current_price=Decimal("230")
                ),
            ],
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=MagicMock(spec=RepositoryPort),
            quota_monitor=QuotaMonitor(),
            notifier=MagicMock(),
            subscribers=[
                Subscriber(email="Owner@Example.com", symbols=frozenset()),
                Subscriber(email=self.FRIEND, symbols=frozenset()),
            ],
            all_symbols=["AAPL", "VOO"],
            context=main_agent.ReportContext(),
        )
        return sent

    def test_fan_out_respects_notifications_disabled(self, settings, mocker) -> None:
        # MAIL-11: `notifications_enabled=false` wyłącza też fan-out do subskrybentów,
        # nie tylko pojedynczą wysyłkę — sam klucz Resend nie wystarcza.
        sent = self._fan_out(settings, mocker, notifications_enabled=False)

        assert sent == {}
        main_agent.ResendNotifier.assert_not_called()

    def test_subscriber_digest_omits_owner_portfolio(self, settings, mocker) -> None:
        sent = self._fan_out(settings, mocker)

        friend = sent[self.FRIEND]
        assert "Twój portfel" not in friend["html_body"]
        assert "$460,00" not in friend["html_body"]
        assert "Twój portfel" not in friend["plain_text"]
        assert "$460,00" not in friend["plain_text"]

    def test_owner_digest_keeps_the_portfolio(self, settings, mocker) -> None:
        # Adres z innym wielkościami liter to nadal właściciel (casefold).
        sent = self._fan_out(settings, mocker)

        owner = sent["Owner@Example.com"]
        assert "Twój portfel" in owner["html_body"]
        assert "$460,00" in owner["plain_text"]

    def test_withheld_portfolio_is_logged_with_a_masked_address(
        self, settings, mocker, caplog
    ) -> None:
        import logging

        with caplog.at_level(logging.INFO, logger="main_agent"):
            self._fan_out(settings, mocker)

        messages = [r.getMessage() for r in caplog.records]
        assert any("f***@example.com" in m and "portfel" in m for m in messages)
        assert not any(self.FRIEND in m for m in messages)

    def test_subscriber_subject_does_not_reveal_the_owner_position(self, settings, mocker) -> None:
        # PORT-10 robi temat z leadu świadomego portfela („… na AAPL (100% portfela)”);
        # temat obcego subskrybenta liczony jest z leadu bez portfela.
        from src.domain.council import InvestorOpinion

        verdict = CouncilVerdict(
            final_recommendation="SELL", consensus_strength=0.9, summary="Rada: SELL.",
            dissenting_views=(),
            investor_opinions=tuple(
                InvestorOpinion(investor_name=n, recommendation="SELL",
                                confidence=0.9, reasoning="x", key_factors=())
                for n in ("A", "B")
            ),
        )
        results = [
            main_agent.SymbolResult(
                symbol="AAPL", status="saved", delta=Decimal("-0.05"),
                current_price=Decimal("95"), council_verdict=verdict,
            ),
        ]

        sent = self._fan_out(settings, mocker, results)

        assert "Twojej pozycji" in sent["Owner@Example.com"]["subject"]
        friend_subject = sent[self.FRIEND]["subject"]
        assert "Twojej pozycji" not in friend_subject
        assert "portfela" not in friend_subject
        assert "AAPL -5,0%, rada zgodna: SPRZEDAJ" in friend_subject


class TestFeatureSetVersionWiring:
    """DAT-09 — flagi z Settings sterują stemplem wersji w Fast Loopie i filtrem
    wersji w Slow Loopie; obie domyślnie off (kolumna dopiero po migracji 027)."""

    def test_flags_default_off(self, settings) -> None:
        assert settings.feature_set_version_enabled is False
        assert settings.feature_set_filter_enabled is False

    def test_fast_loop_passes_the_stamp_flag_to_graph_deps(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.feature_set_version_enabled = True

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].feature_set_version_enabled is True

    def test_slow_loop_trains_on_the_current_version_only_when_enabled(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.application.ml_features import FEATURE_SET_VERSION

        use_case_cls = mocker.patch("main_trainer.TrainModelUseCase")
        settings.feature_set_filter_enabled = True

        main_trainer.build_use_case(settings)

        assert use_case_cls.call_args.kwargs["feature_set_version"] == FEATURE_SET_VERSION

    def test_slow_loop_reads_the_whole_history_when_disabled(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_trainer.TrainModelUseCase")

        main_trainer.build_use_case(settings)

        assert use_case_cls.call_args.kwargs["feature_set_version"] is None


class TestMarketFeaturesWiring:
    """ML-31 — `market_features_enabled` dociera do grafu, a wersja zestawu cech 4
    do filtra treningu i backtestu (ta sama próbka co stempel Fast Loopa)."""

    @pytest.mark.parametrize("enabled", [False, True])
    def test_flag_reaches_graph_deps(
        self, settings, mock_external_clients, mocker, enabled: bool
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.market_features_enabled = enabled

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].market_features_enabled is enabled

    @pytest.mark.parametrize(("market", "expected"), [(False, 3), (True, 4)])
    def test_slow_loop_and_backtest_filter_on_the_flagged_version(
        self, settings, mock_external_clients, mocker, market: bool, expected: int
    ) -> None:
        trainer_cls = mocker.patch("main_trainer.TrainModelUseCase")
        backtest_cls = mocker.patch("main_backtest.BacktestUseCase")
        settings.feature_set_filter_enabled = True
        settings.market_features_enabled = market

        main_trainer.build_use_case(settings)
        main_backtest.build_use_case(settings)

        assert trainer_cls.call_args.kwargs["feature_set_version"] == expected
        assert backtest_cls.call_args.kwargs["feature_set_version"] == expected

    def test_fast_loop_prepares_the_market_features_before_the_graph(
        self, settings, mock_external_clients, mocker
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)

        main_agent.main(settings)

        fake_uc.prepare_market_features.assert_called_once()
        call_ = fake_uc.prepare_market_features.call_args
        assert list(call_.args[0]) == ["AAPL", "VOO"]
        assert call_.kwargs["now"].tzinfo is not None
        # ML-32: klasy aktywów do mediany klasy i szerokości rynku.
        assert set(call_.kwargs["asset_types"]) == {"AAPL", "VOO"}
        # Faza cech idzie przed pierwszym symbolem grafu.
        names = [c[0] for c in fake_uc.method_calls]
        assert names.index("prepare_market_features") < names.index("run")


class TestModelScorecardBaselinesWiring:
    """ML-10 — `model_scorecard_baselines_enabled` dociera do repozytorium, które
    zapisuje karty: use case'u (wiersz ALL) i kroków per symbol w `main()`."""

    def test_flag_defaults_off(self, settings) -> None:
        assert settings.model_scorecard_baselines_enabled is False

    @pytest.mark.parametrize("enabled", [False, True])
    def test_use_case_repository_gets_the_flag(
        self, settings, mock_external_clients, mocker, enabled: bool
    ) -> None:
        settings.model_scorecard_baselines_enabled = enabled
        repo_cls = mocker.patch("main_trainer.SupabaseRepository")

        main_trainer.build_use_case(settings)

        assert repo_cls.call_args.kwargs["scorecard_baselines_enabled"] is enabled

    @pytest.mark.parametrize("enabled", [False, True])
    def test_per_symbol_scorecard_repository_gets_the_flag(
        self, settings, mock_external_clients, mocker, enabled: bool
    ) -> None:
        settings.model_scorecard_baselines_enabled = enabled
        repo_cls = mocker.patch(
            "main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort)
        )
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        main_trainer.main(settings)

        assert repo_cls.call_args_list
        assert all(
            c.kwargs["scorecard_baselines_enabled"] is enabled
            for c in repo_cls.call_args_list
        )


def test_backtest_log_reports_the_drift_and_llm_baselines(
    settings, mock_external_clients, mocker, caplog
):
    # ML-10: werdykt offline harnessa liczy się wobec najsilniejszego baseline'u,
    # więc log pokazuje wszystkie trzy RMSE obok kandydata.
    from src.domain.backtest import BacktestSummary

    fake_uc = MagicMock()
    fake_uc.run.return_value = BacktestSummary(
        n_folds=4,
        mean_candidate_rmse=0.03,
        mean_baseline_rmse=0.05,
        mean_hit_rate=0.6,
        improvement_pct=40.0,
        beats_baseline=False,
        verdict="model nie bije baseline'u „sam LLM”",
        mean_drift_baseline_rmse=0.045,
        mean_llm_baseline_rmse=0.029,
    )
    mocker.patch("main_backtest.BacktestUseCase", return_value=fake_uc)

    with caplog.at_level("INFO", logger="main_backtest"):
        main_backtest.main(settings)

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("AAPL"))
    assert "base_rmse=0.05000" in line
    assert "drift_rmse=0.04500" in line
    assert "llm_rmse=0.02900" in line


class TestSettledAtWindowsWiring:
    """WIA-19 — `settled_at_windows_enabled` przełącza okna „zamknięte w oknie”
    z daty utworzenia na chwilę rozliczenia (migracja 029): sekcja maila bierze
    predykcje rozliczone od startu cyklu, recap — rozliczone w ostatnich 7 dniach.
    Off = zapytania i nagłówek dokładnie jak przed migracją."""

    _STARTED_AT = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    def test_flag_defaults_off(self, settings) -> None:
        assert settings.settled_at_windows_enabled is False

    @staticmethod
    def _repo() -> MagicMock:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_accuracy_stats.return_value = None
        repo.get_recently_resolved_predictions.return_value = []
        repo.get_recent_quota_alerts.return_value = []
        repo.get_council_vote_history.return_value = []
        repo.get_persona_track_record.return_value = {}
        return repo

    def test_report_context_reads_by_creation_date_without_since(self) -> None:
        from src.application.quota_monitor import QuotaMonitor

        repo = self._repo()

        main_agent._fetch_report_context(repo, QuotaMonitor(), ["AAPL"])

        assert repo.get_recently_resolved_predictions.call_args == call(
            main_agent.RESOLVED_PREDICTIONS_HOURS
        )

    def test_report_context_reads_settled_since_the_cycle_start(self) -> None:
        from src.application.quota_monitor import QuotaMonitor

        repo = self._repo()

        main_agent._fetch_report_context(
            repo, QuotaMonitor(), ["AAPL"], resolved_since=self._STARTED_AT
        )

        assert repo.get_recently_resolved_predictions.call_args == call(
            main_agent.RESOLVED_PREDICTIONS_HOURS, since=self._STARTED_AT
        )

    def _dispatch(self, settings: Settings, mocker) -> tuple[MagicMock, MagicMock]:
        from src.application.quota_monitor import QuotaMonitor

        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        fetch = mocker.patch(
            f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], [])
        )
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch("main_agent._build_user_portfolio", return_value=None)
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        mocker.patch("main_agent._publish_public_scorecard")
        report = mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
        main_agent._dispatch_reports(
            results=[],
            started_at=self._STARTED_AT,
            duration=1.0,
            failures=0,
            settings=settings,
            repository=MagicMock(spec=RepositoryPort),
            quota_monitor=QuotaMonitor(),
            notifier=MagicMock(),
            subscribers=[],
            all_symbols=["AAPL"],
            context=main_agent.ReportContext(),
        )
        return fetch, report

    def test_flag_on_reads_this_cycle_and_labels_the_section(self, settings, mocker) -> None:
        settings.settled_at_windows_enabled = True

        fetch, report = self._dispatch(settings, mocker)

        assert fetch.call_args.kwargs["resolved_since"] == self._STARTED_AT
        assert report.call_args_list[0].args[3].resolved_this_cycle is True

    def test_flag_off_keeps_the_creation_date_window(self, settings, mocker) -> None:
        settings.settled_at_windows_enabled = False

        fetch, report = self._dispatch(settings, mocker)

        assert fetch.call_args.kwargs["resolved_since"] is None
        assert report.call_args_list[0].args[3].resolved_this_cycle is False

    def test_failed_settlement_read_is_reported_as_unavailable(self) -> None:
        # MAIL-25: awaria odczytu to None, a nie pusta lista — pusta lista
        # znaczy w mailu „brak rozliczeń”, czego po awarii nie wiemy.
        from src.application.quota_monitor import QuotaMonitor

        repo = self._repo()
        repo.get_recently_resolved_predictions.side_effect = Exception("supabase down")

        _, resolved, *_ = main_agent._fetch_report_context(repo, QuotaMonitor(), ["AAPL"])

        assert resolved is None

    def test_dispatch_marks_the_section_unavailable_after_a_failed_read(
        self, settings, mocker
    ) -> None:
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, None, [], [], []))

        _, report = self._dispatch_with_fetch_patched(settings, mocker)

        context = report.call_args_list[0].args[3]
        assert context.resolved_unavailable is True

    def test_dispatch_keeps_an_empty_read_as_nothing_settled(self, settings, mocker) -> None:
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))

        _, report = self._dispatch_with_fetch_patched(settings, mocker)

        context = report.call_args_list[0].args[3]
        assert context.resolved_unavailable is False
        assert context.resolved_predictions == []

    def _dispatch_with_fetch_patched(
        self, settings: Settings, mocker
    ) -> tuple[MagicMock, MagicMock]:
        from src.application.quota_monitor import QuotaMonitor

        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch("main_agent._build_user_portfolio", return_value=None)
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        mocker.patch("main_agent._publish_public_scorecard")
        report = mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
        notifier = MagicMock()
        main_agent._dispatch_reports(
            results=[],
            started_at=self._STARTED_AT,
            duration=1.0,
            failures=0,
            settings=settings,
            repository=MagicMock(spec=RepositoryPort),
            quota_monitor=QuotaMonitor(),
            notifier=notifier,
            subscribers=[],
            all_symbols=["AAPL"],
            context=main_agent.ReportContext(),
        )
        return notifier, report

    def test_push_digest_says_settlements_are_unavailable_after_a_failed_read(self) -> None:
        push = MagicMock()

        main_agent._dispatch_push_digest(
            push_notifier=push,
            results=[],
            started_at=self._STARTED_AT,
            resolved=[],
            subject="StockAgent",
            resolved_unavailable=True,
        )

        sent = push.send_report.call_args.kwargs["plain_text"]
        assert "rozliczenia niedostępne" in sent

    def _trainer(self, settings, mocker) -> MagicMock:
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch(
            "main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort)
        )
        recap_uc = MagicMock()
        recap_uc.run.return_value = {"status": "skipped_below_threshold", "n_closed": 2}
        mocker.patch("main_trainer.WeeklyRecapUseCase", return_value=recap_uc)
        mocker.patch("main_trainer.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent.build_llm_adapter", return_value=MagicMock())
        return recap_uc

    def test_flag_on_recap_reads_the_week_by_settlement(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.settled_at_windows_enabled = True
        recap_uc = self._trainer(settings, mocker)
        before = datetime.now(UTC)

        main_trainer.main(settings)

        kwargs = recap_uc.run.call_args.kwargs
        assert kwargs["days"] == 7
        since = kwargs["since"]
        assert before - timedelta(days=7) <= since <= datetime.now(UTC) - timedelta(days=7)

    def test_flag_off_recap_keeps_the_creation_date_window(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.settled_at_windows_enabled = False
        recap_uc = self._trainer(settings, mocker)

        main_trainer.main(settings)

        assert recap_uc.run.call_args.kwargs == {"days": 7, "since": None}


class TestRunContextWiring:
    """MAIL-42: `main()` buduje `RunContext` z env GitHub Actions i oddaje go
    raportowi — stopka maila wskazuje przebieg, commit i logi."""

    def _dispatched_context(self, settings, mocker):
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        dispatch = mocker.patch("main_agent._dispatch_reports")

        main_agent.main(settings)

        return dispatch.call_args.kwargs["context"]

    def test_github_actions_run_reaches_the_report(
        self, settings, mock_external_clients, mocker, monkeypatch
    ) -> None:
        from src.application.run_context import RunContext

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_RUN_ID", "77")
        monkeypatch.setenv("GITHUB_SHA", "abc1234def")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        monkeypatch.setenv("GITHUB_REPOSITORY", "x/y")

        context = self._dispatched_context(settings, mocker)

        assert context.run_context == RunContext(
            run_id="77",
            git_sha="abc1234def",
            trigger="workflow_dispatch",
            run_url="https://github.com/x/y/actions/runs/77",
        )

    def test_local_run_reaches_the_report_without_a_logs_url(
        self, settings, mock_external_clients, mocker, monkeypatch
    ) -> None:
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

        context = self._dispatched_context(settings, mocker)

        assert context.run_context.git_sha == "local"
        assert context.run_context.run_url is None


class TestPredictionRecipeWiring:
    """WIA-20: przepis predykcji liczony RAZ na cykl w DI i wstrzykiwany jednym
    polem `AgentGraphDeps.recipe`; kolumna `recipe_version` za własną flagą."""

    def test_recipe_names_the_models_the_run_and_the_config(self, settings) -> None:
        from src.application.run_context import RunContext

        settings.llm_provider = "anthropic"
        settings.anthropic_api_key = "sk-ant-test"
        settings.council_llm_provider = "openai"
        settings.council_llm_model = "gpt-5-mini"

        recipe = main_agent.build_prediction_recipe(
            settings, RunContext(run_id="77", git_sha="abc1234def")
        )

        assert (recipe.llm_provider, recipe.llm_model) == ("anthropic", "claude-sonnet-4-6")
        assert (recipe.council_provider, recipe.council_model) == ("openai", "gpt-5-mini")
        assert (recipe.git_sha, recipe.run_id) == ("abc1234def", "77")
        assert recipe.config_hash == settings.config_hash()
        # `ml_model_path` fixture'a nie istnieje → cold start.
        assert recipe.model_version == "cold_start"
        assert len(recipe.prompt_sha256) == 64
        assert len(recipe.personas_sha256) == 64

    def test_council_without_overrides_is_the_main_llm(self, settings) -> None:
        from src.application.run_context import RunContext

        recipe = main_agent.build_prediction_recipe(settings, RunContext())

        assert (recipe.llm_provider, recipe.llm_model) == ("openai", "gpt-4o")
        assert (recipe.council_provider, recipe.council_model) == ("openai", "gpt-4o")

    def test_model_routing_names_both_models(self, settings) -> None:
        from src.application.run_context import RunContext

        settings.model_routing_enabled = True

        recipe = main_agent.build_prediction_recipe(settings, RunContext())

        assert recipe.llm_provider == "routing"
        assert recipe.llm_model == "openai:gpt-5-mini|anthropic:claude-sonnet-4-6"

    def test_recipe_never_carries_a_secret(self, settings) -> None:
        import json

        from src.application.run_context import RunContext

        settings.openai_api_key = "sk-SECRET-openai"
        settings.supabase_key = "SECRET-supabase"
        settings.digest_to_email = "SECRET-owner@example.com"

        fields = main_agent.build_prediction_recipe(settings, RunContext()).receipt_fields()

        assert "SECRET" not in json.dumps(fields)

    def test_flags_off_build_no_recipe(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")

        main_agent.build_use_case(settings)

        deps = use_case_cls.call_args.args[0]
        assert deps.recipe is None
        assert deps.recipe_version_enabled is False

    def test_receipts_on_inject_the_recipe_of_this_run(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.application.run_context import RunContext

        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.receipts_enabled = True

        main_agent.build_use_case(settings, run_context=RunContext(git_sha="abc"))

        deps = use_case_cls.call_args.args[0]
        assert deps.recipe is not None
        assert deps.recipe.git_sha == "abc"
        assert deps.recipe_version_enabled is False

    def test_attestation_on_builds_the_recipe_for_the_commitment(
        self, settings, mock_external_clients, mocker
    ) -> None:
        # WIA-41/42: commitment v2 wiąże `recipe_version` i hash protokołu, więc
        # przepis powstaje także wtedy, gdy kwit i kolumna przepisu są wyłączone.
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.attestation_enabled = True
        repo = MagicMock(spec=RepositoryPort)
        repo.get_settled_moves.return_value = []
        repo.attestation_columns_available.return_value = True

        main_agent.build_use_case(settings, repository=repo)

        deps = use_case_cls.call_args.args[0]
        assert deps.recipe is not None
        assert len(deps.recipe.protocol_sha256) == 64

    def test_column_flag_reaches_graph_deps_with_a_recipe(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.recipe_version_enabled = True

        main_agent.build_use_case(settings)

        deps = use_case_cls.call_args.args[0]
        assert deps.recipe is not None
        assert deps.recipe_version_enabled is True

    def test_main_builds_the_recipe_from_the_run_context(
        self, settings, mock_external_clients, mocker, monkeypatch
    ) -> None:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_SHA", "abc1234def")
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        build = mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent._dispatch_reports")

        main_agent.main(settings)

        assert build.call_args.kwargs["run_context"].git_sha == "abc1234def"


class TestUniverseVersionInRecipe:
    """WIA-48: przepis niesie wersję uniwersum (symbols + crypto_symbols) —
    dodanie symbolu to nowy segment track recordu, a nie „zmiana trafności”."""

    def test_recipe_carries_the_universe_of_this_config(self, settings) -> None:
        from src.application.run_context import RunContext
        from src.domain.recipe import universe_version

        settings.crypto_symbols = ["BTC"]

        recipe = main_agent.build_prediction_recipe(settings, RunContext())

        assert recipe.universe_version == universe_version(["AAPL", "VOO"], ["BTC"])

    def test_added_symbol_is_a_new_recipe(self, settings) -> None:
        from src.application.run_context import RunContext

        before = main_agent.build_prediction_recipe(settings, RunContext())
        settings.symbols = [*settings.symbols, "PLTR"]

        after = main_agent.build_prediction_recipe(settings, RunContext())

        assert after.universe_version != before.universe_version
        assert after.version != before.version


class TestFeatureSnapshotsWiring:
    """ML-14 — flaga z Settings trafia do `AgentGraphDeps`; off (domyślnie) =
    graf nie dotyka tabeli `feature_snapshots`, której przed migracją 031 nie ma."""

    def test_flag_defaults_off_in_graph_deps(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].feature_snapshots_enabled is False

    def test_flag_reaches_graph_deps(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.feature_snapshots_enabled = True

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].feature_snapshots_enabled is True


class TestMoveIntervalWiring:
    """ML-34 — okno kalibracyjne przedziału ruchu czytane RAZ na cykl (darmowy
    odczyt Supabase, kolumny istniejące) i wstrzyknięte jednym polem deps; zapis
    kolumn i pokrycie w Track Record za flagą `interval_coverage_enabled`."""

    @staticmethod
    def _settled(symbol: str, actual: float) -> dict[str, object]:
        return {
            "symbol": symbol,
            "timestamp": "2026-10-16T17:30:00+00:00",
            "price_at_prediction": 100.0,
            "predicted_target_price": 100.0,
            "actual_price_after_12h": actual,
        }

    def test_calibration_windows_are_split_by_asset_class(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.domain.value_objects import AssetType

        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.crypto_symbols = ["BTC"]
        settings.symbols_etf = ["VOO"]
        repo = MagicMock(spec=RepositoryPort)
        repo.get_settled_moves.return_value = [
            self._settled("AAPL", 101.0),
            self._settled("VOO", 99.0),
            self._settled("BTC", 105.0),
        ]

        main_agent.build_use_case(settings, repository=repo)

        repo.get_settled_moves.assert_called_once_with(90)
        calibration = use_case_cls.call_args.args[0].move_calibration
        assert set(calibration) == {AssetType.STOCK, AssetType.ETF, AssetType.CRYPTO}
        assert calibration[AssetType.CRYPTO][0].realized_return == pytest.approx(0.05)

    def test_calibration_read_failure_leaves_the_cycle_without_intervals(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        repo = MagicMock(spec=RepositoryPort)
        repo.get_settled_moves.side_effect = Exception("supabase down")

        main_agent.build_use_case(settings, repository=repo)

        assert use_case_cls.call_args.args[0].move_calibration is None

    def test_column_flag_defaults_off_and_reaches_graph_deps(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")

        main_agent.build_use_case(settings)
        assert settings.interval_coverage_enabled is False
        assert use_case_cls.call_args.args[0].interval_columns_enabled is False

        settings.interval_coverage_enabled = True
        main_agent.build_use_case(settings)
        assert use_case_cls.call_args.args[0].interval_columns_enabled is True

    def test_track_record_coverage_only_behind_the_flag(self, settings) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.return_value = []
        repo.get_resolved_for_calibration.return_value = []
        repo.get_settled_moves.return_value = [
            {"interval_low": 98.0, "interval_high": 102.0, "actual_price_after_12h": 101.0},
            {"interval_low": 98.0, "interval_high": 102.0, "actual_price_after_12h": 97.0},
        ]

        assert main_agent._build_track_record_stats(settings, repo) is None
        repo.get_settled_moves.assert_not_called()

        settings.interval_coverage_enabled = True
        stats = main_agent._build_track_record_stats(settings, repo)

        repo.get_settled_moves.assert_called_once_with(
            settings.track_record_days, with_intervals=True
        )
        assert stats is not None
        assert stats.coverage is not None
        assert (stats.coverage.hits, stats.coverage.n) == (1, 2)


class TestRegimeTaggingWiring:
    """ML-38 — `regime_tagging_enabled` w `AgentGraphDeps`. Brak wartości w
    Settings (None) = dziedziczy `regime_aware_enabled`; jawna wartość wygrywa."""

    @staticmethod
    def _deps(settings, mocker, **overrides: object):
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        for name, value in overrides.items():
            setattr(settings, name, value)

        main_agent.build_use_case(settings)

        return use_case_cls.call_args.args[0]

    def test_follows_regime_aware_when_unset(
        self, settings, mock_external_clients, mocker
    ) -> None:
        deps = self._deps(settings, mocker, regime_aware_enabled=True)

        assert deps.regime_tagging_enabled is True

    def test_off_when_regime_aware_is_off_and_unset(
        self, settings, mock_external_clients, mocker
    ) -> None:
        deps = self._deps(settings, mocker, regime_aware_enabled=False)

        assert deps.regime_tagging_enabled is False

    def test_explicit_off_overrides_regime_aware(
        self, settings, mock_external_clients, mocker
    ) -> None:
        # Furtka dla deploymentu bez migracji 017 przy włączonym reżimie.
        deps = self._deps(
            settings, mocker, regime_aware_enabled=True, regime_tagging_enabled=False
        )

        assert deps.regime_tagging_enabled is False

    def test_explicit_on_without_regime_aware(
        self, settings, mock_external_clients, mocker
    ) -> None:
        deps = self._deps(
            settings, mocker, regime_aware_enabled=False, regime_tagging_enabled=True
        )

        assert deps.regime_tagging_enabled is True


class TestResolveRegimeHysteresis:
    """ALR-03 — `_resolve_regime` czyta ostatni stan z `regime_history` i zapisuje
    nowy raz na cykl. Flaga off albo błąd odczytu = dokładnie dzisiejszy surowy
    reżim; histereza może tylko przedłużyć RISK_OFF (wyższy próg)."""

    _CYCLE_AT = datetime(2026, 10, 19, 13, 30, tzinfo=UTC)

    @staticmethod
    def _settings(**overrides: object) -> MagicMock:
        settings = MagicMock()
        settings.regime_aware_enabled = True
        settings.yield_curve_enabled = False
        settings.regime_threshold_max_multiplier = 2.0
        settings.regime_hysteresis_enabled = True
        settings.regime_exit_confirmations = 2
        for name, value in overrides.items():
            setattr(settings, name, value)
        return settings

    @staticmethod
    def _calm_macro() -> MagicMock:
        # Makro NORMAL bez sygnałów → surowy reżim RISK_ON (mnożnik 1.0).
        report = MagicMock()
        report.overall_alert = MacroAlertLevel.NORMAL
        return report

    @staticmethod
    def _repo(previous: object = None) -> MagicMock:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_latest_regime_state.return_value = previous
        return repo

    def _resolve(self, settings: MagicMock, repo: MagicMock) -> tuple[str, float]:
        return main_agent._resolve_regime(
            settings, self._calm_macro(), None, repository=repo, cycle_at=self._CYCLE_AT
        )

    def test_flag_off_leaves_the_history_untouched(self) -> None:
        from src.domain.regime import MarketRegime, RegimeState

        repo = self._repo(RegimeState(regime=MarketRegime.RISK_OFF))

        _, multiplier = self._resolve(self._settings(regime_hysteresis_enabled=False), repo)

        assert multiplier == 1.0
        repo.get_latest_regime_state.assert_not_called()
        repo.save_regime_state.assert_not_called()

    def test_regime_off_touches_no_history(self) -> None:
        repo = self._repo()

        result = self._resolve(self._settings(regime_aware_enabled=False), repo)

        assert result == ("", 1.0)
        repo.get_latest_regime_state.assert_not_called()
        repo.save_regime_state.assert_not_called()

    def test_first_cycle_without_history_uses_the_raw_regime(self) -> None:
        from src.domain.regime import MarketRegime, RegimeDetector, RegimeState

        repo = self._repo(None)

        context, multiplier = self._resolve(self._settings(), repo)

        assert (context, multiplier) == (RegimeDetector().label(MarketRegime.RISK_ON), 1.0)
        repo.save_regime_state.assert_called_once_with(
            self._CYCLE_AT, MarketRegime.RISK_ON, RegimeState(regime=MarketRegime.RISK_ON)
        )

    def test_single_calm_reading_keeps_risk_off_and_the_raised_threshold(self) -> None:
        from src.domain.regime import MarketRegime, RegimeDetector, RegimeState

        repo = self._repo(RegimeState(regime=MarketRegime.RISK_OFF))

        context, multiplier = self._resolve(self._settings(), repo)

        assert (context, multiplier) == (RegimeDetector().label(MarketRegime.RISK_OFF), 2.0)
        repo.save_regime_state.assert_called_once_with(
            self._CYCLE_AT,
            MarketRegime.RISK_ON,
            RegimeState(regime=MarketRegime.RISK_OFF, pending=MarketRegime.RISK_ON, streak=1),
        )

    def test_second_calm_reading_exits_risk_off(self) -> None:
        from src.domain.regime import MarketRegime, RegimeState

        repo = self._repo(
            RegimeState(regime=MarketRegime.RISK_OFF, pending=MarketRegime.RISK_ON, streak=1)
        )

        _, multiplier = self._resolve(self._settings(), repo)

        assert multiplier == 1.0
        saved_state = repo.save_regime_state.call_args.args[2]
        assert saved_state == RegimeState(regime=MarketRegime.RISK_ON)

    def test_exit_confirmations_come_from_settings(self) -> None:
        from src.domain.regime import MarketRegime, RegimeState

        repo = self._repo(
            RegimeState(regime=MarketRegime.RISK_OFF, pending=MarketRegime.RISK_ON, streak=1)
        )

        _, multiplier = self._resolve(self._settings(regime_exit_confirmations=3), repo)

        assert multiplier == 2.0

    def test_read_failure_falls_back_to_the_raw_regime_and_logs(self, caplog) -> None:
        import logging

        repo = self._repo()
        repo.get_latest_regime_state.side_effect = RuntimeError("PGRST205 no table")

        with caplog.at_level(logging.ERROR, logger="main_agent"):
            _, multiplier = self._resolve(self._settings(), repo)

        assert multiplier == 1.0
        repo.save_regime_state.assert_not_called()
        assert any("regime_history" in rec.message for rec in caplog.records)

    def test_write_failure_keeps_the_hysteresis_result_and_logs(self, caplog) -> None:
        import logging

        from src.domain.regime import MarketRegime, RegimeState

        repo = self._repo(RegimeState(regime=MarketRegime.RISK_OFF))
        repo.save_regime_state.side_effect = RuntimeError("timeout")

        with caplog.at_level(logging.ERROR, logger="main_agent"):
            _, multiplier = self._resolve(self._settings(), repo)

        assert multiplier == 2.0
        assert any("regime_history" in rec.message for rec in caplog.records)

    def test_main_passes_the_repository_and_cycle_start(
        self, settings, mock_external_clients, mocker
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        resolve = mocker.patch("main_agent._resolve_regime", return_value=("", 1.0))

        main_agent.main(settings)

        resolve.assert_called_once()
        kwargs = resolve.call_args.kwargs
        assert isinstance(kwargs["repository"], RepositoryPort)
        assert kwargs["cycle_at"].tzinfo is not None


class TestRegimeBannerWiring:
    """ALR-05 — `_resolve_regime` oddaje reżim cyklu do nagłówka maila (sink
    `on_banner`), bez zmiany zwracanej pary (kontekst promptu, mnożnik)."""

    _CYCLE_AT = datetime(2026, 10, 19, 13, 30, tzinfo=UTC)

    @staticmethod
    def _settings(**overrides: object) -> MagicMock:
        settings = MagicMock()
        settings.regime_aware_enabled = True
        settings.yield_curve_enabled = False
        settings.regime_threshold_max_multiplier = 1.5
        settings.regime_hysteresis_enabled = False
        settings.volatility_threshold = Decimal("0.02")
        for name, value in overrides.items():
            setattr(settings, name, value)
        return settings

    @staticmethod
    def _macro(level: MacroAlertLevel) -> MagicMock:
        report = MagicMock()
        report.overall_alert = level
        return report

    def test_risk_off_banner_carries_multiplier_threshold_and_reason(self) -> None:
        from src.domain.regime import MarketRegime

        banners: list[object] = []

        result = main_agent._resolve_regime(
            self._settings(), self._macro(MacroAlertLevel.CRITICAL), None,
            on_banner=banners.append,
        )

        assert result == (main_agent.RegimeDetector().label(MarketRegime.RISK_OFF), 1.5)
        [banner] = banners
        assert banner == main_agent.RegimeBanner(
            regime=MarketRegime.RISK_OFF,
            multiplier=1.5,
            base_threshold=Decimal("0.02"),
            reason="alert CRITICAL",
        )

    def test_hysteresis_hold_is_the_reason(self) -> None:
        from src.domain.regime import MarketRegime, RegimeState

        repo = MagicMock(spec=RepositoryPort)
        repo.get_latest_regime_state.return_value = RegimeState(regime=MarketRegime.RISK_OFF)
        banners: list[object] = []

        main_agent._resolve_regime(
            self._settings(regime_hysteresis_enabled=True, regime_exit_confirmations=2),
            self._macro(MacroAlertLevel.NORMAL),
            None,
            repository=repo,
            cycle_at=self._CYCLE_AT,
            on_banner=banners.append,
        )

        [banner] = banners
        assert banner.regime is MarketRegime.RISK_OFF
        assert banner.reason == "histereza: trzymam RISK_OFF (surowy odczyt RISK_ON)"

    def test_regime_off_sends_no_banner(self) -> None:
        banners: list[object] = []

        main_agent._resolve_regime(
            self._settings(regime_aware_enabled=False),
            self._macro(MacroAlertLevel.CRITICAL),
            None,
            on_banner=banners.append,
        )

        assert banners == []

    def test_main_puts_the_banner_into_the_report_context(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.domain.regime import MarketRegime

        banner = main_agent.RegimeBanner(
            regime=MarketRegime.RISK_OFF,
            multiplier=2.0,
            base_threshold=Decimal("0.02"),
            reason="alert CRITICAL",
        )

        def resolve(*_args: object, on_banner=None, **_kwargs: object) -> tuple[str, float]:
            on_banner(banner)
            return "Ucieczka od ryzyka", 2.0

        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent._resolve_regime", side_effect=resolve)
        dispatch = mocker.patch("main_agent._dispatch_reports")

        main_agent.main(settings)

        assert dispatch.call_args.kwargs["context"].regime == banner


class TestGateDeltaWiring:
    """EXP-01 — flaga z Settings trafia do `AgentGraphDeps`; off (domyślnie) =
    rekord predykcji nie niesie `gate_delta`, którego przed migracją 034 nie ma."""

    def test_flag_defaults_off_in_graph_deps(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].gate_delta_enabled is False

    def test_flag_reaches_graph_deps(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.gate_delta_enabled = True

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].gate_delta_enabled is True


class TestDriftGateWiring:
    """ALR-10 — flaga bramki dryfu z Settings trafia do `AgentGraphDeps`; off
    (domyślnie) = use case nie czyta referencji dryfu, bramka jak przed ALR-10."""

    def test_flag_defaults_off_in_graph_deps(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].drift_gate_enabled is False

    def test_flag_reaches_graph_deps(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.drift_gate_enabled = True

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].drift_gate_enabled is True

class TestTrendProbabilitiesWiring:
    """ML-29 — flaga z Settings trafia do `AgentGraphDeps`; off (domyślnie) = prompt
    bez rozkładu i rekord bez `trend_probabilities` (kolumna dopiero w migracji 051)."""

    @pytest.mark.parametrize("enabled", [False, True])
    def test_flag_reaches_graph_deps(
        self, settings, mock_external_clients, mocker, enabled: bool
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.trend_probabilities_enabled = enabled

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].trend_probabilities_enabled is enabled


class TestStaleReferenceWiring:
    """FIN-14 — zegar cyklu dociera do use case'u (wiek referencji ceny liczy się
    od startu cyklu, jak nagłówek maila), a stęchłe referencje dowodzą historii."""

    def test_cycle_clock_reaches_the_use_case(self) -> None:
        use_case = MagicMock()
        use_case.run.return_value = {"status": "ignored", "delta": "0.001"}
        cycle_at = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

        main_agent._process_one_symbol(
            "AAPL",
            peer_ctx=(),
            use_case=use_case,
            crypto_set=set(),
            etf_set=set(),
            regime_context="",
            regime_multiplier=1.0,
            collect_alpha=False,
            alpha_use_case=MagicMock(),
            cycle_at=cycle_at,
        )

        assert use_case.run.call_args.kwargs["now"] == cycle_at

    def test_stale_references_keep_new_symbols_from_a_first_run(self) -> None:
        results = [
            main_agent.SymbolResult(symbol=f"S{i}", status="ignored",
                                    skip_reason=SkipReason.STALE_REFERENCE)
            for i in range(10)
        ]
        results.append(
            main_agent.SymbolResult(symbol="NEW", status="ignored",
                                    skip_reason=SkipReason.COLD_START)
        )

        assert main_agent._classify_cycle_maturity(results) is CycleMaturity.STEADY_STATE


class TestPortfolioChangeDispatch:
    """PORT-06: zmiana portfela od ostatniego maila trafia do właściciela (HTML
    i tekst), nigdy do obcego subskrybenta ani do starego pushu (BEZ-01, decyzja
    2026-10-19), i nie wchodzi do tematu maila."""

    _RESULTS = [
        main_agent.SymbolResult(
            symbol="AAPL", status="saved", delta=Decimal("0.10"),
            current_price=Decimal("110"),
        ),
    ]
    _CHANGE = "+$20,00 (+10,0%)"

    def test_owner_sees_the_change_and_a_subscriber_does_not(self, settings, mocker) -> None:
        sent = TestDispatchReports()._fan_out(settings, mocker, list(self._RESULTS))

        owner = sent["Owner@Example.com"]
        friend = sent[TestDispatchReports.FRIEND]
        assert "Twój portfel od ostatniego maila" in owner["html_body"]
        assert self._CHANGE in owner["html_body"]
        assert "Twój portfel od ostatniego maila" in owner["plain_text"]
        assert "$20,00" not in friend["html_body"]
        assert "$20,00" not in friend["plain_text"]
        assert "od ostatniego maila" not in friend["html_body"]
        assert "od ostatniego maila" not in friend["plain_text"]

    def test_change_stays_out_of_the_subject(self, settings, mocker) -> None:
        sent = TestDispatchReports()._fan_out(settings, mocker, list(self._RESULTS))

        for mail in sent.values():
            assert "Twój portfel" not in mail["subject"]
            assert "$20,00" not in mail["subject"]

    def test_legacy_push_text_carries_no_portfolio_change(self, settings, mocker) -> None:
        from src.application.quota_monitor import QuotaMonitor

        settings.notifications_enabled = True
        settings.resend_api_key = "re_test"
        settings.digest_to_email = "you@example.com"
        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch(
            "main_agent._build_user_portfolio",
            return_value=TestHeldPositionsOutsideWatchlist._portfolio("AAPL"),
        )
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        mocker.patch("main_agent._publish_public_scorecard")
        email = MagicMock()
        legacy_push = MagicMock()

        main_agent._dispatch_reports(
            results=list(self._RESULTS),
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=MagicMock(spec=RepositoryPort),
            quota_monitor=QuotaMonitor(),
            notifier=email,
            subscribers=[],
            all_symbols=["AAPL"],
            context=main_agent.ReportContext(),
            legacy_push_notifier=legacy_push,
        )

        email_text = email.send_report.call_args.kwargs["plain_text"]
        push_text = legacy_push.send_report.call_args.kwargs["plain_text"]
        assert "Twój portfel od ostatniego maila: +10,0% (+$20,00)" in email_text
        assert "od ostatniego maila" not in push_text
        assert "$20,00" not in push_text


class TestPortfolioClustersFromConfigSectors:
    """PORT-05: `peer_groups = {}` w config.toml wyłączało klastry w sizingu
    i ostrzeżenie o koncentracji. Teraz grupy idą z `sectors`."""

    def test_production_config_maps_nvda_to_its_sector(self, monkeypatch):
        monkeypatch.delenv("STOCKAGENT_DISABLE_TOML", raising=False)
        monkeypatch.delenv("STOCKAGENT_TOML_FILE", raising=False)
        settings = Settings(
            _env_file=None,
            openai_api_key="sk-test",
            finnhub_api_key="fh",
            alpha_vantage_api_keys=["av1"],
            supabase_url="https://test.supabase.co",
            supabase_key="anon",
        )

        clusters = main_agent._portfolio_clusters(settings)

        assert clusters.get("NVDA")
        assert clusters["AMD"] == clusters["NVDA"]
        assert "VT" not in clusters
        assert "BTC" not in clusters


class TestPositionRiskWiring:
    """PORT-04: VaR, obsunięcie i hedge realnego portfela zwężają bandę — tylko
    za flagą `portfolio_var_enabled` i tylko przy realnych pozycjach. Odczyt
    darmowej historii cen, zero płatnych portów."""

    @staticmethod
    def _portfolio() -> "main_agent.Portfolio":
        return TestHeldPositionsOutsideWatchlist._portfolio("NVDA", "SH")

    def test_flag_off_reads_nothing(self, settings) -> None:
        repository = MagicMock(spec=RepositoryPort)

        risk = main_agent._build_position_risk(settings, repository, self._portfolio())

        assert risk is None
        repository.get_price_history.assert_not_called()

    def test_no_real_positions_give_no_risk(self, settings) -> None:
        settings.portfolio_var_enabled = True
        repository = MagicMock(spec=RepositoryPort)

        assert main_agent._build_position_risk(settings, repository, None) is None
        repository.get_price_history.assert_not_called()

    def test_flag_on_uses_held_positions_and_equity_hedges_only(
        self, settings, mocker
    ) -> None:
        from src.application.use_cases.portfolio_risk import PositionRisk
        from src.domain.macro_risk import MacroRiskInstrumentType as T

        settings.portfolio_var_enabled = True
        settings.risk_symbols = ["SH", "GLD", "VIXY", "EPOL"]
        settings.risk_symbol_types = {
            "SH": T.INVERSE_EQUITY,
            "GLD": T.SAFE_HAVEN,
            "VIXY": T.VOLATILITY,
            "EPOL": T.SOVEREIGN_PROXY,
        }
        computed = mocker.patch.object(
            main_agent.PortfolioRiskUseCase,
            "position_risk",
            return_value=PositionRisk(var=0.04),
        )
        portfolio = self._portfolio()

        risk = main_agent._build_position_risk(
            settings, MagicMock(spec=RepositoryPort), portfolio
        )

        assert risk == PositionRisk(var=0.04)
        # GLD i EPOL to inwestycja, nie hedge — nie mogą obniżać bandy.
        computed.assert_called_once_with(
            portfolio, hedge_symbols=frozenset({"SH", "VIXY"}), rates=None
        )

    def test_failure_degrades_to_no_risk(self, settings, mocker) -> None:
        settings.portfolio_var_enabled = True
        mocker.patch.object(
            main_agent.PortfolioRiskUseCase,
            "position_risk",
            side_effect=RuntimeError("supabase down"),
        )

        risk = main_agent._build_position_risk(
            settings, MagicMock(spec=RepositoryPort), self._portfolio()
        )

        assert risk is None

    def test_dispatch_hands_position_risk_to_the_report(self, settings, mocker) -> None:
        from src.application.ports import MarketDataPort
        from src.application.use_cases.portfolio_risk import PositionRisk

        expected = PositionRisk(var=0.05)
        build = mocker.patch("main_agent._build_position_risk", return_value=expected)
        portfolio = self._portfolio()

        report, repository, _, _ = TestHeldPositionsOutsideWatchlist()._dispatch(
            settings, mocker, portfolio, MagicMock(spec=MarketDataPort)
        )

        build.assert_called_once_with(settings, repository, portfolio, rates=None)
        assert report.call_args_list[0].args[3].position_risk == expected

    def test_non_owner_context_drops_position_risk(self) -> None:
        from src.application.use_cases.portfolio_risk import PositionRisk

        context = main_agent.ReportContext(position_risk=PositionRisk(var=0.05))

        assert main_agent._without_owner_portfolio(context).position_risk is None



class TestSectorMapFromConfig:
    """PORT-05: etykiety sektorów i peerzy Warte uwagi idą z Settings (config.toml)
    do `ReportContext` — jedno źródło z `peer_groups`."""

    def test_sector_map_combines_sectors_etfs_and_crypto(self, settings) -> None:
        settings.sectors = {"Półprzewodniki/AI": ["NVDA"]}
        settings.symbols_etf = ["VT"]
        settings.crypto_symbols = ["BTC"]

        assert main_agent._sector_map(settings) == {
            "NVDA": "Półprzewodniki/AI", "VT": "ETF", "BTC": "Krypto"
        }

    def test_portfolio_sectors_count_by_the_config_map(self) -> None:
        results = [
            main_agent.SymbolResult(symbol=symbol, status="ignored")
            for symbol in ("NVDA", "AMD", "BTC", "ZZZZ")
        ]
        sector_map = {"NVDA": "Półprzewodniki/AI", "AMD": "Półprzewodniki/AI", "BTC": "Krypto"}

        assert main_agent._portfolio_sectors(results, sector_map) == {
            "Półprzewodniki/AI": 2, "Krypto": 1, "Inne": 1
        }

    def test_dispatch_hands_the_config_map_and_peers_to_the_report(
        self, settings, mocker
    ) -> None:
        from src.application.ports import MarketDataPort

        settings.sectors = {"Big Tech": ["AAPL"]}
        settings.sector_peers = {"Big Tech": ["NFLX"]}

        report, _, _, _ = TestHeldPositionsOutsideWatchlist()._dispatch(
            settings, mocker, None, MagicMock(spec=MarketDataPort)
        )

        context = report.call_args_list[0].args[3]
        assert context.sector_map == {"AAPL": "Big Tech"}
        assert context.sector_peers == {"Big Tech": ["NFLX"]}
        assert context.portfolio_sectors == {"Big Tech": 1}


class TestPositionsPortCurrencyColumns:
    """PORT-18: `positions_currency_enabled` przełącza adapter pozycji na kolumny
    migracji 035 (waluta + prawdziwa data zakupu); bez flagi zostaje stary odczyt."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_flag_reaches_the_positions_adapter(self, settings, mocker, enabled):
        adapter_cls = mocker.patch(
            "src.infrastructure.adapters.positions_repo.SupabasePositionsAdapter"
        )
        settings = settings.model_copy(
            update={"portfolio_positions_enabled": True, "positions_currency_enabled": enabled}
        )

        main_agent.build_positions_port(settings)

        assert adapter_cls.call_args.kwargs["currency_columns"] is enabled


class TestPortfolioFxWiring:
    """PORT-11: kursy NBP (bieżące i z dni zakupu) do „P/L zł” — darmowe, za
    `nbp_enabled`, tylko z portfelem właściciela; cache `fx_rates` za flagą."""

    def test_fx_port_without_the_cache_flag_is_plain_nbp(self, settings, mocker) -> None:
        from src.infrastructure.adapters.nbp_client import NbpFxRateClient

        create = mocker.patch("src.infrastructure.adapters.fx_rates_cache.create_client")

        port = main_agent.build_fx_port(settings)

        assert isinstance(port, NbpFxRateClient)
        create.assert_not_called()

    def test_fx_port_with_the_cache_flag_reads_through_fx_rates(self, settings, mocker) -> None:
        from src.infrastructure.adapters.fx_rates_cache import SupabaseCachedFxRates

        mocker.patch("src.infrastructure.adapters.fx_rates_cache.create_client")
        settings = settings.model_copy(update={"fx_rates_cache_enabled": True})

        assert isinstance(main_agent.build_fx_port(settings), SupabaseCachedFxRates)

    def test_rates_are_collected_for_the_owner_portfolio(self, settings, mocker) -> None:
        from datetime import date

        from src.application.ports import FxRatePort, MarketDataPort
        from src.domain.fx import FxRate
        from src.domain.portfolio import PlnRates

        settings.nbp_enabled = True
        fx = MagicMock(spec=FxRatePort)
        fx.get_rate.side_effect = lambda currency, day: FxRate(
            currency=currency, on_date=day, effective_date=day, rate=Decimal("3.70")
        )
        build_fx = mocker.patch("main_agent.build_fx_port", return_value=fx)
        portfolio = TestHeldPositionsOutsideWatchlist._portfolio("AAPL")

        report, _, _, _ = TestHeldPositionsOutsideWatchlist()._dispatch(
            settings, mocker, portfolio, MagicMock(spec=MarketDataPort)
        )

        build_fx.assert_called_once_with(settings)
        # 05:30 UTC 18.10 to 18.10 w Warszawie — dzień kursu bieżącego.
        assert report.call_args_list[0].args[3].portfolio_fx == PlnRates(
            current={"USD": Decimal("3.70")},
            on_purchase={("USD", date(2026, 10, 1)): Decimal("3.70")},
        )

    def test_no_rates_without_nbp(self, settings, mocker) -> None:
        from src.application.ports import MarketDataPort

        build_fx = mocker.patch("main_agent.build_fx_port")
        report, _, _, _ = TestHeldPositionsOutsideWatchlist()._dispatch(
            settings, mocker, TestHeldPositionsOutsideWatchlist._portfolio("AAPL"),
            MagicMock(spec=MarketDataPort),
        )

        build_fx.assert_not_called()
        assert report.call_args_list[0].args[3].portfolio_fx is None

    def test_no_rates_without_a_portfolio(self, settings, mocker) -> None:
        from src.application.ports import MarketDataPort

        settings.nbp_enabled = True
        build_fx = mocker.patch("main_agent.build_fx_port")
        report, _, _, _ = TestHeldPositionsOutsideWatchlist()._dispatch(
            settings, mocker, None, MagicMock(spec=MarketDataPort)
        )

        build_fx.assert_not_called()
        assert report.call_args_list[0].args[3].portfolio_fx is None

    def test_failed_rate_collection_leaves_the_report_without_rates(
        self, settings, mocker
    ) -> None:
        from src.application.ports import MarketDataPort

        settings.nbp_enabled = True
        mocker.patch("main_agent.build_fx_port", side_effect=RuntimeError("boom"))
        report, _, _, _ = TestHeldPositionsOutsideWatchlist()._dispatch(
            settings, mocker, TestHeldPositionsOutsideWatchlist._portfolio("AAPL"),
            MagicMock(spec=MarketDataPort),
        )

        assert report.call_args_list[0].args[3].portfolio_fx is None

    def test_non_owner_context_drops_the_rates(self) -> None:
        from src.domain.portfolio import PlnRates

        context = main_agent.ReportContext(portfolio_fx=PlnRates(current={"USD": Decimal("4")}))

        assert main_agent._without_owner_portfolio(context).portfolio_fx is None


class TestUserPortfolioPurchaseDates:
    """PORT-11: bez migracji 035 adapter podaje `as_of` jako datę — to data
    aktualizacji tabeli, nie zakupu, więc portfel raportu jej nie dostaje."""

    def _build(self, settings, mocker, *, currency_columns: bool):
        from datetime import date

        from src.domain.portfolio import Position

        port = MagicMock()
        port.get_positions.return_value = [
            Position(
                symbol="AAPL",
                quantity=Decimal("1"),
                purchase_price=Decimal("10"),
                purchase_date=date(2026, 7, 10),
            )
        ]
        port.get_as_of.return_value = datetime(2026, 7, 10, tzinfo=UTC)
        mocker.patch("main_agent.build_positions_port", return_value=port)
        settings = settings.model_copy(
            update={
                "portfolio_positions_enabled": True,
                "positions_currency_enabled": currency_columns,
            }
        )
        return main_agent._build_user_portfolio(settings)

    def test_without_the_migration_purchase_dates_are_unknown(self, settings, mocker) -> None:
        portfolio = self._build(settings, mocker, currency_columns=False)

        assert portfolio is not None
        assert portfolio.positions[0].purchase_date is None

    def test_with_the_migration_purchase_dates_are_kept(self, settings, mocker) -> None:
        from datetime import date

        portfolio = self._build(settings, mocker, currency_columns=True)

        assert portfolio is not None
        assert portfolio.positions[0].purchase_date == date(2026, 7, 10)



class TestPositionRiskInPln:
    """Wagi VaR/obsunięcia/hedge'a portfela w złotych tymi samymi kursami NBP co raport."""

    def test_dispatch_passes_the_portfolio_rates_to_position_risk(self, settings, mocker) -> None:
        from src.application.ports import FxRatePort, MarketDataPort
        from src.domain.fx import FxRate

        settings.nbp_enabled = True
        fx = MagicMock(spec=FxRatePort)
        fx.get_rate.side_effect = lambda currency, day: FxRate(
            currency=currency, on_date=day, effective_date=day, rate=Decimal("3.70")
        )
        mocker.patch("main_agent.build_fx_port", return_value=fx)
        build = mocker.patch("main_agent._build_position_risk", return_value=None)

        report, _, _, _ = TestHeldPositionsOutsideWatchlist()._dispatch(
            settings, mocker, TestHeldPositionsOutsideWatchlist._portfolio("AAPL"),
            MagicMock(spec=MarketDataPort),
        )

        assert build.call_args.kwargs["rates"] == report.call_args_list[0].args[3].portfolio_fx
        assert build.call_args.kwargs["rates"].current == {"USD": Decimal("3.70")}

    def test_position_risk_gets_the_rates(self, settings, mocker) -> None:
        from datetime import date

        from src.application.use_cases.portfolio_risk import PositionRisk
        from src.domain.portfolio import PlnRates, Position

        settings.portfolio_var_enabled = True
        computed = mocker.patch.object(
            main_agent.PortfolioRiskUseCase, "position_risk", return_value=PositionRisk()
        )
        portfolio = main_agent.Portfolio(
            positions=(
                Position(symbol="AAPL", quantity=Decimal("1"), purchase_price=Decimal("1"),
                         purchase_date=date(2026, 1, 2)),
            ),
            as_of=datetime(2026, 10, 18, tzinfo=UTC),
        )
        rates = PlnRates(current={"USD": Decimal("4")})

        main_agent._build_position_risk(
            settings, MagicMock(spec=RepositoryPort), portfolio, rates=rates
        )

        assert computed.call_args.kwargs["rates"] is rates


class TestPaperCostWiring:
    """WIA-50: paper-PnL w mailu, na scorecardzie i w backteście liczony po
    kosztach z config.toml — naliczanych tylko przy zmianie pozycji symbolu."""

    @staticmethod
    def _costs() -> "main_agent.PaperCosts":
        return main_agent.PaperCosts(cost_bps=10.0, crypto_cost_bps=25.0)

    def test_paper_costs_come_from_settings(self, settings) -> None:
        settings.crypto_symbols = ["BTC"]

        assert main_agent._paper_costs(settings) == main_agent.PaperCosts(
            cost_bps=10, crypto_cost_bps=25, crypto_symbols=frozenset({"BTC"})
        )

    def test_track_record_curve_is_net_of_entry_costs(self) -> None:
        settings = TestHonestTrackRecordWiring._settings()

        equity, _, _ = main_agent._build_track_record(
            settings, TestHonestTrackRecordWiring._repo(), self._costs()
        )

        # NVDA i AMD wchodzą w tym samym cyklu: 0.10 − 0.001 na każdej połowie.
        assert equity is not None
        assert equity.total_return == pytest.approx(0.099)

    def test_track_record_stats_carry_the_costs_for_the_label(self) -> None:
        costs = self._costs()

        stats = main_agent._build_track_record_stats(
            TestHonestTrackRecordWiring._settings(), TestHonestTrackRecordWiring._repo(), costs
        )

        assert stats is not None
        assert stats.costs == costs

    def test_costs_alone_follow_the_equity_curve_flag(self) -> None:
        settings = TestHonestTrackRecordWiring._settings(
            equity_curve_enabled=False, calibration_curve_enabled=False
        )

        assert (
            main_agent._build_track_record_stats(
                settings, TestHonestTrackRecordWiring._repo(), self._costs()
            )
            is None
        )

    def test_scorecard_is_net_of_costs_and_names_them(self) -> None:
        publisher = MagicMock()

        main_agent._publish_public_scorecard(
            TestHonestTrackRecordWiring._settings(),
            TestHonestTrackRecordWiring._repo(),
            publisher,
            self._costs(),
        )

        page = publisher.publish.call_args.args[0]
        assert "paper-PnL (koszt 10 bps)" in page
        assert "+9.9%" in page

    def test_dispatch_passes_the_configured_costs_everywhere(self, settings, mocker) -> None:
        from src.application.quota_monitor import QuotaMonitor

        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        track_record = mocker.patch(
            f"{ASSEMBLY}.build_track_record", return_value=(None, None, None)
        )
        stats = mocker.patch(f"{ASSEMBLY}.build_track_record_stats", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch("main_agent._build_user_portfolio", return_value=None)
        mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        scorecard = mocker.patch("main_agent._publish_public_scorecard")
        repository = MagicMock(spec=RepositoryPort)

        main_agent._dispatch_reports(
            results=[],
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=repository,
            quota_monitor=QuotaMonitor(),
            notifier=MagicMock(),
            subscribers=[],
            all_symbols=["AAPL"],
            context=main_agent.ReportContext(),
        )

        expected = main_agent._paper_costs(settings)
        assert track_record.call_args.args[2] == expected
        assert stats.call_args.args[2] == expected
        assert scorecard.call_args.args[3] == expected


def test_backtest_uses_the_paper_cost_and_logs_net_return(
    settings, mock_external_clients, mocker, caplog
):
    from src.domain.backtest import BacktestSummary

    fake_uc = MagicMock()
    fake_uc.run.return_value = BacktestSummary(
        n_folds=4,
        mean_candidate_rmse=0.03,
        mean_baseline_rmse=0.05,
        mean_hit_rate=0.6,
        improvement_pct=40.0,
        beats_baseline=True,
        verdict="model bije baseline o 40.0%",
        mean_net_return=0.0012,
        turnover=0.25,
    )
    use_case_cls = mocker.patch("main_backtest.BacktestUseCase", return_value=fake_uc)

    with caplog.at_level("INFO", logger="main_backtest"):
        main_backtest.main(settings)

    assert use_case_cls.call_args.kwargs["cost_bps"] == settings.paper_cost_bps
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("AAPL"))
    assert "net=0.00120" in line
    assert "turnover=0.25" in line
    assert f"cost={settings.paper_cost_bps:g}bps" in line


class TestBenchmarkWiring:
    """WIA-21: koszyk i IVV z darmowych snapshotów (`price_snapshots`), w tych
    samych oknach i z tymi samymi kosztami co krzywa agenta — zero płatnych."""

    @staticmethod
    def _settings(**overrides: object) -> MagicMock:
        settings = TestHonestTrackRecordWiring._settings(**overrides)
        settings.reflection_min_age_hours = 6
        return settings

    @staticmethod
    def _repo() -> MagicMock:
        from src.domain.value_objects import Money

        repo = TestHonestTrackRecordWiring._repo()
        # Predykcje z 2026-10-17 14:00 UTC; IVV z tego cyklu i z cyklu rozliczenia.
        repo.get_price_history.return_value = [
            (datetime(2026, 10, 17, 13, 58, tzinfo=UTC), Money(Decimal("500"))),
            (datetime(2026, 10, 18, 5, 31, tzinfo=UTC), Money(Decimal("510"))),
        ]
        return repo

    @staticmethod
    def _costs() -> "main_agent.PaperCosts":
        return main_agent.PaperCosts(cost_bps=10.0, crypto_cost_bps=25.0)

    def test_stats_carry_the_comparison_from_free_snapshots(self) -> None:
        repo = self._repo()

        stats = main_agent._build_track_record_stats(self._settings(), repo, self._costs())

        assert stats is not None
        assert stats.benchmark is not None
        assert stats.benchmark.index_symbol == "IVV"
        assert stats.benchmark.index is not None
        assert stats.benchmark.index.total_return == pytest.approx(0.02 - 0.001)
        # Okno IVV o dzień dłuższe: snapshot startu bywa sprzed odcięcia okna predykcji.
        repo.get_price_history.assert_called_once_with("IVV", 31)

    def test_without_costs_no_benchmark_is_read(self) -> None:
        repo = self._repo()

        stats = main_agent._build_track_record_stats(self._settings(), repo)

        assert stats is not None
        assert stats.benchmark is None
        repo.get_price_history.assert_not_called()

    def test_benchmark_read_error_keeps_the_basket(self) -> None:
        repo = self._repo()
        repo.get_price_history.side_effect = Exception("supabase down")

        stats = main_agent._build_track_record_stats(self._settings(), repo, self._costs())

        assert stats is not None
        assert stats.benchmark is not None
        assert stats.benchmark.index is None
        assert stats.benchmark.basket.total_return == pytest.approx(0.099)

    def test_equity_flag_off_reads_no_benchmark(self) -> None:
        repo = self._repo()
        settings = self._settings(equity_curve_enabled=False)

        stats = main_agent._build_track_record_stats(settings, repo, self._costs())

        assert stats is None or stats.benchmark is None
        repo.get_price_history.assert_not_called()

    def test_scorecard_shows_the_comparison(self) -> None:
        publisher = MagicMock()

        main_agent._publish_public_scorecard(
            self._settings(), self._repo(), publisher, self._costs()
        )

        page = publisher.publish.call_args.args[0]
        assert "agent +9.9% · koszyk +9.9% · S&amp;P 500 +1.9%" in page
        assert "IVV" not in page
        assert "NVDA" not in page
        assert "AMD" not in page


class TestExperimentWiring:
    """EXP-05 — definicje eksperymentów i flaga kolumny `experiment_arms`
    (migracja 037) trafiają do `AgentGraphDeps`; flaga domyślnie off."""

    def test_column_flag_defaults_off_and_no_experiments(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")

        main_agent.build_use_case(settings)

        deps = use_case_cls.call_args.args[0]
        assert deps.experiment_arms_enabled is False
        assert deps.experiments == ()

    def test_definitions_and_flag_reach_graph_deps(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from datetime import date

        from src.config import ExperimentSettings

        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.experiment_arms_enabled = True
        settings.experiments = {
            "EXP-03": ExperimentSettings(
                hypothesis="Pamięć pomaga.",
                arms=["memory", "no_memory"],
                start=date(2026, 11, 2),
                end=date(2027, 1, 29),
                primary_metric="hit_rate",
            )
        }

        main_agent.build_use_case(settings)

        deps = use_case_cls.call_args.args[0]
        assert deps.experiment_arms_enabled is True
        assert [experiment.id for experiment in deps.experiments] == ["EXP-03"]


class TestToolUseAdapterGetsSettingsSecrets:
    """Runda 2 BEZ-07: adapter tool-use redaguje błąd narzędzia wartościami
    sekretów, które składa mu DI Container (adapter nie czyta Settings)."""

    @pytest.mark.parametrize(
        ("provider", "client_path"),
        [
            ("openai", "src.infrastructure.llm.tool_calling_openai.OpenAI"),
            ("anthropic", "src.infrastructure.llm.tool_calling_anthropic.Anthropic"),
        ],
    )
    def test_factory_injects_secret_values(self, settings, mocker, provider, client_path):
        mocker.patch(client_path)
        s = settings.model_copy(
            update={
                "tool_use_enabled": True,
                "llm_provider": provider,
                "anthropic_api_key": "sk-ant-factory-test",
            }
        )

        adapter = main_agent.build_tool_use_adapter(s)

        assert adapter is not None
        assert adapter._secret_values == s.secret_values()


class TestSymbolErrorRedactsSettingsSecrets:
    """Runda 2 BEZ-07: błąd symbolu w mailu traci też dosłowne wartości sekretów
    z Settings — kształt bez wzorca (nie URL, nie nagłówek) przechodził przez regex."""

    ODD_SECRET = "Zx9!odd~shape#secret"

    def _kwargs(self, settings, use_case):
        return {
            "use_case": use_case,
            "settings": settings,
            "crypto_set": set(),
            "etf_set": set(),
            "regime_context": "",
            "regime_multiplier": 1.0,
            "collect_alpha": False,
            "alpha_use_case": MagicMock(),
        }

    @pytest.mark.parametrize("concurrency", [1, 4], ids=["sequential", "parallel"])
    def test_error_message_omits_literal_secret(self, settings, concurrency):
        s = settings.model_copy(
            update={"finnhub_api_key": self.ODD_SECRET, "symbol_concurrency": concurrency}
        )
        uc = MagicMock()
        uc.run.side_effect = RuntimeError(
            f"provider rejected credentials {self.ODD_SECRET} (code 7)"
        )

        results, failures, _s, _p = main_agent._analyze_symbols(["AAPL"], **self._kwargs(s, uc))

        assert failures == 1
        message = results[0].error_message
        assert message is not None
        assert "provider rejected credentials" in message
        assert self.ODD_SECRET not in message

    def test_process_one_symbol_redacts_passed_secret_values(self, settings):
        uc = MagicMock()
        uc.run.side_effect = RuntimeError(f"echo {self.ODD_SECRET}")

        proc = main_agent._process_one_symbol(
            "AAPL",
            peer_ctx=(),
            secret_values=(self.ODD_SECRET,),
            **{k: v for k, v in self._kwargs(settings, uc).items() if k != "settings"},
        )

        assert proc.result.error_message is not None
        assert self.ODD_SECRET not in proc.result.error_message


# --- BEZ-08: lokalny run tylko do odczytu wobec wspólnej bazy ----------------


def test_build_repository_blocks_writes_when_db_writes_disabled(settings, mocker) -> None:
    # Lokalny run i GHA piszą do tej samej bazy: bez strażnika każdy lokalny
    # przebieg nadpisywał snapshot i predykcję z tej samej godziny w produkcji.
    from src.domain.value_objects import Money

    client = mocker.patch("src.infrastructure.adapters.supabase_repo.create_client").return_value
    # Jawnie, choć to default (`TestDbWritesFlag`): test nie zależy od env powłoki.
    settings = settings.model_copy(update={"db_writes_enabled": False})

    repository = main_agent.build_repository(settings)
    repository.save_price_snapshot("AAPL", Money(Decimal("100")))
    prediction_id = repository.save_prediction({"symbol": "AAPL"})

    assert client.table.return_value.upsert.call_count == 0
    assert prediction_id.startswith("dry-run-")


@pytest.mark.parametrize("enabled", [False, True])
def test_build_repository_passes_the_council_votes_upsert_flag(
    settings, mocker, enabled: bool
) -> None:
    # NIEZ-14: upsert głosów rady potrzebuje UNIQUE z migracji 054 — za flagą.
    repo_cls = mocker.patch("main_agent.SupabaseRepository")
    settings.council_votes_upsert_enabled = enabled

    main_agent.build_repository(settings)

    assert repo_cls.call_args.kwargs["council_votes_upsert_enabled"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_build_repository_passes_the_persona_vote_outcomes_flag(
    settings, mocker, enabled: bool
) -> None:
    # WIA-51: RPC 045 i pasmo HOLD krypto — flaga i symbole krypto trafiają do adaptera.
    repo_cls = mocker.patch("main_agent.SupabaseRepository")
    settings.persona_vote_outcomes_enabled = enabled
    settings.crypto_symbols = ["BTC", "ETH"]

    main_agent.build_repository(settings)

    assert repo_cls.call_args.kwargs["persona_vote_outcomes_enabled"] is enabled
    assert list(repo_cls.call_args.kwargs["crypto_symbols"]) == ["BTC", "ETH"]


@pytest.mark.parametrize("enabled", [False, True])
def test_report_flags_carry_the_persona_brier_ranking_flag(settings, enabled: bool) -> None:
    # WIA-44: kolejność leaderboardu person to flaga raportu (ReportFlags), nie adaptera.
    settings.persona_brier_ranking_enabled = enabled

    flags = main_agent._report_flags(settings)

    assert flags.persona_brier_ranking_enabled is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_build_repository_passes_the_correction_category_flag(
    settings, mocker, enabled: bool
) -> None:
    # WIA-57: kolumna z migracji 046 — zapis kategorii diagnozy i odczyt lessons.
    repo_cls = mocker.patch("main_agent.SupabaseRepository")
    settings.correction_category_enabled = enabled

    main_agent.build_repository(settings)

    assert repo_cls.call_args.kwargs["correction_category_enabled"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_build_repository_passes_the_exact_rag_search_flag(
    settings, mocker, enabled: bool
) -> None:
    # ML-23: dokładne KNN analogów RAG (funkcja z migracji 063) za flagą.
    repo_cls = mocker.patch("main_agent.SupabaseRepository")
    settings.rag_exact_search_enabled = enabled

    main_agent.build_repository(settings)

    assert repo_cls.call_args.kwargs["rag_exact_search_enabled"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_repository_passes_the_correction_category_flag(
    settings, mocker, enabled: bool
) -> None:
    # WIA-53/57: backfill diagnoz w Slow Loopie zapisuje kategorię tą samą flagą.
    repo_cls = mocker.patch("main_trainer.SupabaseRepository")
    settings.correction_category_enabled = enabled

    main_trainer.build_repository(settings)

    assert repo_cls.call_args.kwargs["correction_category_enabled"] is enabled


def test_build_repository_writes_when_db_writes_enabled(settings, mocker) -> None:
    from src.domain.value_objects import Money
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

    client = mocker.patch("src.infrastructure.adapters.supabase_repo.create_client").return_value
    settings = settings.model_copy(update={"db_writes_enabled": True})

    repository = main_agent.build_repository(settings)
    repository.save_price_snapshot("AAPL", Money(Decimal("100")))

    assert isinstance(repository, SupabaseRepository)
    assert client.table.return_value.upsert.call_count == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_repository_follows_db_writes_flag(settings, mocker, enabled: bool) -> None:
    from src.infrastructure.adapters.write_guard import ReadOnlyRepository

    supabase = MagicMock(spec=RepositoryPort)
    mocker.patch("main_trainer.SupabaseRepository", return_value=supabase)
    settings = settings.model_copy(update={"db_writes_enabled": enabled})

    repository = main_trainer.build_repository(settings)
    repository.save_model_scorecard("AAPL", {"status": "skipped"})

    assert isinstance(repository, ReadOnlyRepository) is not enabled
    assert supabase.save_model_scorecard.call_count == int(enabled)


def test_trainer_use_case_gets_the_guarded_repository(settings, mocker) -> None:
    from src.infrastructure.adapters.write_guard import ReadOnlyRepository

    mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
    use_case_cls = mocker.patch("main_trainer.TrainModelUseCase")
    settings = settings.model_copy(update={"db_writes_enabled": False})

    main_trainer.build_use_case(settings)

    assert isinstance(use_case_cls.call_args.kwargs["db_port"], ReadOnlyRepository)


@pytest.mark.parametrize("enabled", [False, True])
def test_attestation_publisher_follows_db_writes_flag(settings, enabled: bool) -> None:
    # Commitment predykcji `dry-run-…`, której nie ma w bazie, byłby fałszywym
    # wpisem w publicznym logu atestacji.
    from src.infrastructure.adapters.attestation_publisher import FileAttestationPublisher
    from src.infrastructure.adapters.write_guard import ReadOnlyAttestationPublisher

    settings = settings.model_copy(
        update={"attestation_enabled": True, "db_writes_enabled": enabled}
    )

    publisher = main_agent.build_attestation_publisher(settings)

    expected = FileAttestationPublisher if enabled else ReadOnlyAttestationPublisher
    assert type(publisher) is expected


@pytest.mark.parametrize("available", [False, True])
def test_attestation_needs_the_commitment_columns(
    settings, mock_external_clients, mocker, available: bool
) -> None:
    # WIA-70: flaga on przed migracjami 024/043 → cykl bez atestacji, ale z zapisem
    # predykcji (nowe klucze dałyby PGRST204 i zabiły zapis całego wiersza).
    use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
    settings.attestation_enabled = True
    repo = MagicMock(spec=RepositoryPort)
    repo.get_settled_moves.return_value = []
    repo.attestation_columns_available.return_value = available

    main_agent.build_use_case(settings, repository=repo)

    publisher = use_case_cls.call_args.args[0].attestation_publisher
    assert (publisher is not None) is available


@pytest.mark.parametrize("enabled", [False, True])
def test_fx_cache_writes_through_only_with_db_writes(settings, mocker, enabled: bool) -> None:
    cache_cls = mocker.patch("src.infrastructure.adapters.fx_rates_cache.SupabaseCachedFxRates")
    settings = settings.model_copy(
        update={"fx_rates_cache_enabled": True, "db_writes_enabled": enabled}
    )

    main_agent.build_fx_port(settings)

    assert cache_cls.call_args.kwargs["write_through"] is enabled


class TestCouncilRequestMeteringWiring:
    """FIN-02: DI owija port LLM rady w `MeteredLLMPort` z TĄ SAMĄ telemetrią,
    którą dostaje graf — każdy request persony i chairmana trafia do kosztu
    cyklu (węzeł rady tylko loguje)."""

    def test_council_llm_counts_into_the_graph_telemetry(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.application.ports import LLMPort

        council_inner = MagicMock(spec=LLMPort)
        council_inner.analyze.return_value = {}
        mocker.patch("main_agent.build_council_llm_adapter", return_value=council_inner)
        build_council = mocker.patch("main_agent.build_council_adapter")
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")

        main_agent.build_use_case(settings)

        deps = use_case_cls.call_args.args[0]
        council_llm = build_council.call_args.args[1]
        council_llm.analyze("persona prompt")
        council_llm.analyze("chairman prompt")

        council_inner.analyze.assert_called_with("chairman prompt")
        assert deps.telemetry.snapshot().paid_calls["council_llm"] == 2
        # Główny LLM nie jest owinięty — jego wywołania liczy `_log_paid_call`.
        assert deps.llm_port is not council_llm


class TestPaidCallCapWiring:
    """FIN-12: sufit płatnych wywołań LLM z `Settings` trafia do grafu jednym
    polem `AgentGraphDeps.paid_call_cap`; brak sufitu to None (bez limitu)."""

    def test_build_use_case_passes_the_cap_to_graph_deps(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.max_paid_llm_calls_per_cycle = 100

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].paid_call_cap == 100

    def test_no_cap_leaves_the_graph_unbounded(
        self, settings, mock_external_clients, mocker
    ) -> None:
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")

        main_agent.build_use_case(settings)

        assert use_case_cls.call_args.args[0].paid_call_cap is None


class TestSlowLoopPaidBudget:
    """FIN-30: płatne wywołania LLM Slow Loopa (recap, sędzia kalibracji) liczy ten
    sam meter co Fast Loop, a tygodniowy budżet z domeny przycina je PRZED
    wywołaniem dostawcy. Recap ma pierwszeństwo — to on trafia do użytkownika."""

    def _trainer(self, settings, mocker) -> tuple[MagicMock, MagicMock, MagicMock]:
        from src.application.ports import LLMPort

        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        mocker.patch("main_trainer.build_notifier", return_value=MagicMock())
        provider = MagicMock(spec=LLMPort)
        provider.analyze.return_value = {"narrative": "opowieść"}
        provider.analyze_mistake.return_value = "przepewny"
        mocker.patch("main_agent.build_llm_adapter", return_value=provider)
        recap_cls = mocker.patch("main_trainer.WeeklyRecapUseCase")
        recap_cls.return_value.run.return_value = {"status": "skipped_below_threshold"}
        judge_cls = mocker.patch(
            "src.application.use_cases.calibrate_confidence.CalibrateConfidenceUseCase"
        )
        judge_cls.return_value.run.return_value = {"status": "calibrated"}
        settings.symbols = ["AAPL"]
        settings.weekly_recap_enabled = True
        settings.calibration_judge_enabled = True
        # FIN-35: budżet dotyczy prawdziwego dostawcy — lokalna atrapa nie płaci.
        settings.paid_calls_enabled = True
        return provider, recap_cls, judge_cls

    def test_job_past_the_weekly_cap_never_reaches_the_provider(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.application.metering import PaidBudgetExhausted

        provider, recap_cls, judge_cls = self._trainer(settings, mocker)
        settings.slow_loop_max_paid_llm_calls_per_week = 1

        main_trainer.main(settings)

        recap_llm = recap_cls.call_args.kwargs["llm_port"]
        judge_llm = judge_cls.call_args.kwargs["llm_port"]
        assert recap_llm.analyze("narracja") == {"narrative": "opowieść"}
        with pytest.raises(PaidBudgetExhausted):
            judge_llm.analyze_mistake("wniosek")
        provider.analyze_mistake.assert_not_called()
        judge_cls.return_value.run.assert_called_once()

    def test_without_a_cap_both_jobs_reach_the_provider(
        self, settings, mock_external_clients, mocker
    ) -> None:
        provider, recap_cls, judge_cls = self._trainer(settings, mocker)

        main_trainer.main(settings)

        recap_cls.call_args.kwargs["llm_port"].analyze("narracja")
        judge_cls.call_args.kwargs["llm_port"].analyze_mistake("wniosek")
        provider.analyze.assert_called_once_with("narracja")
        provider.analyze_mistake.assert_called_once_with("wniosek")

    def test_paid_calls_of_the_run_are_metered_and_logged(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        import logging

        _, recap_cls, judge_cls = self._trainer(settings, mocker)
        settings.slow_loop_max_paid_llm_calls_per_week = 15

        def recap_run(*_args, **_kwargs):
            recap_cls.call_args.kwargs["llm_port"].analyze("narracja")
            return {"status": "skipped_below_threshold"}

        recap_cls.return_value.run.side_effect = recap_run
        with caplog.at_level(logging.INFO, logger="main_trainer"):
            main_trainer.main(settings)

        assert any(
            "Slow Loop paid LLM calls: 1/15" in rec.message for rec in caplog.records
        ), [rec.message for rec in caplog.records]


class TestDiagnosisBackfillJob:
    """WIA-53: job spłaty długu diagnoz w Slow Loopie — za flagą, z przydziałem
    z tygodniowego budżetu FIN-30 (ostatni w kolejce: recap > sędzia > diagnozy)."""

    def _trainer(self, settings, mocker) -> MagicMock:
        from src.application.ports import LLMPort

        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        mocker.patch("main_trainer.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent.build_llm_adapter", return_value=MagicMock(spec=LLMPort))
        recap_cls = mocker.patch("main_trainer.WeeklyRecapUseCase")
        recap_cls.return_value.run.return_value = {"status": "skipped_below_threshold"}
        judge_cls = mocker.patch(
            "src.application.use_cases.calibrate_confidence.CalibrateConfidenceUseCase"
        )
        judge_cls.return_value.run.return_value = {"status": "calibrated"}
        backfill_cls = mocker.patch("main_trainer.BackfillDiagnosesUseCase")
        backfill_cls.return_value.run.return_value = {"status": "backfilled"}
        settings.symbols = ["AAPL"]
        settings.paid_calls_enabled = True
        return backfill_cls

    def test_off_by_default(self, settings, mock_external_clients, mocker) -> None:
        backfill_cls = self._trainer(settings, mocker)

        main_trainer.main(settings)

        backfill_cls.assert_not_called()

    def test_job_gets_its_cap_through_a_budgeted_port(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.application.metering import BudgetedLLMPort

        backfill_cls = self._trainer(settings, mocker)
        settings.diagnosis_backfill_enabled = True
        settings.diagnosis_backfill_max_calls = 5
        settings.slow_loop_max_paid_llm_calls_per_week = 15

        main_trainer.main(settings)

        kwargs = backfill_cls.call_args.kwargs
        assert kwargs["max_paid_calls"] == 5
        assert isinstance(kwargs["llm_port"], BudgetedLLMPort)
        backfill_cls.return_value.run.assert_called_once()

    def test_weekly_cap_trims_the_backfill_first(
        self, settings, mock_external_clients, mocker
    ) -> None:
        backfill_cls = self._trainer(settings, mocker)
        settings.weekly_recap_enabled = True
        settings.calibration_judge_enabled = True
        settings.diagnosis_backfill_enabled = True
        settings.diagnosis_backfill_max_calls = 5
        settings.slow_loop_max_paid_llm_calls_per_week = 3

        main_trainer.main(settings)

        # Sufit 3: recap 1 + sędzia 1 → na diagnozy zostaje 1.
        assert backfill_cls.call_args.kwargs["max_paid_calls"] == 1


class TestNyseClosedDay:
    """FIN-34: w dzień zamknięcia NYSE (święto, weekend) Fast Loop pomija akcje
    z jawnym logiem. Krypto handluje w święta, więc idzie normalnie; bez krypto
    przebieg kończy się od razu — bez use case'u i bez maila."""

    _CHRISTMAS = datetime(2026, 12, 25, 5, 30, tzinfo=UTC)
    _TRADING_DAY = datetime(2026, 12, 23, 5, 30, tzinfo=UTC)

    def _run(self, settings, mocker, caplog, now: datetime) -> tuple[MagicMock, MagicMock]:
        import logging

        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        build_use_case = mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        notifier = MagicMock()
        mocker.patch("main_agent.build_notifier", return_value=notifier)
        with caplog.at_level(logging.INFO, logger="main_agent"):
            self.exit_code = main_agent.main(settings, clock=lambda: now)
        return build_use_case, notifier

    def test_holiday_without_crypto_ends_without_use_case_and_mail(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        settings.nyse_closure_skip_enabled = True

        build_use_case, notifier = self._run(settings, mocker, caplog, self._CHRISTMAS)

        assert self.exit_code == 0
        build_use_case.assert_not_called()
        notifier.send_report.assert_not_called()
        assert any(
            "święto NYSE" in rec.message and "Boże Narodzenie" in rec.message
            for rec in caplog.records
        ), [rec.message for rec in caplog.records]

    def test_holiday_with_crypto_analyzes_only_crypto(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        settings.nyse_closure_skip_enabled = True
        settings.crypto_symbols = ["BTC"]

        build_use_case, _ = self._run(settings, mocker, caplog, self._CHRISTMAS)

        fake_uc = build_use_case.return_value
        assert [c.args[0] for c in fake_uc.run.call_args_list] == ["BTC"]
        assert any(
            "święto NYSE" in rec.message and "BTC" in rec.message for rec in caplog.records
        )

    def test_weekend_skips_stocks_too(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        settings.nyse_closure_skip_enabled = True
        saturday = datetime(2026, 10, 24, 5, 30, tzinfo=UTC)

        build_use_case, _ = self._run(settings, mocker, caplog, saturday)

        build_use_case.assert_not_called()
        assert any("NYSE zamknięta (weekend)" in rec.message for rec in caplog.records)

    def test_trading_day_runs_every_symbol(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        settings.nyse_closure_skip_enabled = True
        settings.crypto_symbols = ["BTC"]

        build_use_case, _ = self._run(settings, mocker, caplog, self._TRADING_DAY)

        fake_uc = build_use_case.return_value
        assert [c.args[0] for c in fake_uc.run.call_args_list] == ["AAPL", "VOO", "BTC"]

    def test_flag_off_runs_on_a_holiday_as_before(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        build_use_case, _ = self._run(settings, mocker, caplog, self._CHRISTMAS)

        assert build_use_case.return_value.run.call_count == len(settings.symbols)

    def test_cycle_start_comes_from_the_injected_clock(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        dispatch = mocker.patch("main_agent._dispatch_reports")

        self._run(settings, mocker, caplog, self._TRADING_DAY)

        assert dispatch.call_args.kwargs["started_at"] == self._TRADING_DAY


class TestLocalRunWithoutPaidCalls:
    """FIN-35: przy `paid_calls_enabled=False` (domyślnie, lokalnie) DI podstawia
    deterministyczne atrapy zamiast LLM, rady, Alpha Vantage i embeddingów, a
    produkcyjne adaptery w ogóle nie powstają. Jeden WARNING mówi o tym wprost."""

    _PAID_FACTORIES = (
        "build_llm_adapter",
        "build_council_llm_adapter",
        "build_council_adapter",
        "build_embedding_adapter",
        "build_tool_use_adapter",
    )

    def _deps(self, settings, mocker) -> tuple[object, dict[str, MagicMock]]:
        factories = {name: mocker.patch(f"main_agent.{name}") for name in self._PAID_FACTORIES}
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        main_agent.build_use_case(settings)
        return use_case_cls.call_args.args[0], factories

    def test_fast_loop_gets_stubs_and_never_builds_a_paid_adapter(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.infrastructure.adapters.local_stubs import (
            LocalStubNewsAdapter,
            LocalStubSentimentAdapter,
        )
        from src.infrastructure.llm.local_stub import LocalStubLLMAdapter

        settings.paid_calls_enabled = False
        settings.tool_use_enabled = True

        deps, factories = self._deps(settings, mocker)

        for name, factory in factories.items():
            assert not factory.called, name
        assert isinstance(deps.llm_port, LocalStubLLMAdapter)
        assert isinstance(deps.sentiment_port, LocalStubSentimentAdapter)
        assert isinstance(deps.news_port, LocalStubNewsAdapter)
        assert deps.council_port is None
        assert deps.embedding_port is None
        assert deps.tool_use_port is None

    def test_paid_run_wires_the_production_llm(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.application.ports import LLMPort
        from src.infrastructure.adapters.alpha_vantage_adapters import (
            AlphaVantageSentimentAdapter,
        )

        settings.paid_calls_enabled = True
        production = MagicMock(spec=LLMPort)
        mocker.patch("main_agent.build_llm_adapter", return_value=production)
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")

        main_agent.build_use_case(settings)

        deps = use_case_cls.call_args.args[0]
        assert deps.llm_port is production
        assert isinstance(deps.sentiment_port, AlphaVantageSentimentAdapter)
        assert deps.council_port is not None

    def test_earnings_calendar_stays_null_without_paid_calls(self, settings) -> None:
        from src.infrastructure.adapters.alpha_vantage_earnings import (
            NullEarningsCalendarAdapter,
        )

        settings.paid_calls_enabled = False
        settings.earnings_calendar_enabled = True

        enrichment = main_agent.build_alpha_enrichment(settings)

        assert isinstance(enrichment._earnings, NullEarningsCalendarAdapter)

    def test_fast_loop_warns_once_and_hides_the_cycle_cost(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        import logging

        settings.paid_calls_enabled = False
        settings.db_writes_enabled = False
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        dispatch = mocker.patch("main_agent._dispatch_reports")

        with caplog.at_level(logging.WARNING):
            main_agent.main(settings)

        warnings = [
            r.message for r in caplog.records if "lokalnie bez płatnych wywołań" in r.message
        ]
        assert len(warnings) == 1
        # Atrapy liczą się w telemetrii jak prawdziwe wywołania — sekcja kosztu by kłamała.
        assert dispatch.call_args.kwargs["context"].cycle_cost is None

    def test_paid_fast_loop_keeps_quiet_and_reports_the_cost(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        import logging

        settings.paid_calls_enabled = True
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        fake_uc.telemetry.snapshot.return_value.paid_calls = {}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        dispatch = mocker.patch("main_agent._dispatch_reports")

        with caplog.at_level(logging.WARNING):
            main_agent.main(settings)

        assert not any("lokalnie bez płatnych wywołań" in r.message for r in caplog.records)
        assert dispatch.call_args.kwargs["context"].cycle_cost is not None

    def _trainer(self, settings, mocker) -> tuple[MagicMock, MagicMock, MagicMock]:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        mocker.patch("main_trainer.build_notifier", return_value=MagicMock())
        av_fundamentals = mocker.patch("main_trainer.AlphaVantageFundamentalsAdapter")
        build_llm = mocker.patch("main_agent.build_llm_adapter")
        recap_cls = mocker.patch("main_trainer.WeeklyRecapUseCase")
        recap_cls.return_value.run.return_value = {"status": "skipped_below_threshold"}
        settings.symbols = ["AAPL"]
        settings.weekly_recap_enabled = True
        return av_fundamentals, build_llm, recap_cls

    def test_slow_loop_skips_av_fundamentals_and_uses_the_llm_stub(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        import logging

        settings.paid_calls_enabled = False
        settings.db_writes_enabled = False
        av_fundamentals, build_llm, recap_cls = self._trainer(settings, mocker)

        with caplog.at_level(logging.WARNING):
            main_trainer.main(settings)

        av_fundamentals.assert_not_called()
        build_llm.assert_not_called()
        recap_llm = recap_cls.call_args.kwargs["llm_port"]
        assert "narrative" not in recap_llm.analyze("narracja")
        warnings = [
            r.message for r in caplog.records if "lokalnie bez płatnych wywołań" in r.message
        ]
        assert len(warnings) == 1

    def test_paid_slow_loop_refreshes_fundamentals_from_av(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.paid_calls_enabled = True
        av_fundamentals, build_llm, _ = self._trainer(settings, mocker)

        main_trainer.main(settings)

        av_fundamentals.assert_called_once()
        build_llm.assert_called_once()


class TestStubbedPaidPortsRefuseDatabaseWrites:
    """FIN-35 (decyzja 2026-10-20): atrapy płatnych portów razem z zapisem do bazy
    wpisałyby neutralne prognozy atrapy do produkcyjnego track recordu. Fast Loop
    i Slow Loop odmawiają startu, zanim zbudują jakikolwiek adapter."""

    def test_fast_loop_refuses_to_start(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        import logging

        settings.paid_calls_enabled = False
        build_use_case = mocker.patch("main_agent.build_use_case")
        build_repository = mocker.patch("main_agent.build_repository")

        with caplog.at_level(logging.ERROR, logger="main_agent"):
            exit_code = main_agent.main(settings)

        assert exit_code == main_agent.STUB_WRITES_REFUSED_EXIT_CODE
        assert exit_code not in (0, 1)
        build_use_case.assert_not_called()
        build_repository.assert_not_called()
        errors = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any(
            "DB_WRITES_ENABLED=true" in m and "PAID_CALLS_ENABLED=false" in m for m in errors
        ), errors

    def test_slow_loop_refuses_to_start(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        import logging

        settings.paid_calls_enabled = False
        build_use_case = mocker.patch("main_trainer.build_use_case")
        repository_cls = mocker.patch("main_trainer.SupabaseRepository")

        with caplog.at_level(logging.ERROR):
            exit_code = main_trainer.main(settings)

        assert exit_code == main_agent.STUB_WRITES_REFUSED_EXIT_CODE
        build_use_case.assert_not_called()
        repository_cls.assert_not_called()
        assert any("PAID_CALLS_ENABLED=false" in r.message for r in caplog.records)

    def test_read_only_local_run_still_starts(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.paid_calls_enabled = False
        settings.db_writes_enabled = False
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        build_use_case = mocker.patch("main_agent.build_use_case", return_value=fake_uc)

        assert main_agent.main(settings) == 0
        build_use_case.assert_called_once()


class TestClosedDayHeldStockQuotes:
    """FIN-34 (decyzja 2026-10-20): w dzień zamknięcia NYSE akcje nie idą do grafu,
    więc trzymana akcja z watchlisty nie ma ceny z wyników cyklu. Dostaje sam
    darmowy `/quote` (ścieżka PORT-07) — sekcja 💼 jest wyceniona, zero płatnych."""

    _CHRISTMAS = datetime(2026, 12, 25, 5, 30, tzinfo=UTC)
    _PAID_FACTORIES = (
        "build_use_case",
        "build_llm_adapter",
        "build_council_adapter",
        "build_council_llm_adapter",
        "build_embedding_adapter",
        "build_alpha_enrichment",
        "build_tool_use_adapter",
    )

    @staticmethod
    def _portfolio(*symbols: str) -> "main_agent.Portfolio":
        from datetime import date

        from src.domain.portfolio import Position

        return main_agent.Portfolio(
            positions=tuple(
                Position(
                    symbol=symbol,
                    quantity=Decimal("2"),
                    purchase_price=Decimal("100"),
                    purchase_date=date(2026, 10, 1),
                )
                for symbol in symbols
            ),
            as_of=datetime(2026, 12, 24, 5, 0, tzinfo=UTC),
        )

    def test_held_watchlist_stock_gets_a_free_quote_when_only_crypto_ran(
        self, settings, mocker
    ) -> None:
        from src.application.ports import MarketDataPort
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.value_objects import Money

        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch(
            "main_agent._build_user_portfolio", return_value=self._portfolio("AAPL", "BTC")
        )
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        mocker.patch("main_agent._publish_public_scorecard")
        report = mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
        market = MagicMock(spec=MarketDataPort)
        market.get_current_price.return_value = Money(Decimal("229.10"))
        mocker.patch("main_agent.build_market_port", return_value=market)
        paid = {name: mocker.patch(f"main_agent.{name}") for name in self._PAID_FACTORIES}
        repository = MagicMock(spec=RepositoryPort)

        main_agent._dispatch_reports(
            results=[
                main_agent.SymbolResult(
                    symbol="BTC", status="ignored", current_price=Decimal("65000")
                )
            ],
            started_at=self._CHRISTMAS,
            duration=1.0,
            failures=0,
            settings=settings,
            repository=repository,
            quota_monitor=QuotaMonitor(),
            notifier=MagicMock(),
            subscribers=[],
            all_symbols=["AAPL", "VOO", "BTC"],
            cycle_symbols=["BTC"],
            context=main_agent.ReportContext(),
        )

        market.get_current_price.assert_called_once_with("AAPL")
        repository.save_price_snapshot.assert_not_called()
        for name, factory in paid.items():
            assert not factory.called, name
        assert report.call_args_list[0].args[3].held_prices == {"AAPL": Decimal("229.10")}

    def test_closed_day_passes_only_the_cycle_symbols_to_the_report(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.nyse_closure_skip_enabled = True
        settings.crypto_symbols = ["BTC"]
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        dispatch = mocker.patch("main_agent._dispatch_reports")

        main_agent.main(settings, clock=lambda: self._CHRISTMAS)

        kwargs = dispatch.call_args.kwargs
        assert kwargs["cycle_symbols"] == ["BTC"]
        assert kwargs["all_symbols"] == ["AAPL", "VOO", "BTC"]

    def test_trading_day_cycle_covers_the_whole_watchlist(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.nyse_closure_skip_enabled = True
        settings.crypto_symbols = ["BTC"]
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        dispatch = mocker.patch("main_agent._dispatch_reports")

        main_agent.main(settings, clock=lambda: datetime(2026, 12, 23, 5, 30, tzinfo=UTC))

        assert dispatch.call_args.kwargs["cycle_symbols"] == ["AAPL", "VOO", "BTC"]


def _usage(model: str = "claude-sonnet-4-6") -> "object":
    from src.domain.finops import LLMUsage

    return LLMUsage(provider="anthropic", model=model, input_tokens=1000, output_tokens=200)


def _sonnet_price() -> "object":
    from src.config import ModelPriceSettings

    return ModelPriceSettings(input=3.0, output=15.0)


class TestCostLedgerWiring:
    """FIN-13: jeden `UsageLedger` na przebieg — adaptery LLM i embeddingów
    zgłaszają do niego tokeny (port `UsageSink`), mail liczy z nich koszt cyklu,
    a zapis `llm_calls` idzie batchem po cyklu tylko za `cost_ledger_enabled`."""

    _PAID_FACTORIES = (
        "build_llm_adapter",
        "build_council_llm_adapter",
        "build_embedding_adapter",
        "build_tool_use_adapter",
    )

    def test_every_paid_adapter_gets_the_ledger_as_its_usage_sink(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.application.usage_ledger import UsageLedger

        factories = {name: mocker.patch(f"main_agent.{name}") for name in self._PAID_FACTORIES}
        mocker.patch("main_agent.AnalyzeMarketUseCase")
        ledger = UsageLedger()

        main_agent.build_use_case(settings, usage_ledger=ledger)

        for name, factory in factories.items():
            assert factory.call_args.kwargs["usage_sink"] is ledger, name

    def test_local_stub_run_reports_no_usage(self, settings, mock_external_clients, mocker) -> None:
        # FIN-35: atrapy nie wysyłają requestu, więc nie mają czego zgłosić.
        from src.application.usage_ledger import UsageLedger

        settings.paid_calls_enabled = False
        factories = {name: mocker.patch(f"main_agent.{name}") for name in self._PAID_FACTORIES}
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        ledger = UsageLedger()

        main_agent.build_use_case(settings, usage_ledger=ledger)
        deps = use_case_cls.call_args.args[0]
        deps.llm_port.analyze("prompt")
        deps.llm_port.analyze_mistake("prompt")

        assert not any(factory.called for factory in factories.values())
        assert ledger.records() == ()

    def test_main_hands_one_ledger_to_the_graph_and_the_ledger_write(
        self, settings, mock_external_clients, mocker, monkeypatch
    ) -> None:
        # Identyfikator przebiegu ledgera to `GITHUB_RUN_ID` z RunContext (MAIL-42).
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_RUN_ID", "18324")
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        build = mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        persist = mocker.patch("main_agent.persist_llm_calls")
        mocker.patch("main_agent._dispatch_reports")

        main_agent.main(settings)

        ledger = build.call_args.kwargs["usage_ledger"]
        assert persist.call_args.kwargs["ledger"] is ledger
        assert persist.call_args.kwargs["run_context"].run_id == "18324"

    def test_ledger_flag_off_writes_nothing(self, settings) -> None:
        from src.application.usage_ledger import UsageLedger, usage_scope

        ledger = UsageLedger()
        with usage_scope(source="llm", node="predict", symbol="NVDA"):
            ledger.record(_usage())
        repository = MagicMock(spec=RepositoryPort)
        settings.cost_ledger_enabled = False

        main_agent.persist_llm_calls(
            settings=settings,
            repository=repository,
            ledger=ledger,
            run_context=main_agent.RunContext(),
        )

        repository.save_llm_calls.assert_not_called()

    def test_ledger_flag_on_writes_one_priced_batch(self, settings) -> None:
        from datetime import date

        from src.application.usage_ledger import UsageLedger, usage_scope

        ledger = UsageLedger()
        with usage_scope(source="llm", node="predict", symbol="NVDA"):
            ledger.record(_usage())
            ledger.record(_usage(model="claude-unpriced"))
        repository = MagicMock(spec=RepositoryPort)
        settings.cost_ledger_enabled = True
        settings.llm_pricing_as_of = date(2026, 10, 20)
        settings.llm_pricing = {
            "claude-sonnet-4-6": _sonnet_price()
        }

        main_agent.persist_llm_calls(
            settings=settings,
            repository=repository,
            ledger=ledger,
            run_context=main_agent.RunContext(run_id="18324"),
        )

        repository.save_llm_calls.assert_called_once()
        entries = repository.save_llm_calls.call_args.args[0]
        assert [(e.run_id, e.node, e.symbol) for e in entries] == [
            ("18324", "predict", "NVDA"),
            ("18324", "predict", "NVDA"),
        ]
        assert entries[0].cost_usd == pytest.approx(0.006)
        assert entries[0].price_as_of == date(2026, 10, 20)
        assert entries[1].cost_usd is None

    def test_empty_ledger_writes_nothing(self, settings) -> None:
        from src.application.usage_ledger import UsageLedger

        repository = MagicMock(spec=RepositoryPort)
        settings.cost_ledger_enabled = True

        main_agent.persist_llm_calls(
            settings=settings,
            repository=repository,
            ledger=UsageLedger(),
            run_context=main_agent.RunContext(),
        )

        repository.save_llm_calls.assert_not_called()

    def test_failed_ledger_write_does_not_fail_the_cycle(self, settings, caplog) -> None:
        from src.application.usage_ledger import UsageLedger, usage_scope

        ledger = UsageLedger()
        with usage_scope(source="llm", node="predict", symbol="NVDA"):
            ledger.record(_usage())
        repository = MagicMock(spec=RepositoryPort)
        repository.save_llm_calls.side_effect = RuntimeError("PGRST205")
        settings.cost_ledger_enabled = True

        main_agent.persist_llm_calls(
            settings=settings,
            repository=repository,
            ledger=ledger,
            run_context=main_agent.RunContext(),
        )

        assert "llm_calls" in caplog.text

    def test_read_only_run_skips_the_ledger_write_with_a_warning(self, settings, caplog) -> None:
        # BEZ-08: bez `db_writes_enabled` zapis ledgera pomija strażnik zapisu.
        import logging

        from src.application.usage_ledger import UsageLedger, usage_scope
        from src.infrastructure.adapters.write_guard import ReadOnlyRepository

        ledger = UsageLedger()
        with usage_scope(source="llm", node="predict", symbol="NVDA"):
            ledger.record(_usage())
        delegate = MagicMock(spec=RepositoryPort)
        settings.cost_ledger_enabled = True

        with caplog.at_level(logging.WARNING):
            main_agent.persist_llm_calls(
                settings=settings,
                repository=ReadOnlyRepository(delegate),
                ledger=ledger,
                run_context=main_agent.RunContext(),
            )

        delegate.save_llm_calls.assert_not_called()
        assert "save_llm_calls" in caplog.text


class TestCycleCostFromTokens:
    """FIN-13: sekcja „Koszt cyklu” liczy źródło z tokenów, gdy ledger ma jego
    rekordy z wycenionym modelem; reszta źródeł zostaje na `UNIT_COST_USD`.
    Działa bez flagi ledgera — to tylko pamięć przebiegu."""

    def _ledger(self) -> "object":
        from src.application.usage_ledger import UsageLedger, usage_scope

        ledger = UsageLedger()
        with usage_scope(source="llm", node="predict", symbol="NVDA"):
            ledger.record(_usage())
        return ledger

    def test_llm_line_is_priced_from_tokens_and_the_rest_from_the_unit_cost(
        self, settings
    ) -> None:
        from datetime import date

        from src.application.cycle_telemetry import CycleTelemetry
        from src.domain.finops import UNIT_COST_USD

        telemetry = CycleTelemetry()
        telemetry.record_paid_call("llm")
        telemetry.record_paid_call("sentiment")
        settings.cost_ledger_enabled = False
        settings.llm_pricing_as_of = date(2026, 10, 20)
        settings.llm_pricing = {
            "claude-sonnet-4-6": _sonnet_price()
        }

        cost = main_agent.cycle_cost(settings, telemetry, self._ledger())

        lines = {line.source: line for line in cost.lines}
        assert lines["llm"].from_tokens is True
        assert lines["llm"].subtotal_usd == pytest.approx(0.006)
        assert lines["sentiment"].subtotal_usd == pytest.approx(UNIT_COST_USD["sentiment"])
        assert cost.priced_as_of == date(2026, 10, 20)

    def test_without_a_price_list_the_cost_stays_an_estimate(self, settings) -> None:
        from src.application.cycle_telemetry import CycleTelemetry

        telemetry = CycleTelemetry()
        telemetry.record_paid_call("llm")

        cost = main_agent.cycle_cost(settings, telemetry, self._ledger())

        assert cost.from_tokens is False
        assert cost.priced_as_of is None

    def test_local_stub_run_hides_the_section(self, settings) -> None:
        from src.application.cycle_telemetry import CycleTelemetry

        settings.paid_calls_enabled = False

        assert main_agent.cycle_cost(settings, CycleTelemetry(), self._ledger()) is None

    def test_main_puts_the_token_cost_into_the_mail(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from datetime import date

        from src.application.cycle_telemetry import CycleTelemetry

        telemetry = CycleTelemetry()
        telemetry.record_paid_call("llm")
        fake_uc = MagicMock()
        fake_uc.telemetry = telemetry
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent.UsageLedger", return_value=self._ledger())
        dispatch = mocker.patch("main_agent._dispatch_reports")
        settings.llm_pricing_as_of = date(2026, 10, 20)
        settings.llm_pricing = {
            "claude-sonnet-4-6": _sonnet_price()
        }

        main_agent.main(settings)

        cost = dispatch.call_args.kwargs["context"].cycle_cost
        assert cost.lines[0].from_tokens is True
        assert cost.total_usd == pytest.approx(0.006)


class TestSlowLoopCostLedger:
    """FIN-13: recap i sędzia kalibracji zgłaszają tokeny do ledgera przebiegu
    Slow Loopa; zapis `llm_calls` za tą samą flagą co w Fast Loopie."""

    def _trainer(self, settings, mocker) -> tuple[MagicMock, MagicMock]:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        mocker.patch("main_trainer.build_notifier", return_value=MagicMock())
        build_llm = mocker.patch("main_agent.build_llm_adapter")
        recap_cls = mocker.patch("main_trainer.WeeklyRecapUseCase")
        recap_cls.return_value.run.return_value = {"status": "skipped_below_threshold"}
        persist = mocker.patch("main_trainer.persist_llm_calls")
        settings.symbols = ["AAPL"]
        settings.weekly_recap_enabled = True
        settings.paid_calls_enabled = True
        return build_llm, persist

    def test_recap_llm_reports_to_the_run_ledger_that_gets_persisted(
        self, settings, mock_external_clients, mocker, monkeypatch
    ) -> None:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_RUN_ID", "18400")
        build_llm, persist = self._trainer(settings, mocker)

        main_trainer.main(settings)

        ledger = persist.call_args.kwargs["ledger"]
        assert build_llm.call_args.kwargs["usage_sink"] is ledger
        assert persist.call_args.kwargs["run_context"].run_id == "18400"

    def test_recap_usage_is_attributed_to_its_job(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.application.ports import LLMPort

        build_llm, persist = self._trainer(settings, mocker)

        def provider(_settings, quota_monitor=None, usage_sink=None):
            port = MagicMock(spec=LLMPort)
            port.analyze.side_effect = lambda _prompt: usage_sink.record(_usage()) or {}
            return port

        build_llm.side_effect = provider
        recap_cls = main_trainer.WeeklyRecapUseCase
        recap_cls.return_value.run.side_effect = lambda **_: (
            recap_cls.call_args.kwargs["llm_port"].analyze("recap prompt")
            or {"status": "skipped_below_threshold"}
        )

        main_trainer.main(settings)

        records = persist.call_args.kwargs["ledger"].records()
        assert [(r.source, r.node, r.symbol) for r in records] == [("llm", "recap", None)]


class TestTwoPhaseCycle:
    """EXP-27 (decyzja 2026-10-20: za flagą, do porównania z FIN-04): faza 1 dla
    wszystkich symboli to darmowa cena, reflect i bramka domenowa; faza 2 buduje
    `AlphaVantageClient(symbols=przepuszczone)` i płaci tylko za nie, w kolejności
    malejącego |Δ| — sufit FIN-12 odcina najmniejsze ruchy."""

    _NOW = datetime(2026, 10, 20, 21, 30, tzinfo=UTC)

    def _repository(self, *, unverified: bool = False) -> MagicMock:
        from src.domain.prediction import Prediction, TrendDirection
        from src.domain.value_objects import Money

        repository = MagicMock(spec=RepositoryPort)
        repository.get_last_price.return_value = Money(Decimal("100"))
        repository.get_last_price_snapshot.return_value = (
            Money(Decimal("100")),
            self._NOW - timedelta(hours=24),
        )
        repository.get_last_prediction_price.return_value = None
        repository.get_cached_fundamentals.return_value = None
        repository.get_settled_moves.return_value = []
        repository.find_similar_predictions.return_value = []
        repository.get_council_votes_for_prediction.return_value = []
        repository.save_prediction.side_effect = lambda record: f"id-{record['symbol']}"
        repository.get_unverified_prediction.side_effect = (
            (
                lambda symbol, **_: Prediction(
                    id=f"prev-{symbol}",
                    symbol=symbol,
                    predicted_trend=TrendDirection.SIDEWAYS,
                    price_at_prediction=Decimal("100"),
                    predicted_target_price=Decimal("100"),
                )
            )
            if unverified
            else (lambda symbol, **_: None)
        )
        return repository

    def _wire(self, settings, mocker, prices: dict[str, str]) -> tuple[MagicMock, MagicMock]:
        """Prawdziwy graf i prawdziwy klient AV; atrapy tylko na brzegach (ceny,
        LLM, HTTP). Zwraca (klasę klienta AV pod podglądem, mock `Session.get`)."""
        from src.application.ports import LLMPort, MarketDataPort
        from src.domain.value_objects import Money
        from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient

        market = MagicMock(spec=MarketDataPort)
        market.get_current_price.side_effect = lambda symbol: Money(Decimal(prices[symbol]))
        mocker.patch("main_agent.FinnhubAdapter", return_value=market)
        llm = MagicMock(spec=LLMPort)
        llm.analyze.return_value = {
            "trend_direction": "BULLISH",
            "confidence_score": 0.7,
            "av_agreement": 0.6,
            "target_price_12h": 106.0,
            "reasoning": "r",
        }
        llm.analyze_mistake.return_value = "lekcja"
        mocker.patch("main_agent.build_llm_adapter", return_value=llm)
        mocker.patch("main_agent.build_council_adapter", return_value=None)
        mocker.patch("main_agent.build_embedding_adapter", return_value=None)
        client_cls = mocker.patch("main_agent.AlphaVantageClient", wraps=AlphaVantageClient)
        mocker.patch("src.infrastructure.adapters.alpha_vantage_client.time.sleep")
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"feed": []}
        session_get = mocker.patch("requests.Session.get", return_value=response)
        settings.symbols = list(prices)
        settings.two_phase_cycle_enabled = True
        return client_cls, session_get

    def _run(self, settings, repository):
        use_case = main_agent.build_use_case(settings, repository=repository)
        return main_agent._analyze_symbols(
            list(settings.symbols),
            use_case=use_case,
            settings=settings,
            crypto_set=set(),
            etf_set=set(),
            regime_context="",
            regime_multiplier=1.0,
            collect_alpha=False,
            alpha_use_case=MagicMock(),
            cycle_at=self._NOW,
            phase_two_feeds=main_agent.build_phase_two_feeds(settings, quota_monitor=None),
        )

    @staticmethod
    def _news_requests(session_get: MagicMock) -> list[str]:
        return [
            call.kwargs["params"]["tickers"]
            for call in session_get.call_args_list
            if call.kwargs.get("params", {}).get("function") == "NEWS_SENTIMENT"
        ]

    @pytest.mark.parametrize("concurrency", [1, 4])
    def test_av_client_built_only_for_symbols_past_gate(
        self, settings, mock_external_clients, mocker, concurrency
    ) -> None:
        movers = {"S07": "105", "S21": "97", "S40": "104"}
        prices = {f"S{i:02d}": movers.get(f"S{i:02d}", "100.5") for i in range(45)}
        client_cls, session_get = self._wire(settings, mocker, prices)
        settings.symbol_concurrency = concurrency
        repository = self._repository()

        results, failures, _, _ = self._run(settings, repository)

        # Jeden klient AV na cykl, zbudowany po fazie 1 z samymi symbolami za bramką.
        client_cls.assert_called_once()
        assert sorted(client_cls.call_args.kwargs["symbols"]) == ["S07", "S21", "S40"]
        assert sorted(self._news_requests(session_get)) == ["S07", "S21", "S40"]
        assert failures == 0
        statuses = {result.symbol: result.status for result in results}
        assert [result.symbol for result in results] == list(prices)
        assert {s for s, status in statuses.items() if status == "saved"} == set(movers)
        assert sum(1 for status in statuses.values() if status == "ignored") == 42
        # Pod progiem: snapshot jak dziś; za bramką: snapshot z zapisu fazy 2.
        assert repository.save_price_snapshot.call_count == 45

    def test_budget_cap_cuts_the_smallest_moves_first(
        self, settings, mock_external_clients, mocker
    ) -> None:
        # Kolejność z config.toml (A, B, C) odcięłaby C; ranking |Δ| to B, C, A.
        prices = {"A": "103", "B": "105", "C": "96", "D": "100.5"}
        _, session_get = self._wire(settings, mocker, prices)
        settings.max_paid_llm_calls_per_cycle = 2
        repository = self._repository()

        results, _, _, _ = self._run(settings, repository)

        saved = [call.args[0]["symbol"] for call in repository.save_prediction.call_args_list]
        assert saved == ["B", "C"]
        cut = next(result for result in results if result.symbol == "A")
        assert cut.status == "ignored"
        assert cut.skip_reason is main_agent.SkipReason.BUDGET_EXHAUSTED
        # Odcięty symbol zachowuje starą referencję ceny (wyjątek FIN-12)...
        snapshotted = {call.args[0] for call in repository.save_price_snapshot.call_args_list}
        assert snapshotted == {"B", "C", "D"}
        # ...i nie kosztuje requestu Alpha Vantage.
        assert sorted(self._news_requests(session_get)) == ["B", "C"]

    def test_every_symbol_is_settled_before_the_first_paid_feed(
        self, settings, mock_external_clients, mocker
    ) -> None:
        # Self-Reflection przed bramką (CLAUDE.md): faza 1 rozlicza wszystkich.
        prices = {"A": "105", "B": "100.5", "C": "96"}
        _, session_get = self._wire(settings, mocker, prices)
        repository = self._repository(unverified=True)
        events: list[str] = []
        repository.update_prediction_accuracy.side_effect = (
            lambda prediction_id, **_: events.append(f"settle:{prediction_id}")
        )
        response = session_get.return_value
        session_get.side_effect = lambda *_a, **kw: (
            events.append(f"av:{kw['params']['tickers']}") or response
        )

        self._run(settings, repository)

        settled = [event for event in events if event.startswith("settle:")]
        assert settled == ["settle:prev-A", "settle:prev-B", "settle:prev-C"]
        first_feed = next(i for i, event in enumerate(events) if event.startswith("av:"))
        assert first_feed == len(settled)

    def test_build_use_case_defers_the_av_client_to_phase_two(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.infrastructure.adapters.alpha_vantage_adapters import PhaseTwoOnlyFeed

        client_cls = mocker.patch("main_agent.AlphaVantageClient")
        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.two_phase_cycle_enabled = True

        main_agent.build_use_case(settings)

        client_cls.assert_not_called()
        deps = use_case_cls.call_args.args[0]
        assert isinstance(deps.sentiment_port, PhaseTwoOnlyFeed)
        assert isinstance(deps.news_port, PhaseTwoOnlyFeed)

    def test_local_stub_run_binds_stub_feeds_without_an_av_client(self, settings, mocker) -> None:
        from src.infrastructure.adapters.local_stubs import (
            LocalStubNewsAdapter,
            LocalStubSentimentAdapter,
        )

        client_cls = mocker.patch("main_agent.AlphaVantageClient")
        settings.paid_calls_enabled = False

        sentiment, news = main_agent.build_phase_two_feeds(settings, quota_monitor=None)(["A"])

        client_cls.assert_not_called()
        assert isinstance(sentiment, LocalStubSentimentAdapter)
        assert isinstance(news, LocalStubNewsAdapter)

    def test_main_runs_the_two_phase_cycle_only_behind_the_flag(
        self, settings, mock_external_clients, mocker
    ) -> None:
        fake_uc = MagicMock()
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        analyze = mocker.patch("main_agent._analyze_symbols", return_value=([], 0, {}, {}))
        mocker.patch("main_agent._dispatch_reports")

        main_agent.main(settings)
        settings.two_phase_cycle_enabled = True
        main_agent.main(settings)

        flag_off, flag_on = analyze.call_args_list
        assert flag_off.kwargs.get("phase_two_feeds") is None
        assert callable(flag_on.kwargs["phase_two_feeds"])


UNDELIVERED_ANNOTATION = "::error title=StockAgent::Raport nie został dostarczony"


class TestUndeliveredReportSignal:
    """NIEZ-12 (decyzja 2026-10-23, opcja a): exit 0 zgodnie z CLAUDE.md, ale
    niedostarczony raport jest widoczny w UI Actions (adnotacja `::error`) i utrwalony
    jako alert CRITICAL w `quota_alerts`."""

    def _run_main(
        self, settings, mocker, *, delivered: bool, notifier: object | None = None
    ) -> tuple[int, MagicMock]:
        repository = MagicMock(spec=RepositoryPort)
        mocker.patch("main_agent.build_repository", return_value=repository)
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "saved", "delta": Decimal("0.05")}
        fake_uc.telemetry.snapshot.return_value.paid_calls = {}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch(
            "main_agent.build_notifier",
            return_value=MagicMock() if notifier is None else notifier,
        )
        mocker.patch("main_agent._dispatch_reports", return_value=delivered)
        return main_agent.main(settings), repository

    def test_undelivered_report_is_annotated_but_keeps_exit_zero(
        self, settings, mock_external_clients, mocker, capsys
    ) -> None:
        exit_code, _ = self._run_main(settings, mocker, delivered=False)

        assert exit_code == 0
        assert UNDELIVERED_ANNOTATION in capsys.readouterr().out.splitlines()

    def test_undelivered_report_is_persisted_as_a_critical_alert(
        self, settings, mock_external_clients, mocker
    ) -> None:
        _, repository = self._run_main(settings, mocker, delivered=False)

        (alert,) = [c.args[0] for c in repository.save_quota_alert.call_args_list]
        assert alert.severity is QuotaSeverity.CRITICAL
        assert "nie dotarł" in alert.message
        assert alert.action

    def test_delivered_report_leaves_no_annotation_and_no_alert(
        self, settings, mock_external_clients, mocker, capsys
    ) -> None:
        _, repository = self._run_main(settings, mocker, delivered=True)

        assert "::error" not in capsys.readouterr().out
        repository.save_quota_alert.assert_not_called()

    def test_a_run_without_an_email_channel_has_nothing_to_deliver(
        self, settings, mock_external_clients, mocker, capsys
    ) -> None:
        from src.infrastructure.adapters.resend_notifier import NullNotifier

        _, repository = self._run_main(
            settings, mocker, delivered=False, notifier=NullNotifier()
        )

        assert "::error" not in capsys.readouterr().out
        repository.save_quota_alert.assert_not_called()

    def test_alert_persistence_failure_never_changes_the_exit_code(
        self, settings, mock_external_clients, mocker, capsys
    ) -> None:
        repository = MagicMock(spec=RepositoryPort)
        repository.save_quota_alert.side_effect = RuntimeError("db down")
        mocker.patch("main_agent.build_repository", return_value=repository)
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "saved", "delta": Decimal("0.05")}
        fake_uc.telemetry.snapshot.return_value.paid_calls = {}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent._dispatch_reports", return_value=False)

        assert main_agent.main(settings) == 0
        assert UNDELIVERED_ANNOTATION in capsys.readouterr().out.splitlines()


class TestCycleRunRecord:
    """NIEZ-29: jeden wiersz `cycle_runs` (migracja 041) na końcu cyklu Fast Loopa —
    czy cykl ruszył, ile symboli zapisał, ile padło, płatne wywołania i czy mail
    wyszedł. Best-effort za flagą `cycle_runs_enabled`, przez strażnik BEZ-08."""

    _STARTED_AT = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    def _run_main(self, settings, mocker, *, delivered: bool = True) -> MagicMock:
        repository = MagicMock(spec=RepositoryPort)
        mocker.patch("main_agent.build_repository", return_value=repository)
        fake_uc = MagicMock()
        fake_uc.run.side_effect = [
            {"status": "saved", "delta": Decimal("0.05"), "prediction_id": "p1"},
            RuntimeError("Finnhub 503"),
        ]
        fake_uc.telemetry.snapshot.return_value.paid_calls = {"llm": 1, "council_llm": 8}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent._dispatch_reports", return_value=delivered)
        main_agent.main(settings, clock=lambda: self._STARTED_AT)
        return repository

    def test_main_persists_cycle_run_with_delivery_status(
        self, settings, mock_external_clients, mocker, monkeypatch
    ) -> None:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_RUN_ID", "18324")
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
        settings.cycle_runs_enabled = True

        repository = self._run_main(settings, mocker)

        repository.save_cycle_run.assert_called_once()
        run = repository.save_cycle_run.call_args.args[0]
        assert (run.loop, run.run_id, run.run_attempt) == ("fast", "18324", 2)
        assert run.started_at == self._STARTED_AT
        assert (run.eligible, run.saved, run.failures) == (2, 1, 1)
        assert dict(run.paid_calls) == {"llm": 1, "council_llm": 8}
        assert run.report_delivered is True
        assert run.duration_s >= 0

    def test_failed_email_is_recorded_as_not_delivered(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.cycle_runs_enabled = True

        repository = self._run_main(settings, mocker, delivered=False)

        assert repository.save_cycle_run.call_args.args[0].report_delivered is False

    def test_flag_off_writes_no_cycle_run(self, settings, mock_external_clients, mocker) -> None:
        settings.cycle_runs_enabled = False

        repository = self._run_main(settings, mocker)

        repository.save_cycle_run.assert_not_called()

    def test_local_run_has_no_attempt(
        self, settings, mock_external_clients, mocker, monkeypatch
    ) -> None:
        # Poza GHA `run_id` = "local" (RunContext), a ręcznie wyeksportowany
        # GITHUB_RUN_ATTEMPT nie udaje ponowienia — tak jak GITHUB_RUN_ID.
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")
        settings.cycle_runs_enabled = True

        repository = self._run_main(settings, mocker)

        run = repository.save_cycle_run.call_args.args[0]
        assert (run.run_id, run.run_attempt) == ("local", None)

    @pytest.mark.parametrize("attempt", ["", "abc", "0"])
    def test_malformed_attempt_is_none(self, attempt: str) -> None:
        environ = {"GITHUB_ACTIONS": "true", "GITHUB_RUN_ATTEMPT": attempt}

        assert main_agent.run_attempt_from_environ(environ) is None

    def test_failed_cycle_run_write_does_not_fail_the_cycle(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        settings.cycle_runs_enabled = True
        repository = MagicMock(spec=RepositoryPort)
        repository.save_cycle_run.side_effect = RuntimeError("PGRST205")
        mocker.patch("main_agent.build_repository", return_value=repository)
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent._dispatch_reports", return_value=True)

        assert main_agent.main(settings) == 0

        assert "cycle_runs" in caplog.text

    def test_read_only_run_skips_the_cycle_run_with_a_warning(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        # BEZ-08: bez `db_writes_enabled` zapis idzie przez ReadOnlyRepository.
        import logging

        from src.infrastructure.adapters.write_guard import ReadOnlyRepository

        delegate = MagicMock(spec=RepositoryPort)
        settings.cycle_runs_enabled = True

        with caplog.at_level(logging.WARNING):
            main_agent.persist_cycle_run(
                settings=settings,
                repository=ReadOnlyRepository(delegate),
                run=MagicMock(run_id="local"),
            )

        delegate.save_cycle_run.assert_not_called()
        assert "save_cycle_run" in caplog.text


class TestFastLoopWeeklyBudgetBanner:
    """WIA-11: tygodniowy najgorszy przypadek Fast Loopa (sufit FIN-12 × 5 dni
    handlowych) ponad `fast_loop_weekly_budget_usd` to WARNING w logu i w bannerze
    maila przy starcie — wzorzec FIN-32, fail-open."""

    _WATCHLIST = [f"S{i:02d}" for i in range(42)]

    def _alerts(self, settings, mocker) -> list[QuotaAlert]:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        dispatch = mocker.patch("main_agent._dispatch_reports")

        main_agent.main(settings)

        return [
            alert
            for alert in dispatch.call_args.kwargs["quota_monitor"].alerts
            if alert.source == "FinOps"
        ]

    def _production_like(self, settings, budget: float | None) -> Settings:
        # 42 symbole za bramką, sufit 100, cztery symbole naraz, 7 person rady.
        return settings.model_copy(update={
            "symbols": self._WATCHLIST,
            "alpha_vantage_api_keys": ["av1", "av2"],
            "max_paid_llm_calls_per_cycle": 100,
            "symbol_concurrency": 4,
            "fast_loop_weekly_budget_usd": budget,
        })

    def test_worst_case_above_the_budget_lands_in_the_banner_and_the_log(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="main_agent"):
            alerts = self._alerts(self._production_like(settings, budget=5.0), mocker)

        assert [alert.severity for alert in alerts] == [QuotaSeverity.WARNING]
        # 42 × (2 × 0,01 + 0,001 + 2 × 0,00002) + 131 × 0,0025 = 1,2112 USD na cykl.
        assert "6,06" in alerts[0].message
        assert "5,00" in alerts[0].message
        assert "fast_loop_weekly_budget_usd" in alerts[0].action
        assert "6.06" in caplog.text
        assert "5.00" in caplog.text

    def test_budget_covering_the_worst_case_adds_no_alert(
        self, settings, mock_external_clients, mocker
    ) -> None:
        assert self._alerts(self._production_like(settings, budget=7.0), mocker) == []

    def test_without_a_budget_there_is_no_check(
        self, settings, mock_external_clients, mocker
    ) -> None:
        assert self._alerts(self._production_like(settings, budget=None), mocker) == []

    def test_unreadable_personas_skip_the_check_without_stopping_the_run(
        self, settings, mock_external_clients, mocker, caplog, tmp_path
    ) -> None:
        import logging

        broken = self._production_like(settings, budget=5.0).model_copy(
            update={"council_personas_dir": str(tmp_path / "missing")}
        )

        with caplog.at_level(logging.WARNING, logger="main_agent"):
            alerts = self._alerts(broken, mocker)

        assert alerts == []
        assert "weekly budget not checked" in caplog.text


class TestFinnhubSharedRateLimiter:
    """DAT-12: limit Finnhub jest per klucz — każdy adapter Finnhub procesu
    bierze tokeny z tego samego limitera (limiter na instancji by go nie widział)."""

    def test_every_finnhub_adapter_shares_the_process_limiter(
        self, settings, mock_external_clients
    ) -> None:
        from src.domain.macro_risk import MacroRiskInstrumentType

        settings.options_flow_enabled = True
        settings.analyst_consensus_enabled = True
        settings.risk_symbols = ["SH"]
        settings.risk_symbol_types = {"SH": MacroRiskInstrumentType.INVERSE_EQUITY}
        shared = main_agent.FINNHUB_RATE_LIMITER

        enrichment = main_agent.build_alpha_enrichment(settings)
        macro = main_agent.build_macro_risk_use_case(settings, MagicMock(spec=RepositoryPort))
        options = main_agent.build_options_port(settings)
        use_case = main_agent.build_use_case(settings)

        assert main_agent.build_market_port(settings)._equity._rate_limiter is shared
        assert enrichment._analyst._rate_limiter is shared
        assert enrichment._options._rate_limiter is shared
        assert macro is not None and macro._market._rate_limiter is shared
        assert options is not None and options._rate_limiter is shared
        assert use_case._deps.market_port._equity._rate_limiter is shared

class TestPredictionColumnsPreflight:
    """FIN-05: przed pierwszym symbolem Fast Loop sprawdza, czy `prediction_logs` ma
    każdą kolumnę, którą zapisze ten config. Brak = exit 1 i instrukcja DB Migrate,
    zanim cykl zapłaci za analizy, których zapis i tak padnie."""

    _COLUMNS = frozenset({"symbol", "gate_delta"})

    def _run(self, settings, mocker, **preflight):
        from src.infrastructure.adapters.supabase_repo import SupabaseRepository

        check = mocker.patch.object(SupabaseRepository, "missing_prediction_columns", **preflight)
        fake_uc = MagicMock()
        fake_uc.required_prediction_columns.return_value = self._COLUMNS
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        fake_uc.telemetry.snapshot.return_value.paid_calls = {}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        dispatch = mocker.patch("main_agent._dispatch_reports")
        return main_agent.main(settings), check, fake_uc, dispatch

    def test_missing_columns_stop_the_cycle_before_any_symbol(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        import logging

        with caplog.at_level(logging.ERROR, logger="main_agent"):
            exit_code, check, fake_uc, dispatch = self._run(
                settings, mocker, return_value=frozenset({"gate_delta"})
            )

        assert exit_code == 1
        check.assert_called_once_with(self._COLUMNS)
        fake_uc.run.assert_not_called()
        dispatch.assert_not_called()
        assert "gate_delta" in caplog.text
        assert "DB Migrate" in caplog.text

    def test_inconclusive_preflight_lets_the_cycle_run(
        self, settings, mock_external_clients, mocker, caplog
    ) -> None:
        import logging

        import httpx

        with caplog.at_level(logging.WARNING, logger="main_agent"):
            exit_code, _, fake_uc, dispatch = self._run(
                settings, mocker, side_effect=httpx.ConnectError("connection reset")
            )

        assert exit_code == 0
        assert fake_uc.run.called
        dispatch.assert_called_once()
        assert "preflight inconclusive (ConnectError)" in caplog.text

    def test_read_only_run_skips_the_preflight(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.db_writes_enabled = False

        exit_code, check, fake_uc, _ = self._run(
            settings, mocker, return_value=frozenset({"gate_delta"})
        )

        assert exit_code == 0
        check.assert_not_called()
        assert fake_uc.run.called


class TestAlphaEnrichmentWiring:
    """FIN-11: bramka earnings czyta datę raportu z `earnings_cache` (migracja 053),
    który odświeża Slow Loop. Fast Loop nie woła Alpha Vantage przed bramką volatility
    — z flagą `earnings_cache_enabled` dostaje cache z delegatem Null."""

    @staticmethod
    def _cache_on(settings) -> None:
        settings.earnings_calendar_enabled = True
        settings.earnings_cache_enabled = True

    def test_fast_loop_earnings_never_calls_alpha_vantage(self, settings, mocker) -> None:
        import requests

        get = mocker.patch.object(requests.Session, "get")
        self._cache_on(settings)
        repository = MagicMock(spec=RepositoryPort)
        repository.get_cached_earnings.return_value = None

        enrichment = main_agent.build_alpha_enrichment(settings, repository=repository)

        assert enrichment.fetch_earnings("AAPL") is None
        repository.get_cached_earnings.assert_called_once_with("AAPL")
        repository.save_earnings.assert_not_called()
        get.assert_not_called()

    def test_fast_loop_takes_the_report_date_from_the_cache(self, settings, mocker) -> None:
        import requests

        from src.domain.earnings import CachedEarningsDate

        get = mocker.patch.object(requests.Session, "get")
        self._cache_on(settings)
        repository = MagicMock(spec=RepositoryPort)
        soon = datetime.now(UTC).date() + timedelta(days=1)
        repository.get_cached_earnings.return_value = CachedEarningsDate(
            "AAPL", soon, datetime.now(UTC)
        )

        event = main_agent.build_alpha_enrichment(
            settings, repository=repository
        ).fetch_earnings("AAPL")

        assert event == EarningsEvent("AAPL", days_until=1, report_date=soon.isoformat())
        get.assert_not_called()

    def test_without_the_cache_flag_the_live_calendar_stays(self, settings) -> None:
        from src.infrastructure.adapters.alpha_vantage_earnings import (
            AlphaVantageEarningsAdapter,
        )

        settings.earnings_calendar_enabled = True

        enrichment = main_agent.build_alpha_enrichment(
            settings, repository=MagicMock(spec=RepositoryPort)
        )

        assert isinstance(enrichment._earnings, AlphaVantageEarningsAdapter)

    def test_cache_flag_without_a_repository_reads_nothing(self, settings) -> None:
        from src.infrastructure.adapters.alpha_vantage_earnings import (
            NullEarningsCalendarAdapter,
        )

        self._cache_on(settings)

        enrichment = main_agent.build_alpha_enrichment(settings)

        assert isinstance(enrichment._earnings, NullEarningsCalendarAdapter)

    def test_fast_loop_hands_its_repository_to_the_enrichment(
        self, settings, mock_external_clients, mocker
    ) -> None:
        repository = MagicMock(spec=RepositoryPort)
        repository.missing_prediction_columns.return_value = frozenset()
        mocker.patch("main_agent.build_repository", return_value=repository)
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        build = mocker.patch(
            "main_agent.build_alpha_enrichment", side_effect=main_agent.build_alpha_enrichment
        )
        mocker.patch("main_agent._run_risk_watch", return_value=None)
        mocker.patch("main_agent._resolve_regime", return_value=(None, 1.0))
        mocker.patch("main_agent._analyze_symbols", return_value=([], 0, {}, {}))
        mocker.patch("main_agent._build_portfolio_radar", return_value=None)
        mocker.patch("main_agent._dispatch_reports")

        main_agent.main(settings)

        assert build.call_args.kwargs["repository"] is repository


class TestSlowLoopEarningsCache:
    """FIN-11: Slow Loop raz w tygodniu odświeża `earnings_cache` z Alpha Vantage
    całą listą kluczy — tylko dla spółek (ETF i krypto nie mają raportów)."""

    def _refresh(self, settings, mocker) -> tuple[MagicMock, MagicMock]:
        av = mocker.patch("main_trainer.AlphaVantageEarningsAdapter")
        cache = mocker.patch("main_trainer.CachedEarningsAdapter")
        settings.symbols = ["AAPL", "VOO", "MSFT"]
        settings.symbols_etf = ["VOO"]
        settings.crypto_symbols = ["BTC"]
        main_trainer._refresh_earnings_cache(settings, MagicMock(spec=RepositoryPort))
        return av, cache

    def test_refreshes_stocks_with_every_alpha_vantage_key(self, settings, mocker) -> None:
        settings.earnings_calendar_enabled = True
        settings.earnings_cache_enabled = True

        av, cache = self._refresh(settings, mocker)

        av.assert_called_once_with(api_key="av1", reserve_keys=["av2"])
        assert cache.call_args.kwargs["delegate"] is av.return_value
        assert [c.args[0] for c in cache.return_value.get_next_earnings.call_args_list] == [
            "AAPL",
            "MSFT",
        ]

    def test_one_failing_symbol_does_not_stop_the_refresh(self, settings, mocker) -> None:
        settings.earnings_calendar_enabled = True
        settings.earnings_cache_enabled = True
        cache = mocker.patch("main_trainer.CachedEarningsAdapter")
        mocker.patch("main_trainer.AlphaVantageEarningsAdapter")
        cache.return_value.get_next_earnings.side_effect = [RuntimeError("table missing"), None]
        settings.symbols = ["AAPL", "MSFT"]

        main_trainer._refresh_earnings_cache(settings, MagicMock(spec=RepositoryPort))

        assert cache.return_value.get_next_earnings.call_count == 2

    @pytest.mark.parametrize(
        ("calendar", "cache_flag", "paid"),
        [(True, False, True), (False, True, True), (True, True, False)],
        ids=["cache-off", "calendar-off", "local-without-paid-calls"],
    )
    def test_refresh_is_skipped(self, settings, mocker, calendar, cache_flag, paid) -> None:
        settings.earnings_calendar_enabled = calendar
        settings.earnings_cache_enabled = cache_flag
        settings.paid_calls_enabled = paid

        av, cache = self._refresh(settings, mocker)

        av.assert_not_called()
        cache.assert_not_called()

    def test_slow_loop_runs_the_refresh(self, settings, mock_external_clients, mocker) -> None:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        mocker.patch("main_trainer.AlphaVantageFundamentalsAdapter")
        refresh = mocker.patch("main_trainer._refresh_earnings_cache")

        main_trainer.main(settings)

        refresh.assert_called_once()


class TestStaleQuoteWiring:
    """DAT-14: akcje i ETF biorą cenę z /quote razem z jego czasem — port notowań
    to ten sam adapter Finnhub co gałąź equity routera; kolumna `stale_quote`
    (migracja 055) za flagą, w grafie i w repozytorium."""

    def test_quote_port_is_the_finnhub_equity_feed(self, settings, mock_external_clients):
        from src.infrastructure.adapters.finnhub_api import FinnhubAdapter

        deps = main_agent.build_agent_graph_deps(settings)

        assert isinstance(deps.quote_port, FinnhubAdapter)
        assert deps.stale_quote_enabled is settings.stale_quote_enabled

    @pytest.mark.parametrize("enabled", [False, True])
    def test_build_repository_passes_the_stale_quote_flag(
        self, settings, mocker, enabled: bool
    ) -> None:
        repo_cls = mocker.patch("main_agent.SupabaseRepository")
        settings.stale_quote_enabled = enabled

        main_agent.build_repository(settings)

        assert repo_cls.call_args.kwargs["stale_quote_enabled"] is enabled


class TestMarketCapWiring:
    """DAT-21: Slow Loop zapisuje kapitalizację w cache za flagą (migracja 056),
    a Fast Loop czyta ją wyłącznie z cache — dla progu znaczącego zakupu insidera."""

    @pytest.mark.parametrize("enabled", [False, True])
    def test_trainer_repository_follows_the_market_cap_flag(
        self, settings, mocker, enabled: bool
    ) -> None:
        supabase_cls = mocker.patch("main_trainer.SupabaseRepository")
        settings = settings.model_copy(update={"fundamentals_market_cap_enabled": enabled})

        main_trainer.build_repository(settings)

        assert supabase_cls.call_args.kwargs["fundamentals_market_cap_enabled"] is enabled

    def test_alpha_enrichment_reads_the_market_cap_from_the_cache_only(
        self, settings, mocker
    ) -> None:
        get = mocker.patch("requests.Session.get")
        repository = MagicMock(spec=RepositoryPort)
        repository.get_cached_fundamentals.return_value = None
        settings = settings.model_copy(update={"insider_flow_enabled": True})

        enrichment = main_agent.build_alpha_enrichment(settings, repository=repository)
        port = enrichment._market_caps

        assert port is not None
        assert port.get_fundamentals("NVDA") is None
        repository.get_cached_fundamentals.assert_called_with("NVDA")
        get.assert_not_called()

    def test_without_a_repository_the_threshold_stays_fixed(self, settings) -> None:
        settings = settings.model_copy(update={"insider_flow_enabled": True})

        assert main_agent.build_alpha_enrichment(settings)._market_caps is None


class TestRegimeChangeWiring:
    """ALR-05 (krok 2): `_resolve_regime` podaje zmianę reżimu do bannera (a stąd
    do leadu) wyłącznie z histerezy — flaga `regime_hysteresis_enabled` po migracji 033."""

    _CYCLE_AT = datetime(2026, 10, 19, 13, 30, tzinfo=UTC)

    @staticmethod
    def _settings(**overrides: object) -> MagicMock:
        settings = MagicMock()
        settings.regime_aware_enabled = True
        settings.yield_curve_enabled = False
        settings.regime_threshold_max_multiplier = 1.5
        settings.regime_hysteresis_enabled = True
        settings.regime_exit_confirmations = 1
        settings.volatility_threshold = Decimal("0.02")
        for name, value in overrides.items():
            setattr(settings, name, value)
        return settings

    def _banner(self, settings: MagicMock, level: MacroAlertLevel, previous: object) -> object:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_latest_regime_state.return_value = previous
        macro = MagicMock()
        macro.overall_alert = level
        banners: list[object] = []
        main_agent._resolve_regime(
            settings, macro, None, repository=repo, cycle_at=self._CYCLE_AT,
            on_banner=banners.append,
        )
        [banner] = banners
        return banner

    def test_entering_risk_off_reaches_the_banner(self) -> None:
        from src.domain.regime import MarketRegime, RegimeChange, RegimeState

        banner = self._banner(
            self._settings(), MacroAlertLevel.CRITICAL, RegimeState(regime=MarketRegime.NEUTRAL)
        )

        assert banner.change == RegimeChange(
            previous=MarketRegime.NEUTRAL, current=MarketRegime.RISK_OFF
        )

    def test_leaving_risk_off_reaches_the_banner(self) -> None:
        from src.domain.regime import MarketRegime, RegimeState

        banner = self._banner(
            self._settings(), MacroAlertLevel.NORMAL, RegimeState(regime=MarketRegime.RISK_OFF)
        )

        assert banner.change is not None
        assert banner.change.previous is MarketRegime.RISK_OFF

    def test_without_hysteresis_there_is_no_change(self) -> None:
        from src.domain.regime import MarketRegime, RegimeState

        banner = self._banner(
            self._settings(regime_hysteresis_enabled=False),
            MacroAlertLevel.CRITICAL,
            RegimeState(regime=MarketRegime.NEUTRAL),
        )

        assert banner.change is None


class TestInsiderLeadWindowWiring:
    """DAT-21: okno leadu zakupu insidera liczy się od daty cyklu (kalendarz NYSE,
    jak daty filingów EDGAR) z `insider_lead_max_age_days`."""

    def test_report_context_carries_the_lead_window(
        self, settings, mock_external_clients, mocker
    ) -> None:
        from src.domain.insider_flow import InsiderLeadWindow
        from src.domain.market_calendar import to_nyse_time

        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        dispatch = mocker.patch("main_agent._dispatch_reports")
        settings = settings.model_copy(update={"insider_lead_max_age_days": 5})

        main_agent.main(settings)

        kwargs = dispatch.call_args.kwargs
        assert kwargs["context"].insider_lead_window == InsiderLeadWindow(
            as_of=to_nyse_time(kwargs["started_at"]).date(), max_age_days=5
        )


class TestSessionCloseSettlementWiring:
    """WIA-61: zamknięcie sesji obok ceny /quote — flaga w grafie (zapis rozliczenia)
    i w repozytorium (odczyt drugiej trafności), obie za migracją 059."""

    @pytest.mark.parametrize("enabled", [False, True])
    def test_graph_and_repository_follow_the_flag(
        self, settings, mock_external_clients, mocker, enabled: bool
    ) -> None:
        settings = settings.model_copy(update={"session_close_settlement_enabled": enabled})

        deps = main_agent.build_agent_graph_deps(settings)
        repo_cls = mocker.patch("main_agent.SupabaseRepository")
        main_agent.build_repository(settings)

        assert deps.session_close_settlement_enabled is enabled
        assert repo_cls.call_args.kwargs["session_close_settlement_enabled"] is enabled


class TestReportReads:
    """ARCH-16: okno track recordu czytane RAZ na dispatch — rozliczenia i próbki
    kalibracji biorą wszystkie sekcje maila i publiczny scorecard."""

    def test_calibration_and_resolved_rows_are_read_once_per_dispatch(
        self, settings, mocker
    ) -> None:
        from src.application.quota_monitor import QuotaMonitor

        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        settings.equity_curve_enabled = True
        settings.lessons_enabled = True
        settings.calibration_curve_enabled = True
        settings.calibrated_confidence_enabled = True
        settings.model_scorecard_enabled = True
        settings.public_scorecard_enabled = True
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch("main_agent._build_user_portfolio", return_value=None)
        mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
        mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
        repository = MagicMock(spec=RepositoryPort)
        repository.get_resolved_predictions.return_value = [
            {
                "symbol": "AAPL",
                "predicted_trend": "BULLISH",
                "is_trend_correct": True,
                "price_at_prediction": "100",
                "actual_price_after_12h": "101",
                "timestamp": "2026-10-17T05:30:00+00:00",
            }
        ]
        repository.get_resolved_for_calibration.return_value = [
            {"confidence_score": 0.8, "is_trend_correct": True, "predicted_trend": "BULLISH"}
        ]
        repository.get_recent_model_scorecards.return_value = [
            {"symbol": "AAPL", "status": "trained_successfully", "trained_at": "2026-10-17"}
        ]
        repository.get_price_history.return_value = []
        repository.get_settled_moves.return_value = []

        main_agent._dispatch_reports(
            results=[],
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=repository,
            quota_monitor=QuotaMonitor(),
            notifier=MagicMock(),
            subscribers=[],
            all_symbols=["AAPL"],
            context=main_agent.ReportContext(),
        )

        assert repository.get_resolved_for_calibration.call_count == 1
        assert repository.get_resolved_predictions.call_count == 1


def _assembled_report() -> AssembledReport:
    from src.application.report_evidence import PaperCosts
    from src.application.track_record_window import TrackRecordReads

    return AssembledReport(
        context=main_agent.ReportContext(),
        reads=TrackRecordReads(MagicMock(spec=RepositoryPort), 30),
        costs=PaperCosts(),
    )


def test_deliver_report_returns_outcome_per_channel(settings, mocker) -> None:
    # ARCH-08: wysyłka mówi, który kanał wyszedł, a który padł — zamiast
    # jednego bool spod `except Exception` całej fazy.
    from src.application.quota_monitor import QuotaMonitor
    from src.application.report_delivery import ChannelDelivery, DeliveryStatus, ReportChannel

    settings.messenger_digest_enabled = True
    settings.attestation_enabled = False
    settings.public_scorecard_enabled = False
    mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
    mocker.patch("main_agent.build_push_notifier", return_value=MagicMock())
    notifier = MagicMock()
    notifier.send_report.side_effect = RuntimeError("Resend 503")

    outcome = main_agent.deliver_report(
        _assembled_report,
        results=[],
        started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
        duration=1.0,
        failures=0,
        settings=settings,
        repository=MagicMock(spec=RepositoryPort),
        quota_monitor=QuotaMonitor(),
        notifier=notifier,
        subscribers=[],
        all_symbols=["AAPL"],
    )

    assert outcome.channels == (
        ChannelDelivery(ReportChannel.EMAIL, DeliveryStatus.FAILED),
        ChannelDelivery(ReportChannel.PUSH_DIGEST, DeliveryStatus.SENT),
    )
    assert outcome.report_sent is True
    assert outcome.delivered is False


def test_deliver_report_lists_each_subscriber_under_a_masked_address(settings, mocker) -> None:
    # ARCH-08: fan-out (U2) daje kanał na subskrybenta; adres jak w logu, zamaskowany.
    from src.application.quota_monitor import QuotaMonitor
    from src.application.report_delivery import ChannelDelivery, DeliveryStatus, ReportChannel
    from src.domain.subscriber import Subscriber, masked_email

    settings.messenger_digest_enabled = False
    settings.attestation_enabled = False
    settings.public_scorecard_enabled = False
    settings.notifications_enabled = True
    settings.resend_api_key = "re_test"
    mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))

    def resend(*, recipient: str, **_: object) -> MagicMock:
        channel = MagicMock()
        if recipient == "late@example.com":
            channel.send_report.side_effect = RuntimeError("Resend 503")
        return channel

    mocker.patch("main_agent.ResendNotifier", side_effect=resend)

    outcome = main_agent.deliver_report(
        _assembled_report,
        results=[],
        started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
        duration=1.0,
        failures=0,
        settings=settings,
        repository=MagicMock(spec=RepositoryPort),
        quota_monitor=QuotaMonitor(),
        notifier=MagicMock(),
        subscribers=[
            Subscriber(email="early@example.com", symbols=frozenset()),
            Subscriber(email="late@example.com", symbols=frozenset()),
        ],
        all_symbols=["AAPL"],
    )

    assert outcome.channels == (
        ChannelDelivery(
            ReportChannel.EMAIL, DeliveryStatus.SENT, masked_email("early@example.com")
        ),
        ChannelDelivery(
            ReportChannel.EMAIL, DeliveryStatus.FAILED, masked_email("late@example.com")
        ),
    )
    assert outcome.delivered is False
    assert "early@example.com" not in repr(outcome)


class TestAttestationRevealReconciliation:
    """WIA-15 — `revealed_at` stemplowany na starcie runu, z trwałego ledgera
    (checkout `attestation-log`), a nie zaraz po dopisaniu revealu do pliku."""

    @staticmethod
    def _run_main(
        settings, mocker, on_reconcile: Callable[[], object] | None = None
    ) -> tuple[MagicMock, MagicMock]:
        repository = MagicMock(spec=RepositoryPort)
        mocker.patch("main_agent.build_repository", return_value=repository)
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "saved", "delta": Decimal("0.05")}
        fake_uc.telemetry.snapshot.return_value.paid_calls = {}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent._dispatch_reports", return_value=True)
        reveals = MagicMock()
        reveals.reconcile.side_effect = on_reconcile
        mocker.patch("main_agent.attestation_reveals", return_value=reveals)
        main_agent.main(settings)
        return reveals, fake_uc

    def test_fast_loop_stamps_durable_reveals_before_the_cycle(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.attestation_enabled = True
        calls: list[str] = []
        mocker.patch(
            "main_agent._analyze_symbols",
            side_effect=lambda *_a, **_k: calls.append("cycle") or ([], 0, {}, {}),
        )

        reveals, _ = self._run_main(
            settings, mocker, on_reconcile=lambda: calls.append("reconcile")
        )

        reveals.reconcile.assert_called_once_with()
        assert calls == ["reconcile", "cycle"]

    def test_flag_off_stamps_nothing(self, settings, mock_external_clients, mocker) -> None:
        settings.attestation_enabled = False

        reveals, _ = self._run_main(settings, mocker)

        reveals.reconcile.assert_not_called()

    def test_ledger_reads_the_configured_reveals_file(self, settings, tmp_path) -> None:
        reveals = tmp_path / "reveals.jsonl"
        reveals.write_text('{"commitment": "' + "ab" * 32 + '"}\n', encoding="utf-8")
        settings.attestation_reveals_path = str(reveals)
        settings.attestation_commitments_path = str(tmp_path / "commitments.jsonl")

        ledger = main_agent.build_attestation_ledger(settings)

        assert ledger.revealed_commitments() == frozenset({"ab" * 32})

    def test_sweep_after_the_report_defers_stamps_to_the_ledger(self, settings, mocker) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.infrastructure.adapters.attestation_ledger import FileAttestationLedger

        settings.attestation_enabled = True
        settings.public_scorecard_enabled = False
        settings.messenger_digest_enabled = False
        mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
        sweep = mocker.patch("main_agent.RevealAttestationsUseCase")
        repository = MagicMock(spec=RepositoryPort)

        main_agent.deliver_report(
            _assembled_report,
            results=[],
            started_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=repository,
            quota_monitor=QuotaMonitor(),
            notifier=MagicMock(),
            web_publisher=MagicMock(),
            subscribers=[],
            all_symbols=["AAPL"],
        )

        repo_arg, _, ledger = sweep.call_args.args
        assert repo_arg is repository
        assert isinstance(ledger, FileAttestationLedger)
        sweep.return_value.run.assert_called_once_with()


class TestAttestationCycleRoot:
    """WIA-60 — graf dostaje kolektor cyklu, a Fast Loop po przebiegu symboli publikuje
    jedną linię rootu Merkle'a (także gdy przebieg rzucił)."""

    def test_graph_gets_a_cycle_collector(self, settings, mock_external_clients, mocker) -> None:
        from src.application.attestation_cycle import CycleCommitmentCollector

        use_case_cls = mocker.patch("main_agent.AnalyzeMarketUseCase")
        settings.attestation_enabled = True
        repo = MagicMock(spec=RepositoryPort)
        repo.get_settled_moves.return_value = []
        repo.attestation_columns_available.return_value = True

        main_agent.build_use_case(settings, repository=repo)

        publisher = use_case_cls.call_args.args[0].attestation_publisher
        assert isinstance(publisher, CycleCommitmentCollector)

    @staticmethod
    def _wire(settings, mocker, *, analyze: object) -> MagicMock:
        from src.application.attestation_cycle import CycleCommitmentCollector

        mocker.patch("main_agent.build_repository", return_value=MagicMock(spec=RepositoryPort))
        inner = MagicMock()
        inner.publish_commitment.return_value = "public/attestation/commitments.jsonl"
        fake_uc = MagicMock()
        fake_uc.attestation_publisher = CycleCommitmentCollector(inner)
        fake_uc.telemetry.snapshot.return_value.paid_calls = {}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent._dispatch_reports", return_value=True)
        mocker.patch("main_agent.attestation_reveals", return_value=MagicMock())
        mocker.patch("main_agent._analyze_symbols", side_effect=analyze)
        mocker.patch.dict(
            "os.environ",
            {"GITHUB_ACTIONS": "true", "GITHUB_RUN_ID": "987", "GITHUB_RUN_ATTEMPT": "2"},
        )
        return inner

    @staticmethod
    def _main(settings) -> int:
        return main_agent.main(settings, clock=lambda: datetime(2026, 10, 26, 5, 30, tzinfo=UTC))

    def test_fast_loop_publishes_one_root_line_after_the_symbols(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.attestation_enabled = True

        inner = self._wire(settings, mocker, analyze=lambda *_a, **_k: ([], 0, {}, {}))

        self._main(settings)

        line = inner.publish_commitment.call_args.args[0]
        assert line["cycle_id"] == "987.2.20261026T053000Z"
        assert line["n_leaves"] == 0

    def test_root_is_published_even_when_the_symbol_pass_raises(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.attestation_enabled = True

        def crash(*_a: object, **_k: object) -> None:
            raise RuntimeError("unexpected")

        inner = self._wire(settings, mocker, analyze=crash)

        with pytest.raises(RuntimeError, match="unexpected"):
            self._main(settings)

        inner.publish_commitment.assert_called_once()


class TestPublicStatusPage:
    """NIEZ-33 — Fast Loop odświeża stronę statusu po zapisie wiersza `cycle_runs`,
    tylko gdy obie flagi (scorecard i status) są włączone."""

    @staticmethod
    def _run_main(settings, mocker) -> tuple[list[str], MagicMock]:
        calls: list[str] = []
        repository = MagicMock(spec=RepositoryPort)
        repository.save_cycle_run.side_effect = lambda _run: calls.append("cycle_run")
        mocker.patch("main_agent.build_repository", return_value=repository)
        fake_uc = MagicMock()
        fake_uc.telemetry.snapshot.return_value.paid_calls = {}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent._dispatch_reports", return_value=True)
        mocker.patch("main_agent._analyze_symbols", return_value=([], 0, {}, {}))
        status = MagicMock()
        status.run.side_effect = lambda: calls.append("status")
        factory = mocker.patch("main_agent.status_page_use_case", return_value=status)
        settings.cycle_runs_enabled = True
        main_agent.main(settings)
        return calls, factory

    def test_status_page_follows_the_cycle_run_row(
        self, settings, mock_external_clients, mocker
    ) -> None:
        settings.public_scorecard_enabled = True
        settings.public_status_page_enabled = True

        calls, _ = self._run_main(settings, mocker)

        assert calls == ["cycle_run", "status"]

    @pytest.mark.parametrize(("scorecard", "status"), [(True, False), (False, True)])
    def test_status_page_needs_both_public_flags(
        self, settings, mock_external_clients, mocker, scorecard: bool, status: bool
    ) -> None:
        settings.public_scorecard_enabled = scorecard
        settings.public_status_page_enabled = status

        _, factory = self._run_main(settings, mocker)

        factory.assert_not_called()

    def test_status_publisher_writes_inside_the_scorecard_directory(self, settings) -> None:
        from src.infrastructure.adapters.web_publisher import FileWebPublisher, NullWebPublisher

        settings.public_scorecard_enabled = True
        assert isinstance(main_agent.build_public_status_publisher(settings), NullWebPublisher)

        settings.public_status_page_enabled = True
        publisher = main_agent.build_public_status_publisher(settings)

        assert isinstance(publisher, FileWebPublisher)
        assert settings.public_status_path.startswith("public/scorecard/")

    @pytest.mark.parametrize(
        ("max_gap_h", "expected"), [(0.0, "wstrzymany"), (26.0, "brak przebiegu w terminie")]
    )
    def test_status_page_uses_the_deadman_rule_of_the_fast_loop(
        self, settings, mocker, max_gap_h: float, expected: str
    ) -> None:
        from src.application.ports import CycleRun

        now = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
        repository = MagicMock(spec=RepositoryPort)
        repository.get_cycle_runs_since.return_value = [
            CycleRun(
                loop="fast", run_id="1", run_attempt=1, started_at=now - timedelta(days=9),
                duration_s=1.0, eligible=1, saved=1, failures=0, paid_calls={},
                report_delivered=True,
            )
        ]
        publisher = MagicMock()
        mocker.patch("main_agent.build_public_status_publisher", return_value=publisher)
        settings.liveness_fast_loop_max_gap_h = max_gap_h

        main_agent.status_page_use_case(settings, repository, clock=lambda: now).publish()

        assert expected in publisher.publish.call_args.args[0]
