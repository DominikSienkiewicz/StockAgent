"""Smoke testy entry pointów — DI wiring oraz top-level main() z mockami sieci."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, call

import pytest

import main_agent
import main_backtest
import main_trainer
from src.application.alpha_signals import AlphaSignals
from src.application.ports import AttestationPublisherPort, PreviousVerdict, RepositoryPort
from src.config import Settings
from src.domain.consensus_shift import ShiftKind
from src.domain.council import CouncilVerdict
from src.domain.cycle_maturity import CycleMaturity, SkipReason
from src.domain.digest_lead import LeadItem
from src.domain.earnings import EarningsEvent
from src.domain.insider_flow import InsiderFlowSnapshot
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.macro_risk import MacroAlertLevel
from src.domain.quota import QuotaAlert, QuotaSeverity


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
    def test_build_use_case_wires_all_adapters(self, settings, mock_external_clients):
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


class TestMainTrainer:
    def test_build_use_case_wires_supabase_and_xgboost(self, settings, mock_external_clients):
        use_case = main_trainer.build_use_case(settings)
        assert use_case is not None

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

    def test_build_use_case_wires_supabase_and_xgboost(
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
        fake_uc = MagicMock()
        fake_uc.run.side_effect = RuntimeError("XGBoost boom")
        mocker.patch("main_backtest.BacktestUseCase", return_value=fake_uc)

        # Pojedynczy błąd per-symbol nie wywala harnessa.
        assert main_backtest.main(settings) == 0


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
        spy = mocker.patch("main_agent.build_peer_context")
        main_agent._analyze_symbols(["AAPL", "MSFT"], **self._kwargs(s, uc))
        spy.assert_not_called()

    def test_contagion_forces_sequential_despite_concurrency(self, settings, mocker):
        s = settings.model_copy(update={
            "symbol_concurrency": 4, "contagion_enabled": True,
        })
        uc = MagicMock()
        uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        spy = mocker.patch("main_agent.build_peer_context", return_value=())
        main_agent._analyze_symbols(["AAPL", "MSFT"], **self._kwargs(s, uc))
        # Sekwencyjna ścieżka (peer_context budowany) mimo concurrency=4.
        assert spy.call_count == 2

    def test_parallel_uses_thread_pool_executor(self, settings, mocker):
        s = settings.model_copy(update={"symbol_concurrency": 3})
        uc = MagicMock()
        uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        spy = mocker.spy(main_agent, "ThreadPoolExecutor")
        main_agent._analyze_symbols(["AAPL", "MSFT"], **self._kwargs(s, uc))
        spy.assert_called_once()
        assert spy.call_args.kwargs.get("max_workers") == 3

    def test_sequential_path_no_thread_pool(self, settings, mocker):
        s = settings.model_copy(update={"symbol_concurrency": 1})
        uc = MagicMock()
        uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        spy = mocker.spy(main_agent, "ThreadPoolExecutor")
        main_agent._analyze_symbols(["AAPL", "MSFT"], **self._kwargs(s, uc))
        spy.assert_not_called()

    @pytest.mark.parametrize("concurrency", [1, 4])
    def test_cycle_clock_reaches_every_symbol(self, settings, mocker, concurrency):
        # ARCH-26: obie ścieżki (sekwencyjna i równoległa) mapują wynik z zegarem cyklu.
        s = settings.model_copy(update={"symbol_concurrency": concurrency})
        uc = MagicMock()
        uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        spy = mocker.spy(main_agent, "to_symbol_result")
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
        mocker.patch("main_agent._fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch("main_agent._build_track_record", return_value=(None, None, None))
        mocker.patch("main_agent._build_signal_calibration", return_value=None)
        mocker.patch("main_agent._build_model_scorecard", return_value=None)
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
            web_publisher=MagicMock(),
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


class TestAttestationRevealSweep:
    """#16 — sweep ujawnia WSZYSTKIE zapadłe commitmenty. Bez niego predykcje
    symboli usuniętych z config.toml nigdy by się nie odsłoniły, a dla sceptyka
    wygląda to jak ukrywanie nietrafień."""

    @staticmethod
    def _row(pid: str, salt: str = "s" * 32) -> dict:
        return {
            "id": pid,
            "symbol": "GONE",
            "predicted_trend": "BULLISH",
            "predicted_target_price": "110.0",
            "price_at_prediction": "100.0",
            "actual_price_after_12h": "95.0",
            "is_trend_correct": False,
            "commitment_hash": "deadbeef",
            "commitment_salt": salt,
        }

    def test_reveals_every_matured_commitment(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1"), self._row("p-2")]
        pub = MagicMock()

        revealed = main_agent._sweep_reveals(repo, pub)

        assert revealed == 2
        assert pub.publish_reveal.call_count == 2
        assert repo.mark_commitment_revealed.call_count == 2

    def test_reveal_payload_exposes_the_salt(self) -> None:
        # Dopiero teraz sól wychodzi na jaw — to jest cały sens revealu.
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1")]
        pub = MagicMock()

        main_agent._sweep_reveals(repo, pub)

        payload = pub.publish_reveal.call_args.args[0]
        assert payload["salt"] == "s" * 32
        assert payload["commitment"] == "deadbeef"

    def test_a_failed_reveal_does_not_stop_the_sweep(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1"), self._row("p-2")]
        pub = MagicMock()
        pub.publish_reveal.side_effect = [OSError("disk"), "path.jsonl"]

        assert main_agent._sweep_reveals(repo, pub) == 1

    def test_row_is_not_marked_revealed_when_publishing_fails(self) -> None:
        # Inaczej nietrafienie zniknęłoby na zawsze: oznaczone, nigdy ujawnione.
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1")]
        pub = MagicMock()
        pub.publish_reveal.side_effect = OSError("disk")

        main_agent._sweep_reveals(repo, pub)

        repo.mark_commitment_revealed.assert_not_called()

    def test_row_is_not_marked_revealed_when_publisher_returns_empty(self) -> None:
        # Kontrakt portu: "" = wyłączone albo błąd. Adapter połyka OSError i zwraca
        # "", więc bez tej gałęzi wiersz dostałby `revealed_at` bez publikacji.
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1")]
        pub = MagicMock(spec=AttestationPublisherPort)
        pub.publish_reveal.return_value = ""

        assert main_agent._sweep_reveals(repo, pub) == 0
        repo.mark_commitment_revealed.assert_not_called()

    def test_reveal_is_not_counted_when_marking_fails(self) -> None:
        # Wynik sweepu to liczba wierszy z `revealed_at` — nieudany stempel
        # oznacza, że następny sweep ujawni ten wiersz jeszcze raz.
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1")]
        repo.mark_commitment_revealed.side_effect = Exception("db down")
        pub = MagicMock(spec=AttestationPublisherPort)
        pub.publish_reveal.return_value = "public/attestation/reveals.jsonl"

        assert main_agent._sweep_reveals(repo, pub) == 0

    def test_repository_error_degrades_to_no_sweep(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.side_effect = Exception("no column")

        assert main_agent._sweep_reveals(repo, MagicMock()) == 0


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

    def test_repository_error_never_breaks_the_cycle(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.side_effect = Exception("supabase down")

        main_agent._publish_public_scorecard(
            self._settings(True), repo, MagicMock()
        )

    def test_public_publisher_is_a_separate_instance_from_the_digest(self) -> None:
        # Reużycie instancji prywatnego digestu nadpisałoby go publiczną stroną.
        settings = self._settings(True)
        settings.web_digest_enabled = True
        settings.web_digest_path = "public/digest/index.html"

        digest_pub = main_agent.build_web_publisher(settings)
        public_pub = main_agent.build_public_scorecard_publisher(settings)

        assert digest_pub is not public_pub


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
        mocker.patch("main_agent._fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch("main_agent._build_track_record", return_value=(None, None, None))
        mocker.patch("main_agent._build_signal_calibration", return_value=None)
        mocker.patch("main_agent._build_model_scorecard", return_value=None)
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
            web_publisher=MagicMock(),
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
        mocker.patch("main_agent._fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch("main_agent._build_track_record", return_value=(None, None, None))
        mocker.patch("main_agent._build_signal_calibration", return_value=None)
        mocker.patch("main_agent._build_model_scorecard", return_value=None)
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
            web_publisher=MagicMock(),
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
        mocker.patch("main_agent._fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch("main_agent._build_track_record", return_value=(None, None, None))
        mocker.patch("main_agent._build_signal_calibration", return_value=None)
        mocker.patch("main_agent._build_model_scorecard", return_value=None)
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
            web_publisher=MagicMock(),
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
        mocker.patch("main_agent._fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch("main_agent._build_track_record", return_value=(None, None, None))
        mocker.patch("main_agent._build_signal_calibration", return_value=None)
        mocker.patch("main_agent._build_model_scorecard", return_value=None)
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
            web_publisher=MagicMock(),
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
        mocker.patch("main_agent._fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch("main_agent._build_track_record", return_value=(None, None, None))
        mocker.patch("main_agent._build_signal_calibration", return_value=None)
        mocker.patch("main_agent._build_model_scorecard", return_value=None)
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
            web_publisher=MagicMock(),
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
        self, settings: Settings, mocker, results: list | None = None
    ) -> dict[str, dict[str, str]]:
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.subscriber import Subscriber

        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        settings.resend_api_key = "re_test"
        settings.digest_to_email = self.OWNER
        mocker.patch("main_agent._fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch("main_agent._build_track_record", return_value=(None, None, None))
        mocker.patch("main_agent._build_signal_calibration", return_value=None)
        mocker.patch("main_agent._build_model_scorecard", return_value=None)
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
            web_publisher=MagicMock(),
            subscribers=[
                Subscriber(email="Owner@Example.com", symbols=frozenset()),
                Subscriber(email=self.FRIEND, symbols=frozenset()),
            ],
            all_symbols=["AAPL", "VOO"],
            context=main_agent.ReportContext(),
        )
        return sent

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
            "main_agent._fetch_report_context", return_value=(None, [], [], [], [])
        )
        mocker.patch("main_agent._build_track_record", return_value=(None, None, None))
        mocker.patch("main_agent._build_signal_calibration", return_value=None)
        mocker.patch("main_agent._build_model_scorecard", return_value=None)
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
            web_publisher=MagicMock(),
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
        mocker.patch("main_agent._fetch_report_context", return_value=(None, None, [], [], []))

        _, report = self._dispatch_with_fetch_patched(settings, mocker)

        context = report.call_args_list[0].args[3]
        assert context.resolved_unavailable is True

    def test_dispatch_keeps_an_empty_read_as_nothing_settled(self, settings, mocker) -> None:
        mocker.patch("main_agent._fetch_report_context", return_value=(None, [], [], [], []))

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
        mocker.patch("main_agent._build_track_record", return_value=(None, None, None))
        mocker.patch("main_agent._build_signal_calibration", return_value=None)
        mocker.patch("main_agent._build_model_scorecard", return_value=None)
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
            web_publisher=MagicMock(),
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
        mocker.patch("main_agent._fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch("main_agent._build_track_record", return_value=(None, None, None))
        mocker.patch("main_agent._build_signal_calibration", return_value=None)
        mocker.patch("main_agent._build_model_scorecard", return_value=None)
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
            web_publisher=MagicMock(),
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
        computed.assert_called_once_with(portfolio, hedge_symbols=frozenset({"SH", "VIXY"}))

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

        build.assert_called_once_with(settings, repository, portfolio)
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
