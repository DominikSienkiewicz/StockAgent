"""Smoke testy entry pointów — DI wiring oraz top-level main() z mockami sieci."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import main_agent
import main_backtest
import main_trainer
from src.application.ports import RepositoryPort
from src.config import Settings
from src.domain.earnings import EarningsEvent
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.macro_risk import MacroAlertLevel


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
