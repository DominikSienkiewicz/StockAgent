"""Smoke testy entry pointu Slow Loop (`main_trainer`) — DI wiring oraz top-level main()
z mockami sieci."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import main_trainer
from src.application.ports import RepositoryPort
from src.config import Settings
from src.domain.quota import QuotaSeverity
from tests.entrypoint_support import UNDELIVERED_ANNOTATION, build_quota_alert, build_usage


class TestMainTrainer:
    def test_build_use_case_builds_without_error(
        self, settings: Settings, mock_external_clients: None
    ) -> None:
        use_case = main_trainer.build_use_case(settings)
        assert use_case is not None

    def test_refresh_fetches_only_selected_batch(self, settings: Settings, mocker: Any) -> None:
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

    def test_refresh_without_cache_ages_falls_back_to_list_order(self, settings: Settings) -> None:
        # ARCH-33: nieosiągalna baza (taksonomia z repozytorium) → partia bez rotacji.
        from src.application.errors import ProviderUnavailable

        repository = MagicMock(spec=RepositoryPort)
        repository.get_fundamentals_fetched_at.side_effect = ProviderUnavailable("PGRST000")

        ages = main_trainer._fundamentals_fetched_at(repository, ["AAPL", "MSFT"])

        assert ages == {"AAPL": None, "MSFT": None}

    def test_cache_ages_programming_error_is_not_swallowed(self) -> None:
        repository = MagicMock(spec=RepositoryPort)
        repository.get_fundamentals_fetched_at.side_effect = TypeError("bug in the read")

        with pytest.raises(TypeError, match="bug in the read"):
            main_trainer._fundamentals_fetched_at(repository, ["AAPL"])

    def test_main_returns_zero_on_success(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.run.side_effect = RuntimeError("DB down")
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        exit_code = main_trainer.main(settings)

        assert exit_code == 1

    def test_main_returns_one_when_feature_store_refresh_fails(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.refresh_feature_store.side_effect = RuntimeError("DB down")
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        exit_code = main_trainer.main(settings)

        assert exit_code == 1
        fake_uc.run.assert_not_called()


class TestModelScorecardPersistence:
    """#12 — każdy przebieg treningu zapisuje scorecard per symbol, best-effort."""

    def test_scorecard_saved_per_symbol(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
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

    def test_flag_off_writes_nothing(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
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
        self, settings: Settings, mocker: Any, enabled: bool
    ) -> None:
        supabase_cls = mocker.patch("main_trainer.SupabaseRepository")
        settings = settings.model_copy(update={"model_scorecard_run_id_enabled": enabled})

        main_trainer.build_repository(settings)

        assert supabase_cls.call_args.kwargs["scorecard_run_id_enabled"] is enabled

    @pytest.mark.parametrize("enabled", [False, True])
    def test_trainer_repository_follows_the_feature_stats_flag(
        self, settings: Settings, mocker: Any, enabled: bool
    ) -> None:
        # ML-21: kolumna `feature_stats` (migracja 052) tylko za flagą.
        supabase_cls = mocker.patch("main_trainer.SupabaseRepository")
        settings = settings.model_copy(update={"model_scorecard_feature_stats_enabled": enabled})

        main_trainer.build_repository(settings)

        assert supabase_cls.call_args.kwargs["scorecard_feature_stats_enabled"] is enabled

    @pytest.mark.parametrize("enabled", [False, True])
    def test_trainer_repository_follows_the_weights_sha256_flag(
        self, settings: Settings, mocker: Any, enabled: bool
    ) -> None:
        # CI-13: kolumna `weights_sha256` (migracja 060) tylko za flagą.
        supabase_cls = mocker.patch("main_trainer.SupabaseRepository")
        settings = settings.model_copy(update={"model_scorecard_weights_sha256_enabled": enabled})

        main_trainer.build_repository(settings)

        assert supabase_cls.call_args.kwargs["scorecard_weights_sha256_enabled"] is enabled

    @pytest.mark.parametrize("enabled", [False, True])
    def test_trainer_repository_follows_the_provenance_flag(
        self, settings: Settings, mocker: Any, enabled: bool
    ) -> None:
        # ML-22: kolumny proweniencji treningu (migracja 064) tylko za flagą.
        supabase_cls = mocker.patch("main_trainer.SupabaseRepository")
        settings = settings.model_copy(update={"model_scorecard_provenance_enabled": enabled})

        main_trainer.build_repository(settings)

        assert supabase_cls.call_args.kwargs["scorecard_provenance_enabled"] is enabled

    def test_build_use_case_hands_the_commit_sha_to_the_use_case(
        self, settings: Settings, mocker: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ML-22: SHA kodu przebiegu trafia do proweniencji treningu (karta i sidecar).
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef")
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        use_case_cls = mocker.patch("main_trainer.TrainModelUseCase")

        main_trainer.build_use_case(settings, run_id="run-1")

        assert use_case_cls.call_args.kwargs["git_sha"] == "0123456789abcdef"

    @pytest.mark.parametrize("enabled", [False, True])
    def test_build_use_case_follows_the_settled_purge_flag(
        self, settings: Settings, mocker: Any, enabled: bool
    ) -> None:
        # ML-26: purge foldów po `settled_at` tylko za flagą.
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        use_case_cls = mocker.patch("main_trainer.TrainModelUseCase")
        settings = settings.model_copy(update={"ml_settled_purge_enabled": enabled})

        main_trainer.build_use_case(settings)

        assert use_case_cls.call_args.kwargs["settled_purge"] is enabled

    def test_build_use_case_hands_the_run_id_to_the_use_case(
        self, settings: Settings, mocker: Any
    ) -> None:
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        use_case_cls = mocker.patch("main_trainer.TrainModelUseCase")

        main_trainer.build_use_case(settings, run_id="run-1")

        assert use_case_cls.call_args.kwargs["run_id"] == "run-1"


class TestWeeklyRecapWiring:
    """#17 — niedzielny recap. Slow Loop chodzi co tydzień i dotąd nie wysyłał
    użytkownikowi NIC. Koszt: dokładnie 1 wywołanie LLM tygodniowo."""

    def _trainer(
        self, settings: Settings, mocker: Any, recap_status: str
    ) -> tuple[MagicMock, MagicMock]:
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

    def test_recap_email_is_sent_when_ready(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        _, notifier = self._trainer(settings, mocker, "recap_ready")

        main_trainer.main(settings)

        notifier.send_report.assert_called_once()

    def test_below_threshold_no_email_goes_out(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        # Twardy próg próbki: przy chudym tygodniu recap robiłby dramat z szumu.
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        _, notifier = self._trainer(settings, mocker, "skipped_below_threshold")

        main_trainer.main(settings)

        notifier.send_report.assert_not_called()

    def test_recap_knows_the_crypto_symbols(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        # Decyzja 2026-10-19 (a): pasmo FLAT recapu zależy od klasy aktywa.
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        settings.crypto_symbols = ["BTC", "ETH"]
        self._trainer(settings, mocker, "recap_ready")

        main_trainer.main(settings)

        call = cast(Any, main_trainer).WeeklyRecapUseCase.call_args
        assert list(call.kwargs["crypto_symbols"]) == ["BTC", "ETH"]

    @pytest.mark.parametrize("enabled", [False, True])
    def test_recap_gets_the_council_redundancy_flag(
        self, settings: Settings, mock_external_clients: None, mocker: Any, enabled: bool
    ) -> None:
        # WIA-45: linia redundancji w recapie tygodniowym tylko za flagą.
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        settings.council_redundancy_enabled = enabled
        self._trainer(settings, mocker, "recap_ready")

        main_trainer.main(settings)

        call = cast(Any, main_trainer).WeeklyRecapUseCase.call_args
        assert call.kwargs["council_redundancy_enabled"] is enabled

    def test_flag_off_never_builds_the_recap(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        settings.weekly_recap_enabled = False
        settings.symbols = ["AAPL"]
        recap_uc, notifier = self._trainer(settings, mocker, "recap_ready")

        main_trainer.main(settings)

        recap_uc.run.assert_not_called()
        notifier.send_report.assert_not_called()

    def test_recap_failure_does_not_fail_the_slow_loop(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        recap_uc, _ = self._trainer(settings, mocker, "recap_ready")
        recap_uc.run.side_effect = Exception("LLM down")

        assert main_trainer.main(settings) == 0

    def test_recap_that_could_not_be_sent_is_annotated_and_persisted(
        self,
        settings: Settings,
        mock_external_clients: None,
        mocker: Any,
        capsys: pytest.CaptureFixture[str],
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
        assert [a.severity for a in alerts if "nie dotarł" in a.message] == [QuotaSeverity.CRITICAL]

    def test_recap_without_a_failure_leaves_no_annotation(
        self,
        settings: Settings,
        mock_external_clients: None,
        mocker: Any,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        self._trainer(settings, mocker, "recap_ready")

        main_trainer.main(settings)

        assert "::error" not in capsys.readouterr().out

    def test_a_skipped_recap_is_not_an_undelivered_report(
        self,
        settings: Settings,
        mock_external_clients: None,
        mocker: Any,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        self._trainer(settings, mocker, "skipped_below_threshold")

        main_trainer.main(settings)

        assert "::error" not in capsys.readouterr().out

    def test_deterministic_recap_sent_when_llm_adapter_cannot_be_built(
        self, settings: Settings, mock_external_clients: None, mocker: Any
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        # NIEZ-03: alerty kwot zapisujemy przed recapem, więc alert z samej wysyłki
        # recapu (np. Resend 429) przepadał zamiast trafić do poniedziałkowego bannera.
        settings.weekly_recap_enabled = True
        settings.symbols = ["AAPL"]
        _, notifier = self._trainer(settings, mocker, "recap_ready")
        repository = MagicMock(spec=RepositoryPort)
        mocker.patch("main_trainer.SupabaseRepository", return_value=repository)
        alert = build_quota_alert()
        monitors: list[Any] = []

        def build_notifier(_settings: object, quota_monitor: Any = None) -> MagicMock:
            monitors.append(quota_monitor)
            return notifier

        mocker.patch("main_trainer.build_notifier", side_effect=build_notifier)
        notifier.send_report.side_effect = lambda **_: monitors[-1].record(alert)

        main_trainer.main(settings)

        repository.save_quota_alert.assert_any_call(alert)

    def test_calibration_judge_runs_when_llm_adapter_cannot_be_built(
        self, settings: Settings, mock_external_clients: None, mocker: Any
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


class TestCryptoInTrainingLoop:
    """Bug: main_trainer trenował tylko settings.symbols, więc BTC/ETH NIGDY
    nie przechodziły retrainu — model krypto zostawał na zawsze cold-start."""

    def test_crypto_symbols_are_trained_too(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
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


class TestPooledTrainingLoop:
    """ML-13 (decyzja 4): po ewaluacji per symbol Slow Loop trenuje JEDEN wspólny
    model na wszystkich symbolach — jeden zapis wag zamiast last-writer-wins."""

    def _trainer(self, settings: Settings, mocker: Any) -> MagicMock:
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        settings.model_scorecard_enabled = True
        fake_uc = self._trainer(settings, mocker)

        assert main_trainer.main(settings) == 0

        fake_uc.run_pooled.assert_called_once_with(["AAPL", "BTC", "ETH"], record_scorecard=True)

    def test_scorecard_flag_off_reaches_the_pooled_run(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        settings.model_scorecard_enabled = False
        fake_uc = self._trainer(settings, mocker)

        main_trainer.main(settings)

        assert fake_uc.run_pooled.call_args.kwargs == {"record_scorecard": False}

    def test_pooled_training_failure_alone_is_a_partial_failure(
        self,
        settings: Settings,
        mock_external_clients: None,
        mocker: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # NIEZ-38 (decyzja 2026-10-23): exit 1 tylko, gdy padły wszystkie symbole;
        # sam model wspólny to częściowa porażka — WARNING, wagi z poprzedniego tygodnia.
        fake_uc = self._trainer(settings, mocker)
        fake_uc.run_pooled.side_effect = RuntimeError("XGBoost boom")

        with caplog.at_level(logging.WARNING, logger="main_trainer"):
            assert main_trainer.main(settings) == 0

        assert any(
            record.levelno == logging.WARNING and "model wspólny padł" in record.getMessage()
            for record in caplog.records
        )


class TestSnapshotModelsLoop:
    """ML-52: po modelu wspólnym Slow Loop ewaluuje dwa modele z `feature_snapshots`
    (bez zapisu wag). Pomocniczy krok: jego błąd nie zmienia exit code'u."""

    def _trainer(self, settings: Settings, mocker: Any) -> MagicMock:
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
        self, settings: Settings, mock_external_clients: None, mocker: Any, cards: bool
    ) -> None:
        settings.model_scorecard_enabled = cards
        fake_uc = self._trainer(settings, mocker)

        assert main_trainer.main(settings) == 0

        fake_uc.run_snapshot_models.assert_called_once_with(
            ["AAPL", "BTC", "ETH"], record_scorecard=cards
        )

    def test_snapshot_model_failure_does_not_fail_the_run(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        fake_uc = self._trainer(settings, mocker)
        fake_uc.run_snapshot_models.side_effect = RuntimeError("feature_snapshots missing")

        assert main_trainer.main(settings) == 0

    @pytest.mark.parametrize("enabled", [True, False])
    def test_snapshot_training_flag_reaches_the_use_case(
        self, settings: Settings, mock_external_clients: None, enabled: bool
    ) -> None:
        settings.snapshot_training_enabled = enabled

        use_case = main_trainer.build_use_case(settings)

        assert use_case._snapshot_source is enabled

    @pytest.mark.parametrize("enabled", [True, False])
    def test_sigma_target_flag_reaches_the_trainer(
        self, settings: Settings, mock_external_clients: None, enabled: bool
    ) -> None:
        settings.ml_sigma_target_enabled = enabled

        use_case = main_trainer.build_use_case(settings)

        assert use_case._sigma_target is enabled

    def test_use_case_reads_training_data_from_supabase(
        self, settings: Settings, mock_external_clients: None
    ) -> None:
        from src.application.ports import TrainingDataPort

        use_case = main_trainer.build_use_case(settings)

        assert isinstance(use_case._training_data, TrainingDataPort)


class TestModelScorecardBaselinesWiring:
    """ML-10 — `model_scorecard_baselines_enabled` dociera do repozytorium, które
    zapisuje karty: use case'u (wiersz ALL) i kroków per symbol w `main()`."""

    def test_flag_defaults_off(self, settings: Settings) -> None:
        assert settings.model_scorecard_baselines_enabled is False

    @pytest.mark.parametrize("enabled", [False, True])
    def test_use_case_repository_gets_the_flag(
        self, settings: Settings, mock_external_clients: None, mocker: Any, enabled: bool
    ) -> None:
        settings.model_scorecard_baselines_enabled = enabled
        repo_cls = mocker.patch("main_trainer.SupabaseRepository")

        main_trainer.build_use_case(settings)

        assert repo_cls.call_args.kwargs["scorecard_baselines_enabled"] is enabled

    @pytest.mark.parametrize("enabled", [False, True])
    def test_per_symbol_scorecard_repository_gets_the_flag(
        self, settings: Settings, mock_external_clients: None, mocker: Any, enabled: bool
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
            c.kwargs["scorecard_baselines_enabled"] is enabled for c in repo_cls.call_args_list
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_repository_passes_the_correction_category_flag(
    settings: Settings, mocker: Any, enabled: bool
) -> None:
    # WIA-53/57: backfill diagnoz w Slow Loopie zapisuje kategorię tą samą flagą.
    repo_cls = mocker.patch("main_trainer.SupabaseRepository")
    settings.correction_category_enabled = enabled

    main_trainer.build_repository(settings)

    assert repo_cls.call_args.kwargs["correction_category_enabled"] is enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_repository_records_council_rounds_with_debate_mode(
    settings: Settings, mocker: Any, enabled: bool
) -> None:
    # ML-40: recap tygodniowy liczy redundancję rady z rundy 1, gdy debata zapisuje rundy.
    repo_cls = mocker.patch("main_trainer.SupabaseRepository")
    settings.debate_mode_enabled = enabled

    main_trainer.build_repository(settings)

    assert repo_cls.call_args.kwargs["council_vote_rounds_enabled"] is enabled


def test_trainer_repository_reads_council_rounds_from_the_schema(
    settings: Settings, mocker: Any
) -> None:
    # ML-40: redundancja w recapie z rundy 1 po 082, także po wyłączeniu debaty.
    repo_cls = mocker.patch("main_trainer.SupabaseRepository")
    settings.debate_mode_enabled = False

    main_trainer.build_repository(settings)

    assert repo_cls.call_args.kwargs["council_vote_rounds_from_schema"] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_repository_follows_db_writes_flag(
    settings: Settings, mocker: Any, enabled: bool
) -> None:
    from src.infrastructure.adapters.write_guard import ReadOnlyRepository

    supabase = MagicMock(spec=RepositoryPort)
    mocker.patch("main_trainer.SupabaseRepository", return_value=supabase)
    settings = settings.model_copy(update={"db_writes_enabled": enabled})

    repository = main_trainer.build_repository(settings)
    repository.save_model_scorecard("AAPL", {"status": "skipped"})

    assert isinstance(repository, ReadOnlyRepository) is not enabled
    assert supabase.save_model_scorecard.call_count == int(enabled)


def test_trainer_use_case_gets_the_guarded_repository(settings: Settings, mocker: Any) -> None:
    from src.infrastructure.adapters.write_guard import ReadOnlyRepository

    mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
    use_case_cls = mocker.patch("main_trainer.TrainModelUseCase")
    settings = settings.model_copy(update={"db_writes_enabled": False})

    main_trainer.build_use_case(settings)

    assert isinstance(use_case_cls.call_args.kwargs["db_port"], ReadOnlyRepository)


class TestSlowLoopPaidBudget:
    """FIN-30: płatne wywołania LLM Slow Loopa (recap, spłata diagnoz) liczy ten
    sam meter co Fast Loop, a tygodniowy budżet z domeny przycina je PRZED
    wywołaniem dostawcy. Recap ma pierwszeństwo — to on trafia do użytkownika.
    WIA-65: kalibracja pewności nie woła LLM, więc drugim jobem są diagnozy."""

    def _trainer(self, settings: Settings, mocker: Any) -> tuple[MagicMock, MagicMock, MagicMock]:
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
        backfill_cls = mocker.patch("main_trainer.BackfillDiagnosesUseCase")
        backfill_cls.return_value.run.return_value = {"status": "backfilled"}
        settings.symbols = ["AAPL"]
        settings.weekly_recap_enabled = True
        settings.diagnosis_backfill_enabled = True
        settings.diagnosis_backfill_max_calls = 1
        # FIN-35: budżet dotyczy prawdziwego dostawcy — lokalna atrapa nie płaci.
        settings.paid_calls_enabled = True
        return provider, recap_cls, backfill_cls

    def test_job_past_the_weekly_cap_never_reaches_the_provider(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        from src.application.metering import PaidBudgetExhausted

        provider, recap_cls, backfill_cls = self._trainer(settings, mocker)
        settings.slow_loop_max_paid_llm_calls_per_week = 1

        main_trainer.main(settings)

        recap_llm = recap_cls.call_args.kwargs["llm_port"]
        backfill_llm = backfill_cls.call_args.kwargs["llm_port"]
        assert recap_llm.analyze("narracja") == {"narrative": "opowieść"}
        with pytest.raises(PaidBudgetExhausted):
            backfill_llm.analyze_mistake("wniosek")
        provider.analyze_mistake.assert_not_called()
        backfill_cls.return_value.run.assert_called_once()

    def test_without_a_cap_both_jobs_reach_the_provider(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        provider, recap_cls, backfill_cls = self._trainer(settings, mocker)

        main_trainer.main(settings)

        recap_cls.call_args.kwargs["llm_port"].analyze("narracja")
        backfill_cls.call_args.kwargs["llm_port"].analyze_mistake("wniosek")
        provider.analyze.assert_called_once_with("narracja")
        provider.analyze_mistake.assert_called_once_with("wniosek")

    def test_paid_calls_of_the_run_are_metered_and_logged(
        self,
        settings: Settings,
        mock_external_clients: None,
        mocker: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        _, recap_cls, _ = self._trainer(settings, mocker)
        settings.slow_loop_max_paid_llm_calls_per_week = 15

        def recap_run(*_args: Any, **_kwargs: Any) -> dict[str, str]:
            recap_cls.call_args.kwargs["llm_port"].analyze("narracja")
            return {"status": "skipped_below_threshold"}

        recap_cls.return_value.run.side_effect = recap_run
        with caplog.at_level(logging.INFO, logger="main_trainer"):
            main_trainer.main(settings)

        assert any("Slow Loop paid LLM calls: 1/15" in rec.message for rec in caplog.records), [
            rec.message for rec in caplog.records
        ]


class TestDiagnosisBackfillJob:
    """WIA-53: job spłaty długu diagnoz w Slow Loopie — za flagą, z przydziałem
    z tygodniowego budżetu FIN-30 (ostatni w kolejce: recap > diagnozy)."""

    def _trainer(self, settings: Settings, mocker: Any) -> MagicMock:
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
        backfill_cls: MagicMock = mocker.patch("main_trainer.BackfillDiagnosesUseCase")
        backfill_cls.return_value.run.return_value = {"status": "backfilled"}
        settings.symbols = ["AAPL"]
        settings.paid_calls_enabled = True
        return backfill_cls

    def test_off_by_default(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        backfill_cls = self._trainer(settings, mocker)

        main_trainer.main(settings)

        backfill_cls.assert_not_called()

    def test_job_gets_its_cap_through_a_budgeted_port(
        self, settings: Settings, mock_external_clients: None, mocker: Any
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        backfill_cls = self._trainer(settings, mocker)
        settings.weekly_recap_enabled = True
        settings.calibration_judge_enabled = True
        settings.diagnosis_backfill_enabled = True
        settings.diagnosis_backfill_max_calls = 5
        settings.slow_loop_max_paid_llm_calls_per_week = 2

        main_trainer.main(settings)

        # Sufit 2: recap 1 → na diagnozy zostaje 1 (kalibracja bez LLM, WIA-65).
        assert backfill_cls.call_args.kwargs["max_paid_calls"] == 1


class TestBatchDiagnosisJob:
    """EXP-17: tygodniowa diagnoza zbiorcza — za flagą, jedno wywołanie LLM przez
    przydział tygodniowego budżetu FIN-30 (kolejka: recap > zbiorcza > spłata diagnoz)."""

    def _trainer(self, settings: Settings, mocker: Any) -> tuple[MagicMock, MagicMock]:
        from src.application.ports import LLMPort

        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        mocker.patch("main_trainer.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent.build_llm_adapter", return_value=MagicMock(spec=LLMPort))
        recap_cls = mocker.patch("main_trainer.WeeklyRecapUseCase")
        recap_cls.return_value.run.return_value = {"status": "skipped_below_threshold"}
        batch_cls: MagicMock = mocker.patch("main_trainer.BatchDiagnosisUseCase")
        batch_cls.return_value.run.return_value = {"status": "diagnosed"}
        backfill_cls: MagicMock = mocker.patch("main_trainer.BackfillDiagnosesUseCase")
        backfill_cls.return_value.run.return_value = {"status": "backfilled"}
        settings.symbols = ["AAPL"]
        settings.paid_calls_enabled = True
        return batch_cls, backfill_cls

    def test_off_by_default(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        batch_cls, _ = self._trainer(settings, mocker)

        main_trainer.main(settings)

        batch_cls.assert_not_called()

    def test_job_runs_once_through_a_budgeted_port(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        from src.application.metering import BudgetedLLMPort

        batch_cls, _ = self._trainer(settings, mocker)
        settings.batch_diagnosis_enabled = True
        settings.slow_loop_max_paid_llm_calls_per_week = 15

        main_trainer.main(settings)

        assert isinstance(batch_cls.call_args.kwargs["llm_port"], BudgetedLLMPort)
        batch_cls.return_value.run.assert_called_once()

    def test_batch_goes_before_the_backfill_in_the_weekly_cap(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        _, backfill_cls = self._trainer(settings, mocker)
        settings.weekly_recap_enabled = True
        settings.batch_diagnosis_enabled = True
        settings.diagnosis_backfill_enabled = True
        settings.diagnosis_backfill_max_calls = 5
        settings.slow_loop_max_paid_llm_calls_per_week = 3

        main_trainer.main(settings)

        # Sufit 3: recap 1 + diagnoza zbiorcza 1 → na spłatę diagnoz zostaje 1.
        assert backfill_cls.call_args.kwargs["max_paid_calls"] == 1


class TestDiagnosisJudgeJob:
    """EXP-10: sędzia diagnoz z rubryką — za flagą, najwyżej 10 wywołań przez
    przydział tygodniowego budżetu FIN-30, ostatni w kolejce (przycinany pierwszy).
    Odrzuca diagnozy tylko z `diagnosis_judge_rejects_enabled`, bez niej tryb cienia."""

    def _trainer(self, settings: Settings, mocker: Any) -> MagicMock:
        from src.application.ports import LLMPort

        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        mocker.patch("main_trainer.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent.build_llm_adapter", return_value=MagicMock(spec=LLMPort))
        recap_cls = mocker.patch("main_trainer.WeeklyRecapUseCase")
        recap_cls.return_value.run.return_value = {"status": "skipped_below_threshold"}
        backfill_cls = mocker.patch("main_trainer.BackfillDiagnosesUseCase")
        backfill_cls.return_value.run.return_value = {"status": "backfilled"}
        judge_cls: MagicMock = mocker.patch("main_trainer.JudgeDiagnosesUseCase")
        judge_cls.return_value.run.return_value = {"status": "judged"}
        settings.symbols = ["AAPL"]
        settings.paid_calls_enabled = True
        return judge_cls

    def test_off_by_default(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        judge_cls = self._trainer(settings, mocker)

        main_trainer.main(settings)

        judge_cls.assert_not_called()

    def test_job_samples_at_most_ten_through_a_budgeted_port(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        from src.application.metering import BudgetedLLMPort

        judge_cls = self._trainer(settings, mocker)
        settings.diagnosis_judge_enabled = True
        settings.slow_loop_max_paid_llm_calls_per_week = 15

        main_trainer.main(settings)

        kwargs = judge_cls.call_args.kwargs
        assert kwargs["max_samples"] == 10
        assert isinstance(kwargs["llm_port"], BudgetedLLMPort)
        judge_cls.return_value.run.assert_called_once()

    def test_weekly_cap_trims_the_judge_first(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        judge_cls = self._trainer(settings, mocker)
        settings.weekly_recap_enabled = True
        settings.diagnosis_backfill_enabled = True
        settings.diagnosis_backfill_max_calls = 5
        settings.diagnosis_judge_enabled = True
        settings.slow_loop_max_paid_llm_calls_per_week = 9

        main_trainer.main(settings)

        # Sufit 9: recap 1 + spłata diagnoz 5 → sędzia dostaje 3.
        assert judge_cls.call_args.kwargs["max_samples"] == 3

    def test_judge_runs_in_shadow_mode_by_default(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        judge_cls = self._trainer(settings, mocker)
        settings.diagnosis_judge_enabled = True

        main_trainer.main(settings)

        assert judge_cls.call_args.kwargs.get("rejects_enabled") is False

    def test_rejects_setting_lets_the_judge_reject(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        judge_cls = self._trainer(settings, mocker)
        settings.diagnosis_judge_enabled = True
        settings.diagnosis_judge_rejects_enabled = True

        main_trainer.main(settings)

        assert judge_cls.call_args.kwargs.get("rejects_enabled") is True


class TestSlowLoopCostLedger:
    """FIN-13: recap i sędzia kalibracji zgłaszają tokeny do ledgera przebiegu
    Slow Loopa; zapis `llm_calls` za tą samą flagą co w Fast Loopie."""

    def _trainer(self, settings: Settings, mocker: Any) -> tuple[MagicMock, MagicMock]:
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
        self,
        settings: Settings,
        mock_external_clients: None,
        mocker: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_RUN_ID", "18400")
        build_llm, persist = self._trainer(settings, mocker)

        main_trainer.main(settings)

        ledger = persist.call_args.kwargs["ledger"]
        assert build_llm.call_args.kwargs["usage_sink"] is ledger
        assert persist.call_args.kwargs["run_context"].run_id == "18400"

    def test_recap_usage_is_attributed_to_its_job(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        from src.application.ports import LLMPort

        build_llm, persist = self._trainer(settings, mocker)

        def provider(
            _settings: object, quota_monitor: object = None, usage_sink: Any = None
        ) -> MagicMock:
            port = MagicMock(spec=LLMPort)
            port.analyze.side_effect = lambda _prompt: usage_sink.record(build_usage()) or {}
            return port

        build_llm.side_effect = provider
        recap_cls = cast(Any, main_trainer).WeeklyRecapUseCase
        recap_cls.return_value.run.side_effect = lambda **_: (
            recap_cls.call_args.kwargs["llm_port"].analyze("recap prompt")
            or {"status": "skipped_below_threshold"}
        )

        main_trainer.main(settings)

        records = persist.call_args.kwargs["ledger"].records()
        assert [(r.source, r.node, r.symbol) for r in records] == [("llm", "recap", None)]


class TestSlowLoopEarningsCache:
    """FIN-11: Slow Loop raz w tygodniu odświeża `earnings_cache` z Alpha Vantage
    całą listą kluczy — tylko dla spółek (ETF i krypto nie mają raportów)."""

    def _refresh(self, settings: Settings, mocker: Any) -> tuple[MagicMock, MagicMock]:
        av = mocker.patch("main_trainer.AlphaVantageEarningsAdapter")
        cache = mocker.patch("main_trainer.CachedEarningsAdapter")
        settings.symbols = ["AAPL", "VOO", "MSFT"]
        settings.symbols_etf = ["VOO"]
        settings.crypto_symbols = ["BTC"]
        main_trainer._refresh_earnings_cache(settings, MagicMock(spec=RepositoryPort))
        return av, cache

    def test_refreshes_stocks_with_every_alpha_vantage_key(
        self, settings: Settings, mocker: Any
    ) -> None:
        settings.earnings_calendar_enabled = True
        settings.earnings_cache_enabled = True

        av, cache = self._refresh(settings, mocker)

        av.assert_called_once_with(api_key="av1", reserve_keys=["av2"])
        assert cache.call_args.kwargs["delegate"] is av.return_value
        assert [c.args[0] for c in cache.return_value.get_next_earnings.call_args_list] == [
            "AAPL",
            "MSFT",
        ]

    def test_one_failing_symbol_does_not_stop_the_refresh(
        self, settings: Settings, mocker: Any
    ) -> None:
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
    def test_refresh_is_skipped(
        self, settings: Settings, mocker: Any, calendar: bool, cache_flag: bool, paid: bool
    ) -> None:
        settings.earnings_calendar_enabled = calendar
        settings.earnings_cache_enabled = cache_flag
        settings.paid_calls_enabled = paid

        av, cache = self._refresh(settings, mocker)

        av.assert_not_called()
        cache.assert_not_called()

    def test_slow_loop_runs_the_refresh(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        mocker.patch("main_trainer.AlphaVantageFundamentalsAdapter")
        refresh = mocker.patch("main_trainer._refresh_earnings_cache")

        main_trainer.main(settings)

        refresh.assert_called_once()
