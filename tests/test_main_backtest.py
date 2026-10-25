"""Smoke testy entry pointu backtestu (`main_backtest`) — DI wiring oraz top-level
main() z mockami sieci."""

from typing import Any
from unittest.mock import MagicMock

import pytest

import main_backtest
from src.config import Settings
from src.domain.backtest import BacktestSummary


class TestMainBacktest:
    """Offline walk-forward backtest harness (T3) — DI + top-level main()."""

    def test_build_use_case_builds_without_error(
        self, settings: Settings, mock_external_clients: None
    ) -> None:
        use_case = main_backtest.build_use_case(settings)
        assert use_case is not None

    def test_main_runs_backtest_per_symbol(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
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
        self, settings: Settings, mock_external_clients: None, mocker: Any, enabled: bool
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
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
    def _summary() -> BacktestSummary:
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
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
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        build = mocker.patch("main_backtest.build_use_case")
        build.return_value.run.return_value = self._summary()

        main_backtest.main(settings)

        build.assert_called_once()
        assert build.call_args.args == (settings,)


def test_backtest_log_reports_ic_next_to_rmse(
    settings: Settings, mock_external_clients: None, mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
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


def test_backtest_log_reports_the_drift_and_llm_baselines(
    settings: Settings, mock_external_clients: None, mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
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


def test_backtest_uses_the_paper_cost_and_logs_net_return(
    settings: Settings, mock_external_clients: None, mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
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
