"""NIEZ-22: entry pointy dopisują podsumowanie przebiegu do GITHUB_STEP_SUMMARY.

Zapis robi wyłącznie DI (`main_*.py`), renderery są czyste (`report_ops`).
Brak zmiennej = nic się nie dzieje; błąd zapisu nie zmienia wyniku przebiegu.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import main_agent
import main_trainer
import main_watch
from src.application.ports import RepositoryPort
from src.application.quota_monitor import QuotaMonitor
from src.application.report_builder import ReportContext
from src.config import Settings
from tests.settings_kwargs import NO_ENV_FILE

# ARCH-06: sekcje raportu składa `ReportAssembler` — tam patchujemy odczyty.
ASSEMBLY = "src.application.use_cases.assemble_report"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1", "av2"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["QQQA", "QQQB"],
        volatility_threshold=Decimal("0.02"),
        ml_model_path="data/models/missing.ubj",
        db_writes_enabled=True,
        paid_calls_enabled=True,
    )


@pytest.fixture
def mock_external_clients(mocker: Any) -> None:
    mocker.patch("src.infrastructure.adapters.supabase_repo.build_supabase_client")
    mocker.patch("src.infrastructure.llm.openai_client.OpenAI")


@pytest.fixture
def summary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
    return path


def _fast_loop_use_case(mocker: Any) -> MagicMock:
    fake_uc = MagicMock()
    fake_uc.run.side_effect = [
        {"status": "saved", "delta": Decimal("0.05"), "prediction_id": "p1"},
        TimeoutError("read timed out for https://finnhub.io/api/v1/quote?token=sekret"),
    ]
    fake_uc.telemetry.snapshot.return_value.paid_calls = {"llm": 1}
    mocker.patch("main_agent.build_use_case", return_value=fake_uc)
    return fake_uc


class TestFastLoopSummary:
    def test_main_appends_aggregates_to_the_step_summary(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        _fast_loop_use_case(mocker)
        mocker.patch("main_agent._dispatch_reports", return_value=True)

        assert main_agent.main(settings) == 0

        markdown = summary_file.read_text(encoding="utf-8")
        assert "## Fast Loop" in markdown
        assert "| saved | 1 |" in markdown
        assert "| error | 1 |" in markdown
        assert "| TimeoutError | 1 |" in markdown
        assert "| Failures | 1 |" in markdown
        for symbol in settings.symbols:
            assert symbol not in markdown
        assert "sekret" not in markdown
        assert "timed out" not in markdown

    def test_summary_is_appended_not_overwritten(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        summary_file.write_text("previous step\n", encoding="utf-8")
        _fast_loop_use_case(mocker)
        mocker.patch("main_agent._dispatch_reports", return_value=True)

        main_agent.main(settings)

        markdown = summary_file.read_text(encoding="utf-8")
        assert markdown.startswith("previous step\n")
        assert "## Fast Loop" in markdown

    def test_without_the_variable_nothing_is_written(
        self,
        settings: Settings,
        mock_external_clients: None,
        mocker: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        _fast_loop_use_case(mocker)
        mocker.patch("main_agent._dispatch_reports", return_value=True)
        monkeypatch.chdir(tmp_path)

        assert main_agent.main(settings) == 0

        assert list(tmp_path.iterdir()) == []

    def test_unwritable_summary_does_not_change_the_exit_code(
        self,
        settings: Settings,
        mock_external_clients: None,
        mocker: Any,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Katalog zamiast pliku → OSError przy otwarciu do dopisania.
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path))
        _fast_loop_use_case(mocker)
        mocker.patch("main_agent._dispatch_reports", return_value=True)

        with caplog.at_level(logging.WARNING, logger="main_agent"):
            assert main_agent.main(settings) == 0

        assert "step summary" in caplog.text.lower()

    def test_failed_email_is_reported_as_not_delivered(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        _fast_loop_use_case(mocker)
        mocker.patch("main_agent.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent._dispatch_reports", return_value=False)

        main_agent.main(settings)

        assert "| Report email | not delivered |" in summary_file.read_text(encoding="utf-8")

    def test_without_an_email_channel_delivery_is_not_configured(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        # notifications_enabled=false → NullNotifier: nic nie wychodzi, ale to nie awaria.
        _fast_loop_use_case(mocker)
        mocker.patch("main_agent._dispatch_reports", return_value=True)

        main_agent.main(settings)

        assert "| Report email | not configured |" in summary_file.read_text(encoding="utf-8")


class TestDispatchReportsDeliveryResult:
    """`_dispatch_reports` zwraca, czy mail wyszedł — summary i `cycle_runs` go czytają."""

    @staticmethod
    def _dispatch(settings: Settings, mocker: Any, notifier: MagicMock) -> bool:
        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
        mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
        mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
        mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
        mocker.patch("main_agent._build_user_portfolio", return_value=None)
        mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
        mocker.patch("main_agent._publish_public_scorecard")
        return main_agent._dispatch_reports(
            results=[],
            started_at=datetime(2026, 10, 20, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=MagicMock(spec=RepositoryPort),
            quota_monitor=QuotaMonitor(),
            notifier=notifier,
            subscribers=[],
            all_symbols=["QQQA"],
            context=ReportContext(),
        )

    def test_sent_email_is_delivered(self, settings: Settings, mocker: Any) -> None:
        assert self._dispatch(settings, mocker, MagicMock()) is True

    def test_failed_email_is_not_delivered(self, settings: Settings, mocker: Any) -> None:
        notifier = MagicMock()
        notifier.send_report.side_effect = RuntimeError("Resend 503")

        assert self._dispatch(settings, mocker, notifier) is False

    def test_report_that_never_got_built_is_not_delivered(
        self, settings: Settings, mocker: Any
    ) -> None:
        mocker.patch(f"{ASSEMBLY}.build_track_record", side_effect=RuntimeError("boom"))
        settings.messenger_digest_enabled = False
        settings.attestation_enabled = False
        settings.cycle_diff_enabled = False
        mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))

        delivered = main_agent._dispatch_reports(
            results=[],
            started_at=datetime(2026, 10, 20, 5, 30, tzinfo=UTC),
            duration=1.0,
            failures=0,
            settings=settings,
            repository=MagicMock(spec=RepositoryPort),
            quota_monitor=QuotaMonitor(),
            notifier=MagicMock(),
            subscribers=[],
            all_symbols=["QQQA"],
            context=ReportContext(),
        )

        assert delivered is False


class TestWatchSummary:
    def test_main_appends_checked_stale_and_sent(self, mocker: Any, summary_file: Path) -> None:
        settings = Settings(
            **NO_ENV_FILE,
            openai_api_key="sk-test",
            finnhub_api_key="fh",
            alpha_vantage_api_keys=["av1"],
            supabase_url="https://test.supabase.co",
            supabase_key="anon",
            symbols=["QQQA"],
            crypto_symbols=[],
            shock_alerts_enabled=True,
        )
        mocker.patch("main_agent.build_repository")
        mocker.patch("main_agent.build_market_port")
        mocker.patch(
            "main_watch.scan_watch",
            return_value=main_watch.WatchOutcome(sent=1, failures=0, total=3, checked=3, stale=1),
        )

        assert main_watch.main(settings) == 0

        markdown = summary_file.read_text(encoding="utf-8")
        assert "## Shock Watch" in markdown
        assert "| Checked | 3 |" in markdown
        assert "| Stale reference | 1 |" in markdown
        assert "| Alerts sent | 1 |" in markdown
        assert "QQQA" not in markdown

    def test_scan_counts_checked_and_stale_references(self) -> None:
        from datetime import timedelta

        from src.domain.value_objects import Money

        settings = Settings(
            **NO_ENV_FILE,
            openai_api_key="sk-test",
            finnhub_api_key="fh",
            alpha_vantage_api_keys=["av1"],
            supabase_url="https://test.supabase.co",
            supabase_key="anon",
            symbols=["QQQA", "QQQB"],
            crypto_symbols=[],
            shock_alerts_enabled=True,
        )
        moment = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
        repo = MagicMock(spec=RepositoryPort)
        repo.get_sent_shock_alerts.return_value = set()
        repo.get_last_price_snapshot.side_effect = [
            (Money(Decimal("100")), moment - timedelta(hours=1)),
            (Money(Decimal("100")), moment - timedelta(hours=72)),
        ]
        market = MagicMock()
        market.get_current_price.return_value = Money(Decimal("100.5"))

        outcome = main_watch.scan_watch(
            settings, repository=repo, market=market, push_notifier=MagicMock(), now=moment
        )

        assert (outcome.checked, outcome.stale, outcome.sent) == (2, 1, 0)


class TestTrainerSummary:
    def test_main_appends_status_counts(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped", "n_samples": 12}
        fake_uc.run_pooled.return_value = {"status": "skipped", "n_samples": 24}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        assert main_trainer.main(settings) == 0

        markdown = summary_file.read_text(encoding="utf-8")
        assert "## Slow Loop" in markdown
        assert "| skipped | 2 |" in markdown
        assert "| Pooled model | skipped |" in markdown
        assert "| Pooled samples | 24 |" in markdown
        assert "| Failures | 0 / 3 |" in markdown
        for symbol in settings.symbols:
            assert symbol not in markdown

    def test_summary_shows_the_feature_snapshot_models(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        from src.application.use_cases.train_model import SnapshotModelsResult
        from src.domain.model_scorecard import TrainingOutcome, TrainingReport
        from src.domain.training_labels import ReturnDistribution, SelectionReport

        empty = TrainingReport(TrainingOutcome.SKIPPED, n_samples=0)
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped", "n_samples": 12}
        fake_uc.run_pooled.return_value = {"status": "skipped", "n_samples": 24}
        fake_uc.run_snapshot_models.return_value = SnapshotModelsResult(
            free_model=empty,
            llm_model=empty,
            selection=SelectionReport(ReturnDistribution(n=0), ReturnDistribution(n=0)),
            n_snapshots=0,
            n_labeled=0,
        )
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        assert main_trainer.main(settings) == 0

        markdown = summary_file.read_text(encoding="utf-8")
        assert "### Feature snapshot models (ML-52)" in markdown
        assert "| Free features (all cycles) | skipped | 0 | n/a | n/a |" in markdown

    def test_summary_shows_the_label_overlap_of_the_pooled_model(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        # ML-25: liczba nakładających się etykiet z wyniku treningu wspólnego.
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped", "n_samples": 12}
        fake_uc.run_pooled.return_value = {
            "status": "trained_successfully",
            "n_samples": 240,
            "label_overlap_count": 7,
            "label_overlap_measured": 230,
        }
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        assert main_trainer.main(settings) == 0

        markdown = summary_file.read_text(encoding="utf-8")
        assert "| Label overlaps (settle after the next row) | 7 / 230 |" in markdown

    def test_direction_variant_runs_only_with_its_flag(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        # ML-30: wariant binarny to sama ewaluacja w Slow Loopie, za flagą off.
        from src.domain.direction_variant import DirectionReport
        from src.domain.model_scorecard import TrainingOutcome

        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped", "n_samples": 12}
        fake_uc.run_pooled.return_value = {"status": "skipped", "n_samples": 24}
        fake_uc.run_direction_variant.return_value = DirectionReport(
            TrainingOutcome.SKIPPED, n_samples=20, n_in_dead_band=4, dead_band=0.0
        )
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        assert main_trainer.main(settings) == 0
        fake_uc.run_direction_variant.assert_not_called()
        assert "Direction classifier" not in summary_file.read_text(encoding="utf-8")

        settings.ml_direction_classifier_enabled = True
        assert main_trainer.main(settings) == 0

        # Martwa strefa z protokołu ewaluacji (EXP-06), dziś 0.
        assert fake_uc.run_direction_variant.call_args.kwargs == {"dead_band": 0.0}
        assert "### Direction classifier (ML-30)" in summary_file.read_text(encoding="utf-8")

    def test_failed_direction_variant_does_not_fail_the_run(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped", "n_samples": 12}
        fake_uc.run_pooled.return_value = {"status": "skipped", "n_samples": 24}
        fake_uc.run_direction_variant.side_effect = ValueError("Invalid classes")
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        settings.ml_direction_classifier_enabled = True

        assert main_trainer.main(settings) == 0
        assert "Direction classifier" not in summary_file.read_text(encoding="utf-8")

    def test_challenger_review_runs_only_with_the_shadow_flag(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        # ML-33: werdykt promocji challengerów nogi ML w summary, za flagą off.
        from src.domain.promotion import SettledShadowForecast, ShadowForecast

        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped", "n_samples": 12}
        fake_uc.run_pooled.return_value = {"status": "skipped", "n_samples": 24}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        repository = MagicMock(spec=RepositoryPort)
        repository.get_settled_shadow_predictions.return_value = [
            SettledShadowForecast(
                "p1", ShadowForecast("champion", "a" * 64, predicted_return=0.01), 0.02
            ),
            SettledShadowForecast("p1", ShadowForecast("direction", "c" * 64, proba=0.7), 0.02),
        ]
        mocker.patch("main_trainer.build_repository", return_value=repository)

        assert main_trainer.main(settings) == 0
        repository.get_settled_shadow_predictions.assert_not_called()
        assert "Champion/challenger" not in summary_file.read_text(encoding="utf-8")

        settings.shadow_predictions_enabled = True
        assert main_trainer.main(settings) == 0

        markdown = summary_file.read_text(encoding="utf-8")
        assert "### Champion/challenger (ML-33)" in markdown
        assert "| direction | direction Brier | 1 | 1 |" in markdown

    def test_failed_challenger_review_does_not_fail_the_run(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        from src.application.errors import NotFound

        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped", "n_samples": 12}
        fake_uc.run_pooled.return_value = {"status": "skipped", "n_samples": 24}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        repository = MagicMock(spec=RepositoryPort)
        repository.get_settled_shadow_predictions.side_effect = NotFound("no table before 075")
        mocker.patch("main_trainer.build_repository", return_value=repository)
        settings.shadow_predictions_enabled = True

        assert main_trainer.main(settings) == 0
        assert "Champion/challenger" not in summary_file.read_text(encoding="utf-8")

    def test_failed_feature_store_refresh_still_writes_a_summary(
        self, settings: Settings, mock_external_clients: None, mocker: Any, summary_file: Path
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.refresh_feature_store.side_effect = RuntimeError("DB down")
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        assert main_trainer.main(settings) == 1

        markdown = summary_file.read_text(encoding="utf-8")
        assert "## Slow Loop" in markdown
        assert "| Feature store refresh | failed |" in markdown
        assert "| Pooled model | failed |" in markdown
