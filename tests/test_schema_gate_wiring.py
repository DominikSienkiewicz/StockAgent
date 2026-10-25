"""NIEZ-30: Fast Loop i Slow Loop pytają bazę o wersję schematu przy starcie.

Flaga zależna od migracji, której baza jeszcze nie ma, wyłącza się na czas przebiegu
(WARNING w logu, alert w bannerze maila) zamiast kończyć się PGRST204. Gdy wersji nie
da się ustalić, flagi zostają (decyzja właściciela, fail-open), a ostrzeżenie trafia do
logu i do podsumowania przebiegu — bez alertu w mailu.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import main_agent
import main_trainer
import main_watch
from src.application.errors import NotFound
from src.application.ports import RepositoryPort
from src.application.schema_gate import SCHEMA_ALERT_SOURCE
from src.config import Settings
from src.domain.quota import QuotaSeverity
from src.infrastructure.adapters.supabase_repo import SupabaseRepository
from tests.settings_kwargs import NO_ENV_FILE

# 019 < 020 (receipts_enabled) i 021 (shock_alerts_enabled), 042 jest dziś na produkcji.
SCHEMA_BEHIND_RECEIPTS = 19
SCHEMA_AHEAD_OF_EVERYTHING = 71


@pytest.fixture
def settings() -> Settings:
    return Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1", "av2"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL", "VOO"],
        volatility_threshold=Decimal("0.02"),
        ml_model_path="data/models/missing.ubj",
        db_writes_enabled=True,
        paid_calls_enabled=True,
        receipts_enabled=True,
        shock_alerts_enabled=True,
        model_scorecard_enabled=True,
    )


@pytest.fixture
def mock_external_clients(mocker: Any) -> None:
    mocker.patch("src.infrastructure.adapters.supabase_repo.create_client")
    mocker.patch("src.infrastructure.llm.openai_client.OpenAI")


@pytest.fixture
def summary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
    return path


class _FastLoop:
    """Jeden przebieg `main_agent.main` na atrapach: repozytorium z zadaną wersją schematu."""

    def __init__(self, mocker: Any, settings: Settings, schema: object) -> None:
        self.repository = MagicMock(spec=RepositoryPort)
        if isinstance(schema, Exception):
            self.repository.schema_version.side_effect = schema
        else:
            self.repository.schema_version.return_value = schema
        self.build_repository = mocker.patch(
            "main_agent.build_repository", return_value=self.repository
        )
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "saved", "delta": Decimal("0.05")}
        fake_uc.telemetry.snapshot.return_value.paid_calls = {}
        self.build_use_case = mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent._dispatch_reports", return_value=True)
        self.settings = settings

    def run(self) -> int:
        return main_agent.main(self.settings)

    @property
    def settings_used(self) -> Settings:
        used: Settings = self.build_use_case.call_args.args[0]
        return used

    @property
    def alerts(self) -> list[Any]:
        alerts: list[Any] = self.build_use_case.call_args.kwargs["quota_monitor"].alerts
        return alerts


class TestFastLoopSchemaGate:
    def test_flags_ahead_of_the_schema_are_off_for_the_run(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        loop = _FastLoop(mocker, settings, SCHEMA_BEHIND_RECEIPTS)

        assert loop.run() == 0

        assert loop.settings_used.receipts_enabled is False
        assert loop.settings_used.shock_alerts_enabled is False
        assert loop.settings_used.model_scorecard_enabled is True  # migracja 019 jest

    def test_the_banner_gets_one_warning_alert(
        self,
        settings: Settings,
        mock_external_clients: None,
        mocker: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        loop = _FastLoop(mocker, settings, SCHEMA_BEHIND_RECEIPTS)

        with caplog.at_level(logging.WARNING):
            loop.run()

        assert [a.source for a in loop.alerts] == [SCHEMA_ALERT_SOURCE]
        (alert,) = loop.alerts
        assert alert.severity is QuotaSeverity.WARNING
        assert "receipts_enabled" in alert.message
        assert "switched off for this run" in caplog.text

    def test_the_repository_is_rebuilt_with_the_gated_flags(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        loop = _FastLoop(mocker, settings, SCHEMA_BEHIND_RECEIPTS)

        loop.run()

        assert loop.build_repository.call_count == 2
        assert loop.build_repository.call_args.args[0].receipts_enabled is False
        assert loop.build_use_case.call_args.kwargs["repository"] is self._last_repository(loop)

    @staticmethod
    def _last_repository(loop: _FastLoop) -> RepositoryPort:
        built: RepositoryPort = loop.build_repository.return_value
        return built

    def test_a_migrated_database_leaves_settings_and_repository_alone(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        loop = _FastLoop(mocker, settings, SCHEMA_AHEAD_OF_EVERYTHING)

        loop.run()

        assert loop.settings_used is settings
        assert settings.receipts_enabled is True
        assert loop.alerts == []
        assert loop.build_repository.call_count == 1

    def test_an_unavailable_probe_keeps_the_flags_and_notes_it_in_the_summary(
        self,
        settings: Settings,
        mock_external_clients: None,
        mocker: Any,
        summary_file: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        loop = _FastLoop(mocker, settings, NotFound("PGRST202"))

        with caplog.at_level(logging.WARNING):
            assert loop.run() == 0

        assert loop.settings_used.receipts_enabled is True
        assert loop.alerts == []
        assert "left on" in caplog.text
        markdown = summary_file.read_text(encoding="utf-8")
        assert "Schema probe" in markdown
        assert "## Fast Loop cycle summary" in markdown

    def test_a_null_version_is_not_read_as_an_empty_database(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        loop = _FastLoop(mocker, settings, None)

        loop.run()

        assert loop.settings_used.receipts_enabled is True
        assert loop.alerts == []

    def test_a_flag_left_unset_follows_regime_aware_when_checking_the_schema(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        # regime_tagging_enabled = None idzie za regime_aware_enabled i potrzebuje 017.
        settings.regime_aware_enabled = True
        settings.regime_tagging_enabled = None
        loop = _FastLoop(mocker, settings, 16)

        loop.run()

        assert loop.settings_used.regime_tagging_enabled is False


class TestSlowLoopSchemaGate:
    @pytest.fixture
    def trainer(
        self, mocker: Any, settings: Settings, mock_external_clients: None
    ) -> TestSlowLoopSchemaGate:
        self.use_case = MagicMock()
        self.use_case.run.return_value = {"status": "skipped", "n_samples": 1}
        self.use_case.run_pooled.return_value = {"status": "skipped", "n_samples": 1}
        self.build_use_case = mocker.patch(
            "main_trainer.build_use_case", return_value=self.use_case
        )
        self.build_repository = mocker.spy(main_trainer, "build_repository")
        self.saved_alerts = mocker.patch.object(SupabaseRepository, "save_quota_alert")
        return self

    def _run(self, mocker: Any, settings: Settings, schema: object) -> int:
        mocker.patch.object(SupabaseRepository, "schema_version", return_value=schema)
        return main_trainer.main(settings)

    def test_flags_ahead_of_the_schema_are_off_for_the_run(
        self, mocker: Any, settings: Settings, trainer: TestSlowLoopSchemaGate
    ) -> None:
        assert self._run(mocker, settings, SCHEMA_BEHIND_RECEIPTS) == 0

        used: Settings = self.build_use_case.call_args.args[0]
        assert used.receipts_enabled is False
        assert used.shock_alerts_enabled is False
        assert used.model_scorecard_enabled is True
        assert self.build_repository.call_count >= 2
        assert self.build_repository.call_args.args[0].receipts_enabled is False

    def test_the_alert_is_persisted_for_the_next_email_banner(
        self, mocker: Any, settings: Settings, trainer: TestSlowLoopSchemaGate
    ) -> None:
        self._run(mocker, settings, SCHEMA_BEHIND_RECEIPTS)

        sources = [call.args[0].source for call in self.saved_alerts.call_args_list]
        assert sources.count(SCHEMA_ALERT_SOURCE) == 1

    def test_a_migrated_database_changes_nothing(
        self, mocker: Any, settings: Settings, trainer: TestSlowLoopSchemaGate
    ) -> None:
        self._run(mocker, settings, SCHEMA_AHEAD_OF_EVERYTHING)

        assert self.build_use_case.call_args.args[0] is settings
        sources = [call.args[0].source for call in self.saved_alerts.call_args_list]
        assert SCHEMA_ALERT_SOURCE not in sources

    def test_an_unavailable_probe_keeps_the_flags(
        self,
        mocker: Any,
        settings: Settings,
        trainer: TestSlowLoopSchemaGate,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mocker.patch.object(SupabaseRepository, "schema_version", side_effect=NotFound("x"))

        with caplog.at_level(logging.WARNING):
            assert main_trainer.main(settings) == 0

        assert self.build_use_case.call_args.args[0].receipts_enabled is True
        assert "left on" in caplog.text


class TestWatchSchemaGate:
    """W5 (fala 3): godzinny watch też zapisuje do bazy (`shock_alerts`, migracja 021),
    więc pyta o schemat zaraz po `build_repository`, jak Slow Loop."""

    @pytest.fixture
    def watch(self, mocker: Any) -> TestWatchSchemaGate:
        self.repository = MagicMock(spec=RepositoryPort)
        self.build_repository = mocker.patch(
            "main_agent.build_repository", return_value=self.repository
        )
        mocker.patch("main_agent.build_market_port")
        mocker.patch("main_agent.build_push_notifier")
        mocker.patch("main_agent.build_push_channels", return_value=[MagicMock()])
        self.scan_watch = mocker.patch(
            "main_watch.scan_watch",
            return_value=main_watch.WatchOutcome(sent=0, failures=0, total=1),
        )
        self.scan_levels = mocker.patch("main_watch.scan_price_levels")
        mocker.patch("main_watch.scan_invalidations")
        return self

    def test_flags_ahead_of_the_schema_are_off_for_the_watch(
        self, settings: Settings, watch: TestWatchSchemaGate
    ) -> None:
        self.repository.schema_version.return_value = SCHEMA_BEHIND_RECEIPTS

        assert main_watch.main(settings) == 0

        used: Settings = self.scan_watch.call_args.args[0]
        assert used.shock_alerts_enabled is False
        assert self.scan_levels.call_args.args[0].shock_alerts_enabled is False
        assert self.build_repository.call_count == 2
        assert self.build_repository.call_args.args[0].shock_alerts_enabled is False

    def test_a_migrated_database_leaves_the_watch_settings_alone(
        self, settings: Settings, watch: TestWatchSchemaGate
    ) -> None:
        self.repository.schema_version.return_value = SCHEMA_AHEAD_OF_EVERYTHING

        main_watch.main(settings)

        assert self.scan_watch.call_args.args[0] is settings
        assert self.build_repository.call_count == 1
