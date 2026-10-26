"""NIEZ-13: Fast Loop i Slow Loop pingują zewnętrzny heartbeat na końcu przebiegu.

`ok` = przebieg zakończony kodem 0. Wyjątek przebiegu pinguje porażkę i leci
dalej (job w Actions i tak pada), a błąd samego pingu tylko loguje. Brak
`HEARTBEAT_URL` = `NullHeartbeat`, czyli zero requestów.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

import main_agent
import main_trainer
from src.application.ports import HeartbeatPort
from src.config import Settings
from src.infrastructure.adapters.http_heartbeat import HttpHeartbeat, NullHeartbeat
from tests.settings_kwargs import NO_ENV_FILE


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
    )


@pytest.fixture
def mock_external_clients(mocker: Any) -> None:
    mocker.patch("src.infrastructure.adapters.supabase_repo.build_supabase_client")
    mocker.patch("src.infrastructure.llm.openai_client.OpenAI")


@pytest.fixture
def heartbeat(mocker: Any) -> MagicMock:
    port = MagicMock(spec=HeartbeatPort)
    mocker.patch("main_agent.build_heartbeat", return_value=port)
    return port


class TestBuildHeartbeat:
    def test_without_a_url_the_heartbeat_is_null(self, settings: Settings) -> None:
        assert isinstance(main_agent.build_heartbeat(settings), NullHeartbeat)

    def test_blank_url_is_treated_as_missing(self, settings: Settings) -> None:
        settings.heartbeat_url = "  "

        assert isinstance(main_agent.build_heartbeat(settings), NullHeartbeat)

    def test_url_builds_the_http_heartbeat(self, settings: Settings) -> None:
        settings.heartbeat_url = "https://hc-ping.com/5f3c0b9e-7a1d-4c2e-9b8f-0d6a2e4c1b7f"

        assert isinstance(main_agent.build_heartbeat(settings), HttpHeartbeat)


class TestFastLoopHeartbeat:
    def test_successful_cycle_pings_ok(
        self, settings: Settings, mock_external_clients: None, mocker: Any, heartbeat: MagicMock
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent._dispatch_reports", return_value=True)

        assert main_agent.main(settings) == 0

        heartbeat.ping.assert_called_once_with(ok=True)

    def test_cycle_where_every_symbol_failed_pings_a_failure(
        self, settings: Settings, mock_external_clients: None, mocker: Any, heartbeat: MagicMock
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.run.side_effect = RuntimeError("API down")
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent._dispatch_reports", return_value=True)

        assert main_agent.main(settings) == 1

        heartbeat.ping.assert_called_once_with(ok=False)

    def test_refused_start_pings_a_failure(self, settings: Settings, heartbeat: MagicMock) -> None:
        # FIN-35: zapis z atrapami = odmowa startu (exit 2) — to nie jest zdrowy przebieg.
        settings.paid_calls_enabled = False

        assert main_agent.main(settings) == main_agent.STUB_WRITES_REFUSED_EXIT_CODE

        heartbeat.ping.assert_called_once_with(ok=False)

    def test_crashed_cycle_pings_a_failure_and_reraises(
        self, settings: Settings, mock_external_clients: None, mocker: Any, heartbeat: MagicMock
    ) -> None:
        mocker.patch("main_agent.build_use_case", side_effect=RuntimeError("DI broke"))

        with pytest.raises(RuntimeError, match="DI broke"):
            main_agent.main(settings)

        heartbeat.ping.assert_called_once_with(ok=False)

    def test_failed_ping_is_only_logged(
        self,
        settings: Settings,
        mock_external_clients: None,
        mocker: Any,
        heartbeat: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        heartbeat.ping.side_effect = RuntimeError("heartbeat ping failed (ConnectTimeout)")
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent._dispatch_reports", return_value=True)

        with caplog.at_level(logging.WARNING, logger="main_agent"):
            assert main_agent.main(settings) == 0

        assert "heartbeat" in caplog.text.lower()


class TestSlowLoopHeartbeat:
    def test_successful_training_pings_ok(
        self, settings: Settings, mock_external_clients: None, mocker: Any, heartbeat: MagicMock
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "skipped", "n_samples": 3}
        fake_uc.run_pooled.return_value = {"status": "skipped", "n_samples": 6}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        assert main_trainer.main(settings) == 0

        heartbeat.ping.assert_called_once_with(ok=True)

    def test_failed_feature_store_refresh_pings_a_failure(
        self, settings: Settings, mock_external_clients: None, mocker: Any, heartbeat: MagicMock
    ) -> None:
        fake_uc = MagicMock()
        fake_uc.refresh_feature_store.side_effect = RuntimeError("DB down")
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)

        assert main_trainer.main(settings) == 1

        heartbeat.ping.assert_called_once_with(ok=False)
