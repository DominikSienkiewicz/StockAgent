"""Deadman (NIEZ-11): raz dziennie sprawdza ślady życia jobów w bazie i mailuje,
gdy job, który powinien chodzić, zamilkł. Brak maila nie może wyglądać jak spokojny tydzień."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

import main_deadman
from src.application.ports import RepositoryPort
from src.config import Settings
from tests.settings_kwargs import NO_ENV_FILE


@pytest.fixture
def settings() -> Settings:
    return Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL"],
        crypto_symbols=[],
        volatility_threshold=Decimal("0.02"),
        liveness_fast_loop_max_gap_h=0,
        liveness_slow_loop_max_gap_h=192,
    )


def _wire(mocker: Any, repository: MagicMock) -> MagicMock:
    mocker.patch("main_agent.build_repository", return_value=repository)
    notifier = MagicMock()
    mocker.patch("main_agent.build_notifier", return_value=notifier)
    return notifier


def _repo(slow_loop_last_seen: datetime | None) -> MagicMock:
    repository = MagicMock(spec=RepositoryPort)
    repository.get_last_scorecard_at.return_value = slow_loop_last_seen
    return repository


def test_silent_slow_loop_sends_one_mail(settings: Settings, mocker: Any) -> None:
    notifier = _wire(mocker, _repo(datetime.now(UTC) - timedelta(days=9)))

    assert main_deadman.main(settings) == 0

    notifier.send_report.assert_called_once()
    assert "Slow Loop" in notifier.send_report.call_args.kwargs["subject"]


def test_recent_heartbeat_sends_nothing(settings: Settings, mocker: Any) -> None:
    notifier = _wire(mocker, _repo(datetime.now(UTC) - timedelta(days=2)))

    assert main_deadman.main(settings) == 0

    notifier.send_report.assert_not_called()


def test_paused_fast_loop_is_not_checked(settings: Settings, mocker: Any) -> None:
    repository = _repo(datetime.now(UTC) - timedelta(days=2))
    _wire(mocker, repository)

    main_deadman.main(settings)

    repository.get_latest_snapshot_at.assert_not_called()


def test_unreadable_heartbeat_exits_1_so_the_workflow_alert_fires(
    settings: Settings, mocker: Any
) -> None:
    repository = _repo(None)
    repository.get_last_scorecard_at.side_effect = RuntimeError("Supabase 503")
    notifier = _wire(mocker, repository)

    assert main_deadman.main(settings) == 1

    notifier.send_report.assert_not_called()


def test_deadman_builds_no_paid_adapter(settings: Settings, mocker: Any) -> None:
    _wire(mocker, _repo(datetime.now(UTC)))
    paid_builders = [
        mocker.patch(f"main_agent.{name}")
        for name in (
            "build_llm_adapter",
            "build_council_adapter",
            "build_embedding_adapter",
            "build_use_case",
        )
    ]

    main_deadman.main(settings)

    for builder in paid_builders:
        builder.assert_not_called()


# NIEZ-11 (dni handlowe): po wznowieniu Fast Loopa (`liveness_fast_loop_max_gap_h` > 0)
# jego oczekiwanie liczy się w dniach handlowych NYSE — weekend i święto nie alarmują.
FRIDAY_CYCLE = datetime(2026, 10, 16, 5, 35, tzinfo=UTC)
MONDAY_DEADMAN = datetime(2026, 10, 19, 9, 17, tzinfo=UTC)


@pytest.fixture
def resumed_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"liveness_fast_loop_max_gap_h": 26})


def _resumed_repo(fast_loop_last_seen: datetime, now: datetime) -> MagicMock:
    repository = _repo(now - timedelta(days=1))
    repository.get_latest_snapshot_at.return_value = fast_loop_last_seen
    return repository


def test_resumed_fast_loop_after_a_weekend_sends_nothing(
    resumed_settings: Settings, mocker: Any
) -> None:
    notifier = _wire(mocker, _resumed_repo(FRIDAY_CYCLE, MONDAY_DEADMAN))

    assert main_deadman.main(resumed_settings, now=MONDAY_DEADMAN) == 0

    notifier.send_report.assert_not_called()


def test_resumed_fast_loop_silent_for_two_trading_days_sends_one_mail(
    resumed_settings: Settings, mocker: Any
) -> None:
    tuesday_cycle = datetime(2026, 10, 13, 5, 35, tzinfo=UTC)
    thursday_deadman = datetime(2026, 10, 15, 9, 17, tzinfo=UTC)
    notifier = _wire(mocker, _resumed_repo(tuesday_cycle, thursday_deadman))

    assert main_deadman.main(resumed_settings, now=thursday_deadman) == 0

    notifier.send_report.assert_called_once()
    assert "Fast Loop" in notifier.send_report.call_args.kwargs["subject"]


# NIEZ-28: projekt Supabase jest na planie Free, który wstrzymuje projekt po 7 dniach
# bez aktywności. Deadman raz dziennie czyta jeden wiersz (tylko odczyt) i mailuje,
# gdy baza nie odpowiada — także wtedy, gdy wszystkie joby są wstrzymane.


def test_unreachable_database_sends_the_database_alert(settings: Settings, mocker: Any) -> None:
    repository = _repo(datetime.now(UTC))
    repository.probe_database.side_effect = RuntimeError("Supabase 540: project paused")
    notifier = _wire(mocker, repository)

    assert main_deadman.main(settings) == 0

    notifier.send_report.assert_called_once()
    kwargs = notifier.send_report.call_args.kwargs
    assert "Supabase" in kwargs["subject"]
    assert "7 dniach" in kwargs["plain_text"]
    repository.get_last_scorecard_at.assert_not_called()


def test_database_is_probed_even_when_every_job_is_paused(settings: Settings, mocker: Any) -> None:
    repository = _repo(None)
    notifier = _wire(mocker, repository)
    paused = settings.model_copy(update={"liveness_slow_loop_max_gap_h": 0})

    assert main_deadman.main(paused) == 0

    repository.probe_database.assert_called_once_with()
    notifier.send_report.assert_not_called()


def test_failed_database_alert_exits_1_so_the_workflow_alert_fires(
    settings: Settings, mocker: Any
) -> None:
    repository = _repo(datetime.now(UTC))
    repository.probe_database.side_effect = RuntimeError("timeout")
    notifier = _wire(mocker, repository)
    notifier.send_report.side_effect = RuntimeError("Resend 500")

    assert main_deadman.main(settings) == 1


def test_main_builds_settings_via_from_env(settings: Settings, mocker: Any) -> None:
    # ARCH-41: ten sam punkt wejścia konfiguracji co pozostałe entry-pointy
    # (`Settings.from_env`), bez gołego `Settings()` z ignorem mypy.
    settings_cls = mocker.patch("main_deadman.Settings")
    settings_cls.from_env.return_value = settings
    mocker.patch("main_deadman.configure_logging")
    _wire(mocker, _repo(datetime.now(UTC) - timedelta(days=2)))

    main_deadman.main(now=datetime.now(UTC))

    settings_cls.from_env.assert_called_once_with()
