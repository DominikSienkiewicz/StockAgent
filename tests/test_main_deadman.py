"""Deadman (NIEZ-11): raz dziennie sprawdza ślady życia jobów w bazie i mailuje,
gdy job, który powinien chodzić, zamilkł. Brak maila nie może wyglądać jak spokojny tydzień."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import main_deadman
from src.application.ports import RepositoryPort
from src.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
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


def _wire(mocker, repository: MagicMock) -> MagicMock:
    mocker.patch("main_agent.build_repository", return_value=repository)
    notifier = MagicMock()
    mocker.patch("main_agent.build_notifier", return_value=notifier)
    return notifier


def _repo(slow_loop_last_seen: datetime | None) -> MagicMock:
    repository = MagicMock(spec=RepositoryPort)
    repository.get_last_scorecard_at.return_value = slow_loop_last_seen
    return repository


def test_silent_slow_loop_sends_one_mail(settings, mocker) -> None:
    notifier = _wire(mocker, _repo(datetime.now(UTC) - timedelta(days=9)))

    assert main_deadman.main(settings) == 0

    notifier.send_report.assert_called_once()
    assert "Slow Loop" in notifier.send_report.call_args.kwargs["subject"]


def test_recent_heartbeat_sends_nothing(settings, mocker) -> None:
    notifier = _wire(mocker, _repo(datetime.now(UTC) - timedelta(days=2)))

    assert main_deadman.main(settings) == 0

    notifier.send_report.assert_not_called()


def test_paused_fast_loop_is_not_checked(settings, mocker) -> None:
    repository = _repo(datetime.now(UTC) - timedelta(days=2))
    _wire(mocker, repository)

    main_deadman.main(settings)

    repository.get_latest_snapshot_at.assert_not_called()


def test_unreadable_heartbeat_exits_1_so_the_workflow_alert_fires(settings, mocker) -> None:
    repository = _repo(None)
    repository.get_last_scorecard_at.side_effect = RuntimeError("Supabase 503")
    notifier = _wire(mocker, repository)

    assert main_deadman.main(settings) == 1

    notifier.send_report.assert_not_called()


def test_deadman_builds_no_paid_adapter(settings, mocker) -> None:
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
