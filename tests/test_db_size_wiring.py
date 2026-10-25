"""NIEZ-37: raport rozmiaru bazy w Slow Loopie — WARNING od 70% `db_size_budget_mb`,
bez retencji; brak RPC przed migracją 077 nie przerywa przebiegu."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

import main_trainer
from src.application.errors import NotFound, ProviderUnavailable
from src.application.ports import RepositoryPort
from src.config import Settings
from src.domain.db_size import BYTES_PER_MB, RelationSize


def _repo(*megabytes: int) -> MagicMock:
    repo = MagicMock(spec=RepositoryPort)
    repo.get_db_size_report.return_value = [
        RelationSize(name, mb * BYTES_PER_MB)
        for name, mb in zip(("prediction_logs", "price_snapshots"), megabytes, strict=False)
    ]
    return repo


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_seventy_percent_of_the_budget_logs_a_warning(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    settings.db_size_budget_mb = 500

    with caplog.at_level(logging.INFO, logger="main_trainer"):
        main_trainer._report_db_size(settings, _repo(300, 100))

    (warning,) = _warnings(caplog)
    assert "400.0 MB" in warning and "80%" in warning
    assert "prediction_logs=300.0 MB" in warning


def test_below_the_threshold_the_size_is_only_reported(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    settings.db_size_budget_mb = 500

    with caplog.at_level(logging.INFO, logger="main_trainer"):
        main_trainer._report_db_size(settings, _repo(200, 100))

    assert _warnings(caplog) == []
    assert any("DB size 300.0 MB" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("error", "hint"),
    [(NotFound("PGRST202"), "migration 077"), (ProviderUnavailable("503"), "failed")],
)
def test_a_missing_or_failing_rpc_only_logs(
    settings: Settings, caplog: pytest.LogCaptureFixture, error: Exception, hint: str
) -> None:
    repo = MagicMock(spec=RepositoryPort)
    repo.get_db_size_report.side_effect = error

    with caplog.at_level(logging.WARNING, logger="main_trainer"):
        main_trainer._report_db_size(settings, repo)

    assert any(hint in message for message in _warnings(caplog))


def test_slow_loop_reports_the_size_and_a_missing_rpc_keeps_exit_zero(
    settings: Settings, mock_external_clients: None, mocker: Any
) -> None:
    fake_uc = MagicMock()
    fake_uc.run.return_value = {"status": "skipped"}
    mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
    repo = MagicMock(spec=RepositoryPort)
    repo.get_db_size_report.side_effect = NotFound("PGRST202")
    mocker.patch("main_trainer.SupabaseRepository", return_value=repo)

    assert main_trainer.main(settings) == 0

    repo.get_db_size_report.assert_called_once_with()
