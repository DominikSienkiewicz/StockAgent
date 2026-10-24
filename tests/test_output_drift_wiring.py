"""NIEZ-19: krok monitora wyjść LLM w Slow Loopie.

Odczyt bez płatnych wywołań; skok rozkładu to WARNING w monitorze kwot (zapisany
potem do `quota_alerts`, więc trafia do bannera maila), a błąd kroku tylko loguje.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

import main_trainer
from src.application.quota_monitor import QuotaMonitor
from src.application.use_cases.monitor_output_drift import DRIFT_ALERT_SOURCE
from src.config import Settings
from src.domain.quota import QuotaSeverity

NOW = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="sb_secret_test",
    )


def _rows(days_ago: int, bullish: int, other: int) -> list[dict[str, Any]]:
    stamp = (NOW - timedelta(days=days_ago)).isoformat()
    return [
        {
            "predicted_trend": "BULLISH" if index < bullish else "BEARISH",
            "confidence_score": 0.6,
            "price_at_prediction": 100.0,
            "actual_price_after_12h": 101.0 if index % 2 else 99.0,
            "timestamp": stamp,
        }
        for index in range(bullish + other)
    ]


def test_jump_records_a_warning_alert(settings: Settings) -> None:
    monitor = QuotaMonitor()
    rows = _rows(10, 30, 70) + _rows(2, 80, 20)

    main_trainer._run_output_drift(settings, monitor, now=NOW, read_rows=lambda days: rows)

    (alert,) = monitor.alerts
    assert alert.source == DRIFT_ALERT_SOURCE
    assert alert.severity is QuotaSeverity.WARNING


def test_stable_distribution_records_nothing(settings: Settings) -> None:
    monitor = QuotaMonitor()
    rows = _rows(10, 40, 60) + _rows(2, 40, 60)

    main_trainer._run_output_drift(settings, monitor, now=NOW, read_rows=lambda days: rows)

    assert monitor.alerts == []


def test_failed_read_is_logged_and_skipped(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(days: int) -> list[dict[str, Any]]:
        raise RuntimeError("database down")

    monitor = QuotaMonitor()
    main_trainer._run_output_drift(settings, monitor, now=NOW, read_rows=broken)

    assert monitor.alerts == []
    assert "Output drift monitor failed" in caplog.text


@pytest.mark.parametrize("recipe_enabled", [False, True])
def test_reader_asks_for_recipe_only_with_its_migration(
    settings: Settings, mocker: Any, recipe_enabled: bool
) -> None:
    settings.recipe_version_enabled = recipe_enabled
    repository = mocker.patch("main_trainer.SupabaseRepository").return_value
    repository.get_resolved_predictions_for_eval.return_value = []

    main_trainer._drift_rows_reader(settings)(14)

    repository.get_resolved_predictions_for_eval.assert_called_once_with(
        14, with_recipe=recipe_enabled, extra_fields=("confidence_score",)
    )
