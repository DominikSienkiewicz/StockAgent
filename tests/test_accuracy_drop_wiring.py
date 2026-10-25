"""NIEZ-20: Fast Loop liczy detektor trafności w Track Record i dokłada alarm
do bannera maila tego cyklu (bez zapisu — alarm liczy się na nowo co cykl)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

import main_agent
from src.application.ports import RepositoryPort
from src.application.quota_monitor import QuotaMonitor
from src.application.report_builder import ReportContext
from src.application.report_evidence import ACCURACY_DROP_ALERT_SOURCE
from src.application.report_track_record import TrackRecordStats
from src.config import Settings
from src.domain.change_detection import AccuracyDrop
from tests.settings_kwargs import NO_ENV_FILE

# ARCH-06: sekcje raportu składa `ReportAssembler` — tam patchujemy odczyty.
ASSEMBLY = "src.application.use_cases.assemble_report"

STARTED_AT = datetime(2026, 7, 21, 5, 35, tzinfo=UTC)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="sb_secret_test",
        symbols=["AAPL"],
        equity_curve_enabled=True,
    )


def _rows() -> list[dict[str, Any]]:
    first = datetime(2026, 6, 1, 5, 35, tzinfo=UTC)
    rows: list[dict[str, Any]] = []
    for day in range(70):
        hits = 6 if day < 30 else 4
        for index in range(10):
            hit = index < hits
            rows.append(
                {
                    "symbol": f"S{index}",
                    "predicted_trend": "BULLISH",
                    "is_trend_correct": hit,
                    "timestamp": (first + timedelta(days=day)).isoformat(),
                    "price_at_prediction": 100.0,
                    "actual_price_after_12h": 101.0 if hit else 99.0,
                    "reasoning_text": "",
                    "correction_insights": None,
                    "data_quality_flags": [],
                }
            )
    return rows


def test_track_record_stats_carry_the_drop(settings: Settings) -> None:
    repository = Mock(spec=RepositoryPort)
    repository.get_resolved_predictions.return_value = _rows()

    stats = main_agent._build_track_record_stats(settings, repository)

    assert stats is not None
    assert stats.accuracy_drop is not None
    assert stats.accuracy_drop.since == "2026-07-01"


def test_drop_alert_reaches_the_banner_of_this_report(settings: Settings, mocker: Any) -> None:
    settings.messenger_digest_enabled = False
    settings.attestation_enabled = False
    settings.cycle_diff_enabled = False
    drop = AccuracyDrop(
        since="2026-07-01", alarm_cycle="2026-07-20", before_rate=0.6, after_rate=0.4, cycles=70
    )
    mocker.patch(f"{ASSEMBLY}.fetch_report_context", return_value=(None, [], [], [], []))
    mocker.patch(f"{ASSEMBLY}.build_track_record", return_value=(None, None, None))
    mocker.patch(
        f"{ASSEMBLY}.build_track_record_stats",
        return_value=TrackRecordStats(accuracy_drop=drop),
    )
    mocker.patch(f"{ASSEMBLY}.build_signal_calibration", return_value=None)
    mocker.patch(f"{ASSEMBLY}.build_model_scorecard", return_value=None)
    mocker.patch("main_agent._build_user_portfolio", return_value=None)
    mocker.patch("main_agent.build_public_scorecard_publisher", return_value=MagicMock())
    mocker.patch("main_agent._publish_public_scorecard")
    report = mocker.patch("main_agent.build_html_report", return_value=("<html>", "text"))
    quota_monitor = QuotaMonitor()

    main_agent._dispatch_reports(
        results=[],
        started_at=STARTED_AT,
        duration=1.0,
        failures=0,
        settings=settings,
        repository=MagicMock(spec=RepositoryPort),
        quota_monitor=quota_monitor,
        notifier=MagicMock(),
        subscribers=[],
        all_symbols=["AAPL"],
        context=ReportContext(),
    )

    context = report.call_args_list[0].args[3]
    (alert,) = context.quota_alerts
    assert alert.source == ACCURACY_DROP_ALERT_SOURCE
    assert "2026-07-01" in alert.message
    # Alarm liczony co cykl — nie trafia do zapisu `quota_alerts`.
    assert quota_monitor.alerts == []
