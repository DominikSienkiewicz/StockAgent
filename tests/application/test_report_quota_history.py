"""Historia alertów nie udaje awarii w bieżącym cyklu raportu."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.application.errors import ProviderUnavailable
from src.application.ports import RepositoryPort
from src.application.quota_monitor import QuotaMonitor
from src.application.report_builder import ReportContext, build_cycle_lead, build_html_report
from src.application.report_evidence import PaperCosts
from src.application.report_formatting import format_local
from src.application.report_track_record import TrackRecordStats
from src.application.use_cases.assemble_report import ReportAssembler, ReportFlags
from src.domain.change_detection import AccuracyDrop
from src.domain.quota import QuotaAlert, QuotaSeverity

STARTED = datetime(2026, 10, 29, 5, 30, tzinfo=UTC)
HISTORY_TITLE = "HISTORIA ALERTÓW (ostatnie 171 h)"
CURRENT_TITLE = "ALERTY SUBSKRYPCJI / LIMITÓW — BIEŻĄCY CYKL"


def _alert(*, hours_ago: int, severity: QuotaSeverity = QuotaSeverity.CRITICAL) -> QuotaAlert:
    return QuotaAlert(
        source="Alpha Vantage",
        severity=severity,
        message="Dostawca odmówił dostępu do danych.",
        action="Sprawdź uprawnienia konta.",
        occurred_at=STARTED - timedelta(hours=hours_ago),
    )


def _repo(alerts: list[QuotaAlert]) -> Mock:
    repo = Mock(spec=RepositoryPort)
    repo.get_accuracy_stats.return_value = None
    repo.get_recently_resolved_predictions.return_value = []
    repo.get_recent_quota_alerts.return_value = alerts
    repo.get_council_vote_history.return_value = []
    repo.get_persona_track_record.return_value = {}
    return repo


def _assemble(repo: Mock, current: list[QuotaAlert]) -> ReportContext:
    monitor = QuotaMonitor()
    for alert in current:
        monitor.record(alert)
    return (
        ReportAssembler(
            repo,
            ReportFlags(track_record_days=30, reflection_min_age_hours=12),
            PaperCosts(cost_bps=10.0),
        )
        .assemble(
            results=[],
            started_at=STARTED,
            quota_monitor=monitor,
            all_symbols=[],
            context=ReportContext(),
        )
        .context
    )


@pytest.mark.parametrize("compact", [False, True])
def test_historical_critical_alert_is_history_without_current_warning(compact: bool) -> None:
    old = _alert(hours_ago=170)
    repo = _repo([old])

    context = _assemble(repo, [])
    html, text = build_html_report([], STARTED, 1.0, context, compact=compact)

    assert "🚨 ALERTY SUBSKRYPCJI / LIMITÓW" not in html
    assert "🚨 ALERTY SUBSKRYPCJI / LIMITÓW" not in text
    assert HISTORY_TITLE in html and HISTORY_TITLE in text
    assert old.message in html and old.message in text
    assert "Status tych zdarzeń wymaga osobnej weryfikacji." in html
    assert "Status tych zdarzeń wymaga osobnej weryfikacji." in text
    assert context.quota_alerts == []
    assert context.quota_alert_history == [old]
    assert build_cycle_lead([], context) == []
    repo.get_recent_quota_alerts.assert_called_once_with(hours=171)


def test_historical_critical_alert_does_not_become_the_subject() -> None:
    from main_agent import _build_subject

    old = _alert(hours_ago=24)
    context = _assemble(_repo([old]), [])

    subject = _build_subject(
        lead_items=build_cycle_lead([], context),
        analyzed=0,
        failures=0,
        started_at=STARTED,
        prefix="",
    )

    assert old.source not in subject
    assert old.message not in subject
    assert "0 predykcji, 0 błędów" in subject


def test_repeated_current_alert_is_not_hidden_by_its_older_occurrence() -> None:
    old = _alert(hours_ago=24)
    current = _alert(hours_ago=0)

    context = _assemble(_repo([old, current]), [current])

    assert context.quota_alerts == [current]
    assert context.quota_alert_history == [old]
    html, text = build_html_report([], STARTED, 1.0, context)
    assert CURRENT_TITLE in html and CURRENT_TITLE in text
    assert HISTORY_TITLE in html and HISTORY_TITLE in text
    current_text, history_text = text.split(HISTORY_TITLE)
    assert format_local(current.occurred_at, context.display_timezone) in current_text
    assert format_local(old.occurred_at, context.display_timezone) in history_text
    assert (
        f"ostatnio {format_local(current.occurred_at, context.display_timezone)}"
        not in history_text
    )
    assert old.message in history_text
    assert current.message in build_cycle_lead([], context)[0].headline


@pytest.mark.parametrize("failed_operation", [None, "save_quota_alert", "get_recent_quota_alerts"])
def test_current_alert_survives_a_missing_persisted_copy(failed_operation: str | None) -> None:
    current = _alert(hours_ago=0)
    repo = _repo([])
    if failed_operation is not None:
        getattr(repo, failed_operation).side_effect = ProviderUnavailable("Baza niedostępna")

    context = _assemble(repo, [current])
    html, text = build_html_report([], STARTED, 1.0, context)

    assert CURRENT_TITLE in html and CURRENT_TITLE in text
    assert context.quota_alerts == [current]
    assert context.quota_alert_history == []
    assert HISTORY_TITLE not in html and HISTORY_TITLE not in text


def test_history_groups_repeated_events_using_latest_time_and_count() -> None:
    older = _alert(hours_ago=48)
    newer = _alert(hours_ago=24)
    context = _assemble(_repo([older, newer]), [])

    html, text = build_html_report([], STARTED, 1.0, context)

    assert HISTORY_TITLE in html and HISTORY_TITLE in text
    for rendered in (html, text):
        assert rendered.count(older.message) == 1
        assert "Liczba zdarzeń: 2" in rendered
        assert format_local(newer.occurred_at, context.display_timezone) in rendered
        assert format_local(older.occurred_at, context.display_timezone) not in rendered
        assert "Zalecenie zapisane przy zdarzeniu" in rendered


def test_historical_critical_does_not_raise_a_current_warning_to_critical() -> None:
    old = _alert(hours_ago=24)
    warning = replace(_alert(hours_ago=0), severity=QuotaSeverity.WARNING)
    context = _assemble(_repo([old, warning]), [warning])

    assert build_cycle_lead([], context) == []
    _, text = build_html_report([], STARTED, 1.0, context)
    current_text, history_text = text.split(HISTORY_TITLE)
    assert "OSTRZEŻENIE" in current_text
    assert "KRYTYCZNY" not in current_text
    assert "KRYTYCZNY" in history_text


def test_recomputed_accuracy_alarm_remains_current_without_reappearing_in_history() -> None:
    old = _alert(hours_ago=24)
    drop = AccuracyDrop(
        since="2026-10-20", alarm_cycle="2026-10-29", before_rate=0.8, after_rate=0.4, cycles=12
    )
    with patch(
        "src.application.use_cases.assemble_report.build_track_record_stats",
        return_value=TrackRecordStats(accuracy_drop=drop),
    ):
        context = _assemble(_repo([old]), [])

    assert context.quota_alerts is not None
    assert len(context.quota_alerts) == 1
    assert context.quota_alerts[0].source == "Detektor trafności"
    assert context.quota_alert_history == [old]
