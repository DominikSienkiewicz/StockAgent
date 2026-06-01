"""Testy QuotaMonitor — kolektor alertów współdzielony przez adaptery."""

from __future__ import annotations

from datetime import UTC, datetime

from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity


def _alert(source: str = "OpenAI", severity: QuotaSeverity = QuotaSeverity.WARNING) -> QuotaAlert:
    return QuotaAlert(
        source=source,
        severity=severity,
        message="msg",
        action="act",
        occurred_at=datetime(2026, 6, 1, tzinfo=UTC),
    )


class TestRecord:
    def test_empty_monitor_has_no_alerts(self) -> None:
        assert QuotaMonitor().alerts == []

    def test_records_single_alert(self) -> None:
        m = QuotaMonitor()
        a = _alert()
        m.record(a)
        assert m.alerts == [a]

    def test_records_multiple_alerts_preserves_order(self) -> None:
        m = QuotaMonitor()
        a1 = _alert(source="A")
        a2 = _alert(source="B")
        a3 = _alert(source="C")
        m.record(a1)
        m.record(a2)
        m.record(a3)
        assert [a.source for a in m.alerts] == ["A", "B", "C"]


class TestSeverity:
    def test_no_alerts_means_no_max_severity(self) -> None:
        assert QuotaMonitor().max_severity() is None

    def test_max_severity_returns_critical_when_present(self) -> None:
        m = QuotaMonitor()
        m.record(_alert(severity=QuotaSeverity.WARNING))
        m.record(_alert(severity=QuotaSeverity.CRITICAL))
        m.record(_alert(severity=QuotaSeverity.WARNING))
        assert m.max_severity() is QuotaSeverity.CRITICAL


class TestAlertsView:
    def test_alerts_returns_copy_not_internal_list(self) -> None:
        m = QuotaMonitor()
        m.record(_alert())
        snapshot = m.alerts
        snapshot.clear()
        # Wewnętrzny stan musi być nieruszony.
        assert len(m.alerts) == 1
