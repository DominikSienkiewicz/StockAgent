"""Testy QuotaMonitor — kolektor alertów współdzielony przez adaptery."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock

from src.application.ports import AlertNotifierPort
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


class TestRealtimeAlertHook:
    """U4 — opcjonalny real-time push CRITICAL alertów (debounce per source)."""

    def test_no_notifier_is_noop_default_behavior(self) -> None:
        # Brak notifiera = dzisiejsze zachowanie: record po prostu zbiera.
        m = QuotaMonitor()
        m.record(_alert(severity=QuotaSeverity.CRITICAL))
        assert len(m.alerts) == 1

    def test_critical_fires_send_alert_once(self) -> None:
        notifier = Mock(spec=AlertNotifierPort)
        m = QuotaMonitor(alert_notifier=notifier)

        alert = _alert(source="OpenAI", severity=QuotaSeverity.CRITICAL)
        m.record(alert)

        notifier.send_alert.assert_called_once()
        # Przekazany alert to ten właśnie zapisany.
        sent = notifier.send_alert.call_args.args[0]
        assert sent == [alert]

    def test_second_critical_same_source_does_not_refire(self) -> None:
        notifier = Mock(spec=AlertNotifierPort)
        m = QuotaMonitor(alert_notifier=notifier)

        m.record(_alert(source="OpenAI", severity=QuotaSeverity.CRITICAL))
        m.record(_alert(source="OpenAI", severity=QuotaSeverity.CRITICAL))

        # Debounce per źródło na czas życia monitora.
        notifier.send_alert.assert_called_once()

    def test_different_source_fires_separately(self) -> None:
        notifier = Mock(spec=AlertNotifierPort)
        m = QuotaMonitor(alert_notifier=notifier)

        m.record(_alert(source="OpenAI", severity=QuotaSeverity.CRITICAL))
        m.record(_alert(source="Resend", severity=QuotaSeverity.CRITICAL))

        assert notifier.send_alert.call_count == 2

    def test_warning_never_fires(self) -> None:
        notifier = Mock(spec=AlertNotifierPort)
        m = QuotaMonitor(alert_notifier=notifier)

        m.record(_alert(severity=QuotaSeverity.WARNING))

        notifier.send_alert.assert_not_called()

    def test_notifier_exception_is_swallowed(self) -> None:
        notifier = Mock(spec=AlertNotifierPort)
        notifier.send_alert.side_effect = RuntimeError("push transport down")
        m = QuotaMonitor(alert_notifier=notifier)

        # Padający push nie ma prawa wywalić record() ani cyklu.
        m.record(_alert(severity=QuotaSeverity.CRITICAL))

        # Alert i tak został zebrany do raportu.
        assert len(m.alerts) == 1

    def test_failed_push_still_debounces_source(self) -> None:
        notifier = Mock(spec=AlertNotifierPort)
        notifier.send_alert.side_effect = RuntimeError("down")
        m = QuotaMonitor(alert_notifier=notifier)

        m.record(_alert(source="OpenAI", severity=QuotaSeverity.CRITICAL))
        m.record(_alert(source="OpenAI", severity=QuotaSeverity.CRITICAL))

        # Nawet po porażce nie spamujemy ponownie tego samego źródła.
        assert notifier.send_alert.call_count == 1
