"""Testy DelegatingAlertNotifier — real-time push alertów CRITICAL.

Adapter formatuje listę `QuotaAlert` jako tekst i deleguje do wstrzykniętego
`ReportNotifierPort` (reuse transportu Telegram/Slack/Resend).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock

from src.application.ports import AlertNotifierPort, ReportNotifierPort
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.infrastructure.adapters.alert_notifier import DelegatingAlertNotifier


def _alert(
    source: str = "OpenAI",
    severity: QuotaSeverity = QuotaSeverity.CRITICAL,
    message: str = "limit hit",
    action: str = "do something",
) -> QuotaAlert:
    return QuotaAlert(
        source=source,
        severity=severity,
        message=message,
        action=action,
        occurred_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )


class TestDelegatingAlertNotifier:
    def test_implements_port(self) -> None:
        assert issubclass(DelegatingAlertNotifier, AlertNotifierPort)

    def test_delegates_to_report_notifier_send_report(self) -> None:
        transport = Mock(spec=ReportNotifierPort)
        notifier = DelegatingAlertNotifier(transport)

        notifier.send_alert([_alert()])

        transport.send_report.assert_called_once()

    def test_formats_alert_fields_into_plain_text(self) -> None:
        transport = Mock(spec=ReportNotifierPort)
        notifier = DelegatingAlertNotifier(transport)

        notifier.send_alert(
            [_alert(source="Resend", message="email failed", action="upgrade plan")]
        )

        kwargs = transport.send_report.call_args
        # plain_text jest trzecim argumentem pozycyjnym lub kwargiem.
        all_text = " ".join(str(x) for x in kwargs.args) + " " + " ".join(
            str(v) for v in kwargs.kwargs.values()
        )
        assert "Resend" in all_text
        assert "email failed" in all_text
        assert "upgrade plan" in all_text

    def test_empty_list_does_not_call_transport(self) -> None:
        transport = Mock(spec=ReportNotifierPort)
        notifier = DelegatingAlertNotifier(transport)

        notifier.send_alert([])

        transport.send_report.assert_not_called()

    def test_subject_signals_critical(self) -> None:
        transport = Mock(spec=ReportNotifierPort)
        notifier = DelegatingAlertNotifier(transport)

        notifier.send_alert([_alert()])

        subject = transport.send_report.call_args.kwargs.get("subject")
        if subject is None:
            subject = transport.send_report.call_args.args[0]
        assert "QUOTA" in subject.upper() or "ALERT" in subject.upper()
