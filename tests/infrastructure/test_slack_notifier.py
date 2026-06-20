"""Testy SlackNotifier — push raportu przez Slack incoming webhook."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest
import requests

from src.application.ports import ReportNotifierPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaSeverity
from src.infrastructure.adapters.slack_notifier import SlackNotifier

WEBHOOK = "https://hooks.slack.com/services/T000/B000/XXXX"


def _resp(status_code: int, text: str = "ok") -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.text = text
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def notifier() -> SlackNotifier:
    return SlackNotifier(webhook_url=WEBHOOK)


class TestSlackNotifier:
    def test_implements_port(self) -> None:
        assert issubclass(SlackNotifier, ReportNotifierPort)

    def test_posts_to_webhook_url(self, notifier: SlackNotifier, mocker) -> None:
        mock_post = mocker.patch("requests.Session.post", return_value=_resp(200))

        notifier.send_report("Subject", "<p>HTML</p>", "plain text body")

        url = mock_post.call_args.args[0]
        assert url == WEBHOOK

    def test_sends_plain_text_with_subject(self, notifier: SlackNotifier, mocker) -> None:
        mock_post = mocker.patch("requests.Session.post", return_value=_resp(200))

        notifier.send_report("My subject", "<p>html</p>", "Plain body content")

        body = mock_post.call_args.kwargs["json"]
        assert "Plain body content" in body["text"]
        assert "My subject" in body["text"]

    def test_does_not_use_html_body(self, notifier: SlackNotifier, mocker) -> None:
        mock_post = mocker.patch("requests.Session.post", return_value=_resp(200))

        notifier.send_report("S", "<p>UNIQUE_HTML_MARKER</p>", "plain")

        body = mock_post.call_args.kwargs["json"]
        assert "UNIQUE_HTML_MARKER" not in body["text"]

    def test_emits_critical_on_4xx_and_does_not_raise(self, mocker) -> None:
        monitor = QuotaMonitor()
        notifier = SlackNotifier(webhook_url=WEBHOOK, quota_monitor=monitor)
        mocker.patch("requests.Session.post", return_value=_resp(404, "no_service"))

        notifier.send_report("S", "<p>H</p>", "T")

        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].severity is QuotaSeverity.CRITICAL
        assert monitor.alerts[0].source == "Slack"

    def test_emits_critical_on_network_error_and_does_not_raise(self, mocker) -> None:
        monitor = QuotaMonitor()
        notifier = SlackNotifier(webhook_url=WEBHOOK, quota_monitor=monitor)
        mocker.patch(
            "requests.Session.post",
            side_effect=requests.Timeout("slow"),
        )

        notifier.send_report("S", "<p>H</p>", "T")

        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].severity is QuotaSeverity.CRITICAL
        assert monitor.alerts[0].source == "Slack"

    def test_no_alert_on_200(self, mocker) -> None:
        monitor = QuotaMonitor()
        notifier = SlackNotifier(webhook_url=WEBHOOK, quota_monitor=monitor)
        mocker.patch("requests.Session.post", return_value=_resp(200))

        notifier.send_report("S", "<p>H</p>", "T")

        assert monitor.alerts == []

    def test_does_not_raise_without_monitor_on_failure(self, mocker) -> None:
        notifier = SlackNotifier(webhook_url=WEBHOOK)
        mocker.patch("requests.Session.post", return_value=_resp(500))

        notifier.send_report("S", "<p>H</p>", "T")

    def test_uses_quota_monitor_spec_mock(self, mocker) -> None:
        monitor = Mock(spec=QuotaMonitor)
        notifier = SlackNotifier(webhook_url=WEBHOOK, quota_monitor=monitor)
        mocker.patch("requests.Session.post", return_value=_resp(404))

        notifier.send_report("S", "<p>H</p>", "T")

        assert monitor.record.call_count == 1
