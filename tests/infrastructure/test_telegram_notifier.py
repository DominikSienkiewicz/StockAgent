"""Testy TelegramNotifier — push raportu przez Telegram Bot API."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest
import requests

from src.application.ports import ReportNotifierPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaSeverity
from src.infrastructure.adapters.telegram_notifier import TelegramNotifier


def _resp(status_code: int, body: dict | None = None) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = body or {"ok": True}
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def notifier() -> TelegramNotifier:
    return TelegramNotifier(bot_token="123:ABC", chat_id="42")


class TestTelegramNotifier:
    def test_implements_port(self) -> None:
        assert issubclass(TelegramNotifier, ReportNotifierPort)

    def test_posts_to_send_message_endpoint(self, notifier: TelegramNotifier, mocker) -> None:
        mock_post = mocker.patch("requests.Session.post", return_value=_resp(200))

        notifier.send_report("Subject", "<p>HTML</p>", "plain text body")

        url = mock_post.call_args.args[0]
        assert url.endswith("/bot123:ABC/sendMessage")

    def test_sends_chat_id_and_plain_text(self, notifier: TelegramNotifier, mocker) -> None:
        mock_post = mocker.patch("requests.Session.post", return_value=_resp(200))

        notifier.send_report("My subject", "<p>html</p>", "Plain body content")

        body = mock_post.call_args.kwargs["json"]
        assert body["chat_id"] == "42"
        # Kanał nie renderuje HTML — konsumujemy plain_text (subject + treść).
        assert "Plain body content" in body["text"]
        assert "My subject" in body["text"]

    def test_does_not_use_html_body(self, notifier: TelegramNotifier, mocker) -> None:
        mock_post = mocker.patch("requests.Session.post", return_value=_resp(200))

        notifier.send_report("S", "<p>UNIQUE_HTML_MARKER</p>", "plain")

        body = mock_post.call_args.kwargs["json"]
        assert "UNIQUE_HTML_MARKER" not in body["text"]

    def test_emits_critical_on_4xx_and_does_not_raise(self, mocker) -> None:
        monitor = QuotaMonitor()
        notifier = TelegramNotifier(
            bot_token="123:ABC", chat_id="42", quota_monitor=monitor
        )
        mocker.patch("requests.Session.post", return_value=_resp(403))

        # Złamany kanał nie ma prawa wywalić cyklu.
        notifier.send_report("S", "<p>H</p>", "T")

        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].severity is QuotaSeverity.CRITICAL
        assert monitor.alerts[0].source == "Telegram"

    def test_emits_critical_on_network_error_and_does_not_raise(self, mocker) -> None:
        monitor = QuotaMonitor()
        notifier = TelegramNotifier(
            bot_token="123:ABC", chat_id="42", quota_monitor=monitor
        )
        mocker.patch(
            "requests.Session.post",
            side_effect=requests.ConnectionError("boom"),
        )

        notifier.send_report("S", "<p>H</p>", "T")

        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].severity is QuotaSeverity.CRITICAL
        assert monitor.alerts[0].source == "Telegram"

    def test_no_alert_on_200(self, mocker) -> None:
        monitor = QuotaMonitor()
        notifier = TelegramNotifier(
            bot_token="123:ABC", chat_id="42", quota_monitor=monitor
        )
        mocker.patch("requests.Session.post", return_value=_resp(200))

        notifier.send_report("S", "<p>H</p>", "T")

        assert monitor.alerts == []

    def test_does_not_raise_without_monitor_on_failure(self, mocker) -> None:
        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="42")
        mocker.patch("requests.Session.post", return_value=_resp(403))

        # Brak monitora + błąd kanału = nadal brak wyjątku.
        notifier.send_report("S", "<p>H</p>", "T")

    def test_uses_quota_monitor_spec_mock(self, mocker) -> None:
        monitor = Mock(spec=QuotaMonitor)
        notifier = TelegramNotifier(
            bot_token="123:ABC", chat_id="42", quota_monitor=monitor
        )
        mocker.patch("requests.Session.post", return_value=_resp(403))

        notifier.send_report("S", "<p>H</p>", "T")

        assert monitor.record.call_count == 1
