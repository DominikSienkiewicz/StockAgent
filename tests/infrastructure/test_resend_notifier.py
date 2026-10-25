import logging
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
import requests

from src.application.ports import ReportNotifierPort
from src.infrastructure.adapters.resend_notifier import NullNotifier, ResendNotifier


@pytest.fixture
def notifier() -> ResendNotifier:
    return ResendNotifier(
        api_key="re-test-key",
        sender="agent@example.com",
        recipient="dom@example.com",
    )


def _ok_response(payload: dict[str, Any] | None = None) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload or {"id": "msg-uuid-123"}
    response.raise_for_status = MagicMock()
    return response


class TestResendNotifier:
    def test_implements_port(self) -> None:
        assert issubclass(ResendNotifier, ReportNotifierPort)

    def test_posts_to_emails_endpoint(self, notifier: ResendNotifier, mocker: Any) -> None:
        mock_post = mocker.patch(
            "requests.Session.post",
            return_value=_ok_response(),
        )

        notifier.send_report("Subject", "<p>HTML</p>", "plain text")

        url = mock_post.call_args.args[0]
        assert url.endswith("/emails")

    def test_passes_bearer_token_in_authorization(
        self, notifier: ResendNotifier, mocker: Any
    ) -> None:
        mock_post = mocker.patch(
            "requests.Session.post",
            return_value=_ok_response(),
        )

        notifier.send_report("S", "<p>H</p>", "T")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer re-test-key"

    def test_payload_contains_from_to_subject_html_text(
        self, notifier: ResendNotifier, mocker: Any
    ) -> None:
        mock_post = mocker.patch(
            "requests.Session.post",
            return_value=_ok_response(),
        )

        notifier.send_report("My subject", "<p>HTML body</p>", "Plain body")

        body = mock_post.call_args.kwargs["json"]
        assert body["from"] == "agent@example.com"
        assert body["to"] == ["dom@example.com"]
        assert body["subject"] == "My subject"
        assert body["html"] == "<p>HTML body</p>"
        assert body["text"] == "Plain body"

    def test_sends_idempotency_key_header(self, notifier: ResendNotifier, mocker: Any) -> None:
        # MAIL-03: Resend ma retry POST po 5xx — bez klucza idempotencji to dwa maile.
        mock_post = mocker.patch("requests.Session.post", return_value=_ok_response())

        notifier.send_report("s", "<p>h</p>", "t")
        notifier.send_report("s", "<p>h</p>", "t")

        keys = [c.kwargs["headers"]["Idempotency-Key"] for c in mock_post.call_args_list]
        assert all(keys) and keys[0] != keys[1]

    def test_logs_message_id_on_success(
        self, notifier: ResendNotifier, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO)
        mocker.patch(
            "requests.Session.post",
            return_value=_ok_response({"id": "abc-123"}),
        )

        notifier.send_report("S", "H", "T")

        assert "abc-123" in caplog.text
        assert "d***@example.com" in caplog.text

    def test_success_log_masks_recipient(
        self, notifier: ResendNotifier, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        # BEZ-11: repo i logi GHA są publiczne — pełny adres nie trafia do logu.
        caplog.set_level(logging.DEBUG)
        mocker.patch(
            "requests.Session.post",
            return_value=_ok_response({"id": "abc-123"}),
        )

        notifier.send_report("S", "H", "T")

        assert "dom@example.com" not in caplog.text

    def test_raises_on_http_error(self, notifier: ResendNotifier, mocker: Any) -> None:
        response = MagicMock(spec=requests.Response)
        response.status_code = 401
        response.raise_for_status.side_effect = requests.HTTPError("401")
        mocker.patch(
            "requests.Session.post",
            return_value=response,
        )

        with pytest.raises(requests.HTTPError):
            notifier.send_report("S", "H", "T")


class TestNullNotifier:
    def test_implements_port(self) -> None:
        assert issubclass(NullNotifier, ReportNotifierPort)

    def test_does_nothing_no_network(self, mocker: Any) -> None:
        # Jeśli ktoś przypadkiem zaimportuje requests w Null, to przykryjemy:
        post = mocker.patch("requests.post")
        NullNotifier().send_report("S", "H", "T")
        post.assert_not_called()

    def test_logs_at_info_level(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        NullNotifier().send_report("Mój subject", "html", "text")
        assert "Mój subject" in caplog.text
        assert "skip" in caplog.text.lower()


def test_silent_mail_is_the_same_mail(notifier: ResendNotifier, mocker: Any) -> None:
    # ALR-09: mail nie ma dźwięku do wyciszenia — `silent` niczego w nim nie zmienia.
    mock_post = mocker.patch("requests.Session.post", return_value=_ok_response())

    notifier.send_report("S", "<p>H</p>", "T", silent=True)
    silent_payload = mock_post.call_args.kwargs["json"]
    notifier.send_report("S", "<p>H</p>", "T")

    assert mock_post.call_args.kwargs["json"] == silent_payload


def test_null_notifier_accepts_silent(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    NullNotifier().send_report("Cichy subject", "html", "text", silent=True)

    assert "Cichy subject" in caplog.text


@pytest.mark.integration
class TestResendLive:
    def test_sends_real_email(self) -> None:
        import os

        api_key = os.environ.get("RESEND_API_KEY")
        email_to = os.environ.get("DIGEST_TO_EMAIL") or os.environ.get("EMAIL_TO")
        if not api_key or not email_to:
            pytest.skip("RESEND_API_KEY or DIGEST_TO_EMAIL not set")

        sender = (
            os.environ.get("DIGEST_FROM_EMAIL")
            or os.environ.get("EMAIL_FROM")
            or "onboarding@resend.dev"
        )
        notifier = ResendNotifier(api_key=api_key, sender=sender, recipient=email_to)
        notifier.send_report(
            subject="StockAgent — test mail (pytest)",
            html_body="<h1>Hello</h1><p>This is a smoke test.</p>",
            plain_text="Hello — this is a smoke test.",
        )


class TestQuotaAlertEmit:
    def _resp(self, status_code: int, body: dict[str, Any] | None = None) -> Mock:
        import requests

        r = Mock(spec=requests.Response)
        r.status_code = status_code
        r.json.return_value = body or {}
        if status_code >= 400:
            r.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
        else:
            r.raise_for_status = Mock()
        return r

    def test_emits_critical_on_429(self, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity
        from src.infrastructure.adapters.resend_notifier import ResendNotifier

        monitor = QuotaMonitor()
        notifier = ResendNotifier(
            api_key="re_x",
            sender="a@x",
            recipient="b@x",
            quota_monitor=monitor,
        )
        mocker.patch("requests.Session.post", return_value=self._resp(429))

        with pytest.raises(requests.HTTPError):
            notifier.send_report("subj", "<html>", "text")

        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].severity is QuotaSeverity.CRITICAL
        assert monitor.alerts[0].source == "Resend"
        assert "100" in monitor.alerts[0].message  # mails/day

    def test_emits_critical_on_4xx_other(self, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.infrastructure.adapters.resend_notifier import ResendNotifier

        monitor = QuotaMonitor()
        notifier = ResendNotifier(
            api_key="re_x",
            sender="a@x",
            recipient="b@x",
            quota_monitor=monitor,
        )
        mocker.patch("requests.Session.post", return_value=self._resp(401))

        with pytest.raises(requests.HTTPError):
            notifier.send_report("subj", "<html>", "text")

        assert len(monitor.alerts) == 1
        assert "401" in monitor.alerts[0].message

    def test_server_error_emits_critical_alert(self, mocker: Any) -> None:
        # NIEZ-04: docs obiecują, że zablokowany mail nie zniknie w logach — 5xx też.
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity
        from src.infrastructure.adapters.resend_notifier import ResendNotifier

        monitor = QuotaMonitor()
        notifier = ResendNotifier(
            api_key="re_x",
            sender="a@x",
            recipient="b@x",
            quota_monitor=monitor,
        )
        mocker.patch("requests.Session.post", return_value=self._resp(503))

        with pytest.raises(requests.HTTPError):
            notifier.send_report("subj", "<html>", "text")

        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].severity is QuotaSeverity.CRITICAL
        assert "503" in monitor.alerts[0].message

    def test_network_error_emits_critical_alert(self, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.infrastructure.adapters.resend_notifier import ResendNotifier

        monitor = QuotaMonitor()
        notifier = ResendNotifier(
            api_key="re_x",
            sender="a@x",
            recipient="b@x",
            quota_monitor=monitor,
        )
        mocker.patch("requests.Session.post", side_effect=requests.ConnectionError("timeout"))

        with pytest.raises(requests.ConnectionError):
            notifier.send_report("subj", "<html>", "text")

        assert len(monitor.alerts) == 1
        assert "ConnectionError" in monitor.alerts[0].message

    def test_no_alert_on_200(self, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.infrastructure.adapters.resend_notifier import ResendNotifier

        monitor = QuotaMonitor()
        notifier = ResendNotifier(
            api_key="re_x",
            sender="a@x",
            recipient="b@x",
            quota_monitor=monitor,
        )
        mocker.patch(
            "requests.Session.post",
            return_value=self._resp(200, {"id": "msg-123"}),
        )

        notifier.send_report("subj", "<html>", "text")

        assert monitor.alerts == []


class TestPolishAlertText:
    """MAIL-15: treść alertu trafia do bannera, leadu i tematu maila — po
    polsku. Angielska diagnoza zostaje w logach."""

    @staticmethod
    def _failing(status_code: int) -> Mock:
        response = Mock(spec=requests.Response)
        response.status_code = status_code
        response.raise_for_status.side_effect = requests.HTTPError(str(status_code))
        return response

    def test_rate_limit_alert_is_polish_and_logged_in_english(
        self, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        from src.application.quota_monitor import QuotaMonitor

        monitor = QuotaMonitor()
        notifier = ResendNotifier(
            api_key="re_x", sender="a@x", recipient="b@x", quota_monitor=monitor
        )
        mocker.patch("requests.Session.post", return_value=self._failing(429))

        with caplog.at_level(logging.WARNING), pytest.raises(requests.HTTPError):
            notifier.send_report("subj", "<html>", "text")

        alert = monitor.alerts[0]
        assert "nie został dostarczony" in alert.message
        assert "exhausted" not in alert.message
        assert alert.action.startswith("Poczekaj")
        assert "Resend returned 429" in caplog.text

    def test_network_alert_is_polish_and_logged_in_english(
        self, mocker: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        from src.application.quota_monitor import QuotaMonitor

        monitor = QuotaMonitor()
        notifier = ResendNotifier(
            api_key="re_x", sender="a@x", recipient="b@x", quota_monitor=monitor
        )
        mocker.patch("requests.Session.post", side_effect=requests.ConnectionError("x"))

        with caplog.at_level(logging.WARNING), pytest.raises(requests.ConnectionError):
            notifier.send_report("subj", "<html>", "text")

        assert "nieosiągalny" in monitor.alerts[0].message
        assert "Resend unreachable (ConnectionError)" in caplog.text
