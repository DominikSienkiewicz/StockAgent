import logging
from unittest.mock import MagicMock

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


def _ok_response(payload: dict | None = None) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload or {"id": "msg-uuid-123"}
    response.raise_for_status = MagicMock()
    return response


class TestResendNotifier:
    def test_implements_port(self):
        assert issubclass(ResendNotifier, ReportNotifierPort)

    def test_posts_to_emails_endpoint(self, notifier, mocker):
        mock_post = mocker.patch(
            "src.infrastructure.adapters.resend_notifier.requests.post",
            return_value=_ok_response(),
        )

        notifier.send_report("Subject", "<p>HTML</p>", "plain text")

        url = mock_post.call_args.args[0]
        assert url.endswith("/emails")

    def test_passes_bearer_token_in_authorization(self, notifier, mocker):
        mock_post = mocker.patch(
            "src.infrastructure.adapters.resend_notifier.requests.post",
            return_value=_ok_response(),
        )

        notifier.send_report("S", "<p>H</p>", "T")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer re-test-key"

    def test_payload_contains_from_to_subject_html_text(self, notifier, mocker):
        mock_post = mocker.patch(
            "src.infrastructure.adapters.resend_notifier.requests.post",
            return_value=_ok_response(),
        )

        notifier.send_report("My subject", "<p>HTML body</p>", "Plain body")

        body = mock_post.call_args.kwargs["json"]
        assert body["from"] == "agent@example.com"
        assert body["to"] == ["dom@example.com"]
        assert body["subject"] == "My subject"
        assert body["html"] == "<p>HTML body</p>"
        assert body["text"] == "Plain body"

    def test_logs_message_id_on_success(self, notifier, mocker, caplog):
        caplog.set_level(logging.INFO)
        mocker.patch(
            "src.infrastructure.adapters.resend_notifier.requests.post",
            return_value=_ok_response({"id": "abc-123"}),
        )

        notifier.send_report("S", "H", "T")

        assert "abc-123" in caplog.text
        assert "dom@example.com" in caplog.text

    def test_raises_on_http_error(self, notifier, mocker):
        response = MagicMock(spec=requests.Response)
        response.raise_for_status.side_effect = requests.HTTPError("401")
        mocker.patch(
            "src.infrastructure.adapters.resend_notifier.requests.post",
            return_value=response,
        )

        with pytest.raises(requests.HTTPError):
            notifier.send_report("S", "H", "T")


class TestNullNotifier:
    def test_implements_port(self):
        assert issubclass(NullNotifier, ReportNotifierPort)

    def test_does_nothing_no_network(self, mocker):
        # Jeśli ktoś przypadkiem zaimportuje requests w Null, to przykryjemy:
        post = mocker.patch("requests.post")
        NullNotifier().send_report("S", "H", "T")
        post.assert_not_called()

    def test_logs_at_info_level(self, caplog):
        caplog.set_level(logging.INFO)
        NullNotifier().send_report("Mój subject", "html", "text")
        assert "Mój subject" in caplog.text
        assert "skip" in caplog.text.lower()


@pytest.mark.integration
class TestResendLive:
    def test_sends_real_email(self):
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
