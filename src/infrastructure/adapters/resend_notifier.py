"""Adaptery dla `ReportNotifierPort`:
- `ResendNotifier` — Resend.com API (free tier 100 mails/dobę)
- `NullNotifier` — fallback gdy notyfikacje wyłączone (no-op + log)
"""

from __future__ import annotations

import logging

import requests

from src.application.ports import ReportNotifierPort

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.resend.com"
DEFAULT_TIMEOUT = 15


class ResendNotifier(ReportNotifierPort):
    """Adapter dla Resend API.

    Sandbox sender `onboarding@resend.dev` działa od razu bez weryfikacji domeny —
    może wysyłać tylko na adres zarejestrowany w Resend. Po weryfikacji własnej
    domeny można podać dowolnego from/to.
    """

    def __init__(
        self,
        api_key: str,
        sender: str,
        recipient: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._sender = sender
        self._recipient = recipient
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def send_report(self, subject: str, html_body: str, plain_text: str) -> None:
        response = requests.post(
            f"{self._base_url}/emails",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": self._sender,
                "to": [self._recipient],
                "subject": subject,
                "html": html_body,
                "text": plain_text,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        message_id = response.json().get("id", "<no-id>")
        logger.info(
            "Resend: sent report to %s (id=%s)", self._recipient, message_id
        )


class NullNotifier(ReportNotifierPort):
    """No-op fallback gdy `NOTIFICATIONS_ENABLED=false` lub brak kluczy.

    Loguje raport na poziomie INFO — żadna sieć nie jest dotykana.
    Pozwala main_agent.py mieć zawsze `notifier`, niezależnie od konfiguracji.
    """

    def send_report(self, subject: str, html_body: str, plain_text: str) -> None:
        logger.info("Notifications disabled — skipping email. Subject: %s", subject)
        logger.debug("Report (plain text):\n%s", plain_text)
