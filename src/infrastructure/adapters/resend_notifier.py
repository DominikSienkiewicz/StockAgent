"""Adaptery dla `ReportNotifierPort`:
- `ResendNotifier` — Resend.com API (free tier 100 mails/dobę)
- `NullNotifier` — fallback gdy notyfikacje wyłączone (no-op + log)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import requests

from src.application.ports import ReportNotifierPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.infrastructure.adapters._http import build_session

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
        quota_monitor: QuotaMonitor | None = None,
    ) -> None:
        self._api_key = api_key
        self._sender = sender
        self._recipient = recipient
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = build_session()
        self._quota_monitor = quota_monitor

    def send_report(self, subject: str, html_body: str, plain_text: str) -> None:
        response = self._session.post(
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
        # Resend zwraca 429 gdy free tier (100 mails/dobę) wyczerpany.
        # Alert trafi do bazy i pojawi się w NASTĘPNYM mailu (ten paradoksalnie
        # nie poszedł), gdy quota się zresetuje.
        if response.status_code == 429:
            self._emit_quota_alert(
                severity=QuotaSeverity.CRITICAL,
                message=(
                    "Resend returned 429 — free tier (100 mails/day) "
                    "exhausted. This report did not get delivered."
                ),
                action=(
                    "Wait 24h for daily reset, upgrade Resend plan, or "
                    "throttle agent runs (GHA cron less frequently)."
                ),
            )
        elif 400 <= response.status_code < 500:
            self._emit_quota_alert(
                severity=QuotaSeverity.CRITICAL,
                message=(
                    f"Resend returned {response.status_code} — email not "
                    "delivered (auth / domain / quota issue)."
                ),
                action=(
                    "Verify RESEND_API_KEY is valid and DIGEST_FROM_EMAIL "
                    "is verified in your Resend domain settings."
                ),
            )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise
        message_id = response.json().get("id", "<no-id>")
        logger.info(
            "Resend: sent report to %s (id=%s)", self._recipient, message_id
        )

    def _emit_quota_alert(
        self, severity: QuotaSeverity, message: str, action: str
    ) -> None:
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source="Resend",
                severity=severity,
                message=message,
                action=action,
                occurred_at=datetime.now(UTC),
            )
        )


class NullNotifier(ReportNotifierPort):
    """No-op fallback gdy `NOTIFICATIONS_ENABLED=false` lub brak kluczy.

    Loguje raport na poziomie INFO — żadna sieć nie jest dotykana.
    Pozwala main_agent.py mieć zawsze `notifier`, niezależnie od konfiguracji.
    """

    def send_report(self, subject: str, html_body: str, plain_text: str) -> None:
        logger.info("Notifications disabled — skipping email. Subject: %s", subject)
        logger.debug("Report (plain text):\n%s", plain_text)
