"""Adaptery dla `ReportNotifierPort`:
- `ResendNotifier` — Resend.com API (free tier 100 mails/dobę)
- `NullNotifier` — fallback gdy notyfikacje wyłączone (no-op + log)
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

import requests

from src.application.ports import ReportNotifierPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.subscriber import masked_email
from src.infrastructure.adapters._http import build_session, raise_for_status_redacted

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

    def send_report(
        self, subject: str, html_body: str, plain_text: str, *, silent: bool = False
    ) -> None:
        # ALR-09: mail nie ma dźwięku do wyciszenia — `silent` jest ignorowane.
        # Każdy niedostarczony mail kończy się alertem CRITICAL: trafi do bazy i
        # pojawi się w NASTĘPNYM mailu (ten paradoksalnie nie poszedł).
        try:
            response = self._session.post(
                f"{self._base_url}/emails",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    # Sesja powtarza POST po 5xx; klucz sprawia, że Resend wyśle mail raz.
                    "Idempotency-Key": str(uuid.uuid4()),
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
        except requests.RequestException as exc:
            self._emit_quota_alert(
                severity=QuotaSeverity.CRITICAL,
                message=(
                    f"Resend nieosiągalny ({type(exc).__name__}) — mail nie został dostarczony."
                ),
                action=("Sprawdź status Resend i sieć runnera; kolejny przebieg ponowi wysyłkę."),
                log_message=(f"Resend unreachable ({type(exc).__name__}) — email not delivered."),
            )
            raise
        # Resend zwraca 429 gdy free tier (100 mails/dobę) wyczerpany.
        if response.status_code == 429:
            self._emit_quota_alert(
                severity=QuotaSeverity.CRITICAL,
                message=(
                    "Resend zwrócił 429 — wyczerpany darmowy limit (100 maili/dobę). "
                    "Ten raport nie został dostarczony."
                ),
                action=(
                    "Poczekaj 24 h na dobowy reset, podnieś plan Resend albo "
                    "rzadziej uruchamiaj agenta (cron GHA)."
                ),
                log_message=(
                    "Resend returned 429 — free tier (100 mails/day) "
                    "exhausted. This report did not get delivered."
                ),
            )
        elif 400 <= response.status_code < 500:
            self._emit_quota_alert(
                severity=QuotaSeverity.CRITICAL,
                message=(
                    f"Resend zwrócił {response.status_code} — mail nie został "
                    "dostarczony (klucz, domena albo limit)."
                ),
                action=(
                    "Sprawdź, czy RESEND_API_KEY jest ważny, a DIGEST_FROM_EMAIL "
                    "zweryfikowany w ustawieniach domeny w Resend."
                ),
                log_message=(
                    f"Resend returned {response.status_code} — email not "
                    "delivered (auth / domain / quota issue)."
                ),
            )
        elif response.status_code >= 500:
            self._emit_quota_alert(
                severity=QuotaSeverity.CRITICAL,
                message=(
                    f"Resend zwrócił {response.status_code} — mail nie został "
                    "dostarczony (błąd dostawcy)."
                ),
                action="Sprawdź status.resend.com; kolejny przebieg ponowi wysyłkę.",
                log_message=(
                    f"Resend returned {response.status_code} — email not delivered "
                    "(provider error)."
                ),
            )
        raise_for_status_redacted(response)
        message_id = response.json().get("id", "<no-id>")
        # BEZ-11: repo i logi GHA są publiczne — adres odbiorcy tylko zamaskowany.
        logger.info("Resend: sent report to %s (id=%s)", masked_email(self._recipient), message_id)

    def _emit_quota_alert(
        self, severity: QuotaSeverity, message: str, action: str, log_message: str
    ) -> None:
        # MAIL-15: `message`/`action` idą do maila (po polsku), `log_message`
        # zostaje angielską diagnozą w logach.
        level = logging.ERROR if severity is QuotaSeverity.CRITICAL else logging.WARNING
        logger.log(level, "%s", log_message)
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

    def send_report(
        self, subject: str, html_body: str, plain_text: str, *, silent: bool = False
    ) -> None:
        logger.info("Notifications disabled — skipping email. Subject: %s", subject)
        logger.debug("Report (plain text):\n%s", plain_text)
