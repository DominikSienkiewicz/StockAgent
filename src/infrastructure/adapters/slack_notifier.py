"""SlackNotifier — push raportu przez Slack incoming webhook.

Webhook URL jest wstrzykiwany (DI w orkiestratorze z sekretu). Kanał nie renderuje
HTML maila, więc konsumujemy `plain_text` (subject + treść). Adapter NIE rzuca
z `send_report` — złamany kanał push nie może zabić cyklu. Błąd 4xx/5xx/sieci →
CRITICAL `QuotaAlert`.
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

DEFAULT_TIMEOUT = 15


class SlackNotifier(ReportNotifierPort):
    """Adapter dla Slack incoming webhook.

    POST JSON `{"text": ...}` pod wstrzyknięty `webhook_url`. Slack zwraca `ok`
    (200) przy sukcesie i `4xx` z tekstem błędu (np. `no_service`) przy złym URL.
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: int = DEFAULT_TIMEOUT,
        quota_monitor: QuotaMonitor | None = None,
    ) -> None:
        self._webhook_url = webhook_url
        self._timeout = timeout
        self._session = build_session(retry_post=False)
        self._quota_monitor = quota_monitor

    def send_report(
        self, subject: str, html_body: str, plain_text: str, *, silent: bool = False
    ) -> None:
        # ALR-09: incoming webhook nie ma flagi cichej wiadomości; cisza to brak
        # @-wzmianki, a adapter żadnej nie dokleja — `silent` niczego tu nie zmienia.
        # Kanał push konsumuje plain_text — HTML maila jest tu nieprzydatny.
        text = f"{subject}\n\n{plain_text}"
        try:
            response = self._session.post(
                self._webhook_url,
                json={"text": text},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            self._emit_quota_alert(
                # Treść wyjątku niesie URL webhooka — do maila i bazy idzie typ.
                message=f"Push Slack nie wyszedł (błąd sieci: {type(exc).__name__}).",
                action=("Sprawdź SLACK_WEBHOOK_URL i połączenie wychodzące do hooks.slack.com."),
                # Tylko nazwa wyjątku: jego treść niesie URL z sekretem
                # (token / webhook), a logi repo są publiczne.
                log_message=f"Slack push failed (network): {type(exc).__name__}.",
            )
            return

        if response.status_code >= 400:
            self._emit_quota_alert(
                message=(
                    f"Slack zwrócił {response.status_code} — push nie został "
                    "dostarczony (unieważniony albo błędny webhook)."
                ),
                action=("Wygeneruj nowy incoming webhook Slacka i zaktualizuj SLACK_WEBHOOK_URL."),
                log_message=(
                    f"Slack returned {response.status_code} — push not "
                    "delivered (revoked / malformed webhook)."
                ),
            )
            return

        logger.info("Slack: sent report via incoming webhook")

    def _emit_quota_alert(self, message: str, action: str, log_message: str) -> None:
        # MAIL-15: `message`/`action` idą do maila (po polsku), `log_message`
        # zostaje angielską diagnozą w logach.
        logger.error("%s", log_message)
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source="Slack",
                severity=QuotaSeverity.CRITICAL,
                message=message,
                action=action,
                occurred_at=datetime.now(UTC),
            )
        )
