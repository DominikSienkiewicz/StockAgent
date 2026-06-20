"""TelegramNotifier — push raportu przez Telegram Bot API (`sendMessage`).

Kanał nie renderuje HTML maila, więc konsumujemy `plain_text` (subject + treść).
W odróżnieniu od `ResendNotifier` adapter NIE rzuca z `send_report` — złamany
kanał push nie może zabić cyklu. Błąd 4xx/5xx/sieci → CRITICAL `QuotaAlert`.
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

DEFAULT_BASE_URL = "https://api.telegram.org"
DEFAULT_TIMEOUT = 15


class TelegramNotifier(ReportNotifierPort):
    """Adapter dla Telegram Bot API.

    `bot_token` i `chat_id` są wstrzykiwane (DI w orkiestratorze z sekretów).
    Endpoint: `https://api.telegram.org/bot<token>/sendMessage`.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        quota_monitor: QuotaMonitor | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = build_session()
        self._quota_monitor = quota_monitor

    def send_report(self, subject: str, html_body: str, plain_text: str) -> None:
        # Kanał push konsumuje plain_text — HTML maila jest tu nieprzydatny.
        text = f"{subject}\n\n{plain_text}"
        url = f"{self._base_url}/bot{self._bot_token}/sendMessage"
        try:
            response = self._session.post(
                url,
                json={"chat_id": self._chat_id, "text": text},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            # Błąd sieci (timeout, DNS, connection reset) — kanał padł.
            self._emit_quota_alert(
                message=f"Telegram push failed (network): {exc}.",
                action=(
                    "Verify TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID and "
                    "outbound connectivity to api.telegram.org."
                ),
            )
            return

        if response.status_code >= 400:
            self._emit_quota_alert(
                message=(
                    f"Telegram returned {response.status_code} — push not "
                    "delivered (bad token / chat_id / bot blocked)."
                ),
                action=(
                    "Verify TELEGRAM_BOT_TOKEN is valid and the bot has been "
                    "started in the target chat (TELEGRAM_CHAT_ID)."
                ),
            )
            return

        logger.info("Telegram: sent report to chat_id=%s", self._chat_id)

    def _emit_quota_alert(self, message: str, action: str) -> None:
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source="Telegram",
                severity=QuotaSeverity.CRITICAL,
                message=message,
                action=action,
                occurred_at=datetime.now(UTC),
            )
        )
