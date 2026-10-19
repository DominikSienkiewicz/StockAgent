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
        self._session = build_session(retry_post=False)
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
                # Treść wyjątku niesie URL `/bot<token>/` — do maila i bazy idzie typ.
                message=f"Push Telegram nie wyszedł (błąd sieci: {type(exc).__name__}).",
                action=(
                    "Sprawdź TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID i połączenie "
                    "wychodzące do api.telegram.org."
                ),
                # Tylko nazwa wyjątku: jego treść niesie URL z sekretem
                # (token / webhook), a logi repo są publiczne.
                log_message=f"Telegram push failed (network): {type(exc).__name__}.",
            )
            return

        if response.status_code >= 400:
            self._emit_quota_alert(
                message=(
                    f"Telegram zwrócił {response.status_code} — push nie został "
                    "dostarczony (zły token / chat_id albo zablokowany bot)."
                ),
                action=(
                    "Sprawdź, czy TELEGRAM_BOT_TOKEN jest ważny i czy bot został "
                    "uruchomiony w docelowym czacie (TELEGRAM_CHAT_ID)."
                ),
                log_message=(
                    f"Telegram returned {response.status_code} — push not "
                    "delivered (bad token / chat_id / bot blocked)."
                ),
            )
            return

        logger.info("Telegram: sent report to chat_id=%s", self._chat_id)

    def _emit_quota_alert(self, message: str, action: str, log_message: str) -> None:
        # MAIL-15: `message`/`action` idą do maila (po polsku), `log_message`
        # zostaje angielską diagnozą w logach.
        logger.error("%s", log_message)
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
