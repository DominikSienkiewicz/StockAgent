"""DelegatingAlertNotifier — real-time push alertów przez istniejący transport.

Implementuje `AlertNotifierPort.send_alert` formatując listę `QuotaAlert` jako
tekst i delegując do wstrzykniętego `ReportNotifierPort` (Telegram / Slack /
Resend). Dzięki temu nie duplikujemy transportu — alert idzie tym samym kanałem
co raport.
"""

from __future__ import annotations

import logging

from src.application.ports import AlertNotifierPort, ReportNotifierPort
from src.domain.quota import QuotaAlert

logger = logging.getLogger(__name__)

ALERT_SUBJECT = "⚠️ [QUOTA] StockAgent — real-time alert"


class DelegatingAlertNotifier(AlertNotifierPort):
    def __init__(self, transport: ReportNotifierPort) -> None:
        self._transport = transport

    def send_alert(self, alerts: list[QuotaAlert]) -> None:
        if not alerts:
            return
        text = self._format(alerts)
        # Transport rozróżnia html_body/plain_text — dla push liczy się plain_text,
        # ale podajemy tekst też jako html_body (Resend dostanie sensowny body).
        self._transport.send_report(
            subject=ALERT_SUBJECT,
            html_body=text,
            plain_text=text,
        )

    def _format(self, alerts: list[QuotaAlert]) -> str:
        lines = ["Real-time quota alert:"]
        for alert in alerts:
            lines.append("")
            lines.append(f"[{alert.severity.value}] {alert.source}")
            lines.append(f"  {alert.message}")
            lines.append(f"  → {alert.action}")
        return "\n".join(lines)
