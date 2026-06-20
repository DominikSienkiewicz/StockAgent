"""CompositeNotifier — fan-out `send_report` do wielu kanałów.

Pozwala wstrzyknąć JEDEN `ReportNotifierPort` do orkiestratora, a pod spodem
rozesłać raport do N kanałów (Resend + Telegram + Slack). Mirror wzorca
`RoutingMarketDataPort`: warstwa application nie wie o podziale na kanały.

Izolacja per-kanał: wyjątek jednego dziecka jest logowany i połknięty, więc
nie blokuje pozostałych ani nie propaguje do cyklu.
"""

from __future__ import annotations

import logging

from src.application.ports import ReportNotifierPort

logger = logging.getLogger(__name__)


class CompositeNotifier(ReportNotifierPort):
    def __init__(self, notifiers: list[ReportNotifierPort]) -> None:
        self._notifiers = notifiers

    def send_report(self, subject: str, html_body: str, plain_text: str) -> None:
        for notifier in self._notifiers:
            try:
                notifier.send_report(subject, html_body, plain_text)
            except Exception:
                # Jeden zepsuty kanał nie może zatrzymać pozostałych.
                logger.exception(
                    "CompositeNotifier: child %s raised on send_report",
                    type(notifier).__name__,
                )
