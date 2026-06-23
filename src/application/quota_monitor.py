"""QuotaMonitor — kolektor `QuotaAlert` współdzielony przez adaptery.

Wstrzykiwany do każdego adaptera, który ma quota signal (Alpha Vantage,
OpenAI, Finnhub, Resend). Adapter sam decyduje kiedy wystawić alert
(np. wszystkie klucze AV exhausted, OpenAI 429 po retry, Resend 4xx);
monitor tylko agreguje i podaje do raportu na końcu cyklu.

Brak globalnego stanu — każda sesja main_agent ma własną instancję.
"""

from __future__ import annotations

import logging
import threading

from src.application.ports import AlertNotifierPort
from src.domain.quota import QuotaAlert, QuotaSeverity

logger = logging.getLogger(__name__)


class QuotaMonitor:
    def __init__(self, alert_notifier: AlertNotifierPort | None = None) -> None:
        self._alerts: list[QuotaAlert] = []
        # U4 — opcjonalny real-time push. Default None ⇒ dzisiejsze zachowanie.
        self._alert_notifier = alert_notifier
        # Debounce per źródło na czas życia monitora — jedno CRITICAL z danego
        # źródła pushuje raz, kolejne tylko trafiają do raportu dobowego.
        self._alerted_sources: set[str] = set()
        # Serializuje record() — chroni check-then-act na _alerted_sources przy
        # współbieżnym wołaniu (rada na 7 wątkach, równoległe symbole ~N×7).
        self._lock = threading.Lock()

    def record(self, alert: QuotaAlert) -> None:
        with self._lock:
            self._alerts.append(alert)
            self._maybe_push_realtime(alert)

    def _maybe_push_realtime(self, alert: QuotaAlert) -> None:
        if self._alert_notifier is None:
            return
        if alert.severity is not QuotaSeverity.CRITICAL:
            return
        if alert.source in self._alerted_sources:
            return
        # Debounce ustawiamy PRZED wysyłką: nawet gdy push padnie, nie spamujemy
        # ponownie tego samego źródła w tym cyklu.
        self._alerted_sources.add(alert.source)
        try:
            self._alert_notifier.send_alert([alert])
        except Exception:
            # Padający push nie może wywalić record() ani cyklu — alert i tak
            # został już zebrany do raportu dobowego.
            logger.exception(
                "QuotaMonitor: real-time alert push failed for source=%s",
                alert.source,
            )

    @property
    def alerts(self) -> list[QuotaAlert]:
        """Zwraca kopię — caller nie może modyfikować wewnętrznego stanu."""
        return list(self._alerts)

    def max_severity(self) -> QuotaSeverity | None:
        if not self._alerts:
            return None
        return max((a.severity for a in self._alerts), key=lambda s: s.rank)
