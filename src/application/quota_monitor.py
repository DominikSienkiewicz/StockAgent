"""QuotaMonitor — kolektor `QuotaAlert` współdzielony przez adaptery.

Wstrzykiwany do każdego adaptera, który ma quota signal (Alpha Vantage,
OpenAI, Finnhub, Resend). Adapter sam decyduje kiedy wystawić alert
(np. wszystkie klucze AV exhausted, OpenAI 429 po retry, Resend 4xx);
monitor tylko agreguje i podaje do raportu na końcu cyklu.

Brak globalnego stanu — każda sesja main_agent ma własną instancję.
"""

from __future__ import annotations

from src.domain.quota import QuotaAlert, QuotaSeverity


class QuotaMonitor:
    def __init__(self) -> None:
        self._alerts: list[QuotaAlert] = []

    def record(self, alert: QuotaAlert) -> None:
        self._alerts.append(alert)

    @property
    def alerts(self) -> list[QuotaAlert]:
        """Zwraca kopię — caller nie może modyfikować wewnętrznego stanu."""
        return list(self._alerts)

    def max_severity(self) -> QuotaSeverity | None:
        if not self._alerts:
            return None
        return max((a.severity for a in self._alerts), key=lambda s: s.rank)
