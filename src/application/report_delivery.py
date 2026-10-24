"""ARCH-08: wynik wysyłki raportu cyklu — kanał po kanale.

`main_agent.deliver_report` (composition root, bo buduje kanały z `Settings`)
zwraca `DeliveryOutcome` zamiast jednego bool spod `except Exception` całej
fazy. Step summary (NIEZ-22) i `cycle_runs` (NIEZ-29) czytają z niego
`delivered`; lista kanałów mówi, który wyszedł, a który padł.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ReportChannel(StrEnum):
    """Kanał, którym raport cyklu wychodzi do odbiorcy."""

    WEB = "web"
    EMAIL = "email"
    LEGACY_PUSH = "legacy_push"
    PUSH_DIGEST = "push_digest"


class DeliveryStatus(StrEnum):
    """`SKIPPED` — kanał wyłączony (atrapa) albo port zwrócił pusty wynik."""

    SENT = "sent"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ChannelDelivery:
    """Wynik jednego kanału; `recipient` to zamaskowany adres subskrybenta (U2)."""

    channel: ReportChannel
    status: DeliveryStatus
    recipient: str | None = None


@dataclass(frozen=True)
class DeliveryOutcome:
    """`report_sent` — faza wysyłki doszła do końca (raport powstał i poszedł
    do kanałów); wyjątek przed nią albo w jej trakcie zostawia False."""

    report_sent: bool
    channels: tuple[ChannelDelivery, ...] = ()

    @property
    def email_delivered(self) -> bool:
        """Żaden mail cyklu (właściciel albo subskrybent) nie padł."""
        return not any(
            c.channel is ReportChannel.EMAIL and c.status is DeliveryStatus.FAILED
            for c in self.channels
        )

    @property
    def delivered(self) -> bool:
        """Mail cyklu wyszedł bez błędu (NIEZ-22/NIEZ-29)."""
        return self.report_sent and self.email_delivered
