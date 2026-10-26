"""Wspólna sygnalizacja kwot adapterów LLM (ARCH-29).

Każdy adapter LLM miał własną kopię `_emit_quota_alert`: log po angielsku i
`QuotaAlert` po polsku do `QuotaMonitor`. `QuotaEmitter` robi to raz — adapter
podaje źródło alertu (np. `Anthropic (claude-sonnet-4-6)`) i swój logger, więc
nazwy loggerów i treści alertów zostają takie jak przed wydzieleniem.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity


def _utc_now() -> datetime:
    return datetime.now(UTC)


class QuotaEmitter:
    """Alert kwoty jednego źródła: wpis w logu adaptera i — gdy jest monitor —
    `QuotaAlert` do maila.

    MAIL-15: `message` i `action` idą do maila (po polsku), `log_message` zostaje
    angielską diagnozą w logach. CRITICAL loguje na ERROR, reszta na WARNING."""

    def __init__(
        self,
        source: str,
        quota_monitor: QuotaMonitor | None,
        logger: logging.Logger,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._source = source
        self._quota_monitor = quota_monitor
        self._logger = logger
        self._clock = clock

    def emit(self, severity: QuotaSeverity, message: str, action: str, log_message: str) -> None:
        level = logging.ERROR if severity is QuotaSeverity.CRITICAL else logging.WARNING
        self._logger.log(level, "%s", log_message)
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source=self._source,
                severity=severity,
                message=message,
                action=action,
                occurred_at=self._clock(),
            )
        )
