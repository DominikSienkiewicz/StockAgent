"""NIEZ-33 — publiczna strona statusu i JSON odznaki „last digest” jako use case.

Zna tylko porty: repozytorium (darmowy odczyt `cycle_runs` z okna historii) i publisher
strony. Regułę „wstrzymany / spóźniony” dostaje od composition root jako
`LivenessExpectation` Fast Loopa — tę samą, której używa deadman. `main_scorecard`
woła `publish` (błąd odczytu = exit 1), Fast Loop — `run` (best-effort, po zapisie
wiersza `cycle_runs`, więc strona widzi bieżący przebieg).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.application.errors import PortError
from src.application.ports import RepositoryPort, WebPublisherPort
from src.application.report_status_page import (
    BADGE_JSON_NAME,
    STATUS_HISTORY_DAYS,
    build_public_status,
    render_status_badge,
    render_status_page_html,
)
from src.domain.liveness import LivenessExpectation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StatusPageOptions:
    """Reguła Fast Loopa z `Settings` i okno historii w dniach."""

    fast_loop: LivenessExpectation
    history_days: int = STATUS_HISTORY_DAYS


class PublishStatusPageUseCase:
    """NIEZ-33 — strona statusu i obok niej `badge.json` (endpoint shields.io)."""

    def __init__(
        self,
        repository: RepositoryPort,
        publisher: WebPublisherPort,
        options: StatusPageOptions,
        *,
        clock: Callable[[], datetime],
    ) -> None:
        self._repository = repository
        self._publisher = publisher
        self._options = options
        self._clock = clock

    def publish(self) -> bool:
        """True — strona i odznaka zapisane; False — któryś zapis padł. Błąd odczytu
        (`PortError`) leci do wołającego."""
        now = self._clock()
        runs = self._repository.get_cycle_runs_since(
            now - timedelta(days=self._options.history_days)
        )
        status = build_public_status(
            runs,
            now=now,
            fast_loop=self._options.fast_loop,
            history_days=self._options.history_days,
        )
        page_written = bool(self._publisher.publish(render_status_page_html(status)))
        badge_written = bool(
            self._publisher.publish_asset(
                BADGE_JSON_NAME,
                json.dumps(render_status_badge(status), ensure_ascii=False) + "\n",
                "application/json",
            )
        )
        return page_written and badge_written

    def run(self) -> None:
        """Best-effort: nieczytelna historia przebiegów nie wywraca cyklu (WARNING)."""
        try:
            self.publish()
        except PortError as exc:
            logger.warning("Status page not published — cycle_runs unreadable (%s)", exc)
