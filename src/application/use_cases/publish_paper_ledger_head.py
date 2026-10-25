"""WIA-59 — głowa append-only paper ledgera publikowana raz dziennie obok atestacji.

Zna tylko porty: repozytorium (głowa łańcucha `paper_trades`) i publisher głowy.
Fast Loop woła go za `paper_ledger_enabled` i `attestation_enabled`, po zapisach
cyklu (save i reflect już otworzyły i zamknęły pozycje). Opublikowany hash głowy
kotwiczy cały łańcuch przed nią: podmiana wiersza w bazie rozjechałaby się z nim.
Publikacja jest best-effort — nie może zatrzymać raportu.
"""

from __future__ import annotations

import logging
from datetime import date

from src.application.errors import PortError
from src.application.ports import LedgerHeadPublisherPort, RepositoryPort
from src.domain.paper_ledger import head_record, should_publish_head

logger = logging.getLogger(__name__)


class PublishPaperLedgerHeadUseCase:
    """Pierwszy przebieg dnia dopisuje linię z głową; kolejne i pusty ledger — nic."""

    def __init__(self, repository: RepositoryPort, publisher: LedgerHeadPublisherPort) -> None:
        self._repository = repository
        self._publisher = publisher

    def run(self, today: date) -> str:
        """Zwraca ścieżkę publikacji albo "" (już dziś, pusty ledger, błąd odczytu)."""
        if not should_publish_head(self._publisher.last_published_on(), today):
            return ""
        try:
            head = self._repository.get_paper_ledger_head()
        except PortError:
            logger.warning("Paper ledger head unavailable — not published today", exc_info=True)
            return ""
        if head is None:
            return ""
        published = self._publisher.publish_head(head_record(head, today))
        if published:
            logger.info("Paper ledger head %s (seq %d) published", head.row_hash, head.seq)
        return published
