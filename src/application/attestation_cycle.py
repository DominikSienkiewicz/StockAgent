"""WIA-60 — commitmenty cyklu jako liście jednego drzewa Merkle'a.

Graf publikuje commitment każdej zapisanej predykcji przez `AttestationPublisherPort`
jak dotąd (po zapisie wiersza, WIA-01). `CycleCommitmentCollector` stoi między grafem
a publisherem: zbiera hashe jako liście, a po przebiegu symboli DI domyka cykl
(`seal_attestation_cycle`) — do ledgera trafia jedna linia z rootem, `n_leaves`
i liśćmi (`domain.attestation.cycle_root_record`), także dla pustego zbioru.

Revealów nie zbiera: reveal nadpisanej predykcji (WIA-01) wychodzi od razu, bo po
upsercie jej soli nie ma już w bazie.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from src.application.ports import AttestationPublisherPort
from src.domain.attestation import cycle_root_record

logger = logging.getLogger(__name__)

# Wynik `publish_commitment`, gdy hash czeka na root cyklu: nie ścieżka pliku, ale też
# nie "" (kontrakt portu: "" = wyłączone albo błąd).
PENDING_CYCLE_ROOT = "pending:cycle-root"


def attestation_cycle_id(run_id: str, run_attempt: int | None, started_at: datetime) -> str:
    """Identyfikator cyklu w ledgerze: run, próba i start (UTC) — ponowienie runu
    (`GITHUB_RUN_ATTEMPT`) i run lokalny dostają własny cykl."""
    return f"{run_id}.{run_attempt or 1}.{started_at.astimezone(UTC):%Y%m%dT%H%M%SZ}"


class CycleCommitmentCollector(AttestationPublisherPort):
    """Publisher cyklu: commitmenty → liście, revealy → prosto do publishera.

    Wątki symboli (`symbol_concurrency > 1`) dopisują liście równolegle, stąd lock.
    Po `seal` cykl jest zamknięty: kolejny commitment byłby tylko w bazie, więc jest
    odrzucany z WARNING-iem zamiast po cichu wypaść z rootu."""

    def __init__(self, publisher: AttestationPublisherPort) -> None:
        self._publisher = publisher
        self._leaves: set[str] = set()
        self._sealed = False
        self._lock = threading.Lock()

    def publish_commitment(self, record: Mapping[str, Any]) -> str:
        commitment = record.get("commitment")
        if not isinstance(commitment, str) or not commitment:
            logger.warning("Attestation commitment line without a hash — not a leaf")
            return ""
        with self._lock:
            if self._sealed:
                logger.warning("Attestation commitment after the cycle root — not published")
                return ""
            self._leaves.add(commitment.lower())
        return PENDING_CYCLE_ROOT

    def publish_reveal(self, record: Mapping[str, Any]) -> str:
        return self._publisher.publish_reveal(record)

    def seal(self, cycle_id: str, committed_at: datetime) -> str:
        """Publikuje linię rootu cyklu (raz); zwraca wynik publishera albo "" po zamknięciu."""
        with self._lock:
            if self._sealed:
                return ""
            self._sealed = True
            leaves = tuple(self._leaves)
        return self._publisher.publish_commitment(
            cycle_root_record(cycle_id=cycle_id, committed_at=committed_at, commitments=leaves)
        )


def seal_attestation_cycle(
    publisher: AttestationPublisherPort | None, *, cycle_id: str, committed_at: datetime
) -> None:
    """Domyka cykl, gdy graf dostał `CycleCommitmentCollector` (atestacja włączona
    i kolumny w bazie); inny publisher albo None = nic do domknięcia."""
    if not isinstance(publisher, CycleCommitmentCollector):
        return
    if not publisher.seal(cycle_id, committed_at):
        logger.warning(
            "Attestation cycle root %s not published — its predictions will be "
            "revealed as an incomplete cycle",
            cycle_id,
        )
