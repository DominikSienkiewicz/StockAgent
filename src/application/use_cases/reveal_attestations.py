"""#16 — sweep revealów commit-reveal po wysyłce raportu Fast Loopa (ARCH-02).

Zna tylko porty: repozytorium (odczyt zapadłych commitmentów i stempel
`revealed_at`), publisher atestacji i — od WIA-15 — trwały ledger (checkout gałęzi
`attestation-log`). Fast Loop woła go za flagą `attestation_enabled`: `reconcile`
na starcie runu, `run` PO wysyłce raportu — publikacja to artefakt poboczny, nie może
opóźnić ani zablokować maila.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from src.application.errors import PortError
from src.application.ports import (
    AttestationLedgerPort,
    AttestationPublisherPort,
    RepositoryPort,
)

logger = logging.getLogger(__name__)


def reveal_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """#16 — rekord revealu: sól, hash i plaintext predykcji z wynikiem.

    WIA-70: wiersz z `commitment_payload` (043) ujawnia zapisany plaintext
    dosłownie (`payload`) — sceptyk liczy SHA-256(payload + sól) bez rekonstrukcji.
    Wiersz bez niego (sprzed 043) ujawnia kolumny, z których odtwarza się payload v1.
    """
    record: dict[str, Any] = {
        "symbol": row.get("symbol"),
        "commitment": row.get("commitment_hash"),
        # Sól wychodzi na jaw dopiero TERAZ — to jest sens revealu.
        "salt": row.get("commitment_salt"),
        "predicted_trend": row.get("predicted_trend"),
        "predicted_target_price": row.get("predicted_target_price"),
        "price_at_prediction": row.get("price_at_prediction"),
        "actual_price_after_12h": row.get("actual_price_after_12h"),
        "is_trend_correct": row.get("is_trend_correct"),
    }
    payload = row.get("commitment_payload")
    if payload is not None:
        record["payload"] = payload
    return record


def _commitment_of(row: Mapping[str, Any]) -> str:
    value = row.get("commitment_hash")
    return value.lower() if isinstance(value, str) else ""


class RevealAttestationsUseCase:
    """#16 — ujawnia WSZYSTKIE zapadłe commitmenty.

    Sweep, nie „reveal przy rozliczeniu": predykcje symboli usuniętych z
    `config.toml` nigdy nie przeszłyby przez reflect, więc ich commitmenty
    zostałyby na zawsze nieujawnione. Dla sceptyka wygląda to jak ukrywanie
    nietrafień — a to niszczy cały sens weryfikowalnego track recordu. WIA-03:
    dlatego odczyt nie ma ani filtra watchlisty, ani okna `track_record_days`.

    WIA-15: `ledger` (trwały ledger) rozdziela publikację od stempla. Plik revealów
    w runie jest tylko etapem przed pushem, więc `run` publikuje bez stempla,
    a `reconcile` na starcie następnego runu stempluje wyłącznie wiersze, których
    hash jest już w wcheckoutowanym `reveals.jsonl`. Bez `ledger` publikacja jest
    trwała z definicji publishera i stempel idzie od razu po niej.
    """

    def __init__(
        self,
        repository: RepositoryPort,
        publisher: AttestationPublisherPort,
        ledger: AttestationLedgerPort | None = None,
    ) -> None:
        self._repository = repository
        self._publisher = publisher
        self._ledger = ledger

    def reconcile(self) -> int:
        """WIA-15 — stempluje `revealed_at` wierszom z revealem w trwałym ledgerze.

        Idempotentne: ponowny run widzi te same linie i stempluje tylko to, czego
        jeszcze nie ostemplował. Nieczytelny ledger = brak stempli (WARNING)."""
        durable = self._durable_reveals()
        if not durable:
            return 0
        rows = [row for row in self._unrevealed() if _commitment_of(row) in durable]
        return sum(1 for row in rows if self._mark(row))

    def run(self) -> int:
        """Publikuje revealy zapadłych commitmentów; zwraca liczbę domkniętych.

        Bez ledgera: `revealed_at` DOPIERO po udanej publikacji (odwrotna kolejność
        zgubiłaby nietrafienie na zawsze), wynik = wiersze ze stemplem. Z ledgerem:
        wiersze z revealem już w ledgerze są pomijane (ostempluje je `reconcile`),
        reszta jest publikowana bez stempla, wynik = opublikowane. Pusty wynik
        `publish_reveal` to według kontraktu portu porażka, nie sukces.
        Best-effort per wiersz: jeden błąd nie przerywa sweepu.
        """
        if self._ledger is None:
            return sum(1 for row in self._unrevealed() if self._publish(row) and self._mark(row))
        try:
            durable = frozenset(value.lower() for value in self._ledger.revealed_commitments())
        except PortError as exc:
            logger.warning("Attestation ledger unreadable (%s) — reveal sweep skipped", exc)
            return 0
        pending = [row for row in self._unrevealed() if _commitment_of(row) not in durable]
        return sum(1 for row in pending if self._publish(row))

    def _durable_reveals(self) -> frozenset[str]:
        if self._ledger is None:
            return frozenset()
        try:
            return frozenset(value.lower() for value in self._ledger.revealed_commitments())
        except PortError as exc:
            logger.warning("Attestation ledger unreadable (%s) — no reveal stamped", exc)
            return frozenset()

    def _unrevealed(self) -> Sequence[Mapping[str, Any]]:
        try:
            return self._repository.get_unrevealed_commitments()
        except Exception:
            logger.exception("Failed to read unrevealed commitments — sweep skipped")
            return []

    def _publish(self, row: Mapping[str, Any]) -> bool:
        try:
            location = self._publisher.publish_reveal(reveal_record(row))
        except (OSError, PortError, TypeError, ValueError):
            logger.exception("Failed to publish reveal for %s", row.get("id"))
            return False
        if not location:
            logger.warning("Reveal for %s was not published — left unrevealed", row.get("id"))
        return bool(location)

    def _mark(self, row: Mapping[str, Any]) -> bool:
        try:
            self._repository.mark_commitment_revealed(str(row["id"]))
        except Exception:
            logger.exception("Failed to mark reveal for %s", row.get("id"))
            return False
        return True
