"""#16 — sweep revealów commit-reveal po wysyłce raportu Fast Loopa (ARCH-02).

Zna tylko porty: repozytorium (odczyt zapadłych commitmentów i stempel
`revealed_at`) oraz publisher atestacji. Fast Loop woła go za flagą
`attestation_enabled`, PO wysyłce raportu — publikacja to artefakt poboczny,
nie może opóźnić ani zablokować maila.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from src.application.ports import AttestationPublisherPort, RepositoryPort

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


class RevealAttestationsUseCase:
    """#16 — ujawnia WSZYSTKIE zapadłe commitmenty.

    Sweep, nie „reveal przy rozliczeniu": predykcje symboli usuniętych z
    `config.toml` nigdy nie przeszłyby przez reflect, więc ich commitmenty
    zostałyby na zawsze nieujawnione. Dla sceptyka wygląda to jak ukrywanie
    nietrafień — a to niszczy cały sens weryfikowalnego track recordu. WIA-03:
    dlatego odczyt nie ma ani filtra watchlisty, ani okna `track_record_days`.
    """

    def __init__(self, repository: RepositoryPort, publisher: AttestationPublisherPort) -> None:
        self._repository = repository
        self._publisher = publisher

    def run(self) -> int:
        """Zwraca liczbę revealów ze stemplem `revealed_at`.

        `revealed_at` stemplujemy DOPIERO po udanej publikacji. Odwrotna kolejność
        zgubiłaby nietrafienie na zawsze: oznaczone jako ujawnione, nigdy ujawnione.
        Pusty wynik `publish_reveal` to według kontraktu portu porażka, nie sukces.
        Best-effort per wiersz: jeden błąd nie przerywa sweepu.
        """
        try:
            rows = self._repository.get_unrevealed_commitments()
        except Exception:
            logger.exception("Failed to read unrevealed commitments — sweep skipped")
            return 0

        revealed = 0
        for row in rows:
            try:
                location = self._publisher.publish_reveal(reveal_record(row))
            except Exception:
                logger.exception("Failed to publish reveal for %s", row.get("id"))
                continue
            if not location:
                logger.warning("Reveal for %s was not published — left unrevealed", row.get("id"))
                continue
            try:
                self._repository.mark_commitment_revealed(str(row["id"]))
            except Exception:
                logger.exception("Failed to mark reveal for %s", row.get("id"))
                continue
            revealed += 1
        return revealed
