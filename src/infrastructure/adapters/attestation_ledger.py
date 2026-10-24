"""Adapter `AttestationLedgerPort` (WIA-15): odczyt trwałego ledgera atestacji.

Fast Loop checkoutuje gałąź `attestation-log` do katalogu z `attestation_*_path`
PRZED krokiem agenta (WIA-14), więc na starcie runu pliki JSONL to dokładnie to, co
przeżyło push poprzedniego runu. Adapter tylko czyta — zapis należy do
`FileAttestationPublisher`.

Uszkodzona linia jest pomijana z WARNING-iem: reveal, którego tu nie widać, zostaje
nieostemplowany i ujawniony ponownie (duplikat linii jest nieszkodliwy, weryfikator
liczy go raz). Błąd I/O (np. brak uprawnień) to `ProviderUnavailable`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.application.errors import ProviderUnavailable
from src.application.ports import AttestationLedgerPort
from src.infrastructure.error_translation import as_port_error

logger = logging.getLogger(__name__)


class FileAttestationLedger(AttestationLedgerPort):
    """Ledger z plików JSONL checkoutu `attestation-log`."""

    def __init__(self, commitment_path: str, reveal_path: str) -> None:
        self._commitment_path = Path(commitment_path)
        self._reveal_path = Path(reveal_path)

    def revealed_commitments(self) -> frozenset[str]:
        return frozenset(
            commitment.lower()
            for record in _records(self._reveal_path)
            if isinstance(commitment := record.get("commitment"), str) and commitment
        )


def _records(path: Path) -> Iterator[dict[str, Any]]:
    """Obiekty JSON z kolejnych linii pliku; brak pliku = brak rekordów."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except OSError as exc:
        raise as_port_error(exc, ProviderUnavailable) from exc
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if isinstance(record, dict):
            yield record
        else:
            logger.warning("Attestation ledger %s:%d is not a JSON object — skipped", path, number)
