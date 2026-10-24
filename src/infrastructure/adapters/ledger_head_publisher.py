"""Adapter `LedgerHeadPublisherPort` (WIA-59): głowy paper ledgera w append-only JSONL.

Plik leży obok ledgera atestacji (`paper_ledger_heads_path` w config.toml, katalog
checkoutu gałęzi `attestation-log`), więc workflow Fast Loopa wypycha go razem
z commitmentami — historia gałęzi to publiczny ślad, którego nie da się po cichu
przepisać przy branch protection. Zero płatnych wywołań, zero sieci.

Odporność: błąd I/O nigdy nie wywala cyklu — zapis zwraca "", odczyt dnia
ostatniej publikacji zwraca None (wtedy najwyżej druga linia tego samego dnia).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from src.application.ports import LedgerHeadPublisherPort

logger = logging.getLogger(__name__)


class FileLedgerHeadPublisher(LedgerHeadPublisherPort):
    """Jedna linia (zwarty JSON, posortowane klucze) na publikację, tylko dopisywanie."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def last_published_on(self) -> date | None:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("Paper ledger heads unreadable at %s: %s", self._path, exc)
            return None
        for line in reversed(lines):
            if line.strip():
                return _published_on(line)
        return None

    def publish_head(self, record: Mapping[str, Any]) -> str:
        line = json.dumps(dict(record), sort_keys=True, ensure_ascii=False)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            logger.warning("Paper ledger head publish failed for %s: %s", self._path, exc)
            return ""
        return str(self._path.resolve())


def _published_on(line: str) -> date | None:
    """Dzień z linii głowy; linia nieczytelna (uszkodzony plik) → None."""
    try:
        record = json.loads(line)
        return date.fromisoformat(str(record["published_on"]))
    except (ValueError, KeyError, TypeError):
        logger.warning("Paper ledger heads: last line is not a head record")
        return None
