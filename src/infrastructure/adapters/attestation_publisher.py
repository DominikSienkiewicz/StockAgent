"""Adaptery dla `AttestationPublisherPort` (#16 — commit-reveal track record):
- `FileAttestationPublisher` — dopisuje commitmenty i revealy do append-only
  plików JSONL (osobne ścieżki), które workflow Gita może commitować.
- `NullAttestationPublisher` — no-op gdy feature wyłączony (mirror `NullWebPublisher`).

FinOps: zero płatnych wywołań (LLM / rynek / embeddingi). Czysty zapis pliku.

Reguła odporności: publikacja NIGDY nie może wywalić cyklu agenta. Każdy błąd
I/O jest logowany (WARNING) i połykany — publikacja zwraca wtedy "" zamiast
podnosić wyjątek (dokładnie jak `FileWebPublisher`).

UCZCIWOŚĆ CLAIMU (świadomie ograniczona):
    Niezależny timestamp bierze się z historii commitów Gita, a daty commitów Gita
    SĄ FAŁSZOWALNE (`GIT_COMMITTER_DATE` można ustawić dowolnie). To NIE jest
    niepodrabialny znacznik czasu. Realny gwarant to *tamper-evidence przy branch
    protection*: gdy główna gałąź jest chroniona i append-only, cofnięcie lub
    podmiana wcześniej opublikowanego commitmentu zostawia widoczny ślad w historii.
    Ten adapter obiecuje wyłącznie append-only zapis kwitów — nic więcej.

Uwaga architektoniczna: `AttestationPublisherPort` (ABC) żyje w `application/ports.py`;
te adaptery go implementują, a `main_agent.py` wstrzykuje konkretną instancję.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.application.ports import AttestationPublisherPort

logger = logging.getLogger(__name__)


class FileAttestationPublisher(AttestationPublisherPort):
    """Append-only JSONL: osobny plik dla commitmentów, osobny dla revealów.

    Każde wywołanie dopisuje JEDNĄ linię (zwarty JSON), nigdy nie nadpisuje
    wcześniejszych wpisów — to append-only jest fundamentem tamper-evidence.
    Brakujące katalogi nadrzędne są tworzone idempotentnie. Zwraca absolutną
    ścieżkę pliku, do którego dopisano wpis; na błędzie I/O zwraca "".
    """

    def __init__(self, commitment_path: str, reveal_path: str) -> None:
        self._commitment_path = commitment_path
        self._reveal_path = reveal_path

    def publish_commitment(self, record: Mapping[str, Any]) -> str:
        return self._append(self._commitment_path, record)

    def publish_reveal(self, record: Mapping[str, Any]) -> str:
        return self._append(self._reveal_path, record)

    def _append(self, path: str, record: Mapping[str, Any]) -> str:
        target = Path(path)
        # Jedna linia = jeden zwarty JSON. `sort_keys` czyni zapis
        # deterministycznym niezależnie od kolejności kluczy wejścia;
        # `ensure_ascii=False` zachowuje polskie znaki dosłownie.
        line = json.dumps(dict(record), sort_keys=True, ensure_ascii=False)
        try:
            # Tworzymy brakujące katalogi nadrzędne (idempotentnie).
            target.parent.mkdir(parents=True, exist_ok=True)
            # Tryb "a" — dopisujemy, nigdy nie nadpisujemy.
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            # Publikacja jest best-effort — błąd nie może przerwać cyklu.
            logger.warning("Attestation publish failed for %s: %s", path, exc)
            return ""
        resolved = str(target.resolve())
        logger.info("Attestation record appended to %s", resolved)
        return resolved


class NullAttestationPublisher(AttestationPublisherPort):
    """No-op fallback gdy feature commit-reveal wyłączony.

    Nic nie zapisuje, żadne I/O nie jest dotykane — zwraca "".
    Pozwala main_agent.py mieć zawsze `attestation_publisher`, niezależnie
    od konfiguracji.
    """

    def publish_commitment(self, record: Mapping[str, Any]) -> str:
        logger.info("Attestation disabled — skipping commitment publish.")
        return ""

    def publish_reveal(self, record: Mapping[str, Any]) -> str:
        logger.info("Attestation disabled — skipping reveal publish.")
        return ""
