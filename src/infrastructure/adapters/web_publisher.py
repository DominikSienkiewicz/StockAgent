"""Adaptery dla `WebPublisherPort` (feature U3 — statyczny web digest):
- `FileWebPublisher` — zapisuje HTML raportu na dysk jako bookmarkowalny
  artefakt (np. plik serwowany przez GitHub Pages / commit workflow).
- `NullWebPublisher` — no-op gdy feature wyłączony (mirror `NullNotifier`).

FinOps: zero płatnych wywołań (LLM / rynek / embeddingi). Czysty zapis pliku.

Reguła odporności: publikacja NIGDY nie może wywalić cyklu agenta. Każdy błąd
I/O jest logowany (WARNING) i połykany — `publish` zwraca wtedy "" zamiast
podnosić wyjątek.

Uwaga architektoniczna: `WebPublisherPort` (ABC) żyje w `application/ports.py`
i jest dodawany przez orchestratora. Te adaptery są z nim strukturalnie zgodne
(metoda `publish(self, html: str) -> str`); orchestrator wstrzykuje je w miejsce
portu w `main_agent.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.application.ports import WebPublisherPort

logger = logging.getLogger(__name__)


class FileWebPublisher(WebPublisherPort):
    """Zapisuje raport HTML do `output_path`, tworząc brakujące katalogi.

    Zwraca absolutną ścieżkę zapisanego pliku (użyteczny artefakt, który
    workflow GitHub Pages / commit może serwować). Na błędzie zwraca "".
    """

    def __init__(self, output_path: str) -> None:
        self._output_path = output_path

    def publish(self, html: str) -> str:
        target = Path(self._output_path)
        try:
            # Tworzymy brakujące katalogi nadrzędne (idempotentnie).
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(html, encoding="utf-8")
        except OSError as exc:
            # Publikacja jest best-effort — błąd nie może przerwać cyklu.
            logger.warning(
                "Web digest publish failed for %s: %s", self._output_path, exc
            )
            return ""
        resolved = str(target.resolve())
        logger.info("Web digest published to %s", resolved)
        return resolved


class NullWebPublisher(WebPublisherPort):
    """No-op fallback gdy `WEB_DIGEST_ENABLED=false`.

    Nic nie zapisuje, żadne I/O nie jest dotykane — zwraca "".
    Pozwala main_agent.py mieć zawsze `web_publisher`, niezależnie od konfiguracji.
    """

    def publish(self, html: str) -> str:
        logger.info("Web digest disabled — skipping publish.")
        return ""
