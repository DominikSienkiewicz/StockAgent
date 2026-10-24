"""Adaptery dla `WebPublisherPort` (publiczny scorecard #10):
- `FileWebPublisher` — zapisuje stronę HTML na dysk jako artefakt serwowany
  przez GitHub Pages (WIA-16/52).
- `NullWebPublisher` — no-op gdy feature wyłączony (mirror `NullNotifier`).

Prywatny digest nie idzie już na dysk: web digest U3 zastąpiło prywatne
archiwum w Supabase Storage (MAIL-24, BEZ-27).

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
        logger.info("Web page published to %s", resolved)
        return resolved

    def publish_asset(self, name: str, content: str, content_type: str) -> str:
        """Zapisuje `name` w katalogu strony — ten sam katalog idzie na Pages, więc plik
        trafia obok niej. Nazwa z katalogiem albo `..` to błąd wołającego (`ValueError`):
        zapis poza katalogiem strony wyszedłby poza artefakt publiczny."""
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"publish_asset expects a plain file name, got {name!r}")
        target = Path(self._output_path).parent / name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            # Publikacja jest best-effort — błąd nie może przerwać cyklu.
            logger.warning("Web asset publish failed for %s (%s): %s", target, content_type, exc)
            return ""
        resolved = str(target.resolve())
        logger.info("Web asset published to %s", resolved)
        return resolved


class NullWebPublisher(WebPublisherPort):
    """No-op fallback gdy `public_scorecard_enabled=false`.

    Nic nie zapisuje, żadne I/O nie jest dotykane — zwraca "".
    """

    def publish(self, html: str) -> str:
        logger.info("Web page publishing disabled — skipping publish.")
        return ""

    def publish_asset(self, name: str, content: str, content_type: str) -> str:
        logger.info("Web publisher disabled — skipping %s.", name)
        return ""
