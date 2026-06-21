from __future__ import annotations

import logging
from typing import Any

from src.application.ports import LLMPort

logger = logging.getLogger(__name__)

# Domyślny próg pewności: poniżej niego wynik taniego modelu uznajemy za zbyt
# niepewny i (jeśli budżet pozwala) eskalujemy do frontier. 0.6 to rozsądny
# default cost/quality — nadpisywalny z configu.
DEFAULT_ROUTING_FLOOR = 0.6
# Domyślny budżet eskalacji per instancja. Jedna instancja na cykl ⇒ to
# efektywnie cap eskalacji na cykl (ochrona kosztów: nawet jeśli wszystkie
# wyniki są niepewne, nie zalejemy frontier modelu wywołaniami).
DEFAULT_MAX_ESCALATIONS = 5


class CascadingLLMAdapter(LLMPort):
    """Confidence-gated model routing — kompozyt dwóch `LLMPort` (Feature #2).

    Analizę domyślnie wykonuje TANI model. Dopiero gdy jego wynik jest mało
    pewny (`confidence_score < routing_floor`) i mamy jeszcze budżet eskalacji,
    powtarzamy analizę na modelu FRONTIER i zwracamy JEGO wynik. To standardowa
    dźwignia cost/quality: płacimy za drogi model tylko tam, gdzie tani nie był
    pewny.

    Budżet eskalacji jest per-instancja. main_agent tworzy jedną instancję na
    cykl, więc `max_escalations` działa jak cap eskalacji na cykl. Po jego
    wyczerpaniu zawsze zwracamy wynik taniego modelu (i logujemy to RAZ).

    Sam jest `LLMPort`, więc reszta systemu (graf, use case'y) nie wie, że pod
    spodem siedzą dwa modele — wstrzykiwany jest jak każdy inny adapter LLM.
    """

    def __init__(
        self,
        cheap: LLMPort,
        frontier: LLMPort,
        routing_floor: float = DEFAULT_ROUTING_FLOOR,
        max_escalations: int = DEFAULT_MAX_ESCALATIONS,
    ) -> None:
        self._cheap = cheap
        self._frontier = frontier
        self._routing_floor = routing_floor
        self._max_escalations = max_escalations
        # Licznik wykonanych eskalacji (rośnie). Po osiągnięciu max przestajemy
        # eskalować. Per-instancja ⇒ per-cykl przy jednej instancji na cykl.
        self._escalations_used = 0
        # Flaga, by log o wyczerpaniu budżetu poszedł dokładnie raz.
        self._exhaustion_logged = False

    def analyze(self, prompt: str) -> dict[str, Any]:
        cheap_result = self._cheap.analyze(prompt)
        confidence = _extract_confidence(cheap_result)

        # Pewny wynik (lub == floor) — nie eskalujemy, nie ruszamy budżetu.
        if confidence >= self._routing_floor:
            return cheap_result

        # Wynik niepewny — eskalujemy tylko gdy budżet jeszcze jest.
        if self._escalations_used >= self._max_escalations:
            self._log_exhaustion_once()
            return cheap_result

        self._escalations_used += 1
        logger.info(
            "Escalating to frontier LLM: cheap confidence %.3f < floor %.3f "
            "(escalation %d/%d)",
            confidence,
            self._routing_floor,
            self._escalations_used,
            self._max_escalations,
        )
        return self._frontier.analyze(prompt)

    def analyze_mistake(self, prompt: str) -> str:
        # Świadomy wybór: diagnoza błędu to free-form tekst (correction insight),
        # nie warto za nią płacić frontier modelem — delegujemy zawsze do taniego.
        return self._cheap.analyze_mistake(prompt)

    def _log_exhaustion_once(self) -> None:
        if self._exhaustion_logged:
            return
        self._exhaustion_logged = True
        logger.warning(
            "Escalation budget exhausted (%d/%d used) — serving cheap LLM "
            "results for the rest of this instance's lifetime.",
            self._escalations_used,
            self._max_escalations,
        )


def _extract_confidence(result: dict[str, Any]) -> float:
    """Wyciąga `confidence_score` jako float; brak / nie-liczba / None → 0.0.

    Brakujący lub niepoprawny `confidence_score` traktujemy jako MAŁO pewny
    (0.0) — to bezpieczne domyślne (raczej eskalujemy niż przepuścimy słaby
    wynik), spójne z logiką bramki."""
    raw = result.get("confidence_score")
    if isinstance(raw, bool):  # bool jest podtypem int — nie chcemy go jako confidence
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    return 0.0
