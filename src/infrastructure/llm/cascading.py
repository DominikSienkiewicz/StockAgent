from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any

from src.application.ports import LLM_ROUTED_MODEL_KEY, LLMPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity

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

    NIEZ-17: gdy frontier padnie, zwracamy opłacony już wynik taniego modelu
    (WARNING przy każdej awarii, alert do `quota_monitor` raz na instancję —
    niedziałający routing nie może się schować). Rezerwacja eskalacji idzie pod
    lockiem: symbole cyklu biegną równolegle, a sufit jest wspólny.

    FIN-21: z tożsamościami `cheap_model` / `frontier_model` (`dostawca:model`)
    każdy wynik `analyze` niesie `LLM_ROUTED_MODEL_KEY` — model, który faktycznie
    odpowiedział — żeby kwit predykcji pozwolił porównać trafność per model.
    """

    def __init__(
        self,
        cheap: LLMPort,
        frontier: LLMPort,
        routing_floor: float = DEFAULT_ROUTING_FLOOR,
        max_escalations: int = DEFAULT_MAX_ESCALATIONS,
        quota_monitor: QuotaMonitor | None = None,
        *,
        cheap_model: str | None = None,
        frontier_model: str | None = None,
    ) -> None:
        self._cheap = cheap
        self._frontier = frontier
        self._cheap_model = cheap_model
        self._frontier_model = frontier_model
        self._routing_floor = routing_floor
        self._max_escalations = max_escalations
        self._quota_monitor = quota_monitor
        # Licznik wykonanych eskalacji (rośnie). Po osiągnięciu max przestajemy
        # eskalować. Per-instancja ⇒ per-cykl przy jednej instancji na cykl.
        self._escalations_used = 0
        # Flaga, by log o wyczerpaniu budżetu poszedł dokładnie raz.
        self._exhaustion_logged = False
        self._frontier_failure_alerted = False
        # Chroni check-then-act licznika i obu flag przy równoległych symbolach.
        self._lock = threading.Lock()

    def analyze(self, prompt: str) -> dict[str, Any]:
        cheap_result = self._cheap.analyze(prompt)
        confidence = _extract_confidence(cheap_result)

        # Pewny wynik (lub == floor) — nie eskalujemy, nie ruszamy budżetu.
        if confidence >= self._routing_floor:
            return _marked(cheap_result, self._cheap_model)

        # Wynik niepewny — eskalujemy tylko gdy budżet jeszcze jest.
        escalation = self._reserve_escalation()
        if escalation is None:
            return _marked(cheap_result, self._cheap_model)

        logger.info(
            "Escalating to frontier LLM: cheap confidence %.3f < floor %.3f "
            "(escalation %d/%d)",
            confidence,
            self._routing_floor,
            escalation,
            self._max_escalations,
        )
        try:
            frontier_result = self._frontier.analyze(prompt)
        except Exception as exc:
            self._report_frontier_failure(exc)
            return _marked(cheap_result, self._cheap_model)
        return _marked(frontier_result, self._frontier_model)

    def analyze_mistake(self, prompt: str) -> str:
        # Świadomy wybór: diagnoza błędu to free-form tekst (correction insight),
        # nie warto za nią płacić frontier modelem — delegujemy zawsze do taniego.
        return self._cheap.analyze_mistake(prompt)

    def _reserve_escalation(self) -> int | None:
        """Numer zarezerwowanej eskalacji albo None, gdy sufit jest wyczerpany."""
        with self._lock:
            if self._escalations_used >= self._max_escalations:
                self._log_exhaustion_once()
                return None
            self._escalations_used += 1
            return self._escalations_used

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

    def _report_frontier_failure(self, exc: Exception) -> None:
        # Sam typ błędu: komunikat SDK potrafi nieść URL albo fragment odpowiedzi.
        logger.warning(
            "Frontier LLM failed (%s) — serving the paid cheap LLM result instead",
            type(exc).__name__,
        )
        with self._lock:
            first = not self._frontier_failure_alerted
            self._frontier_failure_alerted = True
        if not first or self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source="Routing modeli (frontier)",
                severity=QuotaSeverity.WARNING,
                message=(
                    f"Model frontier nie odpowiedział ({type(exc).__name__}) — niepewne "
                    "wyniki taniego modelu poszły bez eskalacji."
                ),
                action=(
                    "Sprawdź klucz, model i status dostawcy z ROUTER_FRONTIER_PROVIDER / "
                    "ROUTER_FRONTIER_MODEL albo wyłącz MODEL_ROUTING_ENABLED."
                ),
                occurred_at=datetime.now(UTC),
            )
        )


def _marked(result: dict[str, Any], model: str | None) -> dict[str, Any]:
    """FIN-21: kopia wyniku ze znacznikiem modelu, który odpowiedział. Bez tożsamości
    modelu (adapter złożony bez `cheap_model` / `frontier_model`) wynik bez zmian."""
    if model is None:
        return result
    return {**result, LLM_ROUTED_MODEL_KEY: model}


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
