"""EXP-10 — sędzia diagnoz z rubryką na próbce (LLM-as-judge, Slow Loop).

Co tydzień losowa próbka najwyżej `MAX_JUDGED_PER_RUN` diagnoz błędnych predykcji
z ostatnich `days` dni (jeszcze bez oceny) idzie do sędziego, jedno wywołanie LLM na
diagnozę. Rubryka (`src/domain/diagnosis_rubric.py`): (1) wskazuje czynnik obecny
w newsach lub cenach predykcji, (2) jest falsyfikowalna, (3) nie jest ogólnikiem.
Wynik 0–3 trafia do `diagnosis_judge_score` (migracja 078); diagnoza poniżej progu
przechodzi do `rejected_diagnosis`, a `correction_insights` dostaje placeholder
`DIAGNOSIS_REJECTED_INSIGHT`, który lessons, analogi RAG i recap odsiewają.

Sędzia LLM ocenia LLM — to samo uprzedzenie, więc wynik znaczy coś dopiero po
kalibracji na ocenach człowieka (krok właściciela). Odpowiedź bez trzech ocen tak/nie
niczego nie zapisuje: wiersz wraca do kolejnej próbki.

FinOps (FIN-30, wyjątek od bramki volatility): najwyżej `max_samples` wywołań na
przebieg, przydział z `slow_loop_paid_budget`; `BudgetedLLMPort` odrzuca nadmiar.
Szablon promptu żyje tutaj, nie w modułach promptów predykcji (`prompt_sha256`).
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from datetime import datetime
from typing import Any

from src.application._prompt_safety import fence_untrusted
from src.application.errors import PortError
from src.application.metering import PaidBudgetExhausted
from src.application.ports import JudgeableDiagnosis, LLMPort, RepositoryPort
from src.domain.diagnosis_rubric import parse_rubric_verdict
from src.domain.prediction import is_substantive_insight

logger = logging.getLogger(__name__)

# FIN-30: najwyżej tyle płatnych wywołań sędziego na przebieg (tygodniowa próbka).
MAX_JUDGED_PER_RUN = 10
JUDGE_DAYS = 7
_ITEM_MAX_LEN = 300


def diagnosis_judge_prompt(row: JudgeableDiagnosis) -> str:
    """Prompt oceny jednej diagnozy rubryką trzech kryteriów tak/nie."""
    prediction = row.prediction
    headlines = [h for h in (prediction.news_summary or "").split(" | ") if h.strip()]
    evidence = list(headlines)
    if (prediction.reasoning or "").strip():
        evidence.append(f"Uzasadnienie prognozy: {prediction.reasoning}")
    evidence_block = (
        fence_untrusted("EVIDENCE", evidence, max_len=_ITEM_MAX_LEN)
        if evidence
        else "Brak zapisanych newsów i uzasadnienia."
    )
    diagnosis_block = fence_untrusted("DIAGNOSIS", row.insight, max_len=_ITEM_MAX_LEN * 2)
    return f"""
Jesteś sędzią jakości diagnoz błędnych prognoz. Prognoza {prediction.symbol}:
{prediction.predicted_trend}, wejście {prediction.price_at_prediction}, cel
{prediction.predicted_target_price}, rozliczenie {row.actual_price} — prognoza była błędna.

Dowody, którymi dysponowała prognoza:
{evidence_block}

Diagnoza do oceny:
{diagnosis_block}

Oceń diagnozę trzema kryteriami (tak/nie):
1. factor_in_evidence — wskazuje czynnik obecny w dowodach albo w cenach powyżej,
   a nie wymyślony po fakcie;
2. falsifiable — da się ją sprawdzić danymi (zdarzenie, poziom, data), a nie tylko opinią;
3. specific — nie jest ogólnikiem pasującym do każdej pomyłki („rynek był zmienny”).
Zwróć czysty JSON (bez bloków markdown):
{{"factor_in_evidence": true, "falsifiable": false, "specific": true}}
""".strip()


class JudgeDiagnosesUseCase:
    """Ocenia losową próbkę diagnoz tygodnia i odrzuca te poniżej progu rubryki.

    `rng` (wstrzyknięty) losuje próbkę; `max_samples` = przydział wywołań joba.
    Wyczerpany przydział kończy job, błąd LLM albo zapisu pomija tylko swój wiersz."""

    def __init__(
        self,
        repository_port: RepositoryPort,
        llm_port: LLMPort,
        *,
        clock: Callable[[], datetime],
        rng: random.Random,
        max_samples: int = MAX_JUDGED_PER_RUN,
        days: int = JUDGE_DAYS,
    ) -> None:
        self._repo = repository_port
        self._llm = llm_port
        self._clock = clock
        self._rng = rng
        self._max_samples = max(0, min(max_samples, MAX_JUDGED_PER_RUN))
        self._days = days

    def run(self) -> dict[str, Any]:
        if self._max_samples == 0:
            return {"status": "skipped", "reason": "Brak przydziału płatnych wywołań."}
        candidates = [
            row
            for row in self._repo.get_diagnoses_to_judge(self._days)
            if row.prediction.id is not None and is_substantive_insight(row.insight)
        ]
        if not candidates:
            return {"status": "skipped", "reason": "Brak diagnoz tygodnia do oceny."}
        sample = self._rng.sample(candidates, k=min(self._max_samples, len(candidates)))
        judged = rejected = failed = 0
        status = "judged"
        for row in sample:
            try:
                answer = self._llm.analyze_mistake(diagnosis_judge_prompt(row))
            except PaidBudgetExhausted:
                logger.warning("Diagnosis judge — przydział wyczerpany, koniec joba")
                status = "budget_exhausted"
                break
            except (PortError, RuntimeError):
                logger.warning("Diagnosis judge — LLM failed", exc_info=True)
                failed += 1
                continue
            outcome = self._judge(row, answer)
            if outcome is None:
                failed += 1
                continue
            judged += 1
            rejected += int(outcome)
        return {
            "status": status,
            "candidates": len(candidates),
            "judged": judged,
            "rejected": rejected,
            "failed": failed,
        }

    def _judge(self, row: JudgeableDiagnosis, answer: str) -> bool | None:
        """True = odrzucona, False = przyjęta, None = brak oceny albo zapis nie wyszedł."""
        verdict = parse_rubric_verdict(answer)
        prediction_id = row.prediction.id
        if verdict is None or prediction_id is None:
            logger.warning("Diagnosis judge — niepewna ocena dla %s, bez zapisu", prediction_id)
            return None
        try:
            self._repo.save_diagnosis_judgement(
                prediction_id,
                score=verdict.score,
                rejected_insight=None if verdict.passes else row.insight,
            )
        except PortError:
            logger.warning(
                "Diagnosis judge — zapis nie wyszedł dla %s", prediction_id, exc_info=True
            )
            return None
        return not verdict.passes
