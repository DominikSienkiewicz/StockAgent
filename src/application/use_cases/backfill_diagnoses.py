"""WIA-53 — spłata długu diagnoz błędnych predykcji (Slow Loop).

`reflect_node` płaci za diagnozę błędu tylko za bramką volatility. Na spokojnym
cyklu zamyka predykcję z placeholderem (`DIAGNOSIS_SKIPPED_INSIGHT`, w starych
wierszach „odłożona”) i nic jej później nie ponawia, więc lessons i analogi RAG
nigdy nie dostają z niej wniosku. Ten job bierze takie błędne predykcje i każdą
diagnozuje jednym wywołaniem LLM z jej PERSYSTOWANEGO wejścia: nagłówków,
uzasadnienia, ceny wejścia, celu i rozliczenia oraz kwitu decyzji (próg
efektywny, analogi RAG) — materiału, którego `reflect` nie miał.

FinOps (FIN-30, wyjątek od bramki volatility): najwyżej `max_paid_calls`
wywołań na przebieg — przydział z `slow_loop_paid_budget`, a `BudgetedLLMPort`
odrzuca nadmiar przed dostawcą. Job chodzi sekwencyjnie w Slow Loopie, nie
w `reflect`, więc nie ściga się z `symbol_concurrency` Fast Loopa.

Uwaga: zapis podmienia historię — przepisany wniosek trafia do lessons i do
analogów RAG kolejnych predykcji.
"""

from __future__ import annotations

import logging
from typing import Any

from src.application.decision_receipt import read_decision_receipt
from src.application.metering import PaidBudgetExhausted
from src.application.ports import DeferredDiagnosis, LLMPort, RepositoryPort
from src.application.prompts import get_mistake_diagnosis_prompt
from src.domain.mistake_taxonomy import parse_mistake_diagnosis
from src.domain.prediction import is_substantive_insight

logger = logging.getLogger(__name__)


def _prompt(row: DeferredDiagnosis) -> str:
    prediction = row.prediction
    return get_mistake_diagnosis_prompt(
        last_trend=str(prediction.predicted_trend),
        last_news_summary=prediction.news_summary or "",
        actual_price=str(row.actual_price),
        entry_price=str(prediction.price_at_prediction),
        target_price=str(prediction.predicted_target_price),
        last_reasoning=prediction.reasoning or "",
        receipt=read_decision_receipt(row.receipt),
    )


class BackfillDiagnosesUseCase:
    """Diagnozuje najwyżej `max_paid_calls` błędnych predykcji bez diagnozy.

    Błąd LLM albo pusta odpowiedź zostawia placeholder (wiersz wraca w kolejnym
    przebiegu); wyczerpany przydział kończy job. Zapis podmienia tylko ten
    placeholder, który job przeczytał."""

    def __init__(
        self, repository_port: RepositoryPort, llm_port: LLMPort, max_paid_calls: int
    ) -> None:
        self._repo = repository_port
        self._llm = llm_port
        self._max_paid_calls = max(0, max_paid_calls)

    def run(self) -> dict[str, Any]:
        if self._max_paid_calls == 0:
            return {"status": "skipped", "reason": "Brak przydziału płatnych wywołań."}
        rows = self._repo.get_deferred_diagnoses(limit=self._max_paid_calls)
        candidates = [row for row in rows if row.is_trend_correct is False]
        diagnosed = failed = 0
        status = "backfilled"
        for row in candidates[: self._max_paid_calls]:
            try:
                answer = self._llm.analyze_mistake(_prompt(row))
            except PaidBudgetExhausted:
                logger.warning("Diagnosis backfill — przydział wyczerpany, koniec joba")
                status = "budget_exhausted"
                break
            except Exception:
                logger.exception("Diagnosis backfill LLM failed for %s", row.prediction.id)
                failed += 1
                continue
            if self._save(row, answer):
                diagnosed += 1
            else:
                failed += 1
        return {
            "status": status,
            "candidates": len(candidates),
            "diagnosed": diagnosed,
            "failed": failed,
        }

    def _save(self, row: DeferredDiagnosis, answer: str) -> bool:
        diagnosis = parse_mistake_diagnosis(answer)
        prediction_id = row.prediction.id
        if prediction_id is None or not is_substantive_insight(diagnosis.insight):
            logger.warning("Diagnosis backfill — pusta diagnoza dla %s, placeholder zostaje",
                           prediction_id)
            return False
        try:
            self._repo.save_backfilled_diagnosis(
                prediction_id,
                diagnosis.insight,
                replaces=row.placeholder,
                correction_category=diagnosis.category,
            )
        except Exception:
            logger.exception("save_backfilled_diagnosis failed for %s", prediction_id)
            return False
        return True
