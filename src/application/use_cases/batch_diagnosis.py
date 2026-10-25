"""EXP-17 — diagnoza zbiorcza raz w tygodniu (Slow Loop).

Za `batch_diagnosis_enabled` `reflect` w Fast Loopie nie płaci za diagnozę każdego
błędu, tylko zamyka predykcję z placeholderem `DIAGNOSIS_BATCH_PENDING_INSIGHT`.
Ten job bierze błędne predykcje ostatnich `days` dni, które wciąż czekają na
diagnozę, i diagnozuje je JEDNYM wywołaniem LLM (wzorzec `WeeklyRecapUseCase`):
prompt wylicza pudła po numerach, odpowiedź daje diagnozę per numer, a każda
trafia do swojego wiersza (`save_backfilled_diagnosis` podmienia tylko placeholder,
który job przeczytał). Pudło bez wpisu w odpowiedzi zostaje z placeholderem — po
tygodniu przejmuje je spłata długu diagnoz (WIA-53).

FinOps (FIN-30, wyjątek od bramki volatility): `PAID_LLM_CALLS_PER_RUN` = 1 w
tygodniowym budżecie Slow Loopa; `BudgetedLLMPort` odrzuca wywołanie ponad przydział.

Szablon promptu żyje tutaj, nie w `prompts.py`: bajty modułów promptów predykcji
wchodzą do `prompt_sha256` przepisu, a ta diagnoza jest jobem Slow Loopa — przepis
zmienia sama flaga (pole RECIPE), nie kod tego modułu. Nagłówki i uzasadnienie
pochodzą z newsów stron trzecich, więc idą przez `fence_untrusted`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any

from src.application._prompt_safety import fence_untrusted
from src.application.errors import PortError
from src.application.metering import PaidBudgetExhausted
from src.application.ports import DeferredDiagnosis, LLMPort, RepositoryPort
from src.domain.mistake_taxonomy import (
    MistakeCategory,
    MistakeDiagnosis,
    parse_batch_diagnoses,
)

logger = logging.getLogger(__name__)

# FIN-30: płatne wywołania LLM jednego przebiegu — wkład joba do budżetu tygodnia.
PAID_LLM_CALLS_PER_RUN = 1
BATCH_DIAGNOSIS_DAYS = 7
# Górna granica pudeł w jednym prompcie (długość promptu); nadmiar czeka na WIA-53.
MAX_BATCH_MISSES = 30
# Limit jednego nagłówka albo uzasadnienia w fence'u — 30 pudeł mieści się w promptcie.
_ITEM_MAX_LEN = 240
_CATEGORIES = " | ".join(category.value for category in MistakeCategory)


def batch_diagnosis_prompt(misses: Sequence[DeferredDiagnosis]) -> str:
    """Prompt diagnozy zbiorczej: pudła tygodnia po numerach, odpowiedź JSON per numer."""
    blocks = [_miss_block(position, miss) for position, miss in enumerate(misses, start=1)]
    listing = "\n\n".join(blocks)
    return f"""
Jesteś analitykiem korygującym. W tym tygodniu {len(misses)} prognoz kierunku okazało się
błędnych. Dla każdej, po jej numerze, zidentyfikuj, dlaczego dane wprowadziły Cię w błąd
(np. zignorowany kontekst makro, news już w cenie, sentyment bez pokrycia w faktach), a
jeśli kilka błędów ma wspólną przyczynę, nazwij ją w każdym z nich.

{listing}

Zwróć czysty JSON (bez bloków markdown):
{{"diagnoses": [{{"n": 1, "category": "{_CATEGORIES}", "insight": "zwięźle, max 3 zdania"}}]}}
Jeden wpis na każdy numer od 1 do {len(misses)}. Kategoria przyczyny: MACRO = makro, stopy,
dane gospodarcze; EARNINGS = wyniki lub raport spółki; STALE_NEWS = newsy już w cenie albo
nieaktualne; SECTOR_ROTATION = przepływ kapitału między sektorami; SENTIMENT_TRAP =
sentyment bez pokrycia w faktach; NOISE = ruch bez wyraźnej przyczyny; OTHER = nic z tych.
""".strip()


def _miss_block(position: int, miss: DeferredDiagnosis) -> str:
    prediction = miss.prediction
    headlines = [h for h in (prediction.news_summary or "").split(" | ") if h.strip()]
    items = list(headlines)
    if (prediction.reasoning or "").strip():
        items.append(f"Uzasadnienie: {prediction.reasoning}")
    saved_input = (
        fence_untrusted(f"MISS_{position}", items, max_len=_ITEM_MAX_LEN)
        if items
        else "Brak zapisanych newsów i uzasadnienia."
    )
    return (
        f"[{position}] {prediction.symbol}: prognoza {prediction.predicted_trend}, "
        f"wejście {prediction.price_at_prediction}, cel {prediction.predicted_target_price}, "
        f"rozliczenie {miss.actual_price}.\n{saved_input}"
    )


class BatchDiagnosisUseCase:
    """Jedno wywołanie LLM nad pudłami ostatnich `days` dni bez diagnozy.

    0 pudeł → 0 wywołań. Wyczerpany przydział albo błąd LLM zostawia wszystkie
    placeholdery; błąd zapisu jednego wiersza nie zatrzymuje pozostałych."""

    def __init__(
        self,
        repository_port: RepositoryPort,
        llm_port: LLMPort,
        *,
        clock: Callable[[], datetime],
        days: int = BATCH_DIAGNOSIS_DAYS,
        max_misses: int = MAX_BATCH_MISSES,
    ) -> None:
        self._repo = repository_port
        self._llm = llm_port
        self._clock = clock
        self._days = days
        self._max_misses = max_misses

    def run(self) -> dict[str, Any]:
        misses = self._week_misses()
        if not misses:
            return {"status": "skipped", "reason": "Brak pudeł tygodnia bez diagnozy."}
        try:
            answer = self._llm.analyze_mistake(batch_diagnosis_prompt(misses))
        except PaidBudgetExhausted:
            logger.warning("Batch diagnosis — przydział wyczerpany, placeholdery zostają")
            return {"status": "budget_exhausted", "misses": len(misses)}
        except (PortError, RuntimeError):
            logger.warning("Batch diagnosis — LLM failed, placeholdery zostają", exc_info=True)
            return {"status": "llm_failed", "misses": len(misses)}
        diagnoses = parse_batch_diagnoses(answer, len(misses))
        diagnosed = sum(
            self._save(miss, diagnosis)
            for miss, diagnosis in zip(misses, diagnoses, strict=True)
            if diagnosis is not None
        )
        return {"status": "diagnosed", "misses": len(misses), "diagnosed": diagnosed}

    def _week_misses(self) -> list[DeferredDiagnosis]:
        since = self._clock() - timedelta(days=self._days)
        return [
            row
            for row in self._repo.get_deferred_diagnoses(limit=self._max_misses)
            if row.is_trend_correct is False
            and row.prediction.id is not None
            and row.predicted_at is not None
            and row.predicted_at >= since
        ]

    def _save(self, miss: DeferredDiagnosis, diagnosis: MistakeDiagnosis) -> bool:
        prediction_id = miss.prediction.id
        if prediction_id is None:
            return False
        try:
            self._repo.save_backfilled_diagnosis(
                prediction_id,
                diagnosis.insight,
                replaces=miss.placeholder,
                correction_category=diagnosis.category,
            )
        except PortError:
            logger.warning(
                "Batch diagnosis — zapis nie wyszedł dla %s", prediction_id, exc_info=True
            )
            return False
        return True
