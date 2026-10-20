"""NIEZ-44: drugi dostawca LLM po wyczerpaniu retry głównego.

Bez failovera awaria dostawcy po retry kończy się neutralnym fallbackiem grafu
(SIDEWAYS / 0.5) dla każdego symbolu za bramką: cykl płaci za newsy i embeddingi,
a predykcji nie ma. Kompozyt (wzorem `CascadingLLMAdapter`) przełącza na zapasowego
tylko po awarii dostawcy głównego — retry i alerty kwot robi już adapter głównego.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic
import openai

from src.application.ports import LLM_FAILOVER_MODEL_KEY, LLM_FAILOVER_PROVIDER_KEY, LLMPort

logger = logging.getLogger(__name__)

_RATE_LIMITED = 429
_SERVER_ERROR = 500
# Timeout i błąd połączenia obu SDK (`APITimeoutError` dziedziczy po
# `APIConnectionError`) oraz wbudowane — tak kończy się request, który nie dostał
# odpowiedzi dostawcy.
_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.APIConnectionError,
    openai.APIConnectionError,
    TimeoutError,
    ConnectionError,
)


def fails_over(exc: Exception) -> bool:
    """Czy błąd głównego to awaria dostawcy, która uzasadnia request u zapasowego.

    Decyzja właściciela (2026-10-20): przełączają wyłącznie 429, 5xx, timeout i błąd
    sieci/połączenia — każdy po retry adaptera głównego. Pusta albo zła odpowiedź
    JSON (`ValueError`) nie przełącza: dostawca odpowiedział, a graf kończy ją
    neutralnym fallbackiem. Błąd klienta 4xx poza 429 (zły prompt, zły klucz, brak
    modelu) też nie — drugi dostawca dostałby ten sam wadliwy request albo maskowałby
    błąd konfiguracji. `status_code` mają błędy HTTP obu SDK."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == _RATE_LIMITED or status >= _SERVER_ERROR
    return isinstance(exc, _TRANSPORT_ERRORS)


class FailoverLLMAdapter(LLMPort):
    """`LLMPort` z zapasowym dostawcą: `secondary` dostaje request wyłącznie po
    awarii `primary`, którą rozpoznaje `fails_over`.

    `provider` / `model` to tożsamość zapasowego — `analyze` dopisuje ją do wyniku
    (`LLM_FAILOVER_*_KEY`), żeby predykcja niosła flagę jakości i faktyczny przepis.
    `analyze_mistake` zwraca sam tekst, więc diagnoza od zapasowego nie ma znacznika
    (to nie predykcja; jej request liczy meter wokół `secondary`)."""

    def __init__(self, primary: LLMPort, secondary: LLMPort, *, provider: str, model: str) -> None:
        self._primary = primary
        self._secondary = secondary
        self._provider = provider
        self._model = model

    def analyze(self, prompt: str) -> dict[str, Any]:
        try:
            return self._primary.analyze(prompt)
        except Exception as exc:
            if not fails_over(exc):
                raise
            self._log_failover("analyze", exc)
        result = self._secondary.analyze(prompt)
        return {
            **result,
            LLM_FAILOVER_PROVIDER_KEY: self._provider,
            LLM_FAILOVER_MODEL_KEY: self._model,
        }

    def analyze_mistake(self, prompt: str) -> str:
        try:
            return self._primary.analyze_mistake(prompt)
        except Exception as exc:
            if not fails_over(exc):
                raise
            self._log_failover("analyze_mistake", exc)
        return self._secondary.analyze_mistake(prompt)

    def _log_failover(self, method: str, exc: Exception) -> None:
        # Sam typ błędu: komunikat SDK potrafi nieść URL albo fragment odpowiedzi.
        logger.warning(
            "LLM failover: primary %s failed (%s) — retrying on %s/%s",
            method,
            type(exc).__name__,
            self._provider,
            self._model,
        )
