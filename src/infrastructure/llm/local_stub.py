# src/infrastructure/llm/local_stub.py
"""FIN-35: atrapa LLM dla lokalnego runu bez płatnych wywołań (PAID_CALLS_ENABLED=false)."""

from __future__ import annotations

from typing import Any

from src.application.ports import LLMPort

_STUB_REASONING = (
    "Lokalny przebieg bez płatnych wywołań (PAID_CALLS_ENABLED=false) — atrapa LLM, nie analiza."
)
_STUB_INSIGHT = (
    "Lokalny przebieg bez płatnych wywołań (PAID_CALLS_ENABLED=false) — diagnoza LLM nie powstała."
)


class LocalStubLLMAdapter(LLMPort):
    """Deterministyczny LLMPort bez I/O: neutralna analiza „SIDEWAYS 50%” i stała
    diagnoza, obie z jawną adnotacją, że to atrapa. Brak klucza `narrative`, więc
    recap tygodniowy idzie ścieżką bez narracji. W przeciwieństwie do
    `UnavailableLLMAdapter` nie rzuca — lokalny cykl przechodzi cały graf."""

    def analyze(self, prompt: str) -> dict[str, Any]:
        return {
            "trend_direction": "SIDEWAYS",
            "confidence_score": 0.5,
            "av_agreement": 0.5,
            "reasoning": _STUB_REASONING,
        }

    def analyze_mistake(self, prompt: str) -> str:
        return _STUB_INSIGHT
