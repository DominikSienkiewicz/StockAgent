# src/infrastructure/llm/unavailable.py
from __future__ import annotations

from typing import Any

from src.application.ports import LLMPort


class UnavailableLLMAdapter(LLMPort):
    """LLMPort zastępczy, gdy prawdziwego adaptera nie dało się zbudować (np. brak
    SDK albo klucza). Każde wywołanie rzuca RuntimeError, więc use case'y z
    deterministycznym fallbackiem (recap bez narracji, kalibracja bez wniosku)
    idą swoją ścieżką awaryjną, zamiast znikać razem z błędem budowy."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def analyze(self, prompt: str) -> dict[str, Any]:
        raise RuntimeError(f"LLM adapter unavailable: {self._reason}")

    def analyze_mistake(self, prompt: str) -> str:
        raise RuntimeError(f"LLM adapter unavailable: {self._reason}")
