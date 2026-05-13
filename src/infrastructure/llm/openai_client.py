from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.application.ports import LLMPort

DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.2  # niska temperatura — chcemy deterministycznego Quanta


class OpenAIAdapter(LLMPort):
    """Adapter dla OpenAI Chat Completions API (SDK v2+).

    `analyze` używa JSON Mode — wymusza ustrukturyzowany output.
    `analyze_mistake` zwraca free-form tekst (correction insight).

    Przełączenie na Anthropic = podmiana tej klasy w DI Containerze
    (main_agent.py). Reszta systemu nie wie, że jesteśmy na OpenAI.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def analyze(self, prompt: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=self._temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned empty content for analyze().")

        try:
            return json.loads(content)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"OpenAI returned invalid JSON: {content[:200]!r}"
            ) from exc

    def analyze_mistake(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content.strip()
