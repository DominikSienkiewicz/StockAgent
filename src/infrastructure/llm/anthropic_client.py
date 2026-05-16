from __future__ import annotations

import json
import re
from typing import Any

from anthropic import Anthropic

from src.application.ports import LLMPort

# Modele Claude 4.x (stan na maj 2026).
DEFAULT_MODEL = "claude-sonnet-4-6"   # dobre konto kosztów/jakości
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2
# GHA fast loop ma 15 min hard timeout; SDK domyślnie czeka 600s.
DEFAULT_TIMEOUT = 30.0
# Próg włączania ephemeral cache — Anthropic wymaga ≥1024 tok dla sonnet/opus.
# Liczymy ~4 znaki/token jako bezpieczne oszacowanie.
_CACHE_MIN_CHARS = 4096


_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class AnthropicAdapter(LLMPort):
    """Adapter dla Anthropic Messages API.

    W odróżnieniu od OpenAI, Claude nie ma natywnego JSON Mode — instruujemy
    go w prompcie (<output_schema>). Adapter rozpakowuje ewentualne wrappery
    ```json ... ``` z odpowiedzi.

    Przełączenie z OpenAI = `LLM_PROVIDER=anthropic` w `.env` + ANTHROPIC_API_KEY.

    Prompt caching: dla wystarczająco długich promptów (≥1024 tok ≈ 4096 znaków)
    włączamy ephemeral cache_control na content bloku. Cache działa per
    identyczny prefix — przy 2 cyklach dziennie ten sam template promptu
    daje cache hit na strukturalnej części (rola/instrukcje/schemat).
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._client = Anthropic(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @staticmethod
    def _build_user_content(prompt: str) -> Any:
        # Dla krótkich promptów cache_control byłby narzutem bez zysku
        # (Anthropic odrzuca cache bloki <1024 tok). Zwracamy raw string.
        # Typowane Any bo SDK używa wąskich TypedDict'ów (TextBlockParam),
        # których konstruowanie dictem dla cache_control jest poprawne runtime
        # ale niezgodne strukturalnie.
        if len(prompt) < _CACHE_MIN_CHARS:
            return prompt
        return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]

    def _call(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": self._build_user_content(prompt)}],
        )
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response: Any) -> str:
        for block in response.content or []:
            if getattr(block, "type", None) == "text":
                return str(block.text)
        return ""

    @staticmethod
    def _strip_code_block(text: str) -> str:
        match = _CODE_BLOCK_RE.search(text)
        return match.group(1) if match else text

    def analyze(self, prompt: str) -> dict[str, Any]:
        text = self._call(prompt)
        if not text:
            raise ValueError("Anthropic returned empty content for analyze().")
        cleaned = self._strip_code_block(text).strip()
        try:
            return json.loads(cleaned)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Anthropic returned invalid JSON: {cleaned[:200]!r}"
            ) from exc

    def analyze_mistake(self, prompt: str) -> str:
        return self._call(prompt).strip()
