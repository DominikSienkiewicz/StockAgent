from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from anthropic import Anthropic

from src.application.ports import Tool, ToolUseLLMPort
from src.application.quota_monitor import QuotaMonitor
from src.infrastructure.llm.anthropic_client import (
    DEFAULT_TIMEOUT,
    AnthropicAdapter,
    validate_model_id,
)

# #6 — domyślny model do pętli tool-use (taki sam kompromis kosztów/jakości co
# zwykły AnthropicAdapter). Tool-use to wieloetapowa, PŁATNA ścieżka — dlatego
# liczba rund jest twardo ograniczona (`max_rounds`).
DEFAULT_MODEL = "claude-sonnet-4-6"
# Strukturalny JSON werdyktu potrzebuje kilkuset tokenów; ciasny cap tnie koszt.
DEFAULT_MAX_TOKENS = 768
# FinOps: twardy limit rund tool-use. Każda runda to dodatkowy round-trip do
# płatnego modelu — bez capa rozbiegany model mógłby pętlić w nieskończoność.
DEFAULT_MAX_ROUNDS = 3

logger = logging.getLogger(__name__)


class ToolCallingAnthropicAdapter(ToolUseLLMPort):
    """#6 — adapter Anthropic z pętlą tool-use (Messages API tools).

    Model może na żądanie wywołać read-only narzędzia (fundamenty, RAG, makro)
    zanim wyda finalny werdykt JSON. Pętla:
      1. Wysyła prompt + zmapowane narzędzia (Tool → tools schema z
         `input_schema`).
      2. Dopóki `stop_reason == "tool_use"` i jest budżet rund: dokłada turę
         asystenta (z blokami tool_use), wykonuje `tool.func(input)` dla
         każdego bloku (wyjątek per-tool → wynik `{"error": ...}`, pętla nie
         pada), dokłada turę user z blokami tool_result, woła ponownie.
      3. Gdy model wyda finalny tekst LUB wyczerpie `max_rounds` (wtedy wymusza
         finalną odpowiedź BEZ narzędzi) — parsuje JSON.

    Parsowanie finalnego tekstu reużywa robustnej logiki `AnthropicAdapter`
    (rozpakowanie ```json``` + wyłuskanie pierwszego zbalansowanego {...}),
    bo Claude nie ma natywnego JSON Mode. `max_rounds` to twarda bramka FinOps.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        quota_monitor: QuotaMonitor | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        # Fail-fast na oczywiście-złym id modelu (pusty / zły provider prefix).
        validate_model_id(model)
        self._client = Anthropic(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_rounds = max_rounds
        self._quota_monitor = quota_monitor
        self._max_tokens = max_tokens

    @staticmethod
    def _map_tools(tools: Sequence[Tool]) -> list[dict[str, Any]]:
        """Tool (port) → tools schema Anthropic Messages API."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    @staticmethod
    def _execute_tool(tool: Tool, tool_input: Any) -> dict[str, Any]:
        """Wykonuje `tool.func` na wejściu od modelu.

        Każdy wyjątek funkcji zamienia na `{"error": ...}` — pętla nigdy nie
        pada przez pojedyncze narzędzie."""
        args = tool_input if isinstance(tool_input, dict) else {}
        try:
            return tool.func(args)
        except Exception as exc:  # noqa: BLE001 — celowo łapiemy wszystko
            logger.warning("Tool %s raised: %s", tool.name, exc)
            return {"error": str(exc)}

    def analyze_with_tools(
        self, prompt: str, tools: Sequence[Tool]
    ) -> dict[str, Any]:
        tools_by_name = {tool.name: tool for tool in tools}
        tool_schema = self._map_tools(tools)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        rounds = 0
        while True:
            # Po wyczerpaniu budżetu rund wymuszamy finalną odpowiedź: nie
            # przekazujemy narzędzi, więc model musi odpowiedzieć tekstem.
            offer_tools = rounds < self._max_rounds
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": messages,
            }
            if offer_tools:
                kwargs["tools"] = tool_schema

            response = self._client.messages.create(**kwargs)

            if not offer_tools or response.stop_reason != "tool_use":
                # Finalna odpowiedź (albo wymuszona po capie) — parsuj JSON.
                return self._parse_final(response.content)

            # Model prosi o narzędzia — dołóż jego turę (cały content, w tym
            # bloki tool_use), potem wykonaj każdy blok tool_use.
            messages.append(
                {"role": "assistant", "content": response.content}
            )
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                tool = tools_by_name.get(block.name)
                if tool is None:
                    result: dict[str, Any] = {
                        "error": f"unknown tool: {block.name}"
                    }
                else:
                    result = self._execute_tool(tool, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            rounds += 1

    @staticmethod
    def _extract_text(content: Any) -> str:
        for block in content or []:
            if getattr(block, "type", None) == "text":
                return str(block.text)
        return ""

    def _parse_final(self, content: Any) -> dict[str, Any]:
        text = self._extract_text(content)
        if not text:
            raise ValueError(
                "Anthropic tool-use returned empty content for the final answer."
            )
        # Reuse robustnego parsowania z AnthropicAdapter — Claude nie ma JSON
        # Mode, więc owija JSON w ```json``` albo prozę.
        cleaned = AnthropicAdapter._strip_code_block(text).strip()
        try:
            return json.loads(cleaned)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            candidate = AnthropicAdapter._extract_first_json_object(cleaned)
            if candidate is not None:
                try:
                    return json.loads(candidate)  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                f"Anthropic tool-use returned invalid JSON: {cleaned[:200]!r}"
            ) from None
