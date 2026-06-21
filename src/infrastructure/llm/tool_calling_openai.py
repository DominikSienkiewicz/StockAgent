from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from src.application.ports import Tool, ToolUseLLMPort
from src.application.quota_monitor import QuotaMonitor
from src.infrastructure.llm.openai_client import (
    DEFAULT_TIMEOUT,
    validate_model_id,
)

# #6 — tani, deterministyczny model do pętli tool-use (taki sam default jak
# zwykły OpenAIAdapter). Tool-use to wieloetapowa, PŁATNA ścieżka — dlatego
# domyślny model jest tani, a liczba rund twardo ograniczona (`max_rounds`).
DEFAULT_MODEL = "gpt-4o"
# Strukturalny JSON werdyktu potrzebuje kilkuset tokenów; ciasny cap tnie koszt.
DEFAULT_MAX_TOKENS = 768
# FinOps: twardy limit rund tool-use. Każda runda to dodatkowy round-trip do
# płatnego modelu — bez capa rozbiegany model mógłby pętlić w nieskończoność.
DEFAULT_MAX_ROUNDS = 3

logger = logging.getLogger(__name__)


class ToolCallingOpenAIAdapter(ToolUseLLMPort):
    """#6 — adapter OpenAI z pętlą tool-use (function calling).

    Model może na żądanie wywołać read-only narzędzia (fundamenty, RAG, makro)
    zanim wyda finalny werdykt JSON. Pętla:
      1. Wysyła prompt + zmapowane narzędzia (Tool → function schema).
      2. Dopóki model prosi o tool-calls i jest budżet rund: wykonuje
         `tool.func(args)` dla każdego żądania (wyjątek per-tool → wynik
         `{"error": ...}`, pętla nie pada), dokłada wyniki, woła ponownie.
      3. Gdy model wyda finalny tekst LUB wyczerpie `max_rounds` (wtedy
         wymusza finalną odpowiedź bez narzędzi) — parsuje JSON.

    `max_rounds` to twarda bramka FinOps — bez niej rozbiegany model pętliłby
    płatne wywołania bez końca.
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
        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_rounds = max_rounds
        self._quota_monitor = quota_monitor
        self._max_tokens = max_tokens

    @staticmethod
    def _map_tools(tools: Sequence[Tool]) -> list[dict[str, Any]]:
        """Tool (port) → function schema OpenAI Chat Completions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _execute_tool(tool: Tool, raw_args: str) -> dict[str, Any]:
        """Wykonuje `tool.func` na rozparsowanych argumentach.

        Każdy błąd (złe JSON-argumenty, wyjątek funkcji) zamienia na
        `{"error": ...}` — pętla nigdy nie pada przez pojedyncze narzędzie."""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            return {"error": f"invalid tool arguments: {exc}"}
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
            # Po wyczerpaniu budżetu rund wymuszamy finalną odpowiedź:
            # nie przekazujemy narzędzi, więc model musi odpowiedzieć tekstem.
            offer_tools = rounds < self._max_rounds
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": self._max_tokens,
            }
            if offer_tools:
                kwargs["tools"] = tool_schema

            response = self._client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            if not offer_tools or not tool_calls:
                # Finalna odpowiedź (albo wymuszona po capie) — parsuj JSON.
                return self._parse_final(message.content)

            # Model prosi o narzędzia — dołóż jego turę i wykonaj każde żądanie.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )
            for call in tool_calls:
                tool = tools_by_name.get(call.function.name)
                if tool is None:
                    result: dict[str, Any] = {
                        "error": f"unknown tool: {call.function.name}"
                    }
                else:
                    result = self._execute_tool(tool, call.function.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
            rounds += 1

    @staticmethod
    def _parse_final(content: str | None) -> dict[str, Any]:
        if not content:
            raise ValueError(
                "OpenAI tool-use returned empty content for the final answer."
            )
        try:
            return json.loads(content)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"OpenAI tool-use returned invalid JSON: {content[:200]!r}"
            ) from exc
