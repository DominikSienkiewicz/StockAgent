"""NIEZ-44 (c): testy kontraktowe — oba adaptery LLM parsują tę samą odpowiedź tak samo.

Failover przełącza analizę między dostawcami w środku cyklu, więc ten sam tekst
odpowiedzi musi dać ten sam dict u OpenAI i u Anthropic — inaczej predykcja od
zapasowego różniłaby się samym parserem. Każdy fixture przechodzi przez oba
adaptery z zamockowanym SDK (zero sieci).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from src.application.ports import LLMPort
from src.infrastructure.llm.anthropic_client import AnthropicAdapter
from src.infrastructure.llm.openai_client import OpenAIAdapter

_ANALYSIS = {
    "trend_direction": "BULLISH",
    "confidence_score": 0.82,
    "av_agreement": 0.4,
    "target_price_12h": 112.5,
    "reasoning": "Guidance raised; sentiment confirms.",
}

FIXTURES = {
    "plain_json": json.dumps(_ANALYSIS),
    "fenced_json": "```json\n" + json.dumps(_ANALYSIS, indent=2) + "\n```",
    "bare_fence": "```" + json.dumps(_ANALYSIS) + "```",
}


def _openai(content: str | None) -> LLMPort:
    with patch("src.infrastructure.llm.openai_client.OpenAI") as client_class:
        client = MagicMock()
        client_class.return_value = client
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o")
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].finish_reason = "stop"
    client.chat.completions.create.return_value = response
    return adapter


def _anthropic(content: str | None) -> LLMPort:
    with patch("src.infrastructure.llm.anthropic_client.Anthropic") as client_class:
        client = MagicMock()
        client_class.return_value = client
        adapter = AnthropicAdapter(api_key="sk-ant-test", model="claude-sonnet-4-6")
    response = MagicMock()
    if content is None:
        response.content = []
    else:
        block = MagicMock()
        block.type = "text"
        block.text = content
        response.content = [block]
    client.messages.create.return_value = response
    return adapter


ADAPTERS: dict[str, Callable[[str | None], LLMPort]] = {
    "openai": _openai,
    "anthropic": _anthropic,
}


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
@pytest.mark.parametrize("provider", sorted(ADAPTERS))
def test_adapters_parse_same_fixture_identically(provider: str, fixture: str) -> None:
    adapter = ADAPTERS[provider](FIXTURES[fixture])

    assert adapter.analyze("prompt") == _ANALYSIS


@pytest.mark.parametrize("content", [None, ""])
@pytest.mark.parametrize("provider", sorted(ADAPTERS))
def test_adapters_reject_empty_content_the_same_way(provider: str, content: str | None) -> None:
    adapter = ADAPTERS[provider](content)

    with pytest.raises(ValueError, match="empty"):
        adapter.analyze("prompt")


@pytest.mark.parametrize("provider", sorted(ADAPTERS))
def test_adapters_reject_prose_without_json_the_same_way(provider: str) -> None:
    adapter = ADAPTERS[provider]("I cannot answer that.")

    with pytest.raises(ValueError, match="JSON"):
        adapter.analyze("prompt")
