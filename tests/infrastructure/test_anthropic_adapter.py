import json
from unittest.mock import MagicMock

import pytest

from src.application.ports import LLMPort
from src.infrastructure.llm.anthropic_client import AnthropicAdapter


def _message_response(text: str | None) -> MagicMock:
    """Mock odpowiedzi Anthropic Messages API.

    Struktura: response.content -> list[ContentBlock] (każdy z .type/.text).
    """
    response = MagicMock()
    if text is None:
        response.content = []
    else:
        block = MagicMock()
        block.type = "text"
        block.text = text
        response.content = [block]
    return response


@pytest.fixture
def mock_anthropic_class(mocker):
    return mocker.patch("src.infrastructure.llm.anthropic_client.Anthropic")


@pytest.fixture
def adapter_with_client(mock_anthropic_class) -> tuple[AnthropicAdapter, MagicMock]:
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    adapter = AnthropicAdapter(api_key="sk-ant-test", model="claude-sonnet-4-6")
    return adapter, mock_client


class TestAnalyze:
    def test_returns_parsed_json_dict(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.messages.create.return_value = _message_response(
            json.dumps({
                "trend_direction": "BULLISH",
                "confidence_score": 0.85,
                "target_price_12h": 110.0,
                "reasoning": "macro tailwinds",
            })
        )

        result = adapter.analyze("Predict AAPL.")

        assert result["trend_direction"] == "BULLISH"
        assert result["confidence_score"] == 0.85

    def test_passes_model_and_prompt(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.messages.create.return_value = _message_response("{}")

        adapter.analyze("Custom prompt")

        call = client.messages.create.call_args
        assert call.kwargs["model"] == "claude-sonnet-4-6"
        messages = call.kwargs["messages"]
        assert messages[-1] == {"role": "user", "content": "Custom prompt"}

    def test_sets_max_tokens(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.messages.create.return_value = _message_response("{}")

        adapter.analyze("Predict.")

        assert client.messages.create.call_args.kwargs["max_tokens"] > 0

    def test_raises_value_error_on_invalid_json(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.messages.create.return_value = _message_response("not json")

        with pytest.raises(ValueError, match="JSON"):
            adapter.analyze("Predict.")

    def test_raises_value_error_when_content_empty(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.messages.create.return_value = _message_response(None)

        with pytest.raises(ValueError, match="empty"):
            adapter.analyze("Predict.")

    def test_extracts_text_from_markdown_code_block(self, adapter_with_client):
        # Claude czasem owija JSON w ```json ... ``` — adapter musi sobie poradzić
        adapter, client = adapter_with_client
        wrapped = '```json\n{"trend_direction": "BEARISH"}\n```'
        client.messages.create.return_value = _message_response(wrapped)

        result = adapter.analyze("Predict.")

        assert result["trend_direction"] == "BEARISH"


class TestAnalyzeMistake:
    def test_returns_plain_text_content(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.messages.create.return_value = _message_response("ignored macro context")

        result = adapter.analyze_mistake("Diagnose.")

        assert result == "ignored macro context"

    def test_strips_whitespace(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.messages.create.return_value = _message_response("  insight  \n")

        result = adapter.analyze_mistake("Diagnose.")

        assert result == "insight"

    def test_returns_empty_string_when_no_content(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.messages.create.return_value = _message_response(None)

        result = adapter.analyze_mistake("Diagnose.")

        assert result == ""


class TestConfiguration:
    def test_custom_model_passed_to_api(self, mock_anthropic_class):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = _message_response("{}")
        adapter = AnthropicAdapter(api_key="sk-ant-test", model="claude-opus-4-7")

        adapter.analyze("Predict.")

        assert mock_client.messages.create.call_args.kwargs["model"] == "claude-opus-4-7"

    def test_initializes_client_with_api_key(self, mock_anthropic_class):
        AnthropicAdapter(api_key="sk-ant-secret")
        call = mock_anthropic_class.call_args
        assert call.kwargs["api_key"] == "sk-ant-secret"
        # GHA fast loop ma 15 min hard timeout — SDK musi mieć krótszy.
        assert call.kwargs["timeout"] > 0


class TestAdapterImplementsPort:
    def test_is_llm_port(self):
        assert issubclass(AnthropicAdapter, LLMPort)


@pytest.mark.integration
class TestAnthropicLive:
    def test_real_api_returns_json(self):
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        adapter = AnthropicAdapter(api_key=api_key)
        result = adapter.analyze(
            'Return ONLY JSON: {"answer": 42}. No markdown, just JSON.'
        )
        assert isinstance(result, dict)
