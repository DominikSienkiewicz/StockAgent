import json
from unittest.mock import MagicMock

import pytest

from src.application.ports import LLMPort
from src.infrastructure.llm.openai_client import OpenAIAdapter


def _completion(content: str) -> MagicMock:
    """Tworzy mock odpowiedzi OpenAI Chat Completions w nowym SDK (v2+).

    Struktura: response.choices[0].message.content
    """
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


@pytest.fixture
def mock_openai_class(mocker):
    """Mockuje samą klasę OpenAI w module adaptera — konstruktor zwraca MagicMock."""
    return mocker.patch("src.infrastructure.llm.openai_client.OpenAI")


@pytest.fixture
def adapter_with_client(mock_openai_class) -> tuple[OpenAIAdapter, MagicMock]:
    """Zwraca (adapter, mock_client) — client to to, co realnie wywołują metody."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o")
    return adapter, mock_client


class TestAnalyze:
    def test_returns_parsed_json_dict(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion(
            json.dumps({
                "trend_direction": "BULLISH",
                "confidence_score": 0.85,
                "target_price_12h": 110.0,
                "reasoning": "Strong macro tailwinds.",
            })
        )

        result = adapter.analyze("Predict AAPL trend.")

        assert isinstance(result, dict)
        assert result["trend_direction"] == "BULLISH"
        assert result["confidence_score"] == 0.85
        assert result["target_price_12h"] == 110.0

    def test_uses_json_response_format(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion("{}")

        adapter.analyze("Predict.")

        call = client.chat.completions.create.call_args
        assert call.kwargs["response_format"] == {"type": "json_object"}

    def test_passes_model_and_prompt(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion("{}")

        adapter.analyze("Custom prompt")

        call = client.chat.completions.create.call_args
        assert call.kwargs["model"] == "gpt-4o"
        messages = call.kwargs["messages"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Custom prompt"

    def test_raises_value_error_on_invalid_json(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion("not a json")

        with pytest.raises(ValueError, match="JSON"):
            adapter.analyze("Predict.")

    def test_raises_value_error_when_content_empty(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion(None)

        with pytest.raises(ValueError, match="empty"):
            adapter.analyze("Predict.")


class TestAnalyzeMistake:
    def test_returns_plain_text_content(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion(
            "Zignorowałeś szerszy kontekst makro — Fed podniósł stopy."
        )

        result = adapter.analyze_mistake("Diagnose mistake.")

        assert result == "Zignorowałeś szerszy kontekst makro — Fed podniósł stopy."

    def test_does_not_use_json_format_for_freeform_text(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion("free form")

        adapter.analyze_mistake("Diagnose.")

        call = client.chat.completions.create.call_args
        # Plain text — bez wymuszania JSON formatu
        assert call.kwargs.get("response_format") != {"type": "json_object"}

    def test_strips_whitespace_from_response(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion("  insight text  \n")

        result = adapter.analyze_mistake("Diagnose.")

        assert result == "insight text"

    def test_returns_empty_string_when_content_is_none(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion(None)

        result = adapter.analyze_mistake("Diagnose.")

        assert result == ""


class TestConfiguration:
    def test_custom_model_is_passed_to_api(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _completion("{}")
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")

        adapter.analyze("Predict.")

        assert mock_client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"

    def test_initializes_openai_client_with_api_key(self, mock_openai_class):
        OpenAIAdapter(api_key="sk-secret-123")
        mock_openai_class.assert_called_once_with(api_key="sk-secret-123")


class TestAdapterImplementsPort:
    def test_is_llm_port(self):
        assert issubclass(OpenAIAdapter, LLMPort)


@pytest.mark.integration
class TestOpenAILive:
    def test_real_api_returns_json(self):
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        adapter = OpenAIAdapter(api_key=api_key, model="gpt-4o-mini")
        result = adapter.analyze(
            'Return JSON: {"answer": 42}. Only JSON, nothing else.'
        )
        assert isinstance(result, dict)
