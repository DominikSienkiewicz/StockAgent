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

    def test_extracts_json_from_prose_wrapped_response(self, adapter_with_client):
        # Claude nie ma natywnego JSON Mode — czasem owija JSON w prozę.
        # Adapter musi wyłuskać pierwszy zbalansowany obiekt {...}.
        adapter, client = adapter_with_client
        prose = (
            'Here is the analysis: '
            '{"trend_direction":"BULLISH","confidence_score":0.8} '
            'hope this helps'
        )
        client.messages.create.return_value = _message_response(prose)

        result = adapter.analyze("Predict.")

        assert result == {
            "trend_direction": "BULLISH",
            "confidence_score": 0.8,
        }

    def test_ignores_trailing_commentary_after_object(self, adapter_with_client):
        # Tekst po zamykającym nawiasie nie może wywalić parsowania.
        adapter, client = adapter_with_client
        text = '{"trend_direction": "NEUTRAL"}\n\nLet me know if you need more.'
        client.messages.create.return_value = _message_response(text)

        result = adapter.analyze("Predict.")

        assert result["trend_direction"] == "NEUTRAL"

    def test_brace_inside_string_does_not_break_extraction(
        self, adapter_with_client
    ):
        # Nawias klamrowy w stringu nie może mylić skanera balansu.
        adapter, client = adapter_with_client
        text = (
            'Result: {"reasoning": "earnings up }{ noise", '
            '"trend_direction": "BULLISH"} done'
        )
        client.messages.create.return_value = _message_response(text)

        result = adapter.analyze("Predict.")

        assert result["trend_direction"] == "BULLISH"
        assert result["reasoning"] == "earnings up }{ noise"

    def test_raises_value_error_when_no_json_object_present(
        self, adapter_with_client
    ):
        # Genuinie nie-JSON (brak zbalansowanego obiektu) nadal musi failować.
        adapter, client = adapter_with_client
        client.messages.create.return_value = _message_response(
            "I cannot analyze this asset right now."
        )

        with pytest.raises(ValueError, match="JSON"):
            adapter.analyze("Predict.")


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


def _status_error(status: int, retry_after: str | None = None):
    """Buduje anthropic.APIStatusError (lub podklasę) z realnym httpx.Response."""
    import httpx
    from anthropic import APIStatusError, InternalServerError, RateLimitError

    headers = {}
    if retry_after is not None:
        headers["retry-after"] = retry_after
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, headers=headers, request=request)
    if status == 429:
        return RateLimitError("rate limited", response=response, body=None)
    if status >= 500:
        return InternalServerError("overloaded", response=response, body=None)
    return APIStatusError("client error", response=response, body=None)


class TestRetryAndQuota:
    """Główny LLM (domyślnie Anthropic) MUSI mieć tę samą odporność co OpenAI:
    retry na 429/5xx z backoffem + sygnalizacja do QuotaMonitor."""

    def test_retries_on_rate_limit_then_succeeds(
        self, mock_anthropic_class, mocker
    ):
        mocker.patch("src.infrastructure.llm.anthropic_client.time.sleep")
        client = MagicMock()
        mock_anthropic_class.return_value = client
        client.messages.create.side_effect = [
            _status_error(429),
            _message_response("{}"),
        ]
        adapter = AnthropicAdapter(api_key="sk-ant-test", backoff_base=0.01)

        result = adapter.analyze("Predict.")

        assert result == {}
        assert client.messages.create.call_count == 2

    def test_retries_on_server_error_5xx(self, mock_anthropic_class, mocker):
        mocker.patch("src.infrastructure.llm.anthropic_client.time.sleep")
        client = MagicMock()
        mock_anthropic_class.return_value = client
        client.messages.create.side_effect = [
            _status_error(529),  # overloaded
            _message_response('{"trend_direction": "BULLISH"}'),
        ]
        adapter = AnthropicAdapter(api_key="sk-ant-test", backoff_base=0.01)

        result = adapter.analyze("Predict.")

        assert result["trend_direction"] == "BULLISH"
        assert client.messages.create.call_count == 2

    def test_does_not_retry_on_client_4xx(self, mock_anthropic_class, mocker):
        from anthropic import APIStatusError

        mocker.patch("src.infrastructure.llm.anthropic_client.time.sleep")
        client = MagicMock()
        mock_anthropic_class.return_value = client
        client.messages.create.side_effect = _status_error(400)
        adapter = AnthropicAdapter(api_key="sk-ant-test", backoff_base=0.01)

        with pytest.raises(APIStatusError):
            adapter.analyze("Predict.")
        # 400 nie jest transient → żadnego retry
        assert client.messages.create.call_count == 1

    def test_warning_alert_emitted_after_successful_retry(
        self, mock_anthropic_class, mocker
    ):
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity

        mocker.patch("src.infrastructure.llm.anthropic_client.time.sleep")
        client = MagicMock()
        mock_anthropic_class.return_value = client
        client.messages.create.side_effect = [
            _status_error(429),
            _message_response("{}"),
        ]
        monitor = QuotaMonitor()
        adapter = AnthropicAdapter(
            api_key="sk-ant-test", backoff_base=0.01, quota_monitor=monitor
        )

        adapter.analyze("Predict.")

        assert len(monitor.alerts) == 1
        alert = monitor.alerts[0]
        assert alert.severity is QuotaSeverity.WARNING
        assert "Anthropic" in alert.source

    def test_critical_alert_and_raise_when_retries_exhausted(
        self, mock_anthropic_class, mocker
    ):
        from anthropic import APIStatusError

        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity

        mocker.patch("src.infrastructure.llm.anthropic_client.time.sleep")
        client = MagicMock()
        mock_anthropic_class.return_value = client
        client.messages.create.side_effect = _status_error(529)
        monitor = QuotaMonitor()
        adapter = AnthropicAdapter(
            api_key="sk-ant-test",
            backoff_base=0.01,
            max_retries=2,
            quota_monitor=monitor,
        )

        with pytest.raises(APIStatusError):
            adapter.analyze("Predict.")

        assert client.messages.create.call_count == 3  # 1 + 2 retries
        assert monitor.max_severity() is QuotaSeverity.CRITICAL

    def test_honors_retry_after_header(self, mock_anthropic_class, mocker):
        sleep = mocker.patch("src.infrastructure.llm.anthropic_client.time.sleep")
        client = MagicMock()
        mock_anthropic_class.return_value = client
        client.messages.create.side_effect = [
            _status_error(429, retry_after="7"),
            _message_response("{}"),
        ]
        adapter = AnthropicAdapter(api_key="sk-ant-test", backoff_base=0.01)

        adapter.analyze("Predict.")

        sleep.assert_called_once_with(7.0)


class TestModelIdValidation:
    """#30 — Hardcoded model id bez guardu = cichy total outage (pusty/typo'd
    config failuje dopiero przy 1. callu, per symbol, co cykl). Walidacja w
    __init__ łapie oczywiście-zły id natychmiast, z czytelnym komunikatem."""

    def test_empty_model_id_raises_at_construction(self, mock_anthropic_class):
        with pytest.raises(ValueError, match="model"):
            AnthropicAdapter(api_key="sk-ant-test", model="")

    def test_whitespace_model_id_raises_at_construction(self, mock_anthropic_class):
        with pytest.raises(ValueError, match="model"):
            AnthropicAdapter(api_key="sk-ant-test", model="   ")

    def test_wrong_prefix_model_id_raises_at_construction(self, mock_anthropic_class):
        # id bez prefiksu "claude-" jest oczywiście błędny dla Anthropic.
        with pytest.raises(ValueError, match="claude-"):
            AnthropicAdapter(api_key="sk-ant-test", model="gpt-4o")

    def test_valid_model_id_constructs_fine(self, mock_anthropic_class):
        adapter = AnthropicAdapter(api_key="sk-ant-test", model="claude-opus-4-7")
        assert adapter is not None

    def test_default_model_id_constructs_fine(self, mock_anthropic_class):
        # Domyślny model id musi przejść własną walidację.
        adapter = AnthropicAdapter(api_key="sk-ant-test")
        assert adapter is not None


class TestMaxTokensCap:
    """#31 — Schematy JSON potrzebują kilkuset tokenów; uniformny 4096 (w tym
    per-investor council calls) to przeszacowanie. Domyślny cap ~768 tnie koszt
    wyjścia, pozostaje nadpisywalny przez konstruktor."""

    def test_default_max_tokens_is_tight(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.messages.create.return_value = _message_response("{}")

        adapter.analyze("Predict.")

        # Cap dopasowany do schematu JSON (kilkaset tokenów), nie 4096.
        assert client.messages.create.call_args.kwargs["max_tokens"] <= 1024

    def test_max_tokens_overridable_via_constructor(self, mock_anthropic_class):
        client = MagicMock()
        mock_anthropic_class.return_value = client
        client.messages.create.return_value = _message_response("{}")
        adapter = AnthropicAdapter(api_key="sk-ant-test", max_tokens=2048)

        adapter.analyze("Predict.")

        assert client.messages.create.call_args.kwargs["max_tokens"] == 2048

    def test_analyze_mistake_also_has_bounded_cap(self, adapter_with_client):
        adapter, client = adapter_with_client
        client.messages.create.return_value = _message_response("insight")

        adapter.analyze_mistake("Diagnose.")

        cap = client.messages.create.call_args.kwargs["max_tokens"]
        assert 0 < cap <= 4096


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
