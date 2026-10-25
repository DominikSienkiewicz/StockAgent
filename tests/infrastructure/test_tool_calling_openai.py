import contextlib
import json
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from src.application.ports import Tool, ToolUseLLMPort
from src.infrastructure.llm.tool_calling_openai import (
    DEFAULT_MODEL,
    ToolCallingOpenAIAdapter,
)


def _final_message(content: str) -> MagicMock:
    """Mock odpowiedzi OpenAI bez tool-calls — model wydał finalny tekst.

    Struktura: response.choices[0].message.content / .tool_calls (None).
    """
    response = MagicMock()
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    response.choices = [MagicMock(message=message)]
    return response


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> MagicMock:
    """Pojedyncze żądanie wywołania funkcji (jak w `message.tool_calls`)."""
    call = MagicMock()
    call.id = call_id
    call.type = "function"
    call.function = MagicMock()
    call.function.name = name
    # OpenAI serializuje argumenty jako string JSON.
    call.function.arguments = json.dumps(arguments)
    return call


def _tool_request_message(tool_calls: list[MagicMock]) -> MagicMock:
    """Mock odpowiedzi, w której model prosi o wywołanie narzędzi.

    `message.content` bywa None gdy są tool_calls; `tool_calls` to lista.
    """
    response = MagicMock()
    message = MagicMock()
    message.content = None
    message.tool_calls = tool_calls
    response.choices = [MagicMock(message=message)]
    return response


@pytest.fixture
def mock_openai_class(mocker: Any) -> MagicMock:
    """Mockuje klasę OpenAI w module adaptera — konstruktor zwraca MagicMock."""
    return cast(MagicMock, mocker.patch("src.infrastructure.llm.tool_calling_openai.OpenAI"))


@pytest.fixture
def adapter_with_client(
    mock_openai_class: MagicMock,
) -> tuple[ToolCallingOpenAIAdapter, MagicMock]:
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    adapter = ToolCallingOpenAIAdapter(api_key="sk-test", model="gpt-4o")
    return adapter, mock_client


def _make_tool(func: Callable[[dict[str, Any]], dict[str, Any]]) -> Tool:
    return Tool(
        name="get_fundamentals",
        description="Zwraca P/E i EPS spółki.",
        parameters={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
        func=func,
    )


class TestImmediateAnswer:
    """(a) Model odpowiada od razu, bez tool-calls → parsed JSON, brak pętli."""

    def test_returns_parsed_json_without_calling_tools(
        self, adapter_with_client: tuple[ToolCallingOpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _final_message(
            json.dumps({"trend_direction": "BULLISH", "confidence_score": 0.9})
        )
        called = MagicMock(return_value={"pe": 20.0})
        tool = _make_tool(called)

        result = adapter.analyze_with_tools("Predict AAPL.", [tool])

        assert result == {"trend_direction": "BULLISH", "confidence_score": 0.9}
        # Zero rund tool-use ⇒ funkcja narzędzia nigdy nie wołana.
        called.assert_not_called()
        # Dokładnie jedno wywołanie API (sam prompt, bez round-tripów).
        assert client.chat.completions.create.call_count == 1

    def test_tools_mapped_to_function_schema(
        self, adapter_with_client: tuple[ToolCallingOpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _final_message("{}")
        tool = _make_tool(MagicMock(return_value={}))

        adapter.analyze_with_tools("Predict.", [tool])

        kwargs = client.chat.completions.create.call_args.kwargs
        tools_arg = kwargs["tools"]
        assert tools_arg[0]["type"] == "function"
        fn = tools_arg[0]["function"]
        assert fn["name"] == "get_fundamentals"
        assert fn["description"] == "Zwraca P/E i EPS spółki."
        assert fn["parameters"] == tool.parameters


class TestSingleToolRound:
    """(b) Model prosi o jeden tool, dostaje wynik, potem odpowiada."""

    def test_tool_func_invoked_with_parsed_args_then_final_json(
        self, adapter_with_client: tuple[ToolCallingOpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        tool_func = MagicMock(return_value={"pe": 18.5})
        tool = _make_tool(tool_func)
        client.chat.completions.create.side_effect = [
            _tool_request_message(
                [_tool_call("call_1", "get_fundamentals", {"symbol": "AAPL"})]
            ),
            _final_message(json.dumps({"trend_direction": "BEARISH"})),
        ]

        result = adapter.analyze_with_tools("Predict AAPL.", [tool])

        # Funkcja wywołana z rozparsowanymi argumentami (dict, nie string).
        tool_func.assert_called_once_with({"symbol": "AAPL"})
        assert result == {"trend_direction": "BEARISH"}
        assert client.chat.completions.create.call_count == 2

    def test_tool_result_fed_back_as_tool_message(
        self, adapter_with_client: tuple[ToolCallingOpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        tool = _make_tool(MagicMock(return_value={"pe": 18.5}))
        client.chat.completions.create.side_effect = [
            _tool_request_message(
                [_tool_call("call_42", "get_fundamentals", {"symbol": "AAPL"})]
            ),
            _final_message("{}"),
        ]

        adapter.analyze_with_tools("Predict.", [tool])

        # Drugie wywołanie API musi nieść wynik narzędzia jako role=tool.
        second_call = client.chat.completions.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_42"
        assert json.loads(tool_messages[0]["content"]) == {"pe": 18.5}


class TestMaxRoundsCap:
    """(c) FinOps — model w kółko prosi o tool; pętla zatrzymuje się po max_rounds."""

    def test_tool_calls_bounded_by_max_rounds(self, mock_openai_class: MagicMock) -> None:
        client = MagicMock()
        mock_openai_class.return_value = client
        tool_func = MagicMock(return_value={"pe": 1.0})
        tool = _make_tool(tool_func)
        # Model ZAWSZE prosi o tool — nigdy nie wydaje finalnej odpowiedzi.
        client.chat.completions.create.return_value = _tool_request_message(
            [_tool_call("call_x", "get_fundamentals", {"symbol": "AAPL"})]
        )
        adapter = ToolCallingOpenAIAdapter(
            api_key="sk-test", model="gpt-4o", max_rounds=2
        )

        # Nie może crashować — finalny parse na pustej/ostatniej wiadomości
        # może rzucić ValueError, ale liczba wywołań tool MUSI być ograniczona.
        with contextlib.suppress(ValueError):
            adapter.analyze_with_tools("Predict.", [tool])

        # Co najwyżej max_rounds wykonanych rund narzędzia (twardy cap FinOps).
        assert tool_func.call_count <= 2

    def test_forces_final_answer_after_cap(self, mock_openai_class: MagicMock) -> None:
        client = MagicMock()
        mock_openai_class.return_value = client
        tool = _make_tool(MagicMock(return_value={"pe": 1.0}))
        # Pierwsze 2 rundy: tool. Ostatnie (wymuszone) wywołanie: finalny JSON.
        client.chat.completions.create.side_effect = [
            _tool_request_message(
                [_tool_call("c1", "get_fundamentals", {"symbol": "AAPL"})]
            ),
            _tool_request_message(
                [_tool_call("c2", "get_fundamentals", {"symbol": "AAPL"})]
            ),
            _final_message(json.dumps({"trend_direction": "NEUTRAL"})),
        ]
        adapter = ToolCallingOpenAIAdapter(
            api_key="sk-test", model="gpt-4o", max_rounds=2
        )

        result = adapter.analyze_with_tools("Predict.", [tool])

        assert result == {"trend_direction": "NEUTRAL"}


class TestToolErrorResilience:
    """(d) Funkcja narzędzia rzuca → pętla żyje, błąd jako wynik narzędzia."""

    def test_tool_exception_surfaced_as_error_result_no_crash(
        self, adapter_with_client: tuple[ToolCallingOpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client

        def boom(_args: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("network down")

        tool = _make_tool(boom)
        client.chat.completions.create.side_effect = [
            _tool_request_message(
                [_tool_call("call_err", "get_fundamentals", {"symbol": "AAPL"})]
            ),
            _final_message(json.dumps({"trend_direction": "BULLISH"})),
        ]

        result = adapter.analyze_with_tools("Predict.", [tool])

        # Pętla nie wywaliła się — model dostał finalną szansę i odpowiedział.
        assert result == {"trend_direction": "BULLISH"}
        # Wynik narzędzia to {"error": ...}, podany z powrotem do modelu.
        second_call = client.chat.completions.create.call_args_list[1]
        tool_messages = [
            m for m in second_call.kwargs["messages"] if m.get("role") == "tool"
        ]
        payload = json.loads(tool_messages[0]["content"])
        assert "error" in payload
        assert "network down" in payload["error"]

    def test_unknown_tool_name_surfaced_as_error(
        self, adapter_with_client: tuple[ToolCallingOpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        tool = _make_tool(MagicMock(return_value={"pe": 1.0}))
        client.chat.completions.create.side_effect = [
            _tool_request_message(
                [_tool_call("call_u", "nonexistent_tool", {"x": 1})]
            ),
            _final_message("{}"),
        ]

        adapter.analyze_with_tools("Predict.", [tool])

        second_call = client.chat.completions.create.call_args_list[1]
        tool_messages = [
            m for m in second_call.kwargs["messages"] if m.get("role") == "tool"
        ]
        assert "error" in json.loads(tool_messages[0]["content"])


class TestConfiguration:
    def test_is_tool_use_llm_port(self) -> None:
        assert issubclass(ToolCallingOpenAIAdapter, ToolUseLLMPort)

    def test_initializes_client_with_api_key_and_timeout(
        self, mock_openai_class: MagicMock
    ) -> None:
        ToolCallingOpenAIAdapter(api_key="sk-secret")
        call = mock_openai_class.call_args
        assert call.kwargs["api_key"] == "sk-secret"
        assert call.kwargs["timeout"] > 0

    def test_default_model_constructs_fine(self, mock_openai_class: MagicMock) -> None:
        adapter = ToolCallingOpenAIAdapter(api_key="sk-test")
        assert adapter._model == DEFAULT_MODEL

    def test_empty_model_id_raises(self, mock_openai_class: MagicMock) -> None:
        with pytest.raises(ValueError, match="model"):
            ToolCallingOpenAIAdapter(api_key="sk-test", model="")

    def test_passes_model_and_prompt(
        self, adapter_with_client: tuple[ToolCallingOpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _final_message("{}")
        tool = _make_tool(MagicMock(return_value={}))

        adapter.analyze_with_tools("Custom prompt", [tool])

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["messages"][0] == {"role": "user", "content": "Custom prompt"}

    def test_raises_on_invalid_final_json(
        self, adapter_with_client: tuple[ToolCallingOpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _final_message("not json")
        tool = _make_tool(MagicMock(return_value={}))

        with pytest.raises(ValueError, match="JSON"):
            adapter.analyze_with_tools("Predict.", [tool])

    def test_rejects_top_level_json_array(
        self, adapter_with_client: tuple[ToolCallingOpenAIAdapter, MagicMock]
    ) -> None:
        # ARCH-01: lista to poprawny JSON, ale werdykt musi być obiektem.
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _final_message('["BULLISH"]')
        tool = _make_tool(MagicMock(return_value={}))

        with pytest.raises(ValueError, match="JSON object"):
            adapter.analyze_with_tools("Predict.", [tool])


def test_tool_error_returned_to_model_omits_secrets(mock_openai_class: MagicMock) -> None:
    # Runda 2 BEZ-07: błąd narzędzia wraca do modelu jako `{"error": ...}`, więc
    # URL z `?apikey=` i dosłowny klucz z Settings muszą z niego zniknąć wcześniej.
    client = MagicMock()
    mock_openai_class.return_value = client
    adapter = ToolCallingOpenAIAdapter(
        api_key="sk-test", model="gpt-4o", secret_values=("Lit3ral~odd#key",)
    )

    def leaky(_args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            "fetch failed for url https://www.alphavantage.co/query?apikey=SECRET-URL-1 "
            "(provider echoed Lit3ral~odd#key)"
        )

    client.chat.completions.create.side_effect = [
        _tool_request_message([_tool_call("call_s", "get_fundamentals", {"symbol": "AAPL"})]),
        _final_message("{}"),
    ]

    adapter.analyze_with_tools("Predict.", [_make_tool(leaky)])

    second_call = client.chat.completions.create.call_args_list[1]
    tool_message = next(m for m in second_call.kwargs["messages"] if m.get("role") == "tool")
    error = json.loads(tool_message["content"])["error"]
    assert "fetch failed" in error
    assert "SECRET-URL-1" not in error
    assert "Lit3ral~odd#key" not in error


class TestUsageReporting:
    """FIN-13: każdy request pętli tool-use zgłasza swoje tokeny do `UsageSink`."""

    @staticmethod
    def _with_usage(response: MagicMock, prompt: int, completion: int) -> MagicMock:
        response.usage = MagicMock(spec=["prompt_tokens", "completion_tokens"])
        response.usage.prompt_tokens = prompt
        response.usage.completion_tokens = completion
        return response

    def test_every_round_reports_its_usage(self, mock_openai_class: MagicMock) -> None:
        from src.application.ports import UsageSink

        client = MagicMock()
        mock_openai_class.return_value = client
        sink = MagicMock(spec=UsageSink)
        adapter = ToolCallingOpenAIAdapter(api_key="sk-test", model="gpt-4o", usage_sink=sink)
        client.chat.completions.create.side_effect = [
            self._with_usage(
                _tool_request_message([_tool_call("c1", "get_fundamentals", {"symbol": "A"})]),
                800,
                25,
            ),
            self._with_usage(_final_message('{"trend_direction": "BEARISH"}'), 950, 90),
        ]

        adapter.analyze_with_tools("Predict.", [_make_tool(lambda args: {"pe": 20})])

        recorded = [call.args[0] for call in sink.record.call_args_list]
        assert [(u.input_tokens, u.output_tokens) for u in recorded] == [(800, 25), (950, 90)]
        assert {(u.provider, u.model) for u in recorded} == {("openai", "gpt-4o")}


class TestPortErrorTaxonomy:
    """ARCH-33: błąd SDK w pętli tool-use i nieparsowalny werdykt wychodzą jako
    taksonomia portów (nadal typy SDK / `ValueError`)."""

    @staticmethod
    def _status_error(status: int) -> Exception:
        import httpx
        from openai import APIStatusError, InternalServerError, RateLimitError

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(status, request=request)
        if status == 429:
            return RateLimitError(message="rate limited", response=response, body=None)
        if status >= 500:
            return InternalServerError(message="unavailable", response=response, body=None)
        return APIStatusError(message="bad request", response=response, body=None)

    @pytest.mark.parametrize(
        ("status", "kind_name"),
        [(503, "ProviderUnavailable"), (429, "QuotaExhausted"), (400, "PortError")],
    )
    def test_sdk_error_maps_to_its_kind(
        self,
        adapter_with_client: tuple[ToolCallingOpenAIAdapter, MagicMock],
        status: int,
        kind_name: str,
    ) -> None:
        from openai import APIStatusError

        from src.application import errors

        adapter, client = adapter_with_client
        client.chat.completions.create.side_effect = self._status_error(status)

        with pytest.raises(getattr(errors, kind_name)) as excinfo:
            adapter.analyze_with_tools("p", [_make_tool(lambda **_: {})])

        assert isinstance(excinfo.value, APIStatusError)

    def test_unparseable_verdict_is_malformed_payload(
        self, adapter_with_client: tuple[ToolCallingOpenAIAdapter, MagicMock]
    ) -> None:
        from src.application.errors import MalformedPayload

        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _final_message("no json here")

        with pytest.raises(MalformedPayload):
            adapter.analyze_with_tools("p", [_make_tool(lambda **_: {})])
