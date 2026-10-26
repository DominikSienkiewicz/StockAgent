import contextlib
import json
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from src.application.ports import Tool, ToolUseLLMPort
from src.infrastructure.llm.tool_calling_anthropic import (
    DEFAULT_MODEL,
    ToolCallingAnthropicAdapter,
)


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(block_id: str, name: str, tool_input: dict[str, Any]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = block_id
    block.name = name
    # Anthropic zwraca `input` już jako rozparsowany dict.
    block.input = tool_input
    return block


def _final_response(text: str) -> MagicMock:
    """Model wydał finalną odpowiedź (stop_reason=end_turn)."""
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [_text_block(text)]
    return response


def _tool_use_response(blocks: list[MagicMock]) -> MagicMock:
    """Model prosi o wywołanie narzędzi (stop_reason=tool_use)."""
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = blocks
    return response


@pytest.fixture
def mock_anthropic_class(mocker: Any) -> MagicMock:
    return cast(MagicMock, mocker.patch("src.infrastructure.llm.tool_calling_anthropic.Anthropic"))


@pytest.fixture
def adapter_with_client(
    mock_anthropic_class: MagicMock,
) -> tuple[ToolCallingAnthropicAdapter, MagicMock]:
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    adapter = ToolCallingAnthropicAdapter(api_key="sk-ant-test", model="claude-sonnet-4-6")
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
    """(a) Model odpowiada od razu, bez tool-use → parsed JSON, brak pętli."""

    def test_returns_parsed_json_without_calling_tools(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.messages.create.return_value = _final_response(
            json.dumps({"trend_direction": "BULLISH", "confidence_score": 0.9})
        )
        called = MagicMock(return_value={"pe": 20.0})
        tool = _make_tool(called)

        result = adapter.analyze_with_tools("Predict AAPL.", [tool])

        assert result == {"trend_direction": "BULLISH", "confidence_score": 0.9}
        called.assert_not_called()
        assert client.messages.create.call_count == 1

    def test_tools_mapped_to_anthropic_schema(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.messages.create.return_value = _final_response("{}")
        tool = _make_tool(MagicMock(return_value={}))

        adapter.analyze_with_tools("Predict.", [tool])

        kwargs = client.messages.create.call_args.kwargs
        tools_arg = kwargs["tools"]
        assert tools_arg[0]["name"] == "get_fundamentals"
        assert tools_arg[0]["description"] == "Zwraca P/E i EPS spółki."
        # Anthropic używa `input_schema`, nie `parameters`.
        assert tools_arg[0]["input_schema"] == tool.parameters


class TestSingleToolRound:
    """(b) Model prosi o jeden tool, dostaje wynik, potem odpowiada."""

    def test_tool_func_invoked_with_input_then_final_json(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        tool_func = MagicMock(return_value={"pe": 18.5})
        tool = _make_tool(tool_func)
        client.messages.create.side_effect = [
            _tool_use_response([_tool_use_block("tu_1", "get_fundamentals", {"symbol": "AAPL"})]),
            _final_response(json.dumps({"trend_direction": "BEARISH"})),
        ]

        result = adapter.analyze_with_tools("Predict AAPL.", [tool])

        tool_func.assert_called_once_with({"symbol": "AAPL"})
        assert result == {"trend_direction": "BEARISH"}
        assert client.messages.create.call_count == 2

    def test_tool_result_fed_back_as_tool_result_block(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        tool = _make_tool(MagicMock(return_value={"pe": 18.5}))
        client.messages.create.side_effect = [
            _tool_use_response([_tool_use_block("tu_42", "get_fundamentals", {"symbol": "AAPL"})]),
            _final_response("{}"),
        ]

        adapter.analyze_with_tools("Predict.", [tool])

        second_call = client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        # Ostatnia wiadomość to user z blokiem tool_result.
        last = messages[-1]
        assert last["role"] == "user"
        tool_results = [b for b in last["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "tu_42"
        assert json.loads(tool_results[0]["content"]) == {"pe": 18.5}

    def test_assistant_tool_use_turn_appended(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        # Pętla musi dołożyć turę asystenta (z blokiem tool_use) przed wynikami.
        adapter, client = adapter_with_client
        tool = _make_tool(MagicMock(return_value={"pe": 1.0}))
        client.messages.create.side_effect = [
            _tool_use_response([_tool_use_block("tu_a", "get_fundamentals", {"symbol": "AAPL"})]),
            _final_response("{}"),
        ]

        adapter.analyze_with_tools("Predict.", [tool])

        messages = client.messages.create.call_args_list[1].kwargs["messages"]
        roles = [m["role"] for m in messages]
        # user (prompt) → assistant (tool_use) → user (tool_result)
        assert roles == ["user", "assistant", "user"]


class TestMaxRoundsCap:
    """(c) FinOps — model w kółko prosi o tool; pętla kończy po max_rounds."""

    def test_tool_calls_bounded_by_max_rounds(self, mock_anthropic_class: MagicMock) -> None:
        client = MagicMock()
        mock_anthropic_class.return_value = client
        tool_func = MagicMock(return_value={"pe": 1.0})
        tool = _make_tool(tool_func)
        client.messages.create.return_value = _tool_use_response(
            [_tool_use_block("tu_x", "get_fundamentals", {"symbol": "AAPL"})]
        )
        adapter = ToolCallingAnthropicAdapter(
            api_key="sk-ant-test", model="claude-sonnet-4-6", max_rounds=2
        )

        with contextlib.suppress(ValueError):
            adapter.analyze_with_tools("Predict.", [tool])

        assert tool_func.call_count <= 2

    def test_forces_final_answer_after_cap(self, mock_anthropic_class: MagicMock) -> None:
        client = MagicMock()
        mock_anthropic_class.return_value = client
        tool = _make_tool(MagicMock(return_value={"pe": 1.0}))
        client.messages.create.side_effect = [
            _tool_use_response([_tool_use_block("c1", "get_fundamentals", {"symbol": "AAPL"})]),
            _tool_use_response([_tool_use_block("c2", "get_fundamentals", {"symbol": "AAPL"})]),
            _final_response(json.dumps({"trend_direction": "NEUTRAL"})),
        ]
        adapter = ToolCallingAnthropicAdapter(
            api_key="sk-ant-test", model="claude-sonnet-4-6", max_rounds=2
        )

        result = adapter.analyze_with_tools("Predict.", [tool])

        assert result == {"trend_direction": "NEUTRAL"}

    def test_last_forced_call_drops_tools_to_force_text(
        self, mock_anthropic_class: MagicMock
    ) -> None:
        # Po wyczerpaniu rund wymuszamy finalną odpowiedź — bez `tools`,
        # żeby model nie mógł znowu poprosić o narzędzie.
        client = MagicMock()
        mock_anthropic_class.return_value = client
        tool = _make_tool(MagicMock(return_value={"pe": 1.0}))
        client.messages.create.side_effect = [
            _tool_use_response([_tool_use_block("c1", "get_fundamentals", {"symbol": "AAPL"})]),
            _final_response(json.dumps({"trend_direction": "NEUTRAL"})),
        ]
        adapter = ToolCallingAnthropicAdapter(
            api_key="sk-ant-test", model="claude-sonnet-4-6", max_rounds=1
        )

        adapter.analyze_with_tools("Predict.", [tool])

        # Drugie (wymuszone) wywołanie nie przekazuje już narzędzi.
        forced_call = client.messages.create.call_args_list[1]
        assert "tools" not in forced_call.kwargs or not forced_call.kwargs["tools"]


class TestToolErrorResilience:
    """(d) Funkcja narzędzia rzuca → pętla żyje, błąd jako wynik narzędzia."""

    def test_tool_exception_surfaced_as_error_result_no_crash(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client

        def boom(_args: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("network down")

        tool = _make_tool(boom)
        client.messages.create.side_effect = [
            _tool_use_response([_tool_use_block("tu_err", "get_fundamentals", {"symbol": "AAPL"})]),
            _final_response(json.dumps({"trend_direction": "BULLISH"})),
        ]

        result = adapter.analyze_with_tools("Predict.", [tool])

        assert result == {"trend_direction": "BULLISH"}
        second_call = client.messages.create.call_args_list[1]
        last = second_call.kwargs["messages"][-1]
        tool_results = [b for b in last["content"] if b.get("type") == "tool_result"]
        payload = json.loads(tool_results[0]["content"])
        assert "error" in payload
        assert "network down" in payload["error"]

    def test_unknown_tool_name_surfaced_as_error(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        tool = _make_tool(MagicMock(return_value={"pe": 1.0}))
        client.messages.create.side_effect = [
            _tool_use_response([_tool_use_block("tu_u", "nonexistent_tool", {"x": 1})]),
            _final_response("{}"),
        ]

        adapter.analyze_with_tools("Predict.", [tool])

        last = client.messages.create.call_args_list[1].kwargs["messages"][-1]
        tool_results = [b for b in last["content"] if b.get("type") == "tool_result"]
        assert "error" in json.loads(tool_results[0]["content"])


class TestJsonParsing:
    """Reuse parsowania jak w AnthropicAdapter — proza/code-block wokół JSON."""

    def test_extracts_json_from_prose(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        prose = 'Here is the analysis: {"trend_direction":"BULLISH"} hope it helps'
        client.messages.create.return_value = _final_response(prose)
        tool = _make_tool(MagicMock(return_value={}))

        result = adapter.analyze_with_tools("Predict.", [tool])

        assert result == {"trend_direction": "BULLISH"}

    def test_strips_markdown_code_block(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        wrapped = '```json\n{"trend_direction": "BEARISH"}\n```'
        client.messages.create.return_value = _final_response(wrapped)
        tool = _make_tool(MagicMock(return_value={}))

        result = adapter.analyze_with_tools("Predict.", [tool])

        assert result == {"trend_direction": "BEARISH"}

    def test_raises_on_genuine_non_json(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.messages.create.return_value = _final_response("cannot analyze now")
        tool = _make_tool(MagicMock(return_value={}))

        with pytest.raises(ValueError, match="JSON"):
            adapter.analyze_with_tools("Predict.", [tool])

    def test_rejects_top_level_json_scalar(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        # ARCH-01: liczba to poprawny JSON, ale werdykt musi być obiektem.
        adapter, client = adapter_with_client
        client.messages.create.return_value = _final_response("42")
        tool = _make_tool(MagicMock(return_value={}))

        with pytest.raises(ValueError, match="JSON object"):
            adapter.analyze_with_tools("Predict.", [tool])


class TestConfiguration:
    def test_is_tool_use_llm_port(self) -> None:
        assert issubclass(ToolCallingAnthropicAdapter, ToolUseLLMPort)

    def test_initializes_client_with_api_key_and_timeout(
        self, mock_anthropic_class: MagicMock
    ) -> None:
        ToolCallingAnthropicAdapter(api_key="sk-ant-secret")
        call = mock_anthropic_class.call_args
        assert call.kwargs["api_key"] == "sk-ant-secret"
        assert call.kwargs["timeout"] > 0

    def test_default_model_constructs_fine(self, mock_anthropic_class: MagicMock) -> None:
        adapter = ToolCallingAnthropicAdapter(api_key="sk-ant-test")
        assert adapter._model == DEFAULT_MODEL

    def test_wrong_prefix_model_raises(self, mock_anthropic_class: MagicMock) -> None:
        with pytest.raises(ValueError, match="claude-"):
            ToolCallingAnthropicAdapter(api_key="sk-ant-test", model="gpt-4o")

    def test_passes_model_and_prompt(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.messages.create.return_value = _final_response("{}")
        tool = _make_tool(MagicMock(return_value={}))

        adapter.analyze_with_tools("Custom prompt", [tool])

        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["messages"][0] == {"role": "user", "content": "Custom prompt"}


def test_tool_error_returned_to_model_omits_secrets(mock_anthropic_class: MagicMock) -> None:
    # Runda 2 BEZ-07: błąd narzędzia wraca do modelu jako `{"error": ...}`, więc
    # URL z `?apikey=` i dosłowny klucz z Settings muszą z niego zniknąć wcześniej.
    client = MagicMock()
    mock_anthropic_class.return_value = client
    adapter = ToolCallingAnthropicAdapter(
        api_key="sk-ant-test",
        model="claude-sonnet-4-6",
        secret_values=("Lit3ral~odd#key",),
    )

    def leaky(_args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            "fetch failed for url https://www.alphavantage.co/query?apikey=SECRET-URL-1 "
            "(provider echoed Lit3ral~odd#key)"
        )

    client.messages.create.side_effect = [
        _tool_use_response([_tool_use_block("tu_s", "get_fundamentals", {"symbol": "AAPL"})]),
        _final_response("{}"),
    ]

    adapter.analyze_with_tools("Predict.", [_make_tool(leaky)])

    last = client.messages.create.call_args_list[1].kwargs["messages"][-1]
    tool_result = next(b for b in last["content"] if b.get("type") == "tool_result")
    error = json.loads(tool_result["content"])["error"]
    assert "fetch failed" in error
    assert "SECRET-URL-1" not in error
    assert "Lit3ral~odd#key" not in error


class TestUsageReporting:
    """FIN-13: pętla tool-use to kilka płatnych requestów pod jednym licznikiem
    `llm` — każdy z nich zgłasza swoje tokeny do `UsageSink`."""

    @staticmethod
    def _with_usage(response: MagicMock, input_tokens: int, output_tokens: int) -> MagicMock:
        response.usage = MagicMock(spec=["input_tokens", "output_tokens"])
        response.usage.input_tokens = input_tokens
        response.usage.output_tokens = output_tokens
        return response

    def test_every_round_reports_its_usage(self, mock_anthropic_class: MagicMock) -> None:
        from src.application.ports import UsageSink

        client = MagicMock()
        mock_anthropic_class.return_value = client
        sink = MagicMock(spec=UsageSink)
        adapter = ToolCallingAnthropicAdapter(
            api_key="sk-ant-test", model="claude-sonnet-4-6", usage_sink=sink
        )
        client.messages.create.side_effect = [
            self._with_usage(
                _tool_use_response([_tool_use_block("t1", "get_fundamentals", {"symbol": "A"})]),
                1000,
                40,
            ),
            self._with_usage(_final_response('{"trend_direction": "BULLISH"}'), 1300, 120),
        ]

        adapter.analyze_with_tools("Predict.", [_make_tool(lambda args: {"pe": 20})])

        recorded = [call.args[0] for call in sink.record.call_args_list]
        assert [(u.input_tokens, u.output_tokens) for u in recorded] == [(1000, 40), (1300, 120)]
        assert {(u.provider, u.model) for u in recorded} == {("anthropic", "claude-sonnet-4-6")}


class TestPortErrorTaxonomy:
    """ARCH-33: błąd SDK w pętli tool-use i nieparsowalny werdykt wychodzą jako
    taksonomia portów (nadal typy SDK / `ValueError`)."""

    @staticmethod
    def _status_error(status: int) -> Exception:
        import httpx
        from anthropic import APIStatusError, InternalServerError, RateLimitError

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(status, request=request)
        if status == 429:
            return RateLimitError("rate limited", response=response, body=None)
        if status >= 500:
            return InternalServerError("overloaded", response=response, body=None)
        return APIStatusError("client error", response=response, body=None)

    @pytest.mark.parametrize(
        ("status", "kind_name"),
        [(529, "ProviderUnavailable"), (429, "QuotaExhausted"), (400, "PortError")],
    )
    def test_sdk_error_maps_to_its_kind(
        self,
        adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock],
        status: int,
        kind_name: str,
    ) -> None:
        from anthropic import APIStatusError

        from src.application import errors

        adapter, client = adapter_with_client
        client.messages.create.side_effect = self._status_error(status)

        with pytest.raises(getattr(errors, kind_name)) as excinfo:
            adapter.analyze_with_tools("p", [_make_tool(lambda **_: {})])

        assert isinstance(excinfo.value, APIStatusError)

    def test_unparseable_verdict_is_malformed_payload(
        self, adapter_with_client: tuple[ToolCallingAnthropicAdapter, MagicMock]
    ) -> None:
        from src.application.errors import MalformedPayload

        adapter, client = adapter_with_client
        client.messages.create.return_value = _final_response("no json here")

        with pytest.raises(MalformedPayload):
            adapter.analyze_with_tools("p", [_make_tool(lambda **_: {})])
