import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from src.application.ports import LLMPort
from src.infrastructure.llm.openai_client import DEFAULT_MODEL, OpenAIAdapter


def _completion(content: str | None, finish_reason: str = "stop") -> MagicMock:
    """Tworzy mock odpowiedzi OpenAI Chat Completions w nowym SDK (v2+).

    Struktura: response.choices[0].message.content + choices[0].finish_reason.
    `finish_reason` istotny dla reasoning models: "length" = ucięcie zanim
    pojawiła się widoczna treść (rozumowanie zjadło budżet tokenów).
    """
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].finish_reason = finish_reason
    return response


@pytest.fixture
def mock_openai_class(mocker: Any) -> MagicMock:
    """Mockuje samą klasę OpenAI w module adaptera — konstruktor zwraca MagicMock."""
    return cast(MagicMock, mocker.patch("src.infrastructure.llm.openai_client.OpenAI"))


@pytest.fixture
def adapter_with_client(mock_openai_class: MagicMock) -> tuple[OpenAIAdapter, MagicMock]:
    """Zwraca (adapter, mock_client) — client to to, co realnie wywołują metody."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o")
    return adapter, mock_client


class TestAnalyze:
    def test_returns_parsed_json_dict(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion(
            json.dumps(
                {
                    "trend_direction": "BULLISH",
                    "confidence_score": 0.85,
                    "target_price_12h": 110.0,
                    "reasoning": "Strong macro tailwinds.",
                }
            )
        )

        result = adapter.analyze("Predict AAPL trend.")

        assert isinstance(result, dict)
        assert result["trend_direction"] == "BULLISH"
        assert result["confidence_score"] == pytest.approx(0.85)
        assert result["target_price_12h"] == pytest.approx(110.0)

    def test_uses_json_response_format(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion("{}")

        adapter.analyze("Predict.")

        call = client.chat.completions.create.call_args
        assert call.kwargs["response_format"] == {"type": "json_object"}

    def test_passes_model_and_prompt(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion("{}")

        adapter.analyze("Custom prompt")

        call = client.chat.completions.create.call_args
        assert call.kwargs["model"] == "gpt-4o"
        messages = call.kwargs["messages"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Custom prompt"

    def test_raises_value_error_on_invalid_json(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion("not a json")

        with pytest.raises(ValueError, match="JSON"):
            adapter.analyze("Predict.")

    def test_analyze_rejects_top_level_json_array(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock]
    ) -> None:
        # ARCH-01: lista to poprawny JSON, ale nie obiekt — dawniej wracała jako „dict”.
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion('[{"trend": "UP"}]')

        with pytest.raises(ValueError, match="JSON object"):
            adapter.analyze("Predict.")

    def test_raises_value_error_when_content_empty(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion(None)

        with pytest.raises(ValueError, match="empty"):
            adapter.analyze("Predict.")


class TestAnalyzeMistake:
    def test_returns_plain_text_content(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion(
            "Zignorowałeś szerszy kontekst makro — Fed podniósł stopy."
        )

        result = adapter.analyze_mistake("Diagnose mistake.")

        assert result == "Zignorowałeś szerszy kontekst makro — Fed podniósł stopy."

    def test_does_not_use_json_format_for_freeform_text(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion("free form")

        adapter.analyze_mistake("Diagnose.")

        call = client.chat.completions.create.call_args
        # Plain text — bez wymuszania JSON formatu
        assert call.kwargs.get("response_format") != {"type": "json_object"}

    def test_strips_whitespace_from_response(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion("  insight text  \n")

        result = adapter.analyze_mistake("Diagnose.")

        assert result == "insight text"

    def test_returns_empty_string_when_content_is_none(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock]
    ) -> None:
        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion(None)

        result = adapter.analyze_mistake("Diagnose.")

        assert result == ""


class TestConfiguration:
    def test_custom_model_is_passed_to_api(self, mock_openai_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _completion("{}")
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")

        adapter.analyze("Predict.")

        assert mock_client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"

    def test_initializes_openai_client_with_api_key(self, mock_openai_class: MagicMock) -> None:
        OpenAIAdapter(api_key="sk-secret-123")
        # Sprawdzamy że api_key trafia (bez asercji na timeout — to detal),
        # ale timeout MUSI być ustawiony (GHA hard limit 15 min).
        call = mock_openai_class.call_args
        assert call.kwargs["api_key"] == "sk-secret-123"
        assert call.kwargs["timeout"] > 0


class TestAdapterImplementsPort:
    def test_is_llm_port(self) -> None:
        assert issubclass(OpenAIAdapter, LLMPort)


def _rate_limit_error(retry_after_seconds: float | None = None) -> Exception:
    """Buduje wyjątek RateLimitError jak SDK OpenAI (response 429)."""
    from openai import RateLimitError

    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    if retry_after_seconds is not None:
        response.headers["retry-after"] = str(retry_after_seconds)
    body = {"error": {"message": "tpm exceeded", "type": "tokens"}}
    return RateLimitError(message="rate limited", response=response, body=body)


class TestRetryOn429:
    """OpenAI TPM (tokens per minute) potrafi się odbić nagle przy 30+ symbolach
    z radą doradczą. Adapter retry'uje 429 z backoffem, pozwalając SDK i kodowi
    klienta nie martwić się tym (poza prawdziwymi długotrwałymi limitami).
    """

    def test_retries_on_429_then_succeeds(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock], mocker: Any
    ) -> None:
        adapter, client = adapter_with_client
        mocker.patch("src.infrastructure.llm.openai_client.time.sleep")  # bez realnego spania
        client.chat.completions.create.side_effect = [
            _rate_limit_error(),
            _rate_limit_error(),
            _completion(json.dumps({"trend_direction": "BULLISH"})),
        ]

        result = adapter.analyze("Predict.")

        assert result["trend_direction"] == "BULLISH"
        assert client.chat.completions.create.call_count == 3

    def test_raises_after_max_retries(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock], mocker: Any
    ) -> None:
        from openai import RateLimitError

        adapter, client = adapter_with_client
        mocker.patch("src.infrastructure.llm.openai_client.time.sleep")
        client.chat.completions.create.side_effect = _rate_limit_error()

        with pytest.raises(RateLimitError):
            adapter.analyze("Predict.")

        # Defaultowy max_retries=3 + initial = 4 wywołania.
        assert client.chat.completions.create.call_count == 4

    def test_does_not_retry_on_non_429_errors(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock], mocker: Any
    ) -> None:
        adapter, client = adapter_with_client
        mock_sleep = mocker.patch("src.infrastructure.llm.openai_client.time.sleep")
        client.chat.completions.create.side_effect = RuntimeError("network down")

        with pytest.raises(RuntimeError):
            adapter.analyze("Predict.")

        assert client.chat.completions.create.call_count == 1
        mock_sleep.assert_not_called()

    def test_respects_retry_after_header_when_present(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock], mocker: Any
    ) -> None:
        adapter, client = adapter_with_client
        mock_sleep = mocker.patch("src.infrastructure.llm.openai_client.time.sleep")
        client.chat.completions.create.side_effect = [
            _rate_limit_error(retry_after_seconds=7.5),
            _completion("{}"),
        ]

        adapter.analyze("Predict.")

        # Wzięliśmy retry-after = 7.5s zamiast defaultowego backoffu.
        first_sleep = mock_sleep.call_args_list[0].args[0]
        assert first_sleep >= 7.5

    def test_retry_also_works_for_analyze_mistake(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock], mocker: Any
    ) -> None:
        """analyze_mistake (free-form text) musi mieć tę samą logikę retry."""
        adapter, client = adapter_with_client
        mocker.patch("src.infrastructure.llm.openai_client.time.sleep")
        client.chat.completions.create.side_effect = [
            _rate_limit_error(),
            _completion("It was wrong because..."),
        ]

        result = adapter.analyze_mistake("Diagnose.")

        assert "wrong" in result
        assert client.chat.completions.create.call_count == 2


@pytest.mark.integration
class TestOpenAILive:
    def test_real_api_returns_json(self) -> None:
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        adapter = OpenAIAdapter(api_key=api_key, model="gpt-4o-mini")
        result = adapter.analyze('Return JSON: {"answer": 42}. Only JSON, nothing else.')
        assert isinstance(result, dict)


class TestQuotaAlertEmit:
    """429 → QuotaAlert. WARNING gdy retry zadziałał, CRITICAL gdy padło."""

    def test_warning_when_retry_succeeds(self, mock_openai_class: MagicMock, mocker: Any) -> None:
        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity

        monitor = QuotaMonitor()
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o", quota_monitor=monitor)
        mocker.patch("src.infrastructure.llm.openai_client.time.sleep")
        mock_client.chat.completions.create.side_effect = [
            _rate_limit_error(),
            _completion("{}"),
        ]

        adapter.analyze("Predict.")

        assert len(monitor.alerts) == 1
        alert = monitor.alerts[0]
        assert alert.severity is QuotaSeverity.WARNING
        assert "gpt-4o" in alert.source

    def test_critical_when_retries_exhausted(
        self, mock_openai_class: MagicMock, mocker: Any
    ) -> None:
        from openai import RateLimitError

        from src.application.quota_monitor import QuotaMonitor
        from src.domain.quota import QuotaSeverity

        monitor = QuotaMonitor()
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o", quota_monitor=monitor)
        mocker.patch("src.infrastructure.llm.openai_client.time.sleep")
        mock_client.chat.completions.create.side_effect = _rate_limit_error()

        with pytest.raises(RateLimitError):
            adapter.analyze("Predict.")

        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].severity is QuotaSeverity.CRITICAL

    def test_no_alert_on_clean_call(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock]
    ) -> None:
        from src.application.quota_monitor import QuotaMonitor

        monitor = QuotaMonitor()
        adapter, client = adapter_with_client
        adapter._quota_monitor = monitor
        client.chat.completions.create.return_value = _completion("{}")

        adapter.analyze("Predict.")

        assert monitor.alerts == []


class TestModelIdValidation:
    """Audyt z czerwca 2026 — Hardcoded model id bez guardu = cichy total outage. Walidacja w
    __init__ łapie pusty/typo'd id natychmiast, z czytelnym komunikatem."""

    def test_empty_model_id_raises_at_construction(self, mock_openai_class: MagicMock) -> None:
        with pytest.raises(ValueError, match="model"):
            OpenAIAdapter(api_key="sk-test", model="")

    def test_whitespace_model_id_raises_at_construction(self, mock_openai_class: MagicMock) -> None:
        with pytest.raises(ValueError, match="model"):
            OpenAIAdapter(api_key="sk-test", model="   ")

    def test_wrong_prefix_model_id_raises_at_construction(
        self, mock_openai_class: MagicMock
    ) -> None:
        # id bez prefiksu "gpt-"/"o" jest oczywiście błędny dla OpenAI.
        with pytest.raises(ValueError, match="gpt-"):
            OpenAIAdapter(api_key="sk-test", model="claude-sonnet-4-6")

    def test_valid_gpt_model_constructs_fine(self, mock_openai_class: MagicMock) -> None:
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
        assert adapter._model == "gpt-4o-mini"

    def test_valid_o_series_model_constructs_fine(self, mock_openai_class: MagicMock) -> None:
        # Reasoning models (o-series) nie mają prefiksu "gpt-" — muszą przejść.
        adapter = OpenAIAdapter(api_key="sk-test", model="o3-mini")
        assert adapter._model == "o3-mini"

    def test_default_model_id_constructs_fine(self, mock_openai_class: MagicMock) -> None:
        adapter = OpenAIAdapter(api_key="sk-test")
        assert adapter._model == DEFAULT_MODEL


class TestMaxTokensCap:
    """Audyt z czerwca 2026 — Bez `max_tokens`/`max_completion_tokens` output OpenAI jest
    ograniczony tylko defaultem modelu. Schematy JSON potrzebują kilkuset
    tokenów — ustawiamy ciasny, nadpisywalny cap. Reasoning models (gpt-5,
    o-series) używają `max_completion_tokens`, reszta `max_tokens`."""

    def _token_cap(self, kwargs: dict[str, Any]) -> int:
        # Akceptujemy obie nazwy parametru zależnie od rodziny modelu.
        if "max_completion_tokens" in kwargs:
            return cast(int, kwargs["max_completion_tokens"])
        return cast(int, kwargs["max_tokens"])

    def test_gpt4o_sets_max_tokens(self, mock_openai_class: MagicMock) -> None:
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.return_value = _completion("{}")
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o")

        adapter.analyze("Predict.")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "max_tokens" in kwargs
        assert kwargs["max_tokens"] <= 1024

    def test_gpt5_uses_max_completion_tokens(self, mock_openai_class: MagicMock) -> None:
        # Reasoning model — param name to `max_completion_tokens`, nie `max_tokens`.
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.return_value = _completion("{}")
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-5-mini")

        adapter.analyze("Predict.")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" in kwargs
        assert "max_tokens" not in kwargs
        # Reasoning models budżetują rozumowanie + output razem — cap musi mieć
        # zapas na niewidoczne tokeny rozumowania (regresja 2026-06-21), więc
        # NIE jest tak ciasny jak dla gpt-4o.
        assert kwargs["max_completion_tokens"] >= 2048

    def test_o_series_uses_max_completion_tokens(self, mock_openai_class: MagicMock) -> None:
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.return_value = _completion("{}")
        adapter = OpenAIAdapter(api_key="sk-test", model="o3-mini")

        adapter.analyze("Predict.")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" in kwargs
        assert "max_tokens" not in kwargs

    def test_max_tokens_overridable_via_constructor(self, mock_openai_class: MagicMock) -> None:
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.return_value = _completion("{}")
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o", max_tokens=2048)

        adapter.analyze("Predict.")

        assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 2048

    def test_analyze_mistake_also_capped(self, mock_openai_class: MagicMock) -> None:
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.return_value = _completion("insight")
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o")

        adapter.analyze_mistake("Diagnose.")

        kwargs = client.chat.completions.create.call_args.kwargs
        cap = self._token_cap(kwargs)
        assert 0 < cap <= 4096


class TestReasoningModelBudget:
    """Regresja 2026-06-21: COUNCIL_LLM_MODEL=gpt-5-mini, cała rada padała z
    'OpenAI returned empty content' przy HTTP 200. Dla reasoning models
    `max_completion_tokens` budżetuje rozumowanie + widoczny output RAZEM, a
    rozumowanie idzie pierwsze. Cap 768 (dobrany pod widoczny JSON gpt-4o) bywa
    w całości zjadany przez rozumowanie → finish_reason='length', content=''.
    Reasoning models muszą dostać większy domyślny sufit — to SUFIT (płacisz za
    realne tokeny), więc zapas jest darmowy dla krótkich odpowiedzi."""

    def test_reasoning_model_gets_reasoning_safe_default_budget(
        self, mock_openai_class: MagicMock
    ) -> None:
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.return_value = _completion("{}")
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-5-mini")

        adapter.analyze("x")

        kwargs = client.chat.completions.create.call_args.kwargs
        # 768 nie starcza na rozumowanie + JSON; reasoning models potrzebują
        # zapasu na niewidoczne tokeny rozumowania PRZED widoczną treścią.
        assert kwargs["max_completion_tokens"] >= 2048

    def test_non_reasoning_model_keeps_tight_default(self, mock_openai_class: MagicMock) -> None:
        # gpt-4o (nie-reasoning) nie marnuje budżetu na rozumowanie — zostaje
        # przy ciasnym, tanim capie.
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.return_value = _completion("{}")
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o")

        adapter.analyze("x")

        assert client.chat.completions.create.call_args.kwargs["max_tokens"] <= 1024

    def test_explicit_override_wins_for_reasoning_model(self, mock_openai_class: MagicMock) -> None:
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.return_value = _completion("{}")
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-5-mini", max_tokens=512)

        adapter.analyze("x")

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_completion_tokens"] == 512


class TestEmptyContentDiagnostics:
    """Pusta treść przy HTTP 200 to objaw, nie przyczyna. Komunikat błędu musi
    rozróżniać ucięcie z powodu limitu tokenów (finish_reason='length') od
    innych przyczyn — inaczej kolejny incydent jest tak samo nieczytelny jak
    2026-06-21 (cała rada 'empty content', zero wskazówki dlaczego)."""

    def test_length_finish_reason_gives_actionable_error(
        self, mock_openai_class: MagicMock
    ) -> None:
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.return_value = _completion(None, finish_reason="length")
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-5-mini")

        with pytest.raises(ValueError, match="truncat|max_completion_tokens"):
            adapter.analyze("x")

    @pytest.mark.parametrize("content", [None, '{"trend_direction": "BULL'], ids=["empty", "cut"])
    def test_length_finish_reason_raises_truncated(
        self, mock_openai_class: MagicMock, content: str | None
    ) -> None:
        # NIEZ-26: lustro Anthropic `stop_reason = max_tokens` — graf flaguje oba
        # przypadki jako `llm_output_truncated`, niezależnie od dostawcy.
        from src.application.errors import LLMTruncatedError

        client = MagicMock()
        mock_openai_class.return_value = client
        response = _completion(content, finish_reason="length")
        response.usage.completion_tokens = 4096
        client.chat.completions.create.return_value = response
        adapter = OpenAIAdapter(api_key="sk-test", model="gpt-5-mini")

        with pytest.raises(LLMTruncatedError) as excinfo:
            adapter.analyze("x")

        assert excinfo.value.output_tokens == 4096


class TestTemperatureCompatibility:
    """GPT-5 / o-series akceptują tylko domyślną temperature (=1). Adapter NIE
    może wysyłać `temperature` dla takich modeli — inaczej 400 BadRequest
    wywala radę w produkcji (regresja 2026-06-03, COUNCIL_LLM_MODEL=gpt-5-mini)."""

    def _adapter(self, mock_openai_class: MagicMock, model: str) -> tuple[OpenAIAdapter, MagicMock]:
        client = MagicMock()
        mock_openai_class.return_value = client
        return OpenAIAdapter(api_key="sk-test", model=model), client

    def test_gpt4o_still_sends_temperature(self, mock_openai_class: MagicMock) -> None:
        adapter, client = self._adapter(mock_openai_class, "gpt-4o")
        client.chat.completions.create.return_value = _completion("{}")
        adapter.analyze("x")
        assert client.chat.completions.create.call_args.kwargs["temperature"] == pytest.approx(0.2)

    def test_gpt5_mini_omits_temperature_in_analyze(self, mock_openai_class: MagicMock) -> None:
        adapter, client = self._adapter(mock_openai_class, "gpt-5-mini")
        client.chat.completions.create.return_value = _completion("{}")
        adapter.analyze("x")
        assert "temperature" not in client.chat.completions.create.call_args.kwargs

    def test_gpt5_mini_omits_temperature_in_analyze_mistake(
        self, mock_openai_class: MagicMock
    ) -> None:
        adapter, client = self._adapter(mock_openai_class, "gpt-5-mini")
        client.chat.completions.create.return_value = _completion("insight")
        adapter.analyze_mistake("x")
        assert "temperature" not in client.chat.completions.create.call_args.kwargs

    def test_o_series_omits_temperature(self, mock_openai_class: MagicMock) -> None:
        adapter, client = self._adapter(mock_openai_class, "o3-mini")
        client.chat.completions.create.return_value = _completion("{}")
        adapter.analyze("x")
        assert "temperature" not in client.chat.completions.create.call_args.kwargs


def _server_error() -> Exception:
    import httpx2
    from openai import InternalServerError

    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx2.Response(503, request=request)
    return InternalServerError(message="unavailable", response=response, body=None)


class TestSingleRetryLayer:
    """NIEZ-07: SDK nie retry'uje (max_retries=0), więc pętla adaptera musi sama
    łapać 5xx i błędy połączenia, a retry-after ma sufit."""

    def test_sdk_level_retries_disabled(self, mock_openai_class: MagicMock) -> None:
        OpenAIAdapter(api_key="sk-test")
        assert mock_openai_class.call_args.kwargs["max_retries"] == 0

    def test_retries_server_error(self, mock_openai_class: MagicMock, mocker: Any) -> None:
        mocker.patch("src.infrastructure.llm.openai_client.time.sleep")
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.side_effect = [_server_error(), _completion("{}")]

        assert OpenAIAdapter(api_key="sk-test", backoff_base=0.01).analyze("p") == {}
        assert client.chat.completions.create.call_count == 2

    def test_retry_after_is_capped(self, mock_openai_class: MagicMock, mocker: Any) -> None:
        sleep = mocker.patch("src.infrastructure.llm.openai_client.time.sleep")
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.side_effect = [
            _rate_limit_error(retry_after_seconds=3600),
            _completion("{}"),
        ]

        OpenAIAdapter(api_key="sk-test").analyze("p")

        assert sleep.call_args.args[0] <= 20


def test_critical_alert_is_polish_and_logged_in_english(
    mock_openai_class: MagicMock, mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # MAIL-15: komunikat alertu idzie do maila po polsku, diagnoza do logów po angielsku.
    import logging

    from openai import RateLimitError

    from src.application.quota_monitor import QuotaMonitor

    monitor = QuotaMonitor()
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o", quota_monitor=monitor)
    mocker.patch("src.infrastructure.llm.openai_client.time.sleep")
    mock_client.chat.completions.create.side_effect = _rate_limit_error()

    with caplog.at_level(logging.WARNING), pytest.raises(RateLimitError):
        adapter.analyze("Predict.")

    alert = monitor.alerts[0]
    assert "wywołanie nie powiodło się" in alert.message
    assert "call failed" not in alert.message
    assert alert.action.startswith("Podnieś")
    assert "OpenAI gpt-4o returned 429 after" in caplog.text


class TestUsageReporting:
    """FIN-13: `prompt_tokens` OpenAI zawiera już tokeny z cache, więc adapter
    zgłasza `input_tokens` = prompt − cached i `cache_read_tokens` = cached."""

    @staticmethod
    def _adapter(
        mock_openai_class: MagicMock, model: str = "gpt-5-mini"
    ) -> tuple[OpenAIAdapter, MagicMock, MagicMock]:
        from src.application.ports import UsageSink

        client = MagicMock()
        mock_openai_class.return_value = client
        sink = MagicMock(spec=UsageSink)
        return OpenAIAdapter(api_key="sk-test", model=model, usage_sink=sink), client, sink

    @staticmethod
    def _with_usage(response: MagicMock, prompt: int, completion: int, cached: object) -> MagicMock:
        response.usage = MagicMock(
            spec=["prompt_tokens", "completion_tokens", "prompt_tokens_details"]
        )
        response.usage.prompt_tokens = prompt
        response.usage.completion_tokens = completion
        if cached is None:
            response.usage.prompt_tokens_details = None
        else:
            response.usage.prompt_tokens_details = MagicMock(spec=["cached_tokens"])
            response.usage.prompt_tokens_details.cached_tokens = cached
        return response

    def test_usage_is_reported_to_sink(self, mock_openai_class: MagicMock) -> None:
        from src.domain.finops import LLMUsage

        adapter, client, sink = self._adapter(mock_openai_class)
        client.chat.completions.create.return_value = self._with_usage(
            _completion("{}"), prompt=2000, completion=450, cached=1536
        )

        adapter.analyze("Persona prompt")

        sink.record.assert_called_once_with(
            LLMUsage(
                provider="openai",
                model="gpt-5-mini",
                input_tokens=464,
                output_tokens=450,
                cache_read_tokens=1536,
                cache_write_tokens=0,
            )
        )

    def test_usage_without_prompt_details_has_no_cached_tokens(
        self, mock_openai_class: MagicMock
    ) -> None:
        adapter, client, sink = self._adapter(mock_openai_class, model="gpt-4o")
        client.chat.completions.create.return_value = self._with_usage(
            _completion("lekcja"), prompt=120, completion=30, cached=None
        )

        adapter.analyze_mistake("Why?")

        usage = sink.record.call_args.args[0]
        assert (usage.model, usage.input_tokens, usage.cache_read_tokens) == ("gpt-4o", 120, 0)

    def test_missing_usage_records_nothing(self, mock_openai_class: MagicMock) -> None:
        adapter, client, sink = self._adapter(mock_openai_class)
        response = _completion("{}")
        response.usage = None
        client.chat.completions.create.return_value = response

        assert adapter.analyze("prompt") == {}
        sink.record.assert_not_called()

    def test_usage_of_a_truncated_answer_is_still_reported(
        self, mock_openai_class: MagicMock
    ) -> None:
        # finish_reason="length": rozumowanie zjadło budżet, ale tokeny są na rachunku.
        adapter, client, sink = self._adapter(mock_openai_class)
        client.chat.completions.create.return_value = self._with_usage(
            _completion("", finish_reason="length"), prompt=900, completion=4096, cached=0
        )

        with pytest.raises(ValueError):
            adapter.analyze("prompt")

        assert sink.record.call_args.args[0].output_tokens == 4096


def _client_error(status: int = 400) -> Exception:
    import httpx2
    from openai import APIStatusError

    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx2.Response(status, request=request)
    return APIStatusError(message="bad request", response=response, body=None)


class TestPortErrorTaxonomy:
    """ARCH-33: błędy OpenAI po wyczerpaniu ponowień wychodzą jako taksonomia
    portów i nadal jako typy SDK (failover czyta `status_code`)."""

    @staticmethod
    def _adapter(mock_openai_class: MagicMock, mocker: Any, side_effect: object) -> OpenAIAdapter:
        mocker.patch("src.infrastructure.llm.openai_client.time.sleep")
        client = MagicMock()
        mock_openai_class.return_value = client
        client.chat.completions.create.side_effect = side_effect
        return OpenAIAdapter(api_key="sk-test", backoff_base=0.01, max_retries=1)

    def test_server_error_after_retries_is_provider_unavailable(
        self, mock_openai_class: MagicMock, mocker: Any
    ) -> None:
        from openai import InternalServerError

        from src.application.errors import ProviderUnavailable

        adapter = self._adapter(mock_openai_class, mocker, _server_error())

        with pytest.raises(ProviderUnavailable) as excinfo:
            adapter.analyze("p")

        assert isinstance(excinfo.value, InternalServerError)
        assert excinfo.value.status_code == 503

    def test_rate_limit_after_retries_is_quota_exhausted(
        self, mock_openai_class: MagicMock, mocker: Any
    ) -> None:
        from src.application.errors import QuotaExhausted

        adapter = self._adapter(mock_openai_class, mocker, _rate_limit_error())

        with pytest.raises(QuotaExhausted):
            adapter.analyze_mistake("p")

    def test_connection_error_after_retries_is_provider_unavailable(
        self, mock_openai_class: MagicMock, mocker: Any
    ) -> None:
        import httpx2
        from openai import APIConnectionError

        from src.application.errors import ProviderUnavailable

        request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
        adapter = self._adapter(mock_openai_class, mocker, APIConnectionError(request=request))

        with pytest.raises(ProviderUnavailable) as excinfo:
            adapter.analyze("p")

        assert isinstance(excinfo.value, APIConnectionError)

    def test_client_error_is_a_port_error_without_retry(
        self, mock_openai_class: MagicMock, mocker: Any
    ) -> None:
        from src.application.errors import PortError, ProviderUnavailable, QuotaExhausted

        adapter = self._adapter(mock_openai_class, mocker, _client_error(400))

        with pytest.raises(PortError) as excinfo:
            adapter.analyze("p")

        assert not isinstance(excinfo.value, ProviderUnavailable | QuotaExhausted)

    @pytest.mark.parametrize("content", [None, "no json here"], ids=["empty", "invalid-json"])
    def test_unparseable_answer_is_malformed_payload(
        self, adapter_with_client: tuple[OpenAIAdapter, MagicMock], content: str | None
    ) -> None:
        from src.application.errors import MalformedPayload

        adapter, client = adapter_with_client
        client.chat.completions.create.return_value = _completion(content)

        with pytest.raises(MalformedPayload):
            adapter.analyze("p")
