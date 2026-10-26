from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from src.application.errors import LLMTruncatedError, MalformedPayload
from src.application.ports import LLMPort, UsageSink
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaSeverity
from src.infrastructure.error_translation import as_port_error, sdk_error_kind
from src.infrastructure.llm._json_text import parse_json_object, strip_code_block
from src.infrastructure.llm._quota import QuotaEmitter
from src.infrastructure.llm._retry import RetryPolicy, call_with_retry
from src.infrastructure.llm._usage import openai_chat_usage, report_usage

DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.2  # niska temperatura — chcemy deterministycznego Quanta
# Publiczne id modeli OpenAI zaczynają się od "gpt-" (gpt-4o, gpt-5-mini...) albo
# "o" (o1, o3-mini, o4...). `validate_model_id` używa tego do fail-fast na
# oczywiście-złym configu.
_OPENAI_MODEL_PREFIXES = ("gpt-", "o")
# Modele reasoning (GPT-5, o-series) akceptują WYŁĄCZNIE domyślną temperature (=1)
# i zwracają 400 BadRequest na każdą inną. Dla nich `temperature` pomijamy w
# requeście (API użyje defaultu). Te same modele wymagają też `max_completion_tokens`
# zamiast `max_tokens` (patrz `_token_limit_kwarg`). Prefiksy dobrane pod
# faktyczne ograniczenie API.
_NO_CUSTOM_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")
# #31 — Strukturalne schematy JSON potrzebują kilkuset tokenów wyjścia. Bez capa
# OpenAI ogranicza output tylko defaultem modelu. Ciasny, nadpisywalny cap.
DEFAULT_MAX_TOKENS = 768
# Regresja 2026-06-21 (COUNCIL_LLM_MODEL=gpt-5-mini): dla reasoning models
# `max_completion_tokens` budżetuje NIEWIDOCZNE tokeny rozumowania + widoczny
# output RAZEM, a rozumowanie idzie pierwsze. Cap 768 bywał w całości zjadany
# przez rozumowanie → finish_reason="length", content="" → cała rada padała z
# "empty content" przy HTTP 200. Reasoning models dostają większy domyślny sufit.
# To SUFIT (płacisz za realnie wygenerowane tokeny), więc zapas jest darmowy dla
# krótkich odpowiedzi, a chroni przed ucięciem rozumowania. Nadpisywalny przez
# konstruktor (max_tokens=).
REASONING_MAX_TOKENS = 4096
# GHA fast loop ma 15 min hard timeout. SDK domyślnie czeka 600s na response —
# zawieszone wywołanie LLM mogłoby zjeść cały cykl. 30s starcza dla normalnego
# GPT-4o response (typowo 3-8s) i nie maskuje legitnych długich generacji.
DEFAULT_TIMEOUT = 30.0

# Retry-on-429 — TPM limit (tokens per minute) potrafi się odbić nagle przy
# 30+ symbolach z radą doradczą. Backoff: 2s, 4s, 8s (chyba że SDK zwróci
# nagłówek `retry-after` → wtedy honorujemy go).
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0

logger = logging.getLogger(__name__)

T = TypeVar("T")


def validate_model_id(model: str) -> None:
    """Lekka walidacja id modelu — bez sieciowego calla na starcie.

    #30 — Stary/wycofany/typo'd id failuje dopiero przy 1. API callu
    (non-retried 4xx), per symbol, co cykl — cichy total outage wykrywany
    dopiero przez exit code "wszystkie symbole padły". Łapiemy OCZYWISTĄ
    pomyłkę config (pusty string, zły provider prefix) natychmiast, z
    czytelnym komunikatem. Nie weryfikujemy istnienia konkretnego id w API.
    """
    if not model or not model.strip():
        raise ValueError(
            "OpenAI model id is empty — set a valid 'gpt-*' or 'o*' id "
            "(e.g. via COUNCIL_LLM_MODEL or the model= argument)."
        )
    if not model.startswith(_OPENAI_MODEL_PREFIXES):
        raise ValueError(
            f"OpenAI model id {model!r} is malformed — expected an id "
            f"starting with 'gpt-' or 'o' (e.g. 'gpt-4o', 'o3-mini')."
        )


class OpenAIAdapter(LLMPort):
    """Adapter dla OpenAI Chat Completions API (SDK v2+).

    `analyze` używa JSON Mode — wymusza ustrukturyzowany output.
    `analyze_mistake` zwraca free-form tekst (correction insight).

    Przełączenie na Anthropic = podmiana tej klasy w DI Containerze
    (main_agent.py). Reszta systemu nie wie, że jesteśmy na OpenAI.

    Prompt caching: OpenAI cache'uje automatycznie prefixy ≥1024 tokenów
    dla zapytań w 5-10 min oknie (gpt-4o, gpt-4o-mini). Nie wymaga
    cache_control marker'ów po stronie SDK.

    Retry: 429 (TPM exceeded) jest retry'owany z backoffem; wszelkie inne
    błędy lecą natychmiast w górę (niech caller decyduje).

    FIN-13: tokeny udanego requestu (`response.usage`) idą do `usage_sink`, także
    przy pustej treści (np. `finish_reason="length"`) — request był płatny.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        quota_monitor: QuotaMonitor | None = None,
        max_tokens: int | None = None,
        usage_sink: UsageSink | None = None,
    ) -> None:
        # Fail-fast na oczywiście-złym id modelu (pusty / zły provider prefix).
        validate_model_id(model)
        # Retry żyje tylko w pętli adaptera — retry SDK mnożyłby płatne próby (NIEZ-07).
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._quota_monitor = quota_monitor
        self._usage_sink = usage_sink
        # None → domyślny cap zależny od rodziny modelu (reasoning vs nie);
        # jawna wartość → twardy override (per-rodzinny default pomijany).
        self._max_tokens = max_tokens

    def _supports_custom_temperature(self) -> bool:
        """GPT-5 / o-series akceptują tylko default temperature (=1)."""
        model = self._model.lower()
        return not any(model.startswith(p) for p in _NO_CUSTOM_TEMPERATURE_PREFIXES)

    def _is_reasoning_model(self) -> bool:
        """Reasoning models (gpt-5, o-series) różnią się nazwą parametru capa
        wyjścia ORAZ semantyką budżetu tokenów (rozumowanie + output razem)."""
        model = self._model.lower()
        return any(model.startswith(p) for p in _NO_CUSTOM_TEMPERATURE_PREFIXES)

    def _token_limit_kwarg(self) -> str:
        """Nazwa parametru capa wyjścia zależna od rodziny modelu.

        Reasoning models (gpt-5, o-series) odrzucają `max_tokens` i wymagają
        `max_completion_tokens`. Reszta (gpt-4o itd.) używa `max_tokens`.
        Te same prefiksy co dla temperature — to ta sama rodzina modeli."""
        if self._is_reasoning_model():
            return "max_completion_tokens"
        return "max_tokens"

    def _resolved_max_tokens(self) -> int:
        """Efektywny cap wyjścia.

        Jawny `max_tokens=` z konstruktora wygrywa zawsze. Bez niego: reasoning
        models dostają większy zapas (`REASONING_MAX_TOKENS`), bo `max_completion_tokens`
        budżetuje rozumowanie + output razem — 768 bywało zjadane w całości przez
        rozumowanie (regresja 2026-06-21). Nie-reasoning zostaje przy ciasnym
        `DEFAULT_MAX_TOKENS`."""
        if self._max_tokens is not None:
            return self._max_tokens
        return REASONING_MAX_TOKENS if self._is_reasoning_model() else DEFAULT_MAX_TOKENS

    def _create_kwargs(self, prompt: str, *, json_mode: bool) -> dict[str, Any]:
        """Wspólny builder argumentów create() — `temperature` tylko gdy model
        ją wspiera (inaczej API zwraca 400 dla GPT-5/o-series); cap wyjścia
        pod właściwą nazwą parametru (`max_tokens` / `max_completion_tokens`)."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._supports_custom_temperature():
            kwargs["temperature"] = self._temperature
        # #31 — cap wyjścia: bez niego output ograniczony tylko defaultem modelu.
        kwargs[self._token_limit_kwarg()] = self._resolved_max_tokens()
        return kwargs

    def analyze(self, prompt: str) -> dict[str, Any]:
        response = self._call_with_retry(
            lambda: self._client.chat.completions.create(
                **self._create_kwargs(prompt, json_mode=True)
            )
        )
        self._report_usage(response)
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            # Pusta treść przy HTTP 200 to objaw — przyczynę zdradza finish_reason.
            # "length" = model uciął się zanim wyemitował widoczną treść; dla
            # reasoning models (gpt-5/o-series) zwykle znaczy, że rozumowanie
            # zjadło cały `max_completion_tokens` (regresja 2026-06-21).
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                raise LLMTruncatedError(
                    "OpenAI returned empty content for analyze(): response "
                    "truncated (finish_reason='length') before any visible "
                    "output — for reasoning models this means reasoning "
                    "consumed the whole max_completion_tokens budget. Raise "
                    "max_tokens for this model.",
                    output_tokens=_completion_tokens(response),
                )
            raise MalformedPayload(
                f"OpenAI returned empty content for analyze() (finish_reason={finish_reason!r})."
            )

        try:
            return parse_json_object(content)
        except json.JSONDecodeError:
            pass
        # NIEZ-44: JSON w bloku ```json``` parsujemy jak AnthropicAdapter — failover
        # przełącza dostawców, więc ta sama odpowiedź ma dać ten sam dict.
        fenced = strip_code_block(content)
        try:
            return parse_json_object(fenced)
        except json.JSONDecodeError as exc:
            # NIEZ-26: nieparsowalna treść ucięta limitem to ucięcie, nie „zły JSON”.
            if getattr(choice, "finish_reason", None) == "length":
                raise LLMTruncatedError(
                    f"OpenAI {self._model} answer truncated (finish_reason='length') at "
                    f"{self._resolved_max_tokens()} tokens: {content[:200]!r}",
                    output_tokens=_completion_tokens(response),
                ) from exc
            raise MalformedPayload(f"OpenAI returned invalid JSON: {content[:200]!r}") from exc

    def analyze_mistake(self, prompt: str) -> str:
        response = self._call_with_retry(
            lambda: self._client.chat.completions.create(
                **self._create_kwargs(prompt, json_mode=False)
            )
        )
        self._report_usage(response)
        content: str | None = response.choices[0].message.content
        if content is None:
            return ""
        return content.strip()

    def _report_usage(self, response: Any) -> None:
        if self._usage_sink is not None:
            report_usage(self._usage_sink, openai_chat_usage(response, self._model))

    def _call_with_retry(self, fn: Callable[[], T]) -> T:
        """Retry'uje 429, 5xx i błędy połączenia. Honoruje retry-after — wspólna pętla
        `call_with_retry` (ARCH-29).

        Sygnalizacja kwot:
        - sukces po >=1 retry → WARNING (na granicy TPM)
        - wyczerpanie retry → CRITICAL przed propagacją wyjątku

        ARCH-33: błąd wychodzi przetłumaczony na taksonomię portów (`sdk_error_kind`),
        wciąż jako typ SDK z `status_code` — failover i dotychczasowe handlery bez zmian.
        Nieprzejściowy status (zły prompt, klucz, model) wychodzi od razu, bez ponowień.
        """
        policy = RetryPolicy(
            retryable=(APIStatusError, APIConnectionError),
            max_retries=self._max_retries,
            backoff_base=self._backoff_base,
            logger=logger,
            label="OpenAI",
            is_transient=_is_transient,
            reason=_reason,
            translate=_as_port_error,
        )
        return call_with_retry(
            fn,
            policy,
            sleep=time.sleep,
            on_recovered=self._report_recovered,
            on_exhausted=self._report_exhausted,
        )

    def _report_recovered(self, attempts: int) -> None:
        # Sukces po retry — daliśmy radę, ale jesteśmy na granicy.
        self._quota().emit(
            severity=QuotaSeverity.WARNING,
            message=(
                f"OpenAI {self._model}: limit TPM trafiony {attempts}× "
                "w jednym wywołaniu (ponowienie się udało)."
            ),
            action=(
                "Rada działa już na modelu mini — podnieś "
                "COUNCIL_VOLATILITY_THRESHOLD, przejdź na gpt-5-nano "
                "albo podnieś tier konta OpenAI."
            ),
            log_message=(
                f"OpenAI {self._model} hit TPM limit "
                f"{attempts}× during a single call (retry succeeded)."
            ),
        )

    def _report_exhausted(self, exc: Exception) -> None:
        reason = _reason(exc)
        self._quota().emit(
            severity=QuotaSeverity.CRITICAL,
            message=(
                f"OpenAI {self._model} zwrócił {reason} po "
                f"{self._max_retries} ponowieniach — wywołanie nie "
                "powiodło się."
            ),
            action=(
                "Podnieś tier konta OpenAI albo skieruj radę doradczą "
                "na model z wyższym limitem TPM przez COUNCIL_LLM_MODEL."
            ),
            log_message=(
                f"OpenAI {self._model} returned {reason} after "
                f"{self._max_retries} retries — call failed."
            ),
        )

    def _quota(self) -> QuotaEmitter:
        return QuotaEmitter(f"OpenAI ({self._model})", self._quota_monitor, logger)


def _is_transient(exc: Exception) -> bool:
    """429 (TPM), 5xx i błędy połączenia — po klasie wyjątku SDK."""
    return isinstance(exc, RateLimitError | InternalServerError | APIConnectionError)


def _reason(exc: Exception) -> str:
    return "429" if isinstance(exc, RateLimitError) else type(exc).__name__


def _as_port_error(exc: Exception) -> Exception:
    """ARCH-33: błąd SDK w taksonomii portów, wciąż typu SDK."""
    return as_port_error(exc, sdk_error_kind(exc))


def _completion_tokens(response: Any) -> int | None:
    """Tokeny wyjścia z `response.usage` (NIEZ-26); None, gdy SDK ich nie podał."""
    tokens = getattr(getattr(response, "usage", None), "completion_tokens", None)
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None
