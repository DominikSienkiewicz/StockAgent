from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

from openai import OpenAI, RateLimitError

from src.application.ports import LLMPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity

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
    ) -> None:
        # Fail-fast na oczywiście-złym id modelu (pusty / zły provider prefix).
        validate_model_id(model)
        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._quota_monitor = quota_monitor
        # None → domyślny cap zależny od rodziny modelu (reasoning vs nie);
        # jawna wartość → twardy override (per-rodzinny default pomijany).
        self._max_tokens = max_tokens

    def _supports_custom_temperature(self) -> bool:
        """GPT-5 / o-series akceptują tylko default temperature (=1)."""
        model = self._model.lower()
        return not any(
            model.startswith(p) for p in _NO_CUSTOM_TEMPERATURE_PREFIXES
        )

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
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            # Pusta treść przy HTTP 200 to objaw — przyczynę zdradza finish_reason.
            # "length" = model uciął się zanim wyemitował widoczną treść; dla
            # reasoning models (gpt-5/o-series) zwykle znaczy, że rozumowanie
            # zjadło cały `max_completion_tokens` (regresja 2026-06-21).
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                raise ValueError(
                    "OpenAI returned empty content for analyze(): response "
                    "truncated (finish_reason='length') before any visible "
                    "output — for reasoning models this means reasoning "
                    "consumed the whole max_completion_tokens budget. Raise "
                    "max_tokens for this model."
                )
            raise ValueError(
                "OpenAI returned empty content for analyze() "
                f"(finish_reason={finish_reason!r})."
            )

        try:
            return json.loads(content)  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"OpenAI returned invalid JSON: {content[:200]!r}"
            ) from exc

    def analyze_mistake(self, prompt: str) -> str:
        response = self._call_with_retry(
            lambda: self._client.chat.completions.create(
                **self._create_kwargs(prompt, json_mode=False)
            )
        )
        content: str | None = response.choices[0].message.content
        if content is None:
            return ""
        return content.strip()

    def _call_with_retry(self, fn: Callable[[], T]) -> T:
        """Retry'uje wyłącznie RateLimitError (429). Honoruje retry-after.

        Sygnalizacja kwot:
        - sukces po >=1 retry → WARNING (na granicy TPM)
        - wyczerpanie retry → CRITICAL przed propagacją wyjątku
        """
        attempt = 0
        while True:
            try:
                result = fn()
                if attempt > 0:
                    # Sukces po retry — daliśmy radę, ale jesteśmy na granicy.
                    self._emit_quota_alert(
                        severity=QuotaSeverity.WARNING,
                        message=(
                            f"OpenAI {self._model} hit TPM limit "
                            f"{attempt}× during a single call (retry succeeded)."
                        ),
                        action=(
                            "Council already on a mini model — raise "
                            "COUNCIL_VOLATILITY_THRESHOLD, drop to gpt-5-nano, "
                            "or upgrade your OpenAI usage tier."
                        ),
                    )
                return result
            except RateLimitError as exc:
                if attempt >= self._max_retries:
                    self._emit_quota_alert(
                        severity=QuotaSeverity.CRITICAL,
                        message=(
                            f"OpenAI {self._model} returned 429 after "
                            f"{self._max_retries} retries — call failed."
                        ),
                        action=(
                            "Upgrade OpenAI tier or route the advisory "
                            "council to a higher-TPM model via "
                            "COUNCIL_LLM_MODEL."
                        ),
                    )
                    raise
                wait = self._compute_wait(exc, attempt)
                logger.warning(
                    "OpenAI 429 (attempt %d/%d) — sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    wait,
                )
                time.sleep(wait)
                attempt += 1

    def _compute_wait(self, exc: RateLimitError, attempt: int) -> float:
        # Najpierw spróbuj `retry-after` z odpowiedzi serwera — to jego
        # rekomendacja, nie nasze zgadywanie.
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {}) or {}
            raw = headers.get("retry-after")
            if isinstance(raw, str):
                try:
                    return float(raw)
                except ValueError:
                    pass
        # Fallback: backoff wykładniczy 2s, 4s, 8s.
        return float(self._backoff_base * (2**attempt))

    def _emit_quota_alert(
        self, severity: QuotaSeverity, message: str, action: str
    ) -> None:
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source=f"OpenAI ({self._model})",
                severity=severity,
                message=message,
                action=action,
                occurred_at=datetime.now(UTC),
            )
        )
