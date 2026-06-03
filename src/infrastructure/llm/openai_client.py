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
# Modele reasoning (GPT-5, o-series) akceptują WYŁĄCZNIE domyślną temperature (=1)
# i zwracają 400 BadRequest na każdą inną. Dla nich `temperature` pomijamy w
# requeście (API użyje defaultu). Prefiksy dobrane pod faktyczne ograniczenie API.
_NO_CUSTOM_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")
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
    ) -> None:
        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._quota_monitor = quota_monitor

    def _supports_custom_temperature(self) -> bool:
        """GPT-5 / o-series akceptują tylko default temperature (=1)."""
        model = self._model.lower()
        return not any(
            model.startswith(p) for p in _NO_CUSTOM_TEMPERATURE_PREFIXES
        )

    def _create_kwargs(self, prompt: str, *, json_mode: bool) -> dict[str, Any]:
        """Wspólny builder argumentów create() — `temperature` tylko gdy model
        ją wspiera (inaczej API zwraca 400 dla GPT-5/o-series)."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._supports_custom_temperature():
            kwargs["temperature"] = self._temperature
        return kwargs

    def analyze(self, prompt: str) -> dict[str, Any]:
        response = self._call_with_retry(
            lambda: self._client.chat.completions.create(
                **self._create_kwargs(prompt, json_mode=True)
            )
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned empty content for analyze().")

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
