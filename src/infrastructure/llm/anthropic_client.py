from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

from anthropic import Anthropic, APIConnectionError, APIStatusError

from src.application.ports import LLMPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity

# Centralny model id (rodzina Claude 4.x). Świadomie BEZ daty w komentarzu —
# data dryfuje (zostaje "maj 2026" długo po fakcie) i sugeruje fałszywą
# aktualność. Zmianę modelu robi się tu albo przez COUNCIL_LLM_MODEL w configu;
# `validate_model_id` poniżej łapie pusty/typo'd override natychmiast.
DEFAULT_MODEL = "claude-sonnet-4-6"  # dobry kompromis kosztów/jakości
# Wszystkie publiczne id modeli Anthropic zaczynają się od "claude-".
_ANTHROPIC_MODEL_PREFIX = "claude-"
# #31 — Strukturalne schematy JSON (predykcja, council) potrzebują kilkuset
# tokenów wyjścia, nie 4096. Ciasny cap tnie koszt per-investor council call;
# nadpisywalny przez konstruktor (max_tokens=) dla wolnych generacji.
DEFAULT_MAX_TOKENS = 768
DEFAULT_TEMPERATURE = 0.2
# GHA fast loop ma 15 min hard timeout; SDK domyślnie czeka 600s.
DEFAULT_TIMEOUT = 30.0
# Próg włączania ephemeral cache — Anthropic wymaga ≥1024 tok dla sonnet/opus.
# Liczymy ~4 znaki/token jako bezpieczne oszacowanie.
_CACHE_MIN_CHARS = 4096

# Retry na 429 (rate limit) i 5xx (w tym 529 Overloaded) — analogicznie do
# OpenAIAdapter. Claude jest domyślnym, głównym LLM, więc musi mieć tę samą
# odporność: pojedynczy 529 z przeciążonego API nie może ubijać symbolu.
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Bez otaczających `\s*` — `\s*(.*?)\s*` na wejściu z dużą ilością białych
# znaków bez domknięcia fence'a backtrackuje wielomianowo (ReDoS na output LLM).
# Otaczające białe znaki obcinamy w `_strip_code_block` przez `.strip()`.
_CODE_BLOCK_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)


def validate_model_id(model: str) -> None:
    """Lekka walidacja id modelu — bez sieciowego calla na starcie.

    #30 — Stary/wycofany id failuje dopiero przy 1. API callu (non-retried 4xx),
    per symbol, co cykl — cichy total outage wykrywany dopiero przez exit code
    "wszystkie symbole padły". Tutaj łapiemy OCZYWISTĄ pomyłkę config (pusty
    string, zły provider prefix) natychmiast, z czytelnym komunikatem. Nie
    weryfikujemy istnienia konkretnego id w API — to wymagałoby calla i i tak
    by się starzało.
    """
    if not model or not model.strip():
        raise ValueError(
            "Anthropic model id is empty — set a valid 'claude-*' id "
            "(e.g. via COUNCIL_LLM_MODEL or the model= argument)."
        )
    if not model.startswith(_ANTHROPIC_MODEL_PREFIX):
        raise ValueError(
            f"Anthropic model id {model!r} is malformed — expected an id "
            f"starting with {_ANTHROPIC_MODEL_PREFIX!r} "
            f"(e.g. 'claude-sonnet-4-6')."
        )


class AnthropicAdapter(LLMPort):
    """Adapter dla Anthropic Messages API.

    W odróżnieniu od OpenAI, Claude nie ma natywnego JSON Mode — instruujemy
    go w prompcie (<output_schema>). Adapter rozpakowuje ewentualne wrappery
    ```json ... ``` z odpowiedzi.

    Przełączenie z OpenAI = `LLM_PROVIDER=anthropic` w `.env` + ANTHROPIC_API_KEY.

    Prompt caching: dla wystarczająco długich promptów (≥1024 tok ≈ 4096 znaków)
    włączamy ephemeral cache_control na content bloku. Cache działa per
    identyczny prefix — przy 2 cyklach dziennie ten sam template promptu
    daje cache hit na strukturalnej części (rola/instrukcje/schemat).

    Retry: 429 (rate limit) oraz 5xx/529 (Overloaded) i błędy połączenia są
    retry'owane z backoffem; inne błędy (np. 400) lecą natychmiast w górę.
    Sygnał kwot trafia do `QuotaMonitor` (WARNING po udanym retry, CRITICAL
    po wyczerpaniu) — tak samo jak w OpenAIAdapter.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        quota_monitor: QuotaMonitor | None = None,
    ) -> None:
        # Fail-fast na oczywiście-złym id modelu (pusty / zły provider prefix).
        validate_model_id(model)
        self._client = Anthropic(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._quota_monitor = quota_monitor

    @staticmethod
    def _build_user_content(prompt: str) -> Any:
        # Dla krótkich promptów cache_control byłby narzutem bez zysku
        # (Anthropic odrzuca cache bloki <1024 tok). Zwracamy raw string.
        # Typowane Any bo SDK używa wąskich TypedDict'ów (TextBlockParam),
        # których konstruowanie dictem dla cache_control jest poprawne runtime
        # ale niezgodne strukturalnie.
        if len(prompt) < _CACHE_MIN_CHARS:
            return prompt
        return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]

    def _call(self, prompt: str) -> str:
        response = self._call_with_retry(
            lambda: self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[
                    {"role": "user", "content": self._build_user_content(prompt)}
                ],
            )
        )
        return self._extract_text(response)

    def _call_with_retry(self, fn: Callable[[], T]) -> T:
        """Retry'uje wyłącznie błędy transient (429 / 5xx / connection).

        Sygnalizacja kwot:
        - sukces po >=1 retry → WARNING (na granicy limitu),
        - wyczerpanie retry → CRITICAL przed propagacją wyjątku.
        """
        attempt = 0
        while True:
            try:
                result = fn()
                if attempt > 0:
                    self._emit_quota_alert(
                        severity=QuotaSeverity.WARNING,
                        message=(
                            f"Anthropic {self._model} hit a transient error "
                            f"{attempt}× during a single call (retry succeeded)."
                        ),
                        action=(
                            "Transient 429/5xx — usually self-heals. If frequent, "
                            "raise COUNCIL_VOLATILITY_THRESHOLD or check Anthropic "
                            "status / your usage tier."
                        ),
                    )
                return result
            except (APIStatusError, APIConnectionError) as exc:
                if not self._is_transient(exc):
                    raise
                if attempt >= self._max_retries:
                    self._emit_quota_alert(
                        severity=QuotaSeverity.CRITICAL,
                        message=(
                            f"Anthropic {self._model} failed after "
                            f"{self._max_retries} retries "
                            f"({type(exc).__name__}) — call failed."
                        ),
                        action=(
                            "Check Anthropic status page / upgrade usage tier, or "
                            "switch LLM_PROVIDER=openai for this run."
                        ),
                    )
                    raise
                wait = self._compute_wait(exc, attempt)
                logger.warning(
                    "Anthropic transient error %s (attempt %d/%d) — sleeping %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    self._max_retries,
                    wait,
                )
                time.sleep(wait)
                attempt += 1

    @staticmethod
    def _is_transient(exc: APIStatusError | APIConnectionError) -> bool:
        """429 i 5xx (w tym 529 Overloaded) oraz błędy połączenia są transient.

        Błędy klienta (4xx poza 429 — np. 400 zły prompt, 401 zły klucz) nie
        mają sensu retry'ować: powtórzenie nie zmieni wyniku."""
        if isinstance(exc, APIConnectionError):
            return True
        status = getattr(exc, "status_code", None)
        return status == 429 or (isinstance(status, int) and status >= 500)

    def _compute_wait(
        self, exc: APIStatusError | APIConnectionError, attempt: int
    ) -> float:
        # Najpierw `retry-after` z odpowiedzi serwera — to jego rekomendacja.
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {}) or {}
            raw = headers.get("retry-after")
            if isinstance(raw, str):
                try:
                    return float(raw)
                except ValueError:
                    pass
        # Fallback: backoff wykładniczy (2s, 4s, 8s przy base=2.0).
        return float(self._backoff_base * (2**attempt))

    def _emit_quota_alert(
        self, severity: QuotaSeverity, message: str, action: str
    ) -> None:
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source=f"Anthropic ({self._model})",
                severity=severity,
                message=message,
                action=action,
                occurred_at=datetime.now(UTC),
            )
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        for block in response.content or []:
            if getattr(block, "type", None) == "text":
                return str(block.text)
        return ""

    @staticmethod
    def _strip_code_block(text: str) -> str:
        match = _CODE_BLOCK_RE.search(text)
        return match.group(1).strip() if match else text

    @staticmethod
    def _next_string_state(ch: str, escaped: bool) -> tuple[bool, bool]:
        """FSM wnętrza stringu JSON. Zwraca `(wciąż_w_stringu, escaped)`.

        Escape `\\` neutralizuje następny znak; niezescapowany `"` kończy string.
        """
        if escaped:
            return True, False
        if ch == "\\":
            return True, True
        if ch == '"':
            return False, False
        return True, False

    @classmethod
    def _extract_first_json_object(cls, text: str) -> str | None:
        """Wyłuskuje PIERWSZY zbalansowany obiekt {...} z tekstu.

        Claude nie ma natywnego JSON Mode — bywa, że owija JSON w prozę
        ("Here is the analysis: {...} hope that helps") albo dokleja komentarz
        po obiekcie. Skanujemy znak po znaku, licząc głębokość nawiasów i
        respektując stringi (nawias w stringu nie zmienia głębokości; escape
        `\\` nie kończy stringa). Zwraca substring obiektu albo None, gdy
        żaden zbalansowany {...} nie istnieje.
        """
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                in_string, escaped = cls._next_string_state(ch, escaped)
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def analyze(self, prompt: str) -> dict[str, Any]:
        text = self._call(prompt)
        if not text:
            raise ValueError("Anthropic returned empty content for analyze().")
        cleaned = self._strip_code_block(text).strip()
        try:
            return json.loads(cleaned)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            # Fallback: proza wokół JSON-a / komentarz po obiekcie. Wyłuskaj
            # pierwszy zbalansowany {...} i spróbuj sparsować jego.
            candidate = self._extract_first_json_object(cleaned)
            if candidate is not None:
                try:
                    return json.loads(candidate)  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                f"Anthropic returned invalid JSON: {cleaned[:200]!r}"
            ) from None

    def analyze_mistake(self, prompt: str) -> str:
        return self._call(prompt).strip()
