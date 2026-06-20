"""Adapter dla OpenAI Embeddings API.

`text-embedding-3-small` zwraca wektory 1536-wymiarowe — dokładnie tyle,
ile ma kolumna `embedding VECTOR(1536)` w `prediction_logs` (pgvector).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeVar

from openai import OpenAI, RateLimitError

from src.application.ports import EmbeddingPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity

DEFAULT_MODEL = "text-embedding-3-small"  # 1536 wymiarów, tani, szybki
# GHA fast loop ma 15 min hard timeout. SDK domyślnie czeka 600s na response —
# zawieszone wywołanie embeddingów (płatny call na hot pathcie) mogłoby zjeść
# cały budżet cyklu. 30s starcza dla normalnego embeddings response.
DEFAULT_TIMEOUT = 30.0
# Retry-on-429 — TPM limit potrafi się odbić nagle przy 30+ symbolach.
# Backoff: 2s, 4s, 8s (chyba że SDK zwróci `retry-after` → wtedy honorujemy go).
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0
# Cross-cycle cache: news_summary zmienia się wolno (cadence 12h), więc ten sam
# tekst trafia do embed() w wielu cyklach. Bez cache'a każdy cykl re-embeduje go
# od zera — czysty re-burn płatnego calla. Bounded LRU keyed po sha256 tekstu
# (znormalizowanego) eliminuje to bez ryzyka nieograniczonego wzrostu pamięci.
# 256 wpisów ≈ kilka cykli × kilkadziesiąt symboli — z dużym zapasem.
DEFAULT_CACHE_SIZE = 256

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OpenAIEmbeddingAdapter(EmbeddingPort):
    """Adapter dla OpenAI Embeddings API.

    Embeddingi to płatny call na hot pathcie agenta — dostają tę samą
    odporność co OpenAIAdapter: timeout, retry na 429 z backoffem
    (honorującym `retry-after`) i sygnalizację kwot do `QuotaMonitor`
    (WARNING po udanym retry, CRITICAL po wyczerpaniu retry).
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        quota_monitor: QuotaMonitor | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._quota_monitor = quota_monitor
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        # In-process bounded LRU: sha256(znormalizowany tekst) → wektor.
        # OrderedDict zachowuje kolejność wstawień; przekroczenie pojemności
        # eksmituje najstarszy wpis (popitem(last=False)).
        self._cache_size = cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    @staticmethod
    def _cache_key(text: str) -> str:
        # Normalizacja: strip otaczającego whitespace'u — "x" i "  x  " to ten
        # sam content embeddingu (i tak wysyłamy `text`, ale klucz musi być
        # stabilny względem trywialnych różnic formatowania newsów).
        normalized = text.strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> list[float]:
        # Pusty tekst → zerowy wektor (AV nie zwrócił newsów dla tego symbolu).
        # Nie cache'ujemy — to gałąź bez płatnego calla, więc bez zysku.
        if not text or not text.strip():
            return []

        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            # Cache hit — pomijamy płatny call. Odświeżamy pozycję LRU.
            self._cache.move_to_end(key)
            return list(cached)

        response = self._call_with_retry(
            lambda: self._client.embeddings.create(model=self._model, input=text)
        )
        vector = list(response.data[0].embedding)
        self._store_in_cache(key, vector)
        return vector

    def _store_in_cache(self, key: str, vector: list[float]) -> None:
        if self._cache_size <= 0:
            return
        self._cache[key] = list(vector)
        self._cache.move_to_end(key)
        # Bounded: po przekroczeniu pojemności wyrzucamy najstarszy wpis.
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

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
                            f"OpenAI embeddings ({self._model}) hit TPM limit "
                            f"{attempt}× during a single call (retry succeeded)."
                        ),
                        action=(
                            "Raise SYMBOL_THROTTLE_SECONDS to spread embedding "
                            "calls, or upgrade your OpenAI usage tier."
                        ),
                    )
                return result
            except RateLimitError as exc:
                if attempt >= self._max_retries:
                    self._emit_quota_alert(
                        severity=QuotaSeverity.CRITICAL,
                        message=(
                            f"OpenAI embeddings ({self._model}) returned 429 "
                            f"after {self._max_retries} retries — call failed."
                        ),
                        action=(
                            "Upgrade your OpenAI usage tier or raise "
                            "SYMBOL_THROTTLE_SECONDS to reduce embedding "
                            "request pressure."
                        ),
                    )
                    raise
                wait = self._compute_wait(exc, attempt)
                logger.warning(
                    "OpenAI embeddings 429 (attempt %d/%d) — sleeping %.1fs",
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
                source=f"OpenAI embeddings ({self._model})",
                severity=severity,
                message=message,
                action=action,
                occurred_at=datetime.now(UTC),
            )
        )
