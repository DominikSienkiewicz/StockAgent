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
from typing import Any, TypeVar

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

from src.application.ports import EmbeddingPort, UsageSink
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaSeverity
from src.infrastructure.llm._quota import QuotaEmitter
from src.infrastructure.llm._retry import RetryPolicy, call_with_retry
from src.infrastructure.llm._usage import openai_embedding_usage, report_usage

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

    FIN-13: tokeny requestu (`usage.prompt_tokens`) idą do `usage_sink`; trafienie
    w cache LRU nie jest requestem, więc nie ma czego zgłosić.
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
        usage_sink: UsageSink | None = None,
    ) -> None:
        # Retry żyje tylko w pętli adaptera — retry SDK mnożyłby płatne próby (NIEZ-07).
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self._model = model
        self._quota_monitor = quota_monitor
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        # In-process bounded LRU: sha256(znormalizowany tekst) → wektor.
        # OrderedDict zachowuje kolejność wstawień; przekroczenie pojemności
        # eksmituje najstarszy wpis (popitem(last=False)).
        self._cache_size = cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._usage_sink = usage_sink

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
        self._report_usage(response)
        vector = list(response.data[0].embedding)
        self._store_in_cache(key, vector)
        return vector

    def _report_usage(self, response: Any) -> None:
        if self._usage_sink is not None:
            report_usage(self._usage_sink, openai_embedding_usage(response, self._model))

    def _store_in_cache(self, key: str, vector: list[float]) -> None:
        if self._cache_size <= 0:
            return
        self._cache[key] = list(vector)
        self._cache.move_to_end(key)
        # Bounded: po przekroczeniu pojemności wyrzucamy najstarszy wpis.
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _call_with_retry(self, fn: Callable[[], T]) -> T:
        """Retry'uje 429, 5xx i błędy połączenia. Honoruje retry-after — wspólna pętla
        `call_with_retry` (ARCH-29); błąd końcowy wychodzi jako wyjątek SDK bez zmian.

        Sygnalizacja kwot:
        - sukces po >=1 retry → WARNING (na granicy TPM)
        - wyczerpanie retry → CRITICAL przed propagacją wyjątku
        """
        policy = RetryPolicy(
            retryable=(RateLimitError, InternalServerError, APIConnectionError),
            max_retries=self._max_retries,
            backoff_base=self._backoff_base,
            logger=logger,
            label="OpenAI embeddings",
            reason=_reason,
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
                f"Embeddingi OpenAI ({self._model}): limit TPM trafiony "
                f"{attempts}× w jednym wywołaniu (ponowienie się udało)."
            ),
            action=(
                "Podnieś SYMBOL_THROTTLE_SECONDS, żeby rozłożyć wywołania "
                "embeddingów, albo podnieś tier konta OpenAI."
            ),
            log_message=(
                f"OpenAI embeddings ({self._model}) hit TPM limit "
                f"{attempts}× during a single call (retry succeeded)."
            ),
        )

    def _report_exhausted(self, exc: Exception) -> None:
        reason = _reason(exc)
        self._quota().emit(
            severity=QuotaSeverity.CRITICAL,
            message=(
                f"Embeddingi OpenAI ({self._model}) zwróciły {reason} "
                f"po {self._max_retries} ponowieniach — wywołanie nie "
                "powiodło się."
            ),
            action=(
                "Podnieś tier konta OpenAI albo SYMBOL_THROTTLE_SECONDS, "
                "żeby zmniejszyć presję zapytań o embeddingi."
            ),
            log_message=(
                f"OpenAI embeddings ({self._model}) returned {reason} "
                f"after {self._max_retries} retries — call failed."
            ),
        )

    def _quota(self) -> QuotaEmitter:
        return QuotaEmitter(f"OpenAI embeddings ({self._model})", self._quota_monitor, logger)


def _reason(exc: Exception) -> str:
    return "429" if isinstance(exc, RateLimitError) else type(exc).__name__
