"""Wspólna pętla ponowień adapterów LLM (ARCH-29).

Anthropic, OpenAI chat i embeddingi OpenAI miały po kopii tej samej pętli: ponowienie
błędu przejściowego po `retry-after` serwera albo backoffie wykładniczym (sufit
`MAX_RETRY_WAIT_S` — job ma 15 minut), raport po udanym ponowieniu i po wyczerpaniu
ponowień. Pętla jest tu raz; adapter podaje, co łapie, co uznaje za przejściowe i jak
wypuszcza błąd końcowy — semantyka ponowień każdego adaptera zostaje bez zmian:

- Anthropic: status 429 albo ≥ 500 (w tym 529 Overloaded) i błąd połączenia,
- OpenAI chat i embeddingi: `RateLimitError`, `InternalServerError`, błąd połączenia,
- chat (oba) tłumaczy błąd końcowy na taksonomię portów, embeddingi wypuszczają
  wyjątek SDK bez zmian.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

MAX_RETRY_WAIT_S = 20.0


def retry_after_seconds(exc: BaseException, attempt: int, backoff_base: float) -> float:
    """Czekanie przed ponowieniem `attempt` (od 0): najpierw `retry-after` z odpowiedzi
    serwera — to jego rekomendacja — a bez niego `backoff_base * 2**attempt`; jedno i
    drugie najwyżej `MAX_RETRY_WAIT_S`."""
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        raw = headers.get("retry-after")
        if isinstance(raw, str):
            try:
                return min(float(raw), MAX_RETRY_WAIT_S)
            except ValueError:
                pass
    return min(float(backoff_base * (2**attempt)), MAX_RETRY_WAIT_S)


def _always(_exc: Exception) -> bool:
    return True


def _class_name(exc: Exception) -> str:
    return type(exc).__name__


@dataclass(frozen=True)
class RetryPolicy:
    """Co pętla łapie i jak loguje ponowienie.

    `retryable` — wyjątki, które pętla w ogóle ogląda (reszta leci bez zmian);
    `is_transient` — który z nich ponawiać (nieprzejściowy wychodzi od razu, bez
    raportu); `translate` — postać błędu końcowego (None = ten sam wyjątek);
    `label` i `reason` składają linię WARNING ponowienia w logu adaptera."""

    retryable: tuple[type[Exception], ...]
    max_retries: int
    backoff_base: float
    logger: logging.Logger
    label: str
    is_transient: Callable[[Exception], bool] = _always
    reason: Callable[[Exception], str] = _class_name
    translate: Callable[[Exception], Exception] | None = field(default=None)


def call_with_retry[T](
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None],
    on_recovered: Callable[[int], None],
    on_exhausted: Callable[[Exception], None],
) -> T:
    """Woła `fn`, ponawiając błąd przejściowy najwyżej `policy.max_retries` razy.

    Sukces po co najmniej jednym ponowieniu woła `on_recovered(liczba_ponowień)`,
    wyczerpanie ponowień — `on_exhausted(błąd)` tuż przed wypuszczeniem błędu."""
    attempt = 0
    while True:
        try:
            result = fn()
        except policy.retryable as exc:
            transient = policy.is_transient(exc)
            if transient and attempt < policy.max_retries:
                wait = retry_after_seconds(exc, attempt, policy.backoff_base)
                policy.logger.warning(
                    "%s %s (attempt %d/%d) — sleeping %.1fs",
                    policy.label,
                    policy.reason(exc),
                    attempt + 1,
                    policy.max_retries,
                    wait,
                )
                sleep(wait)
                attempt += 1
                continue
            if transient:
                on_exhausted(exc)
            if policy.translate is None:
                raise
            raise policy.translate(exc) from None
        if attempt > 0:
            on_recovered(attempt)
        return result
