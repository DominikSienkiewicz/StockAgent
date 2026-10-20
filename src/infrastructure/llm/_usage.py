"""FIN-13: odczyt tokenów z odpowiedzi SDK Anthropic i OpenAI do `LLMUsage`.

Każdy dostawca liczy inaczej, a `LLMUsage` ma rozłączne kategorie:
  - Anthropic: `input_tokens` NIE obejmuje cache, odczyt i zapis cache są osobno
    (`cache_read_input_tokens`, `cache_creation_input_tokens`);
  - OpenAI chat: `prompt_tokens` OBEJMUJE tokeny z cache
    (`prompt_tokens_details.cached_tokens`), więc wejście po pełnej stawce to różnica;
  - OpenAI embeddings: samo `prompt_tokens`.
Brak usage albo pole, które nie jest liczbą całkowitą, to brak rekordu — nigdy
wyjątek: księgowanie nie może zabić zapłaconego już wywołania.
"""

from __future__ import annotations

import logging
from typing import Any

from src.application.ports import UsageSink
from src.domain.finops import LLMUsage

logger = logging.getLogger(__name__)

ANTHROPIC = "anthropic"
OPENAI = "openai"


def _count(container: Any, name: str) -> int | None:
    value = getattr(container, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def anthropic_usage(response: Any, model: str) -> LLMUsage | None:
    usage = getattr(response, "usage", None)
    input_tokens = _count(usage, "input_tokens")
    output_tokens = _count(usage, "output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    return LLMUsage(
        provider=ANTHROPIC,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=_count(usage, "cache_read_input_tokens") or 0,
        cache_write_tokens=_count(usage, "cache_creation_input_tokens") or 0,
    )


def openai_chat_usage(response: Any, model: str) -> LLMUsage | None:
    usage = getattr(response, "usage", None)
    prompt_tokens = _count(usage, "prompt_tokens")
    completion_tokens = _count(usage, "completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    cached = min(_count(details, "cached_tokens") or 0, prompt_tokens)
    return LLMUsage(
        provider=OPENAI,
        model=model,
        input_tokens=prompt_tokens - cached,
        output_tokens=completion_tokens,
        cache_read_tokens=cached,
    )


def openai_embedding_usage(response: Any, model: str) -> LLMUsage | None:
    prompt_tokens = _count(getattr(response, "usage", None), "prompt_tokens")
    if prompt_tokens is None:
        return None
    return LLMUsage(provider=OPENAI, model=model, input_tokens=prompt_tokens, output_tokens=0)


def report_usage(sink: UsageSink | None, usage: LLMUsage | None) -> None:
    """Oddaje zużycie do portu; błąd odbiorcy tylko loguje (request już zapłacony)."""
    if sink is None or usage is None:
        return
    try:
        sink.record(usage)
    except Exception:
        logger.warning("usage sink failed for %s %s", usage.provider, usage.model, exc_info=True)
