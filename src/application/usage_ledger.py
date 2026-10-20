"""FIN-13: pamięć tokenów przebiegu za portem `UsageSink`.

Adapter LLM / embeddingów zgłasza samo `LLMUsage` (dostawca, model, tokeny).
Kto za to płaci, dopisuje zakres `usage_scope`, który application ustawia wokół
płatnego wywołania: źródło cennika cyklu (`llm`, `council_llm`, `embedding`),
węzeł grafu albo job Slow Loopa i symbol. Zakres żyje w `contextvars`, więc
symbole idące równolegle (`symbol_concurrency`) mają każdy własny; wątki person
rady dostają kopię kontekstu od adaptera rady.

Ledger to tylko pamięć przebiegu: sekcja „Koszt cyklu” liczy z niego koszt
z tokenów bez żadnej flagi, a zapis `llm_calls` robi DI za `cost_ledger_enabled`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import date

from src.application.ports import UsageSink
from src.domain.finops import (
    LLMCallEntry,
    LLMUsage,
    ModelPrice,
    price_usage,
    token_cost_by_source,
)

# Zgłoszenie spoza zakresu (wywołanie, którego application nie otoczyła) nie ginie:
# trafia do ledgera z tym znacznikiem, ale nie podmienia żadnego źródła w mailu.
UNATTRIBUTED = "unattributed"


@dataclass(frozen=True)
class UsageScope:
    """Atrybucja płatnego wywołania; None = pole niepodane w żadnym zakresie."""

    source: str | None = None
    node: str | None = None
    symbol: str | None = None


_NO_SCOPE = UsageScope()
_CURRENT_SCOPE: ContextVar[UsageScope | None] = ContextVar("llm_usage_scope", default=None)


def _current_scope() -> UsageScope:
    return _CURRENT_SCOPE.get() or _NO_SCOPE


@contextmanager
def usage_scope(
    *, source: str | None = None, node: str | None = None, symbol: str | None = None
) -> Iterator[None]:
    """Nakłada podane pola na bieżący zakres na czas bloku; pozostałe dziedziczy.

    Zagnieżdżenie pozwala np. węzłowi rady ustawić węzeł i symbol, a dekoratorowi
    `MeteredLLMPort` samo źródło cennika."""
    current = _current_scope()
    overrides = {
        name: value
        for name, value in (("source", source), ("node", node), ("symbol", symbol))
        if value is not None
    }
    token = _CURRENT_SCOPE.set(replace(current, **overrides))
    try:
        yield
    finally:
        _CURRENT_SCOPE.reset(token)


@dataclass(frozen=True)
class AttributedUsage:
    """Zgłoszenie z atrybucją z zakresu w chwili `record`."""

    source: str
    node: str
    symbol: str | None
    usage: LLMUsage


class UsageLedger(UsageSink):
    """Wątkowo bezpieczny zbiór zgłoszeń jednego przebiegu (Fast albo Slow Loop)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[AttributedUsage] = []

    def record(self, usage: LLMUsage) -> None:
        scope = _current_scope()
        attributed = AttributedUsage(
            source=scope.source or UNATTRIBUTED,
            node=scope.node or UNATTRIBUTED,
            symbol=scope.symbol,
            usage=usage,
        )
        with self._lock:
            self._records.append(attributed)

    def records(self) -> tuple[AttributedUsage, ...]:
        with self._lock:
            return tuple(self._records)

    def token_costs(self, price_table: Mapping[str, ModelPrice]) -> dict[str, float]:
        """Koszt z tokenów per źródło cennika; reguły wyceny trzyma domena."""
        return token_cost_by_source(
            ((record.source, record.usage) for record in self.records()), price_table
        )

    def entries(
        self,
        *,
        run_id: str,
        price_table: Mapping[str, ModelPrice],
        price_as_of: date | None,
    ) -> list[LLMCallEntry]:
        """Wiersze `llm_calls` w kolejności zgłoszeń, każdy wyceniony osobno."""
        entries: list[LLMCallEntry] = []
        for record in self.records():
            cost = price_usage(record.usage, price_table)
            entries.append(
                LLMCallEntry(
                    run_id=run_id,
                    node=record.node,
                    symbol=record.symbol,
                    usage=record.usage,
                    cost_usd=cost,
                    price_as_of=price_as_of if cost is not None else None,
                )
            )
        return entries
