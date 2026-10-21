"""Faza cech rynkowych cyklu (ML-31) — raz na cykl, PRZED grafem.

Za flagą `market_features_enabled` use case zbiera dla każdego symbolu cyklu kurs
(port rynku) i historię snapshotów (`price_snapshots`, darmowy odczyt Supabase),
a domena liczy z nich cechy point-in-time (`price_features`). Cechy trafiają do
stanu grafu i dalej do `feature_snapshots` jako zestaw cech w wersji 4.

FinOps: zero płatnych wywołań. Kurs pobrany tutaj zapamiętuje `CycleQuoteCache`,
więc `check_price` tego samego cyklu go reużywa — liczba requestów o cenę się nie
zmienia, przesuwa się tylko ich chwila na początek cyklu.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from src.application.ports import MarketDataPort, RepositoryPort
from src.domain.price_features import PriceFeatures, price_features
from src.domain.value_objects import Money

logger = logging.getLogger(__name__)

# Okno historii snapshotów: 20 dziennych log-zwrotów potrzebuje 21 zamknięć
# (~30 dni kalendarzowych), plus zapas na święta i pojedyncze dni bez cyklu.
PRICE_HISTORY_DAYS = 45


class CycleQuoteCache(MarketDataPort):
    """Kurs symbolu pobrany najwyżej raz na cykl (faza cech i `check_price`).

    Pamięta tylko udane odczyty — symbol, którego kurs padł w fazie cech,
    `check_price` pyta ponownie, dokładnie jak bez tej fazy. Bezpieczny dla
    wątków (ścieżki równoległe cyklu)."""

    def __init__(self, delegate: MarketDataPort) -> None:
        self._delegate = delegate
        self._quotes: dict[str, Money] = {}
        self._lock = threading.Lock()

    def get_current_price(self, symbol: str) -> Money:
        with self._lock:
            cached = self._quotes.get(symbol)
        if cached is not None:
            return cached
        quote = self._delegate.get_current_price(symbol)
        with self._lock:
            return self._quotes.setdefault(symbol, quote)


def _symbol_price_features(
    symbol: str,
    *,
    market_port: MarketDataPort,
    repository: RepositoryPort,
    as_of: datetime,
) -> PriceFeatures | None:
    """Cechy cenowe symbolu albo None, gdy nie ma kursu (symbol bez cech rynkowych).

    Błąd odczytu historii nie zabiera kursu: cechy wyjdą wtedy puste (None)."""
    try:
        quote = market_port.get_current_price(symbol)
    except Exception:
        logger.warning(
            "ML-31: brak kursu %s w fazie cech — symbol bez cech rynkowych", symbol,
            exc_info=True,
        )
        return None
    try:
        history = repository.get_price_history(symbol, PRICE_HISTORY_DAYS)
    except Exception:
        logger.warning(
            "ML-31: brak historii snapshotów %s — cechy cenowe puste", symbol, exc_info=True
        )
        history = []
    points = [(moment, float(price.amount)) for moment, price in history]
    points.append((as_of, float(quote.amount)))
    return price_features(points, as_of)


def collect_market_features(
    symbols: Sequence[str],
    *,
    market_port: MarketDataPort,
    repository: RepositoryPort,
    as_of: datetime,
) -> dict[str, dict[str, Any]]:
    """Cechy rynkowe każdego symbolu cyklu (`PRICE_FEATURE_NAMES`, None = brak).

    Pojedynczy błąd per symbol nie przerywa fazy — symbol dostaje puste cechy."""
    features: dict[str, dict[str, Any]] = {}
    for symbol in dict.fromkeys(symbols):
        computed = _symbol_price_features(
            symbol, market_port=market_port, repository=repository, as_of=as_of
        )
        features[symbol] = (computed or PriceFeatures()).as_features()
    return features
