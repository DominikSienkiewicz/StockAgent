"""Faza cech rynkowych cyklu (ML-31, ML-32, DAT-22) — raz na cykl, PRZED grafem.

Za flagą `market_features_enabled` use case zbiera dla każdego symbolu cyklu kurs
(port rynku) i historię snapshotów (`price_snapshots`, darmowy odczyt Supabase),
a domena liczy z nich cechy point-in-time (`price_features`). Po zebraniu kursów
WSZYSTKICH symboli domena dokłada cechy cross-sectional (`cross_section_features`):
rangę ruchu w uniwersum, ruch ponad IVV i medianę klasy, szerokość rynku. Cechy
trafiają do stanu grafu i dalej do `feature_snapshots` jako zestaw cech w wersji 4,
razem z cechami makro cyklu z FRED (DAT-22, pobranymi raz przez DI).

FinOps: zero płatnych wywołań. Kurs pobrany tutaj zapamiętuje `CycleQuoteCache`,
więc `check_price` tego samego cyklu go reużywa — liczba requestów o cenę się nie
zmienia, przesuwa się tylko ich chwila na początek cyklu.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from src.application.ports import MarketDataPort, RepositoryPort
from src.domain.cross_section import CrossSectionFeatures, SymbolMove, cross_section_features
from src.domain.price_features import PriceFeatures, price_features
from src.domain.value_objects import AssetType, Money

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


def _universe_moves(
    prices: Mapping[str, PriceFeatures],
    asset_types: Mapping[str, AssetType],
    gate_thresholds: Mapping[AssetType, float],
) -> dict[str, SymbolMove]:
    """ML-32: uniwersum cyklu — symbole ze znanym ret_1. Symbol bez kursu albo bez
    wczorajszego zamknięcia nie wchodzi do rangi, mediany ani szerokości rynku."""
    moves: dict[str, SymbolMove] = {}
    for symbol, features in prices.items():
        if features.ret_1 is None:
            continue
        asset_type = asset_types.get(symbol, AssetType.STOCK)
        moves[symbol] = SymbolMove(
            ret_1=features.ret_1,
            asset_class=asset_type.value,
            gate_threshold=gate_thresholds[asset_type],
        )
    return moves


def collect_market_features(
    symbols: Sequence[str],
    *,
    market_port: MarketDataPort,
    repository: RepositoryPort,
    as_of: datetime,
    asset_types: Mapping[str, AssetType],
    gate_thresholds: Mapping[AssetType, float],
    macro_features: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Cechy rynkowe każdego symbolu cyklu (cenowe + cross-sectional + makro, None = brak).

    `asset_types` (brak = STOCK) wyznacza klasę do mediany, a `gate_thresholds`
    bazowy próg bramki klasy (bez mnożników reżimu i earnings) do szerokości rynku.
    `macro_features` (DAT-22) to cechy makro cyklu — wspólne dla wszystkich symboli.
    Pojedynczy błąd per symbol nie przerywa fazy — symbol dostaje puste cechy."""
    prices: dict[str, PriceFeatures] = {}
    for symbol in dict.fromkeys(symbols):
        computed = _symbol_price_features(
            symbol, market_port=market_port, repository=repository, as_of=as_of
        )
        prices[symbol] = computed or PriceFeatures()
    cross = cross_section_features(_universe_moves(prices, asset_types, gate_thresholds))
    return {
        symbol: {
            **features.as_features(),
            **cross.get(symbol, CrossSectionFeatures()).as_features(),
            **(macro_features or {}),
        }
        for symbol, features in prices.items()
    }
