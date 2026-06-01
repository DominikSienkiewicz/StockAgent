"""Kompozyt MarketDataPort — routing per symbol (crypto vs equity).

Pozwala wstrzyknąć JEDEN port do AnalyzeMarketUseCase, a pod spodem
wybierać między FinnhubAdapter a CoinGeckoAdapter w zależności od tego,
czy symbol jest na liście krypto. Use case i graf nie wiedzą o podziale —
klasyfikacja jest detalem infrastruktury.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.application.ports import MarketDataPort
from src.domain.value_objects import Money


class RoutingMarketDataPort(MarketDataPort):
    def __init__(
        self,
        *,
        equity: MarketDataPort,
        crypto: MarketDataPort,
        crypto_symbols: Iterable[str],
    ) -> None:
        self._equity = equity
        self._crypto = crypto
        self._crypto_symbols = set(crypto_symbols)

    def get_current_price(self, symbol: str) -> Money:
        if symbol in self._crypto_symbols:
            return self._crypto.get_current_price(symbol)
        return self._equity.get_current_price(symbol)
