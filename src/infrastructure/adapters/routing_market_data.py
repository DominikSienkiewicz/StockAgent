"""Kompozyt MarketDataPort — routing per symbol (crypto vs equity).

Pozwala wstrzyknąć JEDEN port do AnalyzeMarketUseCase, a pod spodem
wybierać między FinnhubAdapter a CoinGeckoAdapter w zależności od tego,
czy symbol jest na liście krypto. Use case i graf nie wiedzą o podziale —
klasyfikacja jest detalem infrastruktury.

Tą samą drogą idzie dzienna zmiana ceny (ALR-02): watch szoku dostaje router jako
`DailyChangePort`, więc nie potrzebuje snapshotów Fast Loopa.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.application.ports import DailyChangePort, MarketDataPort
from src.domain.shock import DailyChange
from src.domain.value_objects import Money


class RoutingMarketDataPort(MarketDataPort, DailyChangePort):
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
        return self._source(symbol).get_current_price(symbol)

    def get_daily_change(self, symbol: str) -> DailyChange:
        source = self._source(symbol)
        if not isinstance(source, DailyChangePort):
            raise ValueError(f"Price source for '{symbol}' does not report a daily change.")
        return source.get_daily_change(symbol)

    def _source(self, symbol: str) -> MarketDataPort:
        return self._crypto if symbol in self._crypto_symbols else self._equity
