"""DX-04: ceny cyklu offline z fixture'a scenariusza — bez kluczy i bez sieci.

Jeden egzemplarz na cykl: composition root podaje mapę symbol → cena tego cyklu
(`data/offline/scenario.json`). Symbol spoza mapy to `NotFound` — cykl liczy go
jako błąd symbolu jednym WARNING-iem, jak niedostępny `/quote` dostawcy."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from src.application.errors import NotFound
from src.application.ports import MarketDataPort
from src.domain.value_objects import Money


class FixtureMarketData(MarketDataPort):
    """`MarketDataPort` ze stałej mapy cen jednego cyklu."""

    def __init__(self, prices: Mapping[str, Decimal]) -> None:
        self._prices = dict(prices)

    def get_current_price(self, symbol: str) -> Money:
        price = self._prices.get(symbol)
        if price is None:
            raise NotFound(f"offline fixture has no price for {symbol}")
        return Money(price)
