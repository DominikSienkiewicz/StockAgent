"""Adapter CoinGecko — implementacja MarketDataPort dla krypto.

CoinGecko free tier:
- bez klucza API
- bez znaczącego rate-limitu dla niskiego wolumenu zapytań
- endpoint: GET /api/v3/simple/price?ids={ids}&vs_currencies=usd

Mapping w adapterze: user trzyma w configu czyste tickery (BTC, ETH),
adapter przekłada je na CoinGecko coin IDs (bitcoin, ethereum).
Domyślny słownik pokrywa MVP; do dodania kolejnej monety wystarczy
przekazać własny `ticker_to_id` przy konstrukcji.
"""

from __future__ import annotations

from decimal import Decimal

from src.application.ports import MarketDataPort
from src.domain.value_objects import Money
from src.infrastructure.adapters._http import build_session

DEFAULT_BASE_URL = "https://api.coingecko.com/api/v3"
DEFAULT_TIMEOUT = 10
DEFAULT_VS_CURRENCY = "usd"

# Tickery → CoinGecko coin IDs. Rozszerzaj przez konstruktor zamiast
# modyfikować źródło — zachowuje stabilność domyślnego defaulta.
DEFAULT_TICKER_TO_ID: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
}


class CoinGeckoAdapter(MarketDataPort):
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        ticker_to_id: dict[str, str] | None = None,
        vs_currency: str = DEFAULT_VS_CURRENCY,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._ticker_to_id = ticker_to_id or DEFAULT_TICKER_TO_ID
        self._vs_currency = vs_currency
        self._session = build_session()

    def get_current_price(self, symbol: str) -> Money:
        coin_id = self._ticker_to_id.get(symbol)
        if coin_id is None:
            raise ValueError(
                f"Crypto ticker '{symbol}' is not mapped to a CoinGecko coin "
                f"id. Add it to ticker_to_id (known: "
                f"{sorted(self._ticker_to_id)})."
            )
        response = self._session.get(
            f"{self._base_url}/simple/price",
            params={"ids": coin_id, "vs_currencies": self._vs_currency},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        coin_block = payload.get(coin_id)
        if not isinstance(coin_block, dict):
            raise ValueError(
                f"CoinGecko returned no data for '{symbol}' "
                f"(coin_id={coin_id})."
            )
        price = coin_block.get(self._vs_currency)
        if price is None:
            raise ValueError(
                f"CoinGecko response for '{symbol}' missing "
                f"'{self._vs_currency}' field."
            )
        return Money(Decimal(str(price)))
