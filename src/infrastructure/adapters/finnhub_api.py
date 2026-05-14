from __future__ import annotations

from decimal import Decimal

from src.application.ports import MarketDataPort
from src.domain.value_objects import Money
from src.infrastructure.adapters._http import build_session

DEFAULT_BASE_URL = "https://finnhub.io/api/v1"
DEFAULT_TIMEOUT = 10  # sekundy


class FinnhubAdapter(MarketDataPort):
    """Adapter dla Finnhub Quote API.

    Endpoint: GET /quote?symbol={symbol}&token={api_key}
    Response: {"c": current_price, "d": change, "dp": percent_change, ...}
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = build_session()

    def get_current_price(self, symbol: str) -> Money:
        response = self._session.get(
            f"{self._base_url}/quote",
            params={"symbol": symbol, "token": self._api_key},
            timeout=self._timeout,
        )
        if response.status_code == 403:
            raise ValueError(
                f"Ticker '{symbol}' is not supported by Finnhub free tier (403 Forbidden). "
                "LSE tickers with dot notation (e.g. CSPX.L) require a paid plan."
            )
        response.raise_for_status()

        payload = response.json()
        price = payload.get("c")
        if not price:
            raise ValueError(
                f"Finnhub returned no price for symbol '{symbol}' "
                f"(payload: {payload}). Ticker may be invalid or delisted."
            )

        return Money(Decimal(str(price)))
