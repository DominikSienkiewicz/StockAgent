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

from datetime import UTC, datetime
from decimal import Decimal

from src.application.ports import MarketDataPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
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
        quota_monitor: QuotaMonitor | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._ticker_to_id = ticker_to_id or DEFAULT_TICKER_TO_ID
        self._vs_currency = vs_currency
        self._quota_monitor = quota_monitor
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
        if response.status_code == 429:
            # CoinGecko free tier throttluje ~5-15 req/min. Po wyczerpaniu
            # shared retry (build_session) zostaje twarde 429 — emitujemy
            # alert ZANIM raise_for_status wywali symbol z cyklu (mirror
            # FinnhubAdapter), żeby owner widział powód w bannerze raportu.
            retry_after = self._retry_after_hint(response)
            self._emit_quota_alert(
                severity=QuotaSeverity.CRITICAL,
                message=(
                    f"CoinGecko returned 429 for '{symbol}' — free tier "
                    "rate limit (~5-15 req/min) exhausted."
                ),
                action=(
                    "Slow the crypto polling (raise SYMBOL_THROTTLE_SECONDS), "
                    "reduce crypto SYMBOLS, or add a CoinGecko API key."
                    + retry_after
                ),
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

    @staticmethod
    def _retry_after_hint(response: object) -> str:
        """Wyciąga nagłówek `Retry-After` (jeśli jest) do treści akcji.

        CoinGecko bywa hojny z tym nagłówkiem przy 429; podanie go ownerowi
        daje konkretną liczbę sekund do odczekania zamiast zgadywania.
        """
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("Retry-After")
        if retry_after:
            return f" Retry-After hint: {retry_after}s."
        return ""

    def _emit_quota_alert(
        self, severity: QuotaSeverity, message: str, action: str
    ) -> None:
        """Wystawia alert do QuotaMonitor, jeśli ten jest skonfigurowany."""
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source="CoinGecko",
                severity=severity,
                message=message,
                action=action,
                occurred_at=datetime.now(UTC),
            )
        )
