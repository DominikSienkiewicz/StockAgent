"""Adapter CoinGecko — implementacja MarketDataPort i DailyChangePort (ALR-02) dla krypto.

CoinGecko free tier:
- bez klucza API
- bez znaczącego rate-limitu dla niskiego wolumenu zapytań
- endpoint: GET /api/v3/simple/price?ids={ids}&vs_currencies=usd

Mapping w adapterze: user trzyma w configu czyste tickery (BTC, ETH),
adapter przekłada je na CoinGecko coin IDs (bitcoin, ethereum).
Domyślny słownik pokrywa MVP; do dodania kolejnej monety wystarczy
przekazać własny `ticker_to_id` przy konstrukcji.

ARCH-33: błędy wychodzą w taksonomii portów — sieć i 5xx jako
`ProviderUnavailable`, 429 jako `QuotaExhausted`, nieznana moneta jako `NotFound`
(wciąż `ValueError`), niepełna albo nieparsowalna treść jako `MalformedPayload`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from src.application.errors import MalformedPayload, NotFound
from src.application.ports import DailyChangePort, MarketDataPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.shock import DailyChange
from src.domain.value_objects import Money
from src.infrastructure.adapters._http import (
    build_session,
    json_payload,
    raise_for_status_translated,
    translated_get,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.coingecko.com/api/v3"
DEFAULT_TIMEOUT = 10
DEFAULT_VS_CURRENCY = "usd"

# Tickery → CoinGecko coin IDs. Rozszerzaj przez konstruktor zamiast
# modyfikować źródło — zachowuje stabilność domyślnego defaulta.
DEFAULT_TICKER_TO_ID: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
}


class CoinGeckoCoinNotFound(NotFound, ValueError):
    """Moneta bez mapowania na coin id albo nieznana CoinGecko. `ValueError`
    zostaje bazą dla dotychczasowych wywołujących (ARCH-33)."""


class CoinGeckoAdapter(MarketDataPort, DailyChangePort):
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
        coin_block = self._fetch_coin_block(symbol, {})
        price = coin_block.get(self._vs_currency)
        if price is None:
            raise MalformedPayload(
                f"CoinGecko response for '{symbol}' missing "
                f"'{self._vs_currency}' field."
            )
        return Money(Decimal(str(price)))

    def get_daily_change(self, symbol: str) -> DailyChange:
        """ALR-02: zmiana w oknie 24 h (procent → ułamek) i chwila ostatniej
        aktualizacji kursu — to samo darmowe /simple/price co cena."""
        coin_block = self._fetch_coin_block(
            symbol, {"include_24hr_change": "true", "include_last_updated_at": "true"}
        )
        percent = coin_block.get(f"{self._vs_currency}_24h_change")
        stamp = coin_block.get("last_updated_at")
        if percent is None or not stamp:
            raise MalformedPayload(
                f"CoinGecko response for '{symbol}' missing the 24h change "
                f"(has_change={percent is not None}, has_updated_at={bool(stamp)})."
            )
        return DailyChange(
            fraction=Decimal(str(percent)) / 100,
            as_of=datetime.fromtimestamp(int(stamp), tz=UTC),
        )

    def _fetch_coin_block(self, symbol: str, extra_params: dict[str, str]) -> dict[str, Any]:
        coin_id = self._ticker_to_id.get(symbol)
        if coin_id is None:
            raise CoinGeckoCoinNotFound(
                f"Crypto ticker '{symbol}' is not mapped to a CoinGecko coin "
                f"id. Add it to ticker_to_id (known: "
                f"{sorted(self._ticker_to_id)})."
            )
        response = translated_get(
            self._session,
            f"{self._base_url}/simple/price",
            params={"ids": coin_id, "vs_currencies": self._vs_currency, **extra_params},
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
                    f"CoinGecko zwrócił 429 dla '{symbol}' — wyczerpany darmowy "
                    "limit (~5-15 zapytań/min)."
                ),
                action=(
                    "Spowolnij odpytywanie krypto (podnieś SYMBOL_THROTTLE_SECONDS), "
                    "zmniejsz liczbę krypto w SYMBOLS albo dodaj klucz API CoinGecko."
                    + retry_after
                ),
                log_message=(
                    f"CoinGecko returned 429 for '{symbol}' — free tier "
                    "rate limit (~5-15 req/min) exhausted."
                ),
            )
        raise_for_status_translated(response)
        payload = json_payload(response)
        coin_block = payload.get(coin_id) if isinstance(payload, dict) else None
        if not isinstance(coin_block, dict):
            raise CoinGeckoCoinNotFound(
                f"CoinGecko returned no data for '{symbol}' "
                f"(coin_id={coin_id})."
            )
        return coin_block

    @staticmethod
    def _retry_after_hint(response: object) -> str:
        """Wyciąga nagłówek `Retry-After` (jeśli jest) do treści akcji.

        CoinGecko bywa hojny z tym nagłówkiem przy 429; podanie go ownerowi
        daje konkretną liczbę sekund do odczekania zamiast zgadywania.
        """
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("Retry-After")
        if retry_after:
            return f" Serwer sugeruje odczekać {retry_after} s (Retry-After)."
        return ""

    def _emit_quota_alert(
        self, severity: QuotaSeverity, message: str, action: str, log_message: str
    ) -> None:
        """Wystawia alert do QuotaMonitor, jeśli ten jest skonfigurowany."""
        # MAIL-15: `message`/`action` idą do maila (po polsku), `log_message`
        # zostaje angielską diagnozą w logach.
        level = logging.ERROR if severity is QuotaSeverity.CRITICAL else logging.WARNING
        logger.log(level, "%s", log_message)
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
