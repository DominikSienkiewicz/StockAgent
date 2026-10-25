from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import requests

from src.application.errors import MalformedPayload, NotFound
from src.application.ports import DailyChangePort, MarketDataPort, QuotePort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.shock import DailyChange
from src.domain.value_objects import Money, Quote
from src.infrastructure.adapters._http import (
    RateLimiter,
    build_session,
    raise_for_status_redacted,
    safe_get,
    translate_http_error,
    translate_request_error,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://finnhub.io/api/v1"
DEFAULT_TIMEOUT = 10  # sekundy


class FinnhubSymbolNotFound(NotFound, ValueError):
    """Finnhub nie notuje symbolu: 403 planu darmowego albo `c` = 0 (ticker
    nieznany lub wycofany). `ValueError` zostaje bazą dla dotychczasowych
    wywołujących (ARCH-33, przejście przyrostowe)."""


class FinnhubAdapter(MarketDataPort, DailyChangePort, QuotePort):
    """Adapter dla Finnhub Quote API.

    Endpoint: GET /quote?symbol={symbol}&token={api_key}
    Response: {"c": current_price, "d": change, "dp": percent_change,
    "pc": previous_close, "t": quote_unix_time, ...}
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        quota_monitor: QuotaMonitor | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = build_session()
        self._quota_monitor = quota_monitor
        # DAT-12: wspólny limiter procesu (jeden na klucz Finnhub) z DI.
        self._rate_limiter = rate_limiter

    def get_current_price(self, symbol: str) -> Money:
        payload = self._fetch_quote(symbol)
        price = payload.get("c")
        if not price:
            # Nie logujemy całego payloadu — Finnhub error responses bywają
            # niedeterministyczne i mogą zawierać dane diagnostyczne kontekstu
            # konta. Wystarczy boolowski status pól, których brakuje.
            has_change = "d" in payload
            raise FinnhubSymbolNotFound(
                f"Finnhub returned no price for symbol '{symbol}' "
                f"(has_change_field={has_change}). Ticker may be invalid or delisted."
            )

        return Money(Decimal(str(price)))

    def get_daily_change(self, symbol: str) -> DailyChange:
        """ALR-02: `dp` (procent od poprzedniego zamknięcia) jako ułamek i `t`
        jako chwila notowania — to samo darmowe /quote co cena."""
        payload = self._fetch_quote(symbol)
        percent, stamp = payload.get("dp"), payload.get("t")
        if not payload.get("c") or percent is None or not stamp:
            raise _missing_fields_error(payload)(
                f"Finnhub returned no daily change for symbol '{symbol}' "
                f"(has_dp={percent is not None}, has_t={bool(stamp)})."
            )
        return DailyChange(
            fraction=Decimal(str(percent)) / 100,
            as_of=datetime.fromtimestamp(int(stamp), tz=UTC),
        )

    def get_quote(self, symbol: str) -> Quote:
        """DAT-14: cena `c` z chwilą notowania `t` (plus `pc` i `o`, gdy są) —
        `get_current_price` zostaje bez zmian dla obecnych konsumentów."""
        payload = self._fetch_quote(symbol)
        price, stamp = payload.get("c"), payload.get("t")
        if not price or not stamp:
            raise _missing_fields_error(payload)(
                f"Finnhub returned no quote for symbol '{symbol}' "
                f"(has_price={bool(price)}, has_t={bool(stamp)})."
            )
        return Quote(
            price=Money(Decimal(str(price))),
            quoted_at=datetime.fromtimestamp(int(stamp), tz=UTC),
            prev_close=_optional_money(payload.get("pc")),
            open_price=_optional_money(payload.get("o")),
        )

    def _fetch_quote(self, symbol: str) -> dict[str, Any]:
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            response = safe_get(
                self._session,
                f"{self._base_url}/quote",
                params={"symbol": symbol, "token": self._api_key},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise translate_request_error(exc) from None
        if response.status_code == 403:
            raise FinnhubSymbolNotFound(
                f"Ticker '{symbol}' is not supported by Finnhub free tier (403 Forbidden). "
                "LSE tickers with dot notation (e.g. CSPX.L) require a paid plan."
            )
        if response.status_code == 429:
            # Free tier: 60 req/min. Powyżej tego dostajemy 429.
            self._emit_quota_alert(
                severity=QuotaSeverity.CRITICAL,
                message=(
                    f"Finnhub zwrócił 429 dla '{symbol}' — wyczerpany darmowy "
                    "limit (60 zapytań/min)."
                ),
                action=(
                    "Zmniejsz liczbę SYMBOLS albo podnieś SYMBOL_THROTTLE_SECONDS, "
                    "żeby spowolnić pętlę, albo podnieś plan Finnhub."
                ),
                log_message=(
                    f"Finnhub returned 429 for '{symbol}' — free tier "
                    "limit (60 req/min) exhausted."
                ),
            )
        try:
            raise_for_status_redacted(response)
        except requests.HTTPError as exc:
            raise translate_http_error(exc, response.status_code) from None
        payload: dict[str, Any] = response.json()
        return payload

    def _emit_quota_alert(
        self, severity: QuotaSeverity, message: str, action: str, log_message: str
    ) -> None:
        # MAIL-15: `message`/`action` idą do maila (po polsku), `log_message`
        # zostaje angielską diagnozą w logach.
        level = logging.ERROR if severity is QuotaSeverity.CRITICAL else logging.WARNING
        logger.log(level, "%s", log_message)
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source="Finnhub",
                severity=severity,
                message=message,
                action=action,
                occurred_at=datetime.now(UTC),
            )
        )


def _missing_fields_error(payload: dict[str, Any]) -> type[ValueError]:
    """Brak ceny `c` to nieznany ticker; cena bez reszty pól to niepełny payload."""
    return FinnhubSymbolNotFound if not payload.get("c") else MalformedPayload


def _optional_money(value: Any) -> Money | None:
    """Pole ceny z /quote; brak albo 0 (Finnhub tak oznacza brak danych) → None."""
    if not value:
        return None
    return Money(Decimal(str(value)))
