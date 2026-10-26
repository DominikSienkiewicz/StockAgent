"""Adapter Deribit DVOL — implikowana zmienność BTC i ETH dla implied edge (DAT-18).

Option-chain Finnhuba to endpoint premium, więc R-18 (ruch modelu vs implied move
rynku opcji) nie ma dziś danych. Dla krypto istnieje darmowy, standardowy indeks
IV: DVOL Deribitu (publiczne API, bez klucza):

    GET https://www.deribit.com/api/v2/public/get_volatility_index_data
        ?currency=BTC&start_timestamp=<ms>&end_timestamp=<ms>&resolution=3600
    → {"result": {"data": [[timestamp_ms, open, high, low, close], ...]}}

Bierzemy `close` najświeższej świecy godzinowej z ostatnich `LOOKBACK_HOURS` godzin.
DVOL to annualizowana 30-dniowa IV w punktach procentowych (55.0 = 55%), a port
niesie ułamek (0.55). To 30-dniowa IV, nie ATM najbliższej ekspiracji — ta sama
klasa przybliżenia co średnia z łańcucha Finnhuba. Rynek krypto handluje 24/7,
więc skalowanie godzinami kalendarzowymi w `implied_edge` jest tu poprawne.

Adapter zwraca wyłącznie IV (`put_call_ratio = None`) i jest pytany tylko przez
port opcji grafu (implied edge), nie przez alpha fusion — neutralny głos opcji
rozcieńczałby fuzję. Błąd dostawcy (sieć, status, treść) → None; symbol bez indeksu
DVOL → None bez requestu. Włączenie: `deribit_dvol_enabled` (docs/configuration.md).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import requests

from src.application.errors import PortError
from src.application.ports import OptionsFlowPort
from src.domain.options_flow import OptionsFlowSnapshot
from src.infrastructure.adapters._http import (
    build_session,
    json_payload,
    raise_for_status_translated,
    translated_get,
)

DEFAULT_BASE_URL = "https://www.deribit.com/api/v2"
DEFAULT_TIMEOUT = 10.0
# Indeksy DVOL Deribitu (waluta = ticker krypto w watchliście).
DVOL_CURRENCIES = frozenset({"BTC", "ETH"})
# Świece godzinowe z kilku ostatnich godzin — najświeższa wystarczy, zapas na lukę.
RESOLUTION_SECONDS = 3600
LOOKBACK_HOURS = 6
# DVOL w punktach procentowych → ułamek portu.
_PERCENT = 100.0
_CLOSE_INDEX = 4

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _epoch_ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


class DeribitDvolAdapter(OptionsFlowPort):
    """`OptionsFlowPort` z samą IV dla BTC i ETH (Deribit DVOL)."""

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: str = DEFAULT_BASE_URL,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._session = session if session is not None else build_session()
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")
        self._clock = clock

    def get_options_flow(self, symbol: str) -> OptionsFlowSnapshot | None:
        currency = symbol.strip().upper()
        if currency not in DVOL_CURRENCIES:
            return None
        end = self._clock()
        try:
            response = translated_get(
                self._session,
                f"{self._base_url}/public/get_volatility_index_data",
                params={
                    "currency": currency,
                    "start_timestamp": _epoch_ms(end - timedelta(hours=LOOKBACK_HOURS)),
                    "end_timestamp": _epoch_ms(end),
                    "resolution": str(RESOLUTION_SECONDS),
                },
                timeout=self._timeout,
            )
            raise_for_status_translated(response)
            payload = json_payload(response)
        except PortError:
            logger.warning("Deribit DVOL unavailable for %s", currency, exc_info=True)
            return None
        dvol = _latest_close(payload)
        if dvol is None:
            logger.warning("Deribit DVOL answer for %s has no usable candle", currency)
            return None
        return OptionsFlowSnapshot(symbol=symbol, put_call_ratio=None, implied_vol=dvol / _PERCENT)


def _latest_close(payload: object) -> float | None:
    """`close` najświeższej świecy; brak, śmieć albo wartość niedodatnia → None."""
    result = payload.get("result") if isinstance(payload, dict) else None
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, list) or not data:
        return None
    candle = max(
        (row for row in data if isinstance(row, list) and len(row) > _CLOSE_INDEX),
        key=lambda row: row[0] if isinstance(row[0], int | float) else -math.inf,
        default=None,
    )
    close = candle[_CLOSE_INDEX] if candle is not None else None
    if not isinstance(close, int | float) or not math.isfinite(close) or close <= 0:
        return None
    return float(close)
