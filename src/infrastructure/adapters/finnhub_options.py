"""Adapter Finnhub option-chain → przepływy opcji i IV.

Endpoint: GET /stock/option-chain?symbol={symbol}&token={key}
Odpowiedź:
    {"data": [{"options": {"CALL": [{"volume": .., "impliedVolatility": ..}, ..],
                            "PUT":  [..]}}, ..]}

Agregujemy wolumen putów / calli → put_call_ratio (CALL vol 0 → None, brak div0)
i uśredniamy impliedVolatility po wszystkich kontraktach → implied_vol.

UWAGA: option-chain to PREMIUM endpoint Finnhuba. Na free tier zwraca 403 albo
pustą listę — adapter robi wtedy graceful no-op (None). To opcjonalne źródło
alpha (flag-gated + Null default), więc `get_options_flow` NIGDY nie rzuca.
"""

from __future__ import annotations

import logging

import requests

from src.application.ports import OptionsFlowPort
from src.domain.options_flow import OptionsFlowSnapshot
from src.infrastructure.adapters._http import build_session

DEFAULT_BASE_URL = "https://finnhub.io/api/v1"
DEFAULT_TIMEOUT = 10.0

logger = logging.getLogger(__name__)


class FinnhubOptionsAdapter(OptionsFlowPort):
    """Cienki I/O na Finnhub option-chain. Cała klasyfikacja siedzi w domenie."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session if session is not None else build_session()

    def get_options_flow(self, symbol: str) -> OptionsFlowSnapshot | None:
        try:
            response = self._session.get(
                f"{self._base_url}/stock/option-chain",
                params={"symbol": symbol, "token": self._api_key},
                timeout=self._timeout,
            )
            # Free tier: option-chain jest płatne → 403. Każdy nie-200 = brak
            # danych (no-op), nie podnosimy wyjątku.
            if response.status_code != 200:
                return None
            payload = response.json()
        except Exception:
            # Opcjonalne źródło — sieć, parse, brak pola: zawsze None, nigdy raise.
            logger.debug("Finnhub option-chain failed for %s", symbol, exc_info=True)
            return None

        return self._parse(symbol, payload)

    def _parse(self, symbol: str, payload: object) -> OptionsFlowSnapshot | None:
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return None

        call_volume = 0.0
        put_volume = 0.0
        ivs: list[float] = []

        for expiration in data:
            if not isinstance(expiration, dict):
                continue
            options = expiration.get("options")
            if not isinstance(options, dict):
                continue
            call_volume += self._sum_volume(options.get("CALL"), ivs)
            put_volume += self._sum_volume(options.get("PUT"), ivs)

        if call_volume <= 0:
            # CALL vol 0 → unikamy dzielenia przez zero; brak sygnału = None.
            return None

        put_call_ratio = put_volume / call_volume
        implied_vol = sum(ivs) / len(ivs) if ivs else None
        return OptionsFlowSnapshot(
            symbol=symbol,
            put_call_ratio=put_call_ratio,
            implied_vol=implied_vol,
        )

    @staticmethod
    def _sum_volume(contracts: object, ivs: list[float]) -> float:
        """Sumuje wolumen kontraktów i zbiera IV do uśrednienia (side-effect na
        `ivs`). Brakujące/niepoprawne pola traktujemy jako 0 / pomijamy."""
        if not isinstance(contracts, list):
            return 0.0
        total = 0.0
        for contract in contracts:
            if not isinstance(contract, dict):
                continue
            volume = contract.get("volume")
            if isinstance(volume, (int, float)):
                total += float(volume)
            iv = contract.get("impliedVolatility")
            if isinstance(iv, (int, float)):
                ivs.append(float(iv))
        return total


class NullOptionsFlowAdapter(OptionsFlowPort):
    """Domyślny no-op (brak klucza / Fast-Loop). Zero I/O, zawsze None."""

    def get_options_flow(self, symbol: str) -> OptionsFlowSnapshot | None:
        return None
