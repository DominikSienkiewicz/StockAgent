"""Adapter Alpha Vantage dla danych fundamentalnych spółek.

Woła dwa endpointy: OVERVIEW (P/E ratios) + EARNINGS (quarterly EPS dla growth).
Każde wywołanie = 2 requesty (Alpha Vantage free tier: 25/dobę = max ~12 spółek).

Defensywnie zwraca None przy:
  - pustym JSON (np. dla ETF-ów)
  - błędach sieciowych / rate limit po retry
  - całkowicie nieparsowalnych danych
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from src.application.ports import FundamentalsPort
from src.domain.value_objects import Fundamentals
from src.infrastructure.adapters._http import build_session

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"


def _parse_float(value: Any) -> float | None:
    """AV zwraca wszystko jako string; 'None' / '-' / '' → None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "-", "n/a"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _eps_growth_yoy(earnings_json: dict[str, Any]) -> float | None:
    """YoY growth z quarterlyEarnings: most-recent Q vs Q sprzed roku.

    Wymaga ≥5 kwartałów. Zwraca None gdy ubiegłoroczny EPS ≤ 0
    (dzielenie/sens ekonomiczny)."""
    quarters = earnings_json.get("quarterlyEarnings") or []
    if len(quarters) < 5:
        return None
    recent = _parse_float(quarters[0].get("reportedEPS"))
    year_ago = _parse_float(quarters[4].get("reportedEPS"))
    if recent is None or year_ago is None or year_ago <= 0:
        return None
    return (recent - year_ago) / year_ago


class AlphaVantageFundamentalsAdapter(FundamentalsPort):
    """Implementacja FundamentalsPort oparta o publiczne API Alpha Vantage."""

    def __init__(self, api_keys: list[str]) -> None:
        if not api_keys:
            raise ValueError("api_keys must contain at least one key")
        self._api_keys = api_keys
        self._key_index = 0
        self._session = build_session()

    def _next_key(self) -> str:
        key = self._api_keys[self._key_index % len(self._api_keys)]
        self._key_index += 1
        return key

    def _get(self, function: str, symbol: str) -> dict[str, Any]:
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self._next_key(),
        }
        response = self._session.get(_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return {}
        return data

    def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        try:
            overview = self._get("OVERVIEW", symbol)
            earnings = self._get("EARNINGS", symbol)
        except Exception as exc:
            logger.warning(
                "Alpha Vantage fundamentals fetch failed for %s: %s", symbol, exc
            )
            return None

        trailing_pe = _parse_float(overview.get("PERatio"))
        forward_pe = _parse_float(overview.get("ForwardPE"))
        peg_ratio = _parse_float(overview.get("PEGRatio"))
        eps_growth = _eps_growth_yoy(earnings)

        # Jeśli wszystkie cztery pola puste, nie zwracamy pustki.
        if all(v is None for v in (trailing_pe, forward_pe, peg_ratio, eps_growth)):
            return None

        return Fundamentals(
            trailing_pe=trailing_pe,
            forward_pe=forward_pe,
            peg_ratio=peg_ratio,
            eps_growth_yoy=eps_growth,
            fetched_at=datetime.now(UTC),
        )
