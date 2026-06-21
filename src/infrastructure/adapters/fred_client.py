"""Adapter FRED — krzywa rentowności (10Y / 2Y / fed funds) z St. Louis Fed.

FRED API wymaga darmowego klucza. Pobieramy najświeższą obserwację trzech serii:
    DGS10 — rentowność 10-letnich obligacji skarbowych USA,
    DGS2  — rentowność 2-letnich,
    DFF   — efektywna stopa fed funds.

Endpoint (per seria):
    GET https://api.stlouisfed.org/fred/series/observations
        ?series_id={id}&api_key={key}&file_type=json&sort_order=desc&limit=1

Odpowiedź: {"observations": [{"date": "...", "value": "4.25"}]}. FRED oznacza
brak danych znakiem "." — traktujemy jak None. Adapter jest graceful: każdy błąd
(sieć, parse, brak pola, non-200) na pojedynczej serii daje None dla tej serii.
Gdy wszystkie trzy padną → None (krzywa nieinformatywna). Cienki I/O — cała
klasyfikacja stanu siedzi w domenie (`YieldCurveSnapshot.state`).
"""

from __future__ import annotations

import logging

import requests

from src.application.ports import MacroRatesPort
from src.domain.macro_rates import YieldCurveSnapshot
from src.infrastructure.adapters._http import build_session

DEFAULT_BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_TIMEOUT = 10.0

# Serie FRED mapowane na pola snapshotu krzywej.
SERIES_TEN_YEAR = "DGS10"
SERIES_TWO_YEAR = "DGS2"
SERIES_FED_FUNDS = "DFF"

logger = logging.getLogger(__name__)


class FredClient(MacroRatesPort):
    """Adapter MacroRatesPort dla FRED series/observations API."""

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._session = session if session is not None else build_session()
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")

    def fetch_yield_curve(self) -> YieldCurveSnapshot | None:
        ten_year = self._fetch_latest(SERIES_TEN_YEAR)
        two_year = self._fetch_latest(SERIES_TWO_YEAR)
        fed_funds = self._fetch_latest(SERIES_FED_FUNDS)

        # Wszystkie serie padły → krzywa nieinformatywna, sygnalizujemy brak.
        if ten_year is None and two_year is None and fed_funds is None:
            return None

        return YieldCurveSnapshot(
            ten_year=ten_year,
            two_year=two_year,
            fed_funds=fed_funds,
        )

    def _fetch_latest(self, series_id: str) -> float | None:
        """Najświeższa obserwacja serii jako float; None na dowolnym błędzie."""
        params: dict[str, str | int] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        try:
            response = self._session.get(
                f"{self._base_url}/series/observations",
                params=params,
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
            # Body może być nie-dictem (lista/null z proxy/CDN) — twardo strażujemy
            # kształt przed .get(), inaczej AttributeError wyciekłby z except.
            if not isinstance(payload, dict):
                return None
            observations = payload.get("observations")
            if not isinstance(observations, list) or not observations:
                return None
            first = observations[0]
            if not isinstance(first, dict):
                return None
            value = first.get("value")
            # FRED zwraca "." gdy brak wartości — to nie jest liczba.
            if value is None or value == ".":
                return None
            return float(value)
        except Exception:
            # Opcjonalne źródło alpha — ŻADEN błąd (sieć, parse, nieoczekiwany
            # kształt) nie może wywalić cyklu. Szeroki except celowo.
            logger.warning("FRED fetch failed for series %s", series_id, exc_info=True)
            return None


class NullMacroRatesAdapter(MacroRatesPort):
    """Fast-Loop / brak klucza FRED — zwraca None bez żadnego I/O."""

    def fetch_yield_curve(self) -> YieldCurveSnapshot | None:
        return None
