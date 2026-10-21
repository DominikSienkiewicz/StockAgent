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

DAT-22 (`MacroSeriesPort`): pięć zmiennych stanu rynku — VIXCLS (okno roku do
percentyla), BAMLH0A0HYM2 (spread high yield, okno trzech tygodni do zmiany
5-sesyjnej), VXNCLS, T10Y2Y i tygodniowe NFCI (ostatnia obserwacja z dwóch tygodni).
Okno kończy się na dniu cyklu (`observation_end`), a każda wartość niesie swój dzień
obserwacji — FRED publikuje z opóźnieniem, więc dzień pobrania byłby kłamstwem.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta

import requests

from src.application.ports import MacroRatesPort, MacroSeriesPort
from src.domain.macro_rates import YieldCurveSnapshot
from src.domain.macro_series import MacroSeriesSnapshot, SeriesObservation
from src.infrastructure.adapters._http import (
    build_session,
    raise_for_status_redacted,
    safe_get,
)

DEFAULT_BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_TIMEOUT = 10.0

# Serie FRED mapowane na pola snapshotu krzywej.
SERIES_TEN_YEAR = "DGS10"
SERIES_TWO_YEAR = "DGS2"
SERIES_FED_FUNDS = "DFF"

# DAT-22: zmienne stanu rynku i okna ich odczytu (dni kalendarzowe przed dniem cyklu).
SERIES_VIX = "VIXCLS"
SERIES_VXN = "VXNCLS"
SERIES_HY_OAS = "BAMLH0A0HYM2"
SERIES_TERM_SPREAD = "T10Y2Y"
SERIES_NFCI = "NFCI"
VIX_WINDOW_DAYS = 365
HY_OAS_WINDOW_DAYS = 21
LATEST_WINDOW_DAYS = 14

logger = logging.getLogger(__name__)


class FredClient(MacroRatesPort, MacroSeriesPort):
    """Adapter MacroRatesPort i MacroSeriesPort dla FRED series/observations API."""

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

    def fetch_macro_series(self, as_of: date) -> MacroSeriesSnapshot | None:
        """DAT-22: pięć serii raz na cykl; wszystkie puste → None."""
        snapshot = MacroSeriesSnapshot(
            vix=self._fetch_window(SERIES_VIX, as_of, VIX_WINDOW_DAYS),
            hy_oas=self._fetch_window(SERIES_HY_OAS, as_of, HY_OAS_WINDOW_DAYS),
            vxn=_last(self._fetch_window(SERIES_VXN, as_of, LATEST_WINDOW_DAYS)),
            term_spread=_last(self._fetch_window(SERIES_TERM_SPREAD, as_of, LATEST_WINDOW_DAYS)),
            nfci=_last(self._fetch_window(SERIES_NFCI, as_of, LATEST_WINDOW_DAYS)),
        )
        return None if snapshot.is_empty else snapshot

    def _fetch_window(
        self, series_id: str, as_of: date, days: int
    ) -> tuple[SeriesObservation, ...]:
        """Obserwacje serii z okna `[as_of − days, as_of]` rosnąco; () na dowolnym błędzie."""
        params: dict[str, str | int] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": (as_of - timedelta(days=days)).isoformat(),
            "observation_end": as_of.isoformat(),
            "sort_order": "asc",
        }
        try:
            response = safe_get(
                self._session,
                f"{self._base_url}/series/observations",
                params=params,
                timeout=self._timeout,
            )
            raise_for_status_redacted(response)
            payload = response.json()
            observations = payload.get("observations") if isinstance(payload, dict) else None
            if not isinstance(observations, list):
                return ()
            parsed = (_observation(item) for item in observations)
            return tuple(obs for obs in parsed if obs is not None)
        except Exception:
            # Źródło opcjonalne — błąd serii nie może wywalić cyklu (jak `_fetch_latest`).
            logger.warning("FRED fetch failed for series %s", series_id, exc_info=True)
            return ()

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
            response = safe_get(
                self._session,
                f"{self._base_url}/series/observations",
                params=params,
                timeout=self._timeout,
            )
            raise_for_status_redacted(response)
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


def _observation(item: object) -> SeriesObservation | None:
    """Obserwacja FRED z dniem i liczbą; "." (brak wartości) i śmieci → None."""
    if not isinstance(item, dict):
        return None
    raw_date = item.get("date")
    raw_value = item.get("value")
    if not isinstance(raw_date, str) or not isinstance(raw_value, str) or raw_value == ".":
        return None
    try:
        observation = SeriesObservation(date.fromisoformat(raw_date), float(raw_value))
    except ValueError:
        return None
    return observation if math.isfinite(observation.value) else None


def _last(window: tuple[SeriesObservation, ...]) -> SeriesObservation | None:
    return window[-1] if window else None


class NullMacroRatesAdapter(MacroRatesPort):
    """Fast-Loop / brak klucza FRED — zwraca None bez żadnego I/O."""

    def fetch_yield_curve(self) -> YieldCurveSnapshot | None:
        return None


class NullMacroSeriesAdapter(MacroSeriesPort):
    """DAT-22 off / brak klucza FRED — zwraca None bez żadnego I/O."""

    def fetch_macro_series(self, as_of: date) -> MacroSeriesSnapshot | None:
        return None
