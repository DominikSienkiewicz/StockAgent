"""Adaptery NBP — kursy walut z tabeli A (darmowe API, bez klucza).

`NbpClient` (MacroIndicatorsPort) — EUR/PLN i USD/PLN z 30-dniowego okna:
    https://api.nbp.pl/api/exchangerates/rates/A/{currency}/last/{N}/?format=json
Odpowiedź zawiera listę kursów ("rates") posortowaną chronologicznie —
ostatni element = najświeższy. Adapter używa pierwszego punktu jako
30-dniowej referencji do liczenia procentowej zmiany.

`NbpFxRateClient` (FxRatePort, PORT-11) — kurs z konkretnego dnia (np. zakupu):
    https://api.nbp.pl/api/exchangerates/rates/a/{code}/{date}/?format=json
Dzień bez tabeli (weekend, święto, przed publikacją ok. 12:00) to 404 — adapter
cofa się wtedy dzień po dniu do poprzedniego dnia roboczego, najwyżej
`MAX_LOOKBACK_DAYS` dni. Inny błąd (sieć, 5xx po retry) kończy szukanie → None
i wyłącza klienta do końca jego życia (jeden klient na cykl): przy leżącym NBP
każde zapytanie to timeout z retry, a dziesiątki dat zakupu przekroczyłyby
limit czasu joba.

Cienki I/O — cała logika oceny stresu i P/L w złotych siedzi w domenie
(`PolishMacroSnapshot.evaluate_stress_level`, `Portfolio.pln_pnl`).
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from http import HTTPStatus
from typing import Any

import requests

from src.application.ports import FxRatePort, MacroIndicatorsPort
from src.domain.fx import FxRate
from src.domain.polish_macro import PolishMacroSnapshot
from src.domain.portfolio import HOME_CURRENCY
from src.infrastructure.adapters._http import build_session

DEFAULT_BASE_URL = "https://api.nbp.pl/api/exchangerates/rates/A"
DEFAULT_FX_BASE_URL = "https://api.nbp.pl/api/exchangerates/rates/a"
DEFAULT_TIMEOUT = 10
DEFAULT_WINDOW_DAYS = 30
# Najdłuższa przerwa bez tabeli A (weekend + święta, np. 24–27.12 albo Wielkanoc)
# to 4–5 dni; 10 dni zapasu, zanim uznamy kurs za niedostępny.
MAX_LOOKBACK_DAYS = 10

logger = logging.getLogger(__name__)


class NbpClient(MacroIndicatorsPort):
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._window_days = window_days
        self._session = build_session()

    def fetch_polish_macro(self) -> PolishMacroSnapshot | None:
        eur_rates = self._fetch_rates("EUR")
        usd_rates = self._fetch_rates("USD")
        if not eur_rates or not usd_rates:
            return None
        if len(eur_rates) < 2 or len(usd_rates) < 2:
            # Brak punktu referencyjnego do delty 30d.
            return None

        eur_latest = Decimal(str(eur_rates[-1]["mid"]))
        eur_baseline = Decimal(str(eur_rates[0]["mid"]))
        usd_latest = Decimal(str(usd_rates[-1]["mid"]))
        usd_baseline = Decimal(str(usd_rates[0]["mid"]))

        return PolishMacroSnapshot(
            eur_pln=eur_latest,
            usd_pln=usd_latest,
            eur_pln_30d_change_pct=(eur_latest - eur_baseline) / eur_baseline,
            usd_pln_30d_change_pct=(usd_latest - usd_baseline) / usd_baseline,
            fetched_at=datetime.now(UTC),
        )

    def _fetch_rates(self, currency: str) -> list[dict[str, object]]:
        url = f"{self._base_url}/{currency}/last/{self._window_days}/?format=json"
        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("NBP fetch failed for %s", currency)
            return []
        payload = response.json()
        rates = payload.get("rates")
        if not isinstance(rates, list):
            return []
        return rates


class NbpFxRateClient(FxRatePort):
    """PORT-11: kurs średni NBP z danego dnia, z cofnięciem do dnia roboczego."""

    def __init__(
        self,
        base_url: str = DEFAULT_FX_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_lookback_days: int = MAX_LOOKBACK_DAYS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_lookback_days = max_lookback_days
        self._session = build_session()
        self._unavailable = False

    def get_rate(self, currency: str, on_date: date) -> FxRate | None:
        code = currency.strip().upper()
        if code == HOME_CURRENCY:
            return FxRate(currency=code, on_date=on_date, effective_date=on_date, rate=Decimal(1))
        if self._unavailable:
            return None
        for days_back in range(self._max_lookback_days + 1):
            day = on_date - timedelta(days=days_back)
            url = f"{self._base_url}/{code.lower()}/{day.isoformat()}/?format=json"
            try:
                response = self._session.get(url, timeout=self._timeout)
                if response.status_code == HTTPStatus.NOT_FOUND:
                    continue
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError):
                logger.exception(
                    "NBP rate fetch failed for %s on %s — no more NBP rates this cycle",
                    code,
                    day,
                )
                self._unavailable = True
                return None
            return _parse_day_rate(code, on_date, payload)
        logger.warning(
            "NBP: no table A rate for %s within %d days before %s",
            code,
            self._max_lookback_days,
            on_date,
        )
        return None


def _parse_day_rate(code: str, on_date: date, payload: Any) -> FxRate | None:
    """Odpowiedź `/rates/a/{code}/{date}/` → `FxRate` albo None (zła odpowiedź)."""
    rates = payload.get("rates") if isinstance(payload, dict) else None
    if not isinstance(rates, list) or not rates or not isinstance(rates[-1], dict):
        return None
    entry = rates[-1]
    try:
        return FxRate(
            currency=code,
            on_date=on_date,
            effective_date=date.fromisoformat(str(entry.get("effectiveDate"))),
            rate=Decimal(str(entry["mid"])),
        )
    except (KeyError, ValueError, InvalidOperation):
        logger.warning("NBP: malformed rate payload for %s on %s", code, on_date)
        return None
