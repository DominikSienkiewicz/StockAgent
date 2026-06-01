"""Adapter NBP — kursy EUR/PLN i USD/PLN z 30-dniowej tabeli A.

NBP API jest darmowe i bez klucza. Endpoint:
    https://api.nbp.pl/api/exchangerates/rates/A/{currency}/last/{N}/?format=json

Odpowiedź zawiera listę kursów ("rates") posortowaną chronologicznie —
ostatni element = najświeższy. Adapter używa pierwszego punktu jako
30-dniowej referencji do liczenia procentowej zmiany.

Cienki I/O — cała logika oceny stresu siedzi w domenie
(`PolishMacroSnapshot.evaluate_stress_level`).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal

import requests

from src.application.ports import MacroIndicatorsPort
from src.domain.polish_macro import PolishMacroSnapshot
from src.infrastructure.adapters._http import build_session

DEFAULT_BASE_URL = "https://api.nbp.pl/api/exchangerates/rates/A"
DEFAULT_TIMEOUT = 10
DEFAULT_WINDOW_DAYS = 30

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
