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
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.value_objects import Fundamentals
from src.infrastructure.adapters._http import build_session

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"


class _AlphaVantageQuotaExhausted(Exception):
    """Wewnętrzny sygnał: AV zwróciło soft-limit 200 (pole `Information`/`Note`),
    a nie realne dane. Odróżnia wyczerpanie limitu od zwykłej pustej odpowiedzi,
    żeby `get_fundamentals` nie potraktował go cicho jako brak fundamentów."""


def _is_rate_limited(payload: dict[str, Any]) -> bool:
    """AV w free tier sygnalizuje wyczerpanie dziennego limitu (25/dobę)
    odpowiedzią 200 z polem `Information` (czasem `Note`) i BEZ realnych
    danych. Wykrywamy ten wzorzec, by nie zwracać cicho None."""
    has_soft_limit_marker = "Information" in payload or "Note" in payload
    if not has_soft_limit_marker:
        return False
    # Realna odpowiedź OVERVIEW/EARNINGS niesie te klucze; jeśli któryś
    # jest obecny, to nie jest soft-limit (AV nie miesza danych z notką).
    has_real_data = any(
        key in payload
        for key in ("Symbol", "PERatio", "quarterlyEarnings", "annualEarnings")
    )
    return not has_real_data


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

    def __init__(
        self,
        api_keys: list[str],
        quota_monitor: QuotaMonitor | None = None,
    ) -> None:
        if not api_keys:
            raise ValueError("api_keys must contain at least one key")
        self._api_keys = api_keys
        # Indeks aktywnego klucza — przesuwany WYŁĄCZNIE przy wykrytym
        # soft-limicie (jak rotacja w news clientcie), nie co request.
        # Przy 2 req/symbol (OVERVIEW+EARNINGS) rotacja per-request smaruje
        # load po wszystkich kluczach i bije w dzienny cap równocześnie —
        # zostajemy więc na jednym kluczu aż padnie, reszta jest rezerwą.
        self._key_index = 0
        self._session = build_session()
        self._quota_monitor = quota_monitor

    def _current_key(self) -> str:
        return self._api_keys[self._key_index]

    def _get(self, function: str, symbol: str) -> dict[str, Any]:
        """Woła AV aktualnym kluczem; przy soft-limicie rotuje na kolejny
        i ponawia. Gdy WSZYSTKIE klucze wyczerpane — `_AlphaVantageQuotaExhausted`."""
        while self._key_index < len(self._api_keys):
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self._current_key(),
            }
            response = self._session.get(_BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return {}
            # Soft-limit (200 + `Information`/`Note`, brak danych) musi być
            # rozróżniony od zwykłej pustki — inaczej dane giną po cichu.
            if _is_rate_limited(data):
                note = data.get("Information") or data.get("Note") or ""
                logger.warning(
                    "Alpha Vantage fundamentals key #%d exhausted — rotating.",
                    self._key_index + 1,
                )
                self._key_index += 1
                if self._key_index >= len(self._api_keys):
                    raise _AlphaVantageQuotaExhausted(str(note))
                continue
            return data
        # Klucze wyczerpane jeszcze przed tym wywołaniem (short-circuit) —
        # nie spalamy requestu na martwym kluczu.
        raise _AlphaVantageQuotaExhausted("all keys exhausted")

    def _emit_quota_alert(self, message: str, action: str) -> None:
        """Wystawia CRITICAL alert do QuotaMonitor, jeśli ten jest skonfigurowany."""
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source="Alpha Vantage fundamentals",
                severity=QuotaSeverity.CRITICAL,
                message=message,
                action=action,
                occurred_at=datetime.now(UTC),
            )
        )

    def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        try:
            overview = self._get("OVERVIEW", symbol)
            earnings = self._get("EARNINGS", symbol)
        except _AlphaVantageQuotaExhausted:
            # Wyczerpany dzienny limit: NIE udajemy pustych fundamentów.
            # Emitujemy alert i przerywamy — kolejne klucze i tak są wspólne
            # z resztą integracji AV, więc dalsze próby tylko spalą budżet.
            logger.exception(
                "Alpha Vantage fundamentals daily quota exhausted for %s", symbol
            )
            self._emit_quota_alert(
                message=(
                    "Alpha Vantage fundamentals hit the daily 25 req/day limit; "
                    "P/E and EPS-growth data is missing for this cycle."
                ),
                action=(
                    "Add another key in ALPHA_VANTAGE_API_KEYS (free at "
                    "alphavantage.co), reduce the number of fundamentals symbols, "
                    "or wait for the UTC-midnight reset."
                ),
            )
            return None
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
