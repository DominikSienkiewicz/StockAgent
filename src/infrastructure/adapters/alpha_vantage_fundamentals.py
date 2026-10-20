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
import time
from datetime import UTC, datetime
from typing import Any

from src.application.ports import FundamentalsPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.value_objects import Fundamentals
from src.infrastructure.adapters._alpha_vantage_common import (
    NOTE_BACKOFF_SECONDS,
    is_frequency_limit,
    is_soft_limit,
    soft_limit_text,
)
from src.infrastructure.adapters._http import (
    build_session,
    raise_for_status_redacted,
    safe_get,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"


# Klucze, które niesie realna odpowiedź OVERVIEW/EARNINGS — ich obecność
# odróżnia dane od soft-limitu (DAT-06, `is_soft_limit`).
_FUNDAMENTALS_DATA_KEYS = ("Symbol", "PERatio", "quarterlyEarnings", "annualEarnings")


class _AlphaVantageQuotaExhausted(Exception):
    """Wewnętrzny sygnał: AV zwróciło soft-limit 200 (pole `Information`/`Note`),
    a nie realne dane. Odróżnia wyczerpanie limitu od zwykłej pustej odpowiedzi,
    żeby `get_fundamentals` nie potraktował go cicho jako brak fundamentów.

    `frequency_limit` = ostatni klucz odbił limit minutowy (`Note`), nie dzienny."""

    def __init__(self, message: str, *, frequency_limit: bool = False) -> None:
        super().__init__(message)
        self.frequency_limit = frequency_limit


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
        i ponawia. Gdy WSZYSTKIE klucze wyczerpane — `_AlphaVantageQuotaExhausted`.

        `Note` (limit minutowy) dostaje jeden backoff i ponowienie tym samym
        kluczem, `Information` (limit dzienny) od razu rotuje (DAT-06)."""
        backed_off = False
        while self._key_index < len(self._api_keys):
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self._current_key(),
            }
            response = safe_get(self._session, _BASE_URL, params=params, timeout=15)
            raise_for_status_redacted(response)
            data = response.json()
            if not isinstance(data, dict):
                return {}
            # Soft-limit (200 + `Information`/`Note`, brak danych) musi być
            # rozróżniony od zwykłej pustki — inaczej dane giną po cichu.
            if not is_soft_limit(data, _FUNDAMENTALS_DATA_KEYS):
                return data
            marker = soft_limit_text(data, self._api_keys)
            frequency_limit = is_frequency_limit(data)
            if frequency_limit and not backed_off:
                logger.warning(
                    "Alpha Vantage fundamentals key #%d hit the frequency limit: %s — "
                    "retrying the same key in %.1f s.",
                    self._key_index + 1,
                    marker,
                    NOTE_BACKOFF_SECONDS,
                )
                backed_off = True
                time.sleep(NOTE_BACKOFF_SECONDS)
                continue
            logger.warning(
                "Alpha Vantage fundamentals key #%d exhausted: %s — rotating.",
                self._key_index + 1,
                marker,
            )
            self._key_index += 1
            backed_off = False
            if self._key_index >= len(self._api_keys):
                raise _AlphaVantageQuotaExhausted(marker, frequency_limit=frequency_limit)
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

    def _report_quota_exhausted(self, symbol: str, *, frequency_limit: bool) -> None:
        """Log po angielsku + CRITICAL alert po polsku (MAIL-15); limit minutowy
        (`Note`) ma własny tekst — „dzienny limit 25 zapytań” by mylił (DAT-06)."""
        if frequency_limit:
            logger.exception(
                "Alpha Vantage fundamentals frequency limit hit on every key for %s",
                symbol,
            )
            self._emit_quota_alert(
                message=(
                    "Fundamenty Alpha Vantage odbiły limit częstotliwości zapytań "
                    "(pole Note) mimo ponowienia; w tym cyklu brakuje P/E i wzrostu EPS."
                ),
                action=(
                    "Rozłóż zapytania w czasie albo dodaj kolejny klucz "
                    "w ALPHA_VANTAGE_API_KEYS (darmowy na alphavantage.co)."
                ),
            )
            return
        logger.exception("Alpha Vantage fundamentals daily quota exhausted for %s", symbol)
        self._emit_quota_alert(
            message=(
                "Fundamenty Alpha Vantage wyczerpały dzienny limit 25 zapytań; "
                "w tym cyklu brakuje P/E i wzrostu EPS."
            ),
            action=(
                "Dodaj kolejny klucz w ALPHA_VANTAGE_API_KEYS (darmowy na "
                "alphavantage.co), zmniejsz liczbę symboli z fundamentami "
                "albo poczekaj na reset o północy UTC."
            ),
        )

    def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        try:
            overview = self._get("OVERVIEW", symbol)
            earnings = self._get("EARNINGS", symbol)
        except _AlphaVantageQuotaExhausted as exhausted:
            # Wyczerpany limit: NIE udajemy pustych fundamentów. Emitujemy alert
            # i przerywamy — kolejne klucze i tak są wspólne z resztą integracji
            # AV, więc dalsze próby tylko spalą budżet.
            self._report_quota_exhausted(symbol, frequency_limit=exhausted.frequency_limit)
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
