"""Adapter Alpha Vantage dla danych fundamentalnych spółek.

Woła dwa endpointy: OVERVIEW (P/E ratios) + EARNINGS (quarterly EPS dla growth).
Każde wywołanie = 2 requesty (Alpha Vantage free tier: 25/dobę = max ~12 spółek).

Defensywnie zwraca None przy:
  - pustym JSON (np. dla ETF-ów)
  - błędach sieciowych / rate limit po retry
  - całkowicie nieparsowalnych danych

DAT-11: `Error Message` z mową o `apikey` to odrzucony klucz. Gdy jest kolejny
klucz, adapter przechodzi na niego (alert WARNING, dane dalej płyną); dopiero
odrzucony ostatni klucz daje jeden CRITICAL i adapter martwy do końca przebiegu
(kolejne symbole bez requestów). `Error Message` bez `apikey` (zły symbol, zła
funkcja) to WARNING i None tylko dla tego symbolu.
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
    AlphaVantageFailureKind,
    classify_response,
    failure_action,
    failure_description,
    soft_limit_text,
    summarize_failures,
)
from src.infrastructure.adapters._http import (
    build_session,
    json_payload,
    raise_for_status_translated,
    redact_secrets,
    translated_get,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"


# Klucze, które niesie realna odpowiedź OVERVIEW/EARNINGS — ich obecność
# odróżnia dane od soft-limitu (DAT-06, `is_soft_limit`).
_FUNDAMENTALS_DATA_KEYS = ("Symbol", "PERatio", "quarterlyEarnings", "annualEarnings")

# DAT-11: pole błędu AV (HTTP 200) i fraza, po której poznajemy odrzucony klucz.
_ERROR_MESSAGE_MARKER = "Error Message"
_REJECTED_KEY_PHRASE = "apikey"
_ERROR_LOG_CHARS = 160


class _AlphaVantageKeysUnavailable(Exception):
    """Wszystkie klucze zwróciły marker bez danych; zachowuje rodzaj odmowy."""

    def __init__(self, failure: AlphaVantageFailureKind) -> None:
        super().__init__(failure.value)
        self.failure = failure


class _AlphaVantageRequestRejected(Exception):
    """Wewnętrzny sygnał (DAT-11): AV odpowiedziało `Error Message` o kluczu
    (`apikey` nieważny albo pusty). Kolejne symbole tym kluczem też padną, więc
    `_get` przechodzi na następny klucz, a ten wyjątek wychodzi z niego dopiero
    przy ostatnim — wtedy adapter przestaje pytać."""


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
        # DAT-11: po odrzuceniu klucza adapter nie wysyła już żadnych requestów.
        self._rejected = False
        self._key_failures: set[AlphaVantageFailureKind] = set()
        self._keys_unavailable_reported = False

    def _current_key(self) -> str:
        return self._api_keys[self._key_index]

    def _get(self, function: str, symbol: str) -> dict[str, Any]:
        """Woła AV aktualnym kluczem; przy soft-limicie rotuje na kolejny
        i ponawia. Gdy brak klucza z danymi — `_AlphaVantageKeysUnavailable`.

        Rozpoznany limit częstotliwości dostaje jeden backoff i ponowienie tym
        samym kluczem. Pozostałe markery bez danych od razu rotują.
        Odrzucony klucz (DAT-11) też rotuje; przy ostatnim leci
        `_AlphaVantageRequestRejected`."""
        backed_off = False
        while self._key_index < len(self._api_keys):
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self._current_key(),
            }
            response = translated_get(self._session, _BASE_URL, params=params, timeout=15)
            raise_for_status_translated(response)
            data = json_payload(response)
            if not isinstance(data, dict):
                return {}
            try:
                self._raise_on_error_message(data)
            except _AlphaVantageRequestRejected as rejected:
                if self._key_index + 1 >= len(self._api_keys):
                    raise
                self._report_rotated_key(symbol, str(rejected))
                self._key_index += 1
                backed_off = False
                continue
            # Soft-limit (200 + `Information`/`Note`, brak danych) musi być
            # rozróżniony od zwykłej pustki — inaczej dane giną po cichu.
            failure = classify_response(data, _FUNDAMENTALS_DATA_KEYS)
            if failure is None:
                return data
            marker = soft_limit_text(data, self._api_keys)
            if failure is AlphaVantageFailureKind.FREQUENCY_LIMIT and not backed_off:
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
                "Alpha Vantage fundamentals key #%d unavailable (%s): %s — rotating.",
                self._key_index + 1,
                failure.value,
                marker,
            )
            self._key_failures.add(failure)
            self._key_index += 1
            backed_off = False
            if self._key_index >= len(self._api_keys):
                raise _AlphaVantageKeysUnavailable(summarize_failures(self._key_failures))
        # Klucze wyczerpane jeszcze przed tym wywołaniem (short-circuit) —
        # nie spalamy requestu na martwym kluczu.
        raise _AlphaVantageKeysUnavailable(summarize_failures(self._key_failures))

    def _raise_on_error_message(self, data: dict[str, Any]) -> None:
        """`Error Message` → `_AlphaVantageRequestRejected` (klucz) albo
        `ValueError` (zły symbol / wywołanie). Treść bez kluczy — AV potrafi
        wkleić je do komunikatu."""
        error = data.get(_ERROR_MESSAGE_MARKER)
        if error is None:
            return
        text = redact_secrets(str(error), self._api_keys)[:_ERROR_LOG_CHARS]
        if _REJECTED_KEY_PHRASE in str(error).lower():
            raise _AlphaVantageRequestRejected(text)
        raise ValueError(f"Alpha Vantage returned an error: {text}")

    def _report_rotated_key(self, symbol: str, detail: str) -> None:
        """Log po angielsku + WARNING po polsku: klucz odrzucony, ale kolejny
        przejmuje zapytania, więc dane w cyklu są (DAT-11). Numer klucza, nigdy
        sam klucz."""
        key_number = self._key_index + 1
        logger.warning(
            "Alpha Vantage rejected fundamentals API key #%d at %s: %s — rotating to key #%d",
            key_number,
            symbol,
            detail,
            key_number + 1,
        )
        self._emit_quota_alert(
            severity=QuotaSeverity.WARNING,
            message=(
                f"Alpha Vantage odrzuciło klucz API fundamentów #{key_number} "
                "(Error Message); fundamenty idą kolejnym kluczem."
            ),
            action=(
                f"Sprawdź klucz #{key_number} w ALPHA_VANTAGE_API_KEYS w sekretach "
                "repozytorium — jest nieważny albo pusty; nowy darmowy klucz "
                "wygenerujesz na alphavantage.co."
            ),
        )

    def _report_rejected_key(self, symbol: str, detail: str) -> None:
        """Log po angielsku + jeden CRITICAL po polsku (MAIL-15): odrzucony
        ostatni klucz, więc dalszych fundamentów w tym przebiegu nie będzie."""
        logger.error(
            "Alpha Vantage rejected the last fundamentals API key at %s: %s — "
            "skipping fundamentals for the remaining symbols",
            symbol,
            detail,
        )
        self._emit_quota_alert(
            message=(
                "Alpha Vantage odrzuciło ostatni klucz API fundamentów (Error Message); "
                "w tym przebiegu brakuje P/E i wzrostu EPS dla pozostałych spółek."
            ),
            action=(
                "Sprawdź ALPHA_VANTAGE_API_KEYS w sekretach repozytorium — klucz jest "
                "nieważny albo pusty; nowy darmowy klucz wygenerujesz na alphavantage.co."
            ),
        )

    def _emit_quota_alert(
        self,
        message: str,
        action: str,
        *,
        severity: QuotaSeverity = QuotaSeverity.CRITICAL,
    ) -> None:
        """Wystawia alert (domyślnie CRITICAL) do QuotaMonitor, jeśli ten jest skonfigurowany."""
        if self._quota_monitor is None:
            return
        self._quota_monitor.record(
            QuotaAlert(
                source="Alpha Vantage fundamentals",
                severity=severity,
                message=message,
                action=action,
                occurred_at=datetime.now(UTC),
            )
        )

    def _report_keys_unavailable(self, symbol: str, failure: AlphaVantageFailureKind) -> None:
        """Jeden alert na przebieg; opis przyczyny pozostaje zgodny z odpowiedzią AV."""
        logger.error("Alpha Vantage fundamentals %s for %s", failure.value, symbol)
        self._emit_quota_alert(
            message=(
                f"Klucze fundamentów Alpha Vantage {failure_description(failure)}; "
                "w tym cyklu brakuje P/E i wzrostu EPS dla pozostałych spółek."
            ),
            action=failure_action(failure),
        )

    def get_fundamentals(self, symbol: str) -> Fundamentals | None:
        if self._rejected or self._keys_unavailable_reported:
            return None
        try:
            overview = self._get("OVERVIEW", symbol)
            earnings = self._get("EARNINGS", symbol)
        except _AlphaVantageRequestRejected as rejected:
            self._rejected = True
            self._report_rejected_key(symbol, str(rejected))
            return None
        except _AlphaVantageKeysUnavailable as unavailable:
            self._keys_unavailable_reported = True
            self._report_keys_unavailable(symbol, unavailable.failure)
            return None
        except Exception as exc:
            # ARCH-33: klasa w logu odróżnia awarię dostawcy (taksonomia) od reszty.
            logger.warning(
                "Alpha Vantage fundamentals fetch failed for %s (%s): %s",
                symbol,
                type(exc).__name__,
                exc,
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
            # DAT-21: z tej samej odpowiedzi OVERVIEW — zero nowych requestów.
            market_cap=_parse_float(overview.get("MarketCapitalization")),
        )
