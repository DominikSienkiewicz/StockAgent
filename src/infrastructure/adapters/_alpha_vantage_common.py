"""Odpowiedzi AV bez danych: dostęp do planu, limit doby, częstotliwość lub nieznane.

Nazwa markera `Information`/`Note` nie określa przyczyny. Fraza o endpointcie
premium pochodzi z logu produkcyjnego; sam link do planu premium nie jest odmową
dostępu, bo pojawia się też w komunikatach limitu częstotliwości.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

from src.infrastructure.adapters._http import redact_secrets

INFORMATION_MARKER = "Information"
NOTE_MARKER = "Note"
NOTE_BACKOFF_SECONDS = 2.0
_MARKER_LOG_CHARS = 160


class AlphaVantageFailureKind(Enum):
    ACCESS_DENIED = "access denied"
    DAILY_QUOTA = "daily quota exhausted"
    FREQUENCY_LIMIT = "frequency rate-limit"
    UNKNOWN = "unclassified response without data"


def classify_response(
    payload: Mapping[str, Any], data_keys: Iterable[str]
) -> AlphaVantageFailureKind | None:
    """Rozpoznana przyczyna markera bez danych; brak markera/dane → None."""
    if not is_soft_limit(payload, data_keys):
        return None
    text = " ".join(
        str(payload.get(marker, "")) for marker in (INFORMATION_MARKER, NOTE_MARKER)
    ).lower()
    if any(
        phrase in text
        for phrase in (
            "premium endpoint",
            "premium api function",
            "access denied",
            "not authorized",
            "invalid api key",
        )
    ):
        return AlphaVantageFailureKind.ACCESS_DENIED
    if any(
        phrase in text
        for phrase in ("per minute", "per second", "call frequency", "spread out", "spreading out")
    ):
        return AlphaVantageFailureKind.FREQUENCY_LIMIT
    if any(
        phrase in text
        for phrase in ("requests per day", "calls per day", "daily limit", "daily quota")
    ):
        return AlphaVantageFailureKind.DAILY_QUOTA
    return AlphaVantageFailureKind.UNKNOWN


def summarize_failures(kinds: Iterable[AlphaVantageFailureKind]) -> AlphaVantageFailureKind:
    """Wspólna przyczyna tylko gdy wszystkie odrzucone klucze miały tę samą."""
    distinct = set(kinds)
    return next(iter(distinct)) if len(distinct) == 1 else AlphaVantageFailureKind.UNKNOWN


def failure_description(kind: AlphaVantageFailureKind) -> str:
    """Fragment polskiego alertu opisujący klucze bez dostępnych danych."""
    match kind:
        case AlphaVantageFailureKind.ACCESS_DENIED:
            return (
                "nie mają dostępu do endpointu (wymagany plan premium lub odpowiednie uprawnienia)"
            )
        case AlphaVantageFailureKind.DAILY_QUOTA:
            return "wyczerpały dzienny limit zapytań"
        case AlphaVantageFailureKind.FREQUENCY_LIMIT:
            return "odbiły limit częstotliwości zapytań mimo ponowienia"
        case AlphaVantageFailureKind.UNKNOWN:
            return "zwróciły komunikaty bez danych; przyczyny są niejednoznaczne"


def failure_action(kind: AlphaVantageFailureKind) -> str:
    """Działanie zgodne z rozpoznaną przyczyną, bez obietnic nieznanego resetu."""
    match kind:
        case AlphaVantageFailureKind.ACCESS_DENIED:
            return (
                "Sprawdź plan i uprawnienia kluczy w ALPHA_VANTAGE_API_KEYS do tego endpointu. "
                "Jeżeli masz klucz z wymaganym dostępem, skonfiguruj go dla tego źródła."
            )
        case AlphaVantageFailureKind.DAILY_QUOTA:
            return (
                "Dodaj kolejny klucz z dostępnym limitem w ALPHA_VANTAGE_API_KEYS, "
                "zmniejsz liczbę zapytań albo sprawdź termin odnowienia limitu swojego planu."
            )
        case AlphaVantageFailureKind.FREQUENCY_LIMIT:
            return (
                "Rozłóż zapytania w czasie i sprawdź limit częstotliwości "
                "swojego planu Alpha Vantage."
            )
        case AlphaVantageFailureKind.UNKNOWN:
            return (
                "Sprawdź zredagowane komunikaty Alpha Vantage w logu "
                "oraz plan i uprawnienia kluczy; "
                "nie ustalono jednego rodzaju ograniczenia."
            )


def is_soft_limit(payload: Mapping[str, Any], data_keys: Iterable[str]) -> bool:
    """Marker `Information`/`Note` bez żadnego z `data_keys`.

    Obecny klucz danych znaczy prawdziwą odpowiedź — AV nie miesza danych
    z notką o limicie, więc notka obok danych nie jest soft-limitem."""
    if INFORMATION_MARKER not in payload and NOTE_MARKER not in payload:
        return False
    return not any(key in payload for key in data_keys)


def is_frequency_limit(payload: Mapping[str, Any]) -> bool:
    """Treść markera wskazuje krótkotrwały limit, po którym warto odczekać."""
    return classify_response(payload, ()) is AlphaVantageFailureKind.FREQUENCY_LIMIT


def soft_limit_text(payload: Mapping[str, Any], api_keys: Iterable[str]) -> str:
    """Treść markera do logu, bez kluczy: AV potrafi wkleić klucz do komunikatu.

    Redakcja idzie przed przycięciem — cięcie w środku klucza zostawiłoby jego
    prefiks, którego dosłowna redakcja już nie rozpozna."""
    raw = payload.get(INFORMATION_MARKER) or payload.get(NOTE_MARKER) or ""
    return redact_secrets(str(raw), api_keys)[:_MARKER_LOG_CHARS]
