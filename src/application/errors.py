"""Taksonomia błędów portów (ARCH-33).

Adaptery tłumaczą wyjątki dostawców (`requests.HTTPError`, `anthropic.APIStatusError`,
PostgREST `APIError`) na te klasy, a graf i entry-pointy łapią taksonomię zamiast
`except Exception` — dzięki temu awaria dostawcy, wyczerpana kwota i śmieciowa
odpowiedź są odróżnialne w logach, telemetrii i ścieżkach degradacji.

Przejście jest przyrostowe (adapter po adapterze): przetłumaczony wyjątek nadal jest
instancją oryginalnego typu dostawcy (`src.infrastructure.error_translation`), więc
dotychczasowe `except requests.RequestException` i podobne działają bez zmian.
"""

from __future__ import annotations

from enum import Enum


class PortError(Exception):
    """Błąd zgłoszony przez port — baza całej taksonomii."""


class ProviderUnavailable(PortError):
    """Dostawca nie odpowiedział: 5xx, timeout, zerwane połączenie albo otwarty
    obwód circuit breakera (NIEZ-31). Ponowienie w tym samym przebiegu nie pomoże."""


class QuotaExhausted(PortError):
    """Dostawca odmówił z powodu limitu (429 po ponowieniach, wyczerpana kwota)."""


class MalformedPayload(PortError, ValueError):
    """Dostawca odpowiedział, ale treść nie spełnia kontraktu portu (brak pól,
    niepoprawny JSON). `ValueError` zostaje bazą — tak dotąd sygnalizowały to adaptery."""


class NotFound(PortError, LookupError):
    """Dostawca nie zna zasobu (np. tickera) — to nie awaria, tylko brak danych."""


class LLMOutputIncomplete(MalformedPayload):
    """NIEZ-26: model przerwał odpowiedź, zanim spełniła kontrakt portu. Graf mapuje
    to na flagę jakości `llm_output_truncated` zamiast generycznej awarii LLM."""


class LLMTruncatedError(LLMOutputIncomplete):
    """Odpowiedź ucięta limitem tokenów wyjścia (`stop_reason = max_tokens`,
    `finish_reason = length`). `output_tokens` — ile tokenów zużył (i policzył)
    dostawca; None, gdy go nie podał."""

    def __init__(self, message: str, *, output_tokens: int | None) -> None:
        super().__init__(message)
        self.output_tokens = output_tokens


class LLMRefusalError(LLMOutputIncomplete):
    """Model odmówił odpowiedzi (`stop_reason = refusal`)."""


class ErrorCategory(Enum):
    """MAIL-32: kategoria błędu symbolu dla czytelnika maila — klasa taksonomii
    portów, której instancją jest wyjątek. `PORT_ERROR` to błąd portu bez
    węższej klasy (np. odrzucony klucz API), `UNKNOWN` — wyjątek spoza taksonomii,
    czyli błąd kodu agenta albo adapter, który jeszcze nie tłumaczy wyjątków."""

    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    MALFORMED_PAYLOAD = "MALFORMED_PAYLOAD"
    NOT_FOUND = "NOT_FOUND"
    PORT_ERROR = "PORT_ERROR"
    UNKNOWN = "UNKNOWN"


# Od najwęższej klasy: `PortError` jest bazą wszystkich, więc idzie na końcu.
_CATEGORY_BY_CLASS: tuple[tuple[type[PortError], ErrorCategory], ...] = (
    (ProviderUnavailable, ErrorCategory.PROVIDER_UNAVAILABLE),
    (QuotaExhausted, ErrorCategory.QUOTA_EXHAUSTED),
    (MalformedPayload, ErrorCategory.MALFORMED_PAYLOAD),
    (NotFound, ErrorCategory.NOT_FOUND),
    (PortError, ErrorCategory.PORT_ERROR),
)


def error_category_of(exc: BaseException) -> ErrorCategory:
    """Kategoria z klasy taksonomii; wyjątek spoza niej → `UNKNOWN`. Typ dostawcy
    (`requests`, SDK LLM) nie decyduje — ten sam `requests` stoi w wielu adapterach."""
    for kind, category in _CATEGORY_BY_CLASS:
        if isinstance(exc, kind):
            return category
    return ErrorCategory.UNKNOWN
