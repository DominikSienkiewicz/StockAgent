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
