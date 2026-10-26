"""MAIL-32: kategoria błędu symbolu z taksonomii portów (ARCH-33).

Mail mówi czytelnikowi, czy błąd to awaria dostawcy, limit, zła odpowiedź, brak
danych czy błąd agenta — zamiast surowego `str(exc)`. Kategorię wyznacza klasa
taksonomii, więc przetłumaczony wyjątek dostawcy trafia do niej mimo typu
dostawcy (`ProviderUnavailable[Timeout]`).
"""

from __future__ import annotations

import pytest
import requests

from src.application.errors import (
    ErrorCategory,
    LLMTruncatedError,
    MalformedPayload,
    NotFound,
    PortError,
    ProviderUnavailable,
    QuotaExhausted,
    error_category_of,
)


class _TranslatedTimeout(ProviderUnavailable, requests.Timeout):
    """Kształt mostka `as_port_error` (`ProviderUnavailable[Timeout]`) bez importu
    infrastruktury do testu warstwy aplikacji."""


@pytest.mark.parametrize(
    ("exc", "category"),
    [
        (ProviderUnavailable("down"), ErrorCategory.PROVIDER_UNAVAILABLE),
        (QuotaExhausted("429"), ErrorCategory.QUOTA_EXHAUSTED),
        (MalformedPayload("no field"), ErrorCategory.MALFORMED_PAYLOAD),
        (LLMTruncatedError("cut", output_tokens=10), ErrorCategory.MALFORMED_PAYLOAD),
        (NotFound("ticker"), ErrorCategory.NOT_FOUND),
        (PortError("401"), ErrorCategory.PORT_ERROR),
        (RuntimeError("bug"), ErrorCategory.UNKNOWN),
    ],
    ids=["unavailable", "quota", "malformed", "truncated", "not_found", "port", "unknown"],
)
def test_category_follows_the_port_error_class(exc: Exception, category: ErrorCategory) -> None:
    assert error_category_of(exc) is category


def test_translated_provider_timeout_is_provider_unavailable() -> None:
    timeout = _TranslatedTimeout("Read timed out.")

    assert error_category_of(timeout) is ErrorCategory.PROVIDER_UNAVAILABLE


def test_untranslated_provider_exception_is_unknown() -> None:
    # Adapter, który jeszcze nie tłumaczy (ARCH-33 jest przyrostowe), to nie awaria
    # dostawcy w sensie taksonomii — mail nie zgaduje po typie `requests`.
    assert error_category_of(requests.Timeout("Read timed out.")) is ErrorCategory.UNKNOWN
