"""ARCH-33: przetłumaczony wyjątek jest klasą taksonomii i nadal typem dostawcy."""

from __future__ import annotations

import pytest
import requests

from src.application.errors import NotFound, PortError, ProviderUnavailable, QuotaExhausted
from src.infrastructure.error_translation import as_port_error


def test_translated_error_keeps_provider_type_message_and_attributes() -> None:
    response = requests.Response()
    response.status_code = 503
    original = requests.HTTPError("503 Server Error", response=response)

    translated = as_port_error(original, ProviderUnavailable)

    assert isinstance(translated, ProviderUnavailable)
    assert isinstance(translated, requests.HTTPError)
    assert str(translated) == "503 Server Error"
    assert translated.response is response
    assert type(translated).__name__ == "ProviderUnavailable[HTTPError]"


def test_same_pair_reuses_one_bridge_class() -> None:
    first = as_port_error(requests.ConnectionError("a"), ProviderUnavailable)
    second = as_port_error(requests.ConnectionError("b"), ProviderUnavailable)

    assert type(first) is type(second)


def test_error_already_of_the_kind_is_returned_unchanged() -> None:
    original = QuotaExhausted("limit")

    assert as_port_error(original, QuotaExhausted) is original


def test_translated_error_keeps_the_original_traceback() -> None:
    def fail() -> None:
        raise requests.ConnectionError("down")

    with pytest.raises(requests.ConnectionError) as excinfo:
        fail()

    translated = as_port_error(excinfo.value, ProviderUnavailable)

    assert translated.__traceback__ is excinfo.value.__traceback__


def test_postgrest_error_keeps_code_for_existing_handlers() -> None:
    from postgrest.exceptions import APIError

    translated = as_port_error(APIError({"message": "missing", "code": "42P01"}), NotFound)

    assert isinstance(translated, APIError)
    assert isinstance(translated, PortError)
    assert translated.code == "42P01"
