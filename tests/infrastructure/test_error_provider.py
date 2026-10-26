"""MAIL-32: dostawca błędu symbolu z ramek adaptera na stosie wyjątku."""

from __future__ import annotations

import importlib.util
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from src.application.errors import NotFound, ProviderUnavailable
from src.infrastructure.adapters.finnhub_api import FinnhubAdapter
from src.infrastructure.error_provider import PROVIDER_BY_MODULE, provider_of


def _finnhub_timeout(mocker: Any) -> ProviderUnavailable:
    mocker.patch("requests.Session.get", side_effect=requests.Timeout("Read timed out."))
    with pytest.raises(ProviderUnavailable) as caught:
        FinnhubAdapter(api_key="fh-test").get_current_price("NVDA")
    return caught.value


def test_price_adapter_timeout_names_finnhub(mocker: Any) -> None:
    assert provider_of(_finnhub_timeout(mocker)) == "Finnhub"


def test_error_raised_outside_any_adapter_has_no_provider() -> None:
    with pytest.raises(RuntimeError) as caught:
        raise RuntimeError("bug w grafie")

    assert provider_of(caught.value) is None


def test_wrapped_adapter_error_keeps_its_provider(mocker: Any) -> None:
    # Warstwa wyżej opakowała błąd adaptera (`raise … from exc`) — dostawcę zna przyczyna.
    cause = _finnhub_timeout(mocker)
    with pytest.raises(RuntimeError) as caught:
        raise RuntimeError("graf przerwał symbol") from cause

    assert provider_of(caught.value) == "Finnhub"


def test_adapter_own_error_names_its_provider(mocker: Any) -> None:
    # Adapter rzucił sam (brak ceny w odpowiedzi) — bez wyjątku biblioteki pod spodem.
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = {"c": 0}
    mocker.patch("requests.Session.get", return_value=response)

    with pytest.raises(NotFound) as caught:
        FinnhubAdapter(api_key="fh-test").get_current_price("ZZZQ")

    assert provider_of(caught.value) == "Finnhub"


def test_shared_http_helpers_do_not_hide_the_adapter() -> None:
    # `_http.py` i wrappery nie mają dostawcy — nie ma ich w mapie.
    assert "src.infrastructure.adapters._http" not in PROVIDER_BY_MODULE
    assert "src.infrastructure.adapters.write_guard" not in PROVIDER_BY_MODULE


@pytest.mark.parametrize("module", sorted(PROVIDER_BY_MODULE))
def test_every_mapped_module_exists(module: str) -> None:
    # Zmiana nazwy adaptera bez aktualizacji mapy po cichu gubiłaby dostawcę w mailu.
    assert importlib.util.find_spec(module) is not None
