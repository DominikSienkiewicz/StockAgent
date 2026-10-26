"""BEZ-15: token Finnhub jedzie nagłówkiem `X-Finnhub-Token`, nie parametrem URL-a.

URL żądania trafia do treści wyjątków `requests`, do logów urllib3 i do tracebacków,
więc sekret w query stringu zależał od redakcji każdej z tych ścieżek. Test idzie
przez prawdziwą sesję `build_session()` (atrapa tylko na poziomie transportu
`HTTPAdapter.send`) dla wszystkich czterech adapterów Finnhub.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from datetime import date
from typing import Any

import pytest
import requests
from requests.adapters import HTTPAdapter

from src.application.errors import PortError
from src.infrastructure.adapters.finnhub_analyst import FinnhubAnalystAdapter
from src.infrastructure.adapters.finnhub_api import FinnhubAdapter
from src.infrastructure.adapters.finnhub_earnings import FinnhubEarningsCalendarAdapter
from src.infrastructure.adapters.finnhub_options import FinnhubOptionsAdapter

TOKEN = "fh-header-secret-token"


def _quote(token: str) -> object:
    return FinnhubAdapter(api_key=token).get_quote("AAPL")


def _consensus(token: str) -> object:
    return FinnhubAnalystAdapter(api_key=token).get_consensus("AAPL")


def _earnings(token: str) -> object:
    adapter = FinnhubEarningsCalendarAdapter(token, ["AAPL"], today=lambda: date(2026, 10, 26))
    return adapter.get_next_earnings("AAPL")


def _options(token: str) -> object:
    return FinnhubOptionsAdapter(api_key=token).get_options_flow("AAPL")


ADAPTER_CALLS = [_quote, _consensus, _earnings, _options]


@pytest.mark.parametrize("call", ADAPTER_CALLS)
def test_token_travels_in_a_header_and_never_in_the_url(
    call: Callable[[str], object], mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
    sent: list[requests.PreparedRequest] = []

    def refuse(request: requests.PreparedRequest, **_kwargs: Any) -> requests.Response:
        # Jak `requests`: komunikat błędu sieci niesie pełny URL żądania.
        sent.append(request)
        raise requests.ConnectionError(f"Max retries exceeded with url: {request.url}")

    mocker.patch.object(HTTPAdapter, "send", side_effect=refuse)
    raised = ""
    with caplog.at_level("DEBUG"), suppress(PortError):
        try:
            call(TOKEN)
        except PortError as exc:
            raised = str(exc)
            raise

    assert sent, "adapter nie wysłał żądania"
    for request in sent:
        assert TOKEN not in (request.url or "")
        assert "token=" not in (request.url or "")
        assert request.headers["X-Finnhub-Token"] == TOKEN
    assert TOKEN not in raised
    assert TOKEN not in caplog.text
