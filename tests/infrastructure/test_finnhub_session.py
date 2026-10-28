"""Sesja adapterów Finnhub nie ponawia 429 (Fast Loop i Shock Watch 2026-10-28).

Retry urllib3 działa pod wspólnym `RateLimiter`, więc każde 429 kosztowało cztery
zapytania spoza limitera (jedno i trzy ponowienia w ~3 s), a okno limitu Finnhub trwa
minutę: ponowienia tylko przedłużały blokadę klucza. Test pyta urllib3 o jego własną
decyzję (`Retry.is_retry`) — także dla 429 z nagłówkiem `Retry-After`, który urllib3
ponawia mimo braku statusu na liście. 5xx dalej się ponawia.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, cast

import pytest
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.infrastructure.adapters.finnhub_analyst import FinnhubAnalystAdapter
from src.infrastructure.adapters.finnhub_api import FinnhubAdapter
from src.infrastructure.adapters.finnhub_earnings import FinnhubEarningsCalendarAdapter
from src.infrastructure.adapters.finnhub_options import FinnhubOptionsAdapter


def _quote() -> object:
    return FinnhubAdapter(api_key="k")


def _analyst() -> object:
    return FinnhubAnalystAdapter(api_key="k")


def _earnings() -> object:
    return FinnhubEarningsCalendarAdapter("k", ["AAPL"], today=lambda: date(2026, 10, 28))


def _options() -> object:
    return FinnhubOptionsAdapter(api_key="k")


ADAPTERS = [_quote, _analyst, _earnings, _options]


def _retry(build: Callable[[], object]) -> Retry:
    session = cast(Any, build())._session
    retry = cast(HTTPAdapter, session.get_adapter("https://finnhub.io/api/v1/quote")).max_retries
    assert isinstance(retry, Retry)
    return retry


@pytest.mark.parametrize("has_retry_after", [False, True])
@pytest.mark.parametrize("build", ADAPTERS)
def test_rate_limited_response_is_not_retried(
    build: Callable[[], object], has_retry_after: bool
) -> None:
    assert not _retry(build).is_retry("GET", 429, has_retry_after=has_retry_after)


@pytest.mark.parametrize("build", ADAPTERS)
def test_server_errors_are_still_retried(build: Callable[[], object]) -> None:
    assert _retry(build).is_retry("GET", 503)
