"""Wspólny HTTP client z retry/backoff dla transientnych błędów.

Adaptery (Finnhub, Alpha Vantage, Resend) używają `build_session()` zamiast
gołego `requests` — dzięki temu pojedynczy timeout albo chwilowy 5xx nie
wywala całego symbolu z cyklu.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Retry dla błędów, które zwykle są przejściowe:
#   429 — rate limit, 500/502/503/504 — chwilowa awaria po stronie API.
# backoff_factor=0.5 → przerwy 0.5s, 1s, 2s między próbami.
RETRY_TOTAL = 3
RETRY_BACKOFF_FACTOR = 0.5
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)


def build_session() -> requests.Session:
    """Zwraca `requests.Session` z zamontowanym retry na http:// i https://."""
    session = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,  # status sprawdzamy sami przez raise_for_status()
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
