"""NIEZ-13: ping zewnętrznego heartbeatu po HTTP (zgodny z healthchecks.io).

Udany przebieg robi `GET <URL>`, nieudany `GET <URL>/fail`; brak pingu w oknie
checku to alarm po stronie usługi. URL jest sekretem (UUID checku w ścieżce):
`requests` wkleja go do `HTTPError`, a ścieżkę także do błędów sieci, więc
wyjątek adaptera niesie wyłącznie typ błędu i status HTTP.
"""

from __future__ import annotations

import requests

from src.application.ports import HeartbeatPort
from src.infrastructure.adapters._http import build_session, raise_for_status_redacted, safe_get

DEFAULT_TIMEOUT_S = 10.0
FAIL_SUFFIX = "/fail"


class HeartbeatPingError(RuntimeError):
    """Ping nie doszedł; komunikat bez URL-a (sekretu)."""


class HttpHeartbeat(HeartbeatPort):
    def __init__(self, url: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._url = url.strip()
        self._timeout = timeout
        self._session = build_session()

    def ping(self, ok: bool) -> None:
        url = self._url if ok else self._url.rstrip("/") + FAIL_SUFFIX
        try:
            response = safe_get(self._session, url, timeout=self._timeout)
            raise_for_status_redacted(response)
        except requests.RequestException as exc:
            raise HeartbeatPingError(_describe(exc)) from None


class NullHeartbeat(HeartbeatPort):
    """Brak `HEARTBEAT_URL` — przebieg nie pinguje nikogo."""

    def ping(self, ok: bool) -> None:
        return None


def _describe(exc: requests.RequestException) -> str:
    status = exc.response.status_code if exc.response is not None else None
    detail = type(exc).__name__ if status is None else f"{type(exc).__name__}, HTTP {status}"
    return f"heartbeat ping failed ({detail})"
