"""NIEZ-13: zewnętrzny heartbeat — ping po HTTP na koniec przebiegu.

Watchdog na tym samym GitHub Actions umiera razem z nim, więc żywotność cronów
sprawdza usługa z zewnątrz (np. healthchecks.io): udany przebieg pinguje URL,
nieudany `<URL>/fail`, a brak pingu w oknie to alarm. URL jest sekretem (UUID
checku w ścieżce) — żaden komunikat wyjątku nie może go nieść.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from src.application.ports import HeartbeatPort
from src.infrastructure.adapters.http_heartbeat import (
    HeartbeatPingError,
    HttpHeartbeat,
    NullHeartbeat,
)

_SECRET_UUID = "5f3c0b9e-7a1d-4c2e-9b8f-0d6a2e4c1b7f"
_URL = f"https://hc-ping.com/{_SECRET_UUID}"


def _response(status: int, url: str) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = url
    response.reason = "Not Found" if status == 404 else "OK"
    return response


def test_successful_run_pings_the_url(mocker) -> None:
    get = mocker.patch("requests.Session.get", return_value=_response(200, _URL))

    HttpHeartbeat(_URL).ping(ok=True)

    get.assert_called_once()
    assert get.call_args.args[0] == _URL
    assert get.call_args.kwargs["timeout"] > 0


def test_failed_run_pings_the_fail_endpoint(mocker) -> None:
    get = mocker.patch("requests.Session.get", return_value=_response(200, f"{_URL}/fail"))

    HttpHeartbeat(f"{_URL}/").ping(ok=False)

    assert get.call_args.args[0] == f"{_URL}/fail"


def test_http_error_never_carries_the_url(mocker) -> None:
    mocker.patch("requests.Session.get", return_value=_response(404, _URL))

    with pytest.raises(HeartbeatPingError) as raised:
        HttpHeartbeat(_URL).ping(ok=True)

    message = str(raised.value)
    assert _SECRET_UUID not in message
    assert "hc-ping.com" not in message
    assert "404" in message
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


def test_network_error_never_carries_the_url_or_its_path(mocker) -> None:
    # `requests` wkleja do błędu sieci ścieżkę URL-a — tu siedzi UUID checku.
    mocker.patch(
        "requests.Session.get",
        side_effect=requests.ConnectionError(
            f"HTTPSConnectionPool(host='hc-ping.com', port=443): Max retries exceeded "
            f"with url: /{_SECRET_UUID} (Caused by NewConnectionError)"
        ),
    )

    with pytest.raises(HeartbeatPingError) as raised:
        HttpHeartbeat(_URL).ping(ok=True)

    assert _SECRET_UUID not in str(raised.value)
    assert "ConnectionError" in str(raised.value)


def test_null_heartbeat_does_nothing(mocker) -> None:
    get = mocker.patch("requests.Session.get")

    NullHeartbeat().ping(ok=True)
    NullHeartbeat().ping(ok=False)

    get.assert_not_called()


def test_adapters_implement_the_port() -> None:
    assert isinstance(HttpHeartbeat(_URL), HeartbeatPort)
    assert isinstance(NullHeartbeat(), HeartbeatPort)


def test_port_can_be_mocked() -> None:
    heartbeat = MagicMock(spec=HeartbeatPort)

    heartbeat.ping(ok=True)

    heartbeat.ping.assert_called_once_with(ok=True)
