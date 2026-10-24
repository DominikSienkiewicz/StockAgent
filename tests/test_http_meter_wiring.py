"""NIEZ-24: Fast Loop liczy żądania HTTP wszystkich adapterów jednym miernikiem.

Adaptery budują sesję gołym `build_session()`; miernik instaluje entry point na
cały przebieg, a ten sam obiekt trafia do telemetrii cyklu (step summary).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest
import requests
from requests.adapters import HTTPAdapter

import main_agent
from src.application.http_metering import HttpCallMeter
from src.config import Settings
from src.infrastructure.adapters._http import build_session


class _OkAdapter(HTTPAdapter):
    def send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        response = requests.Response()
        response.status_code = 200
        response.url = request.url
        response.request = request
        response._content = b"{}"
        return response


def test_fast_loop_run_installs_one_meter_and_hands_it_to_the_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[HttpCallMeter | None] = []

    def fake_run(
        settings: Settings,
        clock: Callable[[], datetime],
        http_meter: HttpCallMeter | None = None,
    ) -> int:
        session = build_session()
        session.mount("https://", _OkAdapter())
        session.get("https://finnhub.io/api/v1/quote?token=SECRET123")
        seen.append(http_meter)
        return 0

    monkeypatch.setattr(main_agent, "_run_fast_loop", fake_run)

    exit_code = main_agent._run_fast_loop_metered(
        Settings.model_construct(), clock=lambda: datetime(2026, 10, 21)
    )

    assert exit_code == 0
    (meter,) = seen
    assert meter is not None
    assert meter.snapshot() == {"finnhub.io": {"requests": 1, "2xx": 1}}


def test_meter_is_uninstalled_after_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    meters: list[HttpCallMeter | None] = []
    monkeypatch.setattr(
        main_agent,
        "_run_fast_loop",
        lambda settings, clock, http_meter=None: meters.append(http_meter) or 0,
    )
    main_agent._run_fast_loop_metered(
        Settings.model_construct(), clock=lambda: datetime(2026, 1, 1)
    )

    session = build_session()
    session.mount("https://", _OkAdapter())
    session.get("https://finnhub.io/x")

    (meter,) = meters
    assert meter is not None
    assert meter.snapshot() == {}
