"""NIEZ-31: Fast Loop zapisuje otwarte obwody HTTP do bannera kwot przed raportem."""

from __future__ import annotations

from contextlib import suppress
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

import main_agent
from src.config import Settings
from src.infrastructure.adapters._http import CircuitOpenError, process_circuits
from src.infrastructure.logging_setup import symbol_context
from tests.settings_kwargs import NO_ENV_FILE


@pytest.fixture
def settings() -> Settings:
    return Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1", "av2"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL", "VOO"],
        volatility_threshold=Decimal("0.02"),
        ml_model_path="data/models/missing.ubj",
        db_writes_enabled=True,
        paid_calls_enabled=True,
    )


def _open_finnhub_circuit_skipping(*symbols: str) -> None:
    circuits = process_circuits()
    for _ in range(5):
        circuits.record_failure("finnhub.io")
    for symbol in symbols:
        with symbol_context(symbol), suppress(CircuitOpenError):
            circuits.before_request("finnhub.io")


def test_open_circuit_reaches_the_quota_banner(settings: Settings, mocker: Any) -> None:
    mocker.patch("src.infrastructure.adapters.supabase_repo.build_supabase_client")
    mocker.patch("src.infrastructure.llm.openai_client.OpenAI")
    use_case = MagicMock()

    def run(*_args: object, **_kwargs: object) -> dict[str, object]:
        _open_finnhub_circuit_skipping("AAPL", "VOO")
        return {"status": "ignored", "delta": Decimal("0")}

    use_case.run.side_effect = run
    mocker.patch("main_agent.build_use_case", return_value=use_case)
    dispatch = mocker.patch("main_agent._dispatch_reports", return_value=True)

    assert main_agent.main(settings) == 0

    alerts = dispatch.call_args.kwargs["quota_monitor"].alerts
    assert [a.message for a in alerts if a.source == "Finnhub"] == [
        "Finnhub niedostępny — 2 symbole pominięte."
    ]
