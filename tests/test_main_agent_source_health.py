"""MAIL-21: composition root składa linijkę stanu źródeł z flag, obwodów i licznika HTTP."""

from __future__ import annotations

from decimal import Decimal

import main_agent
from src.application.ports import CircuitState
from src.application.report_builder import SymbolResult
from src.application.source_health import SOURCE_ORDER, SourceHealth, SourceStatus
from src.config import Settings


def _settings(**overrides: object) -> Settings:
    fields: dict[str, object] = {
        "_env_file": None,
        "openai_api_key": "sk-test",
        "anthropic_api_key": "sk-ant-test",
        "finnhub_api_key": "fh",
        "fred_api_key": "fred",
        "alpha_vantage_api_keys": ["av1"],
        "supabase_url": "https://test.supabase.co",
        "supabase_key": "anon",
        "symbols": ["AAPL", "NVDA"],
        "crypto_symbols": ["BTC"],
        "llm_provider": "anthropic",
        "paid_calls_enabled": True,
        "insider_flow_enabled": False,
        "social_velocity_enabled": False,
        "yield_curve_enabled": True,
        "nbp_enabled": True,
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


def test_enabled_sources_follow_flags_keys_and_the_paid_switch() -> None:
    enabled = main_agent.enabled_sources(_settings())

    assert enabled == {
        "Finnhub": True,
        "CoinGecko": True,
        "Alpha Vantage": True,
        "EDGAR": False,
        "Reddit": False,
        "FRED": True,
        "NBP": True,
        "Anthropic": True,
        "OpenAI": True,
    }
    assert set(enabled) == set(SOURCE_ORDER)


def test_paid_sources_are_disabled_without_paid_calls() -> None:
    enabled = main_agent.enabled_sources(_settings(paid_calls_enabled=False))

    assert not enabled["Alpha Vantage"]
    assert not enabled["Anthropic"]
    assert not enabled["OpenAI"]


def test_cycle_source_health_combines_circuits_quality_flags_and_http_counts() -> None:
    results = [
        SymbolResult(
            symbol="NVDA",
            status="saved",
            delta=Decimal("0.03"),
            data_quality_flags=("llm_analysis_failed",),
        )
    ]
    circuits = (
        CircuitState(
            source="Finnhub",
            host="finnhub.io",
            is_open=True,
            consecutive_failures=5,
            skipped_calls=1,
            skipped_symbols=frozenset({"AAPL"}),
        ),
    )

    rows = {
        row.source: row
        for row in main_agent.cycle_source_health(
            _settings(),
            results,
            circuits=circuits,
            http_calls={"api.stlouisfed.org": {"requests": 2, "5xx": 2}},
        )
    }

    assert rows["Finnhub"] == SourceHealth("Finnhub", SourceStatus.UNAVAILABLE, affected=1)
    assert rows["Anthropic"] == SourceHealth("Anthropic", SourceStatus.DEGRADED, affected=1)
    assert rows["FRED"] == SourceHealth("FRED", SourceStatus.DEGRADED, failed_requests=2)
    assert rows["EDGAR"].status is SourceStatus.DISABLED
