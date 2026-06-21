"""Testy AlphaEnrichmentUseCase — agreguje 5 portów alpha per symbol.

Każdy port jest opcjonalnym źródłem; use case zbiera ich snapshoty w AlphaSignals
i NIGDY nie wywala się na błędzie pojedynczego źródła (graceful per-port)."""

from __future__ import annotations

from unittest.mock import Mock

from src.application.ports import (
    AnalystConsensusPort,
    EarningsCalendarPort,
    InsiderFlowPort,
    OptionsFlowPort,
    SocialVelocityPort,
)
from src.application.use_cases.enrich_alpha import AlphaEnrichmentUseCase
from src.domain.analyst_consensus import AnalystConsensus
from src.domain.earnings import EarningsEvent
from src.domain.insider_flow import InsiderFlowSnapshot
from src.domain.options_flow import OptionsFlowSnapshot
from src.domain.social_velocity import SocialVelocitySnapshot


def _ports() -> dict[str, Mock]:
    return {
        "insider_port": Mock(spec=InsiderFlowPort),
        "earnings_port": Mock(spec=EarningsCalendarPort),
        "options_port": Mock(spec=OptionsFlowPort),
        "social_port": Mock(spec=SocialVelocityPort),
        "analyst_port": Mock(spec=AnalystConsensusPort),
    }


def _all_none(ports: dict[str, Mock]) -> None:
    ports["insider_port"].get_insider_flow.return_value = None
    ports["earnings_port"].get_next_earnings.return_value = None
    ports["options_port"].get_options_flow.return_value = None
    ports["social_port"].get_social_velocity.return_value = None
    ports["analyst_port"].get_consensus.return_value = None


def test_collects_all_sources() -> None:
    ports = _ports()
    _all_none(ports)
    ports["insider_port"].get_insider_flow.return_value = InsiderFlowSnapshot(
        "AAPL", net_shares=1000.0, buy_count=3, sell_count=1, window_days=90
    )
    ports["earnings_port"].get_next_earnings.return_value = EarningsEvent(
        "AAPL", days_until=3, report_date="2026-06-25"
    )
    ports["options_port"].get_options_flow.return_value = OptionsFlowSnapshot(
        "AAPL", put_call_ratio=0.6, implied_vol=0.3
    )
    ports["social_port"].get_social_velocity.return_value = SocialVelocitySnapshot(
        "AAPL", mentions_24h=50, baseline_mentions=10.0
    )
    ports["analyst_port"].get_consensus.return_value = AnalystConsensus(
        "AAPL", strong_buy=10, buy=5, price_target=220.0
    )

    signals = AlphaEnrichmentUseCase(**ports).enrich("AAPL")

    assert signals.symbol == "AAPL"
    assert signals.insider is not None
    assert signals.earnings is not None
    assert signals.options is not None
    assert signals.social is not None
    assert signals.analyst is not None
    assert signals.has_any()
    ports["insider_port"].get_insider_flow.assert_called_once_with("AAPL")
    ports["analyst_port"].get_consensus.assert_called_once_with("AAPL")


def test_has_any_false_when_all_sources_empty() -> None:
    ports = _ports()
    _all_none(ports)
    signals = AlphaEnrichmentUseCase(**ports).enrich("AAPL")
    assert not signals.has_any()


def test_graceful_when_a_port_raises() -> None:
    ports = _ports()
    _all_none(ports)
    # Insider rzuca — nie wolno wywalić całej agregacji; pozostałe źródła OK.
    ports["insider_port"].get_insider_flow.side_effect = RuntimeError("EDGAR 503")
    ports["analyst_port"].get_consensus.return_value = AnalystConsensus(
        "AAPL", buy=7
    )

    signals = AlphaEnrichmentUseCase(**ports).enrich("AAPL")

    assert signals.insider is None          # błąd zdegradowany do None
    assert signals.analyst is not None      # reszta nadal zebrana
    assert signals.has_any()
