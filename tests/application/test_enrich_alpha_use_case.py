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


class TestEarningsPrefetchDedup:
    """#6 — bramka earnings pobiera EarningsEvent PRZED grafem. Żeby nie płacić
    drugi raz za to samo AV-wywołanie, orkiestrator wstrzykuje pobrany event
    do `enrich`, a ten pomija własny fetch."""

    def test_fetch_earnings_delegates_to_port(self) -> None:
        ports = _ports()
        event = EarningsEvent("AAPL", days_until=1, report_date="2026-07-11")
        ports["earnings_port"].get_next_earnings.return_value = event

        result = AlphaEnrichmentUseCase(**ports).fetch_earnings("AAPL")

        assert result == event
        ports["earnings_port"].get_next_earnings.assert_called_once_with("AAPL")

    def test_fetch_earnings_is_graceful_on_port_error(self) -> None:
        ports = _ports()
        ports["earnings_port"].get_next_earnings.side_effect = RuntimeError("AV 429")

        assert AlphaEnrichmentUseCase(**ports).fetch_earnings("AAPL") is None

    def test_prefetched_earnings_skips_second_port_call(self) -> None:
        ports = _ports()
        _all_none(ports)
        event = EarningsEvent("AAPL", days_until=1, report_date="2026-07-11")

        signals = AlphaEnrichmentUseCase(**ports).enrich(
            "AAPL", earnings=event, earnings_prefetched=True
        )

        assert signals.earnings == event
        ports["earnings_port"].get_next_earnings.assert_not_called()

    def test_prefetched_none_still_skips_the_fetch(self) -> None:
        # Prefetch, który zwrócił None (brak raportu), NIE może wywołać portu
        # ponownie — inaczej de-dup nie działa dokładnie tam, gdzie brak danych.
        ports = _ports()
        _all_none(ports)

        signals = AlphaEnrichmentUseCase(**ports).enrich(
            "AAPL", earnings=None, earnings_prefetched=True
        )

        assert signals.earnings is None
        ports["earnings_port"].get_next_earnings.assert_not_called()

    def test_without_prefetch_flag_enrich_fetches_as_before(self) -> None:
        ports = _ports()
        _all_none(ports)

        AlphaEnrichmentUseCase(**ports).enrich("AAPL")

        ports["earnings_port"].get_next_earnings.assert_called_once_with("AAPL")


class TestMarketCapForTheInsiderThreshold:
    """DAT-21: kapitalizacja z cache fundamentów skaluje próg znaczącego zakupu —
    czytana tylko, gdy insider coś kupił (inaczej zero odczytów)."""

    @staticmethod
    def _buying_insider() -> InsiderFlowSnapshot:
        from src.domain.insider_flow import InsiderBuy

        return InsiderFlowSnapshot(
            "NVDA",
            10_000.0,
            1,
            0,
            90,
            largest_buy=InsiderBuy(role="CEO", value_usd=1_200_000.0),
        )

    @staticmethod
    def _cache(market_cap: float | None) -> Mock:
        from datetime import UTC, datetime

        from src.application.ports import FundamentalsPort
        from src.domain.value_objects import Fundamentals

        cache = Mock(spec=FundamentalsPort)
        cache.get_fundamentals.return_value = Fundamentals(
            trailing_pe=50.0,
            forward_pe=None,
            peg_ratio=None,
            eps_growth_yoy=None,
            fetched_at=datetime(2026, 10, 18, tzinfo=UTC),
            market_cap=market_cap,
        )
        return cache

    def test_insider_buy_gets_the_cached_market_cap(self) -> None:
        ports = _ports()
        _all_none(ports)
        ports["insider_port"].get_insider_flow.return_value = self._buying_insider()
        cache = self._cache(4e12)

        signals = AlphaEnrichmentUseCase(**ports, market_cap_port=cache).enrich("NVDA")

        assert signals.market_cap == 4e12
        # 1,2 mln USD przy 4 bln USD kapitalizacji to poniżej progu (4 mln USD).
        assert signals.significant_insider_buy() is None
        cache.get_fundamentals.assert_called_once_with("NVDA")

    def test_without_an_insider_buy_the_cache_is_not_read(self) -> None:
        ports = _ports()
        _all_none(ports)
        cache = self._cache(4e12)

        signals = AlphaEnrichmentUseCase(**ports, market_cap_port=cache).enrich("NVDA")

        assert signals.market_cap is None
        cache.get_fundamentals.assert_not_called()

    def test_cache_failure_falls_back_to_the_fixed_threshold(self) -> None:
        ports = _ports()
        _all_none(ports)
        ports["insider_port"].get_insider_flow.return_value = self._buying_insider()
        cache = self._cache(None)
        cache.get_fundamentals.side_effect = RuntimeError("supabase down")

        signals = AlphaEnrichmentUseCase(**ports, market_cap_port=cache).enrich("NVDA")

        assert signals.market_cap is None
        assert signals.significant_insider_buy() is not None
