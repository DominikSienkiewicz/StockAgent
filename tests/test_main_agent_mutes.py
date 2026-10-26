"""MAIL-45 — wyciszenia z bota w Fast Loopie (za `symbol_mutes_enabled`).

Wyciszony symbol nie dostaje płatnej analizy ani sekcji maila, ale dostaje snapshot
ceny (punkt odniesienia po powrocie). Mail niesie linię „watchlista: N z config − K
wyciszone”. Flaga off: zero odczytów tabeli, cykl i mail jak przed MAIL-45.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, Mock

import main_agent
from src.application.errors import ProviderUnavailable
from src.application.ports import MarketDataPort, SymbolMutePort
from src.config import Settings
from src.domain.value_objects import Money
from src.domain.watchlist import SymbolMute
from src.infrastructure.adapters.supabase_repo import SupabaseRepository

NOW = datetime(2026, 12, 23, 5, 30, tzinfo=UTC)


class TestFastLoopMutes:
    def _run(
        self,
        settings: Settings,
        mocker: Any,
        *,
        mutes: list[SymbolMute] | Exception,
        quotes: list[Money | Exception] | None = None,
    ) -> dict[str, MagicMock]:
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mute_port = Mock(spec=SymbolMutePort)
        if isinstance(mutes, Exception):
            mute_port.get_symbol_mutes.side_effect = mutes
        else:
            mute_port.get_symbol_mutes.return_value = mutes
        market = Mock(spec=MarketDataPort)
        if quotes is None:
            market.get_current_price.return_value = Money(Decimal("187.12"))
        else:
            market.get_current_price.side_effect = quotes
        mocks = {
            "use_case": fake_uc,
            "mutes": mute_port,
            "market": market,
            "build_mutes": mocker.patch("main_agent.build_symbol_mutes", return_value=mute_port),
            "snapshot": mocker.patch.object(SupabaseRepository, "save_price_snapshot"),
            "dispatch": mocker.patch("main_agent._dispatch_reports", return_value=True),
        }
        mocker.patch("main_agent.build_use_case", return_value=fake_uc)
        mocker.patch("main_agent.build_market_port", return_value=market)
        mocker.patch("main_agent.build_notifier", return_value=MagicMock())
        self.exit_code = main_agent.main(settings, clock=lambda: NOW)
        return mocks

    def test_muted_symbol_skips_the_graph_but_keeps_its_price_snapshot(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        settings.symbol_mutes_enabled = True
        settings.crypto_symbols = ["BTC"]

        mocks = self._run(settings, mocker, mutes=[SymbolMute("VOO", NOW + timedelta(days=3))])

        analysed = [call.args[0] for call in mocks["use_case"].run.call_args_list]
        assert analysed == ["AAPL", "BTC"]
        mocks["market"].get_current_price.assert_called_once_with("VOO")
        mocks["snapshot"].assert_called_once_with("VOO", Money(Decimal("187.12")))
        context = mocks["dispatch"].call_args.kwargs["context"]
        assert context.watchlist_mutes.config_count == 3
        assert [mute.symbol for mute in context.watchlist_mutes.muted] == ["VOO"]
        assert mocks["dispatch"].call_args.kwargs["cycle_symbols"] == ["AAPL", "BTC"]
        assert self.exit_code == 0

    def test_expired_mute_brings_the_symbol_back(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        settings.symbol_mutes_enabled = True

        mocks = self._run(settings, mocker, mutes=[SymbolMute("VOO", NOW - timedelta(hours=1))])

        analysed = [call.args[0] for call in mocks["use_case"].run.call_args_list]
        assert analysed == ["AAPL", "VOO"]
        mocks["snapshot"].assert_not_called()

    def test_unreadable_mutes_analyse_the_whole_config(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        settings.symbol_mutes_enabled = True

        mocks = self._run(settings, mocker, mutes=ProviderUnavailable("db down"))

        analysed = [call.args[0] for call in mocks["use_case"].run.call_args_list]
        assert analysed == ["AAPL", "VOO"]
        assert mocks["dispatch"].call_args.kwargs["context"].watchlist_mutes.muted == ()

    def test_failed_quote_of_a_muted_symbol_does_not_stop_the_cycle(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:
        settings.symbol_mutes_enabled = True
        settings.symbols = ["AAPL", "VOO", "NVDA"]
        mutes = [SymbolMute(s, NOW + timedelta(days=1)) for s in ("VOO", "NVDA")]

        mocks = self._run(
            settings,
            mocker,
            mutes=mutes,
            quotes=[ProviderUnavailable("finnhub down"), Money(Decimal("120"))],
        )

        mocks["snapshot"].assert_called_once_with("NVDA", Money(Decimal("120")))
        assert self.exit_code == 0

    def test_flag_off_reads_no_mutes_and_keeps_the_context_field_empty(
        self, settings: Settings, mock_external_clients: None, mocker: Any
    ) -> None:

        mocks = self._run(settings, mocker, mutes=[SymbolMute("VOO", NOW + timedelta(days=3))])

        mocks["build_mutes"].assert_not_called()
        analysed = [call.args[0] for call in mocks["use_case"].run.call_args_list]
        assert analysed == ["AAPL", "VOO"]
        assert mocks["dispatch"].call_args.kwargs["context"].watchlist_mutes is None
        mocks["snapshot"].assert_not_called()


def test_owner_mutes_do_not_reach_a_stranger_subscriber() -> None:
    from src.application.report_builder import ReportContext
    from src.domain.watchlist import effective

    context = ReportContext(
        watchlist_mutes=effective(["NVDA"], [SymbolMute("NVDA", NOW + timedelta(days=1))], NOW)
    )

    assert main_agent._without_owner_portfolio(context).watchlist_mutes is None
