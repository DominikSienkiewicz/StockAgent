"""Testy domeny kosztów transakcyjnych paper-PnL (WIA-50).

Koszt `cost_bps` jest jednostronny: naliczany od obrotu, czyli tylko przy
zmianie pozycji symbolu. Wejście z gotówki i wyjście do gotówki (SIDEWAYS) to
jedna noga (1×), odwrócenie long↔short to dwie (2×), trzymanie pozycji jest
darmowe. Pełny round trip kosztuje więc 2 × `cost_bps`.
"""

from __future__ import annotations

import pytest

from src.domain.trading_costs import net_return, trading_cost, turnover


class TestTurnover:
    def test_entry_and_exit_trade_one_unit_of_capital(self) -> None:
        assert turnover(1, 0) == 1
        assert turnover(-1, 0) == 1
        assert turnover(0, 1) == 1

    def test_flip_trades_two_units(self) -> None:
        assert turnover(-1, 1) == 2
        assert turnover(1, -1) == 2

    def test_unchanged_position_trades_nothing(self) -> None:
        assert turnover(1, 1) == 0
        assert turnover(0, 0) == 0


class TestNetReturn:
    def test_directional_entry_pays_one_leg(self) -> None:
        # 10 bps = 0.001 kapitału przy wejściu z gotówki.
        assert net_return(1, 0, 0.0, 10) == pytest.approx(-0.001)

    def test_holding_the_position_is_free(self) -> None:
        assert net_return(1, 1, 0.02, 10) == 0.02

    def test_sideways_after_a_long_pays_the_exit_leg(self) -> None:
        # SIDEWAYS = wyjście z pozycji: zwrot brutto 0, koszt wyjścia.
        assert net_return(0, 1, 0.0, 10) == pytest.approx(-0.001)

    def test_flip_pays_two_legs(self) -> None:
        assert net_return(-1, 1, 0.01, 10) == pytest.approx(0.01 - 0.002)

    def test_staying_in_cash_is_free(self) -> None:
        assert net_return(0, 0, 0.0, 10) == 0.0

    def test_zero_cost_keeps_the_gross_return(self) -> None:
        assert net_return(-1, 1, 0.03, 0) == 0.03


def test_trading_cost_scales_with_the_traded_capital() -> None:
    assert trading_cost(2, 25) == pytest.approx(0.005)
    assert trading_cost(0.5, 10) == pytest.approx(0.0005)
