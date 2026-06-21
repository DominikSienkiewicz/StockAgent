"""Testy domeny InsiderFlowSnapshot.evaluate_signal.

Czysta logika klasyfikacji przepływu insiderów — bez I/O, bez adaptera.
Sprawdzamy progi NET_BUYING / NET_SELLING / NEUTRAL oraz przypadek braku
transakcji (zero buy_count + sell_count → NEUTRAL niezależnie od net_shares).
"""

from __future__ import annotations

from src.domain.insider_flow import InsiderFlowSnapshot, InsiderSignal


def _snap(
    net_shares: float,
    buy_count: int = 0,
    sell_count: int = 0,
    window_days: int = 90,
) -> InsiderFlowSnapshot:
    return InsiderFlowSnapshot(
        symbol="AAPL",
        net_shares=net_shares,
        buy_count=buy_count,
        sell_count=sell_count,
        window_days=window_days,
    )


class TestEvaluateSignal:
    def test_zero_transactions_is_neutral(self) -> None:
        # Brak jakichkolwiek transakcji → NEUTRAL nawet gdy net_shares != 0.
        snap = _snap(net_shares=0.0, buy_count=0, sell_count=0)
        assert snap.evaluate_signal() is InsiderSignal.NEUTRAL

    def test_zero_transactions_neutral_even_with_nonzero_net(self) -> None:
        # net_shares "zabłąkany" przy zerze transakcji nie zmienia werdyktu.
        snap = _snap(net_shares=1000.0, buy_count=0, sell_count=0)
        assert snap.evaluate_signal() is InsiderSignal.NEUTRAL

    def test_positive_net_is_buying(self) -> None:
        snap = _snap(net_shares=500.0, buy_count=2, sell_count=1)
        assert snap.evaluate_signal() is InsiderSignal.NET_BUYING

    def test_negative_net_is_selling(self) -> None:
        snap = _snap(net_shares=-500.0, buy_count=1, sell_count=3)
        assert snap.evaluate_signal() is InsiderSignal.NET_SELLING

    def test_net_zero_with_transactions_is_neutral(self) -> None:
        # Kupno i sprzedaż znoszą się do zera → martwa strefa → NEUTRAL.
        snap = _snap(net_shares=0.0, buy_count=1, sell_count=1)
        assert snap.evaluate_signal() is InsiderSignal.NEUTRAL

    def test_threshold_dead_zone_is_neutral(self) -> None:
        # net_shares w zakresie [-min, +min] → NEUTRAL.
        snap = _snap(net_shares=50.0, buy_count=1, sell_count=0)
        assert snap.evaluate_signal(min_net_shares=100.0) is InsiderSignal.NEUTRAL

    def test_above_threshold_is_buying(self) -> None:
        snap = _snap(net_shares=150.0, buy_count=1, sell_count=0)
        assert snap.evaluate_signal(min_net_shares=100.0) is InsiderSignal.NET_BUYING

    def test_below_negative_threshold_is_selling(self) -> None:
        snap = _snap(net_shares=-150.0, buy_count=0, sell_count=1)
        assert snap.evaluate_signal(min_net_shares=100.0) is InsiderSignal.NET_SELLING

    def test_exactly_at_threshold_is_neutral(self) -> None:
        # Próg jest ścisły (>), więc równość = NEUTRAL.
        snap = _snap(net_shares=100.0, buy_count=1, sell_count=0)
        assert snap.evaluate_signal(min_net_shares=100.0) is InsiderSignal.NEUTRAL
