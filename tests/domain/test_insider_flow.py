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



class TestHasEvidence:
    """DAT-03: snapshot mówi, czy EDGAR dał dowód o insiderach. Niesparsowane Form 4
    są nieodróżnialne od braku transakcji, jeśli liczniki filingów tego nie powiedzą."""

    def test_defaults_keep_the_snapshot_as_evidence(self) -> None:
        snap = _snap(net_shares=0.0)
        assert (snap.filings_seen, snap.filings_parsed) == (0, 0)
        assert snap.has_evidence()

    def test_seen_but_unparsed_filings_are_no_evidence(self) -> None:
        snap = InsiderFlowSnapshot("AAPL", 0.0, 0, 0, 90, filings_seen=3, filings_parsed=0)
        assert not snap.has_evidence()

    def test_one_parsed_filing_is_evidence(self) -> None:
        snap = InsiderFlowSnapshot("AAPL", -500.0, 0, 1, 90, filings_seen=3, filings_parsed=1)
        assert snap.has_evidence()

    def test_no_filings_in_the_window_is_evidence_of_no_trades(self) -> None:
        snap = InsiderFlowSnapshot("AAPL", 0.0, 0, 0, 90, filings_seen=0, filings_parsed=0)
        assert snap.has_evidence()
