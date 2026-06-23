"""Testy domeny AnalystConsensus — koszyki rating(), upside() i agregaty.

Czysta logika, zero I/O. Sprawdzamy mapowanie ważonej średniej na koszyk
oceny, liczenie potencjału wzrostu względem ceny bieżącej (Decimal) oraz
zachowanie przy pustym rozkładzie rekomendacji.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.analyst_consensus import AnalystConsensus, AnalystRating


class TestRatingBuckets:
    def test_strong_buy_when_mean_at_or_above_4_5(self) -> None:
        # Same strong buy → średnia 5.0 → STRONG_BUY.
        consensus = AnalystConsensus(symbol="AAPL", strong_buy=10)
        assert consensus.rating() is AnalystRating.STRONG_BUY

    def test_buy_when_mean_in_3_5_to_4_5(self) -> None:
        # Same buy → średnia 4.0 → BUY.
        consensus = AnalystConsensus(symbol="MSFT", buy=8)
        assert consensus.rating() is AnalystRating.BUY

    def test_hold_when_mean_in_2_5_to_3_5(self) -> None:
        # Same hold → średnia 3.0 → HOLD.
        consensus = AnalystConsensus(symbol="KO", hold=6)
        assert consensus.rating() is AnalystRating.HOLD

    def test_sell_when_mean_in_1_5_to_2_5(self) -> None:
        # Same sell → średnia 2.0 → SELL.
        consensus = AnalystConsensus(symbol="XYZ", sell=4)
        assert consensus.rating() is AnalystRating.SELL

    def test_strong_sell_when_mean_below_1_5(self) -> None:
        # Same strong sell → średnia 1.0 → STRONG_SELL.
        consensus = AnalystConsensus(symbol="ABC", strong_sell=3)
        assert consensus.rating() is AnalystRating.STRONG_SELL

    def test_mixed_distribution_lands_in_buy(self) -> None:
        # 13*5 + 18*4 + 5*3 + 1*2 + 0 = 65 + 72 + 15 + 2 = 154 / 37 ≈ 4.16 → BUY.
        consensus = AnalystConsensus(
            symbol="AAPL",
            strong_buy=13,
            buy=18,
            hold=5,
            sell=1,
            strong_sell=0,
        )
        assert consensus.rating() is AnalystRating.BUY

    def test_empty_distribution_falls_back_to_hold(self) -> None:
        consensus = AnalystConsensus(symbol="NONE")
        assert consensus.rating() is AnalystRating.HOLD


class TestUpside:
    def test_positive_upside_with_decimal_price(self) -> None:
        # cel 150, cena 100 → (150 - 100) / 100 = 0.5.
        consensus = AnalystConsensus(symbol="AAPL", buy=5, price_target=150.0)
        assert consensus.upside(Decimal("100")) == pytest.approx(0.5)

    def test_negative_upside_when_target_below_price(self) -> None:
        # cel 80, cena 100 → -0.2.
        consensus = AnalystConsensus(symbol="AAPL", hold=5, price_target=80.0)
        assert consensus.upside(Decimal("100")) == pytest.approx(-0.2)

    def test_none_when_no_price_target(self) -> None:
        consensus = AnalystConsensus(symbol="AAPL", buy=5, price_target=None)
        assert consensus.upside(Decimal("100")) is None

    def test_none_when_current_price_not_positive(self) -> None:
        consensus = AnalystConsensus(symbol="AAPL", buy=5, price_target=150.0)
        assert consensus.upside(Decimal("0")) is None


class TestAggregates:
    def test_total_sums_all_buckets(self) -> None:
        consensus = AnalystConsensus(
            symbol="AAPL",
            strong_buy=1,
            buy=2,
            hold=3,
            sell=4,
            strong_sell=5,
        )
        assert consensus.total() == 15

    def test_mean_score_none_on_empty(self) -> None:
        consensus = AnalystConsensus(symbol="NONE")
        assert consensus.mean_score() is None
        assert consensus.total() == 0
