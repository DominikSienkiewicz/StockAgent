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


class TestRevision:
    """DAT-13: rewizja konsensusu m/m — UPGRADE / DOWNGRADE / STABLE z progiem
    zmiany średniej oceny (0.3) albo przesunięcia netto analityków (2)."""

    def test_revision_detects_upgrade(self) -> None:
        from src.domain.analyst_consensus import RevisionDirection

        previous = AnalystConsensus(symbol="AAPL", buy=10, hold=10)
        latest = AnalystConsensus(
            symbol="AAPL", strong_buy=3, buy=12, hold=5, previous=previous
        )

        revision = latest.revision()

        assert revision.direction is RevisionDirection.UPGRADE
        assert (revision.strong_buy_change, revision.buy_change, revision.hold_change) == (
            3,
            2,
            -5,
        )
        assert revision.bullish_change == 5

    def test_revision_flat_with_single_period(self) -> None:
        from src.domain.analyst_consensus import RevisionDirection

        revision = AnalystConsensus(symbol="AAPL", strong_buy=3, buy=12).revision()

        assert revision.direction is RevisionDirection.STABLE
        assert revision.bullish_change == 0

    def test_two_analysts_moving_to_hold_is_a_downgrade(self) -> None:
        # Średnia spada tylko o ~0.07, ale dwóch analityków zeszło z kupna.
        from src.domain.analyst_consensus import RevisionDirection

        previous = AnalystConsensus(symbol="MSFT", buy=20, hold=10)
        latest = AnalystConsensus(symbol="MSFT", buy=18, hold=12)

        assert latest.revision_vs(previous).direction is RevisionDirection.DOWNGRADE

    def test_one_analyst_moving_is_stable(self) -> None:
        from src.domain.analyst_consensus import RevisionDirection

        previous = AnalystConsensus(symbol="MSFT", buy=20, hold=10)
        latest = AnalystConsensus(symbol="MSFT", buy=19, hold=11)

        assert latest.revision_vs(previous).direction is RevisionDirection.STABLE

    def test_mean_score_jump_without_net_shift_is_an_upgrade(self) -> None:
        # Hold → strong buy: strona kupna +1, ale średnia rośnie o 0.4.
        from src.domain.analyst_consensus import RevisionDirection

        previous = AnalystConsensus(symbol="KO", buy=1, hold=4)
        latest = AnalystConsensus(symbol="KO", strong_buy=1, buy=1, hold=3)

        assert latest.revision_vs(previous).direction is RevisionDirection.UPGRADE

    def test_contradicting_mean_and_shift_is_stable(self) -> None:
        # Średnia rośnie (4.0 → 4.67), ale strona kupna straciła czterech analityków
        # (pokrycie się skurczyło) — sygnały sprzeczne, bez rewizji.
        from src.domain.analyst_consensus import RevisionDirection

        previous = AnalystConsensus(symbol="XYZ", buy=10)
        latest = AnalystConsensus(symbol="XYZ", strong_buy=4, buy=2)

        assert latest.revision_vs(previous).direction is RevisionDirection.STABLE

    def test_empty_previous_period_counts_only_the_shift(self) -> None:
        from src.domain.analyst_consensus import RevisionDirection

        previous = AnalystConsensus(symbol="NEW")
        latest = AnalystConsensus(symbol="NEW", buy=3)

        assert latest.revision_vs(previous).direction is RevisionDirection.UPGRADE

    def test_mean_change_exactly_at_the_step_counts(self) -> None:
        # 3.5 → 3.8 to w floatach 0.2999…; próg 0.3 ma go objąć. Strona kupna bez
        # zmiany liczebności (3 buy → strong buy), więc decyduje sama średnia.
        from src.domain.analyst_consensus import RevisionDirection

        previous = AnalystConsensus(symbol="KO", buy=5, hold=5)
        latest = AnalystConsensus(symbol="KO", strong_buy=3, buy=2, hold=5)

        assert latest.revision_vs(previous).direction is RevisionDirection.UPGRADE
