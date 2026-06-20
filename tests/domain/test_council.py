# tests/domain/test_council.py
from decimal import Decimal

import pytest

from src.domain.council import (
    CouncilInput,
    CouncilVerdict,
    InvestorOpinion,
    derive_consensus,
)


def _make_opinion(rec: str = "BUY", confidence: float = 0.8) -> InvestorOpinion:
    return InvestorOpinion(
        investor_name="Warren Buffett",
        recommendation=rec,
        confidence=confidence,
        reasoning="Silna fosa ekonomiczna.",
        key_factors=["moat", "FCF", "długi horyzont"],
    )


class TestInvestorOpinion:
    def test_frozen(self):
        op = _make_opinion()
        try:
            op.confidence = 0.1  # type: ignore[misc]
            raise AssertionError("should raise")
        except (AttributeError, TypeError):
            pass

    def test_recommendation_stored(self):
        assert _make_opinion("SELL").recommendation == "SELL"

    def test_key_factors_stored(self):
        op = _make_opinion()
        assert "moat" in op.key_factors


class TestCouncilVerdict:
    def test_frozen(self):
        opinions = [_make_opinion()]
        v = CouncilVerdict(
            final_recommendation="BUY",
            consensus_strength=0.73,
            summary="Większość radzi kupować.",
            dissenting_views=["Soros: zbyt ryzykowne"],
            investor_opinions=opinions,
        )
        try:
            v.consensus_strength = 0.1  # type: ignore[misc]
            raise AssertionError("should raise")
        except (AttributeError, TypeError):
            pass

    def test_holds_all_opinions(self):
        ops = [_make_opinion("BUY"), _make_opinion("SELL")]
        v = CouncilVerdict(
            final_recommendation="BUY",
            consensus_strength=0.5,
            summary="Mieszane opinie.",
            dissenting_views=["Livermore: sprzedaj"],
            investor_opinions=ops,
        )
        assert len(v.investor_opinions) == 2


class TestCouncilVerdictBehavior:
    """Metody domenowe — zastępują ad-hoc logikę w grafie/raporcie.

    Zachowanie należy do agregata, nie do warstwy prezentacji."""

    def _verdict(
        self, opinions: list[InvestorOpinion],
        final: str = "BUY", consensus: float = 0.7,
    ) -> CouncilVerdict:
        return CouncilVerdict(
            final_recommendation=final,  # type: ignore[arg-type]
            consensus_strength=consensus,
            summary="Test.",
            dissenting_views=[],
            investor_opinions=opinions,
        )

    # --- has_strong_consensus ---

    def test_strong_consensus_above_threshold(self):
        v = self._verdict([_make_opinion()], consensus=0.85)
        assert v.has_strong_consensus() is True

    def test_strong_consensus_default_threshold_is_0_7(self):
        # Próg 0.7 — typowy "wyraźny konsensus" dla rady.
        v = self._verdict([_make_opinion()], consensus=0.69)
        assert v.has_strong_consensus() is False
        v2 = self._verdict([_make_opinion()], consensus=0.70)
        assert v2.has_strong_consensus() is True

    def test_strong_consensus_custom_threshold(self):
        v = self._verdict([_make_opinion()], consensus=0.55)
        assert v.has_strong_consensus(threshold=0.5) is True
        assert v.has_strong_consensus(threshold=0.6) is False

    # --- is_split_decision ---

    def test_split_decision_when_both_buy_and_sell_present(self):
        ops = [_make_opinion("BUY"), _make_opinion("SELL"), _make_opinion("HOLD")]
        v = self._verdict(ops)
        assert v.is_split_decision() is True

    def test_not_split_when_only_buy_and_hold(self):
        ops = [_make_opinion("BUY"), _make_opinion("HOLD")]
        v = self._verdict(ops)
        assert v.is_split_decision() is False

    def test_not_split_when_only_sell_and_hold(self):
        ops = [_make_opinion("SELL"), _make_opinion("HOLD")]
        v = self._verdict(ops)
        assert v.is_split_decision() is False

    def test_not_split_when_unanimous(self):
        ops = [_make_opinion("BUY"), _make_opinion("BUY"), _make_opinion("BUY")]
        v = self._verdict(ops)
        assert v.is_split_decision() is False

    # --- vote_distribution ---

    def test_vote_distribution_counts_each_recommendation(self):
        ops = [
            _make_opinion("BUY"),
            _make_opinion("BUY"),
            _make_opinion("SELL"),
            _make_opinion("HOLD"),
        ]
        v = self._verdict(ops)
        dist = v.vote_distribution()
        assert dist == {"BUY": 2, "SELL": 1, "HOLD": 1}

    def test_vote_distribution_missing_recommendations_are_zero(self):
        ops = [_make_opinion("BUY"), _make_opinion("BUY")]
        v = self._verdict(ops)
        # Wszystkie trzy klucze zawsze obecne — łatwiejszy template
        dist = v.vote_distribution()
        assert dist == {"BUY": 2, "SELL": 0, "HOLD": 0}

    def test_vote_distribution_empty_opinions(self):
        v = self._verdict([])
        assert v.vote_distribution() == {"BUY": 0, "SELL": 0, "HOLD": 0}

    # --- dissent_ratio ---

    def test_dissent_ratio_fraction_disagreeing_with_final(self):
        # 2 z 4 nie zgadzają się z finalnym BUY → 0.5
        ops = [
            _make_opinion("BUY"),
            _make_opinion("BUY"),
            _make_opinion("SELL"),
            _make_opinion("HOLD"),
        ]
        v = self._verdict(ops, final="BUY")
        assert v.dissent_ratio() == 0.5

    def test_dissent_ratio_zero_when_unanimous(self):
        ops = [_make_opinion("BUY") for _ in range(5)]
        v = self._verdict(ops, final="BUY")
        assert v.dissent_ratio() == 0.0

    def test_dissent_ratio_zero_when_no_opinions(self):
        v = self._verdict([], final="BUY")
        assert v.dissent_ratio() == 0.0


class TestInvestorOpinionBehavior:
    def test_confidence_label_high(self):
        op = _make_opinion(confidence=0.85)
        assert op.confidence_label() == "HIGH"

    def test_confidence_label_medium(self):
        op = _make_opinion(confidence=0.6)
        assert op.confidence_label() == "MEDIUM"

    def test_confidence_label_low(self):
        op = _make_opinion(confidence=0.3)
        assert op.confidence_label() == "LOW"

    def test_confidence_label_boundary_high(self):
        # ≥ 0.75 = HIGH
        assert _make_opinion(confidence=0.75).confidence_label() == "HIGH"
        assert _make_opinion(confidence=0.749).confidence_label() == "MEDIUM"

    def test_confidence_label_boundary_low(self):
        # < 0.5 = LOW
        assert _make_opinion(confidence=0.5).confidence_label() == "MEDIUM"
        assert _make_opinion(confidence=0.4999).confidence_label() == "LOW"


class TestDeriveConsensus:
    """Czysta funkcja domenowa: rekomendacja + siła konsensusu liczone z
    realnych głosów inwestorów (confidence-weighted), nie z liczby od chairman."""

    def test_unanimous_buy_high_strength(self):
        ops = [_make_opinion("BUY", confidence=0.8) for _ in range(5)]
        rec, strength = derive_consensus(ops)
        assert rec == "BUY"
        # Jednogłośne BUY → cała "masa" pewności na zwycięskim koszyku → 1.0
        assert strength == 1.0

    def test_three_buy_one_sell_three_hold_deterministic(self):
        # 3×BUY, 1×SELL, 3×HOLD — wszystkie confidence=0.8.
        # Suma confidence = 7 * 0.8 = 5.6; zwycięzca to remis BUY/HOLD po 3.
        # Tie-break preferuje BUY > SELL > HOLD → BUY wygrywa.
        # strength = (3 * 0.8) / (7 * 0.8) = 2.4 / 5.6 = 3/7.
        ops = (
            [_make_opinion("BUY", confidence=0.8) for _ in range(3)]
            + [_make_opinion("SELL", confidence=0.8)]
            + [_make_opinion("HOLD", confidence=0.8) for _ in range(3)]
        )
        rec, strength = derive_consensus(ops)
        assert rec == "BUY"
        assert strength == pytest.approx(3 / 7)

    def test_confidence_weighted_winner_not_headcount(self):
        # 1×BUY z confidence 0.9 vs 2×SELL z confidence 0.1 każdy.
        # Headcount mówi SELL (2 głosy), ale waga pewności mówi BUY
        # (0.9 vs 0.2). Funkcja musi ważyć pewnością.
        ops = [
            _make_opinion("BUY", confidence=0.9),
            _make_opinion("SELL", confidence=0.1),
            _make_opinion("SELL", confidence=0.1),
        ]
        rec, strength = derive_consensus(ops)
        assert rec == "BUY"
        assert strength == pytest.approx(0.9 / 1.1)

    def test_empty_opinions_safe_default(self):
        rec, strength = derive_consensus([])
        assert rec == "HOLD"
        assert strength == 0.0

    def test_all_zero_confidence_falls_back_to_headcount(self):
        # Gdy wszyscy mają confidence 0.0, waga jest bezużyteczna — wracamy
        # do liczenia głosów, żeby nie dzielić przez zero.
        ops = [
            _make_opinion("SELL", confidence=0.0),
            _make_opinion("SELL", confidence=0.0),
            _make_opinion("BUY", confidence=0.0),
        ]
        rec, strength = derive_consensus(ops)
        assert rec == "SELL"
        assert strength == pytest.approx(2 / 3)

    def test_strength_in_unit_interval(self):
        ops = [
            _make_opinion("BUY", confidence=0.7),
            _make_opinion("SELL", confidence=0.6),
            _make_opinion("HOLD", confidence=0.55),
        ]
        _, strength = derive_consensus(ops)
        assert 0.0 <= strength <= 1.0


class TestCouncilInput:
    def test_stores_all_fields(self):
        data = CouncilInput(
            symbol="AAPL",
            current_price=Decimal("180.00"),
            price_delta_pct=Decimal("3.5"),
            sentiment_score=0.6,
            news_articles=["Apple beats earnings", "iPhone sales surge"],
            llm_trend="BULLISH",
            llm_confidence=0.82,
            ml_price_target=Decimal("185.00"),
        )
        assert data.symbol == "AAPL"
        assert data.ml_price_target == Decimal("185.00")
