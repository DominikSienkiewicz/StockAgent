# tests/domain/test_council.py
from decimal import Decimal

from src.domain.council import CouncilInput, CouncilVerdict, InvestorOpinion


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
