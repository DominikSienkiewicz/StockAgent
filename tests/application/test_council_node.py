# tests/application/test_council_node.py
from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

from src.application.agent_graph import create_agent_graph
from src.application.ports import (
    AdvisoryCouncilPort,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.value_objects import Money, Threshold


def _make_verdict() -> CouncilVerdict:
    return CouncilVerdict(
        final_recommendation="BUY",
        consensus_strength=0.73,
        summary="Rada radzi kupować.",
        dissenting_views=["Soros: ryzykowne"],
        investor_opinions=[
            InvestorOpinion(
                investor_name="Warren Buffett",
                recommendation="BUY",
                confidence=0.9,
                reasoning="Silna fosa.",
                key_factors=["moat"],
            )
        ],
    )


def _make_graph(council_port: AdvisoryCouncilPort | None = None):
    market = Mock(spec=MarketDataPort)
    market.get_current_price.return_value = Money(Decimal("180.00"))
    sentiment = Mock(spec=SentimentPort)
    sentiment.get_social_score.return_value = {
        "av_sentiment_score": 0.5,
        "av_sentiment_label": "Bullish",
        "av_relevance_avg": 0.7,
        "news_volume_24h": 10,
        "high_relevance_count": 3,
    }
    news = Mock(spec=NewsPort)
    news.get_news_context.return_value = [{"title": "Apple gains", "source": "Reuters"}]
    repo = Mock(spec=RepositoryPort)
    repo.get_unverified_prediction.return_value = None
    repo.save_prediction.return_value = "pred-123"
    ml = Mock(spec=MLPredictionPort)
    ml.is_trained = True
    ml.predict.return_value = Money(Decimal("185.00"))
    llm = Mock(spec=LLMPort)
    llm.analyze.return_value = {
        "trend_direction": "BULLISH",
        "confidence_score": 0.8,
        "av_agreement": 0.9,
        "target_price_12h": 185.0,
        "reasoning": "Strong momentum.",
    }
    return create_agent_graph(
        market_port=market,
        sentiment_port=sentiment,
        news_port=news,
        repository_port=repo,
        ml_port=ml,
        llm_port=llm,
        threshold=Threshold(Decimal("0.02")),
        council_port=council_port,
    )


class TestCouncilNodeIntegration:
    def test_council_verdict_in_state_when_port_provided(self):
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        graph = _make_graph(council_port=council_port)
        app = graph.compile()
        result = app.invoke({"symbol": "AAPL", "previous_price": Decimal("173.00")})
        assert result.get("council_verdict") is not None
        assert result["council_verdict"].final_recommendation == "BUY"

    def test_council_called_with_symbol(self):
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        graph = _make_graph(council_port=council_port)
        app = graph.compile()
        app.invoke({"symbol": "AAPL", "previous_price": Decimal("173.00")})
        council_port.analyze.assert_called_once()
        args = council_port.analyze.call_args
        assert args[0][0] == "AAPL"

    def test_council_verdict_none_when_no_port(self):
        graph = _make_graph(council_port=None)
        app = graph.compile()
        result = app.invoke({"symbol": "AAPL", "previous_price": Decimal("173.00")})
        assert result.get("council_verdict") is None

    def test_council_not_called_when_volatility_gate_blocks(self):
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        graph = _make_graph(council_port=council_port)
        app = graph.compile()
        # previous_price == current_price → delta=0 → gate blocks → council_node not reached
        result = app.invoke({"symbol": "AAPL", "previous_price": Decimal("180.00")})
        council_port.analyze.assert_not_called()
        assert result.get("council_verdict") is None
