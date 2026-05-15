from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.ports import (
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
from src.domain.value_objects import Money, Threshold


@pytest.fixture
def market_port() -> Mock:
    return Mock(spec=MarketDataPort)


@pytest.fixture
def sentiment_port() -> Mock:
    return Mock(spec=SentimentPort)


@pytest.fixture
def news_port() -> Mock:
    return Mock(spec=NewsPort)


@pytest.fixture
def repository_port() -> Mock:
    repo = Mock(spec=RepositoryPort)
    # reflect_node działa w każdym cyklu — domyślnie brak historii do oceny.
    repo.get_unverified_prediction.return_value = None
    return repo


@pytest.fixture
def ml_port() -> Mock:
    return Mock(spec=MLPredictionPort)


@pytest.fixture
def llm_port() -> Mock:
    return Mock(spec=LLMPort)


@pytest.fixture
def use_case(
    market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
) -> AnalyzeMarketUseCase:
    return AnalyzeMarketUseCase(
        market_port=market_port,
        sentiment_port=sentiment_port,
        news_port=news_port,
        repository_port=repository_port,
        ml_port=ml_port,
        llm_port=llm_port,
        threshold=Threshold(Decimal("0.02")),
    )


class TestRun:
    def test_fetches_previous_price_from_repository(
        self, use_case, market_port, repository_port
    ):
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("99.5"))  # -0.5%

        use_case.run("AAPL")

        repository_port.get_last_price.assert_called_once_with("AAPL")

    def test_skips_analysis_on_cold_start_no_previous_price(
        self, use_case, market_port, repository_port, sentiment_port
    ):
        # Brak historii — repo zwraca None
        repository_port.get_last_price.return_value = None
        market_port.get_current_price.return_value = Money(Decimal("100.0"))

        result = use_case.run("AAPL")

        # Cold start: nie ma do czego porównać → ignorujemy (delta=0)
        assert result["status"] == "ignored"
        sentiment_port.get_social_score.assert_not_called()

    def test_runs_full_graph_when_volatility_above_threshold(
        self,
        use_case,
        market_port,
        sentiment_port,
        news_port,
        repository_port,
        ml_port,
        llm_port,
    ):
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        repository_port.get_unverified_prediction.return_value = None
        sentiment_port.get_social_score.return_value = {"galaxy_score": 85}
        news_port.get_news_context.return_value = [{"title": "Reuters story"}]
        llm_port.analyze.return_value = {
            "trend_direction": "BEARISH",
            "confidence_score": 0.8,
            "target_price_12h": 88.0,
            "reasoning": "macro headwinds",
        }
        ml_port.predict.return_value = Money(Decimal("89.0"))
        repository_port.save_prediction.return_value = "new-uuid"

        result = use_case.run("AAPL")

        assert result["status"] == "saved"
        assert result["prediction_id"] == "new-uuid"
        assert result["delta"] == Decimal("-0.10")
        sentiment_port.get_social_score.assert_called_once_with("AAPL")
        news_port.get_news_context.assert_called_once_with("AAPL")

    def test_ignores_minor_change_below_threshold(
        self, use_case, market_port, repository_port, sentiment_port, ml_port
    ):
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("100.5"))  # +0.5%

        result = use_case.run("AAPL")

        assert result["status"] == "ignored"
        sentiment_port.get_social_score.assert_not_called()
        ml_port.predict.assert_not_called()


def test_analyze_market_passes_council_port_to_graph():
    """AnalyzeMarketUseCase forwards council_port — council_node gets invoked."""
    from decimal import Decimal
    from unittest.mock import Mock

    from src.application.ports import (
        AdvisoryCouncilPort,
        LLMPort,
        MarketDataPort,
        MLPredictionPort,
        NewsPort,
        RepositoryPort,
        SentimentPort,
    )
    from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
    from src.domain.council import CouncilVerdict, InvestorOpinion
    from src.domain.value_objects import Money, Threshold

    market = Mock(spec=MarketDataPort)
    market.get_current_price.return_value = Money(Decimal("180.00"))
    sentiment = Mock(spec=SentimentPort)
    sentiment.get_social_score.return_value = {
        "av_sentiment_score": 0.5, "av_sentiment_label": "Bullish",
        "av_relevance_avg": 0.7, "news_volume_24h": 10, "high_relevance_count": 3,
    }
    news = Mock(spec=NewsPort)
    news.get_news_context.return_value = [{"title": "Apple gains"}]
    repo = Mock(spec=RepositoryPort)
    repo.get_unverified_prediction.return_value = None
    repo.save_prediction.return_value = "pred-123"
    ml = Mock(spec=MLPredictionPort)
    ml.is_trained = True
    ml.predict.return_value = Money(Decimal("185.00"))
    llm = Mock(spec=LLMPort)
    llm.analyze.return_value = {
        "trend_direction": "BULLISH", "confidence_score": 0.8,
        "av_agreement": 0.9, "target_price_12h": 185.0, "reasoning": "ok",
    }
    council_port = Mock(spec=AdvisoryCouncilPort)
    verdict = CouncilVerdict(
        final_recommendation="BUY",
        consensus_strength=0.7,
        summary="Rada kupuje.",
        dissenting_views=[],
        investor_opinions=[
            InvestorOpinion(
                investor_name="Warren Buffett",
                recommendation="BUY",
                confidence=0.9,
                reasoning="moat",
                key_factors=["moat"],
            )
        ],
    )
    council_port.analyze.return_value = verdict

    use_case = AnalyzeMarketUseCase(
        market_port=market,
        sentiment_port=sentiment,
        news_port=news,
        repository_port=repo,
        ml_port=ml,
        llm_port=llm,
        threshold=Threshold(Decimal("0.02")),
        council_port=council_port,
    )
    # previous_price much lower to trigger volatility gate
    repo.get_last_price.return_value = Money(Decimal("170.00"))
    use_case.run("AAPL")
    council_port.analyze.assert_called_once()


class TestThresholdInjection:
    def test_respects_custom_threshold(
        self,
        market_port,
        sentiment_port,
        news_port,
        repository_port,
        ml_port,
        llm_port,
    ):
        use_case = AnalyzeMarketUseCase(
            market_port=market_port,
            sentiment_port=sentiment_port,
            news_port=news_port,
            repository_port=repository_port,
            ml_port=ml_port,
            llm_port=llm_port,
            threshold=Threshold(Decimal("0.20")),  # 20% próg
        )
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%

        result = use_case.run("AAPL")

        assert result["status"] == "ignored"
        sentiment_port.get_social_score.assert_not_called()
