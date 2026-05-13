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
    return Mock(spec=RepositoryPort)


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
