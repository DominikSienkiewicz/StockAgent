from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.agent_graph import create_agent_graph
from src.application.ports import (
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.domain.prediction import Prediction, TrendDirection
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
def workflow(
    market_port: Mock,
    sentiment_port: Mock,
    news_port: Mock,
    repository_port: Mock,
    ml_port: Mock,
    llm_port: Mock,
):
    return create_agent_graph(
        market_port=market_port,
        sentiment_port=sentiment_port,
        news_port=news_port,
        repository_port=repository_port,
        ml_port=ml_port,
        llm_port=llm_port,
        threshold=Threshold(Decimal("0.02")),
    )


def _initial_state(previous_price: str) -> dict:
    return {
        "symbol": "AAPL",
        "previous_price": Decimal(previous_price),
    }


def _setup_full_analysis_mocks(
    sentiment_port: Mock,
    news_port: Mock,
    repository_port: Mock,
    ml_port: Mock,
    llm_port: Mock,
    has_prior_prediction: bool = False,
) -> None:
    """Boilerplate — komplet mocków dla pełnej ścieżki analizy."""
    sentiment_port.get_social_score.return_value = {"galaxy_score": 85, "sentiment": "bearish"}
    news_port.get_news_context.return_value = [
        {"title": "Reuters: Fed keeps rates", "source": "Reuters"}
    ]
    if has_prior_prediction:
        repository_port.get_unverified_prediction.return_value = Prediction(
            id="prev-uuid-456",
            symbol="AAPL",
            predicted_trend=TrendDirection.BULLISH,
            price_at_prediction=Decimal("100.0"),
            predicted_target_price=Decimal("105.0"),
        )
    else:
        repository_port.get_unverified_prediction.return_value = None
    repository_port.save_prediction.return_value = "pred-uuid-123"
    llm_port.analyze.return_value = {
        "trend_direction": "BEARISH",
        "confidence_score": 0.85,
        "target_price_12h": 88.0,
        "reasoning": "Strong negative sentiment + macro headwinds.",
    }
    llm_port.analyze_mistake.return_value = "Zignorowałem szerszy kontekst makro."
    ml_port.predict.return_value = Money(Decimal("89.5"))


# ---------------------------------------------------------------------------
# Path 1 — high volatility, no prior prediction (cold start)
# ---------------------------------------------------------------------------


class TestFullAnalysisColdStart:
    def test_runs_all_nodes_when_volatility_high_and_no_prior_prediction(
        self,
        workflow,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=False,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "saved"
        assert final["delta"] == Decimal("-0.10")
        assert final["prediction_id"] == "pred-uuid-123"
        assert final["llm_analysis"]["trend_direction"] == "BEARISH"
        assert final["ml_target_price"] == Decimal("89.5")

        # Kolejność wywołań
        market_port.get_current_price.assert_called_once_with("AAPL")
        repository_port.get_unverified_prediction.assert_called_once_with("AAPL")
        sentiment_port.get_social_score.assert_called_once_with("AAPL")
        news_port.get_news_context.assert_called_once_with("AAPL")
        llm_port.analyze.assert_called_once()
        ml_port.predict.assert_called_once()
        repository_port.save_prediction.assert_called_once()
        # Brak poprzedniej predykcji — nie diagnozujemy błędu
        llm_port.analyze_mistake.assert_not_called()
        repository_port.update_prediction_accuracy.assert_not_called()


# ---------------------------------------------------------------------------
# Path 2 — high volatility, prior prediction was WRONG → mistake diagnosis
# ---------------------------------------------------------------------------


class TestSelfReflectionOnWrongPrediction:
    def test_diagnoses_mistake_when_prior_prediction_was_wrong(
        self,
        workflow,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        # Poprzednio: BULLISH @ 100, target 105. Aktualnie: 90 (spadek) → zły kierunek.
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=True,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        # LLM diagnozuje błąd
        llm_port.analyze_mistake.assert_called_once()
        # Repository dostaje correction insight
        repository_port.update_prediction_accuracy.assert_called_once()
        args = repository_port.update_prediction_accuracy.call_args
        assert args.kwargs.get("insight") == "Zignorowałem szerszy kontekst makro." or \
               args.args[2] == "Zignorowałem szerszy kontekst makro."

        # reflection_context trafia do stanu i zostaje użyty w predykcji
        assert "Zignorowałem szerszy kontekst makro." in final["reflection_context"]
        assert final["status"] == "saved"


# ---------------------------------------------------------------------------
# Path 3 — high volatility, prior prediction CORRECT → reinforcement
# ---------------------------------------------------------------------------


class TestSelfReflectionOnCorrectPrediction:
    def test_reinforces_when_prior_prediction_was_correct(
        self,
        workflow,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        # Poprzednio: BULLISH @ 100. Aktualnie: 110 (wzrost) → trafiona, kierunek OK.
        market_port.get_current_price.return_value = Money(Decimal("110.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=True,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        # Nie wzywamy LLM-a do diagnozy błędu
        llm_port.analyze_mistake.assert_not_called()
        # Repository dostaje pozytywny feedback
        repository_port.update_prediction_accuracy.assert_called_once()
        assert "trafn" in final["reflection_context"].lower()
        assert final["status"] == "saved"


# ---------------------------------------------------------------------------
# Path 4 — low volatility → ignore early (FinOps: ZERO płatnych wywołań)
# ---------------------------------------------------------------------------


class TestVolatilityBelowThreshold:
    def test_ignores_minor_changes_and_skips_all_paid_ports(
        self,
        workflow,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        market_port.get_current_price.return_value = Money(Decimal("99.0"))  # -1%

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "ignored"
        assert final["delta"] == Decimal("-0.01")
        # FINOPS: żadne płatne API nie wolno wywołać
        sentiment_port.get_social_score.assert_not_called()
        news_port.get_news_context.assert_not_called()
        llm_port.analyze.assert_not_called()
        llm_port.analyze_mistake.assert_not_called()
        ml_port.predict.assert_not_called()
        repository_port.save_prediction.assert_not_called()
        repository_port.get_unverified_prediction.assert_not_called()


# ---------------------------------------------------------------------------
# Threshold injection
# ---------------------------------------------------------------------------


class TestCustomThreshold:
    def test_respects_injected_threshold(
        self,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        workflow = create_agent_graph(
            market_port=market_port,
            sentiment_port=sentiment_port,
            news_port=news_port,
            repository_port=repository_port,
            ml_port=ml_port,
            llm_port=llm_port,
            threshold=Threshold(Decimal("0.10")),  # 10%
        )
        market_port.get_current_price.return_value = Money(Decimal("95.0"))  # -5%

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "ignored"
        sentiment_port.get_social_score.assert_not_called()
