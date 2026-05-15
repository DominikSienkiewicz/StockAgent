import inspect
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.ports import (
    AdvisoryCouncilPort,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.domain.council import CouncilInput

ALL_PORTS = [
    MarketDataPort,
    SentimentPort,
    NewsPort,
    RepositoryPort,
    MLPredictionPort,
    LLMPort,
]


@pytest.mark.parametrize("port_cls", ALL_PORTS)
def test_port_cannot_be_instantiated_directly(port_cls):
    with pytest.raises(TypeError, match="abstract"):
        port_cls()  # type: ignore[abstract]


@pytest.mark.parametrize("port_cls", ALL_PORTS)
def test_port_has_at_least_one_abstract_method(port_cls):
    assert len(port_cls.__abstractmethods__) >= 1


def test_advisory_council_port_is_abstract():
    assert inspect.isabstract(AdvisoryCouncilPort)


def test_advisory_council_port_can_be_mocked():
    mock = Mock(spec=AdvisoryCouncilPort)
    data = CouncilInput(
        symbol="AAPL",
        current_price=Decimal("180.00"),
        price_delta_pct=Decimal("3.5"),
        sentiment_score=0.6,
        news_articles=["headline"],
        llm_trend="BULLISH",
        llm_confidence=0.8,
        ml_price_target=Decimal("185.00"),
    )
    mock.analyze("AAPL", data)
    mock.analyze.assert_called_once()
