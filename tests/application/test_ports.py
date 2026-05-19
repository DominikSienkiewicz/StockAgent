import inspect
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.ports import (
    AdvisoryCouncilPort,
    FundamentalsPort,
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
def test_port_cannot_be_instantiated_directly(port_cls: type) -> None:
    with pytest.raises(TypeError, match="abstract"):
        port_cls()


@pytest.mark.parametrize("port_cls", ALL_PORTS)
def test_port_has_at_least_one_abstract_method(port_cls: type) -> None:
    assert len(port_cls.__abstractmethods__) >= 1  # type: ignore[attr-defined]


def test_advisory_council_port_is_abstract() -> None:
    assert inspect.isabstract(AdvisoryCouncilPort)


def test_advisory_council_port_can_be_mocked() -> None:
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


def test_fundamentals_port_has_get_fundamentals() -> None:
    method = FundamentalsPort.get_fundamentals
    sig = inspect.signature(method)
    assert "symbol" in sig.parameters


def test_repository_port_has_fundamentals_methods() -> None:
    assert hasattr(RepositoryPort, "get_cached_fundamentals")
    assert hasattr(RepositoryPort, "save_fundamentals")


def test_repository_port_has_save_council_votes() -> None:
    # Strukturalny audit trail rady doradczej — osobna tabela na głosy
    # umożliwia odpytywanie "jak Burry głosował na NVDA w ostatnim miesiącu"
    # bez parsowania JSONB blob z prediction_logs.council_verdict.
    assert hasattr(RepositoryPort, "save_council_votes")
    sig = inspect.signature(RepositoryPort.save_council_votes)
    assert "prediction_id" in sig.parameters
    assert "votes" in sig.parameters
