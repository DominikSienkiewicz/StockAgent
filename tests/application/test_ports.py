import pytest

from src.application.ports import (
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)

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
