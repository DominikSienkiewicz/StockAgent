"""Wspólne okablowanie grafu Fast Loopa dla testów warstwy application (TEST-33).

`graph_ports` zbiera sześć wymaganych portów `AgentGraphDeps` z fixture'ów
`market_port` ... `llm_port`, a `make_deps` składa z nich zależności grafu.

Fixture'ów `market_port` ... `llm_port` tu NIE definiujemy: każdy moduł testów ma
własne, z własnymi domyślnymi odpowiedziami (np. `repository_port` bez historii),
a `graph_ports` rozwiązuje je po nazwie w module testu, więc niczego nie przesłania.
"""

from decimal import Decimal
from typing import Any, TypedDict
from unittest.mock import Mock

import pytest

from src.application.agent_graph import AgentGraphDeps
from src.application.ports import (
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.domain.value_objects import Threshold


class GraphPorts(TypedDict):
    """Sześć portów, które `AgentGraphDeps` wymaga zawsze (pozostałe są opcjonalne)."""

    market_port: Mock
    sentiment_port: Mock
    news_port: Mock
    repository_port: Mock
    ml_port: Mock
    llm_port: Mock


@pytest.fixture
def graph_ports(
    market_port: Mock,
    sentiment_port: Mock,
    news_port: Mock,
    repository_port: Mock,
    ml_port: Mock,
    llm_port: Mock,
) -> GraphPorts:
    """Te same egzemplarze, które test dostaje przez fixture'y `market_port` ... `llm_port`."""
    return GraphPorts(
        market_port=market_port,
        sentiment_port=sentiment_port,
        news_port=news_port,
        repository_port=repository_port,
        ml_port=ml_port,
        llm_port=llm_port,
    )


def make_deps(**overrides: Any) -> AgentGraphDeps:
    """`AgentGraphDeps` z domyślnym progiem 2% i atrapami `Mock(spec=Port)`.

    Każde pole podane w `overrides` zastępuje domyślne; brakujący z sześciu
    wymaganych portów dostaje świeżą atrapę. Zwykle wołane jako
    `make_deps(**graph_ports, <pola pod test>)`.
    """
    specs: dict[str, type] = {
        "market_port": MarketDataPort,
        "sentiment_port": SentimentPort,
        "news_port": NewsPort,
        "repository_port": RepositoryPort,
        "ml_port": MLPredictionPort,
        "llm_port": LLMPort,
    }
    fields: dict[str, Any] = {
        name: Mock(spec=port) for name, port in specs.items() if name not in overrides
    }
    fields.setdefault("threshold", Threshold(Decimal("0.02")))
    fields.update(overrides)
    return AgentGraphDeps(**fields)

