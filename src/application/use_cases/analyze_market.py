from __future__ import annotations

from decimal import Decimal
from typing import Any

from src.application.agent_graph import AgentState, create_agent_graph
from src.application.ports import (
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.domain.value_objects import Threshold


class AnalyzeMarketUseCase:
    """Fast Loop — pełen cykl decyzyjny dla pojedynczego symbolu.

    Spina:
    1. odczyt poprzedniej ceny z repozytorium,
    2. uruchomienie grafu LangGraph (check_price → reflect/sentiment/news/predict/save).
    """

    def __init__(
        self,
        *,
        market_port: MarketDataPort,
        sentiment_port: SentimentPort,
        news_port: NewsPort,
        repository_port: RepositoryPort,
        ml_port: MLPredictionPort,
        llm_port: LLMPort,
        threshold: Threshold,
    ) -> None:
        self._repository = repository_port
        self._workflow = create_agent_graph(
            market_port=market_port,
            sentiment_port=sentiment_port,
            news_port=news_port,
            repository_port=repository_port,
            ml_port=ml_port,
            llm_port=llm_port,
            threshold=threshold,
        )

    def run(self, symbol: str) -> dict[str, Any]:
        previous = self._repository.get_last_price(symbol)

        # Cold start: brak historii → previous=0 → delta=0 (guard w domenie) → ignore.
        initial_state: AgentState = {
            "symbol": symbol,
            "previous_price": previous.amount if previous else Decimal("0"),
        }

        app = self._workflow.compile()
        result: dict[str, Any] = app.invoke(initial_state)
        return result
