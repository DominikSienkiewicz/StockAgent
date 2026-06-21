from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from src.application.agent_graph import AgentState, create_agent_graph
from src.application.ports import (
    AdvisoryCouncilPort,
    EmbeddingPort,
    FundamentalsPort,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
    Tool,
    ToolUseLLMPort,
)
from src.domain.asset import Asset
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
        embedding_port: EmbeddingPort | None = None,
        council_port: AdvisoryCouncilPort | None = None,
        council_threshold: Threshold | None = None,
        fundamentals_port: FundamentalsPort | None = None,
        crypto_threshold: Threshold | None = None,
        crypto_council_threshold: Threshold | None = None,
        reflection_min_age_hours: int = 0,
        rag_outcome_weight: float = 0.0,
        tool_use_port: ToolUseLLMPort | None = None,
        research_tools: Sequence[Tool] = (),
        tool_use_threshold: Threshold | None = None,
        vector_memory_enabled: bool = False,
    ) -> None:
        self._repository = repository_port
        workflow = create_agent_graph(
            market_port=market_port,
            sentiment_port=sentiment_port,
            news_port=news_port,
            repository_port=repository_port,
            ml_port=ml_port,
            llm_port=llm_port,
            threshold=threshold,
            embedding_port=embedding_port,
            council_port=council_port,
            council_threshold=council_threshold,
            fundamentals_port=fundamentals_port,
            crypto_threshold=crypto_threshold,
            crypto_council_threshold=crypto_council_threshold,
            reflection_min_age_hours=reflection_min_age_hours,
            rag_outcome_weight=rag_outcome_weight,
            tool_use_port=tool_use_port,
            research_tools=research_tools,
            tool_use_threshold=tool_use_threshold,
            vector_memory_enabled=vector_memory_enabled,
        )
        # Kompilacja jest deterministyczna (zależy tylko od topologii + portów),
        # więc kompilujemy RAZ tutaj i reużywamy aplikację w każdym run().
        # Inaczej przy 43 symbolach na cykl rekompilowalibyśmy graf 43×.
        self._app = workflow.compile()

    def run(
        self,
        symbol: str,
        asset: Asset | None = None,
        *,
        regime_context: str = "",
        regime_multiplier: float = 1.0,
        peer_context: tuple[tuple[str, str], ...] = (),
    ) -> dict[str, Any]:
        previous = self._repository.get_last_price(symbol)

        # Cold start: brak historii → previous=0 → delta=0 (guard w domenie) → ignore.
        initial_state: AgentState = {
            "symbol": symbol,
            "previous_price": previous.amount if previous else Decimal("0"),
            # #7 reżim (label + mnożnik progu) i #5 contagion (peery z tego cyklu).
            "regime_context": regime_context,
            "regime_multiplier": regime_multiplier,
            "peer_context": peer_context,
        }
        # Przekazujemy asset z klasyfikacją (STOCK/ETF), gdy dostępny z zewnątrz.
        if asset is not None:
            initial_state["asset"] = asset

        result: dict[str, Any] = self._app.invoke(initial_state)
        return result
