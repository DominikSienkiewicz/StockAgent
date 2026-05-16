"""Adaptery News + Sentiment oparte o wspólny `AlphaVantageClient`.

Oba implementują kontrakty z `application.ports`, dzięki czemu graf LangGraph
nie wie, że pod spodem chodzi jeden batch call.
"""

from __future__ import annotations

from typing import Any

from src.application.ports import NewsPort, SentimentPort
from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient


class AlphaVantageNewsAdapter(NewsPort):
    """Implementacja `NewsPort` — filtrowane po relevance ≥ threshold artykuły."""

    def __init__(self, client: AlphaVantageClient) -> None:
        self._client = client

    def get_news_context(self, symbol: str) -> list[dict[str, Any]]:
        return self._client.articles_for(symbol)


class AlphaVantageSentimentAdapter(SentimentPort):
    """Implementacja `SentimentPort` — multi-feature dict z relevance-weighted sentyment.

    Zwracane pola (kontrakt z `predict_node`):
      - av_sentiment_score:   float [-1.0..1.0] — weighted avg ticker sentiment
      - av_relevance_avg:     float [0..1]      — średnia relevance artykułów
      - news_volume_24h:      int               — liczba relevantnych artykułów
      - high_relevance_count: int               — ile artykułów z relevance ≥ 0.8
      - av_sentiment_label:   str               — kategoria (Bullish/Bearish/...)

    Graf używa tych pól bezpośrednio (po przejściu z LunarCrush na AV NEWS_SENTIMENT).
    """

    def __init__(self, client: AlphaVantageClient) -> None:
        self._client = client

    def get_social_score(self, symbol: str) -> dict[str, Any]:
        return self._client.sentiment_for(symbol)
