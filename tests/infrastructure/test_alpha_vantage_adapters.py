from unittest.mock import MagicMock

import pytest

from src.application.ports import NewsPort, SentimentPort
from src.infrastructure.adapters.alpha_vantage_adapters import (
    AlphaVantageNewsAdapter,
    AlphaVantageSentimentAdapter,
)
from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient


@pytest.fixture
def client() -> MagicMock:
    return MagicMock(spec=AlphaVantageClient)


class TestNewsAdapter:
    def test_implements_news_port(self):
        assert issubclass(AlphaVantageNewsAdapter, NewsPort)

    def test_delegates_to_client_articles_for(self, client):
        client.articles_for.return_value = [{"title": "headline"}]
        adapter = AlphaVantageNewsAdapter(client)

        result = adapter.get_news_context("AAPL")

        assert result == [{"title": "headline"}]
        client.articles_for.assert_called_once_with("AAPL")


class TestSentimentAdapter:
    def test_implements_sentiment_port(self):
        assert issubclass(AlphaVantageSentimentAdapter, SentimentPort)

    def test_delegates_to_client_sentiment_for(self, client):
        client.sentiment_for.return_value = {"av_sentiment_score": 0.5}
        adapter = AlphaVantageSentimentAdapter(client)

        result = adapter.get_social_score("AAPL")

        assert result == {"av_sentiment_score": 0.5}
        client.sentiment_for.assert_called_once_with("AAPL")
