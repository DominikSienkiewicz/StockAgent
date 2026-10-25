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
    def test_implements_news_port(self) -> None:
        assert issubclass(AlphaVantageNewsAdapter, NewsPort)

    def test_delegates_to_client_articles_for(self, client: MagicMock) -> None:
        client.articles_for.return_value = [{"title": "headline"}]
        adapter = AlphaVantageNewsAdapter(client)

        result = adapter.get_news_context("AAPL")

        assert result == [{"title": "headline"}]
        client.articles_for.assert_called_once_with("AAPL")


class TestSentimentAdapter:
    def test_implements_sentiment_port(self) -> None:
        assert issubclass(AlphaVantageSentimentAdapter, SentimentPort)

    def test_delegates_to_client_sentiment_for(self, client: MagicMock) -> None:
        client.sentiment_for.return_value = {"av_sentiment_score": 0.5}
        client.degraded_reason = None
        adapter = AlphaVantageSentimentAdapter(client)

        result = adapter.get_social_score("AAPL")

        assert result == {"av_sentiment_score": 0.5}
        client.sentiment_for.assert_called_once_with("AAPL")

    def test_propagates_degraded_reason_when_client_exhausted(self, client: MagicMock) -> None:
        # Gdy klient AV jest w stanie degradacji (np. wszystkie klucze
        # wyczerpane), sentiment payload niesie ze sobą sygnał `degraded_reason`,
        # który dalej wpada do data_quality_flags w predict_node.
        client.sentiment_for.return_value = {
            "av_sentiment_score": 0.0,
            "news_volume_24h": 0,
        }
        client.degraded_reason = "av_keys_exhausted"
        adapter = AlphaVantageSentimentAdapter(client)

        result = adapter.get_social_score("AAPL")

        assert result["degraded_reason"] == "av_keys_exhausted"

    def test_no_degraded_reason_key_when_client_healthy(self, client: MagicMock) -> None:
        # Brak degradacji = brak klucza w payloadzie (nie pusty string,
        # nie None — żeby graph mógł użyć `if "degraded_reason" in sentiment`).
        client.sentiment_for.return_value = {"av_sentiment_score": 0.4}
        client.degraded_reason = None
        adapter = AlphaVantageSentimentAdapter(client)

        result = adapter.get_social_score("AAPL")

        assert "degraded_reason" not in result


class TestPhaseTwoOnlyFeed:
    """EXP-27: w cyklu dwufazowym deps grafu dostają zaślepkę feedu — klient AV
    powstaje dopiero w fazie 2. Wywołanie zaślepki to błąd orkiestracji, więc
    rzuca zamiast udawać pusty feed (pusty feed wyglądałby jak spokojny dzień)."""

    def test_is_both_feed_ports(self) -> None:
        from src.infrastructure.adapters.alpha_vantage_adapters import PhaseTwoOnlyFeed

        feed = PhaseTwoOnlyFeed()
        assert isinstance(feed, SentimentPort)
        assert isinstance(feed, NewsPort)

    @pytest.mark.parametrize("method", ["get_social_score", "get_news_context"])
    def test_refuses_to_serve_a_feed(self, method) -> None:
        from src.infrastructure.adapters.alpha_vantage_adapters import PhaseTwoOnlyFeed

        with pytest.raises(RuntimeError, match="EXP-27"):
            getattr(PhaseTwoOnlyFeed(), method)("AAPL")
