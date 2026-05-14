from unittest.mock import Mock

import pytest
import requests

from src.application.ports import NewsPort
from src.infrastructure.adapters.news_api import NewsApiAdapter


@pytest.fixture
def adapter() -> NewsApiAdapter:
    return NewsApiAdapter(api_key="news-key-abc")


def _ok_response(payload: dict) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


_SAMPLE_RESPONSE = {
    "status": "ok",
    "totalResults": 2,
    "articles": [
        {
            "source": {"id": "reuters", "name": "Reuters"},
            "author": "Jane Doe",
            "title": "Nvidia faces new regulatory hurdles in Europe",
            "description": "European Commission starts inquiry...",
            "url": "https://reuters.com/...",
            "publishedAt": "2026-05-13T14:20:00Z",
        },
        {
            "source": {"id": "bloomberg", "name": "Bloomberg"},
            "title": "AAPL beats expectations on services revenue",
            "description": "Apple Q2 earnings...",
            "url": "https://bloomberg.com/...",
            "publishedAt": "2026-05-13T12:05:00Z",
        },
    ],
}


class TestGetNewsContext:
    def test_returns_normalized_list_of_articles(self, adapter, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_RESPONSE),
        )

        articles = adapter.get_news_context("AAPL")

        assert isinstance(articles, list)
        assert len(articles) == 2
        first = articles[0]
        assert first["title"] == "Nvidia faces new regulatory hurdles in Europe"
        assert first["source"] == "Reuters"
        assert first["description"] == "European Commission starts inquiry..."
        assert first["published_at"] == "2026-05-13T14:20:00Z"
        assert first["url"] == "https://reuters.com/..."

    def test_passes_symbol_apikey_and_pagination(self, adapter, mocker):
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_RESPONSE),
        )

        adapter.get_news_context("AAPL")

        params = mock_get.call_args.kwargs["params"]
        assert params["q"] == "AAPL"
        assert params["apiKey"] == "news-key-abc"
        assert params["sortBy"] == "publishedAt"
        assert int(params["pageSize"]) >= 1

    def test_sets_request_timeout(self, adapter, mocker):
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_RESPONSE),
        )

        adapter.get_news_context("AAPL")

        assert mock_get.call_args.kwargs["timeout"] > 0

    def test_raises_on_http_error(self, adapter, mocker):
        response = Mock(spec=requests.Response)
        response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
        mocker.patch(
            "requests.Session.get",
            return_value=response,
        )

        with pytest.raises(requests.HTTPError):
            adapter.get_news_context("AAPL")

    def test_returns_empty_list_when_no_articles(self, adapter, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"status": "ok", "articles": []}),
        )

        articles = adapter.get_news_context("UNKNOWN")

        assert articles == []

    def test_handles_article_with_missing_source_name(self, adapter, mocker):
        # Defensywne dekodowanie — niektóre artykuły mogą mieć None w source.name
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({
                "status": "ok",
                "articles": [{"source": {}, "title": "T", "publishedAt": "x"}],
            }),
        )

        articles = adapter.get_news_context("AAPL")
        assert articles[0]["source"] is None


class TestAdapterImplementsPort:
    def test_is_news_port(self):
        assert issubclass(NewsApiAdapter, NewsPort)


@pytest.mark.integration
class TestNewsApiLive:
    def test_fetches_real_articles(self):
        import os

        api_key = os.environ.get("NEWS_API_KEY")
        if not api_key:
            pytest.skip("NEWS_API_KEY not set")

        adapter = NewsApiAdapter(api_key=api_key)
        articles = adapter.get_news_context("AAPL")
        assert isinstance(articles, list)
