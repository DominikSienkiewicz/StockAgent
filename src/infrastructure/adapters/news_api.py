from __future__ import annotations

from typing import Any

from src.application.ports import NewsPort
from src.infrastructure.adapters._http import build_session

DEFAULT_BASE_URL = "https://newsapi.org/v2"
DEFAULT_TIMEOUT = 10
DEFAULT_PAGE_SIZE = 10


class NewsApiAdapter(NewsPort):
    """Adapter dla NewsAPI.org (everything endpoint).

    Endpoint: GET /everything?q={symbol}&apiKey={key}&sortBy=publishedAt
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._page_size = page_size
        self._session = build_session()

    def get_news_context(self, symbol: str) -> list[dict[str, Any]]:
        response = self._session.get(
            f"{self._base_url}/everything",
            params={
                "q": symbol,
                "apiKey": self._api_key,
                "sortBy": "publishedAt",
                "pageSize": str(self._page_size),
                "language": "en",
            },
            timeout=self._timeout,
        )
        response.raise_for_status()

        raw_articles = response.json().get("articles", [])
        return [self._normalize(article) for article in raw_articles]

    @staticmethod
    def _normalize(article: dict[str, Any]) -> dict[str, Any]:
        source = article.get("source") or {}
        return {
            "title": article.get("title"),
            "source": source.get("name"),
            "description": article.get("description"),
            "url": article.get("url"),
            "published_at": article.get("publishedAt"),
        }
