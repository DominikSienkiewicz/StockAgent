from __future__ import annotations

from typing import Any

import requests

from src.application.ports import SentimentPort

DEFAULT_BASE_URL = "https://lunarcrush.com/api4/public"
DEFAULT_TIMEOUT = 10


class LunarCrushAdapter(SentimentPort):
    """Adapter dla LunarCrush API v4 (Social Intelligence).

    Endpoint: GET /stocks/{symbol}/v1 — szczegóły aktywa giełdowego.
    Auth: Bearer token w nagłówku Authorization.

    Uwaga: free tier LunarCrush nie ma dostępu do tych endpointów —
    wymagana płatna subskrypcja (Individual lub wyższa, ~24 USD/msc).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def get_social_score(self, symbol: str) -> dict[str, Any]:
        response = requests.get(
            f"{self._base_url}/stocks/{symbol}/v1",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        response.raise_for_status()

        data = response.json().get("data", {})
        return {
            "symbol": data.get("symbol", symbol),
            "galaxy_score": data.get("galaxy_score"),
            "social_volume": data.get("social_volume"),
            "average_sentiment": data.get("average_sentiment"),
            "social_score": data.get("social_score"),
        }
