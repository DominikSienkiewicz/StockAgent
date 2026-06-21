"""Adapter Reddit — prędkość społeczna z publicznego search.json (bez OAuth).

Reddit udostępnia wyszukiwarkę jako JSON pod
    https://www.reddit.com/r/{subreddits}/search.json
bez logowania i bez klucza API. Wymaga jednak własnego User-Agenta —
domyślne UA bibliotek HTTP są blokowane (429/403).

Adapter robi DWA zapytania na symbol:
  * t=day  → liczba postów z ostatniej doby = `mentions_24h`,
  * t=week → liczba postów z tygodnia, baseline = week / 7.0.
Search nie zwraca sentymentu, więc `avg_sentiment` zawsze None.

Cienki I/O — klasyfikacja trendu siedzi w domenie
(`SocialVelocitySnapshot.trend`). To opcjonalne źródło alpha: każdy błąd
(non-200, brak pól, parse, sieć) kończy się `None`, nigdy wyjątkiem.
"""

from __future__ import annotations

import logging

import requests

from src.application.ports import SocialVelocityPort
from src.domain.social_velocity import SocialVelocitySnapshot
from src.infrastructure.adapters._http import build_session

DEFAULT_SUBREDDITS = "wallstreetbets+stocks+investing"
DEFAULT_USER_AGENT = "StockAgent/1.0"
DEFAULT_TIMEOUT = 10.0
DEFAULT_LIMIT = 100
SEARCH_URL_TEMPLATE = "https://www.reddit.com/r/{subreddits}/search.json"

logger = logging.getLogger(__name__)


class RedditSocialAdapter(SocialVelocityPort):
    """Liczy wzmianki symbolu w wybranych subredditach (24h vs tydzień)."""

    def __init__(
        self,
        subreddits: str = DEFAULT_SUBREDDITS,
        user_agent: str = DEFAULT_USER_AGENT,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._subreddits = subreddits
        self._user_agent = user_agent
        self._timeout = timeout
        self._session = session if session is not None else build_session()

    def get_social_velocity(self, symbol: str) -> SocialVelocitySnapshot | None:
        day_count = self._count_posts(symbol, period="day")
        if day_count is None:
            return None
        week_count = self._count_posts(symbol, period="week")
        if week_count is None:
            return None

        baseline = week_count / 7.0
        return SocialVelocitySnapshot(
            symbol=symbol,
            mentions_24h=day_count,
            baseline_mentions=baseline,
            avg_sentiment=None,
        )

    def _count_posts(self, symbol: str, period: str) -> int | None:
        """Zwraca liczbę postów dla danego okna czasowego lub None na błędzie."""
        url = SEARCH_URL_TEMPLATE.format(subreddits=self._subreddits)
        params: dict[str, str | int] = {
            "q": symbol,
            "restrict_sr": 1,
            "sort": "new",
            "t": period,
            "limit": DEFAULT_LIMIT,
        }
        try:
            response = self._session.get(
                url,
                params=params,
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload["data"]
            children = data["children"]
            if not isinstance(children, list):
                return None
            return len(children)
        except (requests.RequestException, ValueError, KeyError, TypeError):
            logger.exception("Reddit social fetch failed for %s (t=%s)", symbol, period)
            return None


class NullSocialVelocityAdapter(SocialVelocityPort):
    """Domyślny no-op (Fast-Loop / brak źródła): zawsze None, zero I/O."""

    def get_social_velocity(self, symbol: str) -> SocialVelocitySnapshot | None:
        return None
