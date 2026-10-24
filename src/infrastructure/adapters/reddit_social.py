"""Adapter Reddit — prędkość społeczna z publicznego search.json (bez OAuth).

Adapter woła wyszukiwarkę JSON pod
    https://www.reddit.com/r/{subreddits}/search.json
bez logowania i bez klucza API, z własnym User-Agentem. Od co najmniej
2026-07-27 Reddit odpowiada na takie żądania z runnerów GitHub Actions
`403 Blocked` (anonimowy JSON odcięty, oficjalna droga to OAuth), więc
produkcyjny config.toml trzyma `social_velocity_enabled = false` — patrz
docs/configuration.md. Pierwszy 403 kończy Reddit do końca cyklu.

Adapter robi DWA zapytania na symbol:
  * t=day  → liczba postów z ostatniej doby = `mentions_24h`,
  * t=week → liczba postów z tygodnia, baseline = week / 7.0.
Search nie zwraca sentymentu, więc `avg_sentiment` zawsze None.

Jedna strona wyników to najwyżej `DEFAULT_LIMIT` postów. Tydzień, który dobił do
limitu, jest ucięty — to sufit, nie liczba — i zaniżony baseline dawałby
fałszywe SURGING, więc wtedy zwracamy None (bez paginacji: +N requestów na symbol).

Cienki I/O — klasyfikacja trendu siedzi w domenie
(`SocialVelocitySnapshot.trend`). To opcjonalne źródło alpha: każdy błąd
(non-200, brak pól, parse, sieć) kończy się `None`, nigdy wyjątkiem.
"""

from __future__ import annotations

import logging
from http import HTTPStatus

import requests

from src.application.errors import PortError
from src.application.ports import SocialVelocityPort
from src.domain.social_velocity import SocialVelocitySnapshot
from src.infrastructure.adapters._http import (
    build_session,
    json_payload,
    raise_for_status_translated,
    translated_get,
)

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
        # Adapter żyje jeden cykl, a blokada Reddita w trakcie cyklu nie mija.
        self._blocked = False

    def get_social_velocity(self, symbol: str) -> SocialVelocitySnapshot | None:
        if self._blocked:
            return None
        day_count = self._count_posts(symbol, period="day")
        if day_count is None:
            return None
        week_count = self._count_posts(symbol, period="week")
        if week_count is None:
            return None
        if week_count >= DEFAULT_LIMIT:
            logger.warning(
                "Reddit week search for %s hit the %d-post page limit — "
                "baseline unknown, social velocity skipped.",
                symbol,
                DEFAULT_LIMIT,
            )
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
            response = translated_get(
                self._session,
                url,
                params=params,
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
            )
            if response.status_code == HTTPStatus.FORBIDDEN:
                self._blocked = True
                logger.warning(
                    "Reddit answered 403 for %s — no more Reddit requests this cycle "
                    "(see docs/configuration.md, social velocity)",
                    symbol,
                )
                return None
            raise_for_status_translated(response)
            payload = json_payload(response)
            data = payload["data"]
            children = data["children"]
            if not isinstance(children, list):
                return None
            return len(children)
        except (PortError, KeyError, TypeError):
            logger.exception("Reddit social fetch failed for %s (t=%s)", symbol, period)
            return None


class NullSocialVelocityAdapter(SocialVelocityPort):
    """Domyślny no-op (Fast-Loop / brak źródła): zawsze None, zero I/O."""

    def get_social_velocity(self, symbol: str) -> SocialVelocitySnapshot | None:
        return None
