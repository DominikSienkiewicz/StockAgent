"""Adapter Finnhub — konsensus analityków i cel cenowy (sell-side, tło).

Dwa darmowe-/premium-owe endpointy Finnhub:
    GET /stock/recommendation?symbol={s}&token={k}
        → lista miesięcznych snapshotów rozkładu rekomendacji; bierzemy
          najświeższy po polu "period".
    GET /stock/price-target?symbol={s}&token={k}
        → {"targetMedian": .., "targetMean": ..}; bywa płatny (403) — wtedy
          po prostu zostawiamy price_target = None, konsensus i tak zwracamy.

Źródło opcjonalne (alpha): adapter NIGDY nie rzuca. Każdy błąd — sieci,
parsowania, braku pola, non-200 na rekomendacji — kończy się zwróceniem None.
Cienki I/O; cała logika oceny (rating/upside) siedzi w domenie.

DAT-12: każdy request bierze token ze wspólnego limitera Finnhub procesu, a 429
(QuotaAlert) i 401/403 (jeden WARNING) przestały być ciche.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from src.application.ports import AnalystConsensusPort
from src.application.quota_monitor import QuotaMonitor
from src.domain.analyst_consensus import AnalystConsensus
from src.infrastructure.adapters._finnhub_common import FinnhubStatusWatcher
from src.infrastructure.adapters._http import RateLimiter, build_session, safe_get

DEFAULT_BASE_URL = "https://finnhub.io/api/v1"
DEFAULT_TIMEOUT = 10.0

logger = logging.getLogger(__name__)


class FinnhubAnalystAdapter(AnalystConsensusPort):
    """Konsensus analityków z Finnhub recommendation + price-target."""

    def __init__(
        self,
        api_key: str,
        session: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: str = DEFAULT_BASE_URL,
        *,
        quota_monitor: QuotaMonitor | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")
        self._session = session if session is not None else build_session()
        self._rate_limiter = rate_limiter
        self._status = FinnhubStatusWatcher(
            "Finnhub analyst", "konsensus analityków", quota_monitor
        )

    def get_consensus(self, symbol: str) -> AnalystConsensus | None:
        try:
            snapshot = self._latest_recommendation(symbol)
            if snapshot is None:
                return None
            price_target = self._price_target(symbol)
            return AnalystConsensus(
                symbol=symbol,
                strong_buy=_as_int(snapshot.get("strongBuy")),
                buy=_as_int(snapshot.get("buy")),
                hold=_as_int(snapshot.get("hold")),
                sell=_as_int(snapshot.get("sell")),
                strong_sell=_as_int(snapshot.get("strongSell")),
                price_target=price_target,
            )
        except Exception:
            # Źródło opcjonalne — pojedynczy symbol nie może wywalić cyklu.
            logger.exception("Finnhub analyst consensus failed for %s", symbol)
            return None

    def _latest_recommendation(self, symbol: str) -> dict[str, Any] | None:
        """Najświeższy snapshot rozkładu rekomendacji lub None."""
        response = self._fetch("stock/recommendation", symbol)
        if response.status_code != 200:
            return None
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            return None
        # Bierzemy snapshot o maksymalnym "period" (najświeższy miesiąc).
        snapshots = [s for s in payload if isinstance(s, dict)]
        if not snapshots:
            return None
        return max(snapshots, key=lambda s: str(s.get("period", "")))

    def _price_target(self, symbol: str) -> float | None:
        """Mediana (fallback: średnia) celu cenowego. Premium/403/pusto → None."""
        response = self._fetch("stock/price-target", symbol)
        if response.status_code != 200:
            # price-target bywa płatny → 403; nie blokuje to konsensusu.
            return None
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        target = payload.get("targetMedian")
        if target is None:
            target = payload.get("targetMean")
        return _as_float(target)


    def _fetch(self, endpoint: str, symbol: str) -> requests.Response:
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        response = safe_get(
            self._session,
            f"{self._base_url}/{endpoint}",
            params={"symbol": symbol, "token": self._api_key},
            timeout=self._timeout,
        )
        self._status.observe(endpoint, symbol, response.status_code)
        return response


class NullAnalystConsensusAdapter(AnalystConsensusPort):
    """Brak źródła analityków (Fast-Loop / brak klucza) — zawsze None, zero I/O."""

    def get_consensus(self, symbol: str) -> AnalystConsensus | None:
        return None


def _as_int(value: Any) -> int:
    """Bezpieczna konwersja licznika rekomendacji na int (None/śmieci → 0)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    """Bezpieczna konwersja celu cenowego na float; 0/None/śmieci → None."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None
