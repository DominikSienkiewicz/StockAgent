"""AlphaVantageClient — thundering herd na leniwym _cached_feed.

Przy starcie puli kilka wątków widzi `_cached_feed is None` i każdy rusza pełny
fetch feedu (cold-start ~47 s z run #42). AV free tier ma brutalny dzienny limit
(25 req/dobę) — duplikacja fetchu może wyczerpać klucze. Lock → jeden fetch.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock

from src.infrastructure.adapters.alpha_vantage_client import AlphaVantageClient


def _make_response(payload: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def test_concurrent_first_call_fetches_feed_once(mocker) -> None:
    call_count = {"n": 0}
    lock = threading.Lock()

    def _slow_get(*_args, **_kwargs):
        with lock:
            call_count["n"] += 1
        time.sleep(0.02)  # poszerza okno race'a leniwego fetchu
        return _make_response({"feed": []})

    mocker.patch(
        "requests.Session.get",
        side_effect=_slow_get,
    )

    client = AlphaVantageClient(api_keys=["av1"], symbols=["AAPL"])
    n = 6
    barrier = threading.Barrier(n)

    def worker() -> None:
        barrier.wait()
        client.sentiment_for("AAPL")

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Jeden symbol w batchu → dokładnie jeden realny GET mimo 6 współbieżnych callerów.
    assert call_count["n"] == 1
