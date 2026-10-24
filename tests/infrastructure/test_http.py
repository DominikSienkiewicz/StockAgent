"""Testy wspólnego HTTP clienta z retry/backoff."""

from types import SimpleNamespace

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import RequestHistory, Retry

from src.application.http_metering import HttpCallMeter
from src.infrastructure.adapters._http import (
    RETRY_STATUS_FORCELIST,
    RETRY_TOTAL,
    build_session,
    metered_http,
)


class TestBuildSession:
    def test_returns_requests_session(self):
        session = build_session()
        assert isinstance(session, requests.Session)

    def test_mounts_retry_adapter_on_https(self):
        session = build_session()
        adapter = session.get_adapter("https://example.com")
        retry = adapter.max_retries
        assert isinstance(retry, Retry)
        assert retry.total == RETRY_TOTAL

    def test_http_scheme_not_mounted_with_custom_retry(self):
        # https-only: nie montujemy już retry-adaptera na http://. Schemat http
        # spada na domyślny adapter requests (bez naszego retry total=3) — cała
        # komunikacja agenta idzie po https.
        session = build_session()
        adapter = session.get_adapter("http://example.com")
        assert adapter.max_retries.total != RETRY_TOTAL

    def test_retries_transient_status_codes(self):
        session = build_session()
        retry = session.get_adapter("https://x.com").max_retries
        # 429 + 5xx powinny być na liście do retry
        for status in (429, 500, 502, 503, 504):
            assert status in RETRY_STATUS_FORCELIST
            assert status in retry.status_forcelist

    def test_does_not_raise_on_status(self):
        # raise_on_status=False — status sprawdzamy sami przez raise_for_status()
        session = build_session()
        retry = session.get_adapter("https://x.com").max_retries
        assert retry.raise_on_status is False

    def test_retries_get_and_post(self):
        session = build_session()
        retry = session.get_adapter("https://x.com").max_retries
        assert "GET" in retry.allowed_methods
        assert "POST" in retry.allowed_methods


class TestRetryLimits:
    def test_retry_post_false_does_not_retry_post(self):
        # MAIL-03: POST bez klucza idempotencji powtórzony po 5xx dubluje wiadomość.
        retry = build_session(retry_post=False).get_adapter("https://x.com").max_retries
        assert "GET" in retry.allowed_methods
        assert "POST" not in retry.allowed_methods

    def test_retry_after_sleep_is_capped(self, mocker):
        # NIEZ-06: Retry-After potrafi żądać godzin, a job ma 15 minut limitu.
        from urllib3.response import HTTPResponse

        retry = build_session().get_adapter("https://x.com").max_retries
        sleep = mocker.patch("urllib3.util.retry.time.sleep")

        retry.sleep(HTTPResponse(status=429, headers={"Retry-After": "3600"}))

        assert sleep.call_args.args[0] <= 10


class TestRateLimiter:
    """DAT-12: token bucket współdzielony przez adaptery jednego API w procesie."""

    @staticmethod
    def _limiter(rate_per_minute: int):
        from src.infrastructure.adapters._http import RateLimiter

        clock = [0.0]
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock[0] += seconds

        limiter = RateLimiter(rate_per_minute, clock=lambda: clock[0], sleep=sleep)
        return limiter, clock, sleeps

    def test_burst_up_to_the_rate_does_not_wait(self):
        limiter, _, sleeps = self._limiter(55)

        for _ in range(55):
            limiter.acquire()

        assert sleeps == []

    def test_request_over_the_budget_waits_for_one_token(self):
        limiter, _, sleeps = self._limiter(60)

        for _ in range(61):
            limiter.acquire()

        # 60/min = 1 token na sekundę: 61. request czeka pełną sekundę.
        assert sleeps == [1.0]

    def test_tokens_refill_with_time(self):
        limiter, clock, sleeps = self._limiter(60)
        for _ in range(60):
            limiter.acquire()

        clock[0] += 30.0
        for _ in range(30):
            limiter.acquire()

        assert sleeps == []

    def test_rate_must_be_positive(self):
        import pytest

        from src.infrastructure.adapters._http import RateLimiter

        with pytest.raises(ValueError):
            RateLimiter(0)

class _ScriptedAdapter(HTTPAdapter):
    """Atrapa transportu: kolejne statusy z listy, bez sieci."""

    def __init__(self, statuses: list[int], *, retries: int = 0) -> None:
        super().__init__()
        self._statuses = list(statuses)
        self._retries = retries

    def send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        response = requests.Response()
        response.status_code = self._statuses.pop(0)
        response.url = request.url
        response.request = request
        response._content = b"{}"
        if self._retries:
            history = tuple(
                RequestHistory("GET", request.url, None, 429, None)
                for _ in range(self._retries)
            )
            response.raw = SimpleNamespace(retries=Retry(total=3, history=history))
        return response


class TestHttpCallMeter:
    # NIEZ-24: licznik żądań per host przez hook sesji — limity dostawców
    # (Finnhub 60/min, AV 25/dobę/klucz) bez pomiaru były pilnowane po omacku.

    def test_response_hook_counts_requests_per_host_and_status(self):
        meter = HttpCallMeter()
        session = build_session(meter=meter)
        session.mount("https://", _ScriptedAdapter([200, 429]))

        session.get("https://finnhub.io/api/v1/quote?symbol=NVDA&token=SECRET123")
        session.get("https://finnhub.io/api/v1/quote?symbol=AAPL&token=SECRET123")

        assert meter.snapshot()["finnhub.io"] == {"requests": 2, "2xx": 1, "429": 1}

    def test_meter_keys_by_host_only_never_by_url_or_query(self):
        meter = HttpCallMeter()
        session = build_session(meter=meter)
        session.mount("https://", _ScriptedAdapter([200, 503]))

        session.get("https://www.alphavantage.co/query?apikey=SECRET123")
        session.get("https://api.coingecko.com/api/v3/simple/price")

        snapshot = meter.snapshot()
        assert set(snapshot) == {"www.alphavantage.co", "api.coingecko.com"}
        assert snapshot["api.coingecko.com"] == {"requests": 1, "5xx": 1}
        assert "SECRET123" not in repr(snapshot)

    def test_urllib3_retries_are_counted(self):
        meter = HttpCallMeter()
        session = build_session(meter=meter)
        session.mount("https://", _ScriptedAdapter([200], retries=2))

        session.get("https://finnhub.io/api/v1/quote")

        assert meter.snapshot()["finnhub.io"] == {"requests": 1, "2xx": 1, "retries": 2}

    def test_elapsed_time_is_summed_per_host(self):
        meter = HttpCallMeter()
        session = build_session(meter=meter)
        session.mount("https://", _ScriptedAdapter([200, 200]))

        session.get("https://finnhub.io/a")
        session.get("https://finnhub.io/b")

        assert meter.elapsed_seconds()["finnhub.io"] >= 0.0

    def test_installed_meter_counts_sessions_built_without_one(self):
        # Hook obejmuje adaptery bez zmian w ich kodzie: sesja zbudowana gołym
        # `build_session()` liczy się do miernika zainstalowanego na przebieg.
        meter = HttpCallMeter()
        session = build_session()
        session.mount("https://", _ScriptedAdapter([200, 200]))

        with metered_http(meter):
            session.get("https://api.telegram.org/x")
        session.get("https://api.telegram.org/y")

        assert meter.snapshot() == {"api.telegram.org": {"requests": 1, "2xx": 1}}

    def test_session_without_any_meter_still_works(self):
        session = build_session()
        session.mount("https://", _ScriptedAdapter([200]))

        assert session.get("https://finnhub.io/x").status_code == 200
