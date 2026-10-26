"""Testy wspólnego HTTP clienta z retry/backoff."""

from contextlib import suppress
from types import SimpleNamespace
from typing import Any, cast

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import RequestHistory, Retry

from src.application.http_metering import HttpCallMeter
from src.infrastructure.adapters._http import (
    RETRY_STATUS_FORCELIST,
    RETRY_TOTAL,
    RateLimiter,
    build_session,
    metered_http,
)


class TestBuildSession:
    def test_returns_requests_session(self) -> None:
        session = build_session()
        assert isinstance(session, requests.Session)

    def test_mounts_retry_adapter_on_https(self) -> None:
        session = build_session()
        adapter = cast(HTTPAdapter, session.get_adapter("https://example.com"))
        retry = adapter.max_retries
        assert isinstance(retry, Retry)
        assert retry.total == RETRY_TOTAL

    def test_http_scheme_not_mounted_with_custom_retry(self) -> None:
        # https-only: nie montujemy już retry-adaptera na http://. Schemat http
        # spada na domyślny adapter requests (bez naszego retry total=3) — cała
        # komunikacja agenta idzie po https.
        session = build_session()
        adapter = cast(HTTPAdapter, session.get_adapter("http://example.com"))
        assert adapter.max_retries.total != RETRY_TOTAL

    def test_retries_transient_status_codes(self) -> None:
        session = build_session()
        retry = cast(HTTPAdapter, session.get_adapter("https://x.com")).max_retries
        # 429 + 5xx powinny być na liście do retry
        for status in (429, 500, 502, 503, 504):
            assert status in RETRY_STATUS_FORCELIST
            assert status in retry.status_forcelist

    def test_does_not_raise_on_status(self) -> None:
        # raise_on_status=False — status sprawdzamy sami przez raise_for_status()
        session = build_session()
        retry = cast(HTTPAdapter, session.get_adapter("https://x.com")).max_retries
        assert retry.raise_on_status is False

    def test_retries_get_and_post(self) -> None:
        session = build_session()
        retry = cast(HTTPAdapter, session.get_adapter("https://x.com")).max_retries
        assert retry.allowed_methods is not None
        assert "GET" in retry.allowed_methods
        assert "POST" in retry.allowed_methods


class TestRetryLimits:
    def test_retry_post_false_does_not_retry_post(self) -> None:
        # MAIL-03: POST bez klucza idempotencji powtórzony po 5xx dubluje wiadomość.
        session = build_session(retry_post=False)
        retry = cast(HTTPAdapter, session.get_adapter("https://x.com")).max_retries
        assert retry.allowed_methods is not None
        assert "GET" in retry.allowed_methods
        assert "POST" not in retry.allowed_methods

    def test_retry_after_sleep_is_capped(self, mocker: Any) -> None:
        # NIEZ-06: Retry-After potrafi żądać godzin, a job ma 15 minut limitu.
        from urllib3.response import HTTPResponse

        retry = cast(HTTPAdapter, build_session().get_adapter("https://x.com")).max_retries
        sleep = mocker.patch("urllib3.util.retry.time.sleep")

        retry.sleep(HTTPResponse(status=429, headers={"Retry-After": "3600"}))

        assert sleep.call_args.args[0] <= 10


class TestRateLimiter:
    """DAT-12: token bucket współdzielony przez adaptery jednego API w procesie."""

    @staticmethod
    def _limiter(rate_per_minute: int) -> tuple[RateLimiter, list[float], list[float]]:
        from src.infrastructure.adapters._http import RateLimiter

        clock = [0.0]
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock[0] += seconds

        limiter = RateLimiter(rate_per_minute, clock=lambda: clock[0], sleep=sleep)
        return limiter, clock, sleeps

    def test_burst_up_to_the_rate_does_not_wait(self) -> None:
        limiter, _, sleeps = self._limiter(55)

        for _ in range(55):
            limiter.acquire()

        assert sleeps == []

    def test_request_over_the_budget_waits_for_one_token(self) -> None:
        limiter, _, sleeps = self._limiter(60)

        for _ in range(61):
            limiter.acquire()

        # 60/min = 1 token na sekundę: 61. request czeka pełną sekundę.
        assert sleeps == [1.0]

    def test_tokens_refill_with_time(self) -> None:
        limiter, clock, sleeps = self._limiter(60)
        for _ in range(60):
            limiter.acquire()

        clock[0] += 30.0
        for _ in range(30):
            limiter.acquire()

        assert sleeps == []

    def test_rate_must_be_positive(self) -> None:
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

    def send(
        self, request: requests.PreparedRequest, *args: Any, **kwargs: Any
    ) -> requests.Response:
        response = requests.Response()
        response.status_code = self._statuses.pop(0)
        assert request.url is not None
        response.url = request.url
        response.request = request
        response._content = b"{}"
        if self._retries:
            history = tuple(
                RequestHistory("GET", request.url, None, 429, None) for _ in range(self._retries)
            )
            response.raw = SimpleNamespace(retries=Retry(total=3, history=history))
        return response


class TestHttpCallMeter:
    # NIEZ-24: licznik żądań per host przez hook sesji — limity dostawców
    # (Finnhub 60/min, AV 25/dobę/klucz) bez pomiaru były pilnowane po omacku.

    def test_response_hook_counts_requests_per_host_and_status(self) -> None:
        meter = HttpCallMeter()
        session = build_session(meter=meter)
        session.mount("https://", _ScriptedAdapter([200, 429]))

        session.get("https://finnhub.io/api/v1/quote?symbol=NVDA&token=SECRET123")
        session.get("https://finnhub.io/api/v1/quote?symbol=AAPL&token=SECRET123")

        assert meter.snapshot()["finnhub.io"] == {"requests": 2, "2xx": 1, "429": 1}

    def test_meter_keys_by_host_only_never_by_url_or_query(self) -> None:
        meter = HttpCallMeter()
        session = build_session(meter=meter)
        session.mount("https://", _ScriptedAdapter([200, 503]))

        session.get("https://www.alphavantage.co/query?apikey=SECRET123")
        session.get("https://api.coingecko.com/api/v3/simple/price")

        snapshot = meter.snapshot()
        assert set(snapshot) == {"www.alphavantage.co", "api.coingecko.com"}
        assert snapshot["api.coingecko.com"] == {"requests": 1, "5xx": 1}
        assert "SECRET123" not in repr(snapshot)

    def test_urllib3_retries_are_counted(self) -> None:
        meter = HttpCallMeter()
        session = build_session(meter=meter)
        session.mount("https://", _ScriptedAdapter([200], retries=2))

        session.get("https://finnhub.io/api/v1/quote")

        assert meter.snapshot()["finnhub.io"] == {"requests": 1, "2xx": 1, "retries": 2}

    def test_elapsed_time_is_summed_per_host(self) -> None:
        meter = HttpCallMeter()
        session = build_session(meter=meter)
        session.mount("https://", _ScriptedAdapter([200, 200]))

        session.get("https://finnhub.io/a")
        session.get("https://finnhub.io/b")

        assert meter.elapsed_seconds()["finnhub.io"] >= 0.0

    def test_installed_meter_counts_sessions_built_without_one(self) -> None:
        # Hook obejmuje adaptery bez zmian w ich kodzie: sesja zbudowana gołym
        # `build_session()` liczy się do miernika zainstalowanego na przebieg.
        meter = HttpCallMeter()
        session = build_session()
        session.mount("https://", _ScriptedAdapter([200, 200]))

        with metered_http(meter):
            session.get("https://api.telegram.org/x")
        session.get("https://api.telegram.org/y")

        assert meter.snapshot() == {"api.telegram.org": {"requests": 1, "2xx": 1}}

    def test_session_without_any_meter_still_works(self) -> None:
        session = build_session()
        session.mount("https://", _ScriptedAdapter([200]))

        assert session.get("https://finnhub.io/x").status_code == 200


class TestCircuitBreaker:
    """NIEZ-31: po 5 kolejnych porażkach (5xx, timeout, zerwane połączenie) obwód
    hosta otwiera się do końca procesu — kolejne żądania od razu rzucają
    `ProviderUnavailable` bez ruchu sieciowego. 4xx nie liczy się jako porażka."""

    URL = "https://finnhub.io/api/v1/quote"

    @staticmethod
    def _response(status: int) -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response.url = TestCircuitBreaker.URL
        return response

    def test_breaker_opens_after_consecutive_failures(self, mocker: Any) -> None:
        from src.application.errors import ProviderUnavailable

        transport = mocker.patch.object(
            HTTPAdapter, "send", side_effect=requests.ConnectionError("down")
        )
        session = build_session()
        for _ in range(5):
            with pytest.raises(requests.ConnectionError):
                session.get(self.URL)

        with pytest.raises(ProviderUnavailable):
            session.get(self.URL)

        assert transport.call_count == 5

    def test_server_errors_and_timeouts_count_as_failures(self, mocker: Any) -> None:
        from src.application.errors import ProviderUnavailable

        failures = [self._response(503)] * 3 + [requests.ReadTimeout("slow")] * 2
        transport = mocker.patch.object(HTTPAdapter, "send", side_effect=failures)
        session = build_session()
        for _ in range(5):
            with suppress(requests.ReadTimeout):
                session.get(self.URL)

        with pytest.raises(ProviderUnavailable):
            session.get(self.URL)
        assert transport.call_count == 5

    def test_client_errors_do_not_open_the_circuit(self, mocker: Any) -> None:
        transport = mocker.patch.object(HTTPAdapter, "send", return_value=self._response(404))
        session = build_session()

        for _ in range(8):
            assert session.get(self.URL).status_code == 404

        assert transport.call_count == 8

    def test_success_resets_the_consecutive_count(self, mocker: Any) -> None:
        failures = [requests.ConnectionError("down")] * 4
        outcomes = [*failures, self._response(200), *failures, self._response(200)]
        mocker.patch.object(HTTPAdapter, "send", side_effect=outcomes)
        session = build_session()

        statuses: list[int | None] = []
        for _ in outcomes:
            try:
                statuses.append(session.get(self.URL).status_code)
            except requests.ConnectionError:
                statuses.append(None)

        assert statuses[-1] == 200

    def test_circuit_is_per_host(self, mocker: Any) -> None:
        mocker.patch.object(
            HTTPAdapter,
            "send",
            side_effect=[requests.ConnectionError("down")] * 5 + [self._response(200)],
        )
        session = build_session()
        for _ in range(5):
            with pytest.raises(requests.ConnectionError):
                session.get(self.URL)

        assert session.get("https://api.coingecko.com/api/v3/ping").status_code == 200

    def test_open_circuit_error_follows_existing_degradation_paths(self, mocker: Any) -> None:
        # Adaptery łapią dziś `requests.RequestException` (EDGAR, Reddit, NBP, AV) —
        # błąd otwartego obwodu musi trafić na te same ścieżki degradacji.
        mocker.patch.object(HTTPAdapter, "send", side_effect=requests.ConnectionError("down"))
        session = build_session()
        for _ in range(5):
            with pytest.raises(requests.ConnectionError):
                session.get(self.URL)

        with pytest.raises(requests.RequestException) as excinfo:
            session.get(self.URL)

        assert "finnhub.io" in str(excinfo.value)

    def test_circuit_states_report_skipped_symbols(self, mocker: Any) -> None:
        # MAIL-21 i banner kwot czytają stan obwodów przez `CircuitStatePort`.
        from src.application.ports import CircuitState, CircuitStatePort
        from src.infrastructure.adapters._http import process_circuits
        from src.infrastructure.logging_setup import symbol_context

        mocker.patch.object(HTTPAdapter, "send", side_effect=requests.ConnectionError("down"))
        session = build_session()
        for symbol in ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AAPL", "META", "META"]:
            with symbol_context(symbol), suppress(requests.ConnectionError):
                session.get(self.URL)

        circuits: CircuitStatePort = process_circuits()
        assert circuits.circuit_states() == (
            CircuitState(
                source="Finnhub",
                host="finnhub.io",
                is_open=True,
                consecutive_failures=5,
                skipped_calls=3,
                skipped_symbols=frozenset({"AAPL", "META"}),
            ),
        )


class TestFailedRequestsBySource:
    """MAIL-21: błędne odpowiedzi z licznika HTTP przypisane dostawcom po hoście."""

    def test_counts_rate_limits_and_server_errors_per_provider(self) -> None:
        from src.infrastructure.adapters._http import failed_requests_by_source

        snapshot = {
            "finnhub.io": {"requests": 40, "2xx": 35, "429": 2, "5xx": 3},
            "www.alphavantage.co": {"requests": 5, "2xx": 5},
            "unknown.example": {"requests": 2, "5xx": 2},
        }

        assert failed_requests_by_source(snapshot) == {"Finnhub": 5, "unknown.example": 2}

    def test_client_errors_count_only_for_sources_that_403_instead_of_404(self) -> None:
        from src.infrastructure.adapters._http import failed_requests_by_source

        snapshot = {
            "www.sec.gov": {"requests": 3, "4xx": 1},
            "data.sec.gov": {"requests": 2, "4xx": 2},
            "www.reddit.com": {"requests": 1, "4xx": 1},
            "finnhub.io": {"requests": 9, "4xx": 4},
        }

        assert failed_requests_by_source(snapshot) == {"EDGAR": 3, "Reddit": 1}


class TestTranslatedHelpers:
    """ARCH-33: wspólne pomocniki adapterów HTTP — błąd sieci, status i śmieciowa
    treść wychodzą w taksonomii portów, wciąż jako typy `requests` (bez sekretu)."""

    URL = "https://api.example.test/v1/data"

    @staticmethod
    def _response(status: int, body: bytes = b"{}") -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response.reason = "Error"
        response.url = "https://api.example.test/v1/data?apikey=secret-value-123"
        response._content = body
        return response

    def test_transport_error_is_provider_unavailable_without_secret(self, mocker: Any) -> None:
        from src.application.errors import ProviderUnavailable
        from src.infrastructure.adapters._http import translated_get

        mocker.patch(
            "requests.Session.get",
            side_effect=requests.ReadTimeout("Read timed out: /v1/data?apikey=secret-value-123"),
        )

        with pytest.raises(ProviderUnavailable) as excinfo:
            translated_get(requests.Session(), self.URL, timeout=1)

        assert isinstance(excinfo.value, requests.ReadTimeout)
        assert "secret-value-123" not in str(excinfo.value)

    @pytest.mark.parametrize(
        ("status", "kind_name"),
        [(429, "QuotaExhausted"), (503, "ProviderUnavailable"), (404, "PortError")],
    )
    def test_error_status_maps_to_its_kind(self, status: int, kind_name: str) -> None:
        from src.application import errors
        from src.infrastructure.adapters._http import raise_for_status_translated

        with pytest.raises(getattr(errors, kind_name)) as excinfo:
            raise_for_status_translated(self._response(status))

        assert isinstance(excinfo.value, requests.HTTPError)
        assert "secret-value-123" not in str(excinfo.value)

    def test_success_status_does_not_raise(self) -> None:
        from src.infrastructure.adapters._http import raise_for_status_translated

        raise_for_status_translated(self._response(200))

    def test_unparseable_body_is_malformed_payload(self) -> None:
        from src.application.errors import MalformedPayload
        from src.infrastructure.adapters._http import json_payload

        with pytest.raises(MalformedPayload) as excinfo:
            json_payload(self._response(200, b"<html>maintenance</html>"))

        assert isinstance(excinfo.value, requests.JSONDecodeError)

    def test_parsed_body_is_returned(self) -> None:
        from src.infrastructure.adapters._http import json_payload

        assert json_payload(self._response(200, b'{"a": 1}')) == {"a": 1}
