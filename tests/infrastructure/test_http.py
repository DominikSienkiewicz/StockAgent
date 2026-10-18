"""Testy wspólnego HTTP clienta z retry/backoff."""

import requests
from urllib3.util.retry import Retry

from src.infrastructure.adapters._http import (
    RETRY_STATUS_FORCELIST,
    RETRY_TOTAL,
    build_session,
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
