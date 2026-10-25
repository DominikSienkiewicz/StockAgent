"""BEZ-07: redakcja sekretów w komunikatach wyjątków HTTP.

`requests` wkleja pełny URL (z `?token=` / `&apikey=`) do treści `HTTPError`
i `ConnectionError`, a ten tekst trafia do maila, `quota_alerts` i publicznych
logów GHA. Pomocniki z `_http.py` rzucają wyjątek tego samego typu z czystym
komunikatem i `from None`, żeby traceback nie dokleił oryginału z URL-em.
"""

from __future__ import annotations

import traceback
from unittest.mock import Mock

import pytest
import requests

from src.infrastructure.adapters._http import (
    MIN_LITERAL_SECRET_LENGTH,
    raise_for_status_redacted,
    redact_secrets,
    safe_get,
)

SECRET = "s3cr3t-value-123"


def _real_response(status_code: int, url: str, reason: str = "Unauthorized") -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.reason = reason
    response.url = url
    return response


def _printed(exc: BaseException) -> str:
    """Tekst, który `logger.exception` / `traceback` wypisałby dla wyjątku."""
    return "".join(traceback.format_exception(exc))


class TestRedactSecrets:
    @pytest.mark.parametrize("param", ["apikey", "token", "api_key", "key", "APIKEY", "Token"])
    def test_redacts_secret_query_parameters(self, param: str) -> None:
        text = f"for url: https://api.example.com/q?symbol=AAPL&{param}={SECRET}&limit=1"

        redacted = redact_secrets(text)

        assert SECRET not in redacted
        assert "symbol=AAPL" in redacted
        assert "limit=1" in redacted

    def test_redacts_secret_as_first_query_parameter(self) -> None:
        redacted = redact_secrets(f"https://api.example.com/q?apikey={SECRET}")

        assert SECRET not in redacted

    def test_redacts_authorization_header_and_bearer_token(self) -> None:
        text = (
            f"headers={{'Authorization': 'Bearer {SECRET}'}} "
            f"Authorization: Basic {SECRET}x "
            f"curl -H 'Bearer {SECRET}y'"
        )

        redacted = redact_secrets(text)

        assert SECRET not in redacted
        assert "Authorization" in redacted

    def test_redacts_api_key_header_in_dict_form(self) -> None:
        # Supabase/PostgREST niesie klucz w nagłówku `apikey`, Anthropic w `x-api-key`.
        text = f"{{'apikey': '{SECRET}', 'x-api-key': \"{SECRET}z\"}}"

        assert SECRET not in redact_secrets(text)

    def test_redacts_telegram_bot_token_in_path(self) -> None:
        text = "for url: https://api.telegram.org/bot123456:AAH-xyz_TOKEN/sendMessage"

        redacted = redact_secrets(text)

        assert "AAH-xyz_TOKEN" not in redacted
        assert "123456" not in redacted
        assert "/sendMessage" in redacted

    def test_redacts_slack_webhook_path(self) -> None:
        text = "Max retries exceeded with url: https://hooks.slack.com/services/T01/B02/XyZ123"

        redacted = redact_secrets(text)

        assert "T01/B02/XyZ123" not in redacted
        assert "hooks.slack.com" in redacted

    def test_redacts_literal_secret_values_from_settings(self) -> None:
        # Klucz bez rozpoznawalnego kształtu (np. echo w treści błędu dostawcy).
        text = f"provider rejected key {SECRET} and {SECRET}-suffix"

        redacted = redact_secrets(text, secret_values=[SECRET])

        assert SECRET not in redacted

    def test_redacts_longest_literal_first(self) -> None:
        # Klucz będący prefiksem innego nie może zostawić ogona dłuższego klucza.
        longer = f"{SECRET}-extended"

        redacted = redact_secrets(f"value={longer}", secret_values=[SECRET, longer])

        assert "extended" not in redacted

    def test_ignores_literal_values_too_short_to_be_secrets(self) -> None:
        # Krótka wartość (np. testowe `chat_id=42`) zjadałaby zwykły tekst logu.
        short = "x" * (MIN_LITERAL_SECRET_LENGTH - 1)

        assert redact_secrets(f"count {short}", secret_values=[short, ""]) == f"count {short}"

    def test_leaves_text_without_secrets_untouched(self) -> None:
        text = "Finnhub returned no price for symbol 'AAPL' (has_change_field=True)."

        assert redact_secrets(text) == text


class TestRaiseForStatusRedacted:
    def test_http_error_keeps_type_and_response_without_secret(self) -> None:
        response = _real_response(
            401, f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={SECRET}"
        )

        with pytest.raises(requests.HTTPError) as excinfo:
            raise_for_status_redacted(response)

        exc = excinfo.value
        assert type(exc) is requests.HTTPError
        assert exc.response is response
        assert "401 Client Error" in str(exc)
        assert "symbol=AAPL" in str(exc)
        assert SECRET not in str(exc)

    def test_traceback_does_not_chain_the_original_message(self) -> None:
        # `logger.exception` drukuje `__cause__` / `__context__` — oryginał ma URL z kluczem.
        response = _real_response(403, f"https://www.alphavantage.co/query?apikey={SECRET}")

        with pytest.raises(requests.HTTPError) as excinfo:
            raise_for_status_redacted(response)

        assert excinfo.value.__cause__ is None
        assert excinfo.value.__suppress_context__ is True
        assert SECRET not in _printed(excinfo.value)

    def test_success_does_not_raise(self) -> None:
        raise_for_status_redacted(_real_response(200, "https://x.example/q?token=t", "OK"))

    def test_mocked_response_still_raises_its_http_error(self) -> None:
        # Konwencja testów repo: Mock(spec=Response) z side_effect na raise_for_status.
        response = Mock(spec=requests.Response)
        response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        with pytest.raises(requests.HTTPError, match="500 Server Error"):
            raise_for_status_redacted(response)


class TestSafeGet:
    def test_returns_response_and_passes_arguments_through(self) -> None:
        session = Mock(spec=requests.Session)
        response = Mock(spec=requests.Response)
        session.get.return_value = response

        result = safe_get(session, "https://x.example/q", params={"token": SECRET}, timeout=5)

        assert result is response
        session.get.assert_called_once_with(
            "https://x.example/q", params={"token": SECRET}, timeout=5
        )

    @pytest.mark.parametrize(
        "exc_type",
        [
            requests.ConnectionError,
            requests.Timeout,
            requests.ConnectTimeout,
            requests.ReadTimeout,
            requests.exceptions.RetryError,
            requests.exceptions.SSLError,
        ],
    )
    def test_network_error_keeps_type_without_secret(
        self, exc_type: type[requests.RequestException]
    ) -> None:
        session = Mock(spec=requests.Session)
        session.get.side_effect = exc_type(
            "HTTPSConnectionPool(host='finnhub.io', port=443): Max retries exceeded with "
            f"url: /api/v1/quote?symbol=AAPL&token={SECRET} (Caused by NewConnectionError)"
        )

        with pytest.raises(exc_type) as excinfo:
            safe_get(session, "https://finnhub.io/api/v1/quote", timeout=5)

        exc = excinfo.value
        assert type(exc) is exc_type
        assert "Max retries exceeded" in str(exc)
        assert SECRET not in str(exc)
        assert exc.__cause__ is None
        assert SECRET not in _printed(exc)
