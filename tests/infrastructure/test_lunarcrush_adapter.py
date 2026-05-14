from unittest.mock import Mock

import pytest
import requests

from src.application.ports import SentimentPort
from src.infrastructure.adapters.lunarcrush_api import LunarCrushAdapter


@pytest.fixture
def adapter() -> LunarCrushAdapter:
    return LunarCrushAdapter(api_key="lc-token-xyz")


def _ok_response(payload: dict) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


_SAMPLE_RESPONSE = {
    "data": {
        "symbol": "AAPL",
        "galaxy_score": 78,
        "social_volume": 850,
        "average_sentiment": 4.2,
        "social_score": 145020,
    }
}


class TestGetSocialScore:
    def test_returns_normalized_metrics_dict(self, adapter, mocker):
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_RESPONSE),
        )

        result = adapter.get_social_score("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["galaxy_score"] == 78
        assert result["social_volume"] == 850
        assert result["average_sentiment"] == 4.2

    def test_passes_bearer_token_in_authorization_header(self, adapter, mocker):
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_RESPONSE),
        )

        adapter.get_social_score("AAPL")

        headers = mock_get.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer lc-token-xyz"

    def test_url_contains_symbol(self, adapter, mocker):
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_RESPONSE),
        )

        adapter.get_social_score("AAPL")

        call = mock_get.call_args
        url = call.args[0] if call.args else call.kwargs["url"]
        assert "AAPL" in url

    def test_sets_request_timeout(self, adapter, mocker):
        mock_get = mocker.patch(
            "requests.Session.get",
            return_value=_ok_response(_SAMPLE_RESPONSE),
        )

        adapter.get_social_score("AAPL")

        assert mock_get.call_args.kwargs["timeout"] > 0

    def test_raises_on_http_error(self, adapter, mocker):
        response = Mock(spec=requests.Response)
        response.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
        mocker.patch(
            "requests.Session.get",
            return_value=response,
        )

        with pytest.raises(requests.HTTPError):
            adapter.get_social_score("AAPL")

    def test_returns_empty_metrics_when_data_missing(self, adapter, mocker):
        # LunarCrush przy nieznanym tickerze może zwrócić 200 z pustym data
        mocker.patch(
            "requests.Session.get",
            return_value=_ok_response({"data": {}}),
        )

        result = adapter.get_social_score("UNKNOWN")

        assert result["galaxy_score"] is None
        assert result["social_volume"] is None


class TestAdapterImplementsPort:
    def test_is_sentiment_port(self):
        assert issubclass(LunarCrushAdapter, SentimentPort)


@pytest.mark.integration
class TestLunarCrushLive:
    def test_fetches_real_metrics(self):
        import os

        api_key = os.environ.get("LUNARCRUSH_API_KEY")
        if not api_key:
            pytest.skip("LUNARCRUSH_API_KEY not set")

        adapter = LunarCrushAdapter(api_key=api_key)
        result = adapter.get_social_score("AAPL")
        assert "galaxy_score" in result
