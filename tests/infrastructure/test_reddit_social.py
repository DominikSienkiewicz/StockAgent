"""Testy RedditSocialAdapter — SocialVelocityPort z publicznego search.json.

Reddit nie wymaga OAuth dla publicznego wyszukiwania, ale blokuje domyślne
User-Agenty — adapter ustawia własny nagłówek. Adapter robi DWA GET-y
(t=day i t=week), więc mockujemy Session.get przez `side_effect` z dwiema
odpowiedziami. Wszystkie błędy (non-200, brak pól, parse) → None — to
opcjonalne źródło alpha, nigdy nie może rzucić."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import requests

from src.application.ports import SocialVelocityPort
from src.domain.social_velocity import SocialTrend, SocialVelocitySnapshot
from src.infrastructure.adapters.reddit_social import (
    NullSocialVelocityAdapter,
    RedditSocialAdapter,
)


def _listing(num_children: int) -> dict:
    """Format Reddit search.json: {"data": {"children": [...], "dist": N}}."""
    children = [{"data": {"id": str(i)}} for i in range(num_children)]
    return {"data": {"children": children, "dist": num_children}}


def _ok_response(payload: dict) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = Mock()
    return response


def _error_response(status_code: int) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.raise_for_status.side_effect = requests.HTTPError(
        f"{status_code} error"
    )
    return response


@pytest.fixture
def adapter() -> RedditSocialAdapter:
    return RedditSocialAdapter()


class TestGetSocialVelocity:
    def test_returns_snapshot_with_velocity(
        self, adapter: RedditSocialAdapter, mocker
    ) -> None:
        # 30 postów dziś, 70 w tygodniu → baseline = 70/7 = 10, ratio = 3.0.
        mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_response(_listing(30)), _ok_response(_listing(70))],
        )

        snap = adapter.get_social_velocity("GME")

        assert isinstance(snap, SocialVelocitySnapshot)
        assert snap.symbol == "GME"
        assert snap.mentions_24h == 30
        assert snap.baseline_mentions == 10.0
        assert snap.velocity_ratio() == 3.0
        assert snap.trend() is SocialTrend.SURGING

    def test_avg_sentiment_is_none(
        self, adapter: RedditSocialAdapter, mocker
    ) -> None:
        # Search nie daje sentymentu — pole musi być None.
        mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_response(_listing(5)), _ok_response(_listing(14))],
        )

        snap = adapter.get_social_velocity("GME")

        assert snap is not None
        assert snap.avg_sentiment is None

    def test_makes_two_gets_day_and_week(
        self, adapter: RedditSocialAdapter, mocker
    ) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_response(_listing(1)), _ok_response(_listing(7))],
        )

        adapter.get_social_velocity("GME")

        assert mock_get.call_count == 2
        # Okno czasowe (t) i symbol (q) idą jako query params, nie w URL-u.
        periods = [
            call.kwargs.get("params", {}).get("t")
            for call in mock_get.call_args_list
        ]
        queries = [
            call.kwargs.get("params", {}).get("q")
            for call in mock_get.call_args_list
        ]
        assert "day" in periods
        assert "week" in periods
        assert all(q == "GME" for q in queries)

    def test_sets_custom_user_agent_header(
        self, adapter: RedditSocialAdapter, mocker
    ) -> None:
        # Reddit blokuje domyślne UA — własny nagłówek jest obowiązkowy.
        mock_get = mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_response(_listing(1)), _ok_response(_listing(7))],
        )

        adapter.get_social_velocity("GME")

        for call in mock_get.call_args_list:
            headers = call.kwargs.get("headers", {})
            assert headers.get("User-Agent")

    def test_sets_timeout_on_every_call(
        self, adapter: RedditSocialAdapter, mocker
    ) -> None:
        mock_get = mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_response(_listing(1)), _ok_response(_listing(7))],
        )

        adapter.get_social_velocity("GME")

        for call in mock_get.call_args_list:
            assert "timeout" in call.kwargs
            assert call.kwargs["timeout"] > 0

    def test_uses_dist_when_present(
        self, adapter: RedditSocialAdapter, mocker
    ) -> None:
        # dist powinno odpowiadać liczbie children — sprawdzamy spójność liczenia.
        mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_response(_listing(42)), _ok_response(_listing(70))],
        )

        snap = adapter.get_social_velocity("AMC")

        assert snap is not None
        assert snap.mentions_24h == 42

    def test_returns_none_on_http_error(
        self, adapter: RedditSocialAdapter, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get", return_value=_error_response(503)
        )

        assert adapter.get_social_velocity("GME") is None

    def test_returns_none_on_week_error(
        self, adapter: RedditSocialAdapter, mocker
    ) -> None:
        # Pierwszy GET ok, drugi pada — brak baseline'u → None.
        mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_response(_listing(30)), _error_response(503)],
        )

        assert adapter.get_social_velocity("GME") is None

    def test_returns_none_on_missing_fields(
        self, adapter: RedditSocialAdapter, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=[_ok_response({"foo": "bar"}), _ok_response(_listing(7))],
        )

        assert adapter.get_social_velocity("GME") is None

    def test_returns_none_on_request_exception(
        self, adapter: RedditSocialAdapter, mocker
    ) -> None:
        mocker.patch(
            "requests.Session.get",
            side_effect=requests.ConnectionError("boom"),
        )

        assert adapter.get_social_velocity("GME") is None

    def test_returns_none_on_parse_error(
        self, adapter: RedditSocialAdapter, mocker
    ) -> None:
        bad = Mock(spec=requests.Response)
        bad.status_code = 200
        bad.raise_for_status = Mock()
        bad.json.side_effect = ValueError("not json")
        mocker.patch("requests.Session.get", return_value=bad)

        assert adapter.get_social_velocity("GME") is None


class TestNullAdapter:
    def test_returns_none_without_io(self, mocker) -> None:
        # Null nie robi żadnego I/O — patch jako strażnik: nie może być wołany.
        mock_get = mocker.patch("requests.Session.get")
        null = NullSocialVelocityAdapter()

        assert null.get_social_velocity("GME") is None
        mock_get.assert_not_called()

    def test_satisfies_port(self) -> None:
        assert isinstance(NullSocialVelocityAdapter(), SocialVelocityPort)
        assert isinstance(RedditSocialAdapter(), SocialVelocityPort)
