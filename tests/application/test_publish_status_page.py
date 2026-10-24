"""NIEZ-33 — publikacja strony statusu i JSON odznaki jako use case (tylko porty)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

from src.application.errors import ProviderUnavailable
from src.application.ports import CycleRun, RepositoryPort, WebPublisherPort
from src.application.use_cases.publish_status_page import (
    PublishStatusPageUseCase,
    StatusPageOptions,
)
from src.domain.liveness import FAST_LOOP_TRADING_DAY_SLOT, LivenessExpectation

_NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
_OPTIONS = StatusPageOptions(
    fast_loop=LivenessExpectation("Fast Loop", 26.0, trading_day_slot=FAST_LOOP_TRADING_DAY_SLOT)
)


def _repo() -> Mock:
    repo = Mock(spec=RepositoryPort)
    repo.get_cycle_runs_since.return_value = [
        CycleRun(
            loop="fast",
            run_id="1",
            run_attempt=1,
            started_at=_NOW - timedelta(days=2),
            duration_s=10.0,
            eligible=42,
            saved=40,
            failures=2,
            paid_calls={"llm": 7},
            report_delivered=True,
        )
    ]
    return repo


def _publisher(page: str = "status/index.html", badge: str = "status/badge.json") -> Mock:
    publisher = Mock(spec=WebPublisherPort)
    publisher.publish.return_value = page
    publisher.publish_asset.return_value = badge
    return publisher


def _use_case(repo: Mock, publisher: Mock) -> PublishStatusPageUseCase:
    return PublishStatusPageUseCase(repo, publisher, _OPTIONS, clock=lambda: _NOW)


def test_publishes_the_page_and_the_badge_of_the_history_window() -> None:
    repo, publisher = _repo(), _publisher()

    assert _use_case(repo, publisher).publish() is True

    repo.get_cycle_runs_since.assert_called_once_with(_NOW - timedelta(days=90))
    assert "Status StockAgenta" in publisher.publish.call_args.args[0]
    name, content, content_type = publisher.publish_asset.call_args.args
    assert (name, content_type) == ("badge.json", "application/json")
    assert json.loads(content)["message"] == "2 days ago"


def test_a_failed_write_is_reported() -> None:
    assert _use_case(_repo(), _publisher(badge="")).publish() is False
    assert _use_case(_repo(), _publisher(page="")).publish() is False


def test_read_error_reaches_publish_and_run_swallows_it() -> None:
    repo, publisher = _repo(), _publisher()
    repo.get_cycle_runs_since.side_effect = ProviderUnavailable("supabase down")
    use_case = _use_case(repo, publisher)

    use_case.run()

    publisher.publish.assert_not_called()
    try:
        use_case.publish()
    except ProviderUnavailable:
        pass
    else:
        raise AssertionError("publish must let the read error through")
