"""NIEZ-33 — publiczna strona statusu i JSON odznaki „last digest” z `cycle_runs`.

Brak maila wygląda jak spokojny dzień; strona mówi, kiedy ostatnio ruszył Fast Loop,
czy mail wyszedł, czy pętla jest wstrzymana, ile płatnych wywołań poszło w 7 i 30 dni
i jaki odsetek cykli był darmowy. Same agregaty: bez symboli, adresów i kwot.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from src.application.ports import CycleRun
from src.application.report_status_page import (
    build_public_status,
    render_status_badge,
    render_status_page_html,
)
from src.domain.liveness import LivenessExpectation

_NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
_FAST = LivenessExpectation("Fast Loop", 26.0, trading_day_slot=datetime.min.time())
_PAUSED = LivenessExpectation("Fast Loop", 0.0, trading_day_slot=datetime.min.time())


def _run(
    days_ago: float,
    *,
    delivered: bool | None = True,
    paid: dict[str, int] | None = None,
    loop: str = "fast",
    saved: int = 40,
    failures: int = 2,
) -> CycleRun:
    return CycleRun(
        loop=loop,
        run_id="12345",
        run_attempt=1,
        started_at=_NOW - timedelta(days=days_ago),
        duration_s=300.0,
        eligible=42,
        saved=saved,
        failures=failures,
        paid_calls={"llm": 7} if paid is None else paid,
        report_delivered=delivered,
    )


def test_status_page_shows_paused_and_last_run_without_pii() -> None:
    status = build_public_status([_run(60, delivered=False)], now=_NOW, fast_loop=_PAUSED)

    html = render_status_page_html(status)

    assert "wstrzymany" in html
    assert "2026-08-26" in html
    assert "niedoręczony" in html
    assert "@" not in html


def test_running_loop_shows_its_last_run_and_delivered_digest() -> None:
    status = build_public_status([_run(1), _run(2)], now=_NOW, fast_loop=_FAST)

    html = render_status_page_html(status)

    assert "działa" in html
    assert "1 dzień temu" in html
    assert "doręczony" in html
    assert "40 z 42" in html


def test_loop_the_deadman_would_alert_on_is_reported_as_overdue() -> None:
    status = build_public_status([_run(9)], now=_NOW, fast_loop=_FAST)

    assert "brak przebiegu w terminie" in render_status_page_html(status)


def test_without_runs_the_page_says_so() -> None:
    status = build_public_status([], now=_NOW, fast_loop=_FAST)

    html = render_status_page_html(status)

    assert "brak przebiegów w ostatnich 90 dniach" in html
    assert "brak cykli w ostatnich 30 dniach" in html


def test_last_digest_is_the_last_delivered_run_not_the_last_run() -> None:
    status = build_public_status(
        [_run(1, delivered=False), _run(3, delivered=True), _run(5, delivered=None)],
        now=_NOW,
        fast_loop=_FAST,
    )

    assert status.last_digest_at == _NOW - timedelta(days=3)
    assert "Ostatni wysłany digest: 2026-10-22 (3 dni temu)" in render_status_page_html(status)


def test_paid_calls_of_7_and_30_days_and_the_share_of_free_cycles() -> None:
    runs = [
        _run(1, paid={"llm": 7, "embedding": 3}),
        _run(2, paid={}),
        _run(10, paid={"llm": 5}),
        _run(20, paid={"llm": 0}),
        _run(40, paid={"llm": 100}),
        _run(3, paid={"llm": 2}, loop="slow"),
    ]

    status = build_public_status(runs, now=_NOW, fast_loop=_FAST)

    assert (status.paid_calls_7d, status.paid_calls_30d) == (12, 17)
    assert (status.free_cycles_30d, status.cycles_30d) == (2, 4)
    html = render_status_page_html(status)
    assert "7 dni: 12, 30 dni: 17" in html
    assert "2 z 4 (50%)" in html


def test_other_loops_are_listed_with_their_last_run() -> None:
    status = build_public_status(
        [_run(1), _run(6, loop="slow", delivered=None)], now=_NOW, fast_loop=_FAST
    )

    html = render_status_page_html(status)

    assert "Slow Loop" in html
    assert "2026-10-19" in html


def test_page_leaks_no_address_amount_or_currency() -> None:
    status = build_public_status([_run(1)], now=_NOW, fast_loop=_FAST)

    html = render_status_page_html(status)

    for leak in ("@", "$", "USD", "zł", "PLN"):
        assert leak not in html, leak


def test_badge_is_a_shields_endpoint_saying_how_old_the_last_digest_is() -> None:
    status = build_public_status([_run(3)], now=_NOW, fast_loop=_FAST)

    badge = render_status_badge(status)

    assert badge == {
        "schemaVersion": 1,
        "label": "last digest",
        "message": "3 days ago",
        "color": "green",
    }
    json.dumps(badge)


def test_badge_colours_follow_the_age_and_the_pause() -> None:
    def badge(days: float, expectation: LivenessExpectation = _FAST) -> dict[str, object]:
        return render_status_badge(
            build_public_status([_run(days)], now=_NOW, fast_loop=expectation)
        )

    assert (badge(0)["message"], badge(0)["color"]) == ("today", "brightgreen")
    assert (badge(1)["message"], badge(1)["color"]) == ("1 day ago", "brightgreen")
    assert badge(6)["color"] == "yellow"
    assert badge(12)["color"] == "red"
    assert badge(60, _PAUSED)["color"] == "lightgrey"


def test_badge_without_a_delivered_digest() -> None:
    badge = render_status_badge(build_public_status([], now=_NOW, fast_loop=_FAST))

    assert (badge["message"], badge["color"]) == ("none in 90 days", "lightgrey")
