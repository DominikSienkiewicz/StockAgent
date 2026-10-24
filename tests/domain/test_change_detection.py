"""NIEZ-20: detektor Page-Hinkley spadku kroczącej trafności.

Degradację widać dziś tylko jako ruchomy procent w mailu; detektor daje datę
i odróżnia zmianę od szumu. Parametry są pre-rejestrowane w module domeny.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.domain.change_detection import (
    PH_DELTA,
    PH_MIN_SAMPLES,
    PH_THRESHOLD,
    detect_accuracy_drop,
    page_hinkley,
)
from src.domain.track_record import ScoredCall

# Deterministyczny „szum” wokół 0.55: wzorzec o średniej dokładnie 0.55.
_STATIONARY_PATTERN = (0.75, 0.35, 0.65, 0.45, 0.55, 0.70, 0.40, 0.60, 0.50, 0.55)
MAX_DETECTION_DELAY = 25


def test_detects_drop_from_0_6_to_0_4_within_k_steps() -> None:
    outcomes = [0.6] * 30 + [0.4] * 40

    alarm = page_hinkley(outcomes)

    assert alarm is not None
    assert 30 <= alarm.alarm_at <= 30 + MAX_DETECTION_DELAY
    # Początek zmiany: pierwszy cykl po maksimum sumy skumulowanej.
    assert alarm.change_at == 30
    assert alarm.statistic > PH_THRESHOLD


def test_no_alarm_on_stationary_sequence() -> None:
    outcomes = list(_STATIONARY_PATTERN) * 30

    assert page_hinkley(outcomes) is None


def test_rise_in_accuracy_is_not_an_alarm() -> None:
    assert page_hinkley([0.4] * 30 + [0.7] * 40) is None


def test_no_alarm_before_the_burn_in() -> None:
    # Drastyczny spadek, ale zbyt krótka historia, by mieć poziom odniesienia.
    outcomes = [0.9] * 3 + [0.0] * (PH_MIN_SAMPLES - 4)

    assert page_hinkley(outcomes) is None


def test_parameters_can_be_tightened() -> None:
    outcomes = [0.6] * 30 + [0.4] * 40

    loose = page_hinkley(outcomes, delta=PH_DELTA, threshold=PH_THRESHOLD)
    tight = page_hinkley(outcomes, delta=PH_DELTA, threshold=PH_THRESHOLD / 2)

    assert loose is not None and tight is not None
    assert tight.alarm_at < loose.alarm_at


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        page_hinkley([0.5] * 20, delta=-0.1)
    with pytest.raises(ValueError):
        page_hinkley([0.5] * 20, threshold=0.0)


def _cycle_calls(day: str, hits: int, misses: int) -> list[ScoredCall]:
    return [
        ScoredCall(
            cycle=day, predicted_trend="BULLISH", is_correct=index < hits, realized_change=None
        )
        for index in range(hits + misses)
    ]


def _days(start: int, count: int) -> list[str]:
    first = date(2026, 6, 1) + timedelta(days=start)
    return [(first + timedelta(days=offset)).isoformat() for offset in range(count)]


def test_accuracy_drop_reports_the_date_of_the_change() -> None:
    calls: list[ScoredCall] = []
    for day in _days(0, 30):
        calls += _cycle_calls(day, 6, 4)
    for day in _days(30, 40):
        calls += _cycle_calls(day, 4, 6)
    # SIDEWAYS nie stawia kierunku — poza sekwencją trafień.
    sideways = ScoredCall(
        cycle=_days(35, 1)[0], predicted_trend="SIDEWAYS", is_correct=False, realized_change=None
    )
    calls += [sideways] * 50

    drop = detect_accuracy_drop(calls)

    assert drop is not None
    assert drop.since == "2026-07-01"
    assert drop.before_rate == pytest.approx(0.6)
    assert drop.after_rate == pytest.approx(0.4)
    assert "trafność spadła istotnie od 2026-07-01" in drop.describe_pl()


def test_accuracy_drop_is_none_for_a_stable_record() -> None:
    calls: list[ScoredCall] = []
    for day in _days(0, 60):
        calls += _cycle_calls(day, 5, 5)

    assert detect_accuracy_drop(calls) is None
