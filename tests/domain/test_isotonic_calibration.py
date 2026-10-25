"""ML-28: kalibracja izotoniczna (PAV) z temporalną oceną poza próbą."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from src.domain.isotonic_calibration import (
    ISOTONIC_EVAL_WINDOW,
    ISOTONIC_FIT_WINDOW,
    MIN_ISOTONIC_SAMPLES,
    IsotonicMap,
    fit_isotonic,
    select_isotonic_calibrator,
)

NOW = datetime(2026, 10, 25, 5, 30, tzinfo=UTC)


class TestFitIsotonic:
    def test_isotonic_fit_reproduces_pav_example(self) -> None:
        fitted = fit_isotonic([(0.6, False), (0.7, True), (0.8, False), (0.9, True)])

        assert fitted is not None
        assert [fitted.apply(c) for c in (0.6, 0.7, 0.8, 0.9)] == pytest.approx(
            [0.0, 0.5, 0.5, 1.0]
        )

    def test_isotonic_map_is_monotone_non_decreasing(self) -> None:
        # Kubełek 0.8 z gorszą trafnością niż 0.7 — shrinkage kubełkowy odwracał
        # porządek, PAV łączy oba w jeden poziom.
        samples = [
            *[(0.7, hit) for hit in (True,) * 7 + (False,) * 3],
            *[(0.8, hit) for hit in (True,) * 5 + (False,) * 5],
            *[(0.6, hit) for hit in (True,) * 4 + (False,) * 6],
            *[(0.9, hit) for hit in (True,) * 9 + (False,)],
        ]

        fitted = fit_isotonic(samples)

        assert fitted is not None
        grid = [i / 100 for i in range(101)]
        values = [fitted.apply(c) for c in grid]
        assert values == sorted(values)
        assert fitted.apply(0.7) == pytest.approx(fitted.apply(0.8)) == pytest.approx(0.6)

    def test_ties_in_confidence_pool_before_pav(self) -> None:
        fitted = fit_isotonic([(0.5, True), (0.5, False), (0.5, True), (0.9, True)])

        assert fitted == IsotonicMap(knots=(0.5, 0.9), values=(2 / 3, 1.0))

    def test_between_knots_the_map_interpolates_and_outside_it_holds(self) -> None:
        fitted = IsotonicMap(knots=(0.6, 0.8), values=(0.4, 0.8))

        assert fitted.apply(0.7) == pytest.approx(0.6)
        assert fitted.apply(0.1) == pytest.approx(0.4)
        assert fitted.apply(0.95) == pytest.approx(0.8)
        assert math.isnan(fitted.apply(math.nan))

    def test_out_of_range_confidences_are_dropped(self) -> None:
        fitted = fit_isotonic([(85.0, True), (math.nan, False), (0.4, False)])

        assert fitted == IsotonicMap(knots=(0.4,), values=(0.0,))

    def test_no_samples_no_map(self) -> None:
        assert fit_isotonic([]) is None
        assert fit_isotonic([(1.5, True)]) is None


def _timed(
    age: timedelta, confidence: float, hits: int, misses: int
) -> list[tuple[datetime, float, bool]]:
    stamp = NOW - age
    return [(stamp, confidence, True)] * hits + [(stamp, confidence, False)] * misses


def _overconfident_history() -> list[tuple[datetime, float, bool]]:
    """Ten sam wzór w obu oknach: 0.9 trafia rzadziej niż 0.6 — surowa pewność
    i kubełki przegrywają z mapą, która łączy oba poziomy."""
    rows: list[tuple[datetime, float, bool]] = []
    for age in (timedelta(days=45), timedelta(days=10)):
        rows += _timed(age, 0.6, 12, 8)
        rows += _timed(age, 0.9, 11, 9)
    return rows


class TestSelectIsotonicCalibrator:
    def test_windows_are_fit_on_older_and_scored_on_recent_month(self) -> None:
        assert (timedelta(days=60), timedelta(days=30)) == ISOTONIC_FIT_WINDOW
        assert (timedelta(days=30), timedelta(0)) == ISOTONIC_EVAL_WINDOW

    def test_isotonic_wins_out_of_sample_and_enters_the_ranking(self) -> None:
        selection = select_isotonic_calibrator(_overconfident_history(), now=NOW)

        assert selection.use_isotonic is True
        assert selection.calibrator is not None
        assert selection.brier_isotonic is not None
        assert selection.brier_isotonic < min(
            selection.brier_raw or 1.0, selection.brier_buckets or 1.0
        )
        assert (selection.n_fit, selection.n_eval) == (40, 40)

    def test_calibrator_in_the_ranking_is_the_one_scored_out_of_sample(self) -> None:
        history = _overconfident_history()
        older = [(c, hit) for stamp, c, hit in history if NOW - stamp > timedelta(days=30)]

        selection = select_isotonic_calibrator(history, now=NOW)

        assert selection.calibrator == fit_isotonic(older)

    def test_isotonic_losing_out_of_sample_stays_out(self) -> None:
        # Starsze okno uczy odwrotnego porządku niż ostatni miesiąc — poza próbą
        # mapa przegrywa z surową pewnością.
        history = [
            *_timed(timedelta(days=45), 0.6, 18, 2),
            *_timed(timedelta(days=45), 0.9, 2, 18),
            *_timed(timedelta(days=10), 0.6, 2, 18),
            *_timed(timedelta(days=10), 0.9, 18, 2),
        ]

        selection = select_isotonic_calibrator(history, now=NOW)

        assert selection.use_isotonic is False
        assert selection.calibrator is None
        assert "poza próbą" in selection.reason

    def test_too_few_samples_in_a_window_keep_the_buckets(self) -> None:
        history = [
            *_timed(timedelta(days=45), 0.6, MIN_ISOTONIC_SAMPLES - 1, 0),
            *_timed(timedelta(days=10), 0.6, 40, 0),
        ]

        selection = select_isotonic_calibrator(history, now=NOW)

        assert selection.use_isotonic is False
        assert selection.brier_isotonic is None
        assert "za mało" in selection.reason

    def test_rows_outside_both_windows_are_ignored(self) -> None:
        history = [*_overconfident_history(), *_timed(timedelta(days=90), 0.6, 50, 0)]

        selection = select_isotonic_calibrator(history, now=NOW)

        assert selection.n_fit == 40
