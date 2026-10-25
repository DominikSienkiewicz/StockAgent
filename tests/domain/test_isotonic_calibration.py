"""ML-28: kalibracja izotoniczna (PAV) z temporalną oceną poza próbą."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from src.domain.isotonic_calibration import (
    ISOTONIC_EVAL_WINDOW,
    ISOTONIC_FIT_WINDOW,
    MIN_CLASS_ISOTONIC_SAMPLES,
    MIN_ISOTONIC_SAMPLES,
    CalibrationClass,
    IsotonicMap,
    fit_isotonic,
    select_class_isotonic_calibrators,
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


def _classed(
    rows: list[tuple[datetime, float, bool]], calibration_class: CalibrationClass
) -> list[tuple[datetime, float, bool, CalibrationClass]]:
    return [(stamp, confidence, hit, calibration_class) for stamp, confidence, hit in rows]


def _class_history(
    calibration_class: CalibrationClass, hits_at_06: int, hits_at_09: int, per_level: int = 20
) -> list[tuple[datetime, float, bool, CalibrationClass]]:
    """Ten sam wzór w obu oknach: `per_level` predykcji na poziom pewności 0.6 i 0.9."""
    rows: list[tuple[datetime, float, bool]] = []
    for age in (timedelta(days=45), timedelta(days=10)):
        rows += _timed(age, 0.6, hits_at_06, per_level - hits_at_06)
        rows += _timed(age, 0.9, hits_at_09, per_level - hits_at_09)
    return _classed(rows, calibration_class)


def _older(
    history: list[tuple[datetime, float, bool, CalibrationClass]],
) -> list[tuple[float, bool]]:
    return [(c, hit) for stamp, c, hit, _ in history if NOW - stamp > timedelta(days=30)]


class TestSelectClassIsotonicCalibrators:
    """ML-54: osobna mapa dla krypto i dla reszty, gdy klasa ma dość próbek."""

    def test_class_threshold_is_the_per_window_minimum_of_the_global_map(self) -> None:
        assert MIN_CLASS_ISOTONIC_SAMPLES == MIN_ISOTONIC_SAMPLES

    def test_selects_per_class_map_when_class_has_enough_samples(self) -> None:
        # 40 próbek na okno w każdej klasie, krzywe na różnych poziomach: krypto
        # trafia rzadziej przy każdej pewności niż akcje i ETF-y.
        crypto = _class_history(CalibrationClass.CRYPTO, hits_at_06=8, hits_at_09=7)
        rest = _class_history(CalibrationClass.NON_CRYPTO, hits_at_06=12, hits_at_09=11)

        selection = select_class_isotonic_calibrators([*crypto, *rest], now=NOW)

        crypto_map = selection.calibrator_for(CalibrationClass.CRYPTO)
        rest_map = selection.calibrator_for(CalibrationClass.NON_CRYPTO)
        assert crypto_map == fit_isotonic(_older(crypto))
        assert rest_map == fit_isotonic(_older(rest))
        assert crypto_map != rest_map
        assert crypto_map != selection.global_selection.calibrator

    def test_class_below_threshold_falls_back_to_the_global_map(self) -> None:
        # 5 próbek krypto na okno — za mało na własną mapę i jej ocenę poza próbą.
        crypto = _class_history(CalibrationClass.CRYPTO, hits_at_06=3, hits_at_09=2, per_level=5)
        rest = _class_history(CalibrationClass.NON_CRYPTO, hits_at_06=12, hits_at_09=11)

        selection = select_class_isotonic_calibrators([*crypto, *rest], now=NOW)

        global_map = selection.global_selection.calibrator
        assert global_map is not None
        assert selection.calibrator_for(CalibrationClass.CRYPTO) == global_map
        assert CalibrationClass.CRYPTO not in selection.class_selections
        assert selection.calibrator_for(CalibrationClass.NON_CRYPTO) == fit_isotonic(_older(rest))

    def test_class_map_losing_out_of_sample_keeps_the_buckets_for_that_class(self) -> None:
        # Krypto ma dość próbek, ale starsze okno uczy odwrotnego porządku niż ostatni
        # miesiąc — w tej klasie mapa przegrywa, więc nie wchodzi ani ona, ani globalna.
        crypto = _classed(
            [
                *_timed(timedelta(days=45), 0.6, 18, 2),
                *_timed(timedelta(days=45), 0.9, 2, 18),
                *_timed(timedelta(days=10), 0.6, 2, 18),
                *_timed(timedelta(days=10), 0.9, 18, 2),
            ],
            CalibrationClass.CRYPTO,
        )
        rest = _class_history(CalibrationClass.NON_CRYPTO, hits_at_06=12, hits_at_09=11)

        selection = select_class_isotonic_calibrators([*crypto, *rest], now=NOW)

        assert selection.calibrator_for(CalibrationClass.CRYPTO) is None
        assert selection.global_selection.calibrator is not None
        assert selection.class_selections[CalibrationClass.CRYPTO].use_isotonic is False

    def test_unknown_class_uses_the_global_map(self) -> None:
        rest = _class_history(CalibrationClass.NON_CRYPTO, hits_at_06=12, hits_at_09=11)

        selection = select_class_isotonic_calibrators(rest, now=NOW)

        assert selection.calibrator_for(None) == selection.global_selection.calibrator

    def test_global_map_is_the_ml28_selection_over_all_classes(self) -> None:
        crypto = _class_history(CalibrationClass.CRYPTO, hits_at_06=8, hits_at_09=7)
        rest = _class_history(CalibrationClass.NON_CRYPTO, hits_at_06=12, hits_at_09=11)
        unclassed = [(stamp, c, hit) for stamp, c, hit, _ in [*crypto, *rest]]

        selection = select_class_isotonic_calibrators([*crypto, *rest], now=NOW)

        assert selection.global_selection == select_isotonic_calibrator(unclassed, now=NOW)
