"""Testy przedziałów ruchu z conformal prediction (ML-34).

Split-conformal: kwantyl skończonej próby ceil((n+1)(1−α))/n z |reszt| okna
kalibracyjnego. Cold-start: predyktor 0, przedział z empirycznych kwantyli
zwrotów (po α/2 na ogon). ACI: α_t korygowane błędem pokrycia. Gwarancja
pokrycia zakłada wymienność, której rynek nie spełnia — ACI łagodzi, nie gwarantuje.
"""

from __future__ import annotations

import pytest

from src.domain.conformal import (
    ACI_GAMMA,
    MIN_CALIBRATION_SIZE,
    NOMINAL_ALPHA,
    CalibrationPoint,
    IntervalCoverage,
    MoveInterval,
    aci_alpha,
    conformal_quantile,
    empirical_return_interval,
    move_interval,
    realized_coverage,
    split_conformal_interval,
)


def _cold(returns: list[float]) -> list[CalibrationPoint]:
    return [CalibrationPoint(predicted_return=0.0, realized_return=r) for r in returns]


def test_split_conformal_uses_finite_sample_quantile() -> None:
    # n = 10, α = 0.2 → k = ceil(11 · 0.8) = 9: dziewiąta reszta, nie naiwny
    # 80. percentyl (ósma) — poprawka skończonej próby daje gwarancję ≥ 1−α.
    residuals = [0.01 * i for i in range(10, 0, -1)]

    assert conformal_quantile(residuals, 0.2) == pytest.approx(0.09)
    low, high = split_conformal_interval(0.005, residuals, 0.2) or (None, None)
    assert low == pytest.approx(0.005 - 0.09)
    assert high == pytest.approx(0.005 + 0.09)


def test_split_conformal_refuses_a_window_too_small_for_alpha() -> None:
    # n = 3 → k = ceil(4 · 0.8) = 4 > n: żadna skończona reszta nie daje 80%.
    assert conformal_quantile([0.01, 0.02, 0.03], 0.2) is None
    assert split_conformal_interval(0.0, [0.01, 0.02, 0.03], 0.2) is None


def test_cold_start_interval_comes_from_empirical_return_quantiles() -> None:
    # n = 19, α = 0.2 → po 0.1 na ogon: dolny = floor(20 · 0.1) = 2. najmniejszy,
    # górny = ceil(20 · 0.9) = 18. najmniejszy. Rozkład skośny → przedział też.
    returns = [-0.05, -0.03] + [0.001 * i for i in range(15)] + [0.04, 0.09]

    low, high = empirical_return_interval(returns, 0.2) or (None, None)

    assert low == pytest.approx(-0.03)
    assert high == pytest.approx(0.04)


def test_aci_widens_after_misses() -> None:
    # α_{t+1} = α_t + γ(α − err_t): pudło (err = 1) obniża α → szerszy przedział.
    after_miss = aci_alpha([False])
    assert after_miss == pytest.approx(NOMINAL_ALPHA + ACI_GAMMA * (NOMINAL_ALPHA - 1))
    assert aci_alpha([False] * 10) < after_miss < NOMINAL_ALPHA
    assert aci_alpha([True] * 10) > NOMINAL_ALPHA


def test_aci_interval_is_wider_than_nominal_after_a_volatility_regime_shift() -> None:
    # Spokojny początek kalibruje wąski przedział, potem rynek przyspiesza:
    # przedziały z przeszłości pudłują, ACI obniża α i poszerza przedział.
    calm = [0.002 * (1 if i % 2 else -1) * (1 + i % 5) for i in range(30)]
    wild = [0.03 * (1 if i % 2 else -1) * (1 + i % 3) for i in range(30)]
    window = _cold(calm + wild)

    adaptive = move_interval(window, predicted_return=0.0, cold_start=True)
    nominal = empirical_return_interval(calm + wild, NOMINAL_ALPHA)

    assert adaptive.low is not None and adaptive.high is not None
    assert nominal is not None
    assert adaptive.alpha < NOMINAL_ALPHA
    assert adaptive.high - adaptive.low > nominal[1] - nominal[0]


def test_coverage_counts_hits_inside_interval() -> None:
    # Granice wliczają się do pokrycia (przedział domknięty).
    coverage = realized_coverage([(-0.02, 0.02, 0.01), (-0.02, 0.02, 0.03), (-0.02, 0.02, -0.02)])

    assert coverage == IntervalCoverage(hits=2, n=3, nominal=1 - NOMINAL_ALPHA)
    assert coverage.rate == pytest.approx(2 / 3)


def test_empty_coverage_has_no_rate() -> None:
    assert realized_coverage([]).rate is None


def test_too_short_window_reports_insufficient_history() -> None:
    window = _cold([0.01] * (MIN_CALIBRATION_SIZE - 1))

    interval = move_interval(window, predicted_return=0.0, cold_start=True)

    assert interval == MoveInterval(
        low=None,
        high=None,
        calibration_size=MIN_CALIBRATION_SIZE - 1,
        required_size=MIN_CALIBRATION_SIZE,
    )
    assert interval.available is False


def test_trained_model_gets_a_symmetric_interval_around_its_forecast() -> None:
    # Model aktywny: ŷ ± kwantyl |reszt|. Stały rozrzut reszt → bez pudeł w ACI,
    # więc α rośnie nieznacznie, a przedział zostaje symetryczny wokół ŷ.
    window = [
        CalibrationPoint(predicted_return=0.01, realized_return=0.01 + 0.002 * (i % 7 - 3))
        for i in range(40)
    ]

    interval = move_interval(window, predicted_return=0.02, cold_start=False)

    assert interval.available
    assert interval.low is not None and interval.high is not None
    assert interval.high - 0.02 == pytest.approx(0.02 - interval.low)
    assert interval.high - 0.02 <= 0.006 + 1e-12


def test_only_the_most_recent_points_calibrate() -> None:
    # Okno kalibracyjne jest kroczące: stare zwroty nie poszerzają przedziału
    # w nieskończoność, a koszt replayu ACI jest ograniczony.
    old_storm = _cold([0.5, -0.5] * 200)
    recent = _cold([0.001 * (i % 9 - 4) for i in range(250)])

    interval = move_interval(old_storm + recent, predicted_return=0.0, cold_start=True)

    assert interval.calibration_size == 250
    assert interval.high is not None and interval.high < 0.01
