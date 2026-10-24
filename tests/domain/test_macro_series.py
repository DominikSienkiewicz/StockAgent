"""DAT-22 — zmienne stanu z FRED: percentyl VIX, zmiana HY OAS, cechy i głosy."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.domain.macro_risk import MacroAlertLevel
from src.domain.macro_series import (
    MACRO_FEATURE_NAMES,
    MIN_VIX_OBSERVATIONS,
    MacroSeriesSnapshot,
    SeriesObservation,
    credit_spread_alert_level,
    hy_oas_change,
    macro_features,
    macro_series_votes,
    vix_percentile,
)

_DAY = date(2026, 10, 20)


def _window(values: list[float], end: date = _DAY) -> tuple[SeriesObservation, ...]:
    start = end - timedelta(days=len(values) - 1)
    return tuple(SeriesObservation(start + timedelta(days=i), v) for i, v in enumerate(values))


def test_vix_percentile_is_the_share_of_the_year_at_or_below_the_latest() -> None:
    # 1…99 i ostatnia obserwacja 50: ≤ 50 są 1…50 i ona sama — 51 ze 100.
    window = _window([float(v) for v in range(1, 100)] + [50.0])

    assert vix_percentile(window) == pytest.approx(51.0)


def test_vix_percentile_needs_enough_observations() -> None:
    assert vix_percentile(_window([20.0] * (MIN_VIX_OBSERVATIONS - 1))) is None


def test_hy_oas_change_compares_with_five_sessions_back() -> None:
    window = _window([3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6])

    assert hy_oas_change(window) == pytest.approx(0.5)
    assert hy_oas_change(window[:5]) is None


def test_credit_spread_bands() -> None:
    assert credit_spread_alert_level(0.5) is MacroAlertLevel.CRITICAL
    assert credit_spread_alert_level(0.25) is MacroAlertLevel.ELEVATED
    assert credit_spread_alert_level(-0.4) is MacroAlertLevel.NORMAL


def test_votes_skip_series_without_enough_data() -> None:
    calm_year = _window([15.0 + (i % 10) for i in range(200)] + [30.0])
    snapshot = MacroSeriesSnapshot(vix=calm_year, hy_oas=_window([3.0, 3.1]))

    assert macro_series_votes(snapshot) == (MacroAlertLevel.CRITICAL,)
    assert macro_series_votes(MacroSeriesSnapshot()) == ()


def test_features_carry_levels_and_observation_dates() -> None:
    nfci_week = date(2026, 10, 16)
    snapshot = MacroSeriesSnapshot(
        vix=_window([18.0] * 100),
        hy_oas=_window([3.0, 3.0, 3.0, 3.0, 3.0, 3.2]),
        vxn=SeriesObservation(_DAY, 22.5),
        term_spread=SeriesObservation(_DAY, 0.55),
        nfci=SeriesObservation(nfci_week, -0.48),
    )

    features = macro_features(snapshot)

    assert set(features) == set(MACRO_FEATURE_NAMES)
    assert features["vix_level"] == 18.0
    assert features["vix_percentile_1y"] == pytest.approx(100.0)
    assert features["hy_oas_change_5d"] == pytest.approx(0.2)
    assert features["nfci"] == -0.48
    assert features["macro_observed_on"]["NFCI"] == "2026-10-16"
    assert features["macro_observed_on"]["VIXCLS"] == "2026-10-20"


def test_empty_snapshot_gives_null_features() -> None:
    features = macro_features(MacroSeriesSnapshot())

    assert MacroSeriesSnapshot().is_empty
    assert all(features[name] is None for name in MACRO_FEATURE_NAMES[:-1])
    assert features["macro_observed_on"] == {}
