"""ML-21 — dryf rozkładu cech: biny kwantylowe i Population Stability Index."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.domain.drift import (
    MIN_PSI_SAMPLES,
    PSI_DRIFT_THRESHOLD,
    FeatureDrift,
    bin_fractions,
    drifted_features,
    population_stability_index,
    quantile_edges,
    recent_window_psi,
)

_START = datetime(2026, 7, 1, tzinfo=UTC)


def _daily(values: list[float], start: datetime = _START) -> list[datetime]:
    return [start + timedelta(days=i) for i in range(len(values))]


def test_psi_zero_for_identical_bins() -> None:
    fractions = [0.1] * 10

    assert population_stability_index(fractions, fractions) == 0.0


def test_psi_above_threshold_for_shifted_distribution() -> None:
    # 60 dni „starych” wartości z [0, 1) i 30 dni przesuniętych o +0.5.
    old = [i / 60 for i in range(60)]
    recent = [0.5 + i / 30 for i in range(30)]
    stamps = _daily(old + recent)

    psi = recent_window_psi(old + recent, stamps)

    assert psi is not None
    assert psi > PSI_DRIFT_THRESHOLD


def test_psi_stays_low_when_the_recent_window_has_the_same_distribution() -> None:
    old = [(i * 7) % 60 / 60 for i in range(60)]
    recent = [(i * 7) % 30 / 30 for i in range(30)]
    stamps = _daily(old + recent)

    psi = recent_window_psi(old + recent, stamps)

    assert psi is not None
    assert psi < 0.1


def test_recent_window_psi_needs_enough_samples_on_both_sides() -> None:
    values = [float(i) for i in range(MIN_PSI_SAMPLES + 5)]
    # Wszystkie wartości w ostatnich 30 dniach — brak części odniesienia.
    stamps = [_START + timedelta(hours=i) for i in range(len(values))]

    assert recent_window_psi(values, stamps) is None


def test_quantile_edges_deduplicate_a_discrete_feature() -> None:
    signal = [-1.0] * 30 + [0.0] * 40 + [1.0] * 30

    edges = quantile_edges(signal)

    assert edges == (-1.0, 0.0, 1.0)
    assert bin_fractions(signal, edges) == pytest.approx((0.3, 0.4, 0.3, 0.0))


def test_bin_fractions_sum_to_one_and_ignore_non_finite_values() -> None:
    edges = (1.0, 2.0)

    fractions = bin_fractions([0.5, 1.5, 2.5, 3.0, float("nan")], edges)

    assert fractions == pytest.approx((0.25, 0.25, 0.5))


def test_drifted_features_keep_only_psi_above_the_threshold_largest_first() -> None:
    drifts = drifted_features(
        {"price_delta": 0.31, "news_volume_24h": 0.9, "av_sentiment_score": 0.1, "x": None}
    )

    assert drifts == (
        FeatureDrift("news_volume_24h", 0.9),
        FeatureDrift("price_delta", 0.31),
    )


def test_psi_rejects_bins_of_different_length() -> None:
    with pytest.raises(ValueError):
        population_stability_index([0.5, 0.5], [1.0])
