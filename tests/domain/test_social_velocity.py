"""Testy domeny social_velocity — czysta logika velocity_ratio / trend.

Bez I/O: sprawdzamy guard dzielenia przez zero (baseline ≤ 0) oraz progi
klasyfikacji trendu (SURGING / NORMAL / QUIET)."""

from __future__ import annotations

import pytest

from src.domain.social_velocity import SocialTrend, SocialVelocitySnapshot


class TestVelocityRatio:
    def test_ratio_is_mentions_over_baseline(self) -> None:
        snap = SocialVelocitySnapshot(
            symbol="GME", mentions_24h=30, baseline_mentions=10.0
        )
        assert snap.velocity_ratio() == 3.0

    def test_baseline_zero_returns_zero_no_division(self) -> None:
        snap = SocialVelocitySnapshot(
            symbol="GME", mentions_24h=30, baseline_mentions=0.0
        )
        assert snap.velocity_ratio() == 0.0

    def test_negative_baseline_returns_zero(self) -> None:
        snap = SocialVelocitySnapshot(
            symbol="GME", mentions_24h=30, baseline_mentions=-5.0
        )
        assert snap.velocity_ratio() == 0.0


class TestTrend:
    def test_ratio_at_or_above_surge_is_surging(self) -> None:
        snap = SocialVelocitySnapshot(
            symbol="GME", mentions_24h=30, baseline_mentions=10.0
        )
        assert snap.trend() is SocialTrend.SURGING

    def test_surge_boundary_is_inclusive(self) -> None:
        # ratio == 2.0 (domyślny surge_ratio) → SURGING (≥, nie >).
        snap = SocialVelocitySnapshot(
            symbol="GME", mentions_24h=20, baseline_mentions=10.0
        )
        assert snap.trend() is SocialTrend.SURGING

    def test_mid_range_is_normal(self) -> None:
        snap = SocialVelocitySnapshot(
            symbol="GME", mentions_24h=10, baseline_mentions=10.0
        )
        assert snap.trend() is SocialTrend.NORMAL

    def test_quiet_boundary_is_inclusive(self) -> None:
        # ratio == 0.5 (domyślny quiet_ratio) → QUIET (≤).
        snap = SocialVelocitySnapshot(
            symbol="GME", mentions_24h=5, baseline_mentions=10.0
        )
        assert snap.trend() is SocialTrend.QUIET

    def test_no_baseline_classified_quiet(self) -> None:
        # ratio 0.0 (brak baseline'u) jest ≤ quiet_ratio → QUIET.
        snap = SocialVelocitySnapshot(
            symbol="GME", mentions_24h=100, baseline_mentions=0.0
        )
        assert snap.trend() is SocialTrend.QUIET


@pytest.mark.parametrize(
    ("mentions", "baseline", "expected"),
    [
        (40, 10.0, SocialTrend.SURGING),
        (12, 10.0, SocialTrend.NORMAL),
        (3, 10.0, SocialTrend.QUIET),
    ],
)
def test_trend_table(
    mentions: int, baseline: float, expected: SocialTrend
) -> None:
    snap = SocialVelocitySnapshot(
        symbol="GME", mentions_24h=mentions, baseline_mentions=baseline
    )
    assert snap.trend() is expected
