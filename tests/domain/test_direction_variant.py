"""ML-30: wariant binarny modelu — P(wzrost poza martwą strefą) i jego bramka."""

from __future__ import annotations

import math

import pytest

from src.domain.backtest import MIN_OOS_POINTS
from src.domain.direction_variant import (
    DirectionSummary,
    direction_label,
    summarize_direction_folds,
)


class TestDirectionLabel:
    def test_rise_beyond_the_band_is_up(self) -> None:
        assert direction_label(0.003, dead_band=0.001) == 1

    def test_fall_beyond_the_band_is_down(self) -> None:
        assert direction_label(-0.003, dead_band=0.001) == 0

    @pytest.mark.parametrize("move", [0.0, 0.001, -0.001, 0.0005])
    def test_moves_inside_the_band_have_no_label(self, move: float) -> None:
        assert direction_label(move, dead_band=0.001) is None

    def test_zero_band_drops_only_the_flat_move(self) -> None:
        assert direction_label(0.0, dead_band=0.0) is None
        assert direction_label(1e-9, dead_band=0.0) == 1

    def test_nan_return_has_no_label(self) -> None:
        assert direction_label(math.nan, dead_band=0.0) is None

    def test_negative_band_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="dead_band"):
            direction_label(0.01, dead_band=-0.001)


def _fold(n: int, sq: float, base_sq: float, ll: float, base_ll: float) -> dict[str, float]:
    return {
        "n_valid": float(n),
        "sum_sq_error": sq,
        "base_sum_sq_error": base_sq,
        "sum_log_loss": ll,
        "base_sum_log_loss": base_ll,
    }


class TestSummarizeDirectionFolds:
    def test_pooled_scores_weigh_every_out_of_sample_point(self) -> None:
        summary = summarize_direction_folds(
            [_fold(40, 8.0, 10.0, 24.0, 27.0), _fold(20, 4.0, 5.0, 12.0, 13.0)]
        )

        assert summary.n_folds == 2
        assert summary.n_oos_points == 60
        assert summary.brier == pytest.approx(12.0 / 60)
        assert summary.base_rate_brier == pytest.approx(15.0 / 60)
        assert summary.log_loss == pytest.approx(36.0 / 60)
        assert summary.base_rate_log_loss == pytest.approx(40.0 / 60)
        assert summary.skill == pytest.approx(1 - 12.0 / 15.0)

    def test_beating_the_base_rate_by_the_margin_passes(self) -> None:
        summary = summarize_direction_folds([_fold(MIN_OOS_POINTS, 9.0, 10.0, 1.0, 1.0)])

        assert summary.beats_base_rate is True
        assert "bije stopę bazową" in summary.verdict

    def test_a_skill_below_the_margin_fails(self) -> None:
        summary = summarize_direction_folds([_fold(MIN_OOS_POINTS, 9.9, 10.0, 1.0, 1.0)])

        assert summary.beats_base_rate is False
        assert "nie bije" in summary.verdict

    def test_too_few_out_of_sample_points_fail(self) -> None:
        summary = summarize_direction_folds([_fold(MIN_OOS_POINTS - 1, 1.0, 10.0, 1.0, 1.0)])

        assert summary.beats_base_rate is False
        assert "za mało punktów OOS" in summary.verdict

    def test_a_perfect_base_rate_cannot_be_beaten(self) -> None:
        summary = summarize_direction_folds([_fold(MIN_OOS_POINTS, 0.0, 0.0, 0.0, 0.0)])

        assert summary.skill is None
        assert summary.beats_base_rate is False

    def test_no_folds_is_no_data(self) -> None:
        assert summarize_direction_folds([]) == DirectionSummary(
            n_folds=0,
            n_oos_points=0,
            brier=None,
            base_rate_brier=None,
            log_loss=None,
            base_rate_log_loss=None,
            beats_base_rate=False,
            verdict="brak danych",
        )
