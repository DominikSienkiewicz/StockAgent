import pytest

from src.domain.backtest import BacktestSummary, summarize_folds


class TestSummarizeFolds:
    """Agregacja per-fold metryk walk-forward w jeden werdykt backtestu.

    Czysta funkcja: średnie arytmetyczne RMSE/hit-rate, procentowa poprawa
    względem baseline'u (zwrot=0) i binarny werdykt „bije / nie bije baseline".
    """

    def test_means_are_arithmetic_over_folds(self):
        folds = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60},
            {"candidate_rmse": 0.30, "baseline_rmse": 0.40, "hit_rate": 0.80},
        ]

        summary = summarize_folds(folds)

        assert summary.n_folds == 2
        assert abs(summary.mean_candidate_rmse - 0.20) < 1e-9
        assert abs(summary.mean_baseline_rmse - 0.30) < 1e-9
        assert abs(summary.mean_hit_rate - 0.70) < 1e-9

    def test_improvement_pct_relative_to_baseline(self):
        # mean_candidate=0.20, mean_baseline=0.30 → (0.30-0.20)/0.30*100 ≈ 33.33%
        folds = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60},
            {"candidate_rmse": 0.30, "baseline_rmse": 0.40, "hit_rate": 0.80},
        ]

        summary = summarize_folds(folds)

        assert abs(summary.improvement_pct - (10.0 / 30.0 * 100)) < 1e-9

    def test_beats_baseline_true_when_candidate_rmse_lower(self):
        folds = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60},
        ]

        summary = summarize_folds(folds)

        assert summary.beats_baseline is True
        assert "bije baseline" in summary.verdict

    def test_does_not_beat_baseline_when_candidate_rmse_higher(self):
        folds = [
            {"candidate_rmse": 0.40, "baseline_rmse": 0.20, "hit_rate": 0.30},
        ]

        summary = summarize_folds(folds)

        assert summary.beats_baseline is False
        # improvement_pct ujemny: (0.20-0.40)/0.20*100 = -100%
        assert abs(summary.improvement_pct - (-100.0)) < 1e-9
        assert "nie bije baseline" in summary.verdict

    def test_improvement_pct_zero_when_baseline_zero(self):
        # Guard dzielenia przez zero: baseline=0 → improvement_pct=0.0
        folds = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.0, "hit_rate": 0.50},
        ]

        summary = summarize_folds(folds)

        assert summary.improvement_pct == pytest.approx(0.0)
        # candidate 0.10 > baseline 0.0 → nie bije
        assert summary.beats_baseline is False

    def test_verdict_mentions_improvement_percentage_when_beating(self):
        folds = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60},
        ]

        summary = summarize_folds(folds)

        # Werdykt zawiera liczbę procent poprawy (50%).
        assert "50" in summary.verdict

    def test_empty_folds_returns_brak_danych(self):
        summary = summarize_folds([])

        assert summary == BacktestSummary(0, 0, 0, 0, 0, False, "brak danych")
        assert summary.n_folds == 0
        assert summary.beats_baseline is False
        assert summary.verdict == "brak danych"

    def test_summary_is_frozen(self):
        summary = summarize_folds([])

        import dataclasses

        try:
            summary.n_folds = 5  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            pass
        else:  # pragma: no cover
            raise AssertionError("BacktestSummary should be frozen")
