from __future__ import annotations

import math

from src.tools.evaluate import EvalReport, summarize_evaluation


def _row(
    trend: str,
    correct: bool | None,
    price_at: float,
    predicted: float,
    actual: float,
) -> dict:
    return {
        "predicted_trend": trend,
        "is_trend_correct": correct,
        "price_at_prediction": price_at,
        "predicted_target_price": predicted,
        "actual_price_after_12h": actual,
    }


class TestSummarizeEvaluation:
    def test_empty_rows_returns_zeroed_report(self):
        report = summarize_evaluation([])
        assert report.sample_count == 0
        assert report.hit_rate is None
        assert report.model_rmse is None
        assert report.baseline_rmse is None
        assert report.beats_baseline is False

    def test_directional_hit_rate(self):
        rows = [
            _row("BULLISH", True, 100, 105, 106),
            _row("BEARISH", False, 100, 95, 101),
            _row("BULLISH", True, 100, 102, 103),
        ]
        report = summarize_evaluation(rows)
        assert report.sample_count == 3
        assert report.hit_rate == 2 / 3

    def test_model_rmse_and_baseline_rmse(self):
        # 2 rekordy. Model: |pred-actual| = 1 i 1 → RMSE = 1.
        # Baseline (last-value = price_at): |price_at-actual| = 5 i 5 → RMSE = 5.
        rows = [
            _row("BULLISH", True, 100, 104, 105),
            _row("BULLISH", True, 100, 106, 105),
        ]
        report = summarize_evaluation(rows)
        assert math.isclose(report.model_rmse, 1.0, rel_tol=1e-9)
        assert math.isclose(report.baseline_rmse, 5.0, rel_tol=1e-9)
        assert report.beats_baseline is True

    def test_model_worse_than_baseline_does_not_beat(self):
        rows = [
            _row("BULLISH", True, 100, 120, 101),  # model bardzo daleko
            _row("BULLISH", True, 100, 80, 99),
        ]
        report = summarize_evaluation(rows)
        assert report.beats_baseline is False

    def test_skips_rows_with_missing_values_for_rmse(self):
        # Wiersz bez actual nie psuje RMSE (jest pomijany), ale wlicza się do
        # hit-rate jeśli ma is_trend_correct.
        rows = [
            _row("BULLISH", True, 100, 104, 105),
            {"predicted_trend": "BULLISH", "is_trend_correct": True,
             "price_at_prediction": 100, "predicted_target_price": None,
             "actual_price_after_12h": None},
        ]
        report = summarize_evaluation(rows)
        assert report.sample_count == 2
        assert report.hit_rate == 1.0
        # tylko 1 wiersz miał komplet cen → RMSE z jednego punktu
        assert math.isclose(report.model_rmse, 1.0, rel_tol=1e-9)

    def test_report_is_dataclass(self):
        assert isinstance(summarize_evaluation([]), EvalReport)
