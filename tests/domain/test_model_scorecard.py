from datetime import datetime, timedelta

from src.domain.model_scorecard import (
    METRIC_NOTE,
    MODEL_STALE_AFTER_DAYS,
    RunSummary,
    TrainingOutcome,
    classify_status,
    improvement_pct,
    is_stale,
    summarize_run,
)


class TestImprovementPct:
    """Procentowa redukcja RMSE kandydata względem baseline'u.

    Wzór reużyty z `src/domain/backtest.py`:
    `(baseline - candidate) / baseline * 100` (dodatnia = model lepszy).
    """

    def test_positive_when_candidate_beats_baseline(self):
        # baseline=0.30, candidate=0.20 → (0.30-0.20)/0.30*100 ≈ 33.33%
        assert abs(improvement_pct(0.20, 0.30) - (0.10 / 0.30 * 100)) < 1e-9

    def test_negative_when_candidate_worse_than_baseline(self):
        # kandydat gorszy → poprawa ujemna (brutalna prawda, nie udajemy sukcesu)
        assert improvement_pct(0.40, 0.30) < 0.0

    def test_zero_when_baseline_is_zero(self):
        # zabezpieczenie dzielenia przez zero — sentinel 0.0 jak w backtest.py
        assert improvement_pct(0.20, 0.0) == 0.0

    def test_matches_backtest_convention(self):
        from src.domain.backtest import summarize_folds

        folds = [{"candidate_rmse": 0.20, "baseline_rmse": 0.30, "hit_rate": 0.5}]
        summary = summarize_folds(folds)
        assert abs(improvement_pct(0.20, 0.30) - summary.improvement_pct) < 1e-9


class TestStaleness:
    """Reguła świeżości modelu: starszy niż 21 dni → STALE."""

    def test_threshold_constant_is_21_days(self):
        assert MODEL_STALE_AFTER_DAYS == 21

    def test_fresh_model_is_not_stale(self):
        now = datetime(2026, 7, 10, 12, 0, 0)
        trained_at = now - timedelta(days=5)
        assert is_stale(trained_at, now) is False

    def test_exactly_21_days_is_not_stale(self):
        # granica: „starszy NIŻ 21 dni" → dokładnie 21 dni jeszcze świeży
        now = datetime(2026, 7, 10, 12, 0, 0)
        trained_at = now - timedelta(days=21)
        assert is_stale(trained_at, now) is False

    def test_older_than_21_days_is_stale(self):
        now = datetime(2026, 7, 10, 12, 0, 0)
        trained_at = now - timedelta(days=21, seconds=1)
        assert is_stale(trained_at, now) is True


class TestClassifyStatus:
    """Mapowanie stringa `status` z `train()` na domenowy `TrainingOutcome`."""

    def test_trained_successfully_is_accepted(self):
        assert classify_status("trained_successfully") is TrainingOutcome.ACCEPTED

    def test_validation_failed_is_rejected(self):
        assert (
            classify_status("skipped_validation_failed") is TrainingOutcome.REJECTED
        )

    def test_unknown_status_is_skipped(self):
        # np. brak próbek (ValueError w adapterze) → orkiestrator melduje skipped
        assert classify_status("skipped_insufficient_samples") is TrainingOutcome.SKIPPED
        assert classify_status("") is TrainingOutcome.SKIPPED


class TestSummarizeRun:
    """Agregacja per-symbol przebiegu treningu (~43 przebiegi) w jeden obraz."""

    def test_counts_mixed_outcomes(self):
        outcomes = [
            TrainingOutcome.ACCEPTED,
            TrainingOutcome.ACCEPTED,
            TrainingOutcome.REJECTED,
            TrainingOutcome.SKIPPED,
        ]
        summary = summarize_run(outcomes)
        assert isinstance(summary, RunSummary)
        assert summary.total == 4
        assert summary.accepted == 2
        assert summary.rejected == 1
        assert summary.skipped == 1

    def test_all_rejected_does_not_fake_success(self):
        # NAJWIĘKSZE RYZYKO: run, w którym wszystko odrzucone, MUSI to pokazać.
        outcomes = [TrainingOutcome.REJECTED, TrainingOutcome.REJECTED]
        summary = summarize_run(outcomes)
        assert summary.accepted == 0
        assert summary.rejected == 2
        assert summary.total == 2

    def test_empty_run(self):
        summary = summarize_run([])
        assert summary.total == 0
        assert summary.accepted == 0
        assert summary.rejected == 0
        assert summary.skipped == 0

    def test_summary_line_reports_counts_in_polish(self):
        outcomes = [
            TrainingOutcome.ACCEPTED,
            TrainingOutcome.REJECTED,
            TrainingOutcome.REJECTED,
            TrainingOutcome.SKIPPED,
        ]
        summary = summarize_run(outcomes)
        assert "zaakceptowano 1" in summary.summary_line
        assert "odrzucono 2" in summary.summary_line
        assert "pominięto 1" in summary.summary_line

    def test_run_summary_is_frozen(self):
        import dataclasses

        import pytest

        summary = summarize_run([TrainingOutcome.ACCEPTED])
        with pytest.raises(dataclasses.FrozenInstanceError):
            summary.accepted = 99  # type: ignore[misc]

    def test_summarize_run_accepts_status_strings(self):
        # wygoda: agregacja przyjmuje też surowe stringi `status` z train()
        summary = summarize_run(
            ["trained_successfully", "skipped_validation_failed", "boom"]
        )
        assert summary.accepted == 1
        assert summary.rejected == 1
        assert summary.skipped == 1


class TestMetricNote:
    """Zastrzeżenie do copy sekcji „trust" — bez niego byłaby nieuczciwa."""

    def test_note_is_polish_and_mentions_candidate_and_refit(self):
        assert "KANDYDATA" in METRIC_NOTE or "kandydata" in METRIC_NOTE
        assert "walk-forward" in METRIC_NOTE
        assert "refit" in METRIC_NOTE.lower()
