from datetime import datetime, timedelta

from src.domain.model_scorecard import (
    METRIC_NOTE,
    MODEL_STALE_AFTER_DAYS,
    POOLED_MODEL_SYMBOL,
    RUN_MAX_GAP,
    RunSummary,
    TrainingOutcome,
    classify_status,
    improvement_pct,
    is_stale,
    latest_run_indices,
    latest_run_indices_by_id,
    summarize_evaluations,
    summarize_run,
)


class TestImprovementPct:
    """Procentowa redukcja RMSE kandydata względem baseline'u.

    Wzór reużyty z `src/domain/backtest.py`:
    `(baseline - candidate) / baseline * 100` (dodatnia = model lepszy).
    """

    def test_positive_when_candidate_beats_baseline(self) -> None:
        # baseline=0.30, candidate=0.20 → (0.30-0.20)/0.30*100 ≈ 33.33%
        assert abs(improvement_pct(0.20, 0.30) - (0.10 / 0.30 * 100)) < 1e-9

    def test_negative_when_candidate_worse_than_baseline(self) -> None:
        # kandydat gorszy → poprawa ujemna (brutalna prawda, nie udajemy sukcesu)
        assert improvement_pct(0.40, 0.30) < 0.0

    def test_zero_when_baseline_is_zero(self) -> None:
        # zabezpieczenie dzielenia przez zero — sentinel 0.0 jak w backtest.py
        assert improvement_pct(0.20, 0.0) == 0.0

    def test_matches_backtest_convention(self) -> None:
        from src.domain.backtest import summarize_folds

        folds = [{"candidate_rmse": 0.20, "baseline_rmse": 0.30, "hit_rate": 0.5}]
        summary = summarize_folds(folds)
        assert abs(improvement_pct(0.20, 0.30) - summary.improvement_pct) < 1e-9


class TestStaleness:
    """Reguła świeżości modelu: starszy niż 21 dni → STALE."""

    def test_threshold_constant_is_21_days(self) -> None:
        assert MODEL_STALE_AFTER_DAYS == 21

    def test_fresh_model_is_not_stale(self) -> None:
        now = datetime(2026, 7, 10, 12, 0, 0)
        trained_at = now - timedelta(days=5)
        assert is_stale(trained_at, now) is False

    def test_exactly_21_days_is_not_stale(self) -> None:
        # granica: „starszy NIŻ 21 dni" → dokładnie 21 dni jeszcze świeży
        now = datetime(2026, 7, 10, 12, 0, 0)
        trained_at = now - timedelta(days=21)
        assert is_stale(trained_at, now) is False

    def test_older_than_21_days_is_stale(self) -> None:
        now = datetime(2026, 7, 10, 12, 0, 0)
        trained_at = now - timedelta(days=21, seconds=1)
        assert is_stale(trained_at, now) is True


class TestClassifyStatus:
    """Mapowanie stringa `status` z `train()` na domenowy `TrainingOutcome`."""

    def test_trained_successfully_is_accepted(self) -> None:
        assert classify_status("trained_successfully") is TrainingOutcome.ACCEPTED

    def test_validation_failed_is_rejected(self) -> None:
        assert (
            classify_status("skipped_validation_failed") is TrainingOutcome.REJECTED
        )

    def test_unknown_status_is_skipped(self) -> None:
        # np. brak próbek (ValueError w adapterze) → orkiestrator melduje skipped
        assert classify_status("skipped_insufficient_samples") is TrainingOutcome.SKIPPED
        assert classify_status("") is TrainingOutcome.SKIPPED


class TestSummarizeRun:
    """Agregacja per-symbol przebiegu treningu (~43 przebiegi) w jeden obraz."""

    def test_counts_mixed_outcomes(self) -> None:
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

    def test_all_rejected_does_not_fake_success(self) -> None:
        # NAJWIĘKSZE RYZYKO: run, w którym wszystko odrzucone, MUSI to pokazać.
        outcomes = [TrainingOutcome.REJECTED, TrainingOutcome.REJECTED]
        summary = summarize_run(outcomes)
        assert summary.accepted == 0
        assert summary.rejected == 2
        assert summary.total == 2

    def test_empty_run(self) -> None:
        summary = summarize_run([])
        assert summary.total == 0
        assert summary.accepted == 0
        assert summary.rejected == 0
        assert summary.skipped == 0

    def test_summary_line_reports_counts_in_polish(self) -> None:
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

    def test_run_summary_is_frozen(self) -> None:
        import dataclasses

        import pytest

        summary = summarize_run([TrainingOutcome.ACCEPTED])
        with pytest.raises(dataclasses.FrozenInstanceError):
            summary.accepted = 99  # type: ignore[misc]

    def test_summarize_run_accepts_status_strings(self) -> None:
        # wygoda: agregacja przyjmuje też surowe stringi `status` z train()
        summary = summarize_run(
            ["trained_successfully", "skipped_validation_failed", "boom"]
        )
        assert summary.accepted == 1
        assert summary.rejected == 1
        assert summary.skipped == 1


class TestMetricNote:
    """Zastrzeżenie do copy sekcji „trust" — bez niego byłaby nieuczciwa."""

    def test_note_is_polish_and_mentions_candidate_and_refit(self) -> None:
        assert "KANDYDATA" in METRIC_NOTE or "kandydata" in METRIC_NOTE
        assert "walk-forward" in METRIC_NOTE
        assert "refit" in METRIC_NOTE.lower()


class TestLatestRunIndices:
    """ML-03: „przebieg" to wiersze najnowszego treningu, nie całe okno 30 dni —
    inaczej karta sklejała 4–5 tygodniowych przebiegów w jeden."""

    def test_cuts_at_the_first_gap_longer_than_two_hours(self) -> None:
        latest = datetime(2026, 10, 11, 3, 30)
        stamps = [
            latest,
            latest - timedelta(minutes=20),
            latest - timedelta(days=7),
            latest - timedelta(days=7, minutes=10),
        ]

        assert latest_run_indices(stamps) == [0, 1]
        assert timedelta(hours=2) == RUN_MAX_GAP

    def test_order_of_input_rows_does_not_matter(self) -> None:
        latest = datetime(2026, 10, 11, 3, 30)
        stamps = [latest - timedelta(days=7), latest, latest - timedelta(minutes=5)]

        assert latest_run_indices(stamps) == [1, 2]

    def test_gap_is_measured_between_neighbouring_rows(self) -> None:
        # Długi przebieg (45 symboli) nie jest cięty, póki kolejne wiersze dzieli < 2h.
        latest = datetime(2026, 10, 11, 6, 0)
        stamps = [latest - timedelta(minutes=90 * i) for i in range(4)]

        assert latest_run_indices(stamps) == [0, 1, 2, 3]

    def test_rows_without_timestamp_are_left_out(self) -> None:
        latest = datetime(2026, 10, 11, 3, 30)

        assert latest_run_indices([None, latest]) == [1]
        assert latest_run_indices([None]) == []


class TestLatestRunIndicesById:
    """ML-24: wiersze z `run_id` grupuje identyfikator przebiegu, nie przerwa
    czasowa — ręczny re-run kilka minut po cronie to osobny przebieg."""

    def test_rerun_minutes_after_the_cron_is_a_separate_run(self) -> None:
        cron = datetime(2026, 10, 11, 3, 30)
        stamps = [cron + timedelta(minutes=40), cron + timedelta(minutes=35), cron, cron]
        run_ids = ["rerun", "rerun", "cron", "cron"]

        assert latest_run_indices_by_id(stamps, run_ids) == [0, 1]

    def test_newest_row_without_run_id_falls_back_to_the_time_gap(self) -> None:
        # Historia sprzed migracji 049 (albo flaga off) — heurystyka ML-03.
        latest = datetime(2026, 10, 11, 3, 30)
        stamps = [latest, latest - timedelta(minutes=5), latest - timedelta(days=7)]

        assert latest_run_indices_by_id(stamps, [None, None, "older"]) == [0, 1]

    def test_rows_without_timestamp_are_left_out(self) -> None:
        latest = datetime(2026, 10, 11, 3, 30)

        assert latest_run_indices_by_id([None, latest], ["run", "run"]) == [1]
        assert latest_run_indices_by_id([None], [None]) == []


class TestSummarizeEvaluations:
    """Decyzja 3 (2026-10-18): ewaluacje per symbol nigdy nie zapisują wag (ML-13),
    więc karta liczy je osobno od werdyktu modelu wspólnego."""

    def test_counts_passed_failed_and_skipped_evaluations(self) -> None:
        summary = summarize_evaluations(
            ["evaluated_passed_gate", "evaluated_passed_gate", "evaluated_failed_gate", "skipped"]
        )

        assert (summary.total, summary.passed, summary.failed, summary.skipped) == (4, 2, 1, 1)

    def test_summary_line_reports_counts_in_polish(self) -> None:
        summary = summarize_evaluations(["evaluated_passed_gate", "evaluated_failed_gate"])

        assert summary.summary_line == "bramkę przeszło 1, nie przeszło 1, pominięto 0"

    def test_empty_evaluations(self) -> None:
        summary = summarize_evaluations([])

        assert (summary.total, summary.passed, summary.failed, summary.skipped) == (0, 0, 0, 0)

    def test_pooled_symbol_is_shared_with_the_trainer(self) -> None:
        from src.application.use_cases.train_model import (
            POOLED_MODEL_SYMBOL as TRAINER_POOLED_SYMBOL,
        )

        assert POOLED_MODEL_SYMBOL == TRAINER_POOLED_SYMBOL == "ALL"
