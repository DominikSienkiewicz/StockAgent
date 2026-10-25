from src.application.report_snapshot_training import render_snapshot_models_markdown
from src.application.use_cases.train_model import SnapshotModelsResult
from src.domain.model_scorecard import TrainingOutcome, TrainingReport
from src.domain.training_labels import ReturnDistribution, SelectionReport


def _result(
    free: TrainingReport, llm: TrainingReport, selection: SelectionReport
) -> SnapshotModelsResult:
    return SnapshotModelsResult(
        free_model=free, llm_model=llm, selection=selection, n_snapshots=150, n_labeled=140
    )


def test_summary_shows_both_cards_with_their_strongest_baseline() -> None:
    free = TrainingReport(
        TrainingOutcome.EVALUATION_FAILED,
        n_samples=140,
        n_folds=4,
        rmse=0.0123,
        baseline_rmse=0.0130,
        drift_baseline_rmse=0.0119,
    )
    llm = TrainingReport(TrainingOutcome.SKIPPED, n_samples=30, reason="Not enough (30 < 50).")
    selection = SelectionReport(
        gated=ReturnDistribution(
            n=30, mean=0.002, median=0.001, std=0.03, mean_abs=0.025, p10=-0.03, p90=0.04
        ),
        non_gated=ReturnDistribution(
            n=110, mean=0.0, median=0.0, std=0.01, mean_abs=0.01, p10=-0.012, p90=0.011
        ),
    )

    markdown = render_snapshot_models_markdown(_result(free, llm, selection))

    assert "### Feature snapshot models (ML-52)" in markdown
    assert (
        "| Free features (all cycles) | evaluated_failed_gate | 140 | 0.0123 "
        "| 0.0119 (dryf średniej) |"
    ) in markdown
    assert "| LLM features (gated cycles) | skipped | 30 | n/a | n/a |" in markdown
    assert "Feature vectors (set version 4): 150, labeled: 140." in markdown
    assert "| Gated | 30 | 0.0020 | 0.0010 | 0.0300 | 0.0250 | -0.0300 | 0.0400 |" in markdown
    assert "Mean absolute return, gated vs non-gated: 2.50x" in markdown


def test_empty_database_says_there_is_not_enough_data() -> None:
    empty = TrainingReport(TrainingOutcome.SKIPPED, n_samples=0)
    selection = SelectionReport(ReturnDistribution(n=0), ReturnDistribution(n=0))

    markdown = render_snapshot_models_markdown(_result(empty, empty, selection))

    assert "| Non-gated | 0 | n/a | n/a | n/a | n/a | n/a | n/a |" in markdown
    assert "Mean absolute return, gated vs non-gated: n/a" in markdown
