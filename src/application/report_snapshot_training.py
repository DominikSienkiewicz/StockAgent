"""Summary Slow Loopa: modele z `feature_snapshots` i raport selekcji bramki (ML-52).

Dopisywane do step summary GitHub Actions (NIEZ-22) pod tabelą treningu. Czysta
prezentacja — reguły (bramka walk-forward, etykieta, rozkłady) żyją w domenie.
Treść po angielsku jak reszta summary Slow Loopa.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.application.ml_features import SNAPSHOT_TRAINING_FEATURE_SET_VERSION
from src.application.use_cases.train_model import SnapshotModelsResult
from src.domain.backtest import strongest_baseline
from src.domain.model_scorecard import TrainingReport
from src.domain.training_labels import ReturnDistribution

_NOT_AVAILABLE = "n/a"


def _number(value: float | None) -> str:
    return _NOT_AVAILABLE if value is None else f"{value:.4f}"


def _baseline(report: TrainingReport) -> str:
    """Najsilniejszy baseline walk-forward karty (najniższe RMSE) z etykietą."""
    if report.baseline_rmse is None:
        return _NOT_AVAILABLE
    label, rmse = strongest_baseline(
        report.baseline_rmse, report.drift_baseline_rmse, report.llm_baseline_rmse
    )
    return f"{_number(rmse)} ({label})"


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _model_row(name: str, report: TrainingReport) -> str:
    return _row(
        (name, report.status, str(report.n_samples), _number(report.rmse), _baseline(report))
    )


def _distribution_row(name: str, distribution: ReturnDistribution) -> str:
    stats = (
        distribution.mean,
        distribution.median,
        distribution.std,
        distribution.mean_abs,
        distribution.p10,
        distribution.p90,
    )
    return _row((name, str(distribution.n), *(_number(value) for value in stats)))


def render_snapshot_models_markdown(result: SnapshotModelsResult) -> str:
    """Dwie karty modeli z wektorów cykli i rozkład zwrotów etykiety gated vs non-gated."""
    ratio = result.selection.abs_return_ratio
    return "\n".join(
        [
            "### Feature snapshot models (ML-52)",
            "",
            _row(("Model", "Status", "Samples", "Candidate RMSE", "Strongest baseline RMSE")),
            _row(("---",) * 5),
            _model_row("Free features (all cycles)", result.free_model),
            _model_row("LLM features (gated cycles)", result.llm_model),
            "",
            f"Feature vectors (set version {SNAPSHOT_TRAINING_FEATURE_SET_VERSION}): "
            f"{result.n_snapshots}, labeled: {result.n_labeled}.",
            "",
            "### Gate selection (12h label returns)",
            "",
            _row(("Group", "n", "Mean", "Median", "Std", "Mean abs", "P10", "P90")),
            _row(("---",) * 8),
            _distribution_row("Gated", result.selection.gated),
            _distribution_row("Non-gated", result.selection.non_gated),
            "",
            "Mean absolute return, gated vs non-gated: "
            + (_NOT_AVAILABLE if ratio is None else f"{ratio:.2f}x"),
            "",
        ]
    )
