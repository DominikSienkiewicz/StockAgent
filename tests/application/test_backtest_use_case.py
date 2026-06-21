from unittest.mock import Mock

import pandas as pd
import pytest

from src.application.ports import MLPredictionPort, RepositoryPort
from src.application.use_cases.backtest import MIN_BACKTEST_SAMPLES, BacktestUseCase
from src.domain.backtest import BacktestSummary

EXPECTED_ML_FEATURES = [
    "price_delta",
    "av_sentiment_score",
    "av_relevance_avg",
    "news_volume_24h",
    "high_relevance_count",
    "llm_trend_signal",
    "av_llm_agreement",
]


@pytest.fixture
def ml_port() -> Mock:
    return Mock(spec=MLPredictionPort)


@pytest.fixture
def repository_port() -> Mock:
    return Mock(spec=RepositoryPort)


@pytest.fixture
def use_case(ml_port, repository_port) -> BacktestUseCase:
    return BacktestUseCase(ml_port=ml_port, db_port=repository_port)


def _feature_store_rows(n: int) -> list[dict]:
    """Rekordy jak z widoku ml_feature_store — gotowe price_delta + target_return."""
    return [
        {
            "price_delta": (i % 7 - 3) / 100.0,
            "av_sentiment_score": -0.5 + (i % 10) / 10,
            "av_relevance_avg": 0.5 + (i % 5) / 10,
            "news_volume_24h": 1 + (i % 8),
            "high_relevance_count": i % 3,
            "llm_trend_signal": (i % 3) - 1,
            "av_llm_agreement": 0.4 + (i % 6) / 10,
            "target_return": 0.01 * ((i % 5) - 2),
        }
        for i in range(n)
    ]


class TestRun:
    def test_runs_walk_forward_report_and_returns_summary(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = _feature_store_rows(60)
        ml_port.walk_forward_report.return_value = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60},
            {"candidate_rmse": 0.30, "baseline_rmse": 0.40, "hit_rate": 0.80},
        ]

        summary = use_case.run("AAPL")

        ml_port.walk_forward_report.assert_called_once()
        features, target = ml_port.walk_forward_report.call_args.args
        assert isinstance(features, pd.DataFrame)
        assert list(features.columns) == EXPECTED_ML_FEATURES
        assert isinstance(target, pd.Series)
        assert len(features) == len(target) == 60
        # Werdykt pochodzi z summarize_folds nad zwróconymi foldami.
        assert isinstance(summary, BacktestSummary)
        assert summary.n_folds == 2
        assert summary.beats_baseline is True

    def test_too_few_rows_returns_empty_summary_without_calling_ml(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = _feature_store_rows(
            MIN_BACKTEST_SAMPLES - 1
        )

        summary = use_case.run("AAPL")

        ml_port.walk_forward_report.assert_not_called()
        assert summary.n_folds == 0
        assert summary.beats_baseline is False
        # Werdykt sygnalizuje za mało próbek.
        assert "mało" in summary.verdict or "za mało" in summary.verdict.lower()

    def test_empty_rows_returns_empty_summary_without_calling_ml(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = []

        summary = use_case.run("AAPL")

        ml_port.walk_forward_report.assert_not_called()
        assert summary.n_folds == 0

    def test_rows_with_nan_after_dropna_below_minimum_skips_ml(
        self, use_case, repository_port, ml_port
    ):
        # 60 wierszy, ale wszystkie poza 10 mają NaN w target_return → po dropna < 50.
        rows = _feature_store_rows(60)
        for r in rows[10:]:
            r["target_return"] = None  # type: ignore[assignment]
        repository_port.get_feature_store_data.return_value = rows

        summary = use_case.run("AAPL")

        ml_port.walk_forward_report.assert_not_called()
        assert summary.n_folds == 0

    def test_refresh_view_true_refreshes_feature_store(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = []

        use_case.run("AAPL", refresh_view=True)

        repository_port.refresh_feature_store.assert_called_once()

    def test_refresh_view_default_false_does_not_refresh(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = []

        use_case.run("AAPL")

        repository_port.refresh_feature_store.assert_not_called()

    def test_never_raises_on_empty_data(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = []

        # Brak wyjątku = graceful degradacja.
        summary = use_case.run("AAPL")

        assert isinstance(summary, BacktestSummary)
