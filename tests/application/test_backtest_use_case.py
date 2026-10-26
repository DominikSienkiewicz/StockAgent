from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from src.application.ml_features import (
    FEATURE_SET_VERSION,
    complete_model_rows,
    rows_in_label_time_order,
)
from src.application.ports import MLPredictionPort, RepositoryPort
from src.application.use_cases.backtest import MIN_BACKTEST_SAMPLES, BacktestUseCase
from src.domain.backtest import BacktestSummary, walk_forward_folds

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
def use_case(ml_port: Mock, repository_port: Mock) -> BacktestUseCase:
    return BacktestUseCase(ml_port=ml_port, db_port=repository_port)


def _feature_store_rows(n: int) -> list[dict[str, Any]]:
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
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = _feature_store_rows(60)
        ml_port.walk_forward_report.return_value = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60},
            {"candidate_rmse": 0.30, "baseline_rmse": 0.40, "hit_rate": 0.80},
        ]

        summary = use_case.run("AAPL")

        ml_port.walk_forward_report.assert_called_once()
        features, target = ml_port.walk_forward_report.call_args.args
        # ML-55: wiersze cech w kolejności kontraktu zamiast kolumn DataFrame.
        assert [list(row) for row in features] == [EXPECTED_ML_FEATURES] * 60
        assert len(features) == len(target) == 60
        # Werdykt pochodzi z summarize_folds nad zwróconymi foldami.
        assert isinstance(summary, BacktestSummary)
        assert summary.n_folds == 2
        assert summary.beats_baseline is True

    def test_walk_forward_report_receives_plain_sequences(
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
        # ML-55: port dostaje wiersze cech w kolejności kontraktu i target jak `train` —
        # bez DataFrame w warstwie application; ramkę składa adapter.
        repository_port.get_feature_store_data.return_value = _stamped_rows(60)[::-1]
        ml_port.walk_forward_report.return_value = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60}
        ]

        use_case.run("AAPL")

        features, target = ml_port.walk_forward_report.call_args.args
        stamps = ml_port.walk_forward_report.call_args.kwargs["timestamps"]
        assert isinstance(features, list)
        assert all(type(row) is dict for row in features)
        assert [list(row) for row in features] == [EXPECTED_ML_FEATURES] * 60
        assert target == [float(hour) for hour in range(60)]
        assert isinstance(stamps, list)
        assert all(isinstance(stamp, datetime) for stamp in stamps)

    def test_too_few_rows_returns_empty_summary_without_calling_ml(
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
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
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = []

        summary = use_case.run("AAPL")

        ml_port.walk_forward_report.assert_not_called()
        assert summary.n_folds == 0

    def test_rows_with_nan_after_dropna_below_minimum_skips_ml(
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
        # 60 wierszy, ale wszystkie poza 10 mają NaN w target_return → po dropna < 50.
        rows = _feature_store_rows(60)
        for r in rows[10:]:
            r["target_return"] = None
        repository_port.get_feature_store_data.return_value = rows

        summary = use_case.run("AAPL")

        ml_port.walk_forward_report.assert_not_called()
        assert summary.n_folds == 0

    def test_refresh_view_true_refreshes_feature_store(
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = []

        use_case.run("AAPL", refresh_view=True)

        repository_port.refresh_feature_store.assert_called_once()

    def test_refresh_view_default_false_does_not_refresh(
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = []

        use_case.run("AAPL")

        repository_port.refresh_feature_store.assert_not_called()

    def test_never_raises_on_empty_data(
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = []

        # Brak wyjątku = graceful degradacja.
        summary = use_case.run("AAPL")

        assert isinstance(summary, BacktestSummary)


def _stamped_rows(n: int) -> list[dict[str, Any]]:
    """Wiersze z godzinnym `timestamp` (ISO jak z PostgREST); `target_return` koduje
    godzinę, żeby test widział kolejność wierszy w foldach."""
    base = pd.Timestamp("2026-09-01T00:00:00+00:00")
    rows = _feature_store_rows(n)
    for hour, row in enumerate(rows):
        row["timestamp"] = (base + pd.Timedelta(hours=hour)).isoformat()
        row["target_return"] = float(hour)
    return rows


class TestLabelHorizonPurge:
    """Decyzja 4 (2026-10-18): main_backtest purguje horyzont etykiety tak jak model
    wspólny — walk-forward dostaje znaczniki czasu wierszy."""

    def test_passes_row_timestamps_to_walk_forward(
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = _stamped_rows(60)[::-1]
        ml_port.walk_forward_report.return_value = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60}
        ]

        use_case.run("AAPL")

        _, target = ml_port.walk_forward_report.call_args.args
        stamps = list(ml_port.walk_forward_report.call_args.kwargs["timestamps"])
        assert len(stamps) == len(target) == 60
        assert stamps == sorted(stamps)
        assert list(target) == [float(hour) for hour in range(60)]

    def test_rows_without_timestamps_run_unpurged(
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = _feature_store_rows(60)
        ml_port.walk_forward_report.return_value = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60}
        ]

        use_case.run("AAPL")

        assert ml_port.walk_forward_report.call_args.kwargs.get("timestamps") is None


class TestFoldsFromTheDomain:
    """ARCH-27: use case projektuje foldy domenowym `walk_forward_folds` (z purgem po
    znacznikach czasu), a port ML tylko na nich trenuje."""

    def test_stamped_rows_get_purged_domain_folds(
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = _stamped_rows(60)[::-1]
        ml_port.walk_forward_report.return_value = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60}
        ]

        use_case.run("AAPL")

        kwargs = ml_port.walk_forward_report.call_args.kwargs
        assert kwargs.get("folds") == walk_forward_folds(60, kwargs["timestamps"])

    def test_rows_without_timestamps_get_prefix_folds(
        self, use_case: BacktestUseCase, repository_port: Mock, ml_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = _feature_store_rows(60)
        ml_port.walk_forward_report.return_value = [
            {"candidate_rmse": 0.10, "baseline_rmse": 0.20, "hit_rate": 0.60}
        ]

        use_case.run("AAPL")

        assert ml_port.walk_forward_report.call_args.kwargs.get("folds") == walk_forward_folds(60)


class TestTradingCosts:
    """WIA-50: backtest raportuje zwrot strategii znakowej po kosztach obrotu."""

    def test_cost_bps_reaches_the_summary(self, ml_port: Mock, repository_port: Mock) -> None:
        repository_port.get_feature_store_data.return_value = _feature_store_rows(60)
        ml_port.walk_forward_report.return_value = [
            {
                "candidate_rmse": 0.10,
                "baseline_rmse": 0.20,
                "hit_rate": 0.60,
                "sign_strategy_return": 0.003,
                "turnover": 0.5,
            }
        ]
        use_case = BacktestUseCase(ml_port=ml_port, db_port=repository_port, cost_bps=10)

        summary = use_case.run("AAPL")

        assert summary.turnover == pytest.approx(0.5)
        assert summary.mean_net_return == pytest.approx(0.002)

    def test_default_cost_is_zero(
        self, use_case: BacktestUseCase, ml_port: Mock, repository_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = _feature_store_rows(60)
        ml_port.walk_forward_report.return_value = [
            {
                "candidate_rmse": 0.10,
                "baseline_rmse": 0.20,
                "hit_rate": 0.60,
                "sign_strategy_return": 0.003,
                "turnover": 0.5,
            }
        ]

        assert use_case.run("AAPL").mean_net_return == pytest.approx(0.003)


class TestFeatureSetVersion:
    """ML-51: backtest czyta tę samą próbkę co trening — przy fladze
    `feature_set_filter_enabled` wyłącznie wiersze bieżącej wersji zestawu cech."""

    def test_reads_only_the_current_feature_set_version(
        self, ml_port: Mock, repository_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = []
        use_case = BacktestUseCase(
            ml_port=ml_port, db_port=repository_port, feature_set_version=FEATURE_SET_VERSION
        )

        use_case.run("AAPL")

        repository_port.get_feature_store_data.assert_called_once_with(
            "AAPL", feature_set_version=FEATURE_SET_VERSION
        )

    def test_without_a_version_reads_the_whole_history(
        self, use_case: BacktestUseCase, repository_port: Mock
    ) -> None:
        repository_port.get_feature_store_data.return_value = []

        use_case.run("AAPL")

        repository_port.get_feature_store_data.assert_called_once_with("AAPL")


class TestRowsWithoutDataFrame:
    """ML-55: odsiew braków i kolejność po czasie na wierszach widoku zachowują się
    jak dawne `dropna` i `in_label_time_order` na DataFrame."""

    def test_missing_key_none_and_nan_count_as_missing(self) -> None:
        complete = _feature_store_rows(4)
        complete[1]["av_sentiment_score"] = None
        complete[2]["target_return"] = float("nan")
        del complete[3]["llm_trend_signal"]

        assert complete_model_rows(complete) == [complete[0]]

    def test_orders_rows_by_utc_moment_stably(self) -> None:
        rows: list[dict[str, Any]] = [
            {"id": "late", "timestamp": "2026-09-01T12:00:00+02:00"},
            {"id": "naive", "timestamp": "2026-09-01T09:00:00"},
            {"id": "z", "timestamp": "2026-09-01T09:00:00Z"},
            {"id": "aware", "timestamp": datetime(2026, 9, 1, 8, tzinfo=UTC)},
        ]

        ordered, stamps = rows_in_label_time_order(rows)

        assert [row["id"] for row in ordered] == ["aware", "naive", "z", "late"]
        assert stamps == [
            datetime(2026, 9, 1, 8, tzinfo=UTC),
            datetime(2026, 9, 1, 9, tzinfo=UTC),
            datetime(2026, 9, 1, 9, tzinfo=UTC),
            datetime(2026, 9, 1, 10, tzinfo=UTC),
        ]

    def test_rows_without_timestamp_keep_input_order(self) -> None:
        rows = _feature_store_rows(3)[::-1]

        ordered, stamps = rows_in_label_time_order(rows)

        assert ordered == rows
        assert stamps is None
