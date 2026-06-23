from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.application.ports import MLPredictionPort, RepositoryPort
from src.application.use_cases.train_model import (
    MIN_SAMPLES_FOR_TRAINING,
    TrainModelUseCase,
)

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
def use_case(ml_port, repository_port) -> TrainModelUseCase:
    return TrainModelUseCase(ml_port=ml_port, db_port=repository_port)


def _feature_store_rows(n: int) -> list[dict]:
    """Generuje n rekordów jak z widoku ml_feature_store.

    Widok dostarcza GOTOWE `price_delta` (NaN-guarded) i `target_return` (zwrot
    12h) — train_model ich NIE przelicza, tylko selekcjonuje i robi dropna.
    """
    return [
        {
            "price_current": 100.0 + i * 0.5,
            "price_prev_12h": 99.0 + i * 0.5,
            "price_delta": (100.0 + i * 0.5 - (99.0 + i * 0.5)) / (99.0 + i * 0.5),
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
    def test_refresh_feature_store_delegates_to_repository(
        self, use_case, repository_port
    ):
        use_case.refresh_feature_store()

        repository_port.refresh_feature_store.assert_called_once()

    def test_returns_not_enough_data_when_below_minimum(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = _feature_store_rows(
            MIN_SAMPLES_FOR_TRAINING - 1
        )

        result = use_case.run("AAPL")

        assert result["status"] == "skipped"
        assert "not enough" in result["reason"].lower() or "za mało" in result["reason"].lower()
        ml_port.train.assert_not_called()

    def test_returns_not_enough_data_when_empty(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = []

        result = use_case.run("AAPL")

        assert result["status"] == "skipped"
        ml_port.train.assert_not_called()

    def test_can_skip_feature_store_refresh(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = []

        use_case.run("AAPL", refresh_view=False)

        repository_port.refresh_feature_store.assert_not_called()
        ml_port.train.assert_not_called()

    def test_trains_model_with_correct_features_and_target(
        self, use_case, repository_port, ml_port
    ):
        repository_port.get_feature_store_data.return_value = _feature_store_rows(
            MIN_SAMPLES_FOR_TRAINING + 10
        )
        ml_port.train.return_value = {"status": "trained_successfully", "n_samples": 60}

        use_case.run("AAPL")

        ml_port.train.assert_called_once()
        features, target = ml_port.train.call_args.args
        assert isinstance(features, pd.DataFrame)
        assert list(features.columns) == EXPECTED_ML_FEATURES
        # Target = target_return (zwrot 12h z widoku), nie cena bezwzględna
        assert isinstance(target, pd.Series)
        assert len(features) == len(target)

    def test_uses_view_price_delta_without_recompute(
        self, use_case, repository_port, ml_port
    ):
        # Widok dostarcza price_delta GOTOWE — train_model nie wolno go przeliczać
        # (stary recompute (current-prev)/prev dublował logikę i robił +inf przy
        # prev=0). Ustawiamy price_delta na wartość, której recompute NIGDY by nie
        # dał, i sprawdzamy, że trafia do modelu bez zmian.
        rows = _feature_store_rows(MIN_SAMPLES_FOR_TRAINING + 5)
        for r in rows:
            r["price_delta"] = 0.5
        repository_port.get_feature_store_data.return_value = rows
        ml_port.train.return_value = {"status": "trained_successfully"}

        use_case.run("AAPL")

        features, _ = ml_port.train.call_args.args
        # Wszystkie wiersze dostały price_delta=0.5 — porównanie tolerancyjne na float.
        assert np.allclose(features["price_delta"], 0.5)

    def test_drops_rows_when_view_price_delta_is_null(
        self, use_case, repository_port, ml_port
    ):
        # Widok emituje price_delta=NULL gdy price_prev_12h=0/NULL (guard anty-inf).
        # train_model NIE przelicza (stary recompute dawał +inf przeżywające
        # dropna) — wiersz z NULL price_delta musi zostać odrzucony, a do modelu
        # trafiają wyłącznie skończone wartości.
        rows = _feature_store_rows(MIN_SAMPLES_FOR_TRAINING + 5)
        rows[0]["price_delta"] = None  # type: ignore[assignment]
        repository_port.get_feature_store_data.return_value = rows
        ml_port.train.return_value = {"status": "trained_successfully"}

        use_case.run("AAPL")

        features, _ = ml_port.train.call_args.args
        assert len(features) == len(rows) - 1
        assert bool(np.isfinite(features["price_delta"]).all())

    def test_returns_ml_train_result(self, use_case, repository_port, ml_port):
        repository_port.get_feature_store_data.return_value = _feature_store_rows(
            MIN_SAMPLES_FOR_TRAINING + 10
        )
        ml_port.train.return_value = {
            "status": "trained_successfully",
            "n_samples": 60,
            "model_path": "data/models/price_predictor.ubj",
        }

        result = use_case.run("AAPL")

        assert result["status"] == "trained_successfully"
        assert result["n_samples"] == 60
