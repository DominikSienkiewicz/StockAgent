from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.ports import MLPredictionPort
from src.domain.value_objects import Money
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter

FEATURE_NAMES = [
    "price_delta",
    "av_sentiment_score",
    "av_relevance_avg",
    "news_volume_24h",
    "high_relevance_count",
    "llm_trend_signal",
    "av_llm_agreement",
]


def _synthetic_dataset(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Generuje syntetyczny dataset: y = 100 + 50*price_delta + 0.1*sentiment.

    Sygnał na tyle silny, że nawet płytki XGBoost odtworzy zależność.
    """
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        {
            "price_delta": rng.normal(0, 0.05, n),
            "av_sentiment_score": rng.uniform(-1, 1, n),
            "av_relevance_avg": rng.uniform(0, 1, n),
            "news_volume_24h": rng.integers(0, 12, n).astype(float),
            "high_relevance_count": rng.integers(0, 5, n).astype(float),
            "llm_trend_signal": rng.choice([-1, 0, 1], size=n).astype(float),
            "av_llm_agreement": rng.uniform(0, 1, n),
        }
    )
    target = pd.Series(
        100
        + features["price_delta"] * 50
        + features["av_sentiment_score"] * 8
        + features["llm_trend_signal"] * 2
        + features["av_llm_agreement"] * 3
    )
    return features, target


@pytest.fixture
def model_path(tmp_path: Path) -> str:
    return str(tmp_path / "price_predictor.ubj")


@pytest.fixture
def adapter(model_path: str) -> XGBoostAdapter:
    return XGBoostAdapter(model_path=model_path)


@pytest.fixture
def trained_adapter(adapter: XGBoostAdapter) -> XGBoostAdapter:
    features, target = _synthetic_dataset()
    adapter.train(features, target)
    return adapter


class TestTrain:
    def test_returns_success_status(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset()
        result = adapter.train(features, target)
        assert result["status"] == "trained_successfully"

    def test_includes_sample_count_in_status(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset(n=50)
        result = adapter.train(features, target)
        assert result["n_samples"] == 50

    def test_saves_model_to_disk(self, adapter: XGBoostAdapter, model_path: str):
        features, target = _synthetic_dataset()
        adapter.train(features, target)
        assert Path(model_path).exists()
        assert Path(model_path).stat().st_size > 0

    def test_returns_validation_metrics(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset()
        result = adapter.train(features, target)

        assert result["validation_rmse"] < result["baseline_rmse"]

    def test_skips_save_when_model_does_not_beat_baseline(
        self, adapter: XGBoostAdapter, model_path: str
    ):
        features, _ = _synthetic_dataset(n=80)
        target = pd.Series([100.0] * 80)

        result = adapter.train(features, target)

        assert result["status"] == "skipped_validation_failed"
        assert not Path(model_path).exists()
        assert adapter.is_trained is False


class TestPredict:
    def test_returns_money_instance(self, trained_adapter: XGBoostAdapter):
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        features.update({
            "price_delta": 0.02,
            "av_sentiment_score": 0.5,
            "av_relevance_avg": 0.75,
            "news_volume_24h": 4.0,
            "high_relevance_count": 2.0,
            "llm_trend_signal": 1.0,
            "av_llm_agreement": 0.8,
        })
        prediction = trained_adapter.predict(features)
        assert isinstance(prediction, Money)

    def test_prediction_is_finite_positive_for_typical_input(
        self, trained_adapter: XGBoostAdapter
    ):
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        features.update({
            "av_relevance_avg": 0.5,
            "news_volume_24h": 3.0,
            "av_llm_agreement": 0.5,
        })
        prediction = trained_adapter.predict(features)
        assert Decimal("80") < prediction.amount < Decimal("130")

    def test_raises_when_model_not_trained(self, adapter: XGBoostAdapter):
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        with pytest.raises(RuntimeError, match="not trained"):
            adapter.predict(features)


class TestModelPersistence:
    def test_loads_existing_model_from_disk_on_init(
        self, trained_adapter: XGBoostAdapter, model_path: str
    ):
        # Pierwszy adapter trenuje i zapisuje. Drugi czyta z dysku.
        reloaded = XGBoostAdapter(model_path=model_path)
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        features.update({
            "price_delta": 0.02,
            "av_sentiment_score": 0.5,
            "av_relevance_avg": 0.75,
            "news_volume_24h": 4.0,
            "high_relevance_count": 2.0,
            "llm_trend_signal": 1.0,
            "av_llm_agreement": 0.8,
        })

        original = trained_adapter.predict(features)
        from_disk = reloaded.predict(features)

        assert original.amount == from_disk.amount

    def test_cold_start_does_not_raise(self, tmp_path: Path):
        # Brak pliku — konstruktor musi przeżyć (czekamy na pierwszy trening).
        adapter = XGBoostAdapter(model_path=str(tmp_path / "missing.ubj"))
        assert adapter is not None


class TestIsTrained:
    def test_false_on_cold_start(self, adapter: XGBoostAdapter):
        # Brak pliku modelu → is_trained False (Fast Loop użyje baseline)
        assert adapter.is_trained is False

    def test_true_after_training(self, adapter: XGBoostAdapter):
        features, target = _synthetic_dataset()
        adapter.train(features, target)
        assert adapter.is_trained is True

    def test_true_when_model_loaded_from_disk(
        self, trained_adapter: XGBoostAdapter, model_path: str
    ):
        reloaded = XGBoostAdapter(model_path=model_path)
        assert reloaded.is_trained is True


class TestAdapterImplementsPort:
    def test_is_ml_prediction_port(self):
        assert issubclass(XGBoostAdapter, MLPredictionPort)
