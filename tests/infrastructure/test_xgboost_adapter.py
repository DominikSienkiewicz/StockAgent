from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.ports import MLPredictionPort
from src.domain.value_objects import Money
from src.infrastructure.adapters.xgboost_local import XGBoostAdapter

FEATURE_NAMES = ["price_delta", "sentiment_score", "llm_trend_signal"]


def _synthetic_dataset(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Generuje syntetyczny dataset: y = 100 + 50*price_delta + 0.1*sentiment.

    Sygnał na tyle silny, że nawet płytki XGBoost odtworzy zależność.
    """
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        {
            "price_delta": rng.normal(0, 0.05, n),
            "sentiment_score": rng.uniform(0, 100, n),
            "llm_trend_signal": rng.choice([-1, 0, 1], size=n).astype(float),
        }
    )
    target = pd.Series(
        100 + features["price_delta"] * 50 + features["sentiment_score"] * 0.1
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


class TestPredict:
    def test_returns_money_instance(self, trained_adapter: XGBoostAdapter):
        features = {"price_delta": 0.02, "sentiment_score": 75.0, "llm_trend_signal": 1.0}
        prediction = trained_adapter.predict(features)
        assert isinstance(prediction, Money)

    def test_prediction_is_finite_positive_for_typical_input(
        self, trained_adapter: XGBoostAdapter
    ):
        features = {"price_delta": 0.0, "sentiment_score": 50.0, "llm_trend_signal": 0.0}
        prediction = trained_adapter.predict(features)
        # syntetyczny model dla x=0, sentiment=50 → ~105
        assert Decimal("80") < prediction.amount < Decimal("130")

    def test_raises_when_model_not_trained(self, adapter: XGBoostAdapter):
        features = {"price_delta": 0.0, "sentiment_score": 50.0, "llm_trend_signal": 0.0}
        with pytest.raises(RuntimeError, match="not trained"):
            adapter.predict(features)


class TestModelPersistence:
    def test_loads_existing_model_from_disk_on_init(
        self, trained_adapter: XGBoostAdapter, model_path: str
    ):
        # Pierwszy adapter trenuje i zapisuje. Drugi czyta z dysku.
        reloaded = XGBoostAdapter(model_path=model_path)
        features = {"price_delta": 0.02, "sentiment_score": 75.0, "llm_trend_signal": 1.0}

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
