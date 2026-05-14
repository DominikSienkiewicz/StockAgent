from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

from src.application.ports import MLPredictionPort
from src.domain.value_objects import Money

# Hiperparametry dobrane pod dane finansowe (zapobieganie overfittingowi).
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 100,
}


class XGBoostAdapter(MLPredictionPort):
    """Adapter dla lokalnego modelu XGBoost (Local-First AI).

    Model jest persystowany jako plik `.ubj` (UBJSON, natywny format XGBoost).
    Trening i predykcja w pamięci — bez zewnętrznych usług.
    """

    def __init__(self, model_path: str, params: dict[str, Any] | None = None) -> None:
        self._model_path = model_path
        self._params = params or DEFAULT_PARAMS
        self._model: xgb.XGBRegressor | None = None
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not Path(self._model_path).exists():
            return
        model = xgb.XGBRegressor()
        model.load_model(self._model_path)
        self._model = model

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(self, features: Any, target: Any) -> dict[str, Any]:
        model = xgb.XGBRegressor(**self._params)
        model.fit(features, target)
        model.save_model(self._model_path)
        self._model = model
        return {
            "status": "trained_successfully",
            "n_samples": len(target),
            "model_path": self._model_path,
        }

    def predict(self, current_features: dict[str, float]) -> Money:
        if self._model is None:
            raise RuntimeError(
                f"Model not trained — no weights at '{self._model_path}'. "
                "Run TrainModelUseCase first (Slow Loop)."
            )
        frame = pd.DataFrame([current_features])
        prediction = float(self._model.predict(frame)[0])
        return Money(Decimal(str(prediction)))
