from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
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

VALIDATION_FRACTION = 0.2
MIN_VALIDATION_SAMPLES = 10
MIN_RMSE_IMPROVEMENT = 1e-9


class XGBoostAdapter(MLPredictionPort):
    """Adapter dla lokalnego modelu XGBoost (Local-First AI).

    Model jest persystowany jako plik `.ubj` (UBJSON, natywny format XGBoost).
    Trening i predykcja w pamięci — bez zewnętrznych usług.

    Kontrakt targetu: model przewiduje **ZWROT 12h** (`(cena_po_12h - cena) /
    cena`), nie cenę bezwzględną. Dzięki temu:
      - RMSE mierzy trafność RUCHU, nie poziomu ceny (poziom dominował RMSE,
        więc model „echujący" ostatnią cenę wygrywał, nie ucząc się niczego);
      - baseline persystencji to po prostu „zwrot = 0" (cena bez zmian), więc
        bramka „pobij baseline" jest sensowna.
    `predict()` rekonstruuje cenę bezwzględną z przewidzianego zwrotu i bieżącej
    ceny. UWAGA migracyjna: model wytrenowany na starym kontrakcie (cena
    bezwzględna) jest NIEKOMPATYBILNY — po zmianie usuń stary plik `.ubj`
    (cold-start = „brak ruchu" = zwrot 0) i wytrenuj od nowa w Slow Loop.
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
        feature_frame = pd.DataFrame(features).reset_index(drop=True)
        target_series = pd.Series(target).reset_index(drop=True)
        train_x, valid_x, train_y, valid_y = _time_ordered_split(
            feature_frame, target_series
        )

        candidate = xgb.XGBRegressor(**self._params)
        candidate.fit(train_x, train_y)

        candidate_preds = candidate.predict(valid_x)
        validation_rmse = _rmse(valid_y, candidate_preds)
        # Baseline persystencji: „zwrot = 0" (cena 12h naprzód = cena bieżąca).
        # Target to zwrot, więc uczciwy baseline to wektor zer — a NIE rozrzut
        # pojedynczego skalara z train (stary baseline mierzył coś bez sensu).
        baseline_rmse = _rmse(valid_y, np.zeros(len(valid_y)))
        directional_hit_rate = _directional_hit_rate(valid_y, candidate_preds)
        if validation_rmse >= baseline_rmse - MIN_RMSE_IMPROVEMENT:
            return {
                "status": "skipped_validation_failed",
                "reason": (
                    "Model did not beat the zero-return persistence baseline "
                    "on the validation holdout."
                ),
                "n_samples": len(target_series),
                "validation_rmse": validation_rmse,
                "baseline_rmse": baseline_rmse,
                "validation_directional_hit_rate": directional_hit_rate,
                "model_path": self._model_path,
            }

        model = xgb.XGBRegressor(**self._params)
        model.fit(feature_frame, target_series)
        model_path = Path(self._model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(self._model_path)
        self._model = model
        return {
            "status": "trained_successfully",
            "n_samples": len(target_series),
            "validation_rmse": validation_rmse,
            "baseline_rmse": baseline_rmse,
            "validation_directional_hit_rate": directional_hit_rate,
            "model_path": self._model_path,
        }

    def predict(
        self, current_features: dict[str, float], current_price: Decimal
    ) -> Money:
        if self._model is None:
            raise RuntimeError(
                f"Model not trained — no weights at '{self._model_path}'. "
                "Run TrainModelUseCase first (Slow Loop)."
            )
        frame = self._align_features(current_features)
        # Model przewiduje ZWROT 12h; rekonstruujemy cenę bezwzględną:
        # cena_docelowa = cena_bieżąca * (1 + zwrot).
        predicted_return = float(self._model.predict(frame)[0])
        target_price = current_price * (Decimal("1") + Decimal(str(predicted_return)))
        return Money(target_price)

    def _align_features(self, current_features: dict[str, float]) -> pd.DataFrame:
        """Wyrównuje wejście do cech, na których model BYŁ trenowany.

        Bierze dokładnie `feature_names_in_` w kolejności modelu — ignoruje
        nadmiarowe cechy (np. nowa cecha w ML_FEATURE_COLUMNS, której stary
        model w repo jeszcze nie zna) i jawnie zgłasza brakujące (KeyError),
        zamiast wpychać NaN albo wywalać się na mismatchu nazw kolumn.
        Dzięki temu kontrakt cech może ewoluować bez psucia predykcji.
        """
        expected = getattr(self._model, "feature_names_in_", None)
        if expected is None:
            return pd.DataFrame([current_features])
        expected_cols = [str(c) for c in expected]
        missing = [c for c in expected_cols if c not in current_features]
        if missing:
            raise KeyError(
                f"Missing required ML features for prediction: {missing}"
            )
        return pd.DataFrame([{c: current_features[c] for c in expected_cols}])


def _time_ordered_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    validation_size = max(MIN_VALIDATION_SAMPLES, int(len(target) * VALIDATION_FRACTION))
    if validation_size >= len(target):
        raise ValueError("Not enough samples to create validation holdout.")

    train_end = len(target) - validation_size
    return (
        features.iloc[:train_end],
        features.iloc[train_end:],
        target.iloc[:train_end],
        target.iloc[train_end:],
    )


def _rmse(actual: pd.Series, predicted: Any) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual_arr - predicted_arr) ** 2)))


def _directional_hit_rate(actual: pd.Series, predicted: Any) -> float:
    """Frakcja trafień KIERUNKU zwrotu (znak przewidzianego = znak rzeczywistego).

    Uzupełnia RMSE: model może mieć niskie RMSE, a wciąż mylić kierunek ruchu.
    Liczony na tym samym holdoucie walidacyjnym co `validation_rmse`."""
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    if actual_arr.size == 0:
        return 0.0
    return float(np.mean(np.sign(actual_arr) == np.sign(predicted_arr)))
