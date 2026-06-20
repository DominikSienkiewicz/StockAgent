from __future__ import annotations

from typing import Any

import pandas as pd

from src.application.ml_features import ML_FEATURE_COLUMNS, TARGET_COLUMN
from src.application.ports import MLPredictionPort, RepositoryPort

# Minimalna liczba próbek, by trening miał sens (zapobiega overfittingowi na 2 rekordach).
MIN_SAMPLES_FOR_TRAINING = 50


class TrainModelUseCase:
    """Slow Loop — retrenowanie lokalnego modelu XGBoost.

    Pipeline:
    1. Odczyt zmaterializowanego widoku `ml_feature_store` z repozytorium.
    2. Drop NaN (widok już liczy `price_delta` i `target_return`, z guardem
       anty-inf po stronie SQL — train_model NIE dubluje tej logiki).
    3. Separacja X / Y i wywołanie ml_port.train().
    """

    def __init__(self, ml_port: MLPredictionPort, db_port: RepositoryPort) -> None:
        self._ml_port = ml_port
        self._db_port = db_port

    def refresh_feature_store(self) -> None:
        self._db_port.refresh_feature_store()

    def run(self, symbol: str, refresh_view: bool = True) -> dict[str, Any]:
        if refresh_view:
            # Świeży snapshot — wymagane do Continual Learning (model nie może
            # uczyć się na starym MATERIALIZED VIEW z poprzedniego tygodnia).
            self._db_port.refresh_feature_store()

        rows = self._db_port.get_feature_store_data(symbol)

        if len(rows) < MIN_SAMPLES_FOR_TRAINING:
            return {
                "status": "skipped",
                "reason": (
                    f"Not enough samples for '{symbol}' "
                    f"({len(rows)} < {MIN_SAMPLES_FOR_TRAINING})."
                ),
            }

        df = pd.DataFrame(rows)
        # price_delta i target_return przychodzą GOTOWE z widoku (NaN-guarded).
        # Nie przeliczamy ich tutaj — stary recompute (current-prev)/prev dublował
        # logikę widoku i przy prev=0 dawał +inf, który przeżywał dropna i trafiał
        # do XGBoost. dropna odsiewa wiersze, dla których widok zwrócił NULL.
        required_columns = [*ML_FEATURE_COLUMNS, TARGET_COLUMN]
        df = df.dropna(subset=required_columns)

        if len(df) < MIN_SAMPLES_FOR_TRAINING:
            return {
                "status": "skipped",
                "reason": (
                    f"Not enough valid samples for '{symbol}' "
                    f"({len(df)} < {MIN_SAMPLES_FOR_TRAINING})."
                ),
            }

        features = df[list(ML_FEATURE_COLUMNS)]
        target = df[TARGET_COLUMN]

        return self._ml_port.train(features, target)
