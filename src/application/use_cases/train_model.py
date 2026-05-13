from __future__ import annotations

from typing import Any

import pandas as pd

from src.application.ports import MLPredictionPort, RepositoryPort

# Minimalna liczba próbek, by trening miał sens (zapobiega overfittingowi na 2 rekordach).
MIN_SAMPLES_FOR_TRAINING = 50


class TrainModelUseCase:
    """Slow Loop — retrenowanie lokalnego modelu XGBoost.

    Pipeline:
    1. Odczyt zmaterializowanego widoku `ml_feature_store` z repozytorium.
    2. Feature engineering (kalkulacja price_delta).
    3. Drop NaN.
    4. Separacja X / Y i wywołanie ml_port.train().
    """

    def __init__(self, ml_port: MLPredictionPort, db_port: RepositoryPort) -> None:
        self._ml_port = ml_port
        self._db_port = db_port

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
        df["price_delta"] = (
            (df["price_current"] - df["price_prev_12h"]) / df["price_prev_12h"]
        )
        required_columns = [
            "price_delta", "sentiment_score", "llm_trend_signal", "target_price"
        ]
        df = df.dropna(subset=required_columns)

        features = df[["price_delta", "sentiment_score", "llm_trend_signal"]]
        target = df["target_price"]

        return self._ml_port.train(features, target)
