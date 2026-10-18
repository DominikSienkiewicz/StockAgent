from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.application.ml_features import ML_FEATURE_COLUMNS, TARGET_COLUMN
from src.application.ports import MLPredictionPort, RepositoryPort

logger = logging.getLogger(__name__)

# Minimalna liczba próbek, by trening miał sens (zapobiega overfittingowi na 2 rekordach).
# Dla modelu wspólnego (ML-13) dotyczy ŁĄCZNEJ próbki wszystkich symboli.
MIN_SAMPLES_FOR_TRAINING = 50

# Symbol wiersza karty modelu (`model_scorecards`) dla modelu wspólnego.
POOLED_MODEL_SYMBOL = "ALL"


class TrainModelUseCase:
    """Slow Loop — lokalny model XGBoost.

    Decyzja 4 (ML-13): wagi produkuje JEDEN wspólny model na bezwymiarowych cechach
    wszystkich symboli (`run_pooled`). `run` zostaje jako per-symbol EWALUACJA bez
    zapisu wag — dawniej każdy symbol nadpisywał ten sam plik `.ubj`.

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
        """Per-symbol ewaluacja walk-forward — nigdy nie zapisuje wag (ML-13)."""
        if refresh_view:
            # Świeży snapshot — wymagane do Continual Learning (model nie może
            # uczyć się na starym MATERIALIZED VIEW z poprzedniego tygodnia).
            self._db_port.refresh_feature_store()

        rows = self._db_port.get_feature_store_data(symbol)

        if len(rows) < MIN_SAMPLES_FOR_TRAINING:
            return _skipped(f"Not enough samples for '{symbol}'", len(rows))

        df = _valid_rows(rows)

        if len(df) < MIN_SAMPLES_FOR_TRAINING:
            return _skipped(f"Not enough valid samples for '{symbol}'", len(df))

        features = df[list(ML_FEATURE_COLUMNS)]
        target = df[TARGET_COLUMN]

        return self._ml_port.train(features, target, persist=False)

    def run_pooled(
        self, symbols: Sequence[str], *, record_scorecard: bool = False
    ) -> dict[str, Any]:
        """ML-13 — jeden wspólny model na wierszach wszystkich `symbols`, jeden zapis.

        Wiersze są sortowane po czasie GLOBALNYM, a walk-forward dostaje ich
        znaczniki czasu (purge horyzontu etykiety) — inaczej fold walidowałby jeden
        symbol na chwilach sprzed treningu na innym. Z `record_scorecard` wynik
        trafia do `model_scorecards` jako `POOLED_MODEL_SYMBOL`, best-effort.
        """
        result = self._train_pooled(symbols)
        if record_scorecard:
            try:
                self._db_port.save_model_scorecard(POOLED_MODEL_SYMBOL, result)
            except Exception:
                logger.exception("Failed to persist the pooled model scorecard")
        return result

    def _train_pooled(self, symbols: Sequence[str]) -> dict[str, Any]:
        rows = [
            row
            for symbol in dict.fromkeys(symbols)
            for row in self._db_port.get_feature_store_data(symbol)
        ]

        if len(rows) < MIN_SAMPLES_FOR_TRAINING:
            return _skipped("Not enough samples for the pooled model", len(rows))

        df = _valid_rows(rows)

        if len(df) < MIN_SAMPLES_FOR_TRAINING:
            return _skipped("Not enough valid samples for the pooled model", len(df))

        stamps = pd.to_datetime(df["timestamp"], utc=True)
        df = df.assign(_global_time=stamps).sort_values(
            ["_global_time", "symbol"], kind="stable"
        )

        return self._ml_port.train(
            df[list(ML_FEATURE_COLUMNS)],
            df[TARGET_COLUMN],
            timestamps=df["_global_time"],
        )


def _valid_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Wiersze z kompletem cech i targetu.

    price_delta i target_return przychodzą GOTOWE z widoku (NaN-guarded). Nie
    przeliczamy ich tutaj — stary recompute (current-prev)/prev dublował logikę
    widoku i przy prev=0 dawał +inf, który przeżywał dropna i trafiał do XGBoost.
    dropna odsiewa wiersze, dla których widok zwrócił NULL.
    """
    required_columns = [*ML_FEATURE_COLUMNS, TARGET_COLUMN]
    return pd.DataFrame(rows).dropna(subset=required_columns)


def _skipped(reason: str, n_samples: int) -> dict[str, Any]:
    """Wynik pominiętego treningu. `n_samples` jest polem (ML-04), nie tylko tekstem
    powodu — scorecard i logi pokazują, ile próbek brakuje do progu."""
    return {
        "status": "skipped",
        "reason": f"{reason} ({n_samples} < {MIN_SAMPLES_FOR_TRAINING}).",
        "n_samples": n_samples,
    }
