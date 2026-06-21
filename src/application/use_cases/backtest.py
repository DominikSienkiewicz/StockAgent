from __future__ import annotations

import pandas as pd

from src.application.ml_features import ML_FEATURE_COLUMNS, TARGET_COLUMN
from src.application.ports import MLPredictionPort, RepositoryPort
from src.domain.backtest import BacktestSummary, summarize_folds

# Minimalna liczba poprawnych próbek, by walk-forward backtest miał sens.
# Poniżej tego progu foldy out-of-sample są zbyt małe, by werdykt był
# wiarygodny — zwracamy pusty BacktestSummary zamiast wołać model.
MIN_BACKTEST_SAMPLES = 50


class BacktestUseCase:
    """Offline walk-forward backtest lokalnego modelu (harness diagnostyczny).

    Reużywa logikę foldów z `MLPredictionPort.walk_forward_report` (finding #23) —
    NIE re-implementuje splitów. Czysto ewaluacyjny: czyta feature store, składa
    DataFrame i deleguje do portu ML, a wynik agreguje w domenowym
    `summarize_folds`. Nigdy nie rzuca na pustych/krótkich danych — degraduje do
    pustego `BacktestSummary`.
    """

    def __init__(self, ml_port: MLPredictionPort, db_port: RepositoryPort) -> None:
        self._ml_port = ml_port
        self._db_port = db_port

    def run(self, symbol: str, *, refresh_view: bool = False) -> BacktestSummary:
        if refresh_view:
            # Świeży snapshot widoku przed backtestem (opcjonalny — domyślnie
            # backtest jest read-only nad istniejącym ml_feature_store).
            self._db_port.refresh_feature_store()

        rows = self._db_port.get_feature_store_data(symbol)
        if not rows:
            return _too_few_samples_summary(0)

        df = pd.DataFrame(rows)
        # price_delta i target_return przychodzą GOTOWE z widoku (NaN-guarded) —
        # nie przeliczamy ich tutaj (jak train_model). dropna odsiewa wiersze, dla
        # których widok zwrócił NULL.
        required_columns = [*ML_FEATURE_COLUMNS, TARGET_COLUMN]
        df = df.dropna(subset=required_columns)

        if len(df) < MIN_BACKTEST_SAMPLES:
            return _too_few_samples_summary(len(df))

        features = df[list(ML_FEATURE_COLUMNS)]
        target = df[TARGET_COLUMN]

        folds = self._ml_port.walk_forward_report(features, target)
        return summarize_folds(folds)


def _too_few_samples_summary(n_valid: int) -> BacktestSummary:
    """Pusty werdykt z czytelnym powodem (za mało próbek na sensowny backtest)."""
    return BacktestSummary(
        n_folds=0,
        mean_candidate_rmse=0,
        mean_baseline_rmse=0,
        mean_hit_rate=0,
        improvement_pct=0,
        beats_baseline=False,
        verdict=(
            f"za mało próbek na backtest ({n_valid} < {MIN_BACKTEST_SAMPLES})"
        ),
    )
