from __future__ import annotations

from typing import Any

from src.application.ml_features import (
    ML_FEATURE_COLUMNS,
    TARGET_COLUMN,
    complete_model_rows,
    rows_in_label_time_order,
)
from src.application.ports import MLPredictionPort, RepositoryPort
from src.domain.backtest import BacktestSummary, summarize_folds

# Minimalna liczba poprawnych próbek, by walk-forward backtest miał sens.
# Poniżej tego progu foldy out-of-sample są zbyt małe, by werdykt był
# wiarygodny — zwracamy pusty BacktestSummary zamiast wołać model.
MIN_BACKTEST_SAMPLES = 50


class BacktestUseCase:
    """Offline walk-forward backtest lokalnego modelu (harness diagnostyczny).

    Reużywa logikę foldów z `MLPredictionPort.walk_forward_report` (finding #23) —
    NIE re-implementuje splitów. Czysto ewaluacyjny: czyta feature store, podaje
    portowi ML wiersze cech i target jak `train` (ML-55: bez DataFrame w tej
    warstwie), a wynik agreguje w domenowym `summarize_folds`. Nigdy nie rzuca na
    pustych/krótkich danych — degraduje do pustego `BacktestSummary`.
    """

    def __init__(
        self,
        ml_port: MLPredictionPort,
        db_port: RepositoryPort,
        *,
        cost_bps: float = 0.0,
        feature_set_version: int | None = None,
    ) -> None:
        """`feature_set_version` (ML-51) zawęża próbkę do wierszy tej wersji zestawu
        cech, jak w `TrainModelUseCase`; None = cała historia (bez migracji 027)."""
        self._ml_port = ml_port
        self._db_port = db_port
        # WIA-50: jednostronny koszt obrotu do zwrotu netto strategii znakowej.
        self._cost_bps = cost_bps
        self._feature_set_version = feature_set_version

    def run(self, symbol: str, *, refresh_view: bool = False) -> BacktestSummary:
        if refresh_view:
            # Świeży snapshot widoku przed backtestem (opcjonalny — domyślnie
            # backtest jest read-only nad istniejącym ml_feature_store).
            self._db_port.refresh_feature_store()

        rows = self._feature_rows(symbol)
        if not rows:
            return _too_few_samples_summary(0)

        # price_delta i target_return przychodzą GOTOWE z widoku (NaN-guarded) —
        # nie przeliczamy ich tutaj (jak train_model). Odsiewamy wiersze, dla
        # których widok zwrócił NULL.
        complete = complete_model_rows(rows)

        if len(complete) < MIN_BACKTEST_SAMPLES:
            return _too_few_samples_summary(len(complete))

        # Decyzja 4 (2026-10-18): purge horyzontu etykiety jak w modelu wspólnym.
        ordered, stamps = rows_in_label_time_order(complete)

        # Kolejność kluczy = kontrakt `ML_FEATURE_COLUMNS`, ta sama co w `train`.
        features = [{name: float(row[name]) for name in ML_FEATURE_COLUMNS} for row in ordered]
        target = [float(row[TARGET_COLUMN]) for row in ordered]
        folds = self._ml_port.walk_forward_report(features, target, timestamps=stamps)
        return summarize_folds(folds, cost_bps=self._cost_bps)

    def _feature_rows(self, symbol: str) -> list[dict[str, Any]]:
        if self._feature_set_version is None:
            return self._db_port.get_feature_store_data(symbol)
        return self._db_port.get_feature_store_data(
            symbol, feature_set_version=self._feature_set_version
        )


def _too_few_samples_summary(n_valid: int) -> BacktestSummary:
    """Pusty werdykt z czytelnym powodem (za mało próbek na sensowny backtest)."""
    return BacktestSummary(
        n_folds=0,
        mean_candidate_rmse=0,
        mean_baseline_rmse=0,
        mean_hit_rate=0,
        improvement_pct=0,
        beats_baseline=False,
        verdict=(f"za mało próbek na backtest ({n_valid} < {MIN_BACKTEST_SAMPLES})"),
    )
