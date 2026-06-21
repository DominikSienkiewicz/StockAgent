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

# Walk-forward (expanding window): liczba foldów i minimalna liczba próbek, poniżej
# której nie ma sensu robić CV (spadamy do pojedynczego splitu). Każdy fold dokłada
# kolejny chronologiczny blok do treningu i waliduje na następnym — to lepszy
# (mniej hałaśliwy) estymator dla niestacjonarnej serii finansowej niż jeden split.
WALK_FORWARD_FOLDS = 4
MIN_FOLD_SAMPLES = 50

# Komunikat dołączany do wyniku: raportowane metryki dotyczą KANDYDATA na
# holdoucie (foldach), a shipowany model jest refitowany na WSZYSTKICH danych
# i nie jest re-walidowany — więc nie traktuj tych liczb jak jakości produkcyjnej.
METRIC_NOTE = (
    "candidate holdout metrics (walk-forward folds); shipped model refit on "
    "all data and not re-validated"
)


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

        # Walk-forward (expanding window) zamiast jednego splitu 20%: bramka
        # „pobij baseline" opiera się na ŚREDNIEJ z foldów, nie na jednym
        # hałaśliwym estymatorze z jednego reżimu rynku (finding #23).
        candidate_rmse, baseline_rmse, directional_hit_rate, n_folds = (
            self._walk_forward_evaluate(feature_frame, target_series)
        )
        # Per-feature drift stats (mean/std) — obserwowalne niezależnie od
        # werdyktu bramki, żeby porównać rozkład cech z poprzednim runem.
        feature_stats = _feature_distribution_stats(feature_frame)

        if candidate_rmse >= baseline_rmse - MIN_RMSE_IMPROVEMENT:
            return {
                "status": "skipped_validation_failed",
                "reason": (
                    "Candidate did not beat the zero-return persistence "
                    "baseline on average across walk-forward folds."
                ),
                "n_samples": len(target_series),
                "n_folds": n_folds,
                "candidate_holdout_rmse": candidate_rmse,
                "baseline_rmse": baseline_rmse,
                "candidate_holdout_directional_hit_rate": directional_hit_rate,
                "feature_stats": feature_stats,
                "metric_note": METRIC_NOTE,
                "model_path": self._model_path,
            }

        # Shipowany model: refit na WSZYSTKICH danych (train+holdout) — NIE jest
        # re-walidowany, więc raportowane RMSE/hit-rate to metryki KANDYDATA
        # (finding #22, patrz METRIC_NOTE).
        model = xgb.XGBRegressor(**self._params)
        model.fit(feature_frame, target_series)
        model_path = Path(self._model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(self._model_path)
        self._model = model
        return {
            "status": "trained_successfully",
            "n_samples": len(target_series),
            "n_folds": n_folds,
            "candidate_holdout_rmse": candidate_rmse,
            "baseline_rmse": baseline_rmse,
            "candidate_holdout_directional_hit_rate": directional_hit_rate,
            "feature_stats": feature_stats,
            "metric_note": METRIC_NOTE,
            "model_path": self._model_path,
        }

    def _walk_forward_folds_detail(
        self, features: pd.DataFrame, target: pd.Series
    ) -> list[dict[str, float]]:
        """Per-fold metryki walk-forward (expanding window) na danych
        time-ordered. Jedno źródło prawdy dla `_walk_forward_evaluate` (średnia)
        i `walk_forward_report` (szczegół). Poniżej `MIN_FOLD_SAMPLES` spada do
        pojedynczego splitu (sane fallback) → jeden fold.
        """
        folds = _expanding_window_folds(len(target))
        out: list[dict[str, float]] = []
        for train_end, valid_end in folds:
            train_x = features.iloc[:train_end]
            train_y = target.iloc[:train_end]
            valid_x = features.iloc[train_end:valid_end]
            valid_y = target.iloc[train_end:valid_end]

            candidate = xgb.XGBRegressor(**self._params)
            candidate.fit(train_x, train_y)
            preds = candidate.predict(valid_x)

            out.append(
                {
                    "train_end": float(train_end),
                    "valid_end": float(valid_end),
                    "n_train": float(train_end),
                    "n_valid": float(valid_end - train_end),
                    "candidate_rmse": _rmse(valid_y, preds),
                    # Baseline persystencji: „zwrot = 0" (cena 12h = cena bieżąca).
                    "baseline_rmse": _rmse(valid_y, np.zeros(len(valid_y))),
                    "hit_rate": _directional_hit_rate(valid_y, preds),
                }
            )
        return out

    def _walk_forward_evaluate(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[float, float, float, int]:
        """Ewaluacja walk-forward (expanding window) na danych time-ordered.

        Zwraca (avg_candidate_rmse, avg_baseline_rmse, avg_directional_hit_rate,
        n_folds) uśrednione po foldach z `_walk_forward_folds_detail`.
        """
        details = self._walk_forward_folds_detail(features, target)
        return (
            float(np.mean([d["candidate_rmse"] for d in details])),
            float(np.mean([d["baseline_rmse"] for d in details])),
            float(np.mean([d["hit_rate"] for d in details])),
            len(details),
        )

    def walk_forward_report(
        self, features: Any, target: Any
    ) -> list[dict[str, float]]:
        """Per-fold metryki OOS bez persystencji (offline backtest harness).

        Reużywa logikę foldów `train()` (finding #23). Czysto ewaluacyjna —
        nie zapisuje modelu i nie zmienia stanu adaptera (`self._model`)."""
        feature_frame = pd.DataFrame(features).reset_index(drop=True)
        target_series = pd.Series(target).reset_index(drop=True)
        return self._walk_forward_folds_detail(feature_frame, target_series)

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

    def explain(self, current_features: dict[str, float]) -> dict[str, float]:
        """Q3 — znakowane wkłady per-cecha w surową predykcję (SHAP-lite).

        Liczone lokalnie z boostera przez `pred_contribs=True` (dokładne wartości
        SHAP dla modeli drzewiastych) — zero nowych płatnych wywołań. Zwraca
        `{nazwa_cechy: znakowany_wkład}` w jednostkach targetu (zwrot 12h);
        dodatni wkład pchnął prognozę w górę, ujemny w dół. Bias (base value)
        jest pod osobnym kluczem `_bias`, NIE jako cecha — dzięki temu suma
        wszystkich wartości słownika ≈ surowy zwrot modelu (addytywność SHAP).
        Wejście jest wyrównane do `feature_names_in_` tak samo jak w `predict()`
        (nadmiarowe cechy ignorowane, brakujące → KeyError). Bramka „model
        wytrenowany" identyczna jak w `predict()`.
        """
        if self._model is None:
            raise RuntimeError(
                f"Model not trained — no weights at '{self._model_path}'. "
                "Run TrainModelUseCase first (Slow Loop)."
            )
        frame = self._align_features(current_features)
        booster = self._model.get_booster()
        # pred_contribs zwraca macierz [n_próbek, n_cech + 1]; ostatnia kolumna to
        # bias (base value). Mamy jedną próbkę → bierzemy wiersz 0.
        contributions = booster.predict(
            xgb.DMatrix(frame), pred_contribs=True
        )
        row = np.asarray(contributions, dtype=float).reshape(-1)
        feature_names = [str(c) for c in frame.columns]
        result: dict[str, float] = {
            name: float(row[idx]) for idx, name in enumerate(feature_names)
        }
        # Ostatni element to bias — pod dedykowanym kluczem, nie jako cecha.
        result["_bias"] = float(row[len(feature_names)])
        return result

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


def _expanding_window_folds(n_samples: int) -> list[tuple[int, int]]:
    """Zwraca granice (train_end, valid_end) dla foldów expanding-window.

    Dane są time-ordered: każdy fold trenuje na [0, train_end) i waliduje na
    [train_end, valid_end). Kolejne foldy rozszerzają okno treningowe (expanding)
    — replikuje to produkcyjny scenariusz „trenuj na przeszłości, predykuj
    przyszłość". Poniżej `MIN_FOLD_SAMPLES` zwracamy jeden split (sane fallback),
    żeby nie ciąć i tak małej próbki na bezsensownie cienkie foldy.
    """
    single_split = _single_split_bounds(n_samples)
    if n_samples < MIN_FOLD_SAMPLES:
        return [single_split]

    # Walidacyjny blok per fold; reszta na początku idzie do pierwszego treningu.
    block = n_samples // (WALK_FORWARD_FOLDS + 1)
    if block < 1:
        return [single_split]

    folds: list[tuple[int, int]] = []
    for fold_idx in range(1, WALK_FORWARD_FOLDS + 1):
        train_end = block * fold_idx
        valid_end = train_end + block if fold_idx < WALK_FORWARD_FOLDS else n_samples
        if valid_end > train_end and train_end >= 1:
            folds.append((train_end, valid_end))
    return folds or [single_split]


def _single_split_bounds(n_samples: int) -> tuple[int, int]:
    """Granice pojedynczego chronologicznego splitu 20% (fallback / mała próbka)."""
    validation_size = max(MIN_VALIDATION_SAMPLES, int(n_samples * VALIDATION_FRACTION))
    if validation_size >= n_samples:
        raise ValueError("Not enough samples to create validation holdout.")
    train_end = n_samples - validation_size
    return (train_end, n_samples)


def _feature_distribution_stats(
    features: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Per-feature mean/std — prosty drift-detector rozkładu cech wejściowych.

    Porównanie tych statystyk między runami pozwala wykryć dryf danych (zmianę
    rozkładu cech), który może cicho degradować model mimo „zielonej" bramki."""
    stats: dict[str, dict[str, float]] = {}
    for column in features.columns:
        series = pd.to_numeric(features[column], errors="coerce")
        stats[str(column)] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
        }
    return stats


def _rmse(actual: pd.Series, predicted: Any) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual_arr - predicted_arr) ** 2)))


def _directional_hit_rate(actual: pd.Series, predicted: Any) -> float:
    """Frakcja trafień KIERUNKU zwrotu (znak przewidzianego = znak rzeczywistego).

    Uzupełnia RMSE: model może mieć niskie RMSE, a wciąż mylić kierunek ruchu.
    Liczony na tym samym holdoucie foldu co `candidate_holdout_rmse`."""
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    if actual_arr.size == 0:
        return 0.0
    return float(np.mean(np.sign(actual_arr) == np.sign(predicted_arr)))
