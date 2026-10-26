from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.application.errors import MalformedPayload
from src.application.ports import (
    MLPredictionPort,
    ReturnScaleUnavailableError,
    ShadowModelPort,
)
from src.domain.backtest import (
    BacktestSummary,
    WalkForwardFold,
    summarize_folds,
    walk_forward_folds,
)
from src.domain.direction_variant import (
    DirectionReport,
    direction_label,
    summarize_direction_folds,
)
from src.domain.drift import bin_fractions, quantile_edges, recent_window_psi
from src.domain.model_scorecard import TargetUnit, TrainingOutcome, TrainingReport
from src.domain.promotion import ForecastKind, ShadowForecast
from src.domain.training_provenance import (
    DatasetDescription,
    TrainingProvenance,
    dataset_fingerprint,
)
from src.domain.value_objects import Money
from src.infrastructure.error_translation import as_port_error

logger = logging.getLogger(__name__)

# Hiperparametry dobrane pod dane finansowe (zapobieganie overfittingowi).
# ML-22: ziarno jawne — `subsample`/`colsample_bytree` losują, a te same dane z tym
# samym ziarnem mają dać identyczny booster. 0 to dotychczasowy domyślny seed
# XGBoost, więc wagi są bajt w bajt takie jak przed jego dopisaniem.
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 100,
    "random_state": 0,
}

# ML-22: wersja biblioteki w karcie i sidecarze — inna wersja może dać inny booster.
LIBRARY_VERSION = f"xgboost {xgb.__version__}"

# ML-30: wariant binarny — te same drzewa i ziarno, cel logistyczny P(wzrost).
DIRECTION_PARAMS: dict[str, Any] = {**DEFAULT_PARAMS, "objective": "binary:logistic"}
# Przycięcie prawdopodobieństwa w log loss (pewność 0/1 przy pudle = nieskończoność).
_LOG_LOSS_EPS = 1e-6

# ML-10: cecha z kierunkiem LLM (kontrakt `ML_FEATURE_COLUMNS`) — z niej powstaje
# baseline „sam LLM". Bez tej kolumny w danych baseline jest pomijany.
LLM_SIGNAL_FEATURE = "llm_trend_signal"

# Komunikat dołączany do wyniku: raportowane metryki dotyczą KANDYDATA na
# holdoucie (foldach), a shipowany model jest refitowany na WSZYSTKICH danych
# i nie jest re-walidowany — więc nie traktuj tych liczb jak jakości produkcyjnej.
METRIC_NOTE = (
    "candidate holdout metrics (walk-forward folds); shipped model refit on "
    "all data and not re-validated"
)

# ML-11: atrybut boostera zapisywany w pliku `.ubj` — jednostka targetu modelu.
# Plik bez atrybutu (sprzed ML-11) to model w surowych zwrotach.
TARGET_UNIT_ATTR = "target_unit"


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
        self._direction_model: xgb.XGBClassifier | None = None
        self._target_unit = TargetUnit.RETURN
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not Path(self._model_path).exists():
            return
        model = xgb.XGBRegressor()
        model.load_model(self._model_path)
        self._model = model
        self._target_unit = _model_target_unit(model)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(
        self,
        features: Sequence[Mapping[str, float | None]],
        target: Sequence[float],
        *,
        persist: bool = True,
        timestamps: Sequence[datetime] | None = None,
        target_unit: TargetUnit = TargetUnit.RETURN,
        dataset: DatasetDescription | None = None,
        label_ends: Sequence[datetime] | None = None,
        folds: Sequence[WalkForwardFold] | None = None,
    ) -> TrainingReport:
        feature_frame = _feature_frame(features)
        target_series = pd.Series(target).reset_index(drop=True)
        stamps = _global_time_order(timestamps, len(target_series))
        ends = _label_end_order(label_ends, stamps)
        provenance = self._provenance(feature_frame, target_series, stamps, dataset)

        # Walk-forward (expanding window) zamiast jednego splitu 20%: bramka
        # „pobij baseline" opiera się na ŚREDNIEJ z foldów, nie na jednym
        # hałaśliwym estymatorze z jednego reżimu rynku (finding #23). Sama reguła
        # (margines 2%, min. punktów OOS) żyje w domenie — `summarize_folds`.
        summary = summarize_folds(
            self._walk_forward_folds_detail(feature_frame, target_series, stamps, ends, folds=folds)
        )
        # Per-feature drift stats (mean/std, ML-21: biny kwantylowe i PSI ostatnich
        # 30 dni) — obserwowalne niezależnie od werdyktu bramki.
        feature_stats = _feature_distribution_stats(feature_frame, stamps)

        def report(
            outcome: TrainingOutcome,
            *,
            reason: str | None = None,
            model_path: str | None = None,
            weights_sha256: str | None = None,
        ) -> TrainingReport:
            return _training_report(
                outcome,
                summary,
                n_samples=len(target_series),
                feature_stats=feature_stats,
                reason=reason,
                model_path=model_path,
                target_unit=target_unit,
                weights_sha256=weights_sha256,
                provenance=provenance,
            )

        if not persist:
            # ML-13: per-symbol ewaluacja — żadnego zapisu ani podmiany modelu w
            # pamięci; jedyne wagi produkuje model wspólny.
            return report(
                TrainingOutcome.EVALUATION_PASSED
                if summary.beats_baseline
                else TrainingOutcome.EVALUATION_FAILED,
                reason=f"Evaluation only, weights not written: {summary.verdict}.",
            )

        if not summary.beats_baseline:
            return report(
                TrainingOutcome.REJECTED,
                reason=(
                    "Candidate did not pass the walk-forward gate against the "
                    "strongest baseline (zero return, mean drift, LLM only): "
                    f"{summary.verdict}."
                ),
                model_path=self._model_path,
            )

        # Shipowany model: refit na WSZYSTKICH danych (train+holdout) — NIE jest
        # re-walidowany, więc raportowane RMSE/hit-rate to metryki KANDYDATA
        # (finding #22, patrz METRIC_NOTE).
        model = xgb.XGBRegressor(**self._params)
        model.fit(feature_frame, target_series)
        if target_unit is not TargetUnit.RETURN:
            # ML-11: jednostka targetu jedzie w pliku wag — Fast Loop wie z modelu,
            # czy mnożyć prognozę przez σ. Model w zwrotach zostaje bez atrybutu.
            model.get_booster().set_attr(**{TARGET_UNIT_ATTR: target_unit.value})
        model_path = Path(self._model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(self._model_path)
        self._model = model
        self._target_unit = target_unit
        # CI-13: hash bajtów pliku na dysku — tego samego, który workflow wysyła do
        # Release `model-latest`, a Fast Loop haszuje jako `model_version`.
        accepted = report(
            TrainingOutcome.ACCEPTED,
            model_path=self._model_path,
            weights_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        )
        _write_sidecar(model_path, accepted, target_unit)
        return accepted

    def _provenance(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        stamps: list[datetime] | None,
        dataset: DatasetDescription | None,
    ) -> TrainingProvenance:
        """ML-22: odcisk wierszy, na których trenujemy, zakres dat, parametry i wersja
        biblioteki; resztę opisu próbki (`dataset`) zna wołający."""
        moments: list[datetime | None] = (
            list(stamps) if stamps is not None else [None] * len(target)
        )
        rows = zip(
            moments,
            (float(value) for value in target),
            features.to_dict("records"),
            strict=True,
        )
        return TrainingProvenance(
            dataset_sha256=dataset_fingerprint(rows),
            data_start=stamps[0] if stamps else None,
            data_end=stamps[-1] if stamps else None,
            params=dict(self._params),
            library_version=LIBRARY_VERSION,
            dataset=dataset or DatasetDescription(),
        )

    def booster_dump(self) -> str:
        """ML-22: zrzut drzew wczytanego modelu (JSON) — porównanie z refitem."""
        if self._model is None:
            raise RuntimeError(f"Model not trained — no weights at '{self._model_path}'.")
        return _dump(self._model)

    @staticmethod
    def refit_booster_dump(
        features: Sequence[Mapping[str, float | None]],
        target: Sequence[float],
        *,
        params: Mapping[str, object],
    ) -> str:
        """ML-22: refit shipowanego modelu (wszystkie wiersze, te parametry) i zrzut
        jego drzew — bez zapisu i bez zmiany stanu adaptera."""
        model = xgb.XGBRegressor(**params)
        model.fit(_feature_frame(features), pd.Series(target).reset_index(drop=True))
        return _dump(model)

    def _walk_forward_folds_detail(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        timestamps: list[datetime] | None = None,
        label_ends: list[datetime] | None = None,
        *,
        folds: Sequence[WalkForwardFold] | None = None,
    ) -> list[dict[str, float]]:
        """Per-fold metryki walk-forward (expanding window) na danych
        time-ordered. Jedno źródło prawdy dla bramki `train()` (przez domenowe
        `summarize_folds`) i `walk_forward_report` (szczegół).

        ARCH-27: granice foldów i wiersze treningu (purge ML-26 przy `timestamps`)
        projektuje domena (`walk_forward_folds`); adapter tylko trenuje i mierzy —
        na `folds` od wołającego albo, bez nich, na foldach domeny nad tymi wierszami.

        ML-10: baseline'y „dryf średniej" i „sam LLM" biorą skalę WYŁĄCZNIE z
        części treningowej foldu (tej samej, którą widzi kandydat) — żadnego
        podglądu walidacji.
        """
        out: list[dict[str, float]] = []
        designed = (
            walk_forward_folds(len(target), timestamps, label_ends)
            if folds is None
            else _fitting_folds(folds, len(target))
        )
        for fold in designed:
            rows, train_end, valid_end = list(fold.train_rows), fold.train_end, fold.valid_end
            train_x = features.iloc[rows]
            train_y = target.iloc[rows]
            valid_x = features.iloc[train_end:valid_end]
            valid_y = target.iloc[train_end:valid_end]

            candidate = xgb.XGBRegressor(**self._params)
            candidate.fit(train_x, train_y)
            preds = candidate.predict(valid_x)

            out.append(
                {
                    **_training_part_baselines(train_y, valid_x, valid_y),
                    **_paired_loss_sums(train_y, valid_x, valid_y, preds),
                    "train_end": float(train_end),
                    "valid_end": float(valid_end),
                    "n_train": float(len(rows)),
                    "n_valid": float(valid_end - train_end),
                    "candidate_rmse": _rmse(valid_y, preds),
                    # Baseline persystencji: „zwrot = 0" (cena 12h = cena bieżąca).
                    "baseline_rmse": _rmse(valid_y, np.zeros(len(valid_y))),
                    "hit_rate": _directional_hit_rate(valid_y, preds),
                    # ML-20: umiejętność sygnału obok RMSE — porządek i znak.
                    "ic": _rank_ic(valid_y, preds),
                    "sign_strategy_return": _sign_strategy_return(valid_y, preds),
                    "turnover": _sign_turnover(preds),
                }
            )
        return out

    def walk_forward_report(
        self,
        features: Sequence[Mapping[str, float | None]],
        target: Sequence[float],
        *,
        timestamps: Sequence[datetime] | None = None,
        label_ends: Sequence[datetime] | None = None,
        folds: Sequence[WalkForwardFold] | None = None,
    ) -> list[dict[str, float]]:
        """Per-fold metryki OOS bez persystencji (offline backtest harness).

        Reużywa logikę foldów `train()` (finding #23), łącznie z purgem etykiet
        przy podanych `timestamps` (i `label_ends`, ML-26) albo gotowymi `folds`
        (ARCH-27). Czysto ewaluacyjna — nie zapisuje modelu i nie zmienia stanu
        adaptera (`self._model`)."""
        feature_frame = _feature_frame(features)
        target_series = pd.Series(target).reset_index(drop=True)
        stamps = _global_time_order(timestamps, len(target_series))
        return self._walk_forward_folds_detail(
            feature_frame,
            target_series,
            stamps,
            _label_end_order(label_ends, stamps),
            folds=folds,
        )

    def train_direction_classifier(
        self,
        features: Sequence[Mapping[str, float | None]],
        target: Sequence[float],
        *,
        dead_band: float,
        persist: bool = True,
        timestamps: Sequence[datetime] | None = None,
        label_ends: Sequence[datetime] | None = None,
    ) -> DirectionReport:
        """ML-30: klasyfikator P(zwrot > +pasmo) na wierszach poza martwą strefą,
        walk-forward z tym samym purgem co regresja i bramką Briera względem stopy
        bazowej (domena). Zapis tylko po przejściu bramki i tylko do pliku wariantu
        (`*_direction.ubj`) — wagi championa i `self._model` zostają nietknięte."""
        labels = [direction_label(float(value), dead_band=dead_band) for value in target]
        keep = [index for index, label in enumerate(labels) if label is not None]
        frame = _feature_frame(features).iloc[keep].reset_index(drop=True)
        outcome_y = pd.Series([labels[index] for index in keep], dtype=int)
        stamps = _global_time_order(timestamps, len(labels))
        ends = _label_end_order(label_ends, stamps)
        summary = summarize_direction_folds(
            self._direction_folds(
                frame,
                outcome_y,
                None if stamps is None else [stamps[index] for index in keep],
                None if ends is None else [ends[index] for index in keep],
            )
        )

        def report(outcome: TrainingOutcome, model_path: str | None = None) -> DirectionReport:
            return DirectionReport(
                outcome,
                n_samples=len(keep),
                n_in_dead_band=len(labels) - len(keep),
                dead_band=dead_band,
                summary=summary,
                reason=summary.verdict,
                model_path=model_path,
            )

        if not persist:
            return report(
                TrainingOutcome.EVALUATION_PASSED
                if summary.beats_base_rate
                else TrainingOutcome.EVALUATION_FAILED
            )
        if not summary.beats_base_rate:
            return report(TrainingOutcome.REJECTED)
        model = xgb.XGBClassifier(**DIRECTION_PARAMS)
        model.fit(frame, outcome_y)
        path = _direction_model_path(self._model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))
        self._direction_model = model
        return report(TrainingOutcome.ACCEPTED, model_path=str(path))

    def _direction_folds(
        self,
        features: pd.DataFrame,
        outcome_y: pd.Series,
        timestamps: list[datetime] | None,
        label_ends: list[datetime] | None,
    ) -> list[dict[str, float]]:
        """Sumy foldu klasyfikatora: kwadraty błędu i log loss kandydata oraz
        prognozy stopy bazowej części treningowej. Fold z jedną klasą w treningu
        nie ma czego uczyć — pomijany."""
        out: list[dict[str, float]] = []
        for fold in walk_forward_folds(len(outcome_y), timestamps, label_ends):
            rows, train_end, valid_end = list(fold.train_rows), fold.train_end, fold.valid_end
            train_y = outcome_y.iloc[rows]
            if train_y.nunique() < 2:
                continue
            candidate = xgb.XGBClassifier(**DIRECTION_PARAMS)
            candidate.fit(features.iloc[rows], train_y)
            actual = np.asarray(outcome_y.iloc[train_end:valid_end], dtype=float)
            proba = candidate.predict_proba(features.iloc[train_end:valid_end])[:, 1]
            base = np.full(len(actual), float(train_y.mean()))
            out.append(
                {
                    "n_valid": float(len(actual)),
                    "sum_sq_error": float(((proba - actual) ** 2).sum()),
                    "base_sum_sq_error": float(((base - actual) ** 2).sum()),
                    "sum_log_loss": _sum_log_loss(proba, actual),
                    "base_sum_log_loss": _sum_log_loss(base, actual),
                }
            )
        return out

    def predict_proba(self, current_features: dict[str, float]) -> float:
        """ML-30: P(zwrot 12 h > +martwa strefa) z wag wariantu binarnego."""
        model = self._direction_model or self._load_direction_model()
        expected = [str(name) for name in getattr(model, "feature_names_in_", [])]
        frame = (
            pd.DataFrame([{name: current_features[name] for name in expected}])
            if expected
            else pd.DataFrame([current_features])
        )
        return float(model.predict_proba(frame)[0][1])

    def _load_direction_model(self) -> xgb.XGBClassifier:
        path = _direction_model_path(self._model_path)
        if not path.exists():
            raise RuntimeError(
                f"No direction classifier weights at '{path}' — the ML-30 variant "
                "has not passed its gate yet."
            )
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        self._direction_model = model
        return model

    def predict(
        self,
        current_features: dict[str, float],
        current_price: Decimal,
        *,
        return_scale: float | None = None,
    ) -> Money:
        if self._model is None:
            raise RuntimeError(
                f"Model not trained — no weights at '{self._model_path}'. "
                "Run TrainModelUseCase first (Slow Loop)."
            )
        scale = self._return_scale(return_scale)
        frame = self._align_features(current_features)
        # Model przewiduje ZWROT 12h (ML-11: albo zwrot / σ — wtedy × σ);
        # rekonstruujemy cenę bezwzględną: cena_docelowa = cena_bieżąca * (1 + zwrot).
        predicted_return = float(self._model.predict(frame)[0]) * scale
        target_price = current_price * (Decimal("1") + Decimal(str(predicted_return)))
        return Money(target_price)

    def _return_scale(self, return_scale: float | None) -> float:
        """ML-11: mnożnik prognozy — σ dla modelu w jednostkach σ, 1 dla zwrotów."""
        return _scale_for(self._target_unit, return_scale, self._model_path)

    def explain(
        self, current_features: dict[str, float], *, return_scale: float | None = None
    ) -> dict[str, float]:
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
        scale = self._return_scale(return_scale)
        frame = self._align_features(current_features)
        booster = self._model.get_booster()
        # pred_contribs zwraca macierz [n_próbek, n_cech + 1]; ostatnia kolumna to
        # bias (base value). Mamy jedną próbkę → bierzemy wiersz 0.
        contributions = booster.predict(xgb.DMatrix(frame), pred_contribs=True)
        row = np.asarray(contributions, dtype=float).reshape(-1)
        feature_names = [str(c) for c in frame.columns]
        result: dict[str, float] = {
            name: float(row[idx]) * scale for idx, name in enumerate(feature_names)
        }
        # Ostatni element to bias — pod dedykowanym kluczem, nie jako cecha.
        result["_bias"] = float(row[len(feature_names)]) * scale
        return result

    def _align_features(self, current_features: dict[str, float]) -> pd.DataFrame:
        """Wyrównuje wejście do cech, na których model BYŁ trenowany.

        Bierze dokładnie `feature_names_in_` w kolejności modelu — ignoruje
        nadmiarowe cechy (np. nowa cecha w ML_FEATURE_COLUMNS, której stary
        model w repo jeszcze nie zna) i jawnie zgłasza brakujące (KeyError),
        zamiast wpychać NaN albo wywalać się na mismatchu nazw kolumn.
        Dzięki temu kontrakt cech może ewoluować bez psucia predykcji.
        """
        return _aligned_frame(self._model, current_features)


def _scale_for(target_unit: TargetUnit, return_scale: float | None, model_path: str) -> float:
    """ML-11: mnożnik prognozy — σ dla modelu w jednostkach σ, 1 dla zwrotów."""
    if target_unit is TargetUnit.RETURN:
        return 1.0
    if return_scale is None or not math.isfinite(return_scale) or return_scale <= 0:
        raise ReturnScaleUnavailableError(
            f"Model at '{model_path}' predicts return / sigma; no sigma given"
        )
    return return_scale


def _aligned_frame(model: Any, current_features: Mapping[str, float]) -> pd.DataFrame:
    """Jednowierszowa ramka w kolumnach `feature_names_in_` modelu (nadmiarowe cechy
    pominięte, brakujące → KeyError) — kontrakt `XGBoostAdapter._align_features`."""
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return pd.DataFrame([dict(current_features)])
    expected_cols = [str(c) for c in expected]
    missing = [c for c in expected_cols if c not in current_features]
    if missing:
        raise KeyError(f"Missing required ML features for prediction: {missing}")
    return pd.DataFrame([{c: current_features[c] for c in expected_cols}])


def _model_target_unit(model: Any) -> TargetUnit:
    """ML-11: jednostka targetu zapisana w pliku wag; brak atrybutu = surowe zwroty."""
    return TargetUnit(model.get_booster().attr(TARGET_UNIT_ATTR) or TargetUnit.RETURN.value)


def _feature_frame(features: Sequence[Mapping[str, float | None]]) -> pd.DataFrame:
    """Ramka cech z sekwencji wierszy `{cecha: wartość}` (ARCH-09, ML-55: port nie
    niesie DataFrame — ani w `train`, ani w `walk_forward_report`).

    ML-52: brak (None) — także w całej kolumnie, np. cechy makro bez FRED — staje się
    NaN, czyli brakiem XGBoost. Kolumny już liczbowe zostają bez zmian."""
    frame = pd.DataFrame(features).reset_index(drop=True)
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return frame


def _training_report(
    outcome: TrainingOutcome,
    summary: BacktestSummary,
    *,
    n_samples: int,
    feature_stats: dict[str, dict[str, Any]],
    reason: str | None,
    model_path: str | None,
    target_unit: TargetUnit,
    weights_sha256: str | None = None,
    provenance: TrainingProvenance | None = None,
) -> TrainingReport:
    """ARCH-09: werdykt i metryki walk-forward jako domenowy `TrainingReport`.

    Metryki to średnie kandydata na foldach (`METRIC_NOTE`); baseline'y ML-10
    („dryf średniej", „sam LLM") są None, gdy foldy ich nie liczyły. ML-11: przy
    targecie w jednostkach σ metryki też są w σ, a raport niesie `target_unit`."""
    return TrainingReport(
        outcome,
        n_samples=n_samples,
        reason=reason,
        n_folds=summary.n_folds,
        n_oos_points=summary.n_oos_points,
        rmse=summary.mean_candidate_rmse,
        baseline_rmse=summary.mean_baseline_rmse,
        drift_baseline_rmse=summary.mean_drift_baseline_rmse,
        llm_baseline_rmse=summary.mean_llm_baseline_rmse,
        hit_rate=summary.mean_hit_rate,
        feature_stats=feature_stats,
        metric_note=METRIC_NOTE,
        model_path=model_path,
        extras=({} if target_unit is TargetUnit.RETURN else {"target_unit": target_unit.value}),
        weights_sha256=weights_sha256,
        provenance=provenance,
    )


def _dump(model: xgb.XGBRegressor) -> str:
    """Drzewa boostera jako jeden tekst JSON — atrybuty pliku (np. jednostka
    targetu) nie wchodzą, więc porównuje się sam model."""
    return "\n".join(model.get_booster().get_dump(dump_format="json"))


def _write_sidecar(model_path: Path, report: TrainingReport, target_unit: TargetUnit) -> None:
    """ML-22: sidecar `.json` obok pliku wag — proweniencja, hash pliku i jednostka
    targetu. Workflow wysyła go do Release razem z wagami."""
    provenance = report.provenance
    record = provenance.as_record() if provenance is not None else {}
    sidecar = {
        "model_file": model_path.name,
        "weights_sha256": report.weights_sha256,
        "target_unit": target_unit.value,
        "n_samples": report.n_samples,
        **record,
    }
    model_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _baseline_predictions(train_y: pd.Series, valid_x: pd.DataFrame, size: int) -> dict[str, Any]:
    """ML-10: prognozy baseline'ów „dryf" i „llm" na walidacji foldu.

    Dryf przewiduje stały średni target części treningowej. „Sam LLM" przewiduje
    `llm_trend_signal` × średni |target| części treningowej — kierunek z LLM,
    skala z historii. Bez kolumny sygnału LLM zostaje tylko dryf.
    """
    predictions: dict[str, Any] = {"drift": np.full(size, float(train_y.mean()))}
    if LLM_SIGNAL_FEATURE in valid_x.columns:
        scale = float(train_y.abs().mean())
        predictions["llm"] = np.asarray(valid_x[LLM_SIGNAL_FEATURE], dtype=float) * scale
    return predictions


def _training_part_baselines(
    train_y: pd.Series, valid_x: pd.DataFrame, valid_y: pd.Series
) -> dict[str, float]:
    """ML-10: RMSE baseline'ów „dryf średniej" i „sam LLM" na walidacji foldu."""
    return {
        f"{name}_baseline_rmse": _rmse(valid_y, preds)
        for name, preds in _baseline_predictions(train_y, valid_x, len(valid_y)).items()
    }


def _paired_loss_sums(
    train_y: pd.Series, valid_x: pd.DataFrame, valid_y: pd.Series, preds: Any
) -> dict[str, float]:
    """EXP-04: Σd i Σd² foldu, d = strata² baseline'u − strata² kandydata na punkt.

    Klucze bez prefiksu dotyczą „zwrot 0", z prefiksem `drift_` / `llm_` — tych
    baseline'ów; test sparowany liczy z nich domena (`paired_loss_z`)."""
    actual = np.asarray(valid_y, dtype=float)
    candidate_loss = (actual - np.asarray(preds, dtype=float)) ** 2
    baselines = {"": np.zeros(len(actual))}
    for name, baseline in _baseline_predictions(train_y, valid_x, len(actual)).items():
        baselines[f"{name}_"] = baseline
    sums: dict[str, float] = {}
    for prefix, baseline in baselines.items():
        diff = (actual - baseline) ** 2 - candidate_loss
        sums[f"{prefix}sum_loss_diff"] = float(diff.sum())
        sums[f"{prefix}sum_loss_diff_sq"] = float((diff**2).sum())
    return sums


def _global_time_order(
    timestamps: Sequence[datetime] | None, n_samples: int
) -> list[datetime] | None:
    """Znormalizowane (UTC) znaczniki czasu wierszy albo None (trening bez purgu).

    Walk-forward modelu wspólnego musi iść po czasie GLOBALNYM — nieposortowane
    wejście (np. bloki per symbol) walidowałoby na chwilach sprzed treningu."""
    if timestamps is None:
        return None
    parsed = pd.to_datetime(pd.Series(timestamps), utc=True).reset_index(drop=True)
    if len(parsed) != n_samples:
        raise ValueError("timestamps must align with the training rows")
    if not parsed.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted ascending in global time")
    return [stamp.to_pydatetime() for stamp in parsed]


def _direction_model_path(model_path: str) -> Path:
    """ML-30: plik wag wariantu binarnego obok pliku championa (`*_direction.ubj`)."""
    path = Path(model_path)
    return path.with_name(f"{path.stem}_direction{path.suffix}")


def _sum_log_loss(probability: Any, actual: Any) -> float:
    """Suma ujemnej log-wiarygodności z prawdopodobieństwem przyciętym do [eps, 1−eps]."""
    clipped = np.clip(np.asarray(probability, dtype=float), _LOG_LOSS_EPS, 1 - _LOG_LOSS_EPS)
    hits = np.asarray(actual, dtype=float)
    return float(-(hits * np.log(clipped) + (1 - hits) * np.log(1 - clipped)).sum())


def _fitting_folds(folds: Sequence[WalkForwardFold], n_samples: int) -> Sequence[WalkForwardFold]:
    """ARCH-27: foldy od wołającego, o ile mieszczą się w wierszach — walidacja poza
    próbką po cichu skróciłaby się do jej końca i fałszowała `n_valid`."""
    if any(fold.valid_end > n_samples for fold in folds):
        raise ValueError(f"fold validation must end within the {n_samples} rows")
    return folds


def _label_end_order(
    label_ends: Sequence[datetime] | None, stamps: list[datetime] | None
) -> list[datetime] | None:
    """ML-26: chwile końca etykiet (UTC) wyrównane z wierszami albo None (stały
    horyzont). Koniec etykiety bez chwili wiersza nie ma odniesienia — błąd."""
    if label_ends is None:
        return None
    if stamps is None:
        raise ValueError("label_ends need row timestamps")
    parsed = pd.to_datetime(pd.Series(list(label_ends)), utc=True)
    if len(parsed) != len(stamps):
        raise ValueError("label_ends must align with the training rows")
    return [stamp.to_pydatetime() for stamp in parsed]


def _feature_distribution_stats(
    features: pd.DataFrame, stamps: list[datetime] | None = None
) -> dict[str, dict[str, Any]]:
    """Per-feature rozkład cech wejściowych — drift-detector (ML-21).

    `mean`/`std` jak dawniej, plus biny kwantylowe całej próbki (`edges`,
    `fractions`) — odniesienie, z którym da się porównać późniejszy rozkład —
    oraz `psi_recent_30d`: PSI ostatnich 30 dni próbki względem jej starszej
    części (domena `recent_window_psi`). Bez znaczników czasu albo przy zbyt małej
    próbce PSI jest None. Dryf może cicho degradować model mimo „zielonej" bramki."""
    stats: dict[str, dict[str, Any]] = {}
    for column in features.columns:
        series = pd.to_numeric(features[column], errors="coerce")
        values = [float(value) for value in series]
        edges = quantile_edges(values)
        stats[str(column)] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "edges": list(edges),
            "fractions": list(bin_fractions(values, edges)),
            "psi_recent_30d": (recent_window_psi(values, stamps) if stamps is not None else None),
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


def _rank_ic(actual: pd.Series, predicted: Any) -> float:
    """IC: korelacja rang Spearmana prognozy z realizacją (Pearson na rangach).

    Stała seria (prognoza albo realizacja) nie ma porządku — korelacja jest
    wtedy niezdefiniowana i zwracamy 0.0 („brak umiejętności rankingu"), żeby NaN
    nie zatruł średniej IC w `summarize_folds`."""
    actual_ranks = pd.Series(np.asarray(actual, dtype=float)).rank()
    predicted_ranks = pd.Series(np.asarray(predicted, dtype=float)).rank()
    if actual_ranks.nunique() < 2 or predicted_ranks.nunique() < 2:
        return 0.0
    ic = actual_ranks.corr(predicted_ranks)
    return 0.0 if pd.isna(ic) else float(ic)


def _sign_strategy_return(actual: pd.Series, predicted: Any) -> float:
    """Średni zwrot strategii „pozycja = znak prognozy": mean(sign(pred) × target)."""
    actual_arr = np.asarray(actual, dtype=float)
    if actual_arr.size == 0:
        return 0.0
    return float(np.mean(np.sign(np.asarray(predicted, dtype=float)) * actual_arr))


def _sign_turnover(predicted: Any) -> float:
    """Średnia zmiana pozycji znakowej między kolejnymi wierszami foldu ∈ [0, 1].

    Odwrócenie long↔short liczy się jako 1, wejście/wyjście z zera jako 0.5.
    Wiersze idą w kolejności czasu; w modelu wspólnym kolejne wiersze to różne
    symbole, więc tam to tylko przybliżenie obrotu."""
    signs = np.sign(np.asarray(predicted, dtype=float))
    if signs.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(signs)) / 2))


# ML-33: cel boostera, który pasuje do rodzaju challengera — plik innego rodzaju
# (np. klasyfikator podany jako regresor) dałby liczbę w złych jednostkach.
_CHALLENGER_OBJECTIVES: dict[ForecastKind, tuple[str, ...]] = {
    ForecastKind.RETURN: ("reg:",),
    ForecastKind.DIRECTION: ("binary:logistic",),
}


class XGBoostChallenger(ShadowModelPort):
    """ML-33: challenger z własnego pliku wag — regresor zwrotu (`kind = return`)
    albo klasyfikator kierunku ML-30 (`kind = direction`).

    Osobny obiekt i osobny booster: champion (`XGBoostAdapter`) i jego wagi w
    pamięci nie są dotykane. Wersja to sha256 bajtów pliku z chwili wczytania.
    Buduje go `load_challenger`, który odrzuca brakujący, uszkodzony albo
    niepasujący rodzajem plik."""

    def __init__(
        self, name: str, model_path: str, kind: ForecastKind, model: Any, version: str
    ) -> None:
        self._name = name
        self._model_path = model_path
        self._kind = kind
        self._model = model
        self._model_version = version
        self._target_unit = _model_target_unit(model)

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_version(self) -> str:
        return self._model_version

    def forecast(
        self, features: Mapping[str, float], *, return_scale: float | None = None
    ) -> ShadowForecast:
        try:
            frame = _aligned_frame(self._model, features)
            if self._kind is ForecastKind.DIRECTION:
                return ShadowForecast(
                    self._name,
                    self._model_version,
                    proba=float(self._model.predict_proba(frame)[0][1]),
                )
            scale = _scale_for(self._target_unit, return_scale, self._model_path)
            return ShadowForecast(
                self._name,
                self._model_version,
                predicted_return=float(self._model.predict(frame)[0]) * scale,
            )
        except (KeyError, ValueError) as exc:
            # Brak cechy modelu, ramka odrzucona przez XGBoost (XGBoostError to
            # ValueError) albo prognoza poza kontraktem (NaN, p spoza [0, 1]).
            raise as_port_error(exc, MalformedPayload) from None


def load_challenger(name: str, model_path: str, kind: ForecastKind) -> XGBoostChallenger | None:
    """ML-33: challenger z pliku wag albo None — brak pliku (Release jeszcze go nie
    ma), plik uszkodzony albo booster innego rodzaju niż `kind`. Każdy powód idzie
    do logu; Fast Loop leci wtedy bez tego challengera."""
    path = Path(model_path)
    if not path.is_file():
        logger.info("ML-33: brak wag challengera %s (%s) — pomijam", name, model_path)
        return None
    model: Any = xgb.XGBRegressor() if kind is ForecastKind.RETURN else xgb.XGBClassifier()
    try:
        version = hashlib.sha256(path.read_bytes()).hexdigest()
        model.load_model(model_path)
        objective = json.loads(model.get_booster().save_config())["learner"]["objective"]["name"]
    except (OSError, ValueError, KeyError, TypeError):
        # TypeError: wrapper sklearn XGBoost odmawia pliku estymatora innego typu
        # (klasyfikator wczytywany jako regresor i odwrotnie).
        logger.warning("ML-33: wag challengera %s (%s) nie da się wczytać", name, model_path)
        return None
    if not str(objective).startswith(_CHALLENGER_OBJECTIVES[kind]):
        logger.warning(
            "ML-33: challenger %s (%s) ma cel %s, a config mówi %s — pomijam",
            name,
            model_path,
            objective,
            kind.value,
        )
        return None
    return XGBoostChallenger(name, model_path, kind, model, version)
