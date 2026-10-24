from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.application.ports import MLPredictionPort, ReturnScaleUnavailableError
from src.domain.backtest import (
    LABEL_HORIZON,
    BacktestSummary,
    purge_training_indices,
    summarize_folds,
)
from src.domain.drift import bin_fractions, quantile_edges, recent_window_psi
from src.domain.model_scorecard import TargetUnit, TrainingOutcome, TrainingReport
from src.domain.training_provenance import (
    DatasetDescription,
    TrainingProvenance,
    dataset_fingerprint,
)
from src.domain.value_objects import Money

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

VALIDATION_FRACTION = 0.2
MIN_VALIDATION_SAMPLES = 10

# Walk-forward (expanding window): liczba foldów i minimalna liczba próbek, poniżej
# której nie ma sensu robić CV (spadamy do pojedynczego splitu). Każdy fold dokłada
# kolejny chronologiczny blok do treningu i waliduje na następnym — to lepszy
# (mniej hałaśliwy) estymator dla niestacjonarnej serii finansowej niż jeden split.
WALK_FORWARD_FOLDS = 4
MIN_FOLD_SAMPLES = 50

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
        self._target_unit = TargetUnit.RETURN
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not Path(self._model_path).exists():
            return
        model = xgb.XGBRegressor()
        model.load_model(self._model_path)
        self._model = model
        self._target_unit = TargetUnit(
            model.get_booster().attr(TARGET_UNIT_ATTR) or TargetUnit.RETURN.value
        )

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
            self._walk_forward_folds_detail(feature_frame, target_series, stamps, ends)
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
            raise RuntimeError(
                f"Model not trained — no weights at '{self._model_path}'."
            )
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
    ) -> list[dict[str, float]]:
        """Per-fold metryki walk-forward (expanding window) na danych
        time-ordered. Jedno źródło prawdy dla bramki `train()` (przez domenowe
        `summarize_folds`) i `walk_forward_report` (szczegół). Poniżej
        `MIN_FOLD_SAMPLES` spada do pojedynczego splitu (sane fallback) → jeden fold.

        Z `timestamps` (czas globalny, rosnąco) trening foldu to
        `purge_training_indices` — bez wierszy, których etykieta kończy się po
        starcie walidacji (ML-26): koniec z `label_ends` (rozliczenie), a bez nich
        stały horyzont 12 h, czyli prefiks jak dotąd. Purge obejmuje też fallback
        single-split. Fold bez żadnego wiersza treningowego po purgu jest pomijany.

        ML-10: baseline'y „dryf średniej" i „sam LLM" biorą skalę WYŁĄCZNIE z
        części treningowej foldu (tej samej, którą widzi kandydat) — żadnego
        podglądu walidacji.
        """
        folds = _expanding_window_folds(len(target))
        ends = (
            None
            if timestamps is None
            else label_ends or [stamp + LABEL_HORIZON for stamp in timestamps]
        )
        out: list[dict[str, float]] = []
        for train_end, valid_end in folds:
            rows = (
                list(range(train_end))
                if timestamps is None or ends is None
                else list(purge_training_indices(timestamps, ends, train_end))
            )
            if not rows:
                continue
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
        features: Any,
        target: Any,
        *,
        timestamps: Any = None,
        label_ends: Sequence[datetime] | None = None,
    ) -> list[dict[str, float]]:
        """Per-fold metryki OOS bez persystencji (offline backtest harness).

        Reużywa logikę foldów `train()` (finding #23), łącznie z purgem etykiet
        przy podanych `timestamps` (i `label_ends`, ML-26). Czysto ewaluacyjna —
        nie zapisuje modelu i nie zmienia stanu adaptera (`self._model`)."""
        feature_frame = _feature_frame(features)
        target_series = pd.Series(target).reset_index(drop=True)
        stamps = _global_time_order(timestamps, len(target_series))
        return self._walk_forward_folds_detail(
            feature_frame, target_series, stamps, _label_end_order(label_ends, stamps)
        )

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
        if self._target_unit is TargetUnit.RETURN:
            return 1.0
        if return_scale is None or not math.isfinite(return_scale) or return_scale <= 0:
            raise ReturnScaleUnavailableError(
                f"Model at '{self._model_path}' predicts return / sigma; no sigma given"
            )
        return return_scale

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
        contributions = booster.predict(
            xgb.DMatrix(frame), pred_contribs=True
        )
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


def _feature_frame(features: Any) -> pd.DataFrame:
    """Ramka cech z sekwencji wierszy `{cecha: wartość}` (ARCH-09: port nie niesie
    DataFrame; `walk_forward_report` backtestu podaje ją jeszcze wprost).

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
        extras=(
            {} if target_unit is TargetUnit.RETURN else {"target_unit": target_unit.value}
        ),
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


def _baseline_predictions(
    train_y: pd.Series, valid_x: pd.DataFrame, size: int
) -> dict[str, Any]:
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


def _global_time_order(timestamps: Any, n_samples: int) -> list[datetime] | None:
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
            "psi_recent_30d": (
                recent_window_psi(values, stamps) if stamps is not None else None
            ),
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
