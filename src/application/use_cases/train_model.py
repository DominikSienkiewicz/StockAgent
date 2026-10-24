from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypedDict

import pandas as pd

from src.application.ml_features import (
    ML_FEATURE_COLUMNS,
    SETTLED_AT_COLUMN,
    SNAPSHOT_TRAINING_FEATURE_SET_VERSION,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    free_model_vector,
    in_label_time_order,
    llm_model_vector,
    usable_for_llm_model,
)
from src.application.ports import MLPredictionPort, RepositoryPort, TrainingDataPort
from src.domain.backtest import LabelOverlap, count_label_overlaps
from src.domain.model_scorecard import (
    FREE_SNAPSHOT_MODEL_SYMBOL,
    LLM_SNAPSHOT_MODEL_SYMBOL,
    TargetUnit,
    TrainingOutcome,
    TrainingReport,
)

# Jawny re-eksport: symbol wiersza karty modelu wspólnego żyje w domenie karty.
from src.domain.model_scorecard import POOLED_MODEL_SYMBOL as POOLED_MODEL_SYMBOL
from src.domain.price_features import SIGMA_LOOKBACK, return_sigma
from src.domain.training_labels import (
    LabeledSnapshot,
    SelectionReport,
    label_snapshots,
    selection_report,
)

logger = logging.getLogger(__name__)

# Snapshoty cen per symbol `(chwila, cena)` — etykieta i σ liczą się z tego szeregu.
PriceSeries = Mapping[str, Sequence[tuple[datetime, float]]]


class _TargetUnitKwargs(TypedDict, total=False):
    """ML-11: opcjonalny argument `target_unit` wywołania `MLPredictionPort.train`."""

    target_unit: TargetUnit


@dataclass(frozen=True)
class SnapshotModelsResult:
    """ML-52: wynik ewaluacji modeli z `feature_snapshots` — dwie karty i raport
    selekcji bramki. `n_snapshots` to wektory symboli treningu w wersji 4,
    `n_labeled` — te z dojrzałą etykietą."""

    free_model: TrainingReport
    llm_model: TrainingReport
    selection: SelectionReport
    n_snapshots: int
    n_labeled: int

# Minimalna liczba próbek, by trening miał sens (zapobiega overfittingowi na 2 rekordach).
# Dla modelu wspólnego (ML-13) dotyczy ŁĄCZNEJ próbki wszystkich symboli.
MIN_SAMPLES_FOR_TRAINING = 50


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

    ML-52: `run_snapshot_models` ewaluuje (bez zapisu wag) dwa modele na wektorach
    `feature_snapshots` z etykietą z `price_snapshots` — odczyt przez `TrainingDataPort`.
    """

    def __init__(
        self,
        ml_port: MLPredictionPort,
        db_port: RepositoryPort,
        *,
        feature_set_version: int | None = None,
        run_id: str | None = None,
        training_data: TrainingDataPort | None = None,
        snapshot_source: bool = False,
        sigma_target: bool = False,
    ) -> None:
        """`feature_set_version` (DAT-09) zawęża próbkę do wierszy tej wersji
        zestawu cech; None = cała historia (bez migracji 027). `run_id` (ML-24)
        to identyfikator przebiegu Slow Loopa w wierszu karty modelu wspólnego."""
        self._ml_port = ml_port
        self._db_port = db_port
        self._feature_set_version = feature_set_version
        self._run_id = run_id
        self._training_data = training_data
        self._snapshot_source = snapshot_source
        self._sigma_target = sigma_target

    def refresh_feature_store(self) -> None:
        self._db_port.refresh_feature_store()

    def run(self, symbol: str, refresh_view: bool = True) -> TrainingReport:
        """Per-symbol ewaluacja walk-forward — nigdy nie zapisuje wag (ML-13).

        Foldy purgują horyzont etykiety po znacznikach czasu wierszy, tak jak model
        wspólny (decyzja 4 z 2026-10-18)."""
        if refresh_view:
            # Świeży snapshot — wymagane do Continual Learning (model nie może
            # uczyć się na starym MATERIALIZED VIEW z poprzedniego tygodnia).
            self._db_port.refresh_feature_store()

        rows = self._feature_rows(symbol)

        if len(rows) < MIN_SAMPLES_FOR_TRAINING:
            return self._skipped(f"Not enough samples for '{symbol}'", len(rows))

        df = _valid_rows(rows)

        if len(df) < MIN_SAMPLES_FOR_TRAINING:
            return self._skipped(f"Not enough valid samples for '{symbol}'", len(df))

        df, stamps = in_label_time_order(df)

        return self._ml_port.train(
            df[list(ML_FEATURE_COLUMNS)],
            df[TARGET_COLUMN],
            persist=False,
            timestamps=stamps,
        )

    def run_pooled(
        self, symbols: Sequence[str], *, record_scorecard: bool = False
    ) -> TrainingReport:
        """ML-13 — jeden wspólny model na wierszach wszystkich `symbols`, jeden zapis.

        Wiersze są sortowane po czasie GLOBALNYM, a walk-forward dostaje ich
        znaczniki czasu (purge horyzontu etykiety) — inaczej fold walidowałby jeden
        symbol na chwilach sprzed treningu na innym. Z `record_scorecard` wynik
        trafia do `model_scorecards` jako `POOLED_MODEL_SYMBOL`, best-effort.
        """
        result = self._train_pooled(symbols)
        if record_scorecard:
            try:
                self._save_pooled_scorecard(result)
            except Exception:
                logger.exception("Failed to persist the pooled model scorecard")
        return result

    def run_snapshot_models(
        self, symbols: Sequence[str], *, record_scorecard: bool = False
    ) -> SnapshotModelsResult:
        """ML-52 — dwa modele ewaluowane na wektorach `feature_snapshots` (wersja 4)
        z etykietą 12 h z `price_snapshots`, bez zapisu wag (serwowany model bez zmian).

        Model cech darmowych uczy się na WSZYSTKICH cyklach z etykietą, model
        z cechami LLM — na cyklach gated bez flag wykluczających. Każdy ma swoją
        kartę i swoje baseline'y walk-forward na własnej próbce. Raport selekcji
        porównuje zwroty cykli gated i non-gated. Za mało wierszy → `SKIPPED`
        z liczbą próbek (pusta baza mówi to wprost)."""
        n_snapshots, labeled, series = self._labeled_snapshots(symbols)
        free_rows = [
            (free_model_vector(item.snapshot.features), item) for item in labeled
        ]
        llm_rows = _llm_rows(labeled)
        result = SnapshotModelsResult(
            free_model=self._evaluate_snapshot_rows(
                free_rows, series, "the free-feature model"
            ),
            llm_model=self._evaluate_snapshot_rows(
                llm_rows, series, "the LLM-feature model (gated cycles)"
            ),
            selection=selection_report(labeled),
            n_snapshots=n_snapshots,
            n_labeled=len(labeled),
        )
        if record_scorecard:
            for symbol, report in (
                (FREE_SNAPSHOT_MODEL_SYMBOL, result.free_model),
                (LLM_SNAPSHOT_MODEL_SYMBOL, result.llm_model),
            ):
                try:
                    self._save_scorecard(symbol, report)
                except Exception:
                    logger.exception("Failed to persist the %s scorecard", symbol)
        return result

    def _labeled_snapshots(
        self, symbols: Sequence[str]
    ) -> tuple[int, list[LabeledSnapshot], PriceSeries]:
        """Wektory symboli treningu w wersji 4 z etykietą point-in-time i szereg cen
        (z zapasem `SIGMA_LOOKBACK` przed najstarszym wektorem na σ, ML-11)."""
        training_data = self._require_training_data()
        wanted = list(dict.fromkeys(symbols))
        snapshots = [
            snapshot
            for snapshot in training_data.get_feature_snapshots(
                SNAPSHOT_TRAINING_FEATURE_SET_VERSION
            )
            if snapshot.symbol in wanted
        ]
        if not snapshots:
            return 0, [], {}
        series = training_data.get_price_series(
            wanted, since=min(snapshot.as_of for snapshot in snapshots) - SIGMA_LOOKBACK
        )
        return len(snapshots), label_snapshots(snapshots, series), series

    def _in_target_units(
        self,
        rows: Sequence[tuple[Mapping[str, float | None], LabeledSnapshot]],
        series: PriceSeries,
    ) -> list[tuple[Mapping[str, float | None], float, datetime]]:
        """Wiersze (wektor, target, chwila); ML-11: target = zwrot / σ z chwili
        wektora, a wiersz bez σ (za krótka historia) wypada."""
        out: list[tuple[Mapping[str, float | None], float, datetime]] = []
        for vector, item in rows:
            target = item.target_return
            if self._sigma_target:
                sigma = return_sigma(series.get(item.snapshot.symbol, []), item.snapshot.as_of)
                if sigma is None:
                    continue
                target /= sigma
            out.append((vector, target, item.snapshot.as_of))
        return out

    def _snapshot_reason(self, model: str) -> str:
        history = " with a sigma history" if self._sigma_target else ""
        return f"Not enough labeled feature snapshots{history} for {model}"

    def _evaluate_snapshot_rows(
        self,
        rows: Sequence[tuple[Mapping[str, float | None], LabeledSnapshot]],
        series: PriceSeries,
        model: str,
    ) -> TrainingReport:
        training_set = self._in_target_units(rows, series)
        if len(training_set) < MIN_SAMPLES_FOR_TRAINING:
            return _skipped(
                self._snapshot_reason(model),
                len(training_set),
                feature_set_version=SNAPSHOT_TRAINING_FEATURE_SET_VERSION,
            )
        return self._ml_port.train(
            [dict(vector) for vector, _, _ in training_set],
            [target for _, target, _ in training_set],
            persist=False,
            timestamps=[as_of for _, _, as_of in training_set],
            **self._target_unit(),
        )

    def _save_pooled_scorecard(self, result: TrainingReport) -> None:
        self._save_scorecard(POOLED_MODEL_SYMBOL, result)

    def _save_scorecard(self, symbol: str, result: TrainingReport) -> None:
        if self._run_id is None:
            self._db_port.save_model_scorecard(symbol, result)
        else:
            self._db_port.save_model_scorecard(symbol, result, run_id=self._run_id)

    def _train_pooled(self, symbols: Sequence[str]) -> TrainingReport:
        if self._snapshot_source:
            return self._train_pooled_from_snapshots(symbols)
        rows = [
            row
            for symbol in dict.fromkeys(symbols)
            for row in self._feature_rows(symbol)
        ]

        if len(rows) < MIN_SAMPLES_FOR_TRAINING:
            return self._pooled_skipped("Not enough samples for the pooled model", len(rows))

        df = _valid_rows(rows)

        if len(df) < MIN_SAMPLES_FOR_TRAINING:
            return self._pooled_skipped(
                "Not enough valid samples for the pooled model", len(df)
            )

        _log_label_overlap(df)
        stamps = pd.to_datetime(df["timestamp"], utc=True)
        df = df.assign(_global_time=stamps).sort_values(
            ["_global_time", "symbol"], kind="stable"
        )
        target = df[TARGET_COLUMN]
        if self._sigma_target:
            sigmas = self._row_sigmas(df["symbol"], df["_global_time"])
            df, target = df[sigmas.notna()], (target / sigmas)[sigmas.notna()]
            if len(df) < MIN_SAMPLES_FOR_TRAINING:
                return self._pooled_skipped(
                    "Not enough samples with a sigma history for the pooled model", len(df)
                )

        return self._ml_port.train(
            df[list(ML_FEATURE_COLUMNS)],
            target,
            timestamps=df["_global_time"],
            **self._target_unit(),
        )

    def _row_sigmas(self, symbols: pd.Series, stamps: pd.Series) -> pd.Series:
        """ML-11: σ symbolu wiersza w chwili wiersza (`return_sigma`); brak = NaN."""
        series = self._require_training_data().get_price_series(
            list(dict.fromkeys(str(symbol) for symbol in symbols)),
            since=stamps.min().to_pydatetime() - SIGMA_LOOKBACK,
        )
        return pd.Series(
            [
                return_sigma(series.get(str(symbol), []), stamp.to_pydatetime())
                for symbol, stamp in zip(symbols, stamps, strict=True)
            ],
            index=symbols.index,
            dtype=float,
        )

    def _target_unit(self) -> _TargetUnitKwargs:
        """ML-11: jednostka targetu w wywołaniu `train` — tylko przy σ (flaga off =
        wywołanie jak przed ML-11)."""
        return {"target_unit": TargetUnit.SIGMA} if self._sigma_target else {}

    def _require_training_data(self) -> TrainingDataPort:
        if self._training_data is None:
            raise RuntimeError("Training from price snapshots needs a TrainingDataPort (DI)")
        return self._training_data


    def _train_pooled_from_snapshots(self, symbols: Sequence[str]) -> TrainingReport:
        """ML-52 (4): model wspólny z cykli gated w `feature_snapshots` (wektor
        z servingu, etykieta z `price_snapshots`) zamiast z widoku SQL. Jedyny zapis
        wag, jak w ścieżce widoku; poniżej minimum — `SKIPPED` z liczbą próbek."""
        _, labeled, series = self._labeled_snapshots(symbols)
        training_set = self._in_target_units(_llm_rows(labeled), series)
        if len(training_set) < MIN_SAMPLES_FOR_TRAINING:
            return _skipped(
                self._snapshot_reason("the pooled model (gated cycles)"),
                len(training_set),
                feature_set_version=SNAPSHOT_TRAINING_FEATURE_SET_VERSION,
            )
        return self._ml_port.train(
            [dict(vector) for vector, _, _ in training_set],
            [target for _, target, _ in training_set],
            timestamps=[as_of for _, _, as_of in training_set],
            **self._target_unit(),
        )

    def _feature_rows(self, symbol: str) -> list[dict[str, Any]]:
        if self._feature_set_version is None:
            return self._db_port.get_feature_store_data(symbol)
        return self._db_port.get_feature_store_data(
            symbol, feature_set_version=self._feature_set_version
        )

    def _pooled_skipped(self, reason: str, n_samples: int) -> TrainingReport:
        """Pominięty model wspólny. DAT-09: odcięcie historii po wersji zestawu cech
        nie może cicho zatrzymać jedynego zapisu wag — ostrzeżenie z liczbą próbek."""
        if self._feature_set_version is not None:
            logger.warning(
                "Pooled model skipped: feature set version %d leaves %s samples "
                "(minimum %d)",
                self._feature_set_version,
                n_samples,
                MIN_SAMPLES_FOR_TRAINING,
            )
        return self._skipped(reason, n_samples)

    def _skipped(self, reason: str, n_samples: int) -> TrainingReport:
        """Pominięcie treningu; przy filtrze wersji (DAT-09) wynik mówi wprost, że
        próbkę obcięto do bieżącej wersji zestawu cech."""
        if self._feature_set_version is None:
            return _skipped(reason, n_samples)
        return _skipped(
            f"{reason} at feature set version {self._feature_set_version}",
            n_samples,
            feature_set_version=self._feature_set_version,
        )


def _llm_rows(
    labeled: Sequence[LabeledSnapshot],
) -> list[tuple[dict[str, float], LabeledSnapshot]]:
    """Cykle gated bez flag wykluczających, z kompletnym wektorem `ML_FEATURE_COLUMNS`."""
    return [
        (vector, item)
        for item in labeled
        if usable_for_llm_model(item.snapshot.gated, item.snapshot.features)
        and (vector := llm_model_vector(item.snapshot.features)) is not None
    ]


def _valid_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Wiersze z kompletem cech i targetu.

    price_delta i target_return przychodzą GOTOWE z widoku (NaN-guarded). Nie
    przeliczamy ich tutaj — stary recompute (current-prev)/prev dublował logikę
    widoku i przy prev=0 dawał +inf, który przeżywał dropna i trafiał do XGBoost.
    dropna odsiewa wiersze, dla których widok zwrócił NULL.
    """
    required_columns = [*ML_FEATURE_COLUMNS, TARGET_COLUMN]
    return pd.DataFrame(rows).dropna(subset=required_columns)


def _label_overlap(df: pd.DataFrame) -> LabelOverlap | None:
    """ML-25: nakładanie etykiet z `settled_at` (widok po migracji 050).

    None = widok bez kolumny (przed migracją) — odczyt `*` jest tolerancyjny."""
    if not {SETTLED_AT_COLUMN, TIMESTAMP_COLUMN, "symbol"} <= set(df.columns):
        return None
    stamps = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True)
    settled = pd.to_datetime(df[SETTLED_AT_COLUMN], utc=True)
    return count_label_overlaps(
        (
            str(symbol),
            stamp.to_pydatetime(),
            None if pd.isna(settled_at) else settled_at.to_pydatetime(),
        )
        for symbol, stamp, settled_at in zip(df["symbol"], stamps, settled, strict=True)
    )


def _log_label_overlap(df: pd.DataFrame) -> None:
    overlap = _label_overlap(df)
    if overlap is None:
        return
    # Purge walk-forward zakłada stały horyzont `LABEL_HORIZON`; ta liczba mówi,
    # jak często prawdziwa etykieta sięga za następny wiersz symbolu.
    logger.info(
        "Pooled model label overlap: %d of %d rows with settled_at settle after "
        "the next row of their symbol (ML-25)",
        overlap.overlapping,
        overlap.measured,
    )


def _skipped(reason: str, n_samples: int, **extras: Any) -> TrainingReport:
    """Wynik pominiętego treningu. `n_samples` jest polem (ML-04), nie tylko tekstem
    powodu — scorecard i logi pokazują, ile próbek brakuje do progu."""
    return TrainingReport(
        TrainingOutcome.SKIPPED,
        n_samples=n_samples,
        reason=f"{reason} ({n_samples} < {MIN_SAMPLES_FOR_TRAINING}).",
        extras=extras,
    )
