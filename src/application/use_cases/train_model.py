from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypedDict

from src.application.errors import PortError
from src.application.ml_features import (
    ML_FEATURE_COLUMNS,
    SNAPSHOT_TRAINING_FEATURE_SET_VERSION,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    Row,
    complete_model_rows,
    free_model_vector,
    llm_model_vector,
    rows_in_label_time_order,
    settled_moments,
    usable_for_llm_model,
)
from src.application.ports import MLPredictionPort, RepositoryPort, TrainingDataPort
from src.domain.backtest import LabelOverlap, count_label_overlaps, label_end_times
from src.domain.direction_variant import DirectionReport, direction_label
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
from src.domain.training_provenance import DatasetDescription

logger = logging.getLogger(__name__)

# Snapshoty cen per symbol `(chwila, cena)` — etykieta i σ liczą się z tego szeregu.
PriceSeries = Mapping[str, Sequence[tuple[datetime, float]]]


class _TargetUnitKwargs(TypedDict, total=False):
    """ML-11: opcjonalny argument `target_unit` wywołania `MLPredictionPort.train`."""

    target_unit: TargetUnit


class _LabelEndsKwargs(TypedDict, total=False):
    """ML-26: opcjonalny argument `label_ends` wywołania `MLPredictionPort.train`."""

    label_ends: list[datetime]


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


@dataclass(frozen=True)
class TrainingSet:
    """ML-22: próbka modelu wspólnego dokładnie w postaci, w jakiej idzie do
    `MLPredictionPort.train` — wiersze cech, target (ML-11: w jednostkach σ przy
    fladze), znaczniki czasu rosnąco i opis próbki do proweniencji. Z niej
    `reproduce-training` odtwarza odcisk zbioru i booster."""

    features: list[dict[str, float | None]]
    target: list[float]
    timestamps: list[datetime]
    dataset: DatasetDescription
    # ML-26: chwile końca etykiet do purgu foldów; None = stały horyzont 12 h.
    label_ends: list[datetime] | None = None


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
    2. Wiersze z kompletem cech i targetu (`complete_model_rows`; widok już liczy
       `price_delta` i `target_return`, z guardem anty-inf po stronie SQL —
       train_model NIE dubluje tej logiki).
    3. Separacja X / Y i wywołanie ml_port.train() — na wierszach, bez DataFrame
       (ML-58, te same pomocniki co backtest).

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
        git_sha: str | None = None,
        settled_purge: bool = False,
    ) -> None:
        """`feature_set_version` (DAT-09) zawęża próbkę do wierszy tej wersji
        zestawu cech; None = cała historia (bez migracji 027). `run_id` (ML-24)
        to identyfikator przebiegu Slow Loopa w wierszu karty modelu wspólnego.
        `git_sha` (ML-22) — commit kodu przebiegu w proweniencji treningu.
        `settled_purge` (ML-26) — foldy purgują po chwili rozliczenia etykiety
        (`settled_at`), nie tylko po stałym horyzoncie 12 h."""
        self._ml_port = ml_port
        self._db_port = db_port
        self._feature_set_version = feature_set_version
        self._run_id = run_id
        self._training_data = training_data
        self._snapshot_source = snapshot_source
        self._sigma_target = sigma_target
        self._git_sha = git_sha
        self._settled_purge = settled_purge

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

        complete = complete_model_rows(rows)

        if len(complete) < MIN_SAMPLES_FOR_TRAINING:
            return self._skipped(f"Not enough valid samples for '{symbol}'", len(complete))

        ordered, moments = rows_in_label_time_order(complete)

        return self._ml_port.train(
            _feature_vectors(ordered),
            _targets(ordered),
            persist=False,
            timestamps=moments,
            dataset=self._describe({symbol: len(ordered)}, self._feature_set_version),
            **self._label_ends(self._view_label_ends(ordered, moments)),
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
        free_rows = [(free_model_vector(item.snapshot.features), item) for item in labeled]
        llm_rows = _llm_rows(labeled)
        result = SnapshotModelsResult(
            free_model=self._evaluate_snapshot_rows(free_rows, series, "the free-feature model"),
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
                except PortError:
                    logger.exception("Failed to persist the %s scorecard", symbol)
        return result

    def run_direction_variant(self, symbols: Sequence[str], *, dead_band: float) -> DirectionReport:
        """ML-30 — wariant binarny P(zwrot > +`dead_band`) na próbce modelu wspólnego
        (surowe zwroty, także przy flagze ML-11), z tym samym purgem foldów. Wagi
        championa się nie zmieniają; poniżej minimum wierszy poza martwą strefą —
        `SKIPPED` z liczbami."""
        prepared = self.pooled_training_set(symbols, sigma_units=False)
        if isinstance(prepared, TrainingReport):
            return DirectionReport(
                TrainingOutcome.SKIPPED,
                n_samples=prepared.n_samples,
                n_in_dead_band=0,
                dead_band=dead_band,
                reason=prepared.reason,
            )
        outside = sum(
            1
            for value in prepared.target
            if direction_label(value, dead_band=dead_band) is not None
        )
        if outside < MIN_SAMPLES_FOR_TRAINING:
            return DirectionReport(
                TrainingOutcome.SKIPPED,
                n_samples=outside,
                n_in_dead_band=len(prepared.target) - outside,
                dead_band=dead_band,
                reason=(
                    f"Not enough rows outside the dead band ±{dead_band:g} for the "
                    f"direction variant ({outside} < {MIN_SAMPLES_FOR_TRAINING})."
                ),
            )
        return self._ml_port.train_direction_classifier(
            prepared.features,
            prepared.target,
            dead_band=dead_band,
            timestamps=prepared.timestamps,
            **self._label_ends(prepared.label_ends),
        )

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
        *,
        sigma_units: bool | None = None,
    ) -> list[tuple[Mapping[str, float | None], float, LabeledSnapshot]]:
        """Wiersze (wektor, target, wektor z etykietą); ML-11: target = zwrot / σ
        z chwili wektora, a wiersz bez σ (za krótka historia) wypada.
        `sigma_units` None = jak flaga ML-11; False = surowy zwrot (ML-30)."""
        in_sigma = self._sigma_target if sigma_units is None else sigma_units
        out: list[tuple[Mapping[str, float | None], float, LabeledSnapshot]] = []
        for vector, item in rows:
            target = item.target_return
            if in_sigma:
                sigma = return_sigma(series.get(item.snapshot.symbol, []), item.snapshot.as_of)
                if sigma is None:
                    continue
                target /= sigma
            out.append((vector, target, item))
        return out

    def _snapshot_training_set(
        self, rows: Sequence[tuple[Mapping[str, float | None], float, LabeledSnapshot]]
    ) -> TrainingSet:
        """Próbka z wektorów `feature_snapshots` (kolejność `as_of`) z jej opisem."""
        timestamps = [item.snapshot.as_of for _, _, item in rows]
        return TrainingSet(
            features=[dict(vector) for vector, _, _ in rows],
            target=[target for _, target, _ in rows],
            timestamps=timestamps,
            label_ends=(
                label_end_times(timestamps, [item.settled_at for _, _, item in rows])
                if self._settled_purge
                else None
            ),
            dataset=self._describe(
                Counter(item.snapshot.symbol for _, _, item in rows),
                SNAPSHOT_TRAINING_FEATURE_SET_VERSION,
                label_overlap=count_label_overlaps(
                    (item.snapshot.symbol, item.snapshot.as_of, item.settled_at)
                    for _, _, item in rows
                ),
            ),
        )

    def _describe(
        self,
        n_per_symbol: Mapping[str, int],
        feature_set_version: int | None,
        *,
        label_overlap: LabelOverlap | None = None,
    ) -> DatasetDescription:
        """ML-22: opis próbki do proweniencji treningu (karta i sidecar); ML-25:
        z pomiarem nakładania etykiet, gdy próbka zna chwile rozliczeń."""
        return DatasetDescription(
            n_per_symbol=dict(n_per_symbol),
            feature_set_version=feature_set_version,
            git_sha=self._git_sha,
            label_overlap=label_overlap,
        )

    def _snapshot_reason(self, model: str, *, sigma_units: bool | None = None) -> str:
        in_sigma = self._sigma_target if sigma_units is None else sigma_units
        history = " with a sigma history" if in_sigma else ""
        return f"Not enough labeled feature snapshots{history} for {model}"

    def _evaluate_snapshot_rows(
        self,
        rows: Sequence[tuple[Mapping[str, float | None], LabeledSnapshot]],
        series: PriceSeries,
        model: str,
    ) -> TrainingReport:
        in_units = self._in_target_units(rows, series)
        if len(in_units) < MIN_SAMPLES_FOR_TRAINING:
            return _skipped(
                self._snapshot_reason(model),
                len(in_units),
                feature_set_version=SNAPSHOT_TRAINING_FEATURE_SET_VERSION,
            )
        training_set = self._snapshot_training_set(in_units)
        return self._ml_port.train(
            training_set.features,
            training_set.target,
            persist=False,
            timestamps=training_set.timestamps,
            dataset=training_set.dataset,
            **self._target_unit(),
            **self._label_ends(training_set.label_ends),
        )

    def _save_pooled_scorecard(self, result: TrainingReport) -> None:
        self._save_scorecard(POOLED_MODEL_SYMBOL, result)

    def _save_scorecard(self, symbol: str, result: TrainingReport) -> None:
        if self._run_id is None:
            self._db_port.save_model_scorecard(symbol, result)
        else:
            self._db_port.save_model_scorecard(symbol, result, run_id=self._run_id)

    def _train_pooled(self, symbols: Sequence[str]) -> TrainingReport:
        prepared = self.pooled_training_set(symbols)
        if isinstance(prepared, TrainingReport):
            return prepared
        return self._ml_port.train(
            prepared.features,
            prepared.target,
            timestamps=prepared.timestamps,
            dataset=prepared.dataset,
            **self._target_unit(),
            **self._label_ends(prepared.label_ends),
        )

    def pooled_training_set(
        self, symbols: Sequence[str], *, sigma_units: bool | None = None
    ) -> TrainingSet | TrainingReport:
        """ML-22: próbka modelu wspólnego (widok albo, z `snapshot_source`, wektory
        cykli gated) w postaci, w jakiej idzie do `train`, albo raport pominięcia,
        gdy próbka jest poniżej minimum. Sam odczyt — bez treningu i zapisu.
        `sigma_units` None = target jak flaga ML-11; False = surowe zwroty (ML-30)."""
        in_sigma = self._sigma_target if sigma_units is None else sigma_units
        if self._snapshot_source:
            return self._pooled_snapshot_set(symbols, sigma_units=in_sigma)
        rows = [row for symbol in dict.fromkeys(symbols) for row in self._feature_rows(symbol)]

        if len(rows) < MIN_SAMPLES_FOR_TRAINING:
            return self._pooled_skipped("Not enough samples for the pooled model", len(rows))

        complete = complete_model_rows(rows)

        if len(complete) < MIN_SAMPLES_FOR_TRAINING:
            return self._pooled_skipped(
                "Not enough valid samples for the pooled model", len(complete)
            )

        # Porządek globalny (czas, symbol): stabilnie po symbolu, potem stabilnie po czasie.
        ordered, moments = rows_in_label_time_order(sorted(complete, key=_symbol_of))
        if moments is None:
            # Model wspólny bez znaczników czasu nie ma porządku globalnego; widok zawsze
            # ma `timestamp`, a brak kolumny kończył się tym samym wyjątkiem co dotąd.
            raise KeyError(TIMESTAMP_COLUMN)
        overlap = _label_overlap(ordered, moments)
        _log_label_overlap(overlap)
        target = _targets(ordered)
        if in_sigma:
            ordered, moments, target = self._in_sigma_units(ordered, moments, target)
            if len(ordered) < MIN_SAMPLES_FOR_TRAINING:
                return self._pooled_skipped(
                    "Not enough samples with a sigma history for the pooled model", len(ordered)
                )

        return TrainingSet(
            features=_feature_vectors(ordered),
            target=target,
            timestamps=moments,
            label_ends=self._view_label_ends(ordered, moments),
            dataset=self._describe(
                Counter(_symbol_of(row) for row in ordered),
                self._feature_set_version,
                label_overlap=overlap,
            ),
        )

    def _in_sigma_units(
        self, rows: Sequence[Row], moments: Sequence[datetime], target: Sequence[float]
    ) -> tuple[list[Row], list[datetime], list[float]]:
        """ML-11: target = zwrot / σ symbolu wiersza w chwili wiersza (`return_sigma`);
        wiersz bez σ (za krótka historia) wypada razem ze swoim znacznikiem czasu."""
        symbols = [_symbol_of(row) for row in rows]
        series = self._require_training_data().get_price_series(
            list(dict.fromkeys(symbols)), since=min(moments) - SIGMA_LOOKBACK
        )
        kept = [
            (row, moment, value / sigma)
            for row, symbol, moment, value in zip(rows, symbols, moments, target, strict=True)
            if (sigma := return_sigma(series.get(symbol, []), moment)) is not None
        ]
        return (
            [row for row, _, _ in kept],
            [moment for _, moment, _ in kept],
            [value for _, _, value in kept],
        )

    def _view_label_ends(
        self, rows: Sequence[Row], moments: list[datetime] | None
    ) -> list[datetime] | None:
        """ML-26: końce etykiet wierszy widoku z `settled_at` (migracja 050) — tylko
        z flagą; bez kolumny albo znaczników czasu zostaje stały horyzont (None)."""
        if not self._settled_purge or moments is None:
            return None
        settled = settled_moments(rows)
        return None if settled is None else label_end_times(moments, settled)

    @staticmethod
    def _label_ends(label_ends: list[datetime] | None) -> _LabelEndsKwargs:
        """ML-26: `label_ends` w wywołaniu `train` tylko przy purgu po rozliczeniu
        (flaga off = wywołanie jak przed ML-26)."""
        return {} if label_ends is None else {"label_ends": label_ends}

    def _target_unit(self) -> _TargetUnitKwargs:
        """ML-11: jednostka targetu w wywołaniu `train` — tylko przy σ (flaga off =
        wywołanie jak przed ML-11)."""
        return {"target_unit": TargetUnit.SIGMA} if self._sigma_target else {}

    def _require_training_data(self) -> TrainingDataPort:
        if self._training_data is None:
            raise RuntimeError("Training from price snapshots needs a TrainingDataPort (DI)")
        return self._training_data

    def _pooled_snapshot_set(
        self, symbols: Sequence[str], *, sigma_units: bool
    ) -> TrainingSet | TrainingReport:
        """ML-52 (4): model wspólny z cykli gated w `feature_snapshots` (wektor
        z servingu, etykieta z `price_snapshots`) zamiast z widoku SQL. Jedyny zapis
        wag, jak w ścieżce widoku; poniżej minimum — `SKIPPED` z liczbą próbek."""
        _, labeled, series = self._labeled_snapshots(symbols)
        in_units = self._in_target_units(_llm_rows(labeled), series, sigma_units=sigma_units)
        if len(in_units) < MIN_SAMPLES_FOR_TRAINING:
            return _skipped(
                self._snapshot_reason("the pooled model (gated cycles)", sigma_units=sigma_units),
                len(in_units),
                feature_set_version=SNAPSHOT_TRAINING_FEATURE_SET_VERSION,
            )
        return self._snapshot_training_set(in_units)

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
                "Pooled model skipped: feature set version %d leaves %s samples (minimum %d)",
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


def _feature_vectors(rows: Sequence[Row]) -> list[dict[str, float | None]]:
    """ARCH-09: `ML_FEATURE_COLUMNS` wierszy widoku jako słowniki w kolejności
    kontraktu — port nie przyjmuje DataFrame (ramkę składa adapter modelu)."""
    return [{name: float(row[name]) for name in ML_FEATURE_COLUMNS} for row in rows]


def _targets(rows: Sequence[Row]) -> list[float]:
    """`target_return` wierszy widoku. Wartości przychodzą GOTOWE z widoku
    (NaN-guarded) — stary recompute (current-prev)/prev dublował logikę widoku
    i przy prev=0 dawał +inf, który trafiał do XGBoost."""
    return [float(row[TARGET_COLUMN]) for row in rows]


def _symbol_of(row: Row) -> str:
    return str(row["symbol"])


def _label_overlap(rows: Sequence[Row], moments: Sequence[datetime]) -> LabelOverlap | None:
    """ML-25: nakładanie etykiet z `settled_at` (widok po migracji 050).

    None = widok bez kolumny (przed migracją) — odczyt `*` jest tolerancyjny."""
    settled = settled_moments(rows)
    if settled is None:
        return None
    return count_label_overlaps(
        zip((_symbol_of(row) for row in rows), moments, settled, strict=True)
    )


def _log_label_overlap(overlap: LabelOverlap | None) -> None:
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
