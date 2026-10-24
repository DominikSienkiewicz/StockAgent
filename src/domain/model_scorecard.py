# src/domain/model_scorecard.py
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType

# Karta kondycji modelu — czysta domena eksponująca dowód rzetelności retreningu.
# Repo shipuje model tylko, jeśli bije naiwny baseline persystencji („zwrot=0"),
# ale ta bramka jest niewidzialna dla użytkownika. Ten moduł liczy procentową
# poprawę względem baseline'u, ocenia świeżość modelu i agreguje wiersze
# NAJNOWSZEGO przebiegu treningu w uczciwy obraz: werdykt modelu wspólnego „ALL"
# („zaakceptowano N, odrzucono M, pominięto K") i osobno ewaluacje per symbol.
# Bez I/O, bez importów z application.

# Symbol wiersza karty modelu wspólnego (ML-13) — jedynego, który zapisuje wagi.
# Werdykt główny karty bierze się z tego wiersza (decyzja 3 z 2026-10-18).
POOLED_MODEL_SYMBOL = "ALL"

# ML-52: karty modeli ewaluowanych na wektorach `feature_snapshots` (bez zapisu
# wag): cechy darmowe na wszystkich cyklach i wektor modelu tylko na cyklach gated.
# Nie są ewaluacjami per symbol ani werdyktem modelu wspólnego.
FREE_SNAPSHOT_MODEL_SYMBOL = "ALL_FREE"
LLM_SNAPSHOT_MODEL_SYMBOL = "ALL_LLM"
SNAPSHOT_MODEL_SYMBOLS = frozenset({FREE_SNAPSHOT_MODEL_SYMBOL, LLM_SNAPSHOT_MODEL_SYMBOL})

# Model starszy niż tyle dni traktujemy jako nieświeży (STALE). Próg zamrożony
# jako stała modułu — pojedyncze źródło prawdy dla reguły świeżości.
MODEL_STALE_AFTER_DAYS = 21

# ML-15: okno odczytu karty. Musi sięgać dalej niż próg STALE — przy oknie 30 dni
# (`track_record_days`) karta znikała po 30 dniach i odznaka STALE była widoczna
# tylko w dniach 22–30. Dwa progi świeżości dają trzy tygodnie odznaki.
MODEL_SCORECARD_WINDOW_DAYS = 2 * MODEL_STALE_AFTER_DAYS

# Wiersze jednego przebiegu Slow Loopa dzielą minuty; kolejne przebiegi — tydzień.
# Przerwa dłuższa niż ta granica zaczyna poprzedni przebieg (ML-03). Ręczny re-run
# tego samego dnia skleja się z cronem — rozdziela je `run_id` (ML-24), a ta
# heurystyka zostaje dla historii bez identyfikatora przebiegu.
RUN_MAX_GAP = timedelta(hours=2)

# Zastrzeżenie do copy sekcji „trust". Spec mówi wprost: sekcja bez tej noty
# sama byłaby nieuczciwa. Metryki dotyczą KANDYDATA na foldach walk-forward,
# a finalny (shipowany) model jest refitowany na CAŁOŚCI danych i już nie jest
# ponownie walidowany — więc nie są to metryki jakości produkcyjnej.
METRIC_NOTE = (
    "Metryki dotyczą KANDYDATA na foldach walk-forward; finalny (shipowany) "
    "model jest refitowany na całości danych i nie jest ponownie walidowany."
)


def improvement_pct(candidate_rmse: float, baseline_rmse: float) -> float:
    """Procentowa redukcja RMSE kandydata względem baseline'u persystencji.

    Wzór reużyty z `src/domain/backtest.py`:
    `(baseline - candidate) / baseline * 100` — dodatnia gdy kandydat lepszy,
    ujemna gdy gorszy (brutalna prawda, nie udajemy sukcesu). Zwraca `0.0`, gdy
    `baseline_rmse == 0`, by uniknąć dzielenia przez zero (jak w backtest.py).
    """
    if baseline_rmse == 0:
        return 0.0
    return (baseline_rmse - candidate_rmse) / baseline_rmse * 100


def is_stale(trained_at: datetime, now: datetime) -> bool:
    """Czy model jest nieświeży (STALE) — starszy NIŻ `MODEL_STALE_AFTER_DAYS`.

    Granica ostra: dokładnie 21 dni to jeszcze świeży model, dopiero powyżej
    progu staje się STALE. `now` jest wstrzykiwane (domena nie woła zegara).
    """
    return now - trained_at > timedelta(days=MODEL_STALE_AFTER_DAYS)


class TrainingOutcome(Enum):
    """Domenowy werdykt pojedynczego przebiegu treningu albo ewaluacji.

    - `ACCEPTED`  — kandydat pobił baseline, model zaszipowany.
    - `REJECTED`  — kandydat gorszy od baseline'u, retrain odrzucony.
    - `SKIPPED`   — trening nie mógł się odbyć (np. za mało próbek).
    - `EVALUATION_PASSED` / `EVALUATION_FAILED` — sama ewaluacja walk-forward bez
      zapisu wag (ML-13, per symbol): kandydat przeszedł albo nie przeszedł bramki.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    EVALUATION_PASSED = "evaluation_passed"
    EVALUATION_FAILED = "evaluation_failed"


class TargetUnit(Enum):
    """ML-11: jednostka targetu modelu — surowy zwrot albo zwrot / σ symbolu
    (`return_sigma`). Model w jednostkach σ serwuje prognozę × σ z chwili cyklu."""

    RETURN = "return"
    SIGMA = "sigma"


# ARCH-09: status zapisywany w `model_scorecards.status` — kontrakt historii tabeli.
# Werdykt przekracza port jako `TrainingOutcome`, a string powstaje wyłącznie tutaj,
# więc nowy wiersz karty ma zawsze jeden z tych pięciu statusów.
LEGACY_STATUS: Mapping[TrainingOutcome, str] = MappingProxyType(
    {
        TrainingOutcome.ACCEPTED: "trained_successfully",
        TrainingOutcome.REJECTED: "skipped_validation_failed",
        TrainingOutcome.SKIPPED: "skipped",
        TrainingOutcome.EVALUATION_PASSED: "evaluated_passed_gate",
        TrainingOutcome.EVALUATION_FAILED: "evaluated_failed_gate",
    }
)

# Mapa wstecz dla odczytu historii `model_scorecards` (ARCH-09).
_STATUS_TO_OUTCOME = {status: outcome for outcome, status in LEGACY_STATUS.items()}


def classify_status(status: str) -> TrainingOutcome:
    """Mapuje zapisany `status` (`LEGACY_STATUS`) na `TrainingOutcome`.

    Odczyt historii jest tolerancyjny: status spoza mapy (pusty string albo
    dawny „za mało próbek” sprzed ARCH-09) → SKIPPED.
    """
    return _STATUS_TO_OUTCOME.get(status, TrainingOutcome.SKIPPED)


@dataclass(frozen=True)
class TrainingReport(Mapping[str, object]):
    """ARCH-09: wynik `MLPredictionPort.train` — werdykt i metryki walk-forward.

    `outcome` jest typem domenowym; `rmse`, `baseline_rmse` i pozostałe metryki
    to średnie po foldach (None, gdy trening się nie odbył). `extras` niesie pola
    poboczne (np. `feature_set_version` przy pominięciu, DAT-09).

    Widok `Mapping` to kontrakt słownika sprzed ARCH-09 (te same klucze i wartości,
    status z `LEGACY_STATUS`): czytają go zapis karty (`save_model_scorecard`),
    summary Slow Loopa i logi. Metryki są w widoku tylko, gdy policzono foldy.
    """

    outcome: TrainingOutcome
    n_samples: int
    reason: str | None = None
    n_folds: int | None = None
    n_oos_points: int | None = None
    rmse: float | None = None
    baseline_rmse: float | None = None
    drift_baseline_rmse: float | None = None
    llm_baseline_rmse: float | None = None
    hit_rate: float | None = None
    feature_stats: Mapping[str, object] | None = None
    metric_note: str | None = None
    model_path: str | None = None
    extras: Mapping[str, object] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Status do `model_scorecards.status`."""
        return LEGACY_STATUS[self.outcome]

    def as_legacy_dict(self) -> dict[str, object]:
        """Słownik w kontrakcie sprzed ARCH-09, w tej samej kolejności kluczy."""
        legacy: dict[str, object] = {"status": self.status}
        if self.reason is not None:
            legacy["reason"] = self.reason
        legacy["n_samples"] = self.n_samples
        if self.n_folds is not None:
            legacy.update(
                {
                    "n_folds": self.n_folds,
                    "n_oos_points": self.n_oos_points,
                    "candidate_holdout_rmse": self.rmse,
                    "baseline_rmse": self.baseline_rmse,
                    "drift_baseline_rmse": self.drift_baseline_rmse,
                    "llm_baseline_rmse": self.llm_baseline_rmse,
                    "candidate_holdout_directional_hit_rate": self.hit_rate,
                    "feature_stats": self.feature_stats,
                    "metric_note": self.metric_note,
                }
            )
        if self.model_path is not None:
            legacy["model_path"] = self.model_path
        legacy.update(self.extras)
        return legacy

    def __getitem__(self, key: str) -> object:
        return self.as_legacy_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_legacy_dict())

    def __len__(self) -> int:
        return len(self.as_legacy_dict())


@dataclass(frozen=True)
class RunSummary:
    """Zagregowany obraz jednego przebiegu treningu nad wieloma symbolami.

    `total` to liczba symboli, a `accepted`/`rejected`/`skipped` — rozkład ich
    statusów (`accepted + rejected + skipped == total`). `summary_line` to
    gotowa polska etykieta do raportu („zaakceptowano N, odrzucono M,
    pominięto K"). Gdy wszystkie modele odrzucone, `accepted == 0` — karta
    pokazuje brutalną prawdę, nie udaje sukcesu.
    """

    total: int
    accepted: int
    rejected: int
    skipped: int
    summary_line: str


def latest_run_indices(
    stamps: Sequence[datetime | None], max_gap: timedelta = RUN_MAX_GAP
) -> list[int]:
    """Indeksy (rosnąco) wierszy należących do najnowszego przebiegu treningu.

    Idąc od najnowszego znacznika wstecz, przebieg kończy się na pierwszej
    przerwie między sąsiednimi wierszami dłuższej niż `max_gap`. Kolejność
    wejścia nie ma znaczenia; wiersze bez znacznika (`None`) są pomijane.
    """
    dated = sorted(
        ((stamp, idx) for idx, stamp in enumerate(stamps) if stamp is not None),
        reverse=True,
    )
    run: list[int] = []
    previous: datetime | None = None
    for stamp, idx in dated:
        if previous is not None and previous - stamp > max_gap:
            break
        run.append(idx)
        previous = stamp
    return sorted(run)


def latest_run_indices_by_id(
    stamps: Sequence[datetime | None],
    run_ids: Sequence[str | None],
    max_gap: timedelta = RUN_MAX_GAP,
) -> list[int]:
    """Indeksy (rosnąco) wierszy najnowszego przebiegu, grupowane po `run_id` (ML-24).

    Najnowszy wiersz ze znacznikiem wyznacza przebieg: ma `run_id` → przebieg to
    wszystkie wiersze z tym identyfikatorem i znacznikiem; nie ma (historia sprzed
    migracji 049 albo flaga off) → heurystyka przerwy `latest_run_indices`.
    `stamps` i `run_ids` są równoległe (ten sam indeks = ten sam wiersz).
    """
    dated = [(stamp, idx) for idx, stamp in enumerate(stamps) if stamp is not None]
    if not dated:
        return []
    _, newest = max(dated)
    run_id = run_ids[newest]
    if run_id is None:
        return latest_run_indices(stamps, max_gap)
    return [idx for _, idx in dated if run_ids[idx] == run_id]


def summarize_run(
    outcomes: Iterable[TrainingOutcome | str],
) -> RunSummary:
    """Agreguje per-symbol wyniki treningu w jeden `RunSummary`.

    Przyjmuje zarówno `TrainingOutcome`, jak i surowe stringi `status` z
    `train()` (są klasyfikowane przez `classify_status`). Ewaluacje bez zapisu wag
    (`EVALUATION_*`) liczą się jako pominięte, jak przed ARCH-09. Pusty przebieg
    daje zerowy `RunSummary` — nie ma „pustego sukcesu".
    """
    accepted = rejected = skipped = 0
    for item in outcomes:
        outcome = item if isinstance(item, TrainingOutcome) else classify_status(item)
        if outcome is TrainingOutcome.ACCEPTED:
            accepted += 1
        elif outcome is TrainingOutcome.REJECTED:
            rejected += 1
        else:
            skipped += 1

    total = accepted + rejected + skipped
    summary_line = (
        f"zaakceptowano {accepted}, odrzucono {rejected}, pominięto {skipped}"
    )
    return RunSummary(
        total=total,
        accepted=accepted,
        rejected=rejected,
        skipped=skipped,
        summary_line=summary_line,
    )


# Statusy ewaluacji per symbol z `XGBoostAdapter.train(persist=False)` (ML-13).
_EVALUATION_PASSED = LEGACY_STATUS[TrainingOutcome.EVALUATION_PASSED]
_EVALUATION_FAILED = LEGACY_STATUS[TrainingOutcome.EVALUATION_FAILED]


@dataclass(frozen=True)
class EvaluationSummary:
    """Ewaluacje per symbol z jednego przebiegu — osobno od werdyktu modelu wspólnego.

    Ewaluacja nigdy nie zapisuje wag (ML-13), więc nie jest ani „zaakceptowana”,
    ani „odrzucona”: `passed`/`failed` mówią, czy kandydat na danych symbolu
    przeszedł bramkę walk-forward, a `skipped` liczy resztę (np. za mało próbek).
    """

    total: int
    passed: int
    failed: int
    skipped: int
    summary_line: str


def summarize_evaluations(statuses: Iterable[str]) -> EvaluationSummary:
    """Agreguje statusy ewaluacji per symbol w `EvaluationSummary`."""
    passed = failed = skipped = 0
    for status in statuses:
        if status == _EVALUATION_PASSED:
            passed += 1
        elif status == _EVALUATION_FAILED:
            failed += 1
        else:
            skipped += 1
    return EvaluationSummary(
        total=passed + failed + skipped,
        passed=passed,
        failed=failed,
        skipped=skipped,
        summary_line=(
            f"bramkę przeszło {passed}, nie przeszło {failed}, pominięto {skipped}"
        ),
    )
