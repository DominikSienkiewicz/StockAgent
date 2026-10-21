# src/domain/model_scorecard.py
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

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

# Model starszy niż tyle dni traktujemy jako nieświeży (STALE). Próg zamrożony
# jako stała modułu — pojedyncze źródło prawdy dla reguły świeżości.
MODEL_STALE_AFTER_DAYS = 21

# ML-15: okno odczytu karty. Musi sięgać dalej niż próg STALE — przy oknie 30 dni
# (`track_record_days`) karta znikała po 30 dniach i odznaka STALE była widoczna
# tylko w dniach 22–30. Dwa progi świeżości dają trzy tygodnie odznaki.
MODEL_SCORECARD_WINDOW_DAYS = 2 * MODEL_STALE_AFTER_DAYS

# Wiersze jednego przebiegu Slow Loopa dzielą minuty; kolejne przebiegi — tydzień.
# Przerwa dłuższa niż ta granica zaczyna poprzedni przebieg (ML-03). Ręczny re-run
# tego samego dnia skleja się z cronem — rozdzieli je dopiero `run_id` (ML-24).
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
    """Domenowy status pojedynczego przebiegu treningu per-symbol.

    - `ACCEPTED`  — kandydat pobił baseline, model zaszipowany.
    - `REJECTED`  — kandydat gorszy od baseline'u, retrain odrzucony.
    - `SKIPPED`   — trening nie mógł się odbyć (np. za mało próbek).
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SKIPPED = "skipped"


# Mapowanie stringów `status` z `train()` (infrastruktura) na domenowe outcome.
# Trzymane jako literały, bo domena nie importuje infrastruktury; komentarz
# wiąże je z `XGBoostAdapter.train()`.
_STATUS_TO_OUTCOME = {
    "trained_successfully": TrainingOutcome.ACCEPTED,
    "skipped_validation_failed": TrainingOutcome.REJECTED,
}


def classify_status(status: str) -> TrainingOutcome:
    """Mapuje string `status` z `train()` na `TrainingOutcome`.

    Znane statusy: `trained_successfully` → ACCEPTED, `skipped_validation_failed`
    → REJECTED. Wszystko inne (w tym pusty string albo status „za mało próbek"
    zgłoszony przez orkiestratora po `ValueError`) → SKIPPED.
    """
    return _STATUS_TO_OUTCOME.get(status, TrainingOutcome.SKIPPED)


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


def summarize_run(
    outcomes: Iterable[TrainingOutcome | str],
) -> RunSummary:
    """Agreguje per-symbol wyniki treningu w jeden `RunSummary`.

    Przyjmuje zarówno `TrainingOutcome`, jak i surowe stringi `status` z
    `train()` (są klasyfikowane przez `classify_status`). Pusty przebieg daje
    zerowy `RunSummary` — nie ma „pustego sukcesu".
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
_EVALUATION_PASSED = "evaluated_passed_gate"
_EVALUATION_FAILED = "evaluated_failed_gate"


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
