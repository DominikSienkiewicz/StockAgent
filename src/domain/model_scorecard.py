# src/domain/model_scorecard.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Karta kondycji modelu — czysta domena eksponująca dowód rzetelności retreningu.
# Repo shipuje model tylko, jeśli bije naiwny baseline persystencji („zwrot=0"),
# ale ta bramka jest niewidzialna dla użytkownika. Ten moduł liczy procentową
# poprawę względem baseline'u, ocenia świeżość modelu i agreguje PER-SYMBOL
# przebieg treningu (~43 przebiegi nadpisują jeden `.ubj`) w uczciwy obraz:
# „zaakceptowano N, odrzucono M, pominięto K". Bez I/O, bez importów z application.

# Model starszy niż tyle dni traktujemy jako nieświeży (STALE). Próg zamrożony
# jako stała modułu — pojedyncze źródło prawdy dla reguły świeżości.
MODEL_STALE_AFTER_DAYS = 21

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
