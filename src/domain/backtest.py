# src/domain/backtest.py
from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

# Werdykt walk-forward backtestu. Czysta domena — agreguje per-fold metryki
# (RMSE kandydata vs baseline persystencji „zwrot=0", trafność kierunku) w jeden
# obraz: czy lokalny model bije naiwny baseline i o ile. Bez I/O, bez pandas.
# To JEDYNE miejsce reguły „bije baseline" — adapter XGBoost stosuje ten sam
# werdykt przed zapisem wag (ML-05), zamiast trzymać własny próg.

# ML-05: kandydat musi obniżyć RMSE o co najmniej 2% WZGLĘDNIE. Próg ustalony
# przed spojrzeniem na wyniki — obniżanie go pod wynik to p-hacking.
MIN_REL_IMPROVEMENT = 0.02

# ML-06: werdykt wymaga co najmniej tylu punktów out-of-sample łącznie ze
# wszystkich foldów. Przy 4 foldach expanding-window oznacza to n ≥ 75 próbek.
MIN_OOS_POINTS = 60

# Horyzont etykiety: target to zwrot 12h, więc wiersz z chwili t „zna" rynek do
# t + 12h. Wiersz treningowy, którego etykieta dojrzewa po starcie walidacji,
# przecieka do foldu (ML-13).
LABEL_HORIZON = timedelta(hours=12)


@dataclass(frozen=True)
class BacktestSummary:
    """Zagregowany wynik walk-forward backtestu nad N foldami.

    `mean_*` to średnie arytmetyczne metryk per-fold. `improvement_pct` to
    procentowa redukcja RMSE względem baseline'u (dodatnia = model lepszy).
    `beats_baseline` jest True gdy średnie RMSE kandydata jest niższe od
    baseline'u o co najmniej `MIN_REL_IMPROVEMENT` i foldy dały co najmniej
    `MIN_OOS_POINTS` punktów OOS. `n_oos_points` to suma `n_valid` z foldów
    (None, gdy foldy nie niosą tego klucza). `verdict` to polska etykieta do raportu.

    ML-20: `mean_ic` to średni IC (korelacja rang prognozy z realizacją) po
    foldach, `ic_std` jego odchylenie (próbkowe), `ic_t_stat = mean_ic / ic_std ·
    √n_folds`. None, gdy foldy nie niosą `ic`, a sd/t-stat także przy jednym
    foldzie albo zerowym rozrzucie. Przy 4 foldach t-stat jest mało informatywny —
    czytaj go razem z `n_folds`.
    """

    n_folds: int
    mean_candidate_rmse: float
    mean_baseline_rmse: float
    mean_hit_rate: float
    improvement_pct: float
    beats_baseline: bool
    verdict: str
    n_oos_points: int | None = None
    mean_ic: float | None = None
    ic_std: float | None = None
    ic_t_stat: float | None = None


def beats_baseline(candidate_rmse: float, baseline_rmse: float) -> bool:
    """Czy kandydat bije baseline persystencji o co najmniej `MIN_REL_IMPROVEMENT`.

    Bezbłędny baseline (`baseline_rmse == 0`, zwrot zawsze 0) jest nie do pobicia —
    także przez kandydata z RMSE 0, który niczego się nie nauczył.
    """
    if baseline_rmse <= 0:
        return False
    return (baseline_rmse - candidate_rmse) / baseline_rmse >= MIN_REL_IMPROVEMENT


def purged_train_end(
    timestamps: Sequence[datetime], valid_start: int, horizon: timedelta = LABEL_HORIZON
) -> int:
    """Koniec (wyłącznie) treningu foldu walidowanego od wiersza `valid_start`.

    `timestamps` są posortowane rosnąco po czasie GLOBALNYM (wszystkie symbole
    razem). Z treningu wypadają wiersze, których etykieta dojrzewa po starcie
    walidacji: w modelu wspólnym AAPL z 10:00 w treningu i MSFT z 10:01 w walidacji
    dzielą ten sam ruch rynku — to przeciek przekrojowy (purge). Etykieta dojrzała
    dokładnie w chwili startu walidacji jest dozwolona.
    """
    cutoff = timestamps[valid_start] - horizon
    return bisect_right(timestamps, cutoff, 0, valid_start)


def _oos_points(folds: Sequence[Mapping[str, float]]) -> int | None:
    if not all("n_valid" in fold for fold in folds):
        return None
    return int(sum(fold["n_valid"] for fold in folds))


def _ic_stats(
    folds: Sequence[Mapping[str, float]],
) -> tuple[float | None, float | None, float | None]:
    """(średni IC, sd, t-stat) po foldach; None tam, gdzie miara niezdefiniowana."""
    if not all("ic" in fold for fold in folds):
        return None, None, None
    ics = [fold["ic"] for fold in folds]
    mean_ic = statistics.fmean(ics)
    if len(ics) < 2:
        return mean_ic, None, None
    ic_std = statistics.stdev(ics)
    if ic_std == 0:
        return mean_ic, ic_std, None
    return mean_ic, ic_std, mean_ic / ic_std * math.sqrt(len(ics))


def summarize_folds(folds: Sequence[Mapping[str, float]]) -> BacktestSummary:
    """Agreguje per-fold metryki walk-forward w `BacktestSummary`.

    Każdy fold to mapa z kluczami `candidate_rmse`, `baseline_rmse`, `hit_rate`
    i opcjonalnie `n_valid` (kształt z `MLPredictionPort.walk_forward_report`).
    Średnie są arytmetyczne; `improvement_pct = (baseline - candidate) / baseline
    * 100` (0.0 gdy baseline == 0, by uniknąć dzielenia przez zero). Werdykt
    stosuje `beats_baseline` i — gdy foldy niosą `n_valid` — `MIN_OOS_POINTS`.
    Pusta sekwencja zwraca sentinel „brak danych".
    """
    if not folds:
        return BacktestSummary(0, 0, 0, 0, 0, False, "brak danych")

    n = len(folds)
    mean_candidate = sum(f["candidate_rmse"] for f in folds) / n
    mean_baseline = sum(f["baseline_rmse"] for f in folds) / n
    mean_hit_rate = sum(f["hit_rate"] for f in folds) / n

    if mean_baseline == 0:
        improvement_pct = 0.0
    else:
        improvement_pct = (mean_baseline - mean_candidate) / mean_baseline * 100

    n_oos_points = _oos_points(folds)
    mean_ic, ic_std, ic_t_stat = _ic_stats(folds)
    passed = beats_baseline(mean_candidate, mean_baseline)

    if n_oos_points is not None and n_oos_points < MIN_OOS_POINTS:
        passed = False
        verdict = f"za mało punktów OOS ({n_oos_points} < {MIN_OOS_POINTS})"
    elif passed:
        verdict = f"model bije baseline o {improvement_pct:.1f}%"
    elif mean_candidate < mean_baseline:
        verdict = (
            f"model nie bije baseline (poprawa {improvement_pct:.1f}% "
            f"< marginesu {MIN_REL_IMPROVEMENT:.0%})"
        )
    else:
        verdict = "model nie bije baseline"

    return BacktestSummary(
        n_folds=n,
        mean_candidate_rmse=mean_candidate,
        mean_baseline_rmse=mean_baseline,
        mean_hit_rate=mean_hit_rate,
        improvement_pct=improvement_pct,
        beats_baseline=passed,
        verdict=verdict,
        n_oos_points=n_oos_points,
        mean_ic=mean_ic,
        ic_std=ic_std,
        ic_t_stat=ic_t_stat,
    )
