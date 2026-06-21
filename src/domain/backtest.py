# src/domain/backtest.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Werdykt walk-forward backtestu. Czysta domena — agreguje per-fold metryki
# (RMSE kandydata vs baseline persystencji „zwrot=0", trafność kierunku) w jeden
# obraz: czy lokalny model bije naiwny baseline i o ile. Bez I/O, bez pandas.


@dataclass(frozen=True)
class BacktestSummary:
    """Zagregowany wynik walk-forward backtestu nad N foldami.

    `mean_*` to średnie arytmetyczne metryk per-fold. `improvement_pct` to
    procentowa redukcja RMSE względem baseline'u (dodatnia = model lepszy).
    `beats_baseline` jest True gdy średnie RMSE kandydata < średnie RMSE
    baseline'u. `verdict` to krótka polska etykieta do raportu.
    """

    n_folds: int
    mean_candidate_rmse: float
    mean_baseline_rmse: float
    mean_hit_rate: float
    improvement_pct: float
    beats_baseline: bool
    verdict: str


def summarize_folds(folds: Sequence[Mapping[str, float]]) -> BacktestSummary:
    """Agreguje per-fold metryki walk-forward w `BacktestSummary`.

    Każdy fold to mapa z kluczami `candidate_rmse`, `baseline_rmse`, `hit_rate`
    (kształt z `MLPredictionPort.walk_forward_report`). Średnie są arytmetyczne;
    `improvement_pct = (baseline - candidate) / baseline * 100` (0.0 gdy
    baseline == 0, by uniknąć dzielenia przez zero). Pusta sekwencja zwraca
    sentinel „brak danych".
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

    beats_baseline = mean_candidate < mean_baseline

    if beats_baseline:
        verdict = f"model bije baseline o {improvement_pct:.1f}%"
    else:
        verdict = "model nie bije baseline"

    return BacktestSummary(
        n_folds=n,
        mean_candidate_rmse=mean_candidate,
        mean_baseline_rmse=mean_baseline,
        mean_hit_rate=mean_hit_rate,
        improvement_pct=improvement_pct,
        beats_baseline=beats_baseline,
        verdict=verdict,
    )
