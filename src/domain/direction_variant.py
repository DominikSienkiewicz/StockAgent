"""Wariant binarny modelu ML (ML-30): P(zwrot 12 h > +martwa strefa).

Regresja zaszumionego zwrotu z bramką RMSE prawie nigdy nie bije zera (hipoteza
G03), a produkt komunikuje kierunek i pewność. Wariant uczy klasyfikator na
etykiecie 1[zwrot > +pasmo] / 0[zwrot < −pasmo]; wiersze w martwej strefie
wypadają (szum, nie kierunek). Szerokość pasma pochodzi z protokołu ewaluacji
(`directional_dead_zone`, EXP-06) — nie jest strojona pod wynik.

Bramka wariantu to Brier poza próbą względem stopy bazowej części treningowej
foldu (prognoza „zawsze odsetek wzrostów”) z tym samym marginesem i minimum
punktów OOS co bramka regresji. Wariant nie zmienia championa: serwowany model
regresji zostaje bez zmian. Czysta domena, stdlib.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.domain.backtest import MIN_OOS_POINTS, MIN_REL_IMPROVEMENT
from src.domain.model_scorecard import TrainingOutcome

# Klucze foldu (sumy po punktach walidacji) liczone przez adapter klasyfikatora.
_SUM_KEYS = (
    "n_valid",
    "sum_sq_error",
    "base_sum_sq_error",
    "sum_log_loss",
    "base_sum_log_loss",
)


def direction_label(target_return: float, *, dead_band: float) -> int | None:
    """1 = wzrost ponad pasmo, 0 = spadek poniżej −pasma, None = martwa strefa
    (granica włącznie) albo brak zwrotu (NaN). Ujemne pasmo → `ValueError`."""
    if dead_band < 0:
        raise ValueError(f"dead_band must be >= 0, got {dead_band}")
    if math.isnan(target_return):
        return None
    if target_return > dead_band:
        return 1
    if target_return < -dead_band:
        return 0
    return None


@dataclass(frozen=True)
class DirectionSummary:
    """Werdykt walk-forward klasyfikatora kierunku.

    Miary to średnie po WSZYSTKICH punktach OOS foldów (nie średnie foldów):
    Brier i log loss kandydata obok prognozy stopy bazowej części treningowej
    foldu. `skill` = 1 − Brier / Brier stopy bazowej (None, gdy stopa bazowa jest
    bezbłędna). None przy braku foldów."""

    n_folds: int
    n_oos_points: int
    brier: float | None
    base_rate_brier: float | None
    log_loss: float | None
    base_rate_log_loss: float | None
    beats_base_rate: bool
    verdict: str

    @property
    def skill(self) -> float | None:
        if self.brier is None or not self.base_rate_brier:
            return None
        return 1.0 - self.brier / self.base_rate_brier


def summarize_direction_folds(folds: Sequence[Mapping[str, float]]) -> DirectionSummary:
    """Werdykt z sum foldów (`n_valid`, `sum_sq_error`, `base_sum_sq_error`,
    `sum_log_loss`, `base_sum_log_loss`). Kandydat przechodzi, gdy obniża Brier
    stopy bazowej o co najmniej `MIN_REL_IMPROVEMENT` na co najmniej
    `MIN_OOS_POINTS` punktach — ta sama reguła co bramka regresji."""
    n = int(sum(fold["n_valid"] for fold in folds))
    if not folds or n == 0:
        return DirectionSummary(0, 0, None, None, None, None, False, "brak danych")
    sums = {key: sum(fold[key] for fold in folds) for key in _SUM_KEYS}
    brier = sums["sum_sq_error"] / n
    base_brier = sums["base_sum_sq_error"] / n
    passed = base_brier > 0 and (base_brier - brier) / base_brier >= MIN_REL_IMPROVEMENT
    if n < MIN_OOS_POINTS:
        passed = False
        verdict = f"za mało punktów OOS ({n} < {MIN_OOS_POINTS})"
    elif passed:
        verdict = f"klasyfikator bije stopę bazową (Brier {brier:.4f} < {base_brier:.4f})"
    else:
        verdict = (
            f"klasyfikator nie bije stopy bazowej o {MIN_REL_IMPROVEMENT:.0%} "
            f"(Brier {brier:.4f} vs {base_brier:.4f})"
        )
    return DirectionSummary(
        n_folds=len(folds),
        n_oos_points=n,
        brier=brier,
        base_rate_brier=base_brier,
        log_loss=sums["sum_log_loss"] / n,
        base_rate_log_loss=sums["base_sum_log_loss"] / n,
        beats_base_rate=passed,
        verdict=verdict,
    )


@dataclass(frozen=True)
class DirectionReport:
    """Wynik treningu wariantu binarnego (`MLPredictionPort.train_direction_classifier`).

    `n_samples` — wiersze z etykietą (poza martwą strefą), `n_in_dead_band` —
    wiersze odrzucone jako szum, `summary` — werdykt walk-forward (None przy
    pominięciu), `model_path` — plik wag wariantu, gdy go zapisano (nigdy plik
    championa). Wyniki: `ACCEPTED` / `REJECTED` przy zapisie, `EVALUATION_*`
    bez zapisu, `SKIPPED` poniżej minimum próbki."""

    outcome: TrainingOutcome
    n_samples: int
    n_in_dead_band: int
    dead_band: float
    summary: DirectionSummary | None = None
    reason: str | None = None
    model_path: str | None = None
