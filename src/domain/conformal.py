"""Przedziały ruchu z conformal prediction (ML-34).

Produkt podawał gołą cenę docelową (przy cold-starcie równą bieżącej) bez
niepewności. Ten moduł liczy przedział zwrotu do rozliczenia predykcji z okna
kalibracyjnego rozliczonych predykcji tej samej klasy aktywa:

- **model aktywny** — split-conformal: ŷ ± kwantyl skończonej próby
  `ceil((n+1)(1−α))/n` z |reszt| `|y − ŷ|` okna;
- **cold-start** — predyktor 0, przedział z empirycznych kwantyli zwrotów
  (po α/2 na ogon, ta sama poprawka skończonej próby), więc może być skośny;
- **ACI** (adaptive conformal inference) — `α_{t+1} = α_t + γ(α − err_t)`,
  odtwarzane po kolei na oknie: seria pudeł obniża α_t i poszerza przedział.

UCZCIWOŚĆ: gwarancja pokrycia ≥ 1−α zakłada WYMIENNOŚĆ obserwacji okna
i bieżącej predykcji. Rynek jej nie spełnia (zmienność się grupuje, reżimy się
zmieniają), więc nominalne 80% to cel, nie obietnica. ACI łagodzi skutki
zmiany rozkładu, ale też niczego nie gwarantuje — dlatego track record
porównuje pokrycie zrealizowane z nominalnym.

Czysty stdlib.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Decyzja właściciela 2026-10-19: przedział 80%.
NOMINAL_ALPHA = 0.2
# Krok ACI (Gibbs & Candès 2021). Mały krok = α_t reaguje na trwałą serię pudeł,
# a nie na pojedynczy zły dzień.
ACI_GAMMA = 0.005
# Poniżej 20 punktów mail mówi „za mało historii (n/20)" zamiast liczby —
# ten sam próg co dla korelacji w `risk_sample`.
MIN_CALIBRATION_SIZE = 20
# Okno kroczące: najnowsze punkty. Stare reżimy nie poszerzają przedziału bez
# końca, a replay ACI (kwadratowy w n) zostaje tani.
MAX_CALIBRATION_SIZE = 250


@dataclass(frozen=True)
class CalibrationPoint:
    """Rozliczona predykcja: zwrot przewidziany (0 przy cold-starcie) i zrealizowany."""

    predicted_return: float
    realized_return: float


@dataclass(frozen=True)
class MoveInterval:
    """Przedział zwrotu do rozliczenia, jako ułamki (−0.021 = −2.1%).

    `low`/`high` = None → za mało historii (`calibration_size` < `required_size`).
    `alpha` to α_t po ACI, którym policzono przedział (audyt)."""

    low: float | None
    high: float | None
    calibration_size: int
    required_size: int = MIN_CALIBRATION_SIZE
    nominal_coverage: float = 1 - NOMINAL_ALPHA
    alpha: float = NOMINAL_ALPHA

    @property
    def available(self) -> bool:
        return self.low is not None and self.high is not None


@dataclass(frozen=True)
class IntervalCoverage:
    """Pokrycie zrealizowane: ile rozliczonych zwrotów wpadło w swój przedział."""

    hits: int
    n: int
    nominal: float = 1 - NOMINAL_ALPHA

    @property
    def rate(self) -> float | None:
        return self.hits / self.n if self.n else None


def _rank(position: float) -> float:
    # Iloczyny typu 20 · 0.9 wychodzą w floatach jako 18.000000000000004, a ceil
    # dałby wtedy 19 — zaokrąglamy szum arytmetyczny przed zaokrągleniem rangi.
    return round(position, 9)


def _upper_rank(n: int, tail: float) -> int:
    """Ranga (1-indeksowana) kwantyla górnego: ceil((n+1)(1−tail))."""
    return math.ceil(_rank((n + 1) * (1 - tail)))


def _lower_rank(n: int, tail: float) -> int:
    """Ranga (1-indeksowana) kwantyla dolnego: floor((n+1)·tail)."""
    return math.floor(_rank((n + 1) * tail))


def conformal_quantile(scores: Sequence[float], alpha: float) -> float | None:
    """Kwantyl skończonej próby: `ceil((n+1)(1−α))`-ta najmniejsza wartość.

    None, gdy ranga przekracza n — żadna skończona wartość okna nie daje wtedy
    pokrycia 1−α (okno za małe dla tego α)."""
    n = len(scores)
    rank = _upper_rank(n, alpha)
    if n == 0 or rank > n:
        return None
    return sorted(scores)[max(rank, 1) - 1]


def split_conformal_interval(
    point: float, abs_residuals: Sequence[float], alpha: float
) -> tuple[float, float] | None:
    """Split-conformal: `point ± conformal_quantile(|reszt|, α)`; None gdy okno za małe."""
    q = conformal_quantile(abs_residuals, alpha)
    if q is None:
        return None
    return point - q, point + q


def empirical_return_interval(
    returns: Sequence[float], alpha: float
) -> tuple[float, float] | None:
    """Cold-start (predyktor 0): kwantyle skończonej próby zwrotów, po α/2 na ogon.

    Dolny = `floor((n+1)·α/2)`-ty, górny = `ceil((n+1)(1−α/2))`-ty najmniejszy
    zwrot; None, gdy któraś ranga wypada poza okno."""
    n = len(returns)
    low_rank, high_rank = _lower_rank(n, alpha / 2), _upper_rank(n, alpha / 2)
    if n == 0 or low_rank < 1 or high_rank > n:
        return None
    ordered = sorted(returns)
    return ordered[low_rank - 1], ordered[high_rank - 1]


def aci_step(
    alpha_t: float,
    covered: bool,
    *,
    alpha: float = NOMINAL_ALPHA,
    gamma: float = ACI_GAMMA,
) -> float:
    """Jeden krok ACI: `α_{t+1} = α_t + γ(α − err_t)`, err_t = 1 przy pudle."""
    error = 0.0 if covered else 1.0
    return alpha_t + gamma * (alpha - error)


def aci_alpha(
    covered_history: Iterable[bool],
    *,
    alpha: float = NOMINAL_ALPHA,
    gamma: float = ACI_GAMMA,
) -> float:
    """α_t po serii trafień/pudeł, startując od nominalnego α."""
    alpha_t = alpha
    for covered in covered_history:
        alpha_t = aci_step(alpha_t, covered, alpha=alpha, gamma=gamma)
    return alpha_t


def _clamped_interval(
    window: Sequence[CalibrationPoint], center: float, alpha_t: float, *, cold_start: bool
) -> tuple[float, float]:
    """Przedział przy α_t z ACI. α_t może zejść poniżej poziomu, przy którym
    ranga wychodzi poza okno — wtedy bierzemy skrajną wartość okna (najszerszy
    skończony przedział) zamiast przedziału nieskończonego."""
    n = len(window)
    if cold_start:
        ordered = sorted(point.realized_return for point in window)
        low_rank = min(max(_lower_rank(n, alpha_t / 2), 1), n)
        high_rank = min(max(_upper_rank(n, alpha_t / 2), 1), n)
        return ordered[low_rank - 1], ordered[high_rank - 1]
    residuals = sorted(abs(p.realized_return - p.predicted_return) for p in window)
    q = residuals[min(max(_upper_rank(n, alpha_t), 1), n) - 1]
    return center - q, center + q


def move_interval(
    window: Sequence[CalibrationPoint],
    *,
    predicted_return: float,
    cold_start: bool,
    alpha: float = NOMINAL_ALPHA,
    gamma: float = ACI_GAMMA,
    min_size: int = MIN_CALIBRATION_SIZE,
    max_size: int = MAX_CALIBRATION_SIZE,
) -> MoveInterval:
    """Przedział zwrotu bieżącej predykcji z okna kalibracyjnego jej klasy aktywa.

    `window` rosnąco po czasie. Bierzemy `max_size` najnowszych punktów; przy
    mniej niż `min_size` zwracamy przedział niedostępny z licznością okna.
    ACI odtwarzamy na oknie: punkt t dostaje przedział z punktów [0, t) przy
    α_t, a trafienie albo pudło koryguje α_{t+1}. Cold-start → predyktor 0
    i kwantyle zwrotów; model aktywny → ŷ ± kwantyl |reszt|."""
    recent = list(window)[-max_size:]
    n = len(recent)
    if n < min_size:
        return MoveInterval(
            low=None,
            high=None,
            calibration_size=n,
            required_size=min_size,
            nominal_coverage=1 - alpha,
            alpha=alpha,
        )
    alpha_t = alpha
    for t in range(min_size, n):
        point = recent[t]
        center = 0.0 if cold_start else point.predicted_return
        low, high = _clamped_interval(recent[:t], center, alpha_t, cold_start=cold_start)
        alpha_t = aci_step(
            alpha_t, low <= point.realized_return <= high, alpha=alpha, gamma=gamma
        )
    center = 0.0 if cold_start else predicted_return
    low, high = _clamped_interval(recent, center, alpha_t, cold_start=cold_start)
    return MoveInterval(
        low=low,
        high=high,
        calibration_size=n,
        required_size=min_size,
        nominal_coverage=1 - alpha,
        alpha=alpha_t,
    )


def realized_coverage(
    outcomes: Iterable[tuple[float, float, float]],
    nominal: float = 1 - NOMINAL_ALPHA,
) -> IntervalCoverage:
    """Pokrycie zrealizowane z trójek `(dolna, górna, zrealizowana)`; granice
    wliczają się do pokrycia (przedział domknięty)."""
    hits = n = 0
    for low, high, realized in outcomes:
        n += 1
        if low <= realized <= high:
            hits += 1
    return IntervalCoverage(hits=hits, n=n, nominal=nominal)
