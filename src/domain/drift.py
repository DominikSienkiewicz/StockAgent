"""Domena: dryf rozkładu cech (ML-21) — biny kwantylowe i Population Stability Index.

PSI = Σ (a_i − e_i) · ln(a_i / e_i) po binach, gdzie `e` to udział binu w rozkładzie
odniesienia, a `a` w rozkładzie porównywanym. Konwencja branżowa: < 0.1 stabilnie,
0.1–0.25 umiarkowane przesunięcie, > 0.25 istotny dryf. Pusty bin dostaje udział
`PSI_EPSILON`, żeby logarytm był skończony.

Biny to kwantyle rozkładu odniesienia (domyślnie decyle), więc każdy bin ma w nim
podobny udział i PSI nie zależy od skali cechy. Czysta logika, stdlib only.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

# Próg „istotnego dryfu” — powyżej niego karta modelu ostrzega.
PSI_DRIFT_THRESHOLD = 0.25
# Udział przypisywany pustemu binowi (logarytm z zera nie istnieje).
PSI_EPSILON = 1e-4
# Liczba binów kwantylowych (decyle).
DEFAULT_BINS = 10
# Okno „ostatnich dni” porównywane z resztą próbki.
DRIFT_WINDOW_DAYS = 30
# Minimalna liczebność KAŻDEJ z porównywanych prób — poniżej PSI to szum.
MIN_PSI_SAMPLES = 20


def _finite(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(v)]


def _quantile(ordered: Sequence[float], probability: float) -> float:
    """Kwantyl jako odwrotność dystrybuanty empirycznej (bez interpolacji) — granica
    jest zawsze wartością z próby, więc cecha dyskretna nie dostaje granic „pomiędzy”."""
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def quantile_edges(values: Sequence[float], bins: int = DEFAULT_BINS) -> tuple[float, ...]:
    """Wewnętrzne granice binów — kwantyle 1/bins … (bins−1)/bins bez powtórzeń.

    Cecha dyskretna (np. sygnał LLM −1/0/+1) daje mniej binów zamiast pustych
    duplikatów. Brak skończonych wartości → brak granic (jeden bin)."""
    ordered = sorted(_finite(values))
    if not ordered:
        return ()
    edges: list[float] = []
    for k in range(1, bins):
        edge = _quantile(ordered, k / bins)
        if not edges or edge > edges[-1]:
            edges.append(edge)
    return tuple(edges)


def bin_fractions(values: Sequence[float], edges: Sequence[float]) -> tuple[float, ...]:
    """Udziały wartości w binach `(−∞, e_0], (e_0, e_1], …, (e_n, +∞)`.

    Zwraca `len(edges) + 1` udziałów sumujących się do 1; bez skończonych wartości
    — same zera (porównanie z taką próbką nie ma sensu, `recent_window_psi` jej nie
    dopuszcza)."""
    finite = _finite(values)
    counts = [0] * (len(edges) + 1)
    for value in finite:
        counts[bisect_left(edges, value)] += 1
    if not finite:
        return tuple(0.0 for _ in counts)
    return tuple(count / len(finite) for count in counts)


def population_stability_index(expected: Sequence[float], actual: Sequence[float]) -> float:
    """PSI dwóch rozkładów na TYCH SAMYCH binach (udziały, nie liczności)."""
    if len(expected) != len(actual) or not expected:
        raise ValueError("PSI wymaga niepustych udziałów na tych samych binach")
    total = 0.0
    for raw_expected, raw_actual in zip(expected, actual, strict=True):
        e = max(raw_expected, PSI_EPSILON)
        a = max(raw_actual, PSI_EPSILON)
        total += (a - e) * math.log(a / e)
    return total


def recent_window_psi(
    values: Sequence[float],
    stamps: Sequence[datetime],
    *,
    window_days: int = DRIFT_WINDOW_DAYS,
    bins: int = DEFAULT_BINS,
) -> float | None:
    """PSI ostatnich `window_days` dni względem starszej części tej samej próbki.

    Okno liczy się od NAJNOWSZEGO znacznika (deterministycznie, bez zegara).
    Biny to kwantyle starszej części. None, gdy któraś część ma mniej niż
    `MIN_PSI_SAMPLES` skończonych wartości — wtedy PSI byłby szumem."""
    if len(values) != len(stamps):
        raise ValueError("values i stamps muszą być wyrównane")
    if not stamps:
        return None
    cutoff = max(stamps) - timedelta(days=window_days)
    reference = [v for v, s in zip(values, stamps, strict=True) if s <= cutoff]
    recent = [v for v, s in zip(values, stamps, strict=True) if s > cutoff]
    if len(_finite(reference)) < MIN_PSI_SAMPLES or len(_finite(recent)) < MIN_PSI_SAMPLES:
        return None
    edges = quantile_edges(reference, bins)
    return population_stability_index(bin_fractions(reference, edges), bin_fractions(recent, edges))


@dataclass(frozen=True)
class FeatureDrift:
    """Cecha z istotnym dryfem i jej PSI."""

    feature: str
    psi: float


def drifted_features(
    psi_by_feature: Mapping[str, float | None],
    threshold: float = PSI_DRIFT_THRESHOLD,
) -> tuple[FeatureDrift, ...]:
    """Cechy z PSI powyżej progu, od największego dryfu. None (za mało danych) pomijamy."""
    drifts = [
        FeatureDrift(feature, psi)
        for feature, psi in psi_by_feature.items()
        if psi is not None and math.isfinite(psi) and psi > threshold
    ]
    return tuple(sorted(drifts, key=lambda drift: (-drift.psi, drift.feature)))
