"""Kalibracja izotoniczna pewności (ML-28) z temporalną oceną poza próbą.

`calibrated_confidence` (#9) mapuje pewność na hit-rate jej kubełka ze shrinkage,
więc nie jest monotoniczna (kubełek 0.8 może dostać mniej niż 0.7), a kalibratora
nigdy nie oceniono poza próbą, na której powstał. Tu:

- `fit_isotonic` — regresja izotoniczna algorytmem pool-adjacent-violators (PAV):
  niemalejąca mapa pewność → trafność, remisy pewności łączone przed PAV;
- `select_isotonic_calibrator` — mapa dopasowana na [t−60 d, t−30 d) i oceniana
  Brierem na [t−30 d, t) przeciw surowej pewności i obecnemu shrinkage kubełkowemu
  (kubełki z tego samego starszego okna). Do rankingu wchodzi wyłącznie wtedy, gdy
  wygrywa z oboma poza próbą — i jest to dokładnie ta oceniona mapa.

Czysta domena, stdlib. Próbki z pewnością spoza [0, 1] albo NaN odpadają (PORT-09).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.domain.calibration_curve import brier_score, calibrated_confidence, reliability_curve

# Okna jako (wiek najstarszej próbki, wiek najmłodszej) — granica starsza włącznie.
ISOTONIC_FIT_WINDOW = (timedelta(days=60), timedelta(days=30))
ISOTONIC_EVAL_WINDOW = (timedelta(days=30), timedelta(0))
# Minimum próbek w KAŻDYM oknie — przy małym n schodkowy kalibrator jest przeuczony.
MIN_ISOTONIC_SAMPLES = 30

Sample = tuple[float, bool]
TimedSample = tuple[datetime, float, bool]


@dataclass(frozen=True)
class IsotonicMap:
    """Niemalejąca mapa pewność → trafność: węzły (`knots`, rosnąco) i ich
    dopasowane wartości. Między węzłami interpolacja liniowa, poza nimi wartość
    skrajnego węzła; NaN zostaje NaN (nie ma pozycji na osi)."""

    knots: tuple[float, ...]
    values: tuple[float, ...]

    def apply(self, raw: float) -> float:
        if math.isnan(raw):
            return raw
        if raw <= self.knots[0]:
            return self.values[0]
        if raw >= self.knots[-1]:
            return self.values[-1]
        right = bisect_right(self.knots, raw)
        x0, x1 = self.knots[right - 1], self.knots[right]
        y0, y1 = self.values[right - 1], self.values[right]
        return y0 + (y1 - y0) * (raw - x0) / (x1 - x0)


def _pooled_ties(samples: Sequence[Sample]) -> list[tuple[float, float, int]]:
    """(pewność, trafność, waga) per unikalna pewność z [0, 1], rosnąco."""
    hits: dict[float, int] = {}
    counts: dict[float, int] = {}
    for confidence, hit in samples:
        if not 0.0 <= confidence <= 1.0:
            continue
        counts[confidence] = counts.get(confidence, 0) + 1
        hits[confidence] = hits.get(confidence, 0) + (1 if hit else 0)
    return [(c, hits[c] / counts[c], counts[c]) for c in sorted(counts)]


def fit_isotonic(samples: Sequence[Sample]) -> IsotonicMap | None:
    """Regresja izotoniczna PAV na próbkach `(pewność, trafienie)`; None bez próbek.

    Bloki sąsiadów łamiących monotoniczność są łączone średnią ważoną liczbą
    próbek, aż ciąg poziomów jest niemalejący; każdy węzeł dostaje poziom swojego
    bloku."""
    points = _pooled_ties(samples)
    if not points:
        return None
    # Bloki: [suma trafności × waga, waga, liczba węzłów].
    blocks: list[list[float]] = []
    for _, rate, weight in points:
        blocks.append([rate * weight, float(weight), 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, mass, size = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += mass
            blocks[-1][2] += size
    values: list[float] = []
    for total, mass, size in blocks:
        values.extend([total / mass] * int(size))
    return IsotonicMap(knots=tuple(c for c, _, _ in points), values=tuple(values))


@dataclass(frozen=True)
class IsotonicSelection:
    """Decyzja reguły ML-28: czy mapa izotoniczna wchodzi do rankingu sygnałów.

    `calibrator` — oceniona poza próbą mapa (dopasowana na starszym oknie) albo
    None; Briery z okna oceny (None, gdy któreś okno ma za mało próbek);
    `n_fit` / `n_eval` — liczności okien; `reason` — polski powód decyzji."""

    use_isotonic: bool
    calibrator: IsotonicMap | None
    brier_isotonic: float | None
    brier_raw: float | None
    brier_buckets: float | None
    n_fit: int
    n_eval: int
    reason: str


def _window(
    samples: Sequence[TimedSample], now: datetime, window: tuple[timedelta, timedelta]
) -> list[Sample]:
    oldest, youngest = window
    return [
        (confidence, hit)
        for stamp, confidence, hit in samples
        if now - oldest <= stamp < now - youngest and 0.0 <= confidence <= 1.0
    ]


def select_isotonic_calibrator(
    samples: Sequence[TimedSample], *, now: datetime, bucket_min_count: int = 10
) -> IsotonicSelection:
    """Reguła ML-28 na próbkach `(chwila, pewność, trafienie)` predykcji kierunkowych.

    Mapa i kubełki (shrinkage `bucket_min_count`, jak w rankingu #9) powstają na
    [t−60 d, t−30 d), Brier liczy się na [t−30 d, t). Mapa wchodzi do rankingu tylko,
    gdy jej Brier jest niższy od surowej pewności i od kubełków; inaczej ranking
    zostaje przy kubełkach. Okno z mniej niż `MIN_ISOTONIC_SAMPLES` → bez decyzji."""
    fit = _window(samples, now, ISOTONIC_FIT_WINDOW)
    evaluation = _window(samples, now, ISOTONIC_EVAL_WINDOW)
    calibrator = fit_isotonic(fit)
    if calibrator is None or min(len(fit), len(evaluation)) < MIN_ISOTONIC_SAMPLES:
        return IsotonicSelection(
            False,
            None,
            None,
            None,
            None,
            len(fit),
            len(evaluation),
            f"za mało próbek do oceny poza próbą ({len(fit)} / {len(evaluation)} "
            f"< {MIN_ISOTONIC_SAMPLES})",
        )
    buckets = reliability_curve(fit)
    brier_isotonic = brier_score([(calibrator.apply(c), hit) for c, hit in evaluation])
    brier_raw = brier_score(evaluation)
    brier_buckets = brier_score(
        [(calibrated_confidence(c, buckets, bucket_min_count), hit) for c, hit in evaluation]
    )
    wins = brier_isotonic < min(brier_raw, brier_buckets)
    reason = (
        f"izotoniczna wygrywa poza próbą (Brier {brier_isotonic:.4f} vs surowa "
        f"{brier_raw:.4f}, kubełki {brier_buckets:.4f})"
        if wins
        else f"izotoniczna nie wygrywa poza próbą (Brier {brier_isotonic:.4f} vs "
        f"surowa {brier_raw:.4f}, kubełki {brier_buckets:.4f})"
    )
    return IsotonicSelection(
        use_isotonic=wins,
        calibrator=calibrator if wins else None,
        brier_isotonic=brier_isotonic,
        brier_raw=brier_raw,
        brier_buckets=brier_buckets,
        n_fit=len(fit),
        n_eval=len(evaluation),
        reason=reason,
    )
