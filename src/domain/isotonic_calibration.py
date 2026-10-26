"""Kalibracja izotoniczna pewności (ML-28) z temporalną oceną poza próbą.

`calibrated_confidence` (R-9) mapuje pewność na hit-rate jej kubełka ze shrinkage,
więc nie jest monotoniczna (kubełek 0.8 może dostać mniej niż 0.7), a kalibratora
nigdy nie oceniono poza próbą, na której powstał. Tu:

- `fit_isotonic` — regresja izotoniczna algorytmem pool-adjacent-violators (PAV):
  niemalejąca mapa pewność → trafność, remisy pewności łączone przed PAV;
- `select_isotonic_calibrator` — mapa dopasowana na [t−60 d, t−30 d) i oceniana
  Brierem na [t−30 d, t) przeciw surowej pewności i obecnemu shrinkage kubełkowemu
  (kubełki z tego samego starszego okna). Do rankingu wchodzi wyłącznie wtedy, gdy
  wygrywa z oboma poza próbą — i jest to dokładnie ta oceniona mapa;
- `select_class_isotonic_calibrators` (ML-54) — ta sama reguła osobno dla krypto
  i dla reszty, gdy klasa ma dość próbek w obu oknach; mniejsza klasa dostaje
  mapę globalną.

Czysta domena, stdlib. Próbki z pewnością spoza [0, 1] albo NaN odpadają (PORT-09).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from src.domain.calibration_curve import brier_score, calibrated_confidence, reliability_curve

# Okna jako (wiek najstarszej próbki, wiek najmłodszej) — granica starsza włącznie.
ISOTONIC_FIT_WINDOW = (timedelta(days=60), timedelta(days=30))
ISOTONIC_EVAL_WINDOW = (timedelta(days=30), timedelta(0))
# Minimum próbek w KAŻDYM oknie — przy małym n schodkowy kalibrator jest przeuczony.
MIN_ISOTONIC_SAMPLES = 30
# ML-54: klasa aktywa dostaje własną mapę od tylu próbek w KAŻDYM oknie — tyle samo,
# ile potrzebuje ocena poza próbą mapy globalnej, bo klasa przechodzi tę samą ocenę.
MIN_CLASS_ISOTONIC_SAMPLES = MIN_ISOTONIC_SAMPLES

Sample = tuple[float, bool]
TimedSample = tuple[datetime, float, bool]


class CalibrationClass(Enum):
    """ML-54: klasa aktywa mapy kalibracji. Krypto ma inną zmienność i inne pasmo
    SIDEWAYS niż akcje i ETF-y, więc wspólna mapa uśrednia dwie różne krzywe."""

    CRYPTO = "CRYPTO"
    NON_CRYPTO = "NON_CRYPTO"


# Klasa None (wiersz bez symbolu) liczy się wyłącznie do mapy globalnej.
ClassedSample = tuple[datetime, float, bool, CalibrationClass | None]


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

    Mapa i kubełki (shrinkage `bucket_min_count`, jak w rankingu R-9) powstają na
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


@dataclass(frozen=True)
class ClassIsotonicSelection:
    """ML-54: decyzje ML-28 dla rankingu — globalna i per klasa aktywa.

    `class_selections` ma wyłącznie klasy z co najmniej `MIN_CLASS_ISOTONIC_SAMPLES`
    próbkami w każdym oknie; taka klasa ma własną ocenę poza próbą."""

    global_selection: IsotonicSelection
    class_selections: Mapping[CalibrationClass, IsotonicSelection]

    def calibrator_for(self, calibration_class: CalibrationClass | None) -> IsotonicMap | None:
        """Mapa pod ranking symbolu klasy `calibration_class`.

        Klasa z własną oceną dostaje wyłącznie swoją decyzję: mapę klasy, gdy wygrała
        poza próbą w tej klasie, inaczej None (kubełki) — mapa globalna nie była
        oceniana na samej tej klasie. Klasa z małą próbą i klasa nieznana (None)
        dostają mapę globalną."""
        own = (
            self.class_selections.get(calibration_class) if calibration_class is not None else None
        )
        return (own if own is not None else self.global_selection).calibrator


def select_class_isotonic_calibrators(
    samples: Sequence[ClassedSample], *, now: datetime, bucket_min_count: int = 10
) -> ClassIsotonicSelection:
    """ML-54: reguła ML-28 globalnie i osobno dla krypto i dla reszty.

    Mapa globalna to `select_isotonic_calibrator` na wszystkich próbkach. Klasa
    z co najmniej `MIN_CLASS_ISOTONIC_SAMPLES` próbkami w każdym oknie przechodzi
    tę samą ocenę poza próbą na własnych próbkach; mniejsza zostaje przy globalnej."""
    unclassed = [(stamp, confidence, hit) for stamp, confidence, hit, _ in samples]
    global_selection = select_isotonic_calibrator(
        unclassed, now=now, bucket_min_count=bucket_min_count
    )
    class_selections: dict[CalibrationClass, IsotonicSelection] = {}
    for calibration_class in CalibrationClass:
        own = [
            (stamp, confidence, hit)
            for stamp, confidence, hit, sample_class in samples
            if sample_class is calibration_class
        ]
        selection = select_isotonic_calibrator(own, now=now, bucket_min_count=bucket_min_count)
        if min(selection.n_fit, selection.n_eval) >= MIN_CLASS_ISOTONIC_SAMPLES:
            class_selections[calibration_class] = selection
    return ClassIsotonicSelection(
        global_selection=global_selection, class_selections=class_selections
    )
