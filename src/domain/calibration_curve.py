"""Domena krzywej kalibracji zaufania (reliability diagram).

Czysta statystyka, ZERO importów zewnętrznych (tylko stdlib). Partycjonujemy
próbki `(confidence ∈ [0,1], was_correct)` na kubełki o równej szerokości,
liczymy per niepusty kubełek liczność, średnią pewność oraz częstość trafień
(hit_rate), a na bazie krzywej — ECE (Expected Calibration Error),
ważoną licznością średnią rozjazdu |hit_rate − mean_confidence|.

To NIE jest sędzia LLM z `domain/prediction.py` — tu nie ma żadnej oceny
jakościowej, wyłącznie liczbowy diagram niezawodności predykcji.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationBucket:
    """Pojedynczy słupek diagramu niezawodności.

    - `lower`/`upper` — krawędzie przedziału pewności,
    - `count` — liczba próbek w kubełku,
    - `mean_confidence` — średnia deklarowana pewność w kubełku,
    - `hit_rate` — odsetek trafionych predykcji (realizacja).
    Idealna kalibracja oznacza `mean_confidence == hit_rate`.
    """

    lower: float
    upper: float
    count: int
    mean_confidence: float
    hit_rate: float


def reliability_curve(
    samples: Sequence[tuple[float, bool]], n_buckets: int = 10
) -> list[CalibrationBucket]:
    """Buduje krzywą kalibracji z próbek `(confidence, was_correct)`.

    Przedział `[0, 1]` dzielony jest na `n_buckets` kubełków o równej
    szerokości. Każda próbka trafia do kubełka wg swojej pewności; wartość
    dokładnie `1.0` ląduje w OSTATNIM kubełku (a nie poza zakresem). Dla
    każdego NIEPUSTEGO kubełka liczymy `count`, `mean_confidence` (średnia
    pewności w kubełku) i `hit_rate` (odsetek trafień). Puste kubełki są
    pomijane. Wynik posortowany rosnąco po dolnej krawędzi. Puste wejście
    daje `[]`.
    """
    if not samples or n_buckets <= 0:
        return []

    width = 1.0 / n_buckets
    # Akumulatory per indeks kubełka: suma pewności i liczba trafień.
    conf_sum: dict[int, float] = {}
    hit_count: dict[int, int] = {}
    total: dict[int, int] = {}

    for confidence, was_correct in samples:
        idx = int(confidence / width)
        # confidence == 1.0 (lub błąd zmiennoprzecinkowy w górę) → ostatni kubełek.
        idx = min(idx, n_buckets - 1)
        conf_sum[idx] = conf_sum.get(idx, 0.0) + confidence
        hit_count[idx] = hit_count.get(idx, 0) + (1 if was_correct else 0)
        total[idx] = total.get(idx, 0) + 1

    buckets: list[CalibrationBucket] = []
    for idx in sorted(total):
        count = total[idx]
        buckets.append(
            CalibrationBucket(
                lower=idx * width,
                upper=(idx + 1) * width,
                count=count,
                mean_confidence=conf_sum[idx] / count,
                hit_rate=hit_count[idx] / count,
            )
        )
    return buckets


def expected_calibration_error(buckets: Sequence[CalibrationBucket]) -> float:
    """Expected Calibration Error — ważona licznością średnia rozjazdu.

    ECE = Σ (count_b / N) · |hit_rate_b − mean_confidence_b|, gdzie N to
    łączna liczność wszystkich kubełków. Brak kubełków → 0.0. Wynik to
    pojedyncza liczba: im bliżej 0, tym lepiej skalibrowana pewność.
    """
    total = sum(b.count for b in buckets)
    if total == 0:
        return 0.0
    weighted = sum(
        b.count * abs(b.hit_rate - b.mean_confidence) for b in buckets
    )
    return weighted / total
