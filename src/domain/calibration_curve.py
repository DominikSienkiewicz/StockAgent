"""Domena krzywej kalibracji zaufania (reliability diagram).

Czysta statystyka, ZERO importów zewnętrznych (tylko stdlib). Partycjonujemy
próbki `(confidence ∈ [0,1], was_correct)` na kubełki o równej szerokości,
liczymy per niepusty kubełek liczność, średnią pewność oraz częstość trafień
(hit_rate), a na bazie krzywej — ECE (Expected Calibration Error),
ważoną licznością średnią rozjazdu |hit_rate − mean_confidence|.

To NIE jest sędzia LLM z `domain/prediction.py` — tu nie ma żadnej oceny
jakościowej, wyłącznie liczbowy diagram niezawodności predykcji.

ECE jest tylko metryką POMOCNICZĄ (WIA-36): prognoza „zawsze 55%” przy 55%
trafień ma ECE = 0, a zerową umiejętność. Główne metryki pewności to Brier
score, log loss i Brier Skill Score względem prognozy klimatologicznej
(stała stopa bazowa), plus rozkład Murphy'ego BS = REL − RES + UNC.
"""

from __future__ import annotations

import math
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


def find_bucket(
    raw: float, buckets: Sequence[CalibrationBucket]
) -> CalibrationBucket | None:
    """Znajduje kubełek, do którego wpada `raw` (semantyka reliability_curve).

    Dopasowanie w przedziale półotwartym `[lower, upper)` — spójne z tym, jak
    `reliability_curve` przydziela próbki. Górna krawędź `raw == upper` (np.
    `raw == 1.0` dla ostatniego kubełka `[0.9, 1.0]`) trafia do tego kubełka,
    tak samo jak `confidence == 1.0` ląduje w ostatnim słupku diagramu.
    """
    for bucket in buckets:
        if bucket.lower <= raw < bucket.upper:
            return bucket
    # Górna krawędź: raw dokładnie na `upper` (np. 1.0) → ten kubełek.
    for bucket in buckets:
        if raw == bucket.upper:
            return bucket
    return None


def calibrated_confidence(
    raw: float,
    buckets: Sequence[CalibrationBucket],
    min_count: int = 10,
) -> float:
    """Kalibruje surową pewność `raw` historycznym hit-rate jej kubełka.

    Zamiast oddawać użytkownikowi surowe np. 85% pewności, mapujemy `raw` na
    faktyczny hit-rate kubełka, do którego wpada, ale ze SHRINKAGE przy małej
    próbce — kubełek zbudowany na garstce obserwacji nie może w pełni
    przeciągnąć wyniku (niestacjonarność hit-rate: zmiana reżimu unieważnia
    historię, więc małej próbce ufamy proporcjonalnie mniej).

    Jawna formuła shrinkage (waga zaufania do historii kubełka):

        weight = count / (count + min_count)
        calibrated = raw + weight * (hit_rate - raw)

    - `count == min_count` → waga 0.5 (pół drogi między `raw` a `hit_rate`),
    - `count << min_count` → waga bliska 0 (wynik ~ `raw`),
    - `count >> min_count` → waga bliska 1 (wynik ~ `hit_rate` kubełka).

    Kontrakt graceful (brak historii = zachowanie identyczne jak dziś):
    - pusta lista kubełków → zwraca `raw`,
    - brak kubełka pasującego do `raw` → zwraca `raw`,
    - `min_count <= 0` (brak sensownego progu) → zwraca `raw`.

    Wynik jest zawsze przycięty do przedziału `[0, 1]`.
    """
    if not buckets or min_count <= 0:
        return raw
    bucket = find_bucket(raw, buckets)
    if bucket is None:
        return raw
    weight = bucket.count / (bucket.count + min_count)
    calibrated = raw + weight * (bucket.hit_rate - raw)
    # Przycięcie do [0, 1] — zabezpieczenie przed zdegenerowanymi danymi.
    return max(0.0, min(1.0, calibrated))


def brier_score(samples: Sequence[tuple[float, bool]]) -> float:
    """Brier score — średni kwadrat różnicy pewności i wyniku (0/1).

    0.0 to prognoza idealna; brak próbek → 0.0 (spójnie z ECE).
    """
    if not samples:
        return 0.0
    squared_errors = [(confidence - (1.0 if hit else 0.0)) ** 2 for confidence, hit in samples]
    return sum(squared_errors) / len(squared_errors)


def log_loss(samples: Sequence[tuple[float, bool]], eps: float = 1e-6) -> float:
    """Średnia ujemna log-wiarygodność; pewność przycięta do [eps, 1 − eps].

    Bez przycięcia pewność 1.0 przy pudle dawałaby nieskończoność. Brak
    próbek → 0.0.
    """
    if not samples:
        return 0.0
    total = 0.0
    for confidence, hit in samples:
        clipped = min(1.0 - eps, max(eps, confidence))
        total -= math.log(clipped if hit else 1.0 - clipped)
    return total / len(samples)


def murphy_decomposition(
    buckets: Sequence[CalibrationBucket],
) -> tuple[float, float, float]:
    """Rozkład Murphy'ego Briera na kubełkach: `(reliability, resolution, uncertainty)`.

    - reliability (REL) = Σ n_k (f_k − o_k)² / N — kara za rozjazd pewności
      i trafności (im niżej, tym lepiej),
    - resolution (RES) = Σ n_k (o_k − ō)² / N — na ile kubełki ODRÓŻNIAJĄ się
      trafnością od stopy bazowej (im wyżej, tym lepiej),
    - uncertainty (UNC) = ō (1 − ō) — trudność zadania; to zarazem Brier
      prognozy klimatologicznej.

    BS = REL − RES + UNC dokładnie, gdy pewność jest stała w kubełku; przy
    pewnościach rozrzuconych w kubełku to przybliżenie. Pewności skupione
    w 0.6–0.85 dają RES bliskie 0 — to prawda do pokazania, nie do ukrycia.
    """
    total = sum(b.count for b in buckets)
    if total == 0:
        return 0.0, 0.0, 0.0
    base_rate = sum(b.count * b.hit_rate for b in buckets) / total
    reliability = sum(b.count * (b.mean_confidence - b.hit_rate) ** 2 for b in buckets) / total
    resolution = sum(b.count * (b.hit_rate - base_rate) ** 2 for b in buckets) / total
    return reliability, resolution, base_rate * (1.0 - base_rate)


@dataclass(frozen=True)
class ProbabilityScores:
    """Metryki jakości deklarowanej pewności na jednym zbiorze próbek.

    `climatology_brier` to Brier prognozy „zawsze stopa bazowa”; `skill`
    (Brier Skill Score) = 1 − brier / climatology_brier — 0 to brak
    umiejętności, wartości ujemne są gorsze od klimatologii. None, gdy wyniki
    się nie różnią (klimatologia ma Brier 0 i nie ma czego bić).
    """

    n: int
    brier: float
    log_loss: float
    climatology_brier: float
    skill: float | None
    reliability: float
    resolution: float
    uncertainty: float


def probability_scores(
    samples: Sequence[tuple[float, bool]], n_buckets: int = 10
) -> ProbabilityScores | None:
    """Brier, log loss, BSS i rozkład Murphy'ego z próbek `(confidence, was_correct)`.

    Brak próbek → None (sekcja raportu sama się chowa).
    """
    if not samples:
        return None
    brier = brier_score(samples)
    base_rate = sum(1 for _, hit in samples if hit) / len(samples)
    climatology = base_rate * (1.0 - base_rate)
    reliability, resolution, uncertainty = murphy_decomposition(
        reliability_curve(samples, n_buckets)
    )
    return ProbabilityScores(
        n=len(samples),
        brier=brier,
        log_loss=log_loss(samples),
        climatology_brier=climatology,
        skill=None if climatology == 0.0 else 1.0 - brier / climatology,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
    )
