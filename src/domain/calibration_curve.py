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

WIA-58: pewności LLM skupiają się w 0.6–0.85, więc kubełki równej szerokości
dają 2–3 niepuste. `strategy="equal_mass"` dzieli próbki po randze na kubełki
równoliczne (granice kwantylowe). ECE przy małym n jest obciążone w górę —
`debiased_ece` odejmuje ECE oczekiwane przy idealnej kalibracji, a
`calibration_error_evidence` dokłada n_eff i przedział z bootstrapu po cyklach.

WIA-38: kalibracja zakłada, że pewność niesie informację. `discrimination_auc`
(Mann–Whitney) i `risk_coverage_curve` sprawdzają, czy wyższa pewność znaczy
częstsze trafienie, a `confidence_reorders_ranking` to reguła: AUC < 0.55 przy
n_eff ≥ 50 → skalibrowana pewność nie przestawia rankingu sygnałów.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from src.domain.significance import cluster_bootstrap_statistic_ci, effective_n

BucketStrategy = Literal["equal_width", "equal_mass"]

# WIA-58: liczba symulacji Bernoulliego w korekcie obciążenia ECE.
DEFAULT_ECE_SIMULATIONS = 200

# WIA-38: poniżej tego AUC pewność nie odróżnia trafień od pudeł…
MIN_INFORMATIVE_AUC = 0.55
# …ale wyłączamy ranking dopiero przy tej efektywnej liczności — przy mniejszej
# próbie przedział AUC jest za szeroki, żeby o tym rozstrzygać.
MIN_RANKING_DISCRIMINATION_N_EFF = 50
# Pokrycia krzywej ryzyko–pokrycie: top 10%, 20%, …, 100% predykcji wg pewności.
DEFAULT_COVERAGES: tuple[float, ...] = tuple(round(0.1 * k, 1) for k in range(1, 11))


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
    samples: Sequence[tuple[float, bool]],
    n_buckets: int = 10,
    strategy: BucketStrategy = "equal_width",
) -> list[CalibrationBucket]:
    """Buduje krzywą kalibracji z próbek `(confidence, was_correct)`.

    `equal_width` (domyślnie): przedział `[0, 1]` dzielony jest na `n_buckets`
    kubełków o równej szerokości. Każda próbka trafia do kubełka wg swojej
    pewności; wartość dokładnie `1.0` ląduje w OSTATNIM kubełku (a nie poza
    zakresem). Dla każdego NIEPUSTEGO kubełka liczymy `count`,
    `mean_confidence` (średnia pewności w kubełku) i `hit_rate` (odsetek
    trafień). Puste kubełki są pomijane. Wynik posortowany rosnąco po dolnej
    krawędzi. Puste wejście daje `[]`.

    `equal_mass` (WIA-58): patrz `_equal_mass_curve`. Nieznana strategia →
    `ValueError`.

    PORT-09: próbka z pewnością spoza `[0, 1]` (skala 0–100, NaN) nie trafia
    do żadnego kubełka — inaczej 85 lądowało w ostatnim słupku i windowało ECE
    do ~84. Ile takich próbek odrzucono, mówi `out_of_range_confidences`.
    """
    if strategy not in ("equal_width", "equal_mass"):
        raise ValueError(f"Unknown bucket strategy: {strategy!r}")
    samples = [sample for sample in samples if _in_unit_interval(sample[0])]
    if not samples or n_buckets <= 0:
        return []
    if strategy == "equal_mass":
        return _equal_mass_curve(samples, n_buckets)

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


def _equal_mass_curve(
    samples: Sequence[tuple[float, bool]], n_buckets: int
) -> list[CalibrationBucket]:
    """Kubełki równoliczne: próbki posortowane po pewności dzielone po randze.

    Liczności różnią się najwyżej o 1 (pierwsze `n mod k` kubełków ma o jedną
    próbkę więcej); przy mniejszej liczbie próbek niż kubełków każdy kubełek ma
    jedną. Granice są kwantylowe: dolna krawędź to pierwsza pewność kubełka
    (0.0 dla pierwszego), górna — dolna krawędź następnego (1.0 dla
    ostatniego). Remisy pewności mogą trafić do dwóch sąsiednich kubełków —
    podział idzie po randze, więc `find_bucket` służy kubełkom równej szerokości.
    """
    ordered = sorted(samples, key=lambda sample: sample[0])
    k = min(n_buckets, len(ordered))
    base, extra = divmod(len(ordered), k)
    groups: list[list[tuple[float, bool]]] = []
    start = 0
    for index in range(k):
        size = base + (1 if index < extra else 0)
        groups.append(ordered[start : start + size])
        start += size
    lowers = [0.0] + [group[0][0] for group in groups[1:]]
    uppers = lowers[1:] + [1.0]
    return [
        CalibrationBucket(
            lower=lower,
            upper=upper,
            count=len(group),
            mean_confidence=sum(c for c, _ in group) / len(group),
            hit_rate=sum(1 for _, hit in group if hit) / len(group),
        )
        for group, lower, upper in zip(groups, lowers, uppers, strict=True)
    ]


def _in_unit_interval(confidence: float) -> bool:
    """Czy pewność leży w `[0, 1]`; NaN nie leży nigdzie."""
    return 0.0 <= confidence <= 1.0


def out_of_range_confidences(samples: Sequence[tuple[float, bool]]) -> int:
    """Liczba próbek, które `reliability_curve` odrzuca: pewność spoza `[0, 1]`
    albo NaN (PORT-09). Wołający może ją pokazać obok diagramu."""
    return sum(1 for confidence, _ in samples if not _in_unit_interval(confidence))


def expected_calibration_error(buckets: Sequence[CalibrationBucket]) -> float:
    """Expected Calibration Error — ważona licznością średnia rozjazdu.

    ECE = Σ (count_b / N) · |hit_rate_b − mean_confidence_b|, gdzie N to
    łączna liczność wszystkich kubełków. Brak kubełków → 0.0. Wynik to
    pojedyncza liczba: im bliżej 0, tym lepiej skalibrowana pewność.
    """
    total = sum(b.count for b in buckets)
    if total == 0:
        return 0.0
    weighted = sum(b.count * abs(b.hit_rate - b.mean_confidence) for b in buckets)
    return weighted / total


def find_bucket(raw: float, buckets: Sequence[CalibrationBucket]) -> CalibrationBucket | None:
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

    Wynik jest zawsze przycięty do przedziału `[0, 1]` — także na ścieżkach
    graceful (PORT-09), żeby surowe 1.7 nie wyszło z kalibracji jako 170%.
    NaN zostaje NaN: nie ma pozycji na osi, a przycięcie zrobiłoby z niego 100%.
    """
    if not buckets or min_count <= 0:
        return _clamp_unit_interval(raw)
    bucket = find_bucket(raw, buckets)
    if bucket is None:
        return _clamp_unit_interval(raw)
    weight = bucket.count / (bucket.count + min_count)
    calibrated = raw + weight * (bucket.hit_rate - raw)
    # Przycięcie do [0, 1] — zabezpieczenie przed zdegenerowanymi danymi.
    return _clamp_unit_interval(calibrated)


def _clamp_unit_interval(value: float) -> float:
    if math.isnan(value):
        return value
    return max(0.0, min(1.0, value))


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


def _ece(samples: Sequence[tuple[float, bool]], n_buckets: int, strategy: BucketStrategy) -> float:
    return expected_calibration_error(reliability_curve(samples, n_buckets, strategy))


def debiased_ece(
    samples: Sequence[tuple[float, bool]],
    n_buckets: int = 10,
    strategy: BucketStrategy = "equal_mass",
    n_sim: int = DEFAULT_ECE_SIMULATIONS,
    seed: int = 0,
) -> float:
    """WIA-58 — ECE minus ECE oczekiwane przy IDEALNEJ kalibracji tych pewności.

    Nawet idealnie skalibrowana prognoza ma przy małym n dodatnie ECE (szum
    trafień w kubełku). Oczekiwanie szacujemy symulacją: dla tych samych
    pewności losujemy wyniki Bernoulli(confidence) `n_sim` razy (deterministycznie
    dla `seed`) i bierzemy średnie ECE. Wynik może być ujemny — to estymator
    nieobciążony, nie odległość. Brak próbek albo `n_sim <= 0` → surowe ECE.
    """
    observed = _ece(samples, n_buckets, strategy)
    if not samples or n_sim <= 0:
        return observed
    rng = random.Random(seed)
    confidences = [confidence for confidence, _ in samples]
    simulated = [
        _ece([(c, rng.random() < c) for c in confidences], n_buckets, strategy)
        for _ in range(n_sim)
    ]
    return observed - sum(simulated) / len(simulated)


@dataclass(frozen=True)
class CalibrationErrorEvidence:
    """ECE z tym, co pozwala je ocenić (WIA-58).

    - `ece` — ECE na kubełkach `strategy` (domyślnie równolicznych),
    - `debiased` — ECE po korekcie obciążenia (`debiased_ece`),
    - `interval` — percentylowy przedział `ece` z bootstrapu po cyklach;
      None przy < `significance.MIN_BOOTSTRAP_CYCLES` cyklach,
    - `n` / `n_eff` — liczba próbek i efektywna liczność trafień po cyklach.
    """

    n: int
    n_eff: float
    ece: float
    debiased: float
    interval: tuple[float, float] | None


def calibration_error_evidence(
    samples_by_cycle: Sequence[Sequence[tuple[float, bool]]],
    n_buckets: int = 10,
    strategy: BucketStrategy = "equal_mass",
    *,
    n_sim: int = DEFAULT_ECE_SIMULATIONS,
    n_boot: int = 2000,
    seed: int = 0,
) -> CalibrationErrorEvidence | None:
    """ECE, ECE po korekcie, przedział z bootstrapu blokowego i n_eff.

    `samples_by_cycle` to próbki `(confidence, was_correct)` pogrupowane po
    cyklu (dniu UTC): predykcje jednego dnia są skorelowane, więc bootstrap
    losuje całe cykle, a n_eff liczy `effective_n` na trafieniach. Brak próbek
    → None.
    """
    samples = [sample for cycle in samples_by_cycle for sample in cycle]
    if not samples:
        return None
    return CalibrationErrorEvidence(
        n=len(samples),
        n_eff=effective_n([[hit for _, hit in cycle] for cycle in samples_by_cycle]),
        ece=_ece(samples, n_buckets, strategy),
        debiased=debiased_ece(samples, n_buckets, strategy, n_sim=n_sim, seed=seed),
        interval=cluster_bootstrap_statistic_ci(
            samples_by_cycle,
            lambda resample: _ece(resample, n_buckets, strategy),
            n_boot=n_boot,
            seed=seed,
        ),
    )


def discrimination_auc(samples: Sequence[tuple[float, bool]]) -> float | None:
    """WIA-38 — AUC pewności: P(pewność trafienia > pewność pudła), remis = ½.

    Statystyka Manna–Whitneya z rang (remisy dostają średnią rangę), O(n log n).
    0.5 = pewność nie odróżnia trafień od pudeł, 1.0 = każde trafienie ma wyższą
    pewność niż każde pudło. Bez trafień albo bez pudeł → None (nieokreślone).
    """
    hits = sum(1 for _, hit in samples if hit)
    misses = len(samples) - hits
    if hits == 0 or misses == 0:
        return None
    ordered = sorted(samples, key=lambda sample: sample[0])
    hit_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        # Rangi start+1 … end mają wspólną średnią rangę.
        average_rank = (start + 1 + end) / 2
        hit_rank_sum += average_rank * sum(1 for _, hit in ordered[start:end] if hit)
        start = end
    return (hit_rank_sum - hits * (hits + 1) / 2) / (hits * misses)


def risk_coverage_curve(
    samples: Sequence[tuple[float, bool]],
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> list[tuple[float, float]]:
    """WIA-38 — trafność przy zostawieniu top-k% predykcji wg pewności.

    Zwraca pary `(pokrycie, hit-rate)`. Krzywa rosnąca w lewo znaczy, że
    pewność porządkuje trafienia; płaska — że ranking po pewności jest szumem.
    Remis pewności na granicy top-k wchodzi proporcjonalnie (średnią trafność
    remisującej grupy), więc stała pewność daje dokładnie stopę bazową w
    każdym punkcie. Brak próbek → [].
    """
    if not samples:
        return []
    ordered = sorted(samples, key=lambda sample: sample[0], reverse=True)
    n = len(ordered)
    curve: list[tuple[float, float]] = []
    for coverage in coverages:
        k = max(1, min(n, math.ceil(round(coverage * n, 9))))
        threshold = ordered[k - 1][0]
        higher = [hit for confidence, hit in ordered if confidence > threshold]
        tied = [hit for confidence, hit in ordered if confidence == threshold]
        tied_rate = sum(tied) / len(tied)
        hits = sum(higher) + (k - len(higher)) * tied_rate
        curve.append((coverage, hits / k))
    return curve


@dataclass(frozen=True)
class DiscriminationEvidence:
    """AUC pewności z tym, co pozwala je ocenić (WIA-38).

    - `auc` — None, gdy w próbie nie ma trafień albo pudeł,
    - `interval` — percentylowy przedział AUC z bootstrapu po cyklach; None przy
      < `significance.MIN_BOOTSTRAP_CYCLES` cyklach,
    - `n` / `n_eff` — liczba próbek i efektywna liczność trafień po cyklach.
    """

    n: int
    n_eff: float
    auc: float | None
    interval: tuple[float, float] | None


def discrimination_evidence(
    samples_by_cycle: Sequence[Sequence[tuple[float, bool]]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> DiscriminationEvidence | None:
    """AUC, n_eff i przedział z bootstrapu losującego całe cykle; brak próbek → None."""
    samples = [sample for cycle in samples_by_cycle for sample in cycle]
    if not samples:
        return None
    return DiscriminationEvidence(
        n=len(samples),
        n_eff=effective_n([[hit for _, hit in cycle] for cycle in samples_by_cycle]),
        auc=discrimination_auc(samples),
        interval=cluster_bootstrap_statistic_ci(
            samples_by_cycle, discrimination_auc, n_boot=n_boot, seed=seed
        ),
    )


def confidence_reorders_ranking(evidence: DiscriminationEvidence) -> bool:
    """WIA-38 — reguła: czy skalibrowana pewność może przestawiać ranking sygnałów.

    False tylko przy `auc < MIN_INFORMATIVE_AUC` na próbie `n_eff >=
    MIN_RANKING_DISCRIMINATION_N_EFF`: wtedy pewność nie odróżnia trafień od
    pudeł, a kalibracja przesuwałaby wszystkich do stopy bazowej. Mniejsza
    próba albo AUC nieokreślone → True (zachowanie jak przed WIA-38).
    """
    if evidence.auc is None or evidence.n_eff < MIN_RANKING_DISCRIMINATION_N_EFF:
        return True
    return evidence.auc >= MIN_INFORMATIVE_AUC
