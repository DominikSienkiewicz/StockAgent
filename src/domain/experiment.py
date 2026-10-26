"""Eksperymenty w pętli (EXP-05) — deterministyczny przydział ramienia i porównanie ramion.

Pytania „czy RAG pomaga” albo „czy ślepa rada jest lepsza” są przyczynowe.
Porównanie przed/po miesza zmianę z rynkiem, a losowy przydział per
symbol-dzień daje odpowiedź na tym samym okresie i tych samych spółkach.

Przydział to sha256(`id|symbol|data`) mod liczba ramion: deterministyczny,
odtwarzalny i bez stanu — powtórzony cykl dnia trafia do tego samego ramienia,
a id w kluczu robi z dwóch eksperymentów niezależne (ortogonalne) losowania.

Pierwsze ramię jest KONTROLNE w sensie status quo: zachowanie identyczne jak
bez eksperymentu. Pozostałe to zabiegi ze słownika `TREATMENT_ARMS` — kod zna
ich efekt i część promptu, którą zmieniają. Na jedną część promptu może naraz
działać co najwyżej jeden włączony eksperyment (`validate_experiments`), bo dwa
zabiegi na tym samym prompcie mieszają swoje efekty przy małym n na ramię.

Porównanie ramion liczy różnicę (zabieg − kontrola) średniej metryki z n i n_eff
per ramię oraz przedziałem z bootstrapu blokowego po cyklach (dniach UTC).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from src.domain.equity_curve import signal_of
from src.domain.segment_evidence import council_trend, trend_hit
from src.domain.significance import cluster_bootstrap_difference_ci, effective_n_values
from src.domain.track_record import group_by_cycle


class PromptPart(StrEnum):
    """Część promptu, którą zmienia zabieg."""

    PREDICTION = "prediction_prompt"
    COUNCIL = "council_prompt"


class ExperimentMetric(StrEnum):
    """Metryka porównania ramion; każda to średnia wkładów pojedynczych predykcji."""

    HIT_RATE = "hit_rate"
    BRIER = "brier"
    COUNCIL_LLM_AGREEMENT = "council_llm_agreement"
    COUNCIL_HIT_RATE = "council_hit_rate"
    COUNCIL_BRIER = "council_brier"

    @property
    def lower_is_better(self) -> bool:
        return self in _LOWER_IS_BETTER


_LOWER_IS_BETTER = frozenset({ExperimentMetric.BRIER, ExperimentMetric.COUNCIL_BRIER})

# EXP-02: rada bez linii trendu i pewności głównego LLM.
ARM_BLIND_COUNCIL = "blind"
# EXP-03: prompt predykcji bez lekcji z reflect i bez analogów RAG.
ARM_NO_MEMORY = "no_memory"

# Zabiegi znane kodowi → część promptu. Ramię spoza słownika nie miałoby
# żadnego efektu (test A/A udający eksperyment), więc definicja z nim pada.
TREATMENT_ARMS: Mapping[str, PromptPart] = {
    ARM_BLIND_COUNCIL: PromptPart.COUNCIL,
    ARM_NO_MEMORY: PromptPart.PREDICTION,
}

METRICS_BY_PART: Mapping[PromptPart, tuple[ExperimentMetric, ...]] = {
    PromptPart.PREDICTION: (ExperimentMetric.HIT_RATE, ExperimentMetric.BRIER),
    PromptPart.COUNCIL: (
        ExperimentMetric.COUNCIL_LLM_AGREEMENT,
        ExperimentMetric.COUNCIL_HIT_RATE,
        ExperimentMetric.COUNCIL_BRIER,
    ),
}


def assign_arm(experiment_id: str, symbol: str, cycle_date: date, arms: Sequence[str]) -> str:
    """Ramię dla klucza symbol-dzień: sha256(`id|symbol|data ISO`) mod liczba ramion."""
    if not arms:
        raise ValueError("experiment needs at least one arm")
    key = f"{experiment_id}|{symbol}|{cycle_date.isoformat()}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return arms[int.from_bytes(digest, "big") % len(arms)]


@dataclass(frozen=True)
class Experiment:
    """Definicja eksperymentu z `config.toml`.

    Pola:
        id: identyfikator (klucz przydziału i kolumny `experiment_arms`).
        hypothesis: hipoteza do raportu `evaluate`.
        arms: ramiona; pierwsze kontrolne (status quo), reszta to zabiegi
            z `TREATMENT_ARMS` na jednej części promptu.
        start, end: okno aktywności, obie daty włącznie (dzień UTC cyklu).
        primary_metric: metryka główna, jedna z metryk części promptu.
        enabled: wyłączony eksperyment nigdy nie jest aktywny.
    """

    id: str
    hypothesis: str
    arms: tuple[str, ...]
    start: date
    end: date
    primary_metric: ExperimentMetric
    enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "arms", tuple(self.arms))
        if not self.id.strip():
            raise ValueError("experiment id must not be empty")
        if len(self.arms) < 2 or len(set(self.arms)) != len(self.arms):
            raise ValueError(f"{self.id}: needs at least two distinct arms (got {self.arms})")
        if self.arms[0] in TREATMENT_ARMS:
            raise ValueError(
                f"{self.id}: the first arm is the control (status quo), not a treatment"
            )
        unknown = [arm for arm in self.arms[1:] if arm not in TREATMENT_ARMS]
        if unknown:
            raise ValueError(
                f"{self.id}: unknown treatment arms {unknown} (known: {sorted(TREATMENT_ARMS)})"
            )
        if len({TREATMENT_ARMS[arm] for arm in self.arms[1:]}) != 1:
            raise ValueError(f"{self.id}: treatment arms must change one prompt part")
        if self.end < self.start:
            raise ValueError(f"{self.id}: window ends before it starts")
        if self.primary_metric not in self.metrics:
            raise ValueError(
                f"{self.id}: primary metric {self.primary_metric.value} does not fit "
                f"{self.part.value} (allowed: {[m.value for m in self.metrics]})"
            )

    @property
    def control_arm(self) -> str:
        return self.arms[0]

    @property
    def treatment_arms(self) -> tuple[str, ...]:
        return self.arms[1:]

    @property
    def part(self) -> PromptPart:
        return TREATMENT_ARMS[self.arms[1]]

    @property
    def metrics(self) -> tuple[ExperimentMetric, ...]:
        return METRICS_BY_PART[self.part]

    def is_active(self, on: date) -> bool:
        """Aktywny = włączony i `on` w oknie [start, end]."""
        return self.enabled and self.start <= on <= self.end

    def arm_for(self, symbol: str, cycle_date: date) -> str:
        """Ramię symbolu w dniu cyklu; poza aktywnością zawsze kontrolne."""
        if not self.is_active(cycle_date):
            return self.control_arm
        return assign_arm(self.id, symbol, cycle_date, self.arms)


def validate_experiments(experiments: Sequence[Experiment]) -> None:
    """Zatrzymuje start przy zdublowanym id albo dwóch WŁĄCZONYCH eksperymentach
    na tej samej części promptu z nakładającymi się oknami.

    Sprawdzamy okna, nie dzisiejszą datę: konfiguracja poprawna dziś, a sprzeczna
    za tydzień, wywaliłaby cron dopiero w dniu startu drugiego eksperymentu.
    Eksperymenty na różnych częściach promptu mogą biec razem — id w kluczu
    przydziału daje im niezależne losowania (układ czynnikowy)."""
    ids = [experiment.id for experiment in experiments]
    duplicates = sorted({experiment_id for experiment_id in ids if ids.count(experiment_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate experiment ids: {duplicates}")
    enabled = [experiment for experiment in experiments if experiment.enabled]
    for index, first in enumerate(enabled):
        for second in enabled[index + 1 :]:
            overlap = first.start <= second.end and second.start <= first.end
            if first.part is second.part and overlap:
                raise ValueError(
                    f"experiments {first.id} and {second.id} both change {first.part.value} "
                    "in overlapping windows; run one at a time"
                )


def active_arms(experiments: Iterable[Experiment], symbol: str, cycle_date: date) -> dict[str, str]:
    """Ramiona AKTYWNYCH eksperymentów symbolu w dniu cyklu: `{id: ramię}`.
    Puste = brak aktywnego eksperymentu, cykl jak bez eksperymentów."""
    return {
        experiment.id: experiment.arm_for(symbol, cycle_date)
        for experiment in experiments
        if experiment.is_active(cycle_date)
    }


def has_treatment(arms: Mapping[str, str], treatment: str) -> bool:
    """Czy któryś aktywny eksperyment przydzielił ten zabieg. Jednoznaczne, bo
    zabieg należy do jednej części promptu, a na niej działa naraz jeden eksperyment."""
    return treatment in arms.values()


@dataclass(frozen=True)
class ExperimentCall:
    """Jedna rozliczona predykcja z ramieniem eksperymentu, jako prymitywy domeny.

    - `cycle` — klucz cyklu (dzień UTC predykcji, ISO); predykcje dnia są skorelowane,
    - `is_correct` — zapisane `is_trend_correct` trendu LLM,
    - `confidence` — deklarowana pewność LLM,
    - `realized_change` — zrealizowany ruch ceny (ułamek) albo None,
    - `council_recommendation`, `consensus_strength` — werdykt rady albo None.
    """

    cycle: str
    arm: str
    predicted_trend: str
    is_correct: bool | None
    confidence: float | None = None
    realized_change: float | None = None
    council_recommendation: str | None = None
    consensus_strength: float | None = None


def _hit(hit: bool) -> float:
    return 1.0 if hit else 0.0


def metric_value(metric: ExperimentMetric, call: ExperimentCall) -> float | None:
    """Wkład predykcji do metryki albo None, gdy predykcja metryki nie dotyczy.

    - trafność LLM — tylko BULLISH/BEARISH (SIDEWAYS nie stawia zakładu),
    - Brier LLM — pewność vs trafienie na każdym rozliczonym trendzie,
    - zgodność rady z LLM — rekomendacja (BUY/SELL/HOLD) jako trend vs trend LLM,
    - trafność rady — tylko BUY/SELL na zrealizowanym ruchu (`trend_hit`),
    - Brier rady — `consensus_strength` vs trafienie rekomendacji (HOLD = pasmo).
    """
    if metric is ExperimentMetric.HIT_RATE:
        if call.is_correct is None or signal_of(call.predicted_trend) == 0:
            return None
        return _hit(call.is_correct)
    if metric is ExperimentMetric.BRIER:
        if call.is_correct is None or call.confidence is None:
            return None
        return (call.confidence - _hit(call.is_correct)) ** 2
    council = council_trend(call.council_recommendation)
    if council is None:
        return None
    if metric is ExperimentMetric.COUNCIL_LLM_AGREEMENT:
        return _hit(council == call.predicted_trend.upper())
    if call.realized_change is None:
        return None
    council_hit = trend_hit(council, call.realized_change)
    if metric is ExperimentMetric.COUNCIL_HIT_RATE:
        return None if signal_of(council) == 0 else _hit(council_hit)
    if call.consensus_strength is None:
        return None
    return (call.consensus_strength - _hit(council_hit)) ** 2


@dataclass(frozen=True)
class ArmSummary:
    """Ramię w porównaniu: liczba predykcji z wartością metryki, n_eff i średnia."""

    arm: str
    n: int
    n_eff: float
    mean: float | None


@dataclass(frozen=True)
class ArmComparison:
    """Metryka zabiegu na tle kontroli.

    - `difference` — średnia zabiegu − średnia kontroli (None, gdy ramię puste),
    - `interval` — przedział z bootstrapu blokowego po cyklach; None przy
      < `significance.MIN_BOOTSTRAP_CYCLES` cyklach albo pustym ramieniu,
    - `cycles` — liczba cykli z wartością metryki w którymkolwiek ramieniu.
    """

    metric: ExperimentMetric
    control: ArmSummary
    treatment: ArmSummary
    difference: float | None
    interval: tuple[float, float] | None
    cycles: int


def _summarize_arm(arm: str, keyed: Sequence[tuple[str, float]]) -> ArmSummary:
    values = [value for _, value in keyed]
    return ArmSummary(
        arm=arm,
        n=len(values),
        n_eff=effective_n_values(group_by_cycle(keyed)),
        mean=sum(values) / len(values) if values else None,
    )


def _compare(
    metric: ExperimentMetric,
    calls: Sequence[ExperimentCall],
    control: str,
    treatment: str,
    *,
    n_boot: int,
    seed: int,
) -> ArmComparison:
    by_arm: dict[str, list[tuple[str, float]]] = {control: [], treatment: []}
    for call in calls:
        value = metric_value(metric, call)
        if value is not None and call.arm in by_arm:
            by_arm[call.arm].append((call.cycle, value))
    control_summary = _summarize_arm(control, by_arm[control])
    treatment_summary = _summarize_arm(treatment, by_arm[treatment])
    cycles = sorted({cycle for keyed in by_arm.values() for cycle, _ in keyed})
    per_cycle = [
        (
            [value for key, value in by_arm[control] if key == cycle],
            [value for key, value in by_arm[treatment] if key == cycle],
        )
        for cycle in cycles
    ]
    difference = (
        None
        if control_summary.mean is None or treatment_summary.mean is None
        else treatment_summary.mean - control_summary.mean
    )
    return ArmComparison(
        metric=metric,
        control=control_summary,
        treatment=treatment_summary,
        difference=difference,
        interval=cluster_bootstrap_difference_ci(per_cycle, n_boot=n_boot, seed=seed),
        cycles=len(cycles),
    )


def compare_experiment(
    experiment: Experiment,
    calls: Sequence[ExperimentCall],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[ArmComparison, ...]:
    """Każdy zabieg na tle kontroli, dla każdej metryki części promptu
    (w kolejności `METRICS_BY_PART`). Predykcje innych ramion są pomijane."""
    return tuple(
        _compare(metric, calls, experiment.control_arm, treatment, n_boot=n_boot, seed=seed)
        for treatment in experiment.treatment_arms
        for metric in experiment.metrics
    )
