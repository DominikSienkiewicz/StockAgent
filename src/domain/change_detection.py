"""NIEZ-20: detektor zmiany (Page-Hinkley) na kroczącej trafności kierunkowej.

Degradację trafności widać dziś tylko jako ruchomy procent w mailu. Page-Hinkley
daje datę i odróżnia zmianę od szumu: sumuje odchylenia trafności cyklu od
bieżącej średniej (z tolerancją `delta`) i alarmuje, gdy suma spadnie o więcej
niż `threshold` poniżej swojego maksimum. Początek spadku to pierwszy cykl po
tym maksimum.

Parametry są pre-rejestrowane (stałe poniżej), nie dobierane po fakcie. Przy
odchyleniu trafności cyklu ok. 0.2 (kilka predykcji na cykl) fałszywy alarm
pada średnio raz na kilkaset cykli, a spadek o 20 pp wychodzi po ok. 20
cyklach. Przy rzadkich predykcjach (mało cykli z kierunkiem) detektor reaguje
więc z opóźnieniem tygodni — to cena odporności na szum.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.domain.equity_curve import signal_of
from src.domain.track_record import ScoredCall

# Pre-rejestrowane parametry detektora (NIEZ-20).
PH_DELTA = 0.05
PH_THRESHOLD = 2.0
# Przed tyloma cyklami nie ma poziomu odniesienia — alarmu nie ma.
PH_MIN_SAMPLES = 10


@dataclass(frozen=True)
class ChangeAlarm:
    """Alarm detektora: indeks cyklu alarmu, szacowany początek spadku
    (indeks pierwszego cyklu po maksimum sumy) i wartość statystyki."""

    alarm_at: int
    change_at: int
    statistic: float


def page_hinkley(
    outcomes: Sequence[float],
    delta: float = PH_DELTA,
    threshold: float = PH_THRESHOLD,
    *,
    min_samples: int = PH_MIN_SAMPLES,
) -> ChangeAlarm | None:
    """Pierwszy alarm SPADKU średniej w sekwencji `outcomes` albo None.

    m_t = Σ (x_i − x̄_i + delta), gdzie x̄_i to średnia do cyklu i włącznie;
    alarm, gdy max(m) − m_t > threshold (od `min_samples` obserwacji)."""
    if delta < 0:
        raise ValueError("delta must be >= 0")
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    mean = 0.0
    cumulative = 0.0
    peak = 0.0
    peak_at = -1
    for index, value in enumerate(outcomes):
        mean += (value - mean) / (index + 1)
        cumulative += value - mean + delta
        if cumulative >= peak:
            peak, peak_at = cumulative, index
        statistic = peak - cumulative
        if index + 1 >= min_samples and statistic > threshold:
            return ChangeAlarm(alarm_at=index, change_at=peak_at + 1, statistic=statistic)
    return None


@dataclass(frozen=True)
class AccuracyDrop:
    """Istotny spadek trafności: od którego cyklu (`since`, data ISO), kiedy
    padł alarm, trafność przed i od zmiany oraz liczba cykli w sekwencji."""

    since: str
    alarm_cycle: str
    before_rate: float
    after_rate: float
    cycles: int

    def describe_pl(self) -> str:
        return (
            f"Detektor zmiany (Page-Hinkley): trafność spadła istotnie od {self.since} "
            f"({round(self.before_rate * 100)}% → {round(self.after_rate * 100)}%, "
            f"alarm w cyklu {self.alarm_cycle}, {self.cycles} cykli)."
        )


def _pooled_rate(groups: Sequence[Sequence[bool]]) -> float:
    total = sum(len(group) for group in groups)
    return sum(sum(group) for group in groups) / total if total else 0.0


def detect_accuracy_drop(calls: Sequence[ScoredCall]) -> AccuracyDrop | None:
    """Page-Hinkley na trafności kierunkowej per cykl (SIDEWAYS poza
    sekwencją — nie stawia kierunku), w porządku chronologicznym cykli."""
    by_cycle: dict[str, list[bool]] = {}
    for call in calls:
        if signal_of(call.predicted_trend) != 0 and call.cycle:
            by_cycle.setdefault(call.cycle, []).append(call.is_correct)
    keys = sorted(by_cycle)
    groups = [by_cycle[key] for key in keys]
    alarm = page_hinkley([sum(group) / len(group) for group in groups])
    if alarm is None:
        return None
    return AccuracyDrop(
        since=keys[alarm.change_at],
        alarm_cycle=keys[alarm.alarm_at],
        before_rate=_pooled_rate(groups[: alarm.change_at]),
        after_rate=_pooled_rate(groups[alarm.change_at :]),
        cycles=len(groups),
    )
