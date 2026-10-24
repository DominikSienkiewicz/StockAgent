"""Próbki kalibracji pewności z wierszy repozytorium (WIA-39).

`get_resolved_for_calibration` zwraca rozliczone predykcje z pewnością,
werdyktem kierunku, przewidzianym trendem i znacznikiem czasu. Tu zamieniamy je
na próbki `(confidence, was_correct)` domeny `calibration_curve` — osobno dla
predykcji kierunkowych i dla SIDEWAYS. „SIDEWAYS 0.8” trafia za darmo w spokojne
dni; w jednym kubełku z „BULLISH 0.8” zawyżałby hit-rate, którym ranking
sygnałów kalibruje zakłady na kierunek.

Próbki kierunkowe są dodatkowo pogrupowane po cyklu (dzień UTC znacznika, jak
w `report_evidence.cycle_key`): predykcje jednego cyklu są skorelowane, więc
n_eff i bootstrap blokowy (WIA-38, WIA-58) losują całe cykle. Wiersz bez
znacznika ląduje we wspólnym cyklu — to zaniża n_eff zamiast udawać
niezależność.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.domain.calibration_curve import CalibrationBucket, reliability_curve
from src.domain.equity_curve import signal_of
from src.domain.track_record import group_by_cycle

Sample = tuple[float, bool]


@dataclass(frozen=True)
class CalibrationSamples:
    """Próbki kalibracji rozdzielone na kierunkowe (po cyklach) i SIDEWAYS.

    Wiersz bez `predicted_trend` (odczyt sprzed WIA-39) liczy się jako
    kierunkowy — tak, jak traktowała go kalibracja do tej pory.
    """

    directional_by_cycle: tuple[tuple[Sample, ...], ...]
    sideways: tuple[Sample, ...]

    @property
    def directional(self) -> list[Sample]:
        """Próbki kierunkowe w kolejności cykli (chronologicznie)."""
        return [sample for cycle in self.directional_by_cycle for sample in cycle]


def _cycle_of(row: Mapping[str, Any]) -> str:
    """Dzień UTC znacznika `timestamp` w ISO; brak albo zły format → wspólny klucz ""."""
    raw = row.get("timestamp")
    if raw is None:
        return ""
    try:
        stamp = datetime.fromisoformat(str(raw))
    except ValueError:
        return ""
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(UTC)
    return stamp.date().isoformat()


def _is_sideways(row: Mapping[str, Any]) -> bool:
    trend = row.get("predicted_trend")
    return trend is not None and signal_of(str(trend)) == 0


def split_calibration_samples(rows: Sequence[Mapping[str, Any]]) -> CalibrationSamples:
    """Wiersze kalibracji → próbki kierunkowe (po cyklach) i SIDEWAYS.

    Wiersze bez `confidence_score` albo `is_trend_correct` są pomijane.
    """
    directional: list[tuple[str, Sample]] = []
    sideways: list[Sample] = []
    for row in rows:
        confidence = row.get("confidence_score")
        correct = row.get("is_trend_correct")
        if confidence is None or correct is None:
            continue
        sample = (float(confidence), bool(correct))
        if _is_sideways(row):
            sideways.append(sample)
        else:
            directional.append((_cycle_of(row), sample))
    return CalibrationSamples(
        directional_by_cycle=tuple(tuple(cycle) for cycle in group_by_cycle(directional)),
        sideways=tuple(sideways),
    )


def directional_calibration_curve(
    rows: Sequence[Mapping[str, Any]],
) -> list[CalibrationBucket] | None:
    """Krzywa kalibracji z samych predykcji kierunkowych; None bez takich próbek."""
    samples = split_calibration_samples(rows).directional
    return reliability_curve(samples) if samples else None
