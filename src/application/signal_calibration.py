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

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.domain.calibration_curve import CalibrationBucket, reliability_curve
from src.domain.equity_curve import signal_of
from src.domain.isotonic_calibration import CalibrationClass, ClassedSample
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


def timed_directional_samples(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[datetime, float, bool]]:
    """ML-28: próbki kierunkowe z chwilą predykcji `(timestamp, pewność, trafienie)`
    — okna oceny poza próbą tną po czasie. SIDEWAYS (WIA-39) i wiersze bez
    pewności, werdyktu albo poprawnego znacznika czasu odpadają."""
    return [sample for row in rows if (sample := _timed_directional(row)) is not None]


def timed_classed_directional_samples(
    rows: Sequence[Mapping[str, Any]], crypto_symbols: Collection[str]
) -> list[ClassedSample]:
    """ML-54: próbki `timed_directional_samples` z klasą aktywa z `symbol` wiersza.

    Krypto = symbol z `crypto_symbols` (reguła `run_cycle.classify_asset`, ta sama,
    która nadaje `asset_class` wynikom cyklu), reszta = akcje i ETF-y. Wiersz bez
    symbolu nie ma klasy — liczy się tylko do mapy globalnej."""
    crypto = frozenset(crypto_symbols)
    samples: list[ClassedSample] = []
    for row in rows:
        sample = _timed_directional(row)
        if sample is None:
            continue
        stamp, confidence, correct = sample
        samples.append((stamp, confidence, correct, _calibration_class(row, crypto)))
    return samples


def _timed_directional(row: Mapping[str, Any]) -> tuple[datetime, float, bool] | None:
    """Próbka kierunkowa wiersza z chwilą predykcji; SIDEWAYS i braki → None."""
    confidence = row.get("confidence_score")
    correct = row.get("is_trend_correct")
    stamp = _moment_of(row)
    if confidence is None or correct is None or stamp is None or _is_sideways(row):
        return None
    return stamp, float(confidence), bool(correct)


def _calibration_class(row: Mapping[str, Any], crypto: frozenset[str]) -> CalibrationClass | None:
    symbol = row.get("symbol")
    if symbol is None:
        return None
    return CalibrationClass.CRYPTO if symbol in crypto else CalibrationClass.NON_CRYPTO


def _moment_of(row: Mapping[str, Any]) -> datetime | None:
    """Znacznik `timestamp` jako chwila UTC; brak albo zły format → None."""
    raw = row.get("timestamp")
    if raw is None:
        return None
    try:
        stamp = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def directional_calibration_curve(
    rows: Sequence[Mapping[str, Any]],
) -> list[CalibrationBucket] | None:
    """Krzywa kalibracji z samych predykcji kierunkowych; None bez takich próbek."""
    samples = split_calibration_samples(rows).directional
    return reliability_curve(samples) if samples else None
