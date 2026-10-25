"""Chip trafności per symbol w mailu (WIA-35).

Mapuje rozliczone predykcje okna track recordu (`ResolvedPrediction`) na rekordy
domeny (`symbol_track_record`) i składa jedną linię do bloku uzasadnienia i do
wiersza sygnału: „Trafność kierunkowa na NVDA: 4/7 (30 dni, 95% CI 25–84%)”
albo „… za mało historii (n=3 < 5, 30 dni)”. Render-only, zero płatnych wywołań:
dane to ten sam darmowy odczyt `get_resolved_predictions`, co Track Record.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from src.application.report_evidence import interval_label
from src.application.report_models import ResolvedPrediction
from src.domain.symbol_track_record import (
    MIN_SYMBOL_SAMPLE,
    SymbolTrackRecord,
    symbol_track_record,
)


@dataclass(frozen=True)
class SymbolTrackRecordView:
    """Rekordy per symbol i okno (dni), z którego je policzono (pole `ReportContext`)."""

    records: Mapping[str, SymbolTrackRecord]
    window_days: int


def symbol_track_record_view(
    resolved: Sequence[ResolvedPrediction], window_days: int
) -> SymbolTrackRecordView:
    """Rekordy trafności per symbol z rozliczonych predykcji okna."""
    records = symbol_track_record((p.symbol, p.predicted_trend, p.is_correct) for p in resolved)
    return SymbolTrackRecordView(records=records, window_days=window_days)


def symbol_chip_label(view: SymbolTrackRecordView, symbol: str) -> str:
    """Linia chipu; symbol bez historii albo z małą próbą → „za mało historii”."""
    record = view.records.get(symbol)
    days = view.window_days
    if record is None or record.insufficient:
        n = 0 if record is None else record.directional
        minimum = MIN_SYMBOL_SAMPLE if record is None else record.min_sample
        return f"Trafność kierunkowa na {symbol}: za mało historii (n={n} < {minimum}, {days} dni)"
    low, high = record.interval
    return (
        f"Trafność kierunkowa na {symbol}: {record.hits}/{record.directional} "
        f"({days} dni, {interval_label(low, high)})"
    )


def symbol_chip_labels(
    view: SymbolTrackRecordView | None, symbols: Iterable[str]
) -> dict[str, str]:
    """Chipy dla symboli raportu; bez widoku (flaga off, błąd odczytu) — brak chipów."""
    if view is None:
        return {}
    return {symbol: symbol_chip_label(view, symbol) for symbol in symbols}
