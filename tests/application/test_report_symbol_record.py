"""WIA-35: chip trafności per symbol przy predykcji i sygnale.

Decyzja dotyczy konkretnej spółki, więc obok globalnego hit-rate mail mówi, jak
model trafiał na TEJ spółce w oknie track recordu: „na NVDA: 4/7 (30 dni, 95% CI
…)” albo „za mało historii”. n i przedział są zawsze widoczne — mała próba nie
może udawać umiejętności.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_builder import ReportContext, SymbolResult, build_html_report
from src.application.report_models import ResolvedPrediction
from src.application.report_symbol_record import (
    SymbolTrackRecordView,
    symbol_chip_label,
    symbol_track_record_view,
)

_STARTED_AT = datetime(2026, 10, 22, 5, 30, tzinfo=UTC)


def _resolved(symbol: str, trend: str, correct: bool) -> ResolvedPrediction:
    return ResolvedPrediction(symbol=symbol, predicted_trend=trend, is_correct=correct)


def _nvda_history() -> list[ResolvedPrediction]:
    """NVDA: 4 trafione z 7 kierunkowych, plus SIDEWAYS poza mianownikiem."""
    calls = [_resolved("NVDA", "BULLISH", index < 4) for index in range(7)]
    return [*calls, _resolved("NVDA", "SIDEWAYS", True), _resolved("AAPL", "BULLISH", True)]


def _view() -> SymbolTrackRecordView:
    return symbol_track_record_view(_nvda_history(), window_days=30)


def _saved(symbol: str) -> SymbolResult:
    return SymbolResult(
        symbol=symbol,
        status="saved",
        delta=Decimal("0.031"),
        current_price=Decimal("100.00"),
        trend="BULLISH",
        target_price=Decimal("104.00"),
        confidence_score=0.8,
        reasoning="Popyt na GPU rośnie.",
        sentiment_score=0.4,
        sentiment_label="Bullish",
        news_volume=5,
    )


class TestChipLabel:
    def test_sufficient_history_shows_hits_window_and_interval(self) -> None:
        label = symbol_chip_label(_view(), "NVDA")

        assert label.startswith("Trafność kierunkowa na NVDA: 4/7 (30 dni, 95% CI ")

    def test_small_sample_says_there_is_not_enough_history_with_its_n(self) -> None:
        label = symbol_chip_label(_view(), "AAPL")

        assert "na AAPL: za mało historii" in label
        assert "n=1" in label

    def test_symbol_without_any_history_is_not_enough_history(self) -> None:
        assert "na MSFT: za mało historii (n=0" in symbol_chip_label(_view(), "MSFT")


class TestChipInReport:
    def test_reasoning_block_and_signal_row_carry_the_chip(self) -> None:
        context = ReportContext(symbol_track_record=_view())

        html, text = build_html_report([_saved("NVDA")], _STARTED_AT, 1.0, context)

        chip = "Trafność kierunkowa na NVDA: 4/7 (30 dni"
        reasoning = html[html.index("💡 Uzasadnienia") :]
        signals = html[html.index("🎯 Najsilniejsze sygnały") : html.index("💡 Uzasadnienia")]
        assert chip in reasoning
        assert chip in signals
        assert text.count(chip) == 2

    def test_without_per_symbol_history_the_report_has_no_chip(self) -> None:
        html, text = build_html_report([_saved("NVDA")], _STARTED_AT, 1.0, ReportContext())

        assert "Trafność kierunkowa na" not in html
        assert "Trafność kierunkowa na" not in text
