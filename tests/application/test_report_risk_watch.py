"""Testy renderera sekcji Risk Watch (HTML + plain text).

Czysta logika prezentacji — bez I/O. Bierze MacroRiskReport z domeny
(application use case) i produkuje fragmenty do wstawienia w mailu.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_risk_watch import (
    render_risk_watch_html,
    render_risk_watch_text,
)
from src.application.use_cases.monitor_macro_risk import MacroRiskReport
from src.domain.macro_risk import (
    MacroAlertLevel,
    MacroRiskInstrumentType,
    MacroRiskSignal,
)
from src.domain.polish_macro import PolishMacroSnapshot
from src.domain.value_objects import Money


def _signal(
    symbol: str,
    instrument: MacroRiskInstrumentType,
    prev: str,
    cur: str,
) -> MacroRiskSignal:
    return MacroRiskSignal(
        symbol=symbol,
        instrument_type=instrument,
        current_price=Money(Decimal(cur)),
        previous_price=Money(Decimal(prev)),
    )


class TestEmptyReport:
    def test_empty_report_returns_empty_strings(self) -> None:
        report = MacroRiskReport()
        assert render_risk_watch_html(report) == ""
        assert render_risk_watch_text(report) == ""


class TestHtmlRendering:
    def test_renders_section_header(self) -> None:
        report = MacroRiskReport(
            signals=[_signal("SH", MacroRiskInstrumentType.INVERSE_EQUITY, "100", "104")],
            overall_alert=MacroAlertLevel.CRITICAL,
        )
        html = render_risk_watch_html(report)
        assert "Risk Watch" in html

    def test_includes_each_signal_symbol_and_change(self) -> None:
        report = MacroRiskReport(
            signals=[
                _signal("SH", MacroRiskInstrumentType.INVERSE_EQUITY, "100", "104"),
                _signal("EPOL", MacroRiskInstrumentType.SOVEREIGN_PROXY, "20", "19.2"),
            ],
            overall_alert=MacroAlertLevel.CRITICAL,
        )
        html = render_risk_watch_html(report)
        assert "SH" in html
        assert "EPOL" in html
        assert "+4" in html  # +4% change
        assert "-4" in html  # EPOL -4%

    def test_includes_alert_level_label(self) -> None:
        report = MacroRiskReport(
            signals=[_signal("VIXY", MacroRiskInstrumentType.VOLATILITY, "20", "21")],
            overall_alert=MacroAlertLevel.ELEVATED,
        )
        html = render_risk_watch_html(report)
        assert "ELEVATED" in html or "Podwyższony" in html

    def test_includes_polish_macro_when_present(self) -> None:
        snap = PolishMacroSnapshot(
            eur_pln=Decimal("4.45"),
            usd_pln=Decimal("4.10"),
            eur_pln_30d_change_pct=Decimal("0.035"),
            usd_pln_30d_change_pct=Decimal("0.02"),
            fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
        )
        report = MacroRiskReport(polish_macro=snap, overall_alert=MacroAlertLevel.ELEVATED)
        html = render_risk_watch_html(report)
        assert "EUR/PLN" in html
        assert "USD/PLN" in html
        assert "4.45" in html

    def test_skips_polish_macro_block_when_none(self) -> None:
        report = MacroRiskReport(
            signals=[_signal("SH", MacroRiskInstrumentType.INVERSE_EQUITY, "100", "104")],
            polish_macro=None,
            overall_alert=MacroAlertLevel.CRITICAL,
        )
        html = render_risk_watch_html(report)
        assert "EUR/PLN" not in html


class TestPlainTextRendering:
    def test_text_lists_signals(self) -> None:
        report = MacroRiskReport(
            signals=[
                _signal("SH", MacroRiskInstrumentType.INVERSE_EQUITY, "100", "104"),
                _signal("GLD", MacroRiskInstrumentType.SAFE_HAVEN, "180", "183"),
            ],
            overall_alert=MacroAlertLevel.CRITICAL,
        )
        text = render_risk_watch_text(report)
        assert "SH" in text
        assert "GLD" in text
        assert "Risk Watch" in text

    def test_text_includes_polish_macro(self) -> None:
        snap = PolishMacroSnapshot(
            eur_pln=Decimal("4.45"),
            usd_pln=Decimal("4.10"),
            eur_pln_30d_change_pct=Decimal("0.035"),
            usd_pln_30d_change_pct=Decimal("0.02"),
            fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
        )
        report = MacroRiskReport(polish_macro=snap, overall_alert=MacroAlertLevel.ELEVATED)
        text = render_risk_watch_text(report)
        assert "EUR/PLN" in text
        assert "4.45" in text
