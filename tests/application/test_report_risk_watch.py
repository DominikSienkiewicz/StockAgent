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
from src.domain.drawdown import DrawdownSignal
from src.domain.macro_risk import (
    MacroAlertLevel,
    MacroRiskInstrumentType,
    MacroRiskSignal,
)
from src.domain.polish_macro import PolishMacroSnapshot
from src.domain.value_objects import Money


def _drawdown(symbol: str, peak: str, current: str) -> DrawdownSignal:
    return DrawdownSignal(
        symbol=symbol,
        current_price=Decimal(current),
        peak_price=Decimal(peak),
    )


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


class TestHedgeRendering:
    def test_html_renders_hedge_row(self) -> None:
        from src.domain.hedge import HedgeAssessment

        report = MacroRiskReport(
            hedges=[
                HedgeAssessment(
                    hedge_symbol="SH",
                    correlation=-0.92,
                    effectiveness=0.92,
                    label="skuteczny",
                )
            ],
        )
        html = render_risk_watch_html(report)
        # Sekcja renderuje się też bez sygnałów, gdy jest ocena hedge'a.
        assert "Risk Watch" in html
        assert "SH" in html
        assert "skuteczny" in html

    def test_html_escapes_hedge_symbol(self) -> None:
        from src.domain.hedge import HedgeAssessment

        report = MacroRiskReport(
            hedges=[
                HedgeAssessment(
                    hedge_symbol="<b>x</b>",
                    correlation=0.5,
                    effectiveness=-0.5,
                    label="odwrócony",
                )
            ],
        )
        html = render_risk_watch_html(report)
        assert "<b>x</b>" not in html
        assert "&lt;b&gt;" in html

    def test_text_lists_hedges(self) -> None:
        from src.domain.hedge import HedgeAssessment

        report = MacroRiskReport(
            hedges=[
                HedgeAssessment(
                    hedge_symbol="SH",
                    correlation=-0.92,
                    effectiveness=0.92,
                    label="skuteczny",
                )
            ],
        )
        text = render_risk_watch_text(report)
        assert "SH" in text
        assert "skuteczny" in text

    def test_empty_hedges_render_nothing_extra(self) -> None:
        report = MacroRiskReport(hedges=[])
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
        assert "4,45" in html

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
        assert "4,45" in text


class TestDrawdownRendering:
    def test_html_renders_critical_drawdown_row(self) -> None:
        # TSLA: szczyt 100 → 82 = -18% → CRITICAL (> 20%? nie; ale >10% → ELEVATED).
        # Bierzemy -25%, by trafić CRITICAL i czerwony kolor.
        report = MacroRiskReport(
            drawdowns=[_drawdown("TSLA", peak="100", current="75")],
            overall_alert=MacroAlertLevel.CRITICAL,
        )
        html = render_risk_watch_html(report)
        assert "TSLA" in html
        assert "-25" in html  # -25% od szczytu
        # Czerwony kolor wiersza CRITICAL.
        assert "#dc2626" in html

    def test_peak_price_is_written_with_a_decimal_comma(self) -> None:
        # MAIL-58: szczyt to liczba dla czytelnika — „186,00”, nie „186.00”.
        report = MacroRiskReport(
            drawdowns=[_drawdown("AMD", peak="186.00", current="160.00")],
            overall_alert=MacroAlertLevel.ELEVATED,
        )

        assert "186,00" in render_risk_watch_html(report)
        assert "(szczyt 186,00)" in render_risk_watch_text(report)

    def test_html_drawdown_present_even_without_signals(self) -> None:
        report = MacroRiskReport(
            drawdowns=[_drawdown("TSLA", peak="200", current="100")],
            overall_alert=MacroAlertLevel.CRITICAL,
        )
        html = render_risk_watch_html(report)
        assert html != ""
        assert "TSLA" in html

    def test_html_escapes_symbol(self) -> None:
        report = MacroRiskReport(
            drawdowns=[_drawdown("<b>X", peak="100", current="80")],
            overall_alert=MacroAlertLevel.CRITICAL,
        )
        html = render_risk_watch_html(report)
        assert "<b>X" not in html
        assert "&lt;b&gt;X" in html

    def test_text_lists_drawdowns(self) -> None:
        report = MacroRiskReport(
            drawdowns=[_drawdown("TSLA", peak="100", current="82")],
            overall_alert=MacroAlertLevel.ELEVATED,
        )
        text = render_risk_watch_text(report)
        assert "TSLA" in text
        assert "-18" in text

    def test_empty_drawdowns_render_nothing_extra(self) -> None:
        # Brak sygnałów, brak makro, brak drawdownów → sekcja pusta.
        report = MacroRiskReport()
        assert render_risk_watch_html(report) == ""
        assert render_risk_watch_text(report) == ""


class TestCompanyNames:
    """Risk Watch podpisuje instrumenty nazwą z `COMPANY_NAMES`, jak reszta maila —
    goły ticker („PSQ”, „TBT”) nic nie mówi czytelnikowi."""

    @staticmethod
    def _report() -> MacroRiskReport:
        from src.domain.hedge import HedgeAssessment

        return MacroRiskReport(
            signals=[_signal("PSQ", MacroRiskInstrumentType.INVERSE_EQUITY, "100", "104")],
            drawdowns=[_drawdown("TSLA", peak="100", current="82")],
            hedges=[
                HedgeAssessment(
                    hedge_symbol="GLD",
                    correlation=-0.4,
                    effectiveness=0.4,
                    label="skuteczny",
                )
            ],
            overall_alert=MacroAlertLevel.ELEVATED,
        )

    def test_html_labels_signals_drawdowns_and_hedges_with_names(self) -> None:
        html = render_risk_watch_html(self._report())

        assert "PSQ (ProShares Short QQQ)" in html
        assert "TSLA (Tesla Inc.)" in html
        assert "GLD (SPDR Gold Shares)" in html

    def test_text_labels_signals_drawdowns_and_hedges_with_names(self) -> None:
        text = render_risk_watch_text(self._report())

        assert "PSQ (ProShares Short QQQ)" in text
        assert "TSLA (Tesla Inc.)" in text
        assert "GLD (SPDR Gold Shares)" in text

    def test_html_escapes_the_company_name(self) -> None:
        # „S&P” w nazwie SH musi przejść przez html.escape jak każda inna treść.
        report = MacroRiskReport(
            signals=[_signal("SH", MacroRiskInstrumentType.INVERSE_EQUITY, "100", "104")],
            overall_alert=MacroAlertLevel.ELEVATED,
        )

        html = render_risk_watch_html(report)

        assert "SH (ProShares Short S&amp;P 500)" in html
        assert "S&P 500" not in html
