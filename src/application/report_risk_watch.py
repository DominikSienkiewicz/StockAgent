"""Render sekcji 🚨 Risk Watch w raporcie e-mail.

Czysta logika prezentacji — bierze `MacroRiskReport` z use case'a i
produkuje fragment HTML oraz wariant plain text. Sekcja pomijana w
całości, gdy brak sygnałów i brak polskiego makro (`render_*` → "").
Instrumenty podpisane jak w reszcie maila: `company_label` (ticker + nazwa
z `COMPANY_NAMES`), nie goły ticker.
"""

from __future__ import annotations

import html as _html
from decimal import Decimal

from src.application.report_formatting import (
    PALETTE,
    company_label,
    number,
    number_exact,
    pct,
)
from src.application.reporting.section import ReportSection, SectionInput, text_block
from src.application.use_cases.monitor_macro_risk import MacroRiskReport
from src.domain.drawdown import DrawdownSignal
from src.domain.hedge import HedgeAssessment
from src.domain.macro_risk import (
    MacroAlertLevel,
    MacroRiskInstrumentType,
    MacroRiskSignal,
)
from src.domain.polish_macro import MacroStressLevel, PolishMacroSnapshot
from src.domain.value_objects import Ratio

# Domknięcie tabeli HTML — powtarzane w trzech podsekcjach Risk Watch.
_TABLE_CLOSE = "</tbody></table>"

# Progi drawdownu spójne z use case'em — render tylko prezentuje wynik
# domeny, ale poziom alertu liczy tą samą skalą (ELEVATED -10%, CRITICAL -20%).
_DRAWDOWN_ELEVATED_THRESHOLD = Ratio(Decimal("0.10"))
_DRAWDOWN_CRITICAL_THRESHOLD = Ratio(Decimal("0.20"))

_ALERT_LABEL = {
    MacroAlertLevel.NORMAL: "NORMAL",
    MacroAlertLevel.ELEVATED: "ELEVATED",
    MacroAlertLevel.CRITICAL: "CRITICAL",
}

_ALERT_COLOR = {
    MacroAlertLevel.NORMAL: PALETTE.positive,
    MacroAlertLevel.ELEVATED: PALETTE.caution,
    MacroAlertLevel.CRITICAL: PALETTE.negative,
}

_ALERT_BG = {
    MacroAlertLevel.NORMAL: "#f0fdf4",
    MacroAlertLevel.ELEVATED: "#fefce8",
    MacroAlertLevel.CRITICAL: PALETTE.tint_negative,
}

_INSTRUMENT_LABEL = {
    MacroRiskInstrumentType.INVERSE_EQUITY: "Short equity",
    MacroRiskInstrumentType.INVERSE_TREASURY: "Short Treasuries",
    MacroRiskInstrumentType.SAFE_HAVEN: "Safe haven",
    MacroRiskInstrumentType.VOLATILITY: "Volatility (VIX)",
    MacroRiskInstrumentType.SOVEREIGN_PROXY: "Sovereign proxy",
}

_STRESS_LABEL = {
    MacroStressLevel.NORMAL: "NORMAL",
    MacroStressLevel.ELEVATED: "ELEVATED",
    MacroStressLevel.CRITICAL: "CRITICAL",
}


def _format_pct(value: Decimal) -> str:
    """Ułamek jako procent ze znakiem, dwa miejsca po przecinku („+2,10%”, MAIL-58)."""
    return pct(value, signed=True, decimals=2)


def _drawdown_alert(dd: DrawdownSignal) -> MacroAlertLevel:
    return dd.evaluate_drawdown(
        elevated_threshold=_DRAWDOWN_ELEVATED_THRESHOLD,
        critical_threshold=_DRAWDOWN_CRITICAL_THRESHOLD,
    )


def render_risk_watch_html(report: MacroRiskReport) -> str:
    if (
        not report.signals
        and not report.drawdowns
        and not report.hedges
        and report.polish_macro is None
    ):
        return ""

    parts: list[str] = []
    overall_label = _ALERT_LABEL[report.overall_alert]
    overall_color = _ALERT_COLOR[report.overall_alert]
    overall_bg = _ALERT_BG[report.overall_alert]

    parts.append(
        "<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>🚨 Risk Watch — sygnały makro</h2>"
    )
    parts.append(
        f"<div style='padding: 10px 14px; background: {overall_bg}; "
        f"border-left: 3px solid {overall_color}; border-radius: 4px; "
        f"margin-bottom: 12px; font-size: 13px;'>"
        f"<strong>Ogólny poziom alertu:</strong> "
        f"<span style='color: {overall_color}; font-weight: 600;'>"
        f"{overall_label}</span>"
        f"</div>"
    )

    if report.signals:
        parts.append(_render_signals_table_html(report.signals))

    if report.drawdowns:
        parts.append(_render_drawdowns_table_html(report.drawdowns))

    if report.hedges:
        parts.append(_render_hedges_table_html(report.hedges))

    if report.polish_macro is not None:
        parts.append(_render_polish_macro_html(report.polish_macro))

    return "".join(parts)


# Kolor etykiety skuteczności hedge'a: zielony = skuteczny, bursztyn =
# nieskuteczny, czerwony = odwrócony (porusza się jak book, nie chroni).
_HEDGE_COLOR = {
    "skuteczny": PALETTE.positive,
    "nieskuteczny": PALETTE.caution,
    "odwrócony": PALETTE.negative,
}


def _render_hedges_table_html(hedges: list[HedgeAssessment]) -> str:
    """Tabela skuteczności hedge'y (R3) — realized corr instrumentu vs book."""
    rows: list[str] = []
    for h in hedges:
        color = _HEDGE_COLOR.get(h.label, "#6b7280")
        rows.append(
            "<tr>"
            "<td style='padding: 6px 8px;'><strong>"
            f"{_html.escape(company_label(h.hedge_symbol))}</strong></td>"
            f"<td style='padding: 6px 8px; text-align: right;'>"
            f"{number(h.correlation, decimals=2, signed=True)}</td>"
            f"<td style='padding: 6px 8px; text-align: right;'>"
            f"{number(h.effectiveness, decimals=2, signed=True)}</td>"
            f"<td style='padding: 6px 8px; color: {color}; font-weight: 600;'>"
            f"{_html.escape(h.label)}</td>"
            "</tr>"
        )
    return (
        "<p style='font-size: 12px; color: #6b7280; margin: 12px 0 6px 0;'>"
        "🛡️ Skuteczność hedge'y (realized corr vs watchlista):</p>"
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Hedge</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: right;'>Korelacja</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: right;'>Skuteczność</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Ocena</th>"
        "</tr></thead><tbody>" + "".join(rows) + _TABLE_CLOSE
    )


def _render_signals_table_html(signals: list[MacroRiskSignal]) -> str:
    rows: list[str] = []
    for sig in signals:
        alert = sig.evaluate_alert()
        color = _ALERT_COLOR[alert]
        rows.append(
            f"<tr>"
            f"<td style='padding: 6px 8px;'>"
            f"<strong>{_html.escape(company_label(sig.symbol))}</strong></td>"
            f"<td style='padding: 6px 8px; color: #6b7280;'>"
            f"{_INSTRUMENT_LABEL[sig.instrument_type]}</td>"
            f"<td style='padding: 6px 8px; text-align: right;'>"
            f"{_format_pct(sig.change_ratio)}</td>"
            f"<td style='padding: 6px 8px; color: {color}; font-weight: 600;'>"
            f"{_ALERT_LABEL[alert]}</td>"
            f"</tr>"
        )
    return (
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Ticker</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Typ</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: right;'>Δ vs poprz.</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Alert</th>"
        "</tr></thead><tbody>" + "".join(rows) + _TABLE_CLOSE
    )


def _render_drawdowns_table_html(drawdowns: list[DrawdownSignal]) -> str:
    rows: list[str] = []
    for dd in drawdowns:
        alert = _drawdown_alert(dd)
        color = _ALERT_COLOR[alert]
        bg = _ALERT_BG[alert]
        # Czerwone tło wiersza dla CRITICAL ("TSLA -18% od 30d szczytu").
        row_bg = bg if alert is MacroAlertLevel.CRITICAL else "transparent"
        rows.append(
            f"<tr style='background: {row_bg};'>"
            f"<td style='padding: 6px 8px;'>"
            f"<strong>{_html.escape(company_label(dd.symbol))}</strong></td>"
            f"<td style='padding: 6px 8px; text-align: right; "
            f"color: {color}; font-weight: 600;'>"
            f"{_format_pct(dd.drawdown_fraction)}</td>"
            f"<td style='padding: 6px 8px; text-align: right; color: #6b7280;'>"
            f"{number_exact(dd.peak_price)}</td>"
            f"<td style='padding: 6px 8px; color: {color}; font-weight: 600;'>"
            f"{_ALERT_LABEL[alert]}</td>"
            f"</tr>"
        )
    return (
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Ticker</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: right;'>"
        "Spadek od 30d szczytu</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: right;'>Szczyt</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Alert</th>"
        "</tr></thead><tbody>" + "".join(rows) + _TABLE_CLOSE
    )


def _fx_change_html(daily: Ratio | None, monthly: Ratio) -> str:
    """DAT-29: „-0,55% d/d, -2,10% 30d”; bez zmiany dziennej sama zmiana 30d
    (okno podaje wtedy nagłówek bloku, jak przed DAT-29)."""
    if daily is None:
        return _format_pct(monthly)
    return f"{_format_pct(daily)} d/d, {_format_pct(monthly)} 30d"


def _fx_change_text(daily: Ratio | None, monthly: Ratio) -> str:
    """DAT-29: „-0,55% d/d, -2,10% / 30d”; bez zmiany dziennej „-2,10% / 30d”."""
    month = f"{_format_pct(monthly)} / 30d"
    return month if daily is None else f"{_format_pct(daily)} d/d, {month}"


def _render_polish_macro_html(snap: PolishMacroSnapshot) -> str:
    stress = snap.evaluate_stress_level()
    label = _STRESS_LABEL[stress]
    # Z dzienną zmianą (DAT-29) okno stoi przy każdej liczbie, nie w nagłówku.
    daily = (snap.eur_pln_1d_change_ratio, snap.usd_pln_1d_change_ratio)
    window = ", 30d" if None in daily else ""
    return (
        "<div style='padding: 10px 14px; background: #f9fafb; "
        "border-radius: 4px; font-size: 13px; margin-bottom: 16px;'>"
        f"<strong>Polska — kursy referencyjne (NBP{window}):</strong><br>"
        f"EUR/PLN: <strong>{number_exact(snap.eur_pln)}</strong> "
        f"({_fx_change_html(snap.eur_pln_1d_change_ratio, snap.eur_pln_30d_change_ratio)}) · "
        f"USD/PLN: <strong>{number_exact(snap.usd_pln)}</strong> "
        f"({_fx_change_html(snap.usd_pln_1d_change_ratio, snap.usd_pln_30d_change_ratio)})<br>"
        f"Poziom stresu fiskalnego: <strong>{label}</strong>"
        "</div>"
    )


def render_risk_watch_text(report: MacroRiskReport) -> str:
    if (
        not report.signals
        and not report.drawdowns
        and not report.hedges
        and report.polish_macro is None
    ):
        return ""

    lines: list[str] = []
    lines.append("=== 🚨 Risk Watch — sygnały makro ===")
    lines.append(f"Ogólny alert: {_ALERT_LABEL[report.overall_alert]}")
    lines.append("")

    if report.signals:
        for sig in report.signals:
            alert = sig.evaluate_alert()
            lines.append(
                f"  {company_label(sig.symbol):40s} "
                f"{_INSTRUMENT_LABEL[sig.instrument_type]:18s} "
                f"Δ={_format_pct(sig.change_ratio):>7s} "
                f"[{_ALERT_LABEL[alert]}]"
            )
        lines.append("")

    if report.drawdowns:
        lines.append("-- Drawdown od 30d szczytu --")
        for dd in report.drawdowns:
            alert = _drawdown_alert(dd)
            lines.append(
                f"  {company_label(dd.symbol):40s} "
                f"{_format_pct(dd.drawdown_fraction):>8s} od szczytu "
                f"(szczyt {number_exact(dd.peak_price)}) "
                f"[{_ALERT_LABEL[alert]}]"
            )
        lines.append("")

    if report.hedges:
        lines.append("-- Skuteczność hedge'y (corr vs watchlista) --")
        for h in report.hedges:
            lines.append(
                f"  {company_label(h.hedge_symbol):40s} "
                f"corr={number(h.correlation, decimals=2, signed=True)} "
                f"skuteczność={number(h.effectiveness, decimals=2, signed=True)} "
                f"[{h.label}]"
            )
        lines.append("")

    if report.polish_macro is not None:
        snap = report.polish_macro
        lines.append(
            f"  EUR/PLN: {number_exact(snap.eur_pln)} "
            f"({_fx_change_text(snap.eur_pln_1d_change_ratio, snap.eur_pln_30d_change_ratio)})"
        )
        lines.append(
            f"  USD/PLN: {number_exact(snap.usd_pln)} "
            f"({_fx_change_text(snap.usd_pln_1d_change_ratio, snap.usd_pln_30d_change_ratio)})"
        )
        lines.append(f"  Stres fiskalny: {_STRESS_LABEL[snap.evaluate_stress_level()]}")
        lines.append("")

    return "\n".join(lines)


def _section_html(data: SectionInput) -> str:
    report = data.context.macro_risk_report
    return render_risk_watch_html(report) if report else ""


def _section_text(data: SectionInput) -> list[str]:
    report = data.context.macro_risk_report
    return text_block(render_risk_watch_text(report) if report else "")


RISK_WATCH_SECTION = ReportSection(
    key="risk_watch", render_html=_section_html, render_text=_section_text
)
