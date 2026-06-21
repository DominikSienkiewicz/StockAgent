"""Render sekcji 🚨 Risk Watch w raporcie e-mail.

Czysta logika prezentacji — bierze `MacroRiskReport` z use case'a i
produkuje fragment HTML oraz wariant plain text. Sekcja pomijana w
całości, gdy brak sygnałów i brak polskiego makro (`render_*` → "").
"""

from __future__ import annotations

import html as _html
from decimal import Decimal

from src.application.use_cases.monitor_macro_risk import MacroRiskReport
from src.domain.drawdown import DrawdownSignal
from src.domain.hedge import HedgeAssessment
from src.domain.macro_risk import (
    MacroAlertLevel,
    MacroRiskInstrumentType,
    MacroRiskSignal,
)
from src.domain.polish_macro import MacroStressLevel, PolishMacroSnapshot

# Progi drawdownu spójne z use case'em — render tylko prezentuje wynik
# domeny, ale poziom alertu liczy tą samą skalą (ELEVATED -10%, CRITICAL -20%).
_DRAWDOWN_ELEVATED_PCT = Decimal("0.10")
_DRAWDOWN_CRITICAL_PCT = Decimal("0.20")

_ALERT_LABEL = {
    MacroAlertLevel.NORMAL: "NORMAL",
    MacroAlertLevel.ELEVATED: "ELEVATED",
    MacroAlertLevel.CRITICAL: "CRITICAL",
}

_ALERT_COLOR = {
    MacroAlertLevel.NORMAL: "#16a34a",
    MacroAlertLevel.ELEVATED: "#ca8a04",
    MacroAlertLevel.CRITICAL: "#dc2626",
}

_ALERT_BG = {
    MacroAlertLevel.NORMAL: "#f0fdf4",
    MacroAlertLevel.ELEVATED: "#fefce8",
    MacroAlertLevel.CRITICAL: "#fef2f2",
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
    pct = (value * Decimal("100")).quantize(Decimal("0.01"))
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct}%"


def _drawdown_alert(dd: DrawdownSignal) -> MacroAlertLevel:
    return dd.evaluate_drawdown(
        elevated_pct=_DRAWDOWN_ELEVATED_PCT,
        critical_pct=_DRAWDOWN_CRITICAL_PCT,
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
        "<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>"
        "🚨 Risk Watch — sygnały makro</h2>"
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
    "skuteczny": "#16a34a",
    "nieskuteczny": "#ca8a04",
    "odwrócony": "#dc2626",
}


def _render_hedges_table_html(hedges: list[HedgeAssessment]) -> str:
    """Tabela skuteczności hedge'y (R3) — realized corr instrumentu vs book."""
    rows: list[str] = []
    for h in hedges:
        color = _HEDGE_COLOR.get(h.label, "#6b7280")
        rows.append(
            "<tr>"
            "<td style='padding: 6px 8px;'><strong>"
            f"{_html.escape(h.hedge_symbol)}</strong></td>"
            f"<td style='padding: 6px 8px; text-align: right;'>"
            f"{h.correlation:+.2f}</td>"
            f"<td style='padding: 6px 8px; text-align: right;'>"
            f"{h.effectiveness:+.2f}</td>"
            f"<td style='padding: 6px 8px; color: {color}; font-weight: 600;'>"
            f"{_html.escape(h.label)}</td>"
            "</tr>"
        )
    return (
        "<p style='font-size: 12px; color: #6b7280; margin: 12px 0 6px 0;'>"
        "🛡️ Skuteczność hedge'y (realized corr vs portfel):</p>"
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th style='padding: 6px 8px; text-align: left;'>Hedge</th>"
        "<th style='padding: 6px 8px; text-align: right;'>Korelacja</th>"
        "<th style='padding: 6px 8px; text-align: right;'>Skuteczność</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Ocena</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_signals_table_html(signals: list[MacroRiskSignal]) -> str:
    rows: list[str] = []
    for sig in signals:
        alert = sig.evaluate_alert()
        color = _ALERT_COLOR[alert]
        rows.append(
            f"<tr>"
            f"<td style='padding: 6px 8px;'><strong>{_html.escape(sig.symbol)}</strong></td>"
            f"<td style='padding: 6px 8px; color: #6b7280;'>"
            f"{_INSTRUMENT_LABEL[sig.instrument_type]}</td>"
            f"<td style='padding: 6px 8px; text-align: right;'>"
            f"{_format_pct(sig.change_pct)}</td>"
            f"<td style='padding: 6px 8px; color: {color}; font-weight: 600;'>"
            f"{_ALERT_LABEL[alert]}</td>"
            f"</tr>"
        )
    return (
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th style='padding: 6px 8px; text-align: left;'>Ticker</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Typ</th>"
        "<th style='padding: 6px 8px; text-align: right;'>Δ vs poprz.</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Alert</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
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
            f"<strong>{_html.escape(dd.symbol)}</strong></td>"
            f"<td style='padding: 6px 8px; text-align: right; "
            f"color: {color}; font-weight: 600;'>"
            f"{_format_pct(dd.drawdown_fraction)}</td>"
            f"<td style='padding: 6px 8px; text-align: right; color: #6b7280;'>"
            f"{dd.peak_price}</td>"
            f"<td style='padding: 6px 8px; color: {color}; font-weight: 600;'>"
            f"{_ALERT_LABEL[alert]}</td>"
            f"</tr>"
        )
    return (
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th style='padding: 6px 8px; text-align: left;'>Ticker</th>"
        "<th style='padding: 6px 8px; text-align: right;'>"
        "Spadek od 30d szczytu</th>"
        "<th style='padding: 6px 8px; text-align: right;'>Szczyt</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Alert</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_polish_macro_html(snap: PolishMacroSnapshot) -> str:
    stress = snap.evaluate_stress_level()
    label = _STRESS_LABEL[stress]
    return (
        "<div style='padding: 10px 14px; background: #f9fafb; "
        "border-radius: 4px; font-size: 13px; margin-bottom: 16px;'>"
        "<strong>Polska — kursy referencyjne (NBP, 30d):</strong><br>"
        f"EUR/PLN: <strong>{snap.eur_pln}</strong> "
        f"({_format_pct(snap.eur_pln_30d_change_pct)}) · "
        f"USD/PLN: <strong>{snap.usd_pln}</strong> "
        f"({_format_pct(snap.usd_pln_30d_change_pct)})<br>"
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
    lines.append(
        f"Ogólny alert: {_ALERT_LABEL[report.overall_alert]}"
    )
    lines.append("")

    if report.signals:
        for sig in report.signals:
            alert = sig.evaluate_alert()
            lines.append(
                f"  {sig.symbol:6s} "
                f"{_INSTRUMENT_LABEL[sig.instrument_type]:18s} "
                f"Δ={_format_pct(sig.change_pct):>7s} "
                f"[{_ALERT_LABEL[alert]}]"
            )
        lines.append("")

    if report.drawdowns:
        lines.append("-- Drawdown od 30d szczytu --")
        for dd in report.drawdowns:
            alert = _drawdown_alert(dd)
            lines.append(
                f"  {dd.symbol:6s} "
                f"{_format_pct(dd.drawdown_fraction):>8s} od szczytu "
                f"(szczyt {dd.peak_price}) "
                f"[{_ALERT_LABEL[alert]}]"
            )
        lines.append("")

    if report.hedges:
        lines.append("-- Skuteczność hedge'y (corr vs portfel) --")
        for h in report.hedges:
            lines.append(
                f"  {h.hedge_symbol:6s} "
                f"corr={h.correlation:+.2f} "
                f"skuteczność={h.effectiveness:+.2f} "
                f"[{h.label}]"
            )
        lines.append("")

    if report.polish_macro is not None:
        snap = report.polish_macro
        lines.append(
            f"  EUR/PLN: {snap.eur_pln} "
            f"({_format_pct(snap.eur_pln_30d_change_pct)} / 30d)"
        )
        lines.append(
            f"  USD/PLN: {snap.usd_pln} "
            f"({_format_pct(snap.usd_pln_30d_change_pct)} / 30d)"
        )
        lines.append(
            f"  Stres fiskalny: {_STRESS_LABEL[snap.evaluate_stress_level()]}"
        )
        lines.append("")

    return "\n".join(lines)
