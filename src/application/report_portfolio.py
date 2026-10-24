"""Render sekcji 🎯 Portfolio Radar (korelacja / koncentracja) w raporcie.

Czysta logika prezentacji — bierze PortfolioRiskReport z use case'a i
produkuje fragment HTML: werdykt o koncentracji + lista klastrów symboli,
które poruszają się razem ("43 symbole = 3 zakłady"). Sekcja pomijana w
całości, gdy nie ma czego pokazać (`render_*` → "").

PORT-22: statystyka pominięta z braku historii (`PortfolioRiskReport.insufficient`)
renderuje się jawnie jako „za mało historii (n/wymagane)” — pustka albo liczba
z kilku obserwacji byłyby gorsze niż ta informacja.

Nazwy symboli i tekst werdyktu pochodzą z konfiguracji/obliczeń, ale i tak
escapujemy je przez `html.escape` — sekcja raportu musi być HTML-safe
niezależnie od źródła.
"""

from __future__ import annotations

import html as _html

from src.application.report_formatting import PALETTE, pct
from src.application.reporting.section import ReportSection, SectionInput, text_block
from src.application.use_cases.portfolio_risk import PortfolioRiskReport
from src.domain.risk_sample import RiskStat

_HEADING = "🎯 Portfolio Radar — koncentracja i ryzyko"

# Nazwy statystyk w adnotacji „za mało historii”.
_STAT_LABELS: dict[RiskStat, str] = {
    "correlation": "Klastry korelacji",
    "var": "Value-at-Risk",
    "beta": "Stress-test (β względem watchlisty)",
}


def _insufficient_notes(report: PortfolioRiskReport) -> list[str]:
    """Adnotacje „<statystyka>: za mało historii (n/wymagane)” w kolejności raportu."""
    return [
        f"{_STAT_LABELS[stat]}: za mało historii ({n}/{required})"
        for stat, n, required in report.insufficient
    ]


def render_correlation_html(report: PortfolioRiskReport) -> str:
    """Fragment HTML sekcji Portfolio Radar (koncentracja + ryzyko).

    Składa się z niezależnych podsekcji: klastry korelacji (jeśli są), VaR/CVaR
    (R1, jeśli policzone) i stress-test (R2, jeśli są scenariusze). Pusty string,
    gdy ŻADNA podsekcja nie ma czego pokazać (sekcja chowa się w całości)."""
    subsections: list[str] = []
    if report.clusters:
        subsections.append(_render_clusters_block_html(report))
    if report.var is not None:
        subsections.append(_render_var_html(report))
    if report.scenarios:
        subsections.append(_render_stress_html(report))
    if report.insufficient:
        subsections.append(_render_insufficient_html(report))

    if not subsections:
        return ""

    header = (
        "<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>"
        f"{_HEADING}</h2>"
    )
    return header + "".join(subsections)


def _render_insufficient_html(report: PortfolioRiskReport) -> str:
    """PORT-22: lista statystyk pominiętych z braku historii."""
    items = "".join(
        f"<li style='padding: 2px 0;'>{_html.escape(note)}</li>"
        for note in _insufficient_notes(report)
    )
    return (
        "<div style='padding: 10px 14px; background: #f9fafb; "
        "border-left: 3px solid #9ca3af; border-radius: 4px; "
        "margin-bottom: 12px; font-size: 13px; color: #4b5563;'>"
        "⏳ Liczby z kilku obserwacji udawałyby wiedzę — pokażemy je, "
        "gdy zbierze się historia:"
        f"<ul style='margin: 4px 0 0 18px; padding: 0;'>{items}</ul>"
        "</div>"
    )


def render_correlation_text(report: PortfolioRiskReport) -> str:
    """Wariant plain-text sekcji Portfolio Radar; "" gdy nie ma czego pokazać."""
    lines: list[str] = []
    if report.clusters:
        if report.verdict:
            lines.append(f"  Werdykt: {report.verdict}")
        for idx, cluster in enumerate(report.clusters, start=1):
            lines.append(
                f"  Klaster {idx} ({len(cluster)} symboli): {' · '.join(cluster)}"
            )
    if report.var is not None:
        conf_pct = round(report.var_confidence * 100)
        lines.append(
            f"  📉 Value-at-Risk ({conf_pct}%): {pct(report.var, decimals=1)} "
            f"(CVaR {pct(report.cvar or 0.0, decimals=1)})"
        )
    for impact in report.scenarios:
        worst = impact.worst_symbol if impact.worst_symbol is not None else "—"
        lines.append(
            f"  🧪 {impact.scenario_name} ({pct(impact.market_shock, signed=True, decimals=0)}): "
            f"watchlista {pct(impact.portfolio_pnl, signed=True, decimals=1)}, najsłabszy {worst} "
            f"({pct(impact.worst_pnl, signed=True, decimals=1)})"
        )
    lines.extend(f"  ⏳ {note}" for note in _insufficient_notes(report))
    if not lines:
        return ""
    return "\n".join([f"=== {_HEADING.upper()} ===", *lines, ""])


def _render_clusters_block_html(report: PortfolioRiskReport) -> str:
    """Podsekcja klastrów korelacji (werdykt + lista). Bez własnego nagłówka."""
    parts: list[str] = []
    if report.verdict:
        parts.append(
            "<div style='padding: 10px 14px; background: #fef2f2; "
            "border-left: 3px solid #dc2626; border-radius: 4px; "
            "margin-bottom: 12px; font-size: 13px;'>"
            "<strong>Werdykt:</strong> "
            f"{_html.escape(report.verdict)}"
            "</div>"
        )
    threshold_pct = round(report.threshold * 100)
    parts.append(
        "<p style='font-size: 12px; color: #6b7280; margin: 0 0 8px 0;'>"
        f"Klastry symboli o korelacji zwrotów ≥ {threshold_pct}% "
        f"(z {report.watchlist_size} instrumentów z historią):"
        "</p>"
    )
    parts.append(_render_clusters_html(report.clusters))
    return "".join(parts)


def _render_var_html(report: PortfolioRiskReport) -> str:
    """Podsekcja VaR/CVaR (R1) — strata ogona watchlisty o równych wagach (MAIL-31)."""
    if report.var is None:
        return ""
    conf_pct = round(report.var_confidence * 100)
    tail_pct = round((1.0 - report.var_confidence) * 100)
    var_pct = pct(report.var, decimals=1)
    cvar_pct = pct(report.cvar or 0.0, decimals=1)
    return (
        "<div style='padding: 10px 14px; background: #fffbeb; "
        "border-left: 3px solid #f59e0b; border-radius: 4px; "
        "margin-bottom: 12px; font-size: 13px;'>"
        f"<strong>📉 Value-at-Risk ({conf_pct}%):</strong> w najgorszych "
        f"{tail_pct}% okresów dzienna strata watchlisty (równe wagi) sięga "
        f"≥ <strong>{var_pct}</strong>; średnia strata w tym ogonie "
        f"(CVaR / expected shortfall) to <strong>{cvar_pct}</strong>."
        "</div>"
    )


def _render_stress_html(report: PortfolioRiskReport) -> str:
    """Podsekcja stress-test (R2) — wpływ scenariuszy szoku na watchlistę."""
    if not report.scenarios:
        return ""
    rows: list[str] = []
    for impact in report.scenarios:
        worst = (
            _html.escape(impact.worst_symbol)
            if impact.worst_symbol is not None
            else "—"
        )
        rows.append(
            "<tr>"
            "<td style='padding: 6px 8px;'><strong>"
            f"{_html.escape(impact.scenario_name)}</strong></td>"
            f"<td style='padding: 6px 8px; text-align: right; color: #6b7280;'>"
            f"{pct(impact.market_shock, signed=True, decimals=0)}</td>"
            f"<td style='padding: 6px 8px; text-align: right; "
            f"color: {PALETTE.negative if impact.portfolio_pnl < 0 else PALETTE.positive};'>"
            f"{pct(impact.portfolio_pnl, signed=True, decimals=1)}</td>"
            f"<td style='padding: 6px 8px;'>{worst} "
            f"({pct(impact.worst_pnl, signed=True, decimals=1)})</td>"
            "</tr>"
        )
    return (
        "<p style='font-size: 12px; color: #6b7280; margin: 12px 0 6px 0;'>"
        "🧪 Stress-test (β względem watchlisty × szok rynku):</p>"
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Scenariusz</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: right;'>Szok</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: right;'>Watchlista</th>"
        "<th scope='col' style='padding: 6px 8px; text-align: left;'>Najsłabszy</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_clusters_html(clusters: list[tuple[str, ...]]) -> str:
    """Lista klastrów — każdy jako wiersz z symbolami poruszającymi się razem."""
    rows: list[str] = []
    for idx, cluster in enumerate(clusters, start=1):
        members = " · ".join(_html.escape(sym) for sym in cluster)
        rows.append(
            "<li style='padding: 6px 0; border-bottom: 1px solid #f3f4f6; "
            "font-size: 13px;'>"
            f"<span style='color: #6b7280;'>Klaster {idx} "
            f"({len(cluster)} symboli):</span> "
            f"<strong>{members}</strong>"
            "</li>"
        )
    return (
        "<ul style='list-style: none; padding: 0; margin: 0 0 16px 0;'>"
        + "".join(rows)
        + "</ul>"
    )


def _section_html(data: SectionInput) -> str:
    report = data.context.portfolio_risk_report
    return render_correlation_html(report) if report else ""


def _section_text(data: SectionInput) -> list[str]:
    report = data.context.portfolio_risk_report
    return text_block(render_correlation_text(report) if report else "")


# Q4/PORT-22: koncentracja i ryzyko watchlisty (klastry, VaR, stress test).
PORTFOLIO_RADAR_SECTION = ReportSection(
    key="portfolio_radar",
    render_html=_section_html,
    render_text=_section_text,
    context_fields=frozenset({"portfolio_risk_report"}),
)
