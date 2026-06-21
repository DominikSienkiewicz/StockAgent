"""Render sekcji 🎯 Portfolio Radar (korelacja / koncentracja) w raporcie.

Czysta logika prezentacji — bierze PortfolioRiskReport z use case'a i
produkuje fragment HTML: werdykt o koncentracji + lista klastrów symboli,
które poruszają się razem ("43 symbole = 3 zakłady"). Sekcja pomijana w
całości, gdy brak klastrów (`render_*` → "").

Nazwy symboli i tekst werdyktu pochodzą z konfiguracji/obliczeń, ale i tak
escapujemy je przez `html.escape` — sekcja raportu musi być HTML-safe
niezależnie od źródła.
"""

from __future__ import annotations

import html as _html

from src.application.use_cases.portfolio_risk import PortfolioRiskReport


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

    if not subsections:
        return ""

    header = (
        "<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>"
        "🎯 Portfolio Radar — koncentracja i ryzyko</h2>"
    )
    return header + "".join(subsections)


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
    """Podsekcja VaR/CVaR (R1) — strata ogona portfela równoważonego."""
    if report.var is None:
        return ""
    conf_pct = round(report.var_confidence * 100)
    tail_pct = round((1.0 - report.var_confidence) * 100)
    var_pct = f"{report.var * 100:.1f}%"
    cvar_pct = f"{(report.cvar or 0.0) * 100:.1f}%"
    return (
        "<div style='padding: 10px 14px; background: #fffbeb; "
        "border-left: 3px solid #f59e0b; border-radius: 4px; "
        "margin-bottom: 12px; font-size: 13px;'>"
        f"<strong>📉 Value-at-Risk ({conf_pct}%):</strong> w najgorszych "
        f"{tail_pct}% okresów dzienna strata portfela równoważonego sięga "
        f"≥ <strong>{var_pct}</strong>; średnia strata w tym ogonie "
        f"(CVaR / expected shortfall) to <strong>{cvar_pct}</strong>."
        "</div>"
    )


def _render_stress_html(report: PortfolioRiskReport) -> str:
    """Podsekcja stress-test (R2) — wpływ scenariuszy szoku na portfel."""
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
            f"{impact.market_shock * 100:+.0f}%</td>"
            f"<td style='padding: 6px 8px; text-align: right; "
            f"color: {'#dc2626' if impact.portfolio_pnl < 0 else '#059669'};'>"
            f"{impact.portfolio_pnl * 100:+.1f}%</td>"
            f"<td style='padding: 6px 8px;'>{worst} "
            f"({impact.worst_pnl * 100:+.1f}%)</td>"
            "</tr>"
        )
    return (
        "<p style='font-size: 12px; color: #6b7280; margin: 12px 0 6px 0;'>"
        "🧪 Stress-test (β-to-portfel × szok rynku):</p>"
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 16px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th style='padding: 6px 8px; text-align: left;'>Scenariusz</th>"
        "<th style='padding: 6px 8px; text-align: right;'>Szok</th>"
        "<th style='padding: 6px 8px; text-align: right;'>Portfel</th>"
        "<th style='padding: 6px 8px; text-align: left;'>Najsłabszy</th>"
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
