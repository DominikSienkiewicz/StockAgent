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
    """Fragment HTML sekcji Portfolio Radar. Pusty string, gdy brak klastrów."""
    if not report.clusters:
        return ""

    parts: list[str] = []
    parts.append(
        "<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>"
        "🎯 Portfolio Radar — korelacja i koncentracja</h2>"
    )

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
