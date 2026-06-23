"""Render sekcji 📈 Track Record w raporcie e-mail.

Trzy podsekcje track-recordu agenta, render-only (zero płatnych wywołań):
  - T1: krzywa kapitału (paper-PnL ze zrealizowanych predykcji) — podsumowanie
    + zwięzła sparklinia tabelaryczna (email-safe, bez zewnętrznych obrazków),
  - T2: krzywa kalibracji (reliability diagram) — bucket pewności → realny
    hit-rate + ECE,
  - T4: panel „lessons learned" — błędne predykcje z diagnozą i motywem.

Każda podsekcja chowa się, gdy nie ma czego pokazać; cała sekcja znika, gdy
wszystkie trzy są puste (`render_* → ""`)."""

from __future__ import annotations

import html as _html

from src.application.report_lessons import LessonsReport
from src.domain.calibration_curve import (
    CalibrationBucket,
    expected_calibration_error,
)
from src.domain.equity_curve import EquityCurve

# Maksymalna liczba słupków sparklini krzywej kapitału (ostatnie N transakcji) —
# trzymamy zwięźle, by sekcja była czytelna i mailowalna.
_SPARKLINE_BARS = 24
_SPARKLINE_MAX_PX = 36


def render_track_record_html(
    equity_curve: EquityCurve | None,
    calibration_buckets: list[CalibrationBucket] | None,
    lessons: LessonsReport | None,
) -> str:
    """Fragment HTML sekcji Track Record. Pusty string, gdy wszystkie trzy
    podsekcje są puste."""
    subsections: list[str] = []
    if equity_curve is not None and equity_curve.n_trades > 0:
        subsections.append(_render_equity_html(equity_curve))
    if calibration_buckets:
        subsections.append(_render_calibration_html(calibration_buckets))
    if lessons is not None and lessons.items:
        subsections.append(_render_lessons_html(lessons))

    if not subsections:
        return ""

    header = (
        "<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>"
        "📈 Track Record — skuteczność i kalibracja</h2>"
    )
    return header + "".join(subsections)


def _render_equity_html(curve: EquityCurve) -> str:
    """T1 — podsumowanie krzywej kapitału + sparklinia tabelaryczna."""
    total_color = "#059669" if curve.total_return >= 0 else "#dc2626"
    summary = (
        "<div style='padding: 10px 14px; background: #eff6ff; "
        "border-left: 3px solid #2563eb; border-radius: 4px; "
        "margin-bottom: 10px; font-size: 13px;'>"
        "<strong>Krzywa kapitału (paper-PnL):</strong> wynik łączny "
        f"<span style='color: {total_color}; font-weight: 600;'>"
        f"{curve.total_return * 100:+.1f}%</span> · trafność "
        f"{curve.win_rate * 100:.1f}% · max obsunięcie "
        f"{curve.max_drawdown * 100:.1f}% · transakcji: {curve.n_trades}"
        "</div>"
    )
    return summary + _render_sparkline_html(curve)


def _render_sparkline_html(curve: EquityCurve) -> str:
    """Zwięzła sparklinia equity jako tabela słupków (email-safe, bez SVG/obrazków).

    Słupki skalowane do zakresu [min, max] equity w oknie ostatnich N punktów."""
    points = list(curve.points)[-_SPARKLINE_BARS:]
    if not points:
        return ""
    equities = [p.equity for p in points]
    lo, hi = min(equities), max(equities)
    span = (hi - lo) or 1.0
    cells: list[str] = []
    for p in points:
        # Wysokość ≥2px, by nawet płaski odcinek był widoczny.
        height = 2 + int((p.equity - lo) / span * (_SPARKLINE_MAX_PX - 2))
        color = "#0ea5e9" if p.period_return >= 0 else "#f97316"
        cells.append(
            "<td style='vertical-align: bottom; padding: 0 1px;'>"
            f"<div style='width: 6px; height: {height}px; "
            f"background: {color}; border-radius: 1px;'></div>"
            "</td>"
        )
    return (
        f"<table style='border-collapse: collapse; height: {_SPARKLINE_MAX_PX}px; "
        "margin: 0 0 16px 0;'><tr>" + "".join(cells) + "</tr></table>"
    )


def _severity_color(value: float) -> str:
    """Próg rozjazdu → kolor: ≥0.15 czerwony, ≥0.07 żółty, inaczej zielony."""
    if value >= 0.15:
        return "#dc2626"
    if value >= 0.07:
        return "#ca8a04"
    return "#16a34a"


def _render_calibration_html(buckets: list[CalibrationBucket]) -> str:
    """T2 — reliability diagram jako tabela: bucket pewności → realny hit-rate."""
    ece = expected_calibration_error(buckets)
    rows: list[str] = []
    for b in buckets:
        gap = abs(b.hit_rate - b.mean_confidence)
        # Im większy rozjazd pewności vs realnego hit-rate, tym bardziej czerwono.
        color = _severity_color(gap)
        rows.append(
            "<tr>"
            f"<td style='padding: 6px 8px;'>{b.lower * 100:.0f}–{b.upper * 100:.0f}%</td>"
            f"<td style='padding: 6px 8px; text-align: right;'>{b.count}</td>"
            f"<td style='padding: 6px 8px; text-align: right;'>"
            f"{b.mean_confidence * 100:.0f}%</td>"
            f"<td style='padding: 6px 8px; text-align: right; color: {color}; "
            f"font-weight: 600;'>{b.hit_rate * 100:.0f}%</td>"
            "</tr>"
        )
    ece_color = _severity_color(ece)
    return (
        "<p style='font-size: 12px; color: #6b7280; margin: 12px 0 6px 0;'>"
        "🎯 Krzywa kalibracji (deklarowana pewność vs realny hit-rate):</p>"
        "<table style='width: 100%; border-collapse: collapse; "
        "font-size: 13px; margin-bottom: 8px;'>"
        "<thead><tr style='background: #f9fafb; color: #6b7280; "
        "text-transform: uppercase; font-size: 11px;'>"
        "<th style='padding: 6px 8px; text-align: left;'>Pewność</th>"
        "<th style='padding: 6px 8px; text-align: right;'>N</th>"
        "<th style='padding: 6px 8px; text-align: right;'>Śr. pewność</th>"
        "<th style='padding: 6px 8px; text-align: right;'>Hit-rate</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        "<p style='font-size: 12px; color: #6b7280; margin: 0 0 16px 0;'>"
        f"Błąd kalibracji (ECE): <strong style='color: {ece_color};'>"
        f"{ece * 100:.1f}%</strong> — im niżej, tym lepiej pewność odpowiada "
        "realnej trafności.</p>"
    )


def _render_lessons_html(lessons: LessonsReport) -> str:
    """T4 — panel „lessons learned": błędne predykcje z diagnozą + motyw."""
    parts: list[str] = []
    theme_note = ""
    if lessons.recurring_theme:
        theme_note = (
            " · powracający motyw: "
            f"<strong>{_html.escape(lessons.recurring_theme)}</strong>"
        )
    parts.append(
        "<p style='font-size: 12px; color: #6b7280; margin: 12px 0 6px 0;'>"
        f"📓 Lessons learned ({lessons.total_mistakes} błędów z diagnozą)"
        f"{theme_note}:</p>"
    )
    items: list[str] = []
    for item in lessons.items:
        change = (
            f" ({item.actual_change_pct * 100:+.1f}%)"
            if item.actual_change_pct is not None
            else ""
        )
        items.append(
            "<li style='padding: 6px 0; border-bottom: 1px solid #f3f4f6; "
            "font-size: 13px;'>"
            f"<strong>{_html.escape(item.symbol)}</strong> "
            f"<span style='color: #6b7280;'>"
            f"[{_html.escape(item.predicted_trend)}{change}]</span> "
            f"{_html.escape(item.insight)}"
            "</li>"
        )
    parts.append(
        "<ul style='list-style: none; padding: 0; margin: 0 0 16px 0;'>"
        + "".join(items)
        + "</ul>"
    )
    return "".join(parts)
