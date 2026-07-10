"""Wykresy raportu e-mail — email-safe, inline HTML (koniec z QuickChart).

Roadmap #2: wcześniej trzy wykresy były `<img>` z QuickChart.io. To dwa
problemy naraz: Gmail/Outlook z blokadą zdalnych obrazków pokazywały dziury
w sercu maila, a **dane portfela wyciekały w URL-u** do zewnętrznego serwisu.

Teraz każda funkcja zwraca FRAGMENT HTML (div-bary / tabela kwadrantów),
bez żadnego zewnętrznego hosta. Technika słupków tabelarycznych jest wprost
zapożyczona z przetestowanego `_render_sparkline_html`
(`src/application/report_track_record.py`).

Wszystkie trzy funkcje zwracają FRAGMENT HTML gotowy do wstawienia w mail —
nie URL. Dawniej nazywały się `build_*_url` i oddawały link do quickchart.io;
nazwa zmieniła się razem z kontraktem, bo `*_url` zwracające HTML kłamałoby.
Brak danych → pusty string (`""`), nigdy pusty `<div>` ani placeholder.
"""

from __future__ import annotations

import html as _html
from decimal import Decimal

from src.application.report_formatting import delta_color, pct
from src.application.report_models import SymbolResult
from src.application.report_signals import (
    DIVERGENCE_PRICE_THRESHOLD,
    DIVERGENCE_SENTIMENT_THRESHOLD,
)

# Minimalna i maksymalna szerokość słupka w px. Minimum > 0, by najmniejszy
# ruch pozostał widoczny; jednocześnie na tyle małe względem maksimum, żeby
# mały ruch (np. -0.5%) NIE wyglądał jak duży (np. -8%) — inaczej normalizacja
# zrównałaby je wizualnie. Wartość liczbowa i tak zawsze towarzyszy słupkowi.
_BAR_MIN_PX = 3
_BAR_MAX_PX = 240


def _render_bar_chart_html(title: str, rows: list[tuple[str, Decimal]]) -> str:
    """Renderuje poziomy bar-chart jako tabelę div-słupków (email-safe).

    `rows` to pary (symbol, wartość ułamkowa, np. `Decimal("0.025")` = +2.5%).
    Szerokość słupka jest proporcjonalna do |wartość| względem maksimum w
    zestawie, z twardym minimum `_BAR_MIN_PX` — mały ruch pozostaje widoczny,
    a mimo to wyraźnie węższy od dużego. Każdy wiersz ZAWSZE niesie wartość
    liczbową obok słupka (samo minimum szerokości nie wystarcza do odczytu).
    Pusta lista → "".
    """
    if not rows:
        return ""
    max_abs = max((abs(v) for _, v in rows), default=Decimal("0")) or Decimal("1")
    cells: list[str] = []
    for symbol, value in rows:
        frac = float(abs(value) / max_abs)
        width = _BAR_MIN_PX + int(frac * (_BAR_MAX_PX - _BAR_MIN_PX))
        color = delta_color(value)
        cells.append(
            "<tr>"
            "<td style='padding: 2px 8px 2px 0; font-size: 12px; "
            f"white-space: nowrap; color: #374151;'>{_html.escape(symbol)}</td>"
            "<td style='padding: 2px 0;'>"
            f"<div style='display: inline-block; width: {width}px; height: 12px; "
            f"background: {color}; border-radius: 2px; vertical-align: middle;'></div>"
            "<span style='font-size: 12px; color: #374151; margin-left: 6px;'>"
            f"{pct(value, signed=True)}</span>"
            "</td></tr>"
        )
    return (
        "<div style='margin: 16px 0;'>"
        "<div style='font-size: 14px; font-weight: 600; margin-bottom: 8px; "
        f"color: #1f2937;'>{_html.escape(title)}</div>"
        "<table style='border-collapse: collapse; width: 100%;'><tbody>"
        + "".join(cells)
        + "</tbody></table></div>"
    )


def build_delta_chart_html(results: list[SymbolResult]) -> str:
    """Fragment HTML: poziomy bar-chart zmiany ceny (Δ) per symbol.

    Etykieta mówi „(cykl)", bo pętla chodzi raz na dobę handlową (nazwy `*_12h`
    są przeżytkiem). Brak ważnych danych (pusto / same błędy / brak Δ) → "".
    """
    valid = [r for r in results if r.delta is not None and r.status != "error"]
    if not valid:
        return ""
    valid_sorted = sorted(valid, key=lambda r: abs(r.delta or Decimal("0")), reverse=True)
    rows = [(r.symbol, r.delta or Decimal("0")) for r in valid_sorted]
    return _render_bar_chart_html("Zmiana ceny (cykl)", rows)


def build_forecast_chart_html(results: list[SymbolResult]) -> str:
    """Fragment HTML: poziomy bar-chart prognozowanej zmiany ceny.

    Bierze tylko symbole `saved` z policzalnym `expected_change`.
    Brak takich symboli → "".
    """
    saved = [r for r in results if r.status == "saved" and r.expected_change is not None]
    if not saved:
        return ""
    saved_sorted = sorted(
        saved, key=lambda r: abs(r.expected_change or Decimal("0")), reverse=True
    )
    rows = [(r.symbol, r.expected_change or Decimal("0")) for r in saved_sorted]
    return _render_bar_chart_html("Prognoza ruchu ceny (cykl)", rows)


def _render_quadrant_chip(symbol: str, delta: Decimal, is_divergence: bool) -> str:
    """Chip pojedynczego symbolu w tabeli kwadrantów: nazwa + Δ, kolor Δ.

    Silny rozjazd (przekroczenie OBU progów DIVERGENCE_*) dostaje prefiks ⚠.
    Symbol jest escapowany — może pochodzić z danych zewnętrznych."""
    warn = "⚠ " if is_divergence else ""
    color = delta_color(delta)
    return (
        "<span style='display: inline-block; margin: 2px 4px 2px 0; "
        "padding: 1px 6px; border-radius: 3px; font-size: 12px; "
        f"background: #f3f4f6; color: {color};'>"
        f"{warn}{_html.escape(symbol)} {pct(delta, signed=True)}</span>"
    )


def _quadrant_cell(chips: list[str]) -> str:
    """Komórka jednego kwadrantu — chipy symboli albo '—', gdy pusto."""
    inner = "".join(chips) if chips else "<span style='color: #9ca3af;'>—</span>"
    return (
        "<td style='padding: 8px; border: 1px solid #e5e7eb; "
        "vertical-align: top; width: 40%;'>" + inner + "</td>"
    )


def build_correlation_chart_html(results: list[SymbolResult]) -> str:
    """Fragment HTML: tabela kwadrantów sentyment × ruch ceny.

    Zastępuje dawny scatter (QuickChart). Symbole trafiają do jednej z
    czterech ćwiartek wg znaku sentymentu (oś X) i znaku Δ ceny (oś Y).
    Silny rozjazd — przekroczenie OBU progów `DIVERGENCE_PRICE_THRESHOLD` /
    `DIVERGENCE_SENTIMENT_THRESHOLD` z `report_signals` (cena w jedną stronę,
    sentyment mocno w drugą) — jest oznaczony ⚠.
    Mniej niż 3 punkty (sentyment + Δ) → "".
    """
    points = [
        r for r in results
        if r.sentiment_score is not None and r.delta is not None and r.status != "error"
    ]
    if len(points) < 3:
        return ""

    # (cena_rośnie, sentyment_pozytywny) → lista chipów
    quadrants: dict[tuple[bool, bool], list[str]] = {
        (True, True): [], (True, False): [],
        (False, True): [], (False, False): [],
    }
    for r in points:
        delta = r.delta or Decimal("0")
        sentiment = r.sentiment_score or 0.0
        is_divergence = (
            (delta > DIVERGENCE_PRICE_THRESHOLD
             and sentiment < -DIVERGENCE_SENTIMENT_THRESHOLD)
            or (delta < -DIVERGENCE_PRICE_THRESHOLD
                and sentiment > DIVERGENCE_SENTIMENT_THRESHOLD)
        )
        chip = _render_quadrant_chip(r.symbol, delta, is_divergence)
        quadrants[(delta >= 0, sentiment >= 0)].append(chip)

    header_style = (
        "padding: 6px 8px; font-size: 11px; color: #6b7280; "
        "text-transform: uppercase; border: 1px solid #e5e7eb; background: #f9fafb;"
    )
    row_label_style = (
        "padding: 8px; font-size: 12px; font-weight: 600; color: #374151; "
        "border: 1px solid #e5e7eb; background: #f9fafb; white-space: nowrap;"
    )
    return (
        "<div style='margin: 16px 0;'>"
        "<div style='font-size: 14px; font-weight: 600; margin-bottom: 8px; "
        "color: #1f2937;'>Sentyment a ruch ceny (kwadranty)</div>"
        "<table style='border-collapse: collapse; width: 100%; "
        "table-layout: fixed;'><tbody>"
        "<tr>"
        f"<td style='{header_style}'></td>"
        f"<td style='{header_style}'>Sentyment pozytywny</td>"
        f"<td style='{header_style}'>Sentyment negatywny</td>"
        "</tr>"
        "<tr>"
        f"<td style='{row_label_style}'>Cena rośnie</td>"
        + _quadrant_cell(quadrants[(True, True)])
        + _quadrant_cell(quadrants[(True, False)])
        + "</tr>"
        "<tr>"
        f"<td style='{row_label_style}'>Cena spada</td>"
        + _quadrant_cell(quadrants[(False, True)])
        + _quadrant_cell(quadrants[(False, False)])
        + "</tr>"
        "</tbody></table></div>"
    )
