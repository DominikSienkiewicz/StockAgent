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

from src.application.report_formatting import direction_pct, pct
from src.application.report_models import SymbolResult
from src.application.report_signals import (
    DIVERGENCE_PRICE_THRESHOLD,
    DIVERGENCE_SENTIMENT_THRESHOLD,
)
from src.application.report_styles import delta_tone, fill_tone

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
        # MAIL-06 (c): układ i kolor w klasach arkusza (`report_styles`), szerokość
        # słupka liczona z danych zostaje inline.
        cells.append(
            "<tr>"
            f"<td class='bl'>{_html.escape(symbol)}</td>"
            "<td class='bv'>"
            f"<div class='br {fill_tone(value)}' style='width:{width}px'></div>"
            f"<span class='bn'>{pct(value, signed=True)}</span>"
            "</td></tr>"
        )
    return (
        "<div class='gw'>"
        f"<div class='gt'>{_html.escape(title)}</div>"
        "<table role='presentation' class='bt'><tbody>"
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

    Bierze tylko symbole `saved` z policzalnym `expected_change` i prognozą
    modelu — baseline cold-startu (+0.00%) nie jest prognozą. Brak takich → "".
    """
    saved = [
        r
        for r in results
        if r.status == "saved" and r.expected_change is not None and not r.ml_cold_start
    ]
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
    return (
        f"<span class='qc {delta_tone(delta)}'>"
        f"{warn}{_html.escape(symbol)} {direction_pct(delta)}</span>"
    )


def _quadrant_cell(chips: list[str]) -> str:
    """Komórka jednego kwadrantu — chipy symboli albo '—', gdy pusto."""
    inner = "".join(chips) if chips else "<span class='nm'>—</span>"
    return "<td class='qd'>" + inner + "</td>"


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

    # MAIL-06 (c): układ tabeli w klasach arkusza (`report_styles`).
    return (
        "<div class='gw'>"
        "<div class='gt'>Sentyment a ruch ceny (kwadranty)</div>"
        "<table class='qt'><tbody>"
        "<tr>"
        "<td class='qh'></td>"
        "<td class='qh'>Sentyment pozytywny</td>"
        "<td class='qh'>Sentyment negatywny</td>"
        "</tr>"
        "<tr>"
        "<td class='ql'>Cena rośnie</td>"
        + _quadrant_cell(quadrants[(True, True)])
        + _quadrant_cell(quadrants[(True, False)])
        + "</tr>"
        "<tr>"
        "<td class='ql'>Cena spada</td>"
        + _quadrant_cell(quadrants[(False, True)])
        + _quadrant_cell(quadrants[(False, False)])
        + "</tr>"
        "</tbody></table></div>"
    )
