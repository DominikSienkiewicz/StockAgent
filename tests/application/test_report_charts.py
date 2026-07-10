"""Testy modułu wykresów raportu — wersja email-safe (inline HTML div-bary).

Kontrakt po roadmap #2: trzy funkcje `build_*_url` zwracają teraz FRAGMENT
HTML (nie URL), bez żadnego zewnętrznego hosta (koniec z QuickChart.io).
Dane portfela nie mogą wyciekać w URL-u do obcego serwisu, a Gmail/Outlook
z blokadą zdalnych obrazków muszą zobaczyć wykres, nie dziurę."""

from __future__ import annotations

import re
from decimal import Decimal

from src.application.report_charts import (
    build_correlation_chart_html,
    build_delta_chart_html,
    build_forecast_chart_html,
)
from src.application.report_models import SymbolResult


def _saved(symbol: str, delta: str) -> SymbolResult:
    return SymbolResult(symbol=symbol, status="saved", delta=Decimal(delta))


def _bar_width_for(html: str, symbol: str) -> int:
    """Wyłuskuje px szerokości słupka należącego do danego symbolu.

    Słupek symbolu leży w tym samym <tr> co jego etykieta, więc szukamy
    pierwszego `width: Npx` po nazwie symbolu."""
    idx = html.index(symbol)
    match = re.search(r"width:\s*(\d+)px", html[idx:])
    assert match is not None, f"brak słupka dla {symbol}"
    return int(match.group(1))


# ---------------------------------------------------------------------------
# build_delta_chart_html — Δ ceny per cykl
# ---------------------------------------------------------------------------
class TestBuildChartUrl:
    def test_returns_inline_html_not_url(self) -> None:
        results = [_saved("AAPL", "0.025"), _saved("MSFT", "-0.005")]
        html = build_delta_chart_html(results)
        assert "quickchart" not in html
        assert "http" not in html  # brak zewnętrznego hosta
        assert "<div" in html and "<table" in html
        assert "AAPL" in html and "MSFT" in html

    def test_label_uses_cykl_not_12h(self) -> None:
        html = build_delta_chart_html([_saved("AAPL", "0.02"), _saved("MSFT", "0.01")])
        assert "12h" not in html
        assert "cykl" in html.lower()

    def test_empty_results_yield_empty_string(self) -> None:
        assert build_delta_chart_html([]) == ""

    def test_error_results_yield_empty_string(self) -> None:
        results = [SymbolResult(symbol="X", status="error", error_message="boom")]
        assert build_delta_chart_html(results) == ""

    def test_results_without_delta_yield_empty_string(self) -> None:
        assert build_delta_chart_html([SymbolResult(symbol="X", status="ignored")]) == ""

    def test_value_label_carried_in_body(self) -> None:
        html = build_delta_chart_html([_saved("AAPL", "0.025"), _saved("MSFT", "-0.015")])
        assert "+2.50%" in html
        assert "-1.50%" in html

    def test_small_move_not_visually_equal_to_large_move(self) -> None:
        """NAJWIĘKSZE RYZYKO zamrożone: -0.5% NIE może wyglądać jak -8%.

        Słupek -8% ma być WYRAŹNIE szerszy niż słupek -0.5%, a oba muszą
        nieść swoją wartość liczbową w treści (samo minimum szerokości nie
        wystarcza do odczytu)."""
        results = [_saved("BIG", "-0.08"), _saved("SMALL", "-0.005")]
        html = build_delta_chart_html(results)

        w_big = _bar_width_for(html, "BIG")
        w_small = _bar_width_for(html, "SMALL")

        # Wyraźnie szerszy: co najmniej dwukrotnie (w praktyce dużo więcej).
        assert w_big > w_small * 2
        # Oba niosą wartość liczbową.
        assert "-8.00%" in html
        assert "-0.50%" in html

    def test_bar_has_minimum_width(self) -> None:
        """Nawet mikroskopijny ruch renderuje widoczny słupek (>=1px)."""
        results = [_saved("BIG", "-0.08"), _saved("TINY", "-0.0001")]
        html = build_delta_chart_html(results)
        assert _bar_width_for(html, "TINY") >= 1

    def test_symbol_is_html_escaped(self) -> None:
        results = [
            SymbolResult(symbol="<b>X</b>", status="saved", delta=Decimal("0.02")),
            _saved("MSFT", "0.01"),
        ]
        html = build_delta_chart_html(results)
        assert "<b>X</b>" not in html
        assert "&lt;b&gt;X&lt;/b&gt;" in html


# ---------------------------------------------------------------------------
# build_forecast_chart_html — prognoza
# ---------------------------------------------------------------------------
class TestForecastChartUrl:
    def test_returns_inline_html_for_saved(self) -> None:
        results = [
            SymbolResult(
                symbol="AAPL", status="saved",
                current_price=Decimal("100"), target_price=Decimal("105"),
            )
        ]
        html = build_forecast_chart_html(results)
        assert "quickchart" not in html
        assert "<table" in html
        assert "AAPL" in html
        assert "+5.00%" in html

    def test_ignored_yield_empty_string(self) -> None:
        results = [SymbolResult(symbol="X", status="ignored", delta=Decimal("0.005"))]
        assert build_forecast_chart_html(results) == ""

    def test_saved_without_target_yield_empty_string(self) -> None:
        results = [
            SymbolResult(symbol="X", status="saved", current_price=Decimal("100"))
        ]
        assert build_forecast_chart_html(results) == ""

    def test_label_uses_cykl_not_12h(self) -> None:
        results = [
            SymbolResult(
                symbol="AAPL", status="saved",
                current_price=Decimal("100"), target_price=Decimal("103"),
            )
        ]
        html = build_forecast_chart_html(results)
        assert "12h" not in html
        assert "cykl" in html.lower()


# ---------------------------------------------------------------------------
# build_correlation_chart_html — scatter → tabela kwadrantów
# ---------------------------------------------------------------------------
def _pt(symbol: str, sentiment: float, delta: str) -> SymbolResult:
    return SymbolResult(
        symbol=symbol, status="saved",
        sentiment_score=sentiment, delta=Decimal(delta),
    )


class TestCorrelationChartUrl:
    def test_returns_quadrant_table_for_3_plus_points(self) -> None:
        results = [_pt("A", 0.3, "0.05"), _pt("B", -0.2, "-0.03"), _pt("C", 0.1, "0.01")]
        html = build_correlation_chart_html(results)
        assert "quickchart" not in html
        assert "scatter" not in html.lower()
        assert "<table" in html
        assert "A" in html and "B" in html and "C" in html
        # Tabela kwadrantów: osie sentymentu i ceny opisane po polsku.
        assert "Sentyment" in html
        assert "Cena" in html

    def test_fewer_than_3_points_yield_empty_string(self) -> None:
        results = [_pt("A", 0.3, "0.05"), _pt("B", -0.2, "-0.03")]
        assert build_correlation_chart_html(results) == ""

    def test_no_sentiment_excluded(self) -> None:
        results = [
            _pt("A", 0.3, "0.05"),
            _pt("B", -0.2, "-0.03"),
            SymbolResult(symbol="C", status="saved", delta=Decimal("0.01")),
        ]
        # Tylko 2 punkty z sentymentem → za mało.
        assert build_correlation_chart_html(results) == ""

    def test_strong_divergence_is_flagged(self) -> None:
        """Symbol przekraczający OBA progi DIVERGENCE_* (cena w górę, sentyment
        mocno w dół) dostaje oznaczenie rozjazdu ⚠."""
        results = [
            _pt("DIV", -0.5, "0.05"),  # cena +5%, sentyment -0.5 → silny rozjazd
            _pt("OK", 0.3, "0.04"),
            _pt("C", 0.1, "0.01"),
        ]
        html = build_correlation_chart_html(results)
        # Oznaczenie ⚠ pojawia się przy symbolu rozjazdu.
        div_idx = html.index("DIV")
        ok_idx = html.index("OK")
        assert "⚠" in html
        # ⚠ jest bliżej DIV niż OK (leży w jego chipie).
        warn_idx = html.index("⚠")
        assert abs(warn_idx - div_idx) < abs(warn_idx - ok_idx)

    def test_weak_divergence_not_flagged(self) -> None:
        """Rozjazd poniżej progów NIE jest oznaczany ⚠."""
        results = [_pt("A", -0.1, "0.01"), _pt("B", 0.05, "-0.01"), _pt("C", 0.1, "0.005")]
        html = build_correlation_chart_html(results)
        assert "⚠" not in html
