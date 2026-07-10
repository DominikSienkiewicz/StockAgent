"""Testy renderera sekcji "Koszt cyklu" (FinOps) w raporcie e-mail.

Kluczowa decyzja produktowa: cykl DARMOWY to najciekawsza informacja tej sekcji
(bramka volatility odcięła wszystkie symbole), więc renderujemy go jawnie —
nie chowamy. `None` (brak licznika) → "".
"""

from __future__ import annotations

from src.application.report_finops import (
    render_finops_html,
    render_finops_text,
)
from src.domain.finops import estimate_cycle_cost


def test_none_suppresses_section_html() -> None:
    """Brak danych o koszcie (None) → sekcja się chowa."""
    assert render_finops_html(None) == ""


def test_none_suppresses_section_text() -> None:
    assert render_finops_text(None) == ""


def test_free_cycle_renders_explicitly_html() -> None:
    """Zero wywołań → sekcja renderuje się i MÓWI WPROST, że cykl był darmowy."""
    cost = estimate_cycle_cost({})
    html = render_finops_html(cost)
    assert html != ""
    assert "0 płatnych wywołań" in html
    assert "darmow" in html.lower()
    assert "bramka volatility" in html.lower()


def test_free_cycle_renders_explicitly_text() -> None:
    cost = estimate_cycle_cost({})
    text = render_finops_text(cost)
    assert text != ""
    assert "0 płatnych wywołań" in text
    assert "darmow" in text.lower()


def test_priced_cycle_shows_breakdown_and_total_html() -> None:
    """Cykl z kosztem: rozbicie per źródło + suma, ze słowem 'szacunkow'."""
    cost = estimate_cycle_cost({"llm": 2, "council_llm": 1})
    html = render_finops_html(cost)
    assert "szacunkow" in html.lower()
    # Suma sformatowana jest obecna w treści.
    assert f"{cost.total_usd:.4f}" in html
    # Liczba płatnych wywołań widoczna.
    assert "3" in html


def test_priced_cycle_shows_breakdown_and_total_text() -> None:
    cost = estimate_cycle_cost({"llm": 2, "council_llm": 1})
    text = render_finops_text(cost)
    assert "szacunkow" in text.lower()
    assert f"{cost.total_usd:.4f}" in text


def test_unknown_source_is_signalled_and_does_not_crash_html() -> None:
    """Nieznane źródło jest jawnie zasygnalizowane w sekcji, bez wysypki."""
    cost = estimate_cycle_cost({"llm": 1, "mystery_api": 2})
    html = render_finops_html(cost)
    assert "mystery_api" in html
    assert "cennik" in html.lower()


def test_unknown_source_is_signalled_text() -> None:
    cost = estimate_cycle_cost({"llm": 1, "mystery_api": 2})
    text = render_finops_text(cost)
    assert "mystery_api" in text


def test_unknown_source_html_is_escaped() -> None:
    """Klucz nieznanego źródła jest escapowany (żadnego wstrzyknięcia HTML)."""
    cost = estimate_cycle_cost({"<script>": 1})
    html = render_finops_html(cost)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
