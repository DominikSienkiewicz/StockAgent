"""Testy renderera sekcji "Koszt cyklu" (FinOps) w raporcie e-mail.

Kluczowa decyzja produktowa: cykl DARMOWY to najciekawsza informacja tej sekcji
(bramka volatility odcięła wszystkie symbole), więc renderujemy go jawnie —
nie chowamy. `None` (brak licznika) → "".
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_builder import ReportContext, build_html_report
from src.application.report_finops import (
    render_finops_html,
    render_finops_text,
)
from src.application.report_models import SymbolResult
from src.domain.cycle_maturity import SkipReason
from src.domain.finops import BudgetBreach, BudgetWindow, CycleCost, estimate_cycle_cost


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
    assert f"{cost.total_usd:.4f}".replace(".", ",") in html
    # Liczba płatnych wywołań widoczna.
    assert "3" in html


def test_priced_cycle_shows_breakdown_and_total_text() -> None:
    cost = estimate_cycle_cost({"llm": 2, "council_llm": 1})
    text = render_finops_text(cost)
    assert "szacunkow" in text.lower()
    assert f"{cost.total_usd:.4f}".replace(".", ",") in text


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


# --- FIN-13: koszt z tokenów obok szacunku ze stałej ----------------------------


def _mixed_cost() -> CycleCost:
    from datetime import date

    return estimate_cycle_cost(
        {"llm": 2, "council_llm": 8, "sentiment": 2},
        token_costs={"llm": 0.0123, "council_llm": 0.00456},
        priced_as_of=date(2026, 10, 20),
    )


def test_token_lines_are_marked_as_computed_from_tokens_html() -> None:
    html = render_finops_html(_mixed_cost())

    assert "Model predykcyjny (LLM): 2 wywołania z tokenów = <strong>$0,0123</strong>" in html
    assert "Rada doradcza (LLM): 8 wywołań z tokenów = <strong>$0,0046</strong>" in html
    # Źródło bez danych o tokenach zostaje szacunkiem ze stałej, jawnie oznaczonym.
    assert "Sentyment (Alpha Vantage): 2 × ~$0,0005 = <strong>~$0,0010</strong> (szacunek)" in html


def test_token_lines_are_marked_as_computed_from_tokens_text() -> None:
    text = render_finops_text(_mixed_cost())

    assert "Model predykcyjny (LLM): 2 wywołania z tokenów = $0,0123" in text
    assert "Sentyment (Alpha Vantage): 2 × ~$0,0005 = ~$0,0010 (szacunek)" in text


def test_mixed_cycle_header_says_what_is_from_tokens() -> None:
    cost = _mixed_cost()
    html = render_finops_html(cost)
    text = render_finops_text(cost)
    total = f"{cost.total_usd:.4f}".replace(".", ",")

    for rendered in (html, text):
        assert f"~${total}" in rendered
        assert "część z tokenów, reszta szacunkiem" in rendered
        # Cennik dryfuje — data cennika stoi przy kwocie z tokenów.
        assert "cennik z 20.10.2026" in rendered


def test_fully_token_priced_cycle_has_no_estimate_marker() -> None:
    from datetime import date

    cost = estimate_cycle_cost(
        {"llm": 1}, token_costs={"llm": 0.0061}, priced_as_of=date(2026, 10, 20)
    )
    html = render_finops_html(cost)
    text = render_finops_text(cost)

    for rendered in (html, text):
        assert "Koszt cyklu z tokenów:" in rendered
        assert "$0,0061" in rendered
        assert "~$0,0061" not in rendered
        assert "(szacunek)" not in rendered
        assert "nie faktura" in rendered


def test_unit_only_cycle_renders_as_before() -> None:
    # Bez danych o tokenach (np. tokeny nieznane albo cennik pusty) sekcja jest
    # dokładnie taka jak przed FIN-13 — golden snapshot maila bez zmian.
    from src.domain.finops import estimate_cycle_cost as estimate

    calls = {"llm": 2, "council_llm": 1}
    assert render_finops_html(estimate(calls, token_costs={})) == render_finops_html(
        estimate(calls)
    )
    assert "(szacunek)" not in render_finops_text(estimate(calls))


def test_shadow_variant_calls_get_their_own_readable_line() -> None:
    # EXP-26: wywołania wariantu w cieniu to osobny klucz cennika — mail pokazuje je
    # jako osobną linię kosztu, nie jako surowy klucz.
    from datetime import date

    cost = estimate_cycle_cost(
        {"llm": 2, "shadow_llm": 1},
        token_costs={"llm": 0.0123, "shadow_llm": 0.0061},
        priced_as_of=date(2026, 10, 20),
    )

    html = render_finops_html(cost)
    text = render_finops_text(cost)

    assert "Wariant promptu w cieniu (LLM): 1 wywołanie z tokenów" in html
    assert "Wariant promptu w cieniu (LLM): 1 wywołanie z tokenów" in text


# --- FIN-36: trzy rozłączne przyczyny zatrzymania płatnych wywołań ----------------


def _usd_limit_breach() -> BudgetBreach:
    return BudgetBreach(BudgetWindow.DAY, spent_usd=2.134, cap_usd=2.0)


def _budget_cut(symbol: str) -> SymbolResult:
    return SymbolResult(
        symbol=symbol,
        status="ignored",
        delta=Decimal("-0.035"),
        current_price=Decimal("140"),
        skip_reason=SkipReason.BUDGET_EXHAUSTED,
    )


def test_cost_section_names_usd_limit_when_cycle_was_stopped_by_it() -> None:
    """Cykl zatrzymany dziennym limitem USD z ledgera: sekcja kosztu wskazuje limit,
    nie bramkę volatility, a notka budżetu nie mówi o limicie „cyklu”."""
    context = ReportContext(
        cycle_cost=estimate_cycle_cost({}), llm_budget_breach=_usd_limit_breach()
    )

    html, text = build_html_report(
        [_budget_cut("NVDA")], datetime(2026, 10, 20, 5, 30, tzinfo=UTC), 1.0, context
    )

    for rendered in (html, text):
        cost_section = rendered.split("Koszt cyklu" if rendered is html else "KOSZT CYKLU")[1]
        assert "dzienny limit USD" in cost_section
        assert "bramka volatility" not in cost_section
        assert "Pominięte — wyczerpany dzienny limit USD" in html
        assert "budżet cyklu" not in rendered.lower()
        assert "limit płatnych wywołań" not in rendered
    assert "POMINIĘTE — WYCZERPANY DZIENNY LIMIT USD" in text


def test_cost_section_names_the_monthly_usd_limit() -> None:
    breach = BudgetBreach(BudgetWindow.MONTH, spent_usd=30.5, cap_usd=30.0)

    html = render_finops_html(estimate_cycle_cost({}), budget_breach=breach)
    text = render_finops_text(estimate_cycle_cost({}), budget_breach=breach)

    for rendered in (html, text):
        assert "miesięczny limit USD" in rendered
        assert "$30,50 z $30,00" in rendered
        assert "darmow" not in rendered.lower()


def test_free_cycle_without_breach_still_credits_the_volatility_gate() -> None:
    html = render_finops_html(estimate_cycle_cost({}), budget_breach=None)

    assert "bramka volatility odcięła wszystkie symbole" in html
    assert "limit USD" not in html


def test_budget_note_keeps_the_cycle_ceiling_without_a_usd_breach() -> None:
    html, text = build_html_report(
        [_budget_cut("NVDA")], datetime(2026, 10, 20, 5, 30, tzinfo=UTC), 1.0
    )

    assert "Pominięte — wyczerpany budżet cyklu" in html
    assert "limit płatnych wywołań" in text
    assert "limit USD" not in html
