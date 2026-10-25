"""FIN-06: `cost_envelope` — tabela min/max płatnych wywołań i kosztu cyklu
z górnej granicy domeny (`cycle_call_upper_bound`), do wklejenia w docs."""

from __future__ import annotations

import pytest

from src.domain.finops import UNIT_COST_USD, cycle_call_upper_bound
from src.tools.cost_envelope import EnvelopeInputs, cost_envelope, render_cost_envelope

_INPUTS = EnvelopeInputs(n_gated=41, n_av_symbols=42, personas=7, debate=False, tool_rounds=0)


def test_worst_case_per_cycle_matches_unit_costs() -> None:
    envelope = cost_envelope(_INPUTS)
    bound = cycle_call_upper_bound(
        n_gated=41, n_av_symbols=42, personas=7, debate=False, tool_rounds=0
    )

    expected = sum(calls * UNIT_COST_USD[source] for source, calls in bound.calls.items())
    assert envelope.max_usd == pytest.approx(expected)
    assert envelope.max_calls == sum(bound.calls.values())
    # FIN-04: jeden request AV na symbol za bramką — min(41, 42).
    assert envelope.av_requests == 41
    for row in envelope.rows:
        assert row.max_calls == bound.calls[row.source]
        assert row.max_usd == pytest.approx(row.max_calls * UNIT_COST_USD[row.source])


def test_minimum_is_the_fully_gated_free_cycle() -> None:
    envelope = cost_envelope(_INPUTS)

    assert envelope.min_usd == 0.0
    assert all(row.min_calls == 0 and row.min_usd == 0.0 for row in envelope.rows)


def test_rows_follow_the_price_list_order() -> None:
    assert [row.source for row in cost_envelope(_INPUTS).rows] == list(UNIT_COST_USD)


def test_render_prints_an_estimated_markdown_table() -> None:
    text = render_cost_envelope(cost_envelope(_INPUTS))

    assert "estimated" in text.lower()
    assert "| Source | Calls min | Calls max | Unit cost (USD) | USD min | USD max |" in text
    assert "| `council_llm` | 0 | 328 | 0.0025 | 0.0000 | 0.8200 |" in text
    assert "| `embedding` | 0 | 82 | 0.00002 | 0.0000 | 0.0016 |" in text
    assert "| **Total** | 0 | 574 |  | 0.0000 | 1.6826 |" in text
    assert "0–41" in text
    assert "41 symbols" in text


# Wynik `cost-envelope` dla `config.toml` z 2026-10-20 (docs/configuration.md):
# bez routingu modeli i bez wariantu w cieniu tabela ma zostać bajt w bajt ta sama.
_TABLE_WITHOUT_ROUTING_AND_SHADOW = "\n".join([
    "Estimated cost envelope of one Fast Loop cycle (`UNIT_COST_USD`, "
    "an order-of-magnitude model, not an invoice).",
    "Worst case: 42 symbols past the volatility gate, 7 council personas, debate off, "
    "tool use off. Minimum: every symbol below the gate.",
    "",
    "| Source | Calls min | Calls max | Unit cost (USD) | USD min | USD max |",
    "|---|---:|---:|---:|---:|---:|",
    "| `llm` | 0 | 84 | 0.01 | 0.0000 | 0.8400 |",
    "| `council_llm` | 0 | 336 | 0.0025 | 0.0000 | 0.8400 |",
    "| `sentiment` | 0 | 42 | 0.0005 | 0.0000 | 0.0210 |",
    "| `news` | 0 | 42 | 0.0005 | 0.0000 | 0.0210 |",
    "| `embedding` | 0 | 84 | 0.00002 | 0.0000 | 0.0017 |",
    "| **Total** | 0 | 588 |  | 0.0000 | 1.7237 |",
    "",
    "Alpha Vantage NEWS_SENTIMENT requests: 0–42 per cycle (key quota, not priced).",
])

_DOCS_INPUTS = EnvelopeInputs(
    n_gated=42, n_av_symbols=42, personas=7, debate=False, tool_rounds=0
)


def test_envelope_without_routing_and_shadow_renders_as_before() -> None:
    explicit_zeros = EnvelopeInputs(
        n_gated=42, n_av_symbols=42, personas=7, debate=False, tool_rounds=0,
        max_escalations=0, max_shadow_calls=0,
    )

    assert render_cost_envelope(cost_envelope(_DOCS_INPUTS)) == _TABLE_WITHOUT_ROUTING_AND_SHADOW
    assert render_cost_envelope(cost_envelope(explicit_zeros)) == (
        _TABLE_WITHOUT_ROUTING_AND_SHADOW
    )


def test_frontier_escalations_and_shadow_calls_join_the_envelope() -> None:
    # WIA-11: eskalacje FIN-21 dokładają do `llm`, cień EXP-26 to osobny wiersz
    # wyceniony stałą predykcji; suma i opis scenariusza je obejmują.
    inputs = EnvelopeInputs(
        n_gated=42, n_av_symbols=42, personas=7, debate=False, tool_rounds=0,
        max_escalations=5, max_shadow_calls=3,
    )

    text = render_cost_envelope(cost_envelope(inputs))

    assert "| `llm` | 0 | 89 | 0.01 | 0.0000 | 0.8900 |" in text
    # Wiersz cienia idzie po wierszach cennika produkcji.
    assert (
        "| `embedding` | 0 | 84 | 0.00002 | 0.0000 | 0.0017 |\n"
        "| `shadow_llm` | 0 | 3 | 0.01 | 0.0000 | 0.0300 |"
    ) in text
    assert "| **Total** | 0 | 596 |  | 0.0000 | 1.8037 |" in text
    assert "model routing up to 5 frontier escalations" in text
    assert "shadow prompt variant up to 3 calls" in text


def test_render_names_debate_and_tool_use_when_on() -> None:
    inputs = EnvelopeInputs(n_gated=2, n_av_symbols=2, personas=7, debate=True, tool_rounds=3)

    text = render_cost_envelope(cost_envelope(inputs))

    assert "debate on" in text
    assert "tool use up to 3 rounds" in text


# --- WIA-11: wiersz tygodniowy ------------------------------------------------


def test_weekly_row_names_the_ceiling_the_cycles_and_the_verdict() -> None:
    from src.domain.finops import WeeklyCostBudget
    from src.tools.cost_envelope import render_weekly_budget

    within = WeeklyCostBudget(cycle_worst_usd=1.21118, cycles_per_week=5, budget_usd=7.0)
    over = WeeklyCostBudget(cycle_worst_usd=1.21118, cycles_per_week=10, budget_usd=7.0)

    assert render_weekly_budget(within) == (
        "Weekly worst case at the FIN-12 ceiling: 5 cycles (1 a day × 5 trading days) × "
        "1.2112 USD = 6.0559 USD; `fast_loop_weekly_budget_usd` = 7.00 USD, within the budget."
    )
    assert "10 cycles (2 a day × 5 trading days)" in render_weekly_budget(over)
    assert render_weekly_budget(over).endswith("OVER the budget.")


def test_weekly_row_without_a_budget_says_so() -> None:
    from src.domain.finops import WeeklyCostBudget
    from src.tools.cost_envelope import render_weekly_budget

    text = render_weekly_budget(
        WeeklyCostBudget(cycle_worst_usd=0.5, cycles_per_week=5, budget_usd=None)
    )

    assert text.endswith("= 2.5000 USD; no `fast_loop_weekly_budget_usd` set.")
