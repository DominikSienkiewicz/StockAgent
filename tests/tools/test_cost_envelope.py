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


def test_render_names_debate_and_tool_use_when_on() -> None:
    inputs = EnvelopeInputs(n_gated=2, n_av_symbols=2, personas=7, debate=True, tool_rounds=3)

    text = render_cost_envelope(cost_envelope(inputs))

    assert "debate on" in text
    assert "tool use up to 3 rounds" in text
