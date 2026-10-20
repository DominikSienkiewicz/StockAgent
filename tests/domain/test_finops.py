"""Testy domenowego modelu kosztu cyklu (FinOps).

Sedno reguły: cykl odcięty bramką volatility jest DARMOWY (0 wywołań → 0.0),
nieznane źródło nie wywala się (wkład 0 + sygnał), a suma zawsze równa sumie
rozbicia. Cennik to stała modułu — jedno źródło prawdy.
"""

from __future__ import annotations

import pytest

from src.domain.finops import (
    UNIT_COST_USD,
    CycleCallBound,
    CycleCost,
    cycle_call_upper_bound,
    estimate_cycle_cost,
)


def test_zero_calls_is_free() -> None:
    """Pusty cykl (bramka volatility odcięła wszystko) kosztuje dokładnie 0.0."""
    cost = estimate_cycle_cost({})
    assert isinstance(cost, CycleCost)
    assert cost.total_usd == 0.0
    assert cost.total_calls == 0
    assert cost.lines == ()
    assert cost.unknown_sources == ()


def test_zero_counts_are_ignored() -> None:
    """Źródło z licznikiem 0 nie tworzy linii ani kosztu — cykl nadal darmowy."""
    cost = estimate_cycle_cost({"llm": 0, "council_llm": 0})
    assert cost.total_usd == 0.0
    assert cost.total_calls == 0
    assert cost.lines == ()


def test_negative_counts_are_ignored() -> None:
    """Ujemny licznik (błąd wołającego) jest pomijany, nie zaniża sumy."""
    cost = estimate_cycle_cost({"llm": -5})
    assert cost.total_usd == 0.0
    assert cost.total_calls == 0


def test_unknown_source_never_raises_and_is_signalled() -> None:
    """Nieznane źródło → wkład 0, jawnie zasygnalizowane, żadnego KeyError."""
    cost = estimate_cycle_cost({"quantum_oracle": 3})
    assert cost.total_usd == 0.0
    assert "quantum_oracle" in cost.unknown_sources
    # Realne wywołania i tak się odbyły — liczymy je, choć nie umiemy wycenić.
    assert cost.total_calls == 3
    assert cost.lines == ()


def test_total_equals_sum_of_breakdown() -> None:
    """Suma całkowita == suma subtotali z rozbicia (niezmiennik księgowy)."""
    cost = estimate_cycle_cost(
        {"llm": 2, "council_llm": 1, "sentiment": 4, "news": 1, "embedding": 3}
    )
    assert cost.total_usd == sum(line.subtotal_usd for line in cost.lines)
    assert cost.total_calls == sum(line.calls for line in cost.lines)


def test_pricing_comes_from_module_constant() -> None:
    """Wycena liczona JEST z `UNIT_COST_USD`, nie z rozsianej magicznej liczby."""
    cost = estimate_cycle_cost({"llm": 3})
    assert cost.total_usd == UNIT_COST_USD["llm"] * 3
    assert cost.lines[0].unit_cost_usd == UNIT_COST_USD["llm"]


def test_required_sources_are_priced() -> None:
    """Cennik pokrywa co najmniej wymagane źródła kosztu."""
    for source in ("llm", "council_llm", "sentiment", "news", "embedding"):
        assert source in UNIT_COST_USD
        assert UNIT_COST_USD[source] >= 0.0


def test_breakdown_sorted_by_cost_desc() -> None:
    """Rozbicie posortowane malejąco po koszcie — najdroższe źródło na górze."""
    cost = estimate_cycle_cost({"embedding": 1, "council_llm": 1})
    subtotals = [line.subtotal_usd for line in cost.lines]
    assert subtotals == sorted(subtotals, reverse=True)


def test_mixed_known_and_unknown() -> None:
    """Znane i nieznane źródła współistnieją: cena ze znanych, sygnał z nieznanych."""
    cost = estimate_cycle_cost({"llm": 2, "mystery": 1})
    assert cost.total_usd == UNIT_COST_USD["llm"] * 2
    assert cost.unknown_sources == ("mystery",)
    assert [line.source for line in cost.lines] == ["llm"]
    assert cost.total_calls == 3


def test_council_is_priced_per_persona_or_chairman_call() -> None:
    """FIN-03: `council_llm` liczy się per request (FIN-02), więc stawka jest per
    wywołanie persony lub chairmana — rada 7 person + chairman = 8 × 0.0025."""
    assert UNIT_COST_USD["council_llm"] == 0.0025

    cost = estimate_cycle_cost({"council_llm": 8})

    assert abs(cost.total_usd - 0.02) < 1e-12
    assert cost.lines[0].calls == 8


# --- FIN-06: górna granica płatnych wywołań cyklu ------------------------------


def _bound(
    *,
    n_gated: int = 1,
    n_av_symbols: int = 1,
    personas: int = 7,
    debate: bool = False,
    tool_rounds: int = 0,
) -> CycleCallBound:
    return cycle_call_upper_bound(
        n_gated=n_gated,
        n_av_symbols=n_av_symbols,
        personas=personas,
        debate=debate,
        tool_rounds=tool_rounds,
    )


def test_upper_bound_counts_council_as_personas_plus_chairman() -> None:
    assert _bound().calls["council_llm"] == 8
    assert _bound(n_gated=3).calls["council_llm"] == 24


def test_upper_bound_counts_the_debate_revision_round() -> None:
    assert _bound(debate=True).calls["council_llm"] == 15


def test_upper_bound_without_personas_has_no_council_calls() -> None:
    assert _bound(personas=0).calls["council_llm"] == 0


def test_upper_bound_llm_is_diagnosis_plus_prediction_per_symbol() -> None:
    # Diagnoza pomyłki w reflect (ta sama bramka) + jedno `analyze`.
    assert _bound(n_gated=4).calls["llm"] == 8


def test_upper_bound_llm_counts_every_tool_use_request() -> None:
    # Pętla tool-use: `tool_rounds` rund z toolami + jedna wymuszona bez nich.
    assert _bound(tool_rounds=3).calls["llm"] == 1 + 4


def test_upper_bound_meters_sentiment_news_and_embeddings_per_symbol() -> None:
    calls = _bound(n_gated=5).calls

    assert calls["sentiment"] == 5
    assert calls["news"] == 5
    # Embedding RAG w predict + ponowna próba w save po nieudanym RAG.
    assert calls["embedding"] == 10


def test_upper_bound_prices_every_source_of_the_price_list() -> None:
    assert set(_bound().calls) == set(UNIT_COST_USD)


def test_upper_bound_av_requests_follow_the_gated_symbols() -> None:
    # FIN-04: klient leniwy — jeden request AV na symbol za bramką, nie feed
    # wszystkich tickerów przy pierwszym symbolu.
    assert _bound(n_gated=1, n_av_symbols=42).av_requests == 1
    assert _bound(n_gated=50, n_av_symbols=42).av_requests == 42


def test_fully_gated_cycle_has_a_zero_bound() -> None:
    bound = _bound(n_gated=0, n_av_symbols=42, debate=True, tool_rounds=3)

    assert sum(bound.calls.values()) == 0
    assert bound.av_requests == 0
    assert estimate_cycle_cost(bound.calls).total_usd == 0.0


@pytest.mark.parametrize(
    "field", ["n_gated", "n_av_symbols", "personas", "tool_rounds"]
)
def test_upper_bound_rejects_negative_inputs(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _bound(**{field: -1})


# --- FIN-32: niezmiennik dobowego budżetu Alpha Vantage -------------------------


def test_av_budget_flags_worst_case_above_capacity() -> None:
    from src.domain.finops import AV_DAILY_REQUESTS_PER_KEY, av_daily_budget

    over = av_daily_budget(
        n_av_symbols=43, requests_per_symbol=1, cycles_per_day=1, n_keys=1
    )

    assert AV_DAILY_REQUESTS_PER_KEY == 25
    assert over.ok is False
    assert over.capacity == 25
    assert over.worst_case == 43
    assert av_daily_budget(43, 1, 1, 2).ok is True


def test_av_budget_multiplies_symbols_requests_and_cycles() -> None:
    from src.domain.finops import av_daily_budget

    # Kalendarz wyników = drugi request per symbol; dwa cykle na dobę.
    assert av_daily_budget(10, 2, 2, 1).worst_case == 40


def test_av_budget_at_capacity_is_ok() -> None:
    from src.domain.finops import av_daily_budget

    assert av_daily_budget(25, 1, 1, 1).ok is True


@pytest.mark.parametrize(
    "field", ["n_av_symbols", "requests_per_symbol", "cycles_per_day", "n_keys"]
)
def test_av_budget_rejects_negative_inputs(field: str) -> None:
    from src.domain.finops import av_daily_budget

    arguments = {"n_av_symbols": 1, "requests_per_symbol": 1, "cycles_per_day": 1, "n_keys": 1}
    with pytest.raises(ValueError, match=field):
        av_daily_budget(**{**arguments, field: -1})
