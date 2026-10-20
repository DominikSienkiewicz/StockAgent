"""Testy domenowego modelu kosztu cyklu (FinOps).

Sedno reguły: cykl odcięty bramką volatility jest DARMOWY (0 wywołań → 0.0),
nieznane źródło nie wywala się (wkład 0 + sygnał), a suma zawsze równa sumie
rozbicia. Cennik to stała modułu — jedno źródło prawdy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from src.domain.finops import (
    UNIT_COST_USD,
    CycleCallBound,
    CycleCost,
    cycle_call_upper_bound,
    estimate_cycle_cost,
)

if TYPE_CHECKING:
    from src.domain.finops import ModelPrice


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
        _bound(**cast(dict[str, Any], {field: -1}))


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


# --- FIN-12: budżet płatnych wywołań LLM na cykl ---------------------------------


def test_paid_budget_allows_until_cap() -> None:
    from src.domain.finops import paid_budget_allows

    # Miękki sufit: wpuszcza, dopóki wydane < cap (symbol w toku może go przebić).
    assert paid_budget_allows(0, 10) is True
    assert paid_budget_allows(9, 10) is True
    assert paid_budget_allows(10, 10) is False
    assert paid_budget_allows(18, 10) is False
    # None = bez limitu (zachowanie sprzed FIN-12).
    assert paid_budget_allows(18, None) is True


def test_paid_budget_rejects_negative_spend() -> None:
    from src.domain.finops import paid_budget_allows

    with pytest.raises(ValueError, match="spent_calls"):
        paid_budget_allows(-1, 10)


def test_paid_llm_calls_count_the_prediction_and_the_council_only() -> None:
    from src.domain.finops import PAID_LLM_SOURCES, paid_llm_calls

    calls = {"llm": 2, "council_llm": 8, "sentiment": 1, "news": 1, "embedding": 2}

    assert frozenset({"llm", "council_llm"}) == PAID_LLM_SOURCES
    assert paid_llm_calls(calls) == 10
    assert paid_llm_calls({}) == 0


def test_rank_by_magnitude_breaks_ties_alphabetically() -> None:
    from decimal import Decimal

    from src.domain.finops import rank_by_magnitude

    deltas = {
        "MSFT": Decimal("0.03"),
        "AAPL": Decimal("-0.05"),
        "NVDA": Decimal("0.05"),
        "AMD": Decimal("-0.03"),
        "BTC": Decimal("0"),
    }

    # Malejąco po |Δ|; remis po |Δ| rozstrzyga alfabet, nie kolejność z config.toml.
    assert rank_by_magnitude(deltas) == ["AAPL", "NVDA", "AMD", "MSFT", "BTC"]
    assert rank_by_magnitude({}) == []


# --- FIN-30: tygodniowy budżet płatnych wywołań LLM Slow Loopa -------------------


def test_slow_loop_budget_trims_jobs_over_weekly_cap() -> None:
    from src.domain.finops import slow_loop_paid_budget

    requested = {"recap": 1, "judge": 1, "backfill": 5}

    # Kolejność jobów to pierwszeństwo: ostatni dostaje resztę budżetu.
    assert slow_loop_paid_budget(requested, 6) == {"recap": 1, "judge": 1, "backfill": 4}
    assert slow_loop_paid_budget(requested, None) == requested


def test_slow_loop_budget_gives_nothing_past_an_exhausted_cap() -> None:
    from src.domain.finops import slow_loop_paid_budget

    assert slow_loop_paid_budget({"recap": 1, "judge": 1}, 1) == {"recap": 1, "judge": 0}
    assert slow_loop_paid_budget({"recap": 1, "judge": 1}, 0) == {"recap": 0, "judge": 0}


@pytest.mark.parametrize(
    ("requested", "cap", "field"),
    [({"recap": -1}, 6, "recap"), ({"recap": 1}, -1, "cap")],
)
def test_slow_loop_budget_rejects_negative_inputs(
    requested: dict[str, int], cap: int, field: str
) -> None:
    from src.domain.finops import slow_loop_paid_budget

    with pytest.raises(ValueError, match=field):
        slow_loop_paid_budget(requested, cap)


# --- FIN-13: koszt z prawdziwych tokenów (cennik per model) ----------------------


def _sonnet_rates() -> dict[str, ModelPrice]:
    from src.domain.finops import ModelPrice

    return {
        "claude-sonnet-4-6": ModelPrice(
            input_usd_per_mtok=3.0,
            output_usd_per_mtok=15.0,
            cache_read_usd_per_mtok=0.30,
            cache_write_usd_per_mtok=3.75,
        ),
        "text-embedding-3-small": ModelPrice(input_usd_per_mtok=0.02),
    }


def test_price_usage_uses_per_model_rates() -> None:
    from src.domain.finops import LLMUsage, price_usage

    usage = LLMUsage(
        provider="anthropic", model="claude-sonnet-4-6", input_tokens=1000, output_tokens=200
    )

    # 1000 × $3/1M + 200 × $15/1M = 0.003 + 0.003.
    assert price_usage(usage, _sonnet_rates()) == pytest.approx(0.006)


def test_price_usage_prices_cache_reads_and_writes() -> None:
    from src.domain.finops import LLMUsage, price_usage

    usage = LLMUsage(
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )

    assert price_usage(usage, _sonnet_rates()) == pytest.approx(0.30 + 3.75)


def test_price_usage_of_an_unknown_model_is_none() -> None:
    from src.domain.finops import LLMUsage, price_usage

    usage = LLMUsage(provider="openai", model="gpt-9-ultra", input_tokens=10, output_tokens=10)

    # Nie zgadujemy stawki: brak modelu w cenniku = brak wyceny (fallback na stałą).
    assert price_usage(usage, _sonnet_rates()) is None


def test_price_usage_refuses_tokens_of_an_unpriced_category() -> None:
    from src.domain.finops import LLMUsage, price_usage

    # Cennik embeddingu ma tylko input — output bez stawki to brak wyceny, nie 0.
    usage = LLMUsage(
        provider="openai", model="text-embedding-3-small", input_tokens=500, output_tokens=3
    )

    assert price_usage(usage, _sonnet_rates()) is None


def test_price_usage_of_input_only_usage_needs_only_the_input_rate() -> None:
    from src.domain.finops import LLMUsage, price_usage

    usage = LLMUsage(
        provider="openai", model="text-embedding-3-small", input_tokens=500, output_tokens=0
    )

    assert price_usage(usage, _sonnet_rates()) == pytest.approx(500 * 0.02 / 1_000_000)


def test_llm_usage_rejects_negative_token_counts() -> None:
    from src.domain.finops import LLMUsage

    with pytest.raises(ValueError, match="cache_read_tokens"):
        LLMUsage(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=-1,
        )


def test_token_cost_by_source_sums_each_source() -> None:
    from src.domain.finops import LLMUsage, token_cost_by_source

    sonnet = LLMUsage(
        provider="anthropic", model="claude-sonnet-4-6", input_tokens=1000, output_tokens=200
    )
    embedding = LLMUsage(
        provider="openai", model="text-embedding-3-small", input_tokens=1000, output_tokens=0
    )

    costs = token_cost_by_source(
        [("llm", sonnet), ("llm", sonnet), ("embedding", embedding)], _sonnet_rates()
    )

    assert costs == {
        "llm": pytest.approx(0.012),
        "embedding": pytest.approx(1000 * 0.02 / 1_000_000),
    }


def test_token_cost_by_source_drops_a_source_with_an_unpriced_call() -> None:
    from src.domain.finops import LLMUsage, token_cost_by_source

    priced = LLMUsage(
        provider="anthropic", model="claude-sonnet-4-6", input_tokens=10, output_tokens=10
    )
    unpriced = LLMUsage(provider="openai", model="gpt-9-ultra", input_tokens=10, output_tokens=10)

    # Częściowa wycena zaniżałaby źródło — całe źródło wraca do szacunku ze stałej.
    costs = token_cost_by_source(
        [("council_llm", priced), ("council_llm", unpriced), ("llm", priced)], _sonnet_rates()
    )

    assert set(costs) == {"llm"}


def test_cycle_cost_takes_token_costs_and_falls_back_per_source() -> None:
    from datetime import date

    calls = {"llm": 2, "sentiment": 2, "council_llm": 8}

    cost = estimate_cycle_cost(
        calls, token_costs={"llm": 0.012}, priced_as_of=date(2026, 10, 20)
    )

    lines = {line.source: line for line in cost.lines}
    assert lines["llm"].from_tokens is True
    assert lines["llm"].subtotal_usd == pytest.approx(0.012)
    assert lines["llm"].unit_cost_usd == pytest.approx(0.006)
    assert lines["council_llm"].from_tokens is False
    assert lines["council_llm"].subtotal_usd == pytest.approx(8 * UNIT_COST_USD["council_llm"])
    assert lines["sentiment"].from_tokens is False
    assert cost.total_usd == pytest.approx(sum(line.subtotal_usd for line in cost.lines))
    assert cost.total_calls == 12
    assert cost.priced_as_of == date(2026, 10, 20)
    assert cost.from_tokens is True


def test_cycle_cost_without_token_costs_is_the_unit_estimate() -> None:
    cost = estimate_cycle_cost({"llm": 2})

    assert cost.lines[0].from_tokens is False
    assert cost.from_tokens is False
    assert cost.priced_as_of is None


def test_token_cost_of_a_source_without_counted_calls_is_ignored() -> None:
    # Licznik cyklu decyduje, które źródła są w rozbiciu; sam rekord tokenów bez
    # policzonego wywołania nie dopisuje linii.
    cost = estimate_cycle_cost({"llm": 1}, token_costs={"embedding": 0.5})

    assert [line.source for line in cost.lines] == ["llm"]
    assert cost.total_usd == pytest.approx(UNIT_COST_USD["llm"])


def test_llm_call_entry_carries_no_prompt_or_answer() -> None:
    import dataclasses

    from src.domain.finops import LLMCallEntry, LLMUsage

    fields = {field.name for field in dataclasses.fields(LLMCallEntry)}
    usage_fields = {field.name for field in dataclasses.fields(LLMUsage)}

    # Kontekst portfela: ledger zapisuje liczby, nigdy treść promptu ani odpowiedzi.
    assert fields == {"run_id", "node", "symbol", "usage", "cost_usd", "price_as_of"}
    assert usage_fields == {
        "provider",
        "model",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    }


# --- WIA-11: tygodniowy najgorszy przypadek Fast Loopa przy suficie FIN-12 ------


def test_capped_bound_without_a_ceiling_is_the_plain_bound() -> None:
    from src.domain.finops import capped_cycle_call_upper_bound

    capped = capped_cycle_call_upper_bound(
        n_gated=42, n_av_symbols=42, personas=7, debate=False, tool_rounds=0,
        cap=None, concurrency=4,
    )
    plain = cycle_call_upper_bound(
        n_gated=42, n_av_symbols=42, personas=7, debate=False, tool_rounds=0
    )

    assert dict(capped.calls) == dict(plain.calls)
    assert capped.av_requests == plain.av_requests


def test_ceiling_bounds_council_requests_by_cap_and_concurrency() -> None:
    # Licznik sufitu przed ostatnim wpuszczonym symbolem ≤ 99, a po nim radę kończą
    # najwyżej 4 symbole w locie po 8 requestów (7 person + chairman): 99 + 32.
    from src.domain.finops import capped_cycle_call_upper_bound

    bound = capped_cycle_call_upper_bound(
        n_gated=42, n_av_symbols=42, personas=7, debate=False, tool_rounds=0,
        cap=100, concurrency=4,
    )

    assert bound.calls["council_llm"] == 131
    # Symbol z ruchem poniżej progu rady wydaje tylko predykcję, więc sufit 100 nie
    # tnie 42 symboli: predykcja, sentyment, newsy i embeddingi zostają pełne.
    assert bound.calls["llm"] == 84
    assert bound.calls["sentiment"] == 42
    assert bound.calls["news"] == 42
    assert bound.calls["embedding"] == 84
    assert bound.av_requests == 42


def test_ceiling_cuts_analysed_symbols_but_not_their_diagnoses() -> None:
    # Bez rady każdy wpuszczony symbol dokłada do licznika co najmniej predykcję:
    # wpuszczonych ≤ (3 − 1) + 1 = 3. Diagnoza pomyłki w reflect idzie przed sufitem,
    # więc pozostałe 7 symboli za bramką płaci po jednej.
    from src.domain.finops import capped_cycle_call_upper_bound

    bound = capped_cycle_call_upper_bound(
        n_gated=10, n_av_symbols=10, personas=0, debate=False, tool_rounds=0,
        cap=3, concurrency=1,
    )

    assert bound.calls["llm"] == 3 * 2 + 7
    assert bound.calls["council_llm"] == 0
    assert bound.calls["sentiment"] == 3
    assert bound.calls["news"] == 3
    assert bound.calls["embedding"] == 6
    assert bound.av_requests == 3


def test_ceiling_never_raises_the_plain_bound() -> None:
    from src.domain.finops import capped_cycle_call_upper_bound

    bound = capped_cycle_call_upper_bound(
        n_gated=2, n_av_symbols=1, personas=7, debate=True, tool_rounds=3,
        cap=1000, concurrency=4,
    )
    plain = cycle_call_upper_bound(
        n_gated=2, n_av_symbols=1, personas=7, debate=True, tool_rounds=3
    )

    assert dict(bound.calls) == dict(plain.calls)
    assert bound.av_requests == 1


@pytest.mark.parametrize(("cap", "concurrency"), [(0, 1), (100, 0)])
def test_capped_bound_rejects_an_empty_ceiling_or_pool(cap: int, concurrency: int) -> None:
    from src.domain.finops import capped_cycle_call_upper_bound

    with pytest.raises(ValueError, match="must be >= 1"):
        capped_cycle_call_upper_bound(
            n_gated=1, n_av_symbols=1, personas=7, debate=False, tool_rounds=0,
            cap=cap, concurrency=concurrency,
        )


def test_weekly_worst_case_is_the_cycle_bound_times_five_trading_days() -> None:
    from src.domain.finops import TRADING_DAYS_PER_WEEK, weekly_cost_budget

    bound = cycle_call_upper_bound(
        n_gated=2, n_av_symbols=2, personas=7, debate=False, tool_rounds=0
    )
    cycle_usd = estimate_cycle_cost(bound.calls).total_usd

    weekly = weekly_cost_budget(bound, cycles_per_day=2, budget_usd=None)

    assert TRADING_DAYS_PER_WEEK == 5
    assert weekly.cycles_per_week == 10
    assert weekly.cycle_worst_usd == pytest.approx(cycle_usd)
    assert weekly.worst_case_usd == pytest.approx(cycle_usd * 10)


def test_weekly_budget_passes_only_when_it_covers_the_worst_case() -> None:
    from src.domain.finops import weekly_cost_budget

    bound = cycle_call_upper_bound(
        n_gated=10, n_av_symbols=10, personas=7, debate=False, tool_rounds=0
    )
    # 10 × (2 × 0.01 + 8 × 0.0025 + 0.0005 + 0.0005 + 2 × 0.00002) = 0.4104 na cykl.
    worst = weekly_cost_budget(bound, cycles_per_day=1, budget_usd=None).worst_case_usd

    assert worst == pytest.approx(2.052)
    assert weekly_cost_budget(bound, cycles_per_day=1, budget_usd=3.0).ok is True
    assert weekly_cost_budget(bound, cycles_per_day=1, budget_usd=2.0).ok is False
    # Bez budżetu nie ma z czym porównać — kontrola milczy.
    assert weekly_cost_budget(bound, cycles_per_day=1, budget_usd=None).ok is True


@pytest.mark.parametrize(
    ("cycles_per_day", "budget_usd", "field"),
    [(0, None, "cycles_per_day"), (1, -1.0, "budget_usd")],
)
def test_weekly_budget_rejects_invalid_inputs(
    cycles_per_day: int, budget_usd: float | None, field: str
) -> None:
    from src.domain.finops import weekly_cost_budget

    bound = cycle_call_upper_bound(
        n_gated=1, n_av_symbols=1, personas=7, debate=False, tool_rounds=0
    )

    with pytest.raises(ValueError, match=field):
        weekly_cost_budget(bound, cycles_per_day=cycles_per_day, budget_usd=budget_usd)
