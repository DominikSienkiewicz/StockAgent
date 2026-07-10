"""Testy domenowego modelu kosztu cyklu (FinOps).

Sedno reguły: cykl odcięty bramką volatility jest DARMOWY (0 wywołań → 0.0),
nieznane źródło nie wywala się (wkład 0 + sygnał), a suma zawsze równa sumie
rozbicia. Cennik to stała modułu — jedno źródło prawdy.
"""

from __future__ import annotations

from src.domain.finops import (
    UNIT_COST_USD,
    CycleCost,
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
