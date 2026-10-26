"""FIN-06: koperta kosztu jednego cyklu Fast Loopa — czysta logika narzędzia.

Operator ma znać budżet przed włączeniem crona. Granicę wywołań liczy domena
(`cycle_call_upper_bound`), cenę — cennik `UNIT_COST_USD`; tu tylko składamy
tabelę min/max per źródło i drukujemy ją jako Markdown do docs/configuration.md.
Minimum to cykl, w którym bramka volatility odcina każdy symbol: zero wywołań.
`Settings` i persony składa `main_tools.py cost-envelope` (ARCH-39).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.finops import (
    EXPERIMENT_UNIT_COST_USD,
    TRADING_DAYS_PER_WEEK,
    UNIT_COST_USD,
    WeeklyCostBudget,
    cycle_call_upper_bound,
    estimate_cycle_cost,
)


@dataclass(frozen=True)
class EnvelopeInputs:
    """Wejście najgorszego przypadku: ile symboli przechodzi bramkę, ile tickerów
    ma feed AV, skład rady i tryby, które mnożą wywołania LLM. WIA-11: sufit
    eskalacji routingu (FIN-21) i limit wariantu w cieniu (EXP-26); 0 = flaga off."""

    n_gated: int
    n_av_symbols: int
    personas: int
    debate: bool
    tool_rounds: int
    max_escalations: int = 0
    max_shadow_calls: int = 0


@dataclass(frozen=True)
class EnvelopeRow:
    source: str
    min_calls: int
    max_calls: int
    unit_cost_usd: float
    min_usd: float
    max_usd: float


@dataclass(frozen=True)
class CostEnvelope:
    inputs: EnvelopeInputs
    rows: tuple[EnvelopeRow, ...]
    min_usd: float
    max_usd: float
    max_calls: int
    av_requests: int


def cost_envelope(inputs: EnvelopeInputs) -> CostEnvelope:
    """Tabela min/max per źródło cennika; suma maksimum = `estimate_cycle_cost`.
    Źródło eksperymentu (EXP-26) dostaje wiersz tylko, gdy granica go liczy."""
    bound = cycle_call_upper_bound(
        n_gated=inputs.n_gated,
        n_av_symbols=inputs.n_av_symbols,
        personas=inputs.personas,
        debate=inputs.debate,
        tool_rounds=inputs.tool_rounds,
        max_escalations=inputs.max_escalations,
        max_shadow_calls=inputs.max_shadow_calls,
    )
    worst = estimate_cycle_cost(bound.calls)
    experiments = {
        source: unit_cost
        for source, unit_cost in EXPERIMENT_UNIT_COST_USD.items()
        if bound.calls.get(source, 0) > 0
    }
    rows = tuple(
        EnvelopeRow(
            source=source,
            min_calls=0,
            max_calls=bound.calls[source],
            unit_cost_usd=unit_cost,
            min_usd=0.0,
            max_usd=bound.calls[source] * unit_cost,
        )
        for source, unit_cost in {**UNIT_COST_USD, **experiments}.items()
    )
    return CostEnvelope(
        inputs=inputs,
        rows=rows,
        min_usd=estimate_cycle_cost({}).total_usd,
        max_usd=worst.total_usd,
        max_calls=worst.total_calls,
        av_requests=bound.av_requests,
    )


def _unit(cost: float) -> str:
    return f"{cost:.5f}".rstrip("0").rstrip(".")


def _usd(amount: float) -> str:
    return f"{amount:.4f}"


def _scenario(inputs: EnvelopeInputs) -> str:
    debate = "debate on" if inputs.debate else "debate off"
    tool_use = (
        f"tool use up to {inputs.tool_rounds} rounds" if inputs.tool_rounds > 0 else "tool use off"
    )
    modes = [debate, tool_use]
    if inputs.max_escalations > 0:
        modes.append(f"model routing up to {inputs.max_escalations} frontier escalations")
    if inputs.max_shadow_calls > 0:
        modes.append(f"shadow prompt variant up to {inputs.max_shadow_calls} calls")
    return (
        f"Worst case: {inputs.n_gated} symbols past the volatility gate, "
        f"{inputs.personas} council personas, {', '.join(modes)}. "
        "Minimum: every symbol below the gate."
    )


def render_cost_envelope(envelope: CostEnvelope) -> str:
    """Markdown do wklejenia w docs: opis scenariusza, tabela i requesty AV."""
    lines = [
        "Estimated cost envelope of one Fast Loop cycle "
        "(`UNIT_COST_USD`, an order-of-magnitude model, not an invoice).",
        _scenario(envelope.inputs),
        "",
        "| Source | Calls min | Calls max | Unit cost (USD) | USD min | USD max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{row.source}` | {row.min_calls} | {row.max_calls} | {_unit(row.unit_cost_usd)} "
        f"| {_usd(row.min_usd)} | {_usd(row.max_usd)} |"
        for row in envelope.rows
    )
    lines.append(
        f"| **Total** | 0 | {envelope.max_calls} |  "
        f"| {_usd(envelope.min_usd)} | {_usd(envelope.max_usd)} |"
    )
    lines.extend(
        [
            "",
            f"Alpha Vantage NEWS_SENTIMENT requests: 0–{envelope.av_requests} per cycle "
            "(key quota, not priced).",
        ]
    )
    return "\n".join(lines)


def render_weekly_budget(weekly: WeeklyCostBudget) -> str:
    """WIA-11: wiersz tygodniowy pod tabelą — najgorszy tydzień przy suficie FIN-12
    i jego porównanie z `fast_loop_weekly_budget_usd`."""
    cycles_per_day = weekly.cycles_per_week // TRADING_DAYS_PER_WEEK
    line = (
        f"Weekly worst case at the FIN-12 ceiling: {weekly.cycles_per_week} cycles "
        f"({cycles_per_day} a day × {TRADING_DAYS_PER_WEEK} trading days) × "
        f"{_usd(weekly.cycle_worst_usd)} USD = {_usd(weekly.worst_case_usd)} USD"
    )
    if weekly.budget_usd is None:
        return f"{line}; no `fast_loop_weekly_budget_usd` set."
    verdict = "within the budget" if weekly.ok else "OVER the budget"
    return f"{line}; `fast_loop_weekly_budget_usd` = {weekly.budget_usd:.2f} USD, {verdict}."
