"""FIN-13: `UsageLedger` — pamięć tokenów cyklu za portem `UsageSink`.

Adapter zgłasza samo zużycie (dostawca, model, tokeny); kto płacił (źródło
cennika), z którego węzła i za jaki symbol, dopisuje zakres `usage_scope`
ustawiony w application — adapter nie zna grafu ani Supabase.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

from src.application.ports import UsageSink
from src.application.usage_ledger import UNATTRIBUTED, UsageLedger, usage_scope
from src.domain.finops import LLMUsage, ModelPrice

_SONNET = LLMUsage(
    provider="anthropic", model="claude-sonnet-4-6", input_tokens=1000, output_tokens=200
)
_UNKNOWN = LLMUsage(provider="openai", model="gpt-9-ultra", input_tokens=10, output_tokens=10)
_PRICES = {"claude-sonnet-4-6": ModelPrice(input_usd_per_mtok=3.0, output_usd_per_mtok=15.0)}


def test_ledger_is_a_usage_sink() -> None:
    assert isinstance(UsageLedger(), UsageSink)


def test_record_takes_the_attribution_of_the_current_scope() -> None:
    ledger = UsageLedger()

    with usage_scope(source="llm", node="predict", symbol="NVDA"):
        ledger.record(_SONNET)

    (record,) = ledger.records()
    assert (record.source, record.node, record.symbol, record.usage) == (
        "llm",
        "predict",
        "NVDA",
        _SONNET,
    )


def test_nested_scope_overrides_only_the_given_fields() -> None:
    ledger = UsageLedger()

    with usage_scope(node="council", symbol="AAPL"), usage_scope(source="council_llm"):
        ledger.record(_SONNET)

    (record,) = ledger.records()
    assert (record.source, record.node, record.symbol) == ("council_llm", "council", "AAPL")


def test_scope_is_restored_after_the_block() -> None:
    ledger = UsageLedger()

    with usage_scope(source="llm", node="reflect", symbol="AAPL"):
        pass
    ledger.record(_SONNET)

    (record,) = ledger.records()
    assert (record.source, record.node, record.symbol) == (UNATTRIBUTED, UNATTRIBUTED, None)


def test_concurrent_records_are_not_lost() -> None:
    ledger = UsageLedger()
    barrier = threading.Barrier(8)

    def record_many(symbol: str) -> None:
        barrier.wait()
        with usage_scope(source="llm", node="predict", symbol=symbol):
            for _ in range(250):
                ledger.record(_SONNET)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record_many, [f"S{i}" for i in range(8)]))

    records = ledger.records()
    assert len(records) == 2000
    # Zakres jest per wątek: żaden rekord nie dostał symbolu innego wątku w złej liczbie.
    assert {record.symbol for record in records} == {f"S{i}" for i in range(8)}
    assert all(
        sum(1 for record in records if record.symbol == f"S{i}") == 250 for i in range(8)
    )


def test_token_costs_price_each_source_and_skip_partially_unpriced_ones() -> None:
    ledger = UsageLedger()
    with usage_scope(source="llm", node="predict", symbol="NVDA"):
        ledger.record(_SONNET)
        ledger.record(_SONNET)
    with usage_scope(source="council_llm", node="council", symbol="NVDA"):
        ledger.record(_SONNET)
        ledger.record(_UNKNOWN)

    assert ledger.token_costs(_PRICES) == {"llm": pytest.approx(0.012)}


def test_entries_price_every_call_and_stamp_the_run() -> None:
    ledger = UsageLedger()
    with usage_scope(source="llm", node="predict", symbol="NVDA"):
        ledger.record(_SONNET)
    with usage_scope(source="llm", node="weekly_recap"):
        ledger.record(_UNKNOWN)

    entries = ledger.entries(run_id="123456", price_table=_PRICES, price_as_of=date(2026, 10, 20))

    assert [(e.run_id, e.node, e.symbol) for e in entries] == [
        ("123456", "predict", "NVDA"),
        ("123456", "weekly_recap", None),
    ]
    assert entries[0].cost_usd == pytest.approx(0.006)
    assert entries[0].price_as_of == date(2026, 10, 20)
    # Nieznany model: koszt NULL, ale tokeny zostają w ledgerze do rekoncyliacji.
    assert entries[1].cost_usd is None
    assert entries[1].usage == _UNKNOWN


def test_empty_ledger_has_no_entries_and_no_token_costs() -> None:
    ledger = UsageLedger()

    assert ledger.records() == ()
    assert ledger.token_costs(_PRICES) == {}
    assert ledger.entries(run_id="local", price_table=_PRICES, price_as_of=None) == []
