"""DAT-19: rotacja spółek w tygodniowym refreshu fundamentów Alpha Vantage.

Przy jednym kluczu (25 requestów/dobę, 2 na spółkę) refresh zawsze kończył się
na tych samych pierwszych 12 spółkach z listy — reszta nigdy nie dostawała
świeżych fundamentów. Reguła: najpierw spółki bez wpisu w cache, potem najstarszy
`fetched_at`, remis alfabetycznie; partia = budżet requestów / 2.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.domain.fundamentals_refresh import refresh_budget, select_refresh_batch

_NOW = datetime(2026, 10, 18, 6, 0, tzinfo=UTC)


def test_selects_budget_oldest_with_missing_first() -> None:
    # 15 spółek: 2 bez wpisu, 13 z wpisem odświeżonym kolejno 1..13 dni temu.
    fetched_at: dict[str, datetime | None] = {"NEW1": None, "NEW2": None}
    for age in range(1, 14):
        fetched_at[f"S{age:02d}"] = _NOW - timedelta(days=age)

    batch = select_refresh_batch(fetched_at, budget=12)

    oldest_ten = [f"S{age:02d}" for age in range(13, 3, -1)]
    assert batch == ["NEW1", "NEW2", *oldest_ten]


def test_ties_are_broken_alphabetically() -> None:
    same = _NOW - timedelta(days=7)
    fetched_at: dict[str, datetime | None] = {"MSFT": same, "AAPL": same, "ZZZ": None, "AMZN": None}

    assert select_refresh_batch(fetched_at, budget=3) == ["AMZN", "ZZZ", "AAPL"]


def test_budget_larger_than_the_list_takes_everything() -> None:
    fetched_at: dict[str, datetime | None] = {"AAPL": _NOW, "MSFT": None}

    assert select_refresh_batch(fetched_at, budget=12) == ["MSFT", "AAPL"]


def test_zero_or_negative_budget_selects_nothing() -> None:
    assert select_refresh_batch({"AAPL": None}, budget=0) == []
    assert select_refresh_batch({"AAPL": None}, budget=-3) == []


@pytest.mark.parametrize(("keys", "budget"), [(0, 0), (1, 12), (2, 25), (3, 37)])
def test_budget_is_half_of_the_daily_requests_of_all_keys(keys: int, budget: int) -> None:
    # 25 requestów na klucz dziennie, 2 requesty na spółkę (OVERVIEW + EARNINGS).
    assert refresh_budget(keys) == budget
