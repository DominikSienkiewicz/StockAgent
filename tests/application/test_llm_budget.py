"""FIN-25: odczyt limitu USD z ledgera `llm_calls` przed pierwszym płatnym wywołaniem.

Okna liczy domena (`budget_window_start`, `budget_exceeded`); tu jest kolejność
odczytów (doba, potem miesiąc) i decyzja fail-open: nieudany odczyt ledgera nie
zamyka płatnych wywołań, tylko zostawia WARNING (decyzja 2026-10-24).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from src.application.errors import ProviderUnavailable
from src.application.llm_budget import find_budget_breach
from src.application.ports import RepositoryPort
from src.domain.finops import BudgetBreach, BudgetWindow

NOW = datetime(2026, 10, 25, 5, 30, tzinfo=UTC)
DAY_START = datetime(2026, 10, 25, tzinfo=UTC)
MONTH_START = datetime(2026, 10, 1, tzinfo=UTC)


def _repository(spent: dict[datetime, float]) -> Mock:
    repository = Mock(spec=RepositoryPort)
    repository.get_llm_cost_since.side_effect = lambda since: spent[since]
    return repository


def test_spent_daily_cap_is_a_daily_breach() -> None:
    repository = _repository({DAY_START: 2.5, MONTH_START: 9.0})

    breach = find_budget_breach(
        repository,
        now=NOW,
        caps={BudgetWindow.DAY: 2.0, BudgetWindow.MONTH: 30.0},
    )

    assert breach == BudgetBreach(BudgetWindow.DAY, spent_usd=2.5, cap_usd=2.0)
    # Doba wyczerpana — miesiąca nie trzeba czytać.
    repository.get_llm_cost_since.assert_called_once_with(DAY_START)


def test_spent_monthly_cap_is_a_monthly_breach_within_the_daily_cap() -> None:
    repository = _repository({DAY_START: 0.4, MONTH_START: 30.2})

    breach = find_budget_breach(
        repository,
        now=NOW,
        caps={BudgetWindow.DAY: 2.0, BudgetWindow.MONTH: 30.0},
    )

    assert breach == BudgetBreach(BudgetWindow.MONTH, spent_usd=30.2, cap_usd=30.0)


def test_spend_within_both_caps_is_no_breach() -> None:
    repository = _repository({DAY_START: 0.4, MONTH_START: 12.0})

    assert (
        find_budget_breach(
            repository, now=NOW, caps={BudgetWindow.DAY: 2.0, BudgetWindow.MONTH: 30.0}
        )
        is None
    )
    assert repository.get_llm_cost_since.call_count == 2


def test_uncapped_windows_read_nothing() -> None:
    repository = _repository({})

    breach = find_budget_breach(
        repository, now=NOW, caps={BudgetWindow.DAY: None, BudgetWindow.MONTH: None}
    )

    assert breach is None
    repository.get_llm_cost_since.assert_not_called()


def test_failed_ledger_read_keeps_paying_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    repository = Mock(spec=RepositoryPort)
    repository.get_llm_cost_since.side_effect = ProviderUnavailable("PostgREST timeout")

    with caplog.at_level(logging.WARNING, logger="src.application.llm_budget"):
        breach = find_budget_breach(
            repository, now=NOW, caps={BudgetWindow.DAY: 2.0, BudgetWindow.MONTH: 30.0}
        )

    assert breach is None
    assert "fail-open" in caplog.text
    assert "PostgREST timeout" in caplog.text
