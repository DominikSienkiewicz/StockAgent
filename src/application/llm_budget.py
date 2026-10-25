"""FIN-25: dzienny i miesięczny limit USD z ledgera `llm_calls` przed cyklem.

Entry point czyta wydatek okien RAZ, przed pierwszym płatnym wywołaniem cyklu.
Wyczerpane okno zamyka płatne wywołania całego cyklu (graf idzie ścieżką FIN-12
`budget_exhausted`), a mail dostaje banner. Reguły okna i progu żyją w domenie
(`budget_window_start`, `budget_exceeded`); tu jest kolejność odczytów i decyzja
fail-open: ledger, którego nie da się przeczytać, nie zatrzymuje analizy
(decyzja 2026-10-24) — limit chroni przed pętlą dispatchy, a nie zastępuje cyklu.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime

from src.application.errors import PortError
from src.application.ports import RepositoryPort
from src.domain.finops import BudgetBreach, BudgetWindow, budget_exceeded, budget_window_start

logger = logging.getLogger(__name__)

# Doba przed miesiącem: wyczerpana doba rozstrzyga bez drugiego odczytu.
_WINDOW_ORDER = (BudgetWindow.DAY, BudgetWindow.MONTH)


def find_budget_breach(
    repository: RepositoryPort,
    *,
    now: datetime,
    caps: Mapping[BudgetWindow, float | None],
) -> BudgetBreach | None:
    """Pierwsze wyczerpane okno (doba, potem miesiąc) albo None.

    None także wtedy, gdy odczyt ledgera padł (`PortError`) — cykl płaci dalej,
    a log niesie WARNING z przyczyną. Okno bez limitu nie jest czytane."""
    for window in _WINDOW_ORDER:
        cap = caps.get(window)
        if cap is None:
            continue
        since = budget_window_start(window, now)
        try:
            spent = repository.get_llm_cost_since(since)
        except PortError as exc:
            logger.warning(
                "LLM budget: ledger read for the %s window failed (%s) — "
                "fail-open, the cycle keeps its paid calls",
                window.value,
                exc,
            )
            return None
        if budget_exceeded(spent, cap):
            logger.warning(
                "LLM budget: %s cap spent (%.4f/%.2f USD since %s) — "
                "this cycle makes no paid calls",
                window.value,
                spent,
                cap,
                since.isoformat(),
            )
            return BudgetBreach(window=window, spent_usd=spent, cap_usd=cap)
    return None
