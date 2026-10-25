"""NIEZ-38: wspólny kontrakt kodów wyjścia entry-pointów (decyzja 2026-10-23).

Exit 1 wyłącznie wtedy, gdy padły WSZYSTKIE jednostki przebiegu (symbole); częściowa
porażka to exit 0, WARNING w logu i liczby w step summary. Czerwień od jednego
symbolu uczy ją ignorować, a zielony job przy całkowitej awarii ją ukrywa.
"""

from __future__ import annotations

ALL_FAILED_EXIT_CODE = 1
# Jedna linia WARNING dla każdego entry pointa: (przebieg, padło, wszystkich).
PARTIAL_FAILURE_LOG = "%s: częściowa porażka — padło %d/%d, exit 0 (NIEZ-38)"


def run_exit_code(failures: int, total: int) -> int:
    """1, gdy przebieg miał jednostki i padły wszystkie; w każdym innym razie 0."""
    if total > 0 and failures >= total:
        return ALL_FAILED_EXIT_CODE
    return 0


def is_partial_failure(failures: int, total: int) -> bool:
    """Część jednostek padła, ale nie wszystkie — exit 0 z WARNING-iem."""
    return 0 < failures < total
