"""NIEZ-37 — budżet rozmiaru bazy: suma relacji `public` i ostrzeżenie od 70%."""

from __future__ import annotations

import pytest

from src.domain.db_size import (
    BYTES_PER_MB,
    DB_SIZE_WARNING_FRACTION,
    LARGEST_RELATIONS,
    RelationSize,
    assess_db_size,
)


def _sizes(*megabytes: int) -> list[RelationSize]:
    return [RelationSize(f"t{i}", mb * BYTES_PER_MB) for i, mb in enumerate(megabytes)]


@pytest.mark.parametrize(
    ("megabytes", "warn"),
    [((200, 149), False), ((200, 150), True), ((400, 200), True)],
)
def test_warning_starts_at_seventy_percent_of_the_budget(
    megabytes: tuple[int, ...], warn: bool
) -> None:
    assessment = assess_db_size(_sizes(*megabytes), budget_mb=500)

    assert DB_SIZE_WARNING_FRACTION == 0.7
    assert assessment.total_bytes == sum(megabytes) * BYTES_PER_MB
    assert assessment.needs_warning is warn


def test_without_a_budget_there_is_no_fraction_and_no_warning() -> None:
    assessment = assess_db_size(_sizes(900), budget_mb=None)

    assert assessment.budget_bytes is None
    assert assessment.budget_fraction is None
    assert assessment.needs_warning is False


def test_largest_relations_come_first_and_are_capped() -> None:
    sizes = [RelationSize(name, size) for name, size in [("b", 5), ("a", 5), ("c", 9)]]
    sizes += [RelationSize(f"small{i}", 1) for i in range(10)]

    largest = assess_db_size(sizes, budget_mb=500).largest

    assert [size.name for size in largest[:3]] == ["c", "a", "b"]
    assert len(largest) == LARGEST_RELATIONS


def test_empty_report_is_zero_bytes() -> None:
    assessment = assess_db_size([], budget_mb=500)

    assert assessment.total_bytes == 0
    assert assessment.budget_fraction == 0.0
    assert assessment.largest == ()
