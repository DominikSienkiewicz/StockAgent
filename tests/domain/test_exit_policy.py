"""NIEZ-38: reguła kodu wyjścia wspólna dla Fast Loopa, Slow Loopa, watcha i backtestu."""

import pytest

from src.domain.exit_policy import ALL_FAILED_EXIT_CODE, is_partial_failure, run_exit_code


@pytest.mark.parametrize(
    ("failures", "total", "expected"),
    [
        (0, 3, 0),
        (1, 3, 0),
        (2, 3, 0),
        (3, 3, ALL_FAILED_EXIT_CODE),
        (0, 0, 0),
    ],
)
def test_exit_one_only_when_every_unit_failed(failures: int, total: int, expected: int) -> None:
    assert run_exit_code(failures, total) == expected


@pytest.mark.parametrize(
    ("failures", "total", "expected"),
    [(0, 3, False), (1, 3, True), (2, 3, True), (3, 3, False), (0, 0, False)],
)
def test_partial_failure_is_some_but_not_all(failures: int, total: int, expected: bool) -> None:
    assert is_partial_failure(failures, total) is expected
