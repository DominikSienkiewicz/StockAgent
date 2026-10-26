"""CI-16: liczby testów do podsumowania joba liczone z raportu JUnit, nie z `def test_`."""

from __future__ import annotations

from src.tools.junit_summary import (
    NO_REPORT,
    UNREADABLE_REPORT,
    JunitTotals,
    junit_totals,
    render_mutation_summary,
    render_test_summary,
    totals_or_reason,
)

PYTEST_REPORT = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<testsuites name="pytest tests">'
    '<testsuite name="pytest" errors="1" failures="2" skipped="3" tests="10" time="1.5">'
    '<testcase classname="tests.test_a" name="test_ok" time="0.1"/>'
    "</testsuite></testsuites>"
)


def test_junit_totals_sum_every_testsuite() -> None:
    report = (
        "<testsuites>"
        '<testsuite tests="4" failures="1" errors="0" skipped="1"/>'
        '<testsuite tests="6" failures="0" errors="2" skipped="0"/>'
        "</testsuites>"
    )

    assert junit_totals(report) == JunitTotals(tests=10, failures=1, errors=2, skipped=1)


def test_junit_totals_read_a_bare_testsuite_root() -> None:
    report = '<testsuite tests="3" failures="0" errors="0" skipped="0"/>'

    assert junit_totals(report).passed == 3


def test_junit_totals_read_the_pytest_report() -> None:
    totals = junit_totals(PYTEST_REPORT)

    assert totals == JunitTotals(tests=10, failures=2, errors=1, skipped=3)
    assert totals.passed == 4


def test_test_summary_lists_every_step_and_the_total() -> None:
    summary = render_test_summary(
        [
            ("unit", JunitTotals(tests=10, failures=2, errors=1, skipped=3)),
            ("containers", JunitTotals(tests=5, failures=0, errors=0, skipped=0)),
        ]
    )

    assert "| Step | Tests | Passed | Skipped | Failures | Errors |" in summary
    assert "| unit | 10 | 4 | 3 | 2 | 1 |" in summary
    assert "| containers | 5 | 5 | 0 | 0 | 0 |" in summary
    assert "| **Total** | 15 | 9 | 3 | 2 | 1 |" in summary


def test_test_summary_names_a_step_without_a_report() -> None:
    summary = render_test_summary(
        [("unit", JunitTotals(tests=1, failures=0, errors=0, skipped=0)), ("containers", NO_REPORT)]
    )

    assert "| containers | no report | | | | |" in summary
    assert "| **Total** | 1 | 1 | 0 | 0 | 0 |" in summary


def test_mutation_summary_reads_survivors_as_failures_and_timeouts_as_errors() -> None:
    # mutmut junitxml: każdy mutant to testcase; przeżył = failure, timeout = error,
    # nieprzetestowany (`--untested-policy=skipped`) = skipped.
    summary = render_mutation_summary(
        [("var", JunitTotals(tests=45, failures=16, errors=1, skipped=0)), ("asset", NO_REPORT)]
    )

    assert "| Module | Mutants | Killed | Survived | Timeouts | Untested | Score |" in summary
    assert "| var | 45 | 28 | 16 | 1 | 0 | 62.2% |" in summary
    assert "| asset | no report | | | | | |" in summary


def test_mutation_score_is_empty_without_a_tested_mutant() -> None:
    summary = render_mutation_summary(
        [("var", JunitTotals(tests=0, failures=0, errors=0, skipped=0))]
    )

    assert "| var | 0 | 0 | 0 | 0 | 0 | — |" in summary


def test_a_missing_or_broken_report_gives_a_reason_instead_of_numbers() -> None:
    assert totals_or_reason(None) == NO_REPORT
    assert totals_or_reason("<testsuites") == UNREADABLE_REPORT
    assert totals_or_reason(PYTEST_REPORT) == junit_totals(PYTEST_REPORT)
