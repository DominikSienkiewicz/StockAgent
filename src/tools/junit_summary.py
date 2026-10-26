"""CI-16: liczby testów z raportów JUnit do podsumowania joba w GitHub Actions.

Liczby z commitów (np. 9d98e16) były nieodtwarzalne, a liczenie `def test_` nie widzi
parametryzacji ani skipów. Raport JUnit (pytest `--junitxml`, `mutmut junitxml`) mówi,
co naprawdę się wykonało. Czyste funkcje: suma z XML-a i tabela Markdown; pliki
i `GITHUB_STEP_SUMMARY` obsługuje `main_tools.py junit-summary`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from xml.etree import ElementTree

_COUNTED = ("tests", "failures", "errors", "skipped")
_TEST_COLUMNS = ("Step", "Tests", "Passed", "Skipped", "Failures", "Errors")
_MUTATION_COLUMNS = ("Module", "Mutants", "Killed", "Survived", "Timeouts", "Untested", "Score")


@dataclass(frozen=True)
class JunitTotals:
    """Sumy atrybutów `<testsuite>` jednego raportu."""

    tests: int
    failures: int
    errors: int
    skipped: int

    @property
    def passed(self) -> int:
        return self.tests - self.failures - self.errors - self.skipped

    def __add__(self, other: JunitTotals) -> JunitTotals:
        return JunitTotals(
            tests=self.tests + other.tests,
            failures=self.failures + other.failures,
            errors=self.errors + other.errors,
            skipped=self.skipped + other.skipped,
        )


_ZERO = JunitTotals(tests=0, failures=0, errors=0, skipped=0)
NO_REPORT = "no report"
UNREADABLE_REPORT = "unreadable report"


def junit_totals(xml_text: str) -> JunitTotals:
    """Suma po każdym `<testsuite>` (korzeń `<testsuites>` albo sam `<testsuite>`).

    Rzuca `xml.etree.ElementTree.ParseError` dla nieczytelnego raportu.
    """
    root = ElementTree.fromstring(xml_text)
    total = _ZERO
    for suite in root.iter("testsuite"):
        counts = {name: int(suite.get(name, "0")) for name in _COUNTED}
        total += JunitTotals(**counts)
    return total


def totals_or_reason(xml_text: str | None) -> JunitTotals | str:
    """Sumy raportu albo powód ich braku: `None` = brak pliku (krok nie ruszył)."""
    if xml_text is None:
        return NO_REPORT
    try:
        return junit_totals(xml_text)
    except ElementTree.ParseError:
        return UNREADABLE_REPORT


def render_test_summary(rows: Sequence[tuple[str, JunitTotals | str]]) -> str:
    """Tabela kroków pytest z wierszem sumy; napis zamiast sum = powód braku liczb."""
    lines = ["### 🧪 Tests (JUnit)", "", *_header(_TEST_COLUMNS)]
    total = _ZERO
    for label, totals in rows:
        if isinstance(totals, str):
            lines.append(_missing_row(label, totals, _TEST_COLUMNS))
            continue
        total += totals
        lines.append(_row(label, *_test_cells(totals)))
    lines.append(_row("**Total**", *_test_cells(total)))
    return "\n".join(lines) + "\n"


def render_mutation_summary(rows: Sequence[tuple[str, JunitTotals | str]]) -> str:
    """Tabela modułów z raportu `mutmut junitxml`: przeżył = failure, timeout = error,
    nieprzetestowany = skipped. Wynik = zabite / przetestowane, bez progu (TEST-35)."""
    lines = ["### 🧬 Mutation testing (mutmut)", "", *_header(_MUTATION_COLUMNS)]
    for label, totals in rows:
        if isinstance(totals, str):
            lines.append(_missing_row(label, totals, _MUTATION_COLUMNS))
            continue
        tested = totals.tests - totals.skipped
        score = f"{100 * totals.passed / tested:.1f}%" if tested else "—"
        cells = (totals.tests, totals.passed, totals.failures, totals.errors, totals.skipped)
        lines.append(_row(label, *(str(cell) for cell in cells), score))
    return "\n".join(lines) + "\n"


def _test_cells(totals: JunitTotals) -> tuple[str, ...]:
    cells = (totals.tests, totals.passed, totals.skipped, totals.failures, totals.errors)
    return tuple(str(cell) for cell in cells)


def _header(columns: Sequence[str]) -> list[str]:
    return [_row(*columns), "|---|" + "---:|" * (len(columns) - 1)]


def _row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _missing_row(label: str, reason: str, columns: Sequence[str]) -> str:
    """Wiersz bez liczb: powód w drugiej kolumnie, reszta pusta."""
    return f"| {label} | {reason} |" + " |" * (len(columns) - 2)
