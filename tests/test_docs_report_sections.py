"""DOC-15: każdy nagłówek sekcji maila jest opisany w docs/report-email.md.

Anatomia w docs opisywała ok. 15 pozycji, a raport ma ponad 25 sekcji. Strażnik
czyta nagłówki `<h2>` / `<h3>` (i `_section_h2(...)`) z modułów
`src/application/report_*.py` przez AST — także złożone z `_HEADING` albo innej
stałej modułu — i wymaga, żeby każdy stał w docs dosłownie, z emoji. Strona
publicznego scorecardu (`report_public_scorecard.py`) to osobny artefakt, nie mail.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
REPORT_DOC = ROOT / "docs" / "report-email.md"
NOT_THE_EMAIL = frozenset({"report_public_scorecard.py"})
_HEADING = re.compile(r"<h[23][^>]*>(.*?)</h[23]>", re.DOTALL)
# Znacznik miejsca wartości dynamicznej (np. okno „ostatnie 24h”) — nagłówek
# kończy się przed nim.
_DYNAMIC = "\x00"


def _module_constants(tree: ast.Module) -> dict[str, str]:
    return {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _render(node: ast.expr, constants: dict[str, str]) -> str | None:
    """Tekst wyrażenia ze stałych i f-stringów; wartości spoza stałych → `_DYNAMIC`."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
            elif (
                isinstance(part, ast.FormattedValue)
                and isinstance(part.value, ast.Name)
                and part.value.id in constants
            ):
                parts.append(constants[part.value.id])
            else:
                parts.append(_DYNAMIC)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _render(node.left, constants), _render(node.right, constants)
        return None if left is None or right is None else left + right
    return None


def _clean(heading: str) -> str:
    return heading.split(_DYNAMIC, 1)[0].strip().rstrip("(").strip()


def _headings_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = _module_constants(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_section_h2"
            and node.args
        ):
            title = _render(node.args[0], constants)
            if title is not None:
                found.add(_clean(title))
        elif isinstance(node, ast.expr):
            text = _render(node, constants)
            if text is not None:
                found.update(_clean(match) for match in _HEADING.findall(text))
    return {heading for heading in found if heading}


def email_headings() -> set[str]:
    modules = sorted((ROOT / "src" / "application").glob("report_*.py"))
    return {
        heading
        for path in modules
        if path.name not in NOT_THE_EMAIL
        for heading in _headings_of(path)
    }


def test_the_extractor_sees_the_known_headings() -> None:
    # Bez tego strażnik niżej byłby pusty, gdyby AST przestało cokolwiek łapać.
    assert {
        "💸 Koszt cyklu",
        "🩺 Karta kondycji modelu",
        "🎯 Portfolio Radar — koncentracja i ryzyko",
        "📊 Zamknięte predykcje",
        "💸 Pominięte — wyczerpany budżet cyklu",
        "🛰️ Alpha Signals — dodatkowe źródła danych",
        "💼 Twój portfel",
        "📅 Tydzień StockAgenta",
        "💡 Uzasadnienia",
    } <= email_headings()


@pytest.mark.parametrize("heading", sorted(email_headings()))
def test_report_email_doc_names_every_section_heading(heading: str) -> None:
    assert heading in REPORT_DOC.read_text(encoding="utf-8"), (
        f"docs/report-email.md nie opisuje sekcji „{heading}”"
    )
