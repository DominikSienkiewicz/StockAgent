"""Zapadka ARCH-33: liczba szerokich handlerów wyjątków w kodzie produkcyjnym nie rośnie.

Szeroki handler to `except Exception`, `except BaseException` i goły `except:` —
także jako element krotki. Każdy taki handler połyka błąd bez względu na jego
klasę, więc cichej degradacji nie da się odróżnić od awarii, a telemetria nie
policzy błędów per klasa. Nowy kod łapie taksonomię portów
(`src.application.errors`) albo konkretne wyjątki.

Liczba może tylko maleć: gdy zmiana usuwa szeroki handler, obniż
`BROAD_HANDLER_BASELINE` do nowej wartości w tym samym commicie.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parent.parent
# Stan na 7b44c99 (2026-10-24), policzony tym samym AST-em po src/ i main_*.py.
BROAD_HANDLER_BASELINE = 157
_BROAD_NAMES = frozenset({"Exception", "BaseException"})


def _production_files() -> list[Path]:
    return sorted([*(ROOT / "src").rglob("*.py"), *ROOT.glob("main_*.py")])


def _is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    caught = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(name, ast.Name) and name.id in _BROAD_NAMES for name in caught)


def count_broad_handlers(source: str) -> int:
    tree = ast.parse(source)
    return sum(
        1 for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler) and _is_broad(node)
    )


def test_counts_bare_tuple_and_named_broad_handlers() -> None:
    source = (
        "try:\n    pass\nexcept Exception:\n    pass\n"
        "try:\n    pass\nexcept (ValueError, BaseException):\n    pass\n"
        "try:\n    pass\nexcept:\n    pass\n"
        "try:\n    pass\nexcept ValueError:\n    pass\n"
    )

    assert count_broad_handlers(source) == 3


def test_broad_except_count_does_not_grow() -> None:
    per_file = {
        path.relative_to(ROOT).as_posix(): count_broad_handlers(path.read_text(encoding="utf-8"))
        for path in _production_files()
    }
    total = sum(per_file.values())

    assert total <= BROAD_HANDLER_BASELINE, (
        f"{total} szerokich handlerów (`except Exception`/`BaseException`/goły `except`) "
        f"> zapadka {BROAD_HANDLER_BASELINE}. Łap taksonomię z src.application.errors "
        "albo konkretny wyjątek. Najwięcej: "
        + ", ".join(
            f"{name}={count}"
            for name, count in sorted(per_file.items(), key=lambda item: -item[1])[:5]
        )
    )
