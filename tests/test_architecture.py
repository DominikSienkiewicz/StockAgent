"""Strażnik ARCH-39: `src/tools/` trzyma czyste funkcje narzędzi jednorazowych.

`Settings` i konkretne adaptery składa wyłącznie `main_tools.py` (composition
root, podkomenda na narzędzie). Bez tego strażnika każde nowe narzędzie powtórzy
dawny wzorzec `evaluate.py`, który sam importował `Settings` i
`SupabaseRepository`. Test czyta importy z AST, więc łapie też importy lokalne
w funkcjach i pod `TYPE_CHECKING`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
TOOLS_DIR = ROOT / "src" / "tools"
# Składniki kompozycji: konfiguracja, adaptery, klienci LLM i same entry-pointy.
COMPOSITION_MODULES = (
    "src.config",
    "src.infrastructure.adapters",
    "src.infrastructure.llm",
    "main_",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("module", sorted(path.name for path in TOOLS_DIR.glob("*.py")))
def test_src_tools_do_not_compose_settings_or_adapters(module: str) -> None:
    offending = sorted(
        name
        for name in _imported_modules(TOOLS_DIR / module)
        if name.startswith(COMPOSITION_MODULES)
    )

    assert not offending, (
        f"src/tools/{module} importuje {offending} — składanie Settings i adapterów "
        "należy do main_tools.py (ARCH-39)"
    )


def test_the_guard_sees_the_old_evaluate_pattern(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.py"
    legacy.write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from src.infrastructure.adapters.supabase_repo import SupabaseRepository\n"
        "def main():\n"
        "    from src.config import Settings\n",
        encoding="utf-8",
    )

    assert {"src.config", "src.infrastructure.adapters.supabase_repo"} <= _imported_modules(
        legacy
    )
