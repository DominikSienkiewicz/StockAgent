"""Strażnik ARCH-39: `src/tools/` trzyma czyste funkcje narzędzi jednorazowych.

`Settings` i konkretne adaptery składa wyłącznie `main_tools.py` (composition
root, podkomenda na narzędzie). Bez tego strażnika każde nowe narzędzie powtórzy
dawny wzorzec `evaluate.py`, który sam importował `Settings` i
`SupabaseRepository`. Test czyta importy z AST, więc łapie też importy lokalne
w funkcjach i pod `TYPE_CHECKING`.

TEST-05: kierunek zależności `domain ← application ← infrastructure`. Domena
importuje wyłącznie stdlib i `src.domain`, application nie importuje
infrastruktury ani entry-pointów, a `typing.Any` w domenie ma tylko
`attestation.py` (kanonizacja dowolnego JSON-a przed hashem).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
TOOLS_DIR = ROOT / "src" / "tools"
DOMAIN_DIR = ROOT / "src" / "domain"
APPLICATION_DIR = ROOT / "src" / "application"
# Infrastruktura i composition rooty są „nad” application — import w tę stronę odwraca strzałkę.
ABOVE_APPLICATION = ("src.infrastructure", "main_")
DOMAIN_ANY_ALLOWED = frozenset({"attestation.py"})
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

    assert {"src.config", "src.infrastructure.adapters.supabase_repo"} <= _imported_modules(legacy)


# --- TEST-05: czystość domeny i kierunek zależności ---------------------------


def _absolute_and_relative_imports(source: str) -> set[str]:
    """Każdy import modułu w `source` (też lokalny i pod `TYPE_CHECKING`).

    Import względny wraca z kropkami (`..application.ports`), bo jego cel zależy
    od pakietu pliku — warstwa ma go nie używać, więc helper go zgłasza."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add("." * node.level + (node.module or ""))
    return modules


def domain_import_violations(source: str) -> set[str]:
    """Importy domeny spoza stdlib i `src.domain` (third-party, w górę, względne)."""
    return {
        module
        for module in _absolute_and_relative_imports(source)
        if not (
            module.split(".", 1)[0] in sys.stdlib_module_names
            or module == "src.domain"
            or module.startswith("src.domain.")
        )
    }


def application_import_violations(source: str) -> set[str]:
    """Importy application z infrastruktury albo z entry-pointów `main_*`."""
    return {
        module
        for module in _absolute_and_relative_imports(source)
        if module.startswith(ABOVE_APPLICATION)
    }


def uses_typing_any(source: str) -> bool:
    """`from typing import Any` albo `typing.Any` / `<alias typing>.Any` w `source`."""
    tree = ast.parse(source)
    typing_aliases = {"typing"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            typing_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name == "typing"
            )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "typing"
            and any(alias.name == "Any" for alias in node.names)
        ):
            return True
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "Any"
            and isinstance(node.value, ast.Name)
            and node.value.id in typing_aliases
        ):
            return True
    return False


def _layer_modules(layer: Path) -> list[str]:
    return sorted(str(path.relative_to(layer)) for path in layer.rglob("*.py"))


def test_helper_flags_third_party_and_upward_imports() -> None:
    source = (
        "from __future__ import annotations\n"
        "import dataclasses\n"
        "from collections.abc import Sequence\n"
        "from typing import TYPE_CHECKING\n"
        "from src.domain.asset import Asset\n"
        "if TYPE_CHECKING:\n"
        "    from src.application.ports import LLMPort\n"
        "def fetch():\n"
        "    import requests\n"
        "    from ..infrastructure import adapters\n"
    )

    assert domain_import_violations(source) == {
        "src.application.ports",
        "requests",
        "..infrastructure",
    }
    assert application_import_violations(
        "def build():\n    from src.infrastructure.adapters.supabase_repo import X\n"
        "import main_agent\nfrom src.domain.asset import Asset\n"
    ) == {"src.infrastructure.adapters.supabase_repo", "main_agent"}


def test_helper_sees_every_spelling_of_typing_any() -> None:
    assert uses_typing_any("from typing import Any\n")
    assert uses_typing_any("import typing\nx: typing.Any\n")
    assert uses_typing_any("import typing as t\nx: t.Any\n")
    assert not uses_typing_any("from typing import Literal\nAny = 1\n")


@pytest.mark.parametrize("module", _layer_modules(DOMAIN_DIR))
def test_domain_imports_only_stdlib_and_domain(module: str) -> None:
    violations = domain_import_violations((DOMAIN_DIR / module).read_text(encoding="utf-8"))

    assert not violations, (
        f"src/domain/{module} importuje {sorted(violations)} — domena to czysty Python "
        "(stdlib + src.domain), I/O i biblioteki należą do infrastruktury"
    )


@pytest.mark.parametrize("module", _layer_modules(APPLICATION_DIR))
def test_application_does_not_import_infrastructure(module: str) -> None:
    violations = application_import_violations(
        (APPLICATION_DIR / module).read_text(encoding="utf-8")
    )

    assert not violations, (
        f"src/application/{module} importuje {sorted(violations)} — application zna porty, "
        "a adaptery wstrzykuje composition root (main_*.py)"
    )


@pytest.mark.parametrize(
    "module", [name for name in _layer_modules(DOMAIN_DIR) if name not in DOMAIN_ANY_ALLOWED]
)
def test_domain_has_no_typing_any_outside_attestation(module: str) -> None:
    assert not uses_typing_any((DOMAIN_DIR / module).read_text(encoding="utf-8")), (
        f"src/domain/{module} używa typing.Any — domena typuje swoje wartości wprost"
    )


def test_layer_globs_see_the_layers() -> None:
    # Strażnik pustego wyniku: zła ścieżka dałaby zero parametrów i zielony test.
    assert "asset.py" in _layer_modules(DOMAIN_DIR)
    assert "ports.py" in _layer_modules(APPLICATION_DIR)
    assert set(_layer_modules(DOMAIN_DIR)) >= DOMAIN_ANY_ALLOWED


USE_CASES_DIR = APPLICATION_DIR / "use_cases"


@pytest.mark.parametrize("module", _layer_modules(USE_CASES_DIR))
def test_use_cases_do_not_import_pandas(module: str) -> None:
    # ML-58: use case'y idą na wierszach (`ml_features`); DataFrame składa adapter modelu.
    imported = _absolute_and_relative_imports((USE_CASES_DIR / module).read_text(encoding="utf-8"))

    assert not {name for name in imported if name.split(".", 1)[0] == "pandas"}, (
        f"src/application/use_cases/{module} importuje pandas — wiersze widoku obsługują "
        "pomocniki `src/application/ml_features.py`"
    )


def test_use_cases_glob_sees_the_use_cases() -> None:
    assert {"train_model.py", "backtest.py"} <= set(_layer_modules(USE_CASES_DIR))
