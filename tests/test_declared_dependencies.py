"""Strażnik deklaracji zależności (CI-01): `pyproject.toml` ↔ importy w kodzie.

`openai` był importowany przez adaptery, a nie był zadeklarowany — instalował się
tylko przypadkiem, jako zależność martwego `langchain-openai`. Odwrotnie
`langchain` stał pod nagłówkiem „Agentic Framework & RAG”, choć nic go nie
importuje. Test czyta importy z AST (także lokalne w funkcjach i pod
`TYPE_CHECKING`) i mapuje moduł na dystrybucję przez `importlib.metadata`.
Bez `yaml` i `packaging`: sam strażnik nie może zależeć od niezadeklarowanych
pakietów.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from collections.abc import Iterable
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).parent.parent
RUNTIME_GLOBS = ("src/**/*.py", "main_*.py")
TEST_GLOBS = ("tests/**/*.py",)
FIRST_PARTY = frozenset({"src", "tests", "__future__"})
# Zależności rdzeniowe bez bezpośredniego importu — ładuje je inna biblioteka.
LOADED_BY_ANOTHER_PACKAGE = {
    "scikit-learn": "xgboost.XGBRegressor (API sklearn) importuje sklearn sam",
    "python-dotenv": "pydantic-settings czyta `.env` (env_file) przez python-dotenv",
}
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(requirements: Iterable[str]) -> set[str]:
    names: set[str] = set()
    for requirement in requirements:
        match = _REQUIREMENT_NAME.match(requirement.strip())
        assert match, f"nieczytelny wpis zależności: {requirement!r}"
        names.add(_normalize(match.group(0)))
    return names


def _pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _runtime_declared() -> set[str]:
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    return _requirement_names(project["dependencies"])


def _dev_declared() -> set[str]:
    groups = _pyproject()["dependency-groups"]
    assert isinstance(groups, dict)
    return _requirement_names(groups["dev"])


def _is_first_party(top: str) -> bool:
    return top in FIRST_PARTY or (top.startswith("main_") and (ROOT / f"{top}.py").is_file())


def third_party_imports(source: str) -> set[str]:
    """Moduły najwyższego poziomu spoza stdlib i spoza repo, importowane w `source`."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    tops = {module.split(".", 1)[0] for module in modules}
    return {top for top in tops if top not in sys.stdlib_module_names and not _is_first_party(top)}


def _imports_by_module(globs: Iterable[str]) -> dict[str, set[str]]:
    """Moduł third-party → pliki repo, które go importują."""
    found: dict[str, set[str]] = {}
    for pattern in globs:
        for path in sorted(ROOT.glob(pattern)):
            for top in third_party_imports(path.read_text(encoding="utf-8")):
                found.setdefault(top, set()).add(str(path.relative_to(ROOT)))
    return found


def _distributions(module: str) -> set[str]:
    return {_normalize(dist) for dist in packages_distributions().get(module, [])}


def undeclared_imports(imports: dict[str, set[str]], declared: set[str]) -> set[str]:
    """Moduły, których żadna dostarczająca je dystrybucja nie jest zadeklarowana."""
    return {module for module in imports if not _distributions(module) & declared}


def test_helper_sees_local_and_type_checking_imports() -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "import src.domain.asset\n"
        "if TYPE_CHECKING:\n"
        "    from openai import OpenAI\n"
        "def build():\n"
        "    import yaml.loader\n"
        "    from . import sibling\n"
    )

    assert third_party_imports(source) == {"openai", "yaml"}


def test_every_third_party_import_is_declared() -> None:
    imports = _imports_by_module(RUNTIME_GLOBS)

    # Strażnik pustego wyniku: adaptery zawsze importują `requests`.
    assert "requests" in imports
    missing = undeclared_imports(imports, _runtime_declared())
    assert missing == set(), {module: sorted(imports[module]) for module in missing}


def test_every_third_party_test_import_is_declared() -> None:
    imports = _imports_by_module(TEST_GLOBS)

    assert "pytest" in imports
    missing = undeclared_imports(imports, _runtime_declared() | _dev_declared())
    assert missing == set(), {module: sorted(imports[module]) for module in missing}


def test_no_declared_runtime_dependency_is_unused() -> None:
    imported_dists: set[str] = set()
    for module in _imports_by_module(RUNTIME_GLOBS):
        imported_dists |= _distributions(module)

    unused = _runtime_declared() - imported_dists - set(LOADED_BY_ANOTHER_PACKAGE)
    assert unused == set(), f"zadeklarowane, ale nieimportowane w src/ ani main_*.py: {unused}"


def test_exemptions_are_still_declared() -> None:
    assert set(LOADED_BY_ANOTHER_PACKAGE) <= _runtime_declared()
