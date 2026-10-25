"""DOC-18: odtwarzalne liczby repo — czyste funkcje nad treścią plików.

Docs cytowały „640+ tests”, gdy funkcji testowych było kilka razy więcej, a liczb
z dawnych commitów nie dało się odtworzyć. `compute_repo_stats` liczy je z treści
plików (AST, bez importowania modułów); czytanie repo i zapis
`docs/repo-stats.json` robi podkomenda `main_tools.py repo-stats` (ARCH-39).
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

REPO_STATS_NAME = "repo-stats.json"

# Te same reguły co `supabase db push`: NNN_nazwa.sql w supabase/migrations/.
_MIGRATION_NAME = re.compile(r"^\d{3}_.+\.sql$")
_PORTS_MODULE = "src/application/ports.py"
_ADAPTER_DIRS = ("src/infrastructure/adapters/", "src/infrastructure/llm/")
_DOMAIN_DIR = "src/domain/"


@dataclass(frozen=True)
class RepoStats:
    """Liczby, które docs mogą cytować; `domain_stdlib_only` to wynik sprawdzenia
    importów domeny (stdlib, `__future__` i `src.domain`)."""

    test_functions: int
    migrations: int
    ports: int
    adapter_modules: int
    domain_modules: int
    domain_stdlib_only: bool

    def to_json(self) -> str:
        """Deterministyczny JSON (klucze posortowane, bez daty) do commitu."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def _module_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _is_public_module(path: str, directory: str) -> bool:
    name = _module_name(path)
    return (
        path.startswith(directory)
        and "/" not in path[len(directory) :]
        and name.endswith(".py")
        and not name.startswith("_")
    )


def _count_test_functions(sources: Mapping[str, str]) -> int:
    return sum(
        1
        for path, text in sources.items()
        if path.startswith("tests/")
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.startswith("test_")
    )


def _count_ports(ports_source: str) -> int:
    return sum(
        1
        for node in ast.parse(ports_source).body
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == "ABC" for base in node.bases)
    )


def _is_allowed_domain_import(module: str) -> bool:
    top = module.split(".", 1)[0]
    return (
        module == "__future__"
        or module.startswith("src.domain")
        or (top in sys.stdlib_module_names)
    )


def _imports_stdlib_only(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        else:
            continue
        if not all(_is_allowed_domain_import(module) for module in modules):
            return False
    return True


def compute_repo_stats(sources: Mapping[str, str], migration_names: Iterable[str]) -> RepoStats:
    """Liczby repo z `sources` (ścieżka względem roota w stylu POSIX → treść pliku
    `.py`) i nazw plików z `supabase/migrations/`."""
    domain = {
        path: text
        for path, text in sources.items()
        if path.startswith(_DOMAIN_DIR) and "/" not in path[len(_DOMAIN_DIR) :]
    }
    return RepoStats(
        test_functions=_count_test_functions(sources),
        migrations=sum(1 for name in migration_names if _MIGRATION_NAME.match(name)),
        ports=_count_ports(sources.get(_PORTS_MODULE, "")),
        adapter_modules=sum(
            1
            for path in sources
            for directory in _ADAPTER_DIRS
            if _is_public_module(path, directory)
        ),
        domain_modules=sum(1 for path in domain if _is_public_module(path, _DOMAIN_DIR)),
        domain_stdlib_only=all(_imports_stdlib_only(text) for text in domain.values()),
    )
