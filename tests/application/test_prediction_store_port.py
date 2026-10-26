"""ARCH-32 (krok 1): Fast Loop zależy od wąskiego portu magazynu predykcji.

`AgentGraphDeps.repository_port` deklaruje tylko metody, które woła graf i jego use
case (`AnalyzeMarketUseCase` z fazą cech rynkowych), zamiast całego `RepositoryPort`
(kilkadziesiąt metod raportu, Slow Loopa i watcha). Zbiór metod liczy AST z kodu,
więc port nie może ani niczego pominąć, ani urosnąć o metodę, której Fast Loop nie woła.
"""

from __future__ import annotations

import ast
import inspect
import typing
from pathlib import Path

from src.application.agent_graph import AgentGraphDeps
from src.application.ports import PredictionStorePort, RepositoryPort

ROOT = Path(__file__).parents[2]
# Moduł → nazwy, pod którymi Fast Loop trzyma repozytorium (`deps.repository_port`,
# lokalna zmienna, parametr pomocnika, pole use case'u, parametr fazy cech).
FAST_LOOP_REPOSITORY_NAMES: dict[str, frozenset[str]] = {
    "src/application/agent_graph.py": frozenset({"repository_port"}),
    "src/application/use_cases/analyze_market.py": frozenset({"_repository"}),
    "src/application/market_features.py": frozenset({"repository"}),
}


def _holder_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _fast_loop_repository_calls() -> set[str]:
    called: set[str] = set()
    for relative, names in FAST_LOOP_REPOSITORY_NAMES.items():
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and _holder_name(node.func.value) in names
            ):
                called.add(node.func.attr)
    return called


def _store_type() -> type:
    store = typing.get_type_hints(AgentGraphDeps)["repository_port"]
    assert isinstance(store, type)
    return store


def _public_methods(cls: type) -> set[str]:
    return {
        name
        for name, member in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


def test_graph_deps_depend_on_exactly_the_methods_the_fast_loop_calls() -> None:
    store = _store_type()

    assert store is not RepositoryPort, "Fast Loop zależy od całego RepositoryPort"
    assert _public_methods(store) == _fast_loop_repository_calls()


def test_prediction_store_mirrors_repository_port_signatures() -> None:
    # Protocol z inną sygnaturą niż ABC dałby repozytorium, które spełnia jedno,
    # a nie drugie — mypy odkryłby to dopiero przy kolejnym adapterze.
    methods = _public_methods(PredictionStorePort)

    assert methods <= _public_methods(RepositoryPort)
    for name in sorted(methods):
        assert inspect.signature(getattr(PredictionStorePort, name)) == inspect.signature(
            getattr(RepositoryPort, name)
        ), name
