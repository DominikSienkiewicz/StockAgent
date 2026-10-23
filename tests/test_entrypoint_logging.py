"""Strażnik BEZ-02: każdy entry-point `main_*.py` konfiguruje logi przez
`configure_logging` z wartościami sekretów z `Settings`.

Gołe `logging.basicConfig` w nowym entry-poincie puściłoby traceback z kluczem
do publicznego logu GHA. Test czyta wywołania z AST, więc łapie też wywołania
w funkcjach pomocniczych (np. `main_tools._load_settings`).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
ENTRY_POINTS = sorted(path.name for path in ROOT.glob("main_*.py"))


def _calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _callee(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _passes_settings_secrets(call: ast.Call) -> bool:
    arguments = [*call.args, *(keyword.value for keyword in call.keywords)]
    return any(
        isinstance(argument, ast.Call) and _callee(argument) == "secret_values"
        for argument in arguments
    )


def test_guard_sees_every_entry_point() -> None:
    assert {
        "main_agent.py",
        "main_backtest.py",
        "main_deadman.py",
        "main_tools.py",
        "main_trainer.py",
        "main_watch.py",
    } <= set(ENTRY_POINTS)


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_entry_point_configures_redacting_logging_with_settings_secrets(module: str) -> None:
    calls = _calls(ROOT / module)

    configure = [call for call in calls if _callee(call) == "configure_logging"]
    assert configure, f"{module} nie woła configure_logging (BEZ-02)"
    assert all(_passes_settings_secrets(call) for call in configure), (
        f"{module} woła configure_logging bez settings.secret_values()"
    )
    assert not [call for call in calls if _callee(call) == "basicConfig"], (
        f"{module} konfiguruje logi gołym basicConfig — sekrety pójdą do logu bez redakcji"
    )


def test_guard_rejects_a_bare_basic_config(tmp_path: Path) -> None:
    legacy = tmp_path / "main_legacy.py"
    legacy.write_text(
        "import logging\n"
        "def main():\n"
        "    logging.basicConfig(level=logging.INFO)\n"
        "    configure_logging()\n",
        encoding="utf-8",
    )

    calls = _calls(legacy)

    assert [_callee(call) for call in calls].count("basicConfig") == 1
    assert not any(
        _passes_settings_secrets(call) for call in calls if _callee(call) == "configure_logging"
    )


def _imports_configure_logging_from(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "configure_logging" for alias in node.names)
    }


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_entry_point_logs_with_the_run_context(module: str) -> None:
    """NIEZ-21: entry-point bierze `configure_logging` z `logging_setup`, nie z warstwy
    redakcji — inaczej jego linie nie mają run_id, entry_point ani symbolu (JSON w GHA)."""
    assert _imports_configure_logging_from(ROOT / module) == {"src.infrastructure.logging_setup"}, (
        f"{module} konfiguruje logi z pominięciem logging_setup (NIEZ-21)"
    )
