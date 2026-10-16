"""Strażnik jednej bramki jakości: CI ↔ `[tool.mypy]` ↔ pliki agentowe ↔ docs.

Bramka żyła w kilku rozjechanych wersjach (`CLAUDE.md`, `AGENTS.md`, `ci.yml`,
`docs/running-locally.md`, `pyproject.toml`). Skutek: `main_watch.py` — wtedy
jedyny entry-point na aktywnym cronie — nie był w CI ani lintowany, ani
typowany, a gołe `uv run mypy` zgodne z konfiguracją dawało czerwony wynik.

Kontrakt:
1. CI lintuje i typuje KAŻDY `main_*.py` (jawnie albo przez `.` / gołe `mypy`
   z `[tool.mypy].files`).
2. Dokumenty dla agentów i ludzi cytują dokładnie te same komendy co CI.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
GATE_DOCS = ("CLAUDE.md", "AGENTS.md", "docs/running-locally.md")
GATE_TOOLS = ("ruff", "mypy")


def _entrypoints() -> list[str]:
    return sorted(path.name for path in ROOT.glob("main_*.py"))


def _from_tool(command: str, tool: str) -> list[str]:
    """Tokeny komendy od nazwy narzędzia, bez komentarza i flag `uv run`.

    `uv run --frozen --no-build --no-sync ruff check .` i `uv run ruff check .`
    normalizują się do tego samego `["ruff", "check", "."]`.
    """
    tokens = command.split("#", 1)[0].split()
    return tokens[tokens.index(tool) :] if tool in tokens else []


def _ci_commands() -> dict[str, list[str]]:
    steps = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["quality"]["steps"]
    commands: dict[str, list[str]] = {}
    for step in steps:
        run = str(step.get("run", ""))
        for tool in GATE_TOOLS:
            tokens = _from_tool(run, tool)
            if tokens and tool not in commands:
                commands[tool] = tokens
    return commands


def _mypy_files() -> list[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    files: list[str] = config["tool"]["mypy"]["files"]
    return files


def test_ci_quality_job_runs_ruff_and_mypy() -> None:
    assert set(_ci_commands()) == set(GATE_TOOLS)


@pytest.mark.parametrize("entrypoint", _entrypoints())
def test_ci_lints_and_typechecks_every_entrypoint(entrypoint: str) -> None:
    commands = _ci_commands()

    ruff_paths = commands["ruff"][2:]
    assert "." in ruff_paths or entrypoint in ruff_paths, (
        f"ci.yml nie lintuje {entrypoint}: {' '.join(commands['ruff'])}"
    )

    mypy_paths = [token for token in commands["mypy"][1:] if not token.startswith("-")]
    typed = mypy_paths or _mypy_files()
    assert entrypoint in typed, (
        f"ci.yml nie typuje {entrypoint}: {' '.join(commands['mypy'])} "
        f"(pliki: {typed})"
    )


def test_bare_mypy_config_does_not_cover_untyped_tests() -> None:
    # Decyzja 2026-10-16: `tests/` spłacamy ratchetem po katalogach. Cały
    # katalog w `files` czerwieni gołe `mypy` (1326 błędów), zanim dług zniknie.
    assert "tests" not in _mypy_files()


@pytest.mark.parametrize("doc", GATE_DOCS)
@pytest.mark.parametrize("tool", GATE_TOOLS)
def test_agent_docs_quote_the_ci_gate(doc: str, tool: str) -> None:
    ci_command = _ci_commands()[tool]
    quoted = [
        _from_tool(line, tool)
        for line in (ROOT / doc).read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith(f"uv run {tool}")
    ]
    assert quoted, f"{doc} nie cytuje komendy `{tool}` z bramki"
    assert all(tokens == ci_command for tokens in quoted), (
        f"{doc} cytuje {[' '.join(t) for t in quoted]}, a CI uruchamia "
        f"`{' '.join(ci_command)}`."
    )
